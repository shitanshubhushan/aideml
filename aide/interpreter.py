"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("aide")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [
                line
                for line in tb_lines
                if "aide/" not in line and "importlib" not in line
            ]
        )
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue, timeout=5):
        self.queue = queue
        self.timeout = timeout

    def write(self, msg):
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            logger.warning("Queue write timed out")

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()
        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in the data preview)
            os.remove(self.agent_file_name)

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup_session(self):
        if self.process is None:
            return
        try:
            # Reduce grace period from 2 seconds to 0.5
            self.process.terminate()
            self.process.join(timeout=0.5)

            if self.process.exitcode is None:
                logger.warning("Process failed to terminate, killing immediately")
                self.process.kill()
                self.process.join(timeout=0.5)

                if self.process.exitcode is None:
                    logger.error("Process refuses to die, using SIGKILL")
                    os.kill(self.process.pid, signal.SIGKILL)
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")
        finally:
            if self.process is not None:
                if self.process.exitcode is not None:
                    self.process.close()
                else:
                    logger.warning("Process still running after cleanup attempts, cannot close.")
                self.process = None

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command by saving it to scripts/MyMethod.py
        and running the custom evaluation script.
        """
        logger.debug(f"REPL is executing code (reset_session={reset_session})")
        
        # Create scripts directory if it doesn't exist
        scripts_dir = self.working_dir / "input/methods"
        scripts_dir.mkdir(exist_ok=True)
        
        # Save the code to scripts/MyMethod.py
        method_file = scripts_dir / "LLMMethod.py"
        with open(method_file, "w") as f:
            f.write(code)
        
        start_time = time.time()
        
        # Run the custom evaluation script
        import subprocess
        process = subprocess.Popen(
            ["python", "input/main.py", "-m", "llm_method", "-p", "dev"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.working_dir)
        )
        
        # Set a timeout
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
            output = []
            
            # Capture stdout and stderr as separate lines
            if stdout:
                output.extend(stdout.splitlines())
            if stderr:
                output.extend(stderr.splitlines())
                
            exec_time = time.time() - start_time
            returncode = process.returncode
            
            # Try to extract exception information from stderr
            exc_type = None
            exc_info = None
            exc_stack = None
            
            if returncode != 0:
                # Try to parse exception information from stderr
                import re
                exc_match = re.search(r"([A-Za-z]*Error|[A-Za-z]*Exception):", stderr)
                if exc_match:
                    exc_type = exc_match.group(1)
                    exc_info = {"msg": stderr.strip()}
                    # Create a simplified stack trace from the error message
                    stack_lines = stderr.splitlines()
                    exc_stack = [(f"MyMethod.py", 0, "unknown", line) for line in stack_lines if "File " in line]
                else:
                    exc_type = "RuntimeError"
                    exc_info = {"msg": f"Process exited with code {returncode}"}
                
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            output = []
            
            if stdout:
                output.extend(stdout.splitlines())
            if stderr:
                output.extend(stderr.splitlines())
                
            output.append(f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}")
            exec_time = self.timeout
            exc_type = "TimeoutError"
            exc_info = {"msg": "Execution timed out"}
            exc_stack = None
        
        # Add execution time message
        if exc_type != "TimeoutError":
            output.append(f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)}).")
        
        return ExecutionResult(output, exec_time, exc_type, exc_info, exc_stack)
