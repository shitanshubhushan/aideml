"""Backend for OpenAI API."""

import json
import os
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai

import dotenv

dotenv.load_dotenv()

logger = logging.getLogger("aide")

#_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

MODEL2PRICE = {
        "gpt-4o-2024-11-20" : {
            "input" : 2.5 / 1e6,
            "output" : 10 / 1e6,
            },
        "gpt-4o-mini-2024-07-18" : {
            "input" : 0.15 / 1e6,
            "output" : 0.6 / 1e6,
            },
        "o1-mini" : {
            "input" : 3 / 1e6,
            "output" : 12 / 1e6,
            },
        "o3-mini" : {
            "input" : 1.1 / 1e6,
            "output" : 4.4 / 1e6,
            },
        "o1-preview" : {
            "input" : 15 / 1e6,
            "output" : 60 / 1e6,
            },
        "o1" : {
            "input" : 15 / 1e6,
            "output" : 60 / 1e6,
            },
        }

@once
def _setup_openai_client():
    global _client
    openai_api_key = os.getenv('MY_OPENAI_API_KEY')
    openai_api_base = os.getenv('MY_AZURE_OPENAI_ENDPOINT')
    _client = openai.AzureOpenAI(
            max_retries=0,
            azure_endpoint=openai_api_base,
            api_key=openai_api_key,
            api_version="2024-12-01-preview",
            )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    If the model doesn't support function calling, gracefully degrade to text generation.
    """
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    # Convert system/user messages to the format required by the client
    messages = opt_messages_to_list(system_message, user_message)

    # If function calling is requested, attach the function spec
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    completion = None
    t0 = time.time()

    # Attempt the API call
    try:
        completion = backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except openai.BadRequestError as e:
        # Check whether the error indicates that function calling is not supported
        if "function calling" in str(e).lower() or "tools" in str(e).lower():
            logger.warning(
                "Function calling was attempted but is not supported by this model. "
                "Falling back to plain text generation."
            )
            # Remove function-calling parameters and retry
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)

            # Retry without function calling
            completion = backoff_create(
                _client.chat.completions.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        else:
            # If it's some other error, re-raise
            raise

    req_time = time.time() - t0
    choice = completion.choices[0]

    # Decide how to parse the response
    if func_spec is None or "tools" not in filtered_kwargs:
        # No function calling was ultimately used
        output = choice.message.content
    else:
        # Attempt to extract tool calls
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            logger.warning(
                "No function call was used despite function spec. Fallback to text.\n"
                f"Message content: {choice.message.content}"
            )
            output = choice.message.content
        else:
            first_call = tool_calls[0]
            # Optional: verify that the function name matches
            if first_call.function.name != func_spec.name:
                logger.warning(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {first_call.function.name}. Fallback to text."
                )
                output = choice.message.content
            else:
                try:
                    output = json.loads(first_call.function.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(
                        "Error decoding function arguments:\n"
                        f"{first_call.function.arguments}"
                    )
                    raise ex

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens
    curr_cost = in_tokens * MODEL2PRICE[completion.model]["input"] + out_tokens * MODEL2PRICE[completion.model]["output"] 

    LOG_DIR = os.getenv("LOG_DIR", "logs/")
    cost_file = os.path.join(LOG_DIR, "api_cost.json")
    content = dict()
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as reader:
            content = json.load(reader)

    with open(cost_file, 'w') as writer:
        updated_content = {
                "total_cost" : content.get("total_cost", 0) + curr_cost,
                "total_num_prompt_tokens" : content.get("total_num_prompt_tokens", 0) + in_tokens,
                "total_num_sample_tokens" : content.get("total_num_sample_tokens", 0) + out_tokens,
                }
        json.dump(updated_content, writer, indent=2)

    prompt_log_file = os.path.join(LOG_DIR, "history.txt")
    with open(prompt_log_file, 'a') as writer:
        writer.write(f"\n\n================= prompt ==================\n")
        for msg in messages:
            writer.write(f"{msg['role']}: {msg['content']}\n")
        writer.write(f"\n================= response ==================\n")
        writer.write(str(output))


    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
