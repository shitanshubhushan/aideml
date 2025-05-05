from methods.MyMethod import MyMethod
from methods.LLMMethod import LLMMethod

def all_method_handlers():
    """Enumerate and Load (import) all implemented methods."""
    loaded_methods = {
        "my_method": MyMethod,
        "llm_method": LLMMethod,
        # Add more methods here if needed
    }
    return loaded_methods

