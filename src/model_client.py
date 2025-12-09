from typing import Optional
from functools import wraps


class ModelClient:
    def __init__(self):
        pass

    def get_response(self, prompt: str, previous_messages: list = []) -> str:
        pass


# this is the global api client
# reused across modules
_model_client: Optional[ModelClient] = None


def set_global_model_client(client: ModelClient) -> None:
    """Set the global API client (to be used for dependency injection)."""
    global _model_client
    _model_client = client


def get_model_client() -> ModelClient:
    """Get the global API client instance. Raises an error if not initialized."""
    if _model_client is None:
        raise ValueError("API client not initialized.")
    return _model_client


def inject_model_client(func):
    """
    A decorator that injects the API client into the function.
    The function must accept an 'model_client' keyword argument.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "model_client" not in kwargs:
            kwargs["model_client"] = get_model_client()
        return func(*args, **kwargs)
    return wrapper
