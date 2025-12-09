from typing import List

from openai import OpenAI

from model_client import ModelClient


class OpenAIEndpoint(ModelClient):
    def __init__(self, base_url: str, api_key: str, model_name: str, system_prompt: str = "You are a helpful assistant"):
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = {
            "role": "system",
            "content": system_prompt
        }

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        available_models = self.get_models()

        if not model_name in available_models:
            raise ValueError(
                f"No model {model_name} available at endpoint {base_url}!\nAvailable models: {available_models}"
            )

        self.model_name = model_name

    def get_response(self, prompt: str, previous_messages: list = []) -> str:
        messages = []
        if not previous_messages:
            messages.append(self.system_prompt)
        else:
            messages = previous_messages

        messages.append({"role": "user", "content": prompt})

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )

        return chat_completion.choices[0].message.content or ""

    def get_models(self) -> List[str]:
        models = self.client.models.list().data
        model_names = [m.id for m in models if m.object == 'model']

        return model_names
