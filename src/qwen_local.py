import os

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig

from model_client import ModelClient


class QwenLocal(ModelClient):
    def __init__(self, model_path: str, system_prompt: str = "", max_tokens: int = 32768, device="cuda", enable_thinking=False, adapter_path=None):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.device = device
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.adapter_path = adapter_path

        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}!")

        print(f"loading model at {model_path}")
        self.load_model(model_path, adapter_path)
        print(f"Done.")

    def get_response(self, prompt: str = "", messages: list = None) -> str:
        if messages is not None and len(messages) > 0:
            input_messages = messages
        else:
            input_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        text = self.tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # parsing thinking content
        try:
            # 151668 == </think>
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content

    def load_model(self, model_path, adapter_path=None):
        print("Loading Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Loading Config")
        self.config = AutoConfig.from_pretrained(model_path)
        print(f"Loaded: {self.config}")
        print("Loading Model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            device_map=self.device
        )
        if adapter_path and os.path.isdir(adapter_path):
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # merge LoRA weights for faster inference
            self.model = self.model.merge_and_unload()
