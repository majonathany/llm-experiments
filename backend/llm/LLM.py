from typing import Optional, List, Any, Mapping

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM, LLM
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper(BaseLLM):
    model_name_or_path: str = None  # Assuming dynamic definition is not desired/possible
    model: AutoModelForCausalLM = None  # Assuming dynamic definition is not desired/possible
    tokenizer: AutoTokenizer = None  # Assuming dynamic definition is not desired/possible


    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

        if self.is_awq():
            from awq import AutoAWQForCausalLM


            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            # Load model
            model = AutoAWQForCausalLM.from_quantized(model_name_or_path, trust_remote_code=True,
                                                  fuse_layers=True,
                                                  safetensors=True,
                                                  use_cache=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.tokenizer = tokenizer
        self.model = model

    def is_awq(self):
        return "awq" in self.model_name_or_path.lower()

    def run(self, prompt, **kwargs):
        if self.is_awq():
            from awq import AutoAWQForCausalLM

        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Generate a response
        output = self.model.generate(**inputs, **kwargs)
        # Decode the generated tokens to a string
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def _generate(self, prompt, **kwargs):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Generate a response
        output = self.model.generate(**inputs, **kwargs)
        # Decode the generated tokens to a string
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _llm_type(self):
        # Return a string that describes the type of LLM or model
        return "Mistral7B128kAWQ"

class CustomLLM(LLM):
    model_name_or_path: str
    model: str
    tokenizer: str

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        self.tokenizer = AutoTokenizer.from_pretrained('Supabase/gte-small', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('Supabase/gte-small', trust_remote_code=True)


        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Generate a response
        output = self.model.generate(**inputs, **kwargs)
        # Decode the generated tokens to a string
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name_or_path": self.model_name_or_path, "model": self.model, "tokenizer": self.tokenizer}

class CustomLL(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}




def initialize_model():
    model_name_or_path = "/home/ubuntu/llm_experiments/Yarn-Mistral-7B-128k-AWQ"

    return CustomLLM(model_name_or_path=model_name_or_path, model=model_name_or_path, tokenizer=model_name_or_path)