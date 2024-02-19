from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models import BaseLLM

class LLMWrapper(BaseLLM):
    def __init__(self, model_name_or_path):
        from awq import AutoAWQForCausalLM

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # Load model
        model = AutoAWQForCausalLM.from_quantized(model_name_or_path, trust_remote_code=True,
                                              fuse_layers=True,
                                              safetensors=True,
                                              use_cache=False)


        self.tokenizer = tokenizer
        self.model = model

    def run(self, prompt, **kwargs):
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


def initialize_model():
    model_name_or_path = "/home/ubuntu/llm_experiments/Yarn-Mistral-7B-128k-AWQ"


    return LLMWrapper(model_name_or_path)