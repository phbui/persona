import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM:
    def __init__(self, secret_key, model_name="mistralai/Mistral-7B-Instruct-v0.3", peft_dir=None):
        # Set device
        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA is available. Loading model on GPU...")
        else:
            self.device = "cpu"
            print("Warning: CUDA is not available. Running on CPU.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=secret_key
        )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=secret_key
        )
        
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate output tokens
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Decode the output tokens to a string
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
