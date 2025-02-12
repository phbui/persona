import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('hf_key')

class LLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA is available. Loading model on GPU...")
        else:
            self.device = "cpu"
            print("Warning: CUDA is not available. Running on CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=secret_key  # updated parameter name for clarity
        )
        
        # Set up bitsandbytes quantization configuration for 4-bit (Q4) mode
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',  # or 'fp4' depending on your needs
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            token=secret_key  # ensure you authenticate if needed
        )
            
    def generate_response(self, prompt: str, max_new_tokens: int = 128) -> str:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            prompt_length = inputs["input_ids"].shape[1]
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False 
            )
            
            new_tokens = outputs[0][prompt_length:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)