import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from accelerate import Accelerator
from ...meta.meta_singleton import Meta_Singleton
from ...log.logger import Logger

load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_LLM(metaclass=Meta_Singleton):
    """
    Singleton class for managing the LLM model.
    """
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.logger = Logger()
        
        if torch.cuda.is_available():
            self.device = "cuda"
            self.logger.add_log("INFO", "llm", "Manager_LLM", "__init__", "CUDA is available. Loading model on GPU...")
        else:
            self.device = "cpu"
            self.logger.add_log("WARNING", "llm", "Manager_LLM", "__init__", "CUDA is not available. Running on CPU.")

        try:
            # Load the tokenizer.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=secret_key
            )
            
            # Set up bitsandbytes quantization configuration for 4-bit (Q4) mode.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load the model with quantization.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                token=secret_key
            )
            
            # Use torch.compile to optimize the model (PyTorch 2.0+).
            if hasattr(torch, "compile"):
                self.logger.add_log("INFO", "llm", "Manager_LLM", "__init__", "Compiling model with torch.compile for speed improvements...")
                self.model = torch.compile(self.model)
            
            # Set up the accelerator and prepare the model.
            self.accelerator = Accelerator()
            self.model = self.accelerator.prepare(self.model)
            
            self.logger.add_log("INFO", "llm", "Manager_LLM", "__init__", "Model successfully loaded and initialized.")
        except Exception as e:
            self.logger.add_log("ERROR", "llm", "Manager_LLM", "__init__", f"Failed to load model: {str(e)}")
            raise e
                
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0) -> str:
        try:
            with torch.inference_mode():
                inputs = self.tokenizer(prompt, return_tensors="pt")
                # Move inputs to the correct device.
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                prompt_length = inputs["input_ids"].shape[1]

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=temperature
                )

                new_tokens = outputs[0][prompt_length:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                self.logger.add_log("INFO", "response", "Manager_LLM", "generate_response", "Generated response successfully.")
                return response
        except Exception as e:
            self.logger.add_log("ERROR", "response", "Manager_LLM", "generate_response", f"Error generating response: {str(e)}")
            return ""