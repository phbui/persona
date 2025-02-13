import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from accelerate import Accelerator

load_dotenv()
secret_key = os.getenv('hf_key')

class LLM:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLM, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA is available. Loading model on GPU...")
        else:
            self.device = "cpu"
            print("Warning: CUDA is not available. Running on CPU.")

        # Load the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=secret_key  # ensure you authenticate if needed
        )
        
        # Set up bitsandbytes quantization configuration for 4-bit (Q4) mode.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',  # or 'fp4' depending on your needs
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
            print("Compiling model with torch.compile for speed improvements...")
            self.model = torch.compile(self.model)
        
        # Set up the accelerator and prepare the model.
        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)
                
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0) -> str:
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
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
