import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from dotenv import load_dotenv
from accelerate import Accelerator
load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_Encoder():
    def __init__(self, model_name="google/flan-t5-base"):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = self._load_tokenizer(model_name)
            quant_config = self._get_quantization_config()
            self.model = self._load_model(model_name, quant_config)
            self._ensure_hidden_states_enabled()
            self.model = self._compile_model(self.model)
            self.accelerator = Accelerator()
            self.model = self.accelerator.prepare(self.model)
        except Exception as e:
            raise e

    def _load_tokenizer(self, model_name: str):
        return AutoTokenizer.from_pretrained(model_name, token=secret_key)

    def _get_quantization_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )

    def _load_model(self, model_name: str, quant_config):
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            token=secret_key
        )

    def _ensure_hidden_states_enabled(self):
        if not self.model.config.output_hidden_states:
            self.model.config.output_hidden_states = True

    def _compile_model(self, model):
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        return model

    def generate_embedding(self, text: str) -> list:
        try:
            with torch.inference_mode():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                encoder_outputs = self.model.get_encoder()(**inputs)
                hidden_states = encoder_outputs.last_hidden_state  # Shape: (1, seq_len, hidden_size)

                if hidden_states is None:
                    print("Zero vector detected.")
                    return [0.0] * self.model.config.d_model  # Fallback to zero vector

                pooled = torch.mean(hidden_states, dim=1)  # Shape: (1, hidden_size)
                embedding_vector = pooled.squeeze(0).cpu().tolist()  # Convert to list

                # Ensure fixed size
                hidden_size = self.model.config.d_model  # Get model's hidden size
                if len(embedding_vector) != hidden_size:
                    embedding_vector = (embedding_vector + [0.0] * hidden_size)[:hidden_size]  # Pad or truncate

                return embedding_vector  # Guaranteed fixed size
        except Exception as e:
            return [0.0] * self.model.config.d_model  # Return zero vector on failure
