import os
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from dotenv import load_dotenv
from accelerate import Accelerator
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger

load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_LLM(metaclass=Meta_Singleton):

    # Extraction: "google/flan-t5-base"
    # Generation & Preference: "meta-llama/Llama-2-7b-chat-hf"
   
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.logger = Logger()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.logger.add_log("INFO", "llm", "Manager_LLM", "__init__", "CUDA is available. Loading model on GPU...")
        else:
            self.logger.add_log("WARNING", "llm", "Manager_LLM", "__init__", "CUDA is not available. Running on CPU.")
        
        try:
            self.tokenizer = self._load_tokenizer(model_name)
            config = self._load_config(model_name)
            quant_config = self._get_quantization_config()
            self.model = self._load_model(model_name, config, quant_config)
            self._ensure_hidden_states_enabled()
            self.model = self._compile_model(self.model)
            self.accelerator = Accelerator()
            self.model = self.accelerator.prepare(self.model)
            self.logger.add_log("INFO", "llm", "Manager_LLM", "__init__", "Model successfully loaded and initialized.")
        except Exception as e:
            self.logger.add_log("ERROR", "llm", "Manager_LLM", "__init__", f"Failed to load model: {str(e)}")
            raise e

    def _load_tokenizer(self, model_name: str):
        self.logger.add_log("INFO", "llm", "Manager_LLM", "_load_tokenizer", f"Loading tokenizer for {model_name}...")
        return AutoTokenizer.from_pretrained(model_name, token=secret_key)

    def _load_config(self, model_name: str):
        self.logger.add_log("INFO", "llm", "Manager_LLM", "_load_config", f"Loading configuration for {model_name}...")
        return AutoConfig.from_pretrained(model_name, token=secret_key)

    def _get_quantization_config(self):
        self.logger.add_log("INFO", "llm", "Manager_LLM", "_get_quantization_config", "Setting up quantization config for 4-bit mode...")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )

    def _load_model(self, model_name: str, config, quant_config):
        if config.is_encoder_decoder:
            self.logger.add_log("INFO", "llm", "Manager_LLM", "_load_model", "Model is encoder-decoder. Using AutoModelForSeq2SeqLM.")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quant_config,
                token=secret_key
            )
        else:
            self.logger.add_log("INFO", "llm", "Manager_LLM", "_load_model", "Model is causal. Using AutoModelForCausalLM.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quant_config,
                token=secret_key
            )
        return model

    def _ensure_hidden_states_enabled(self):
        if not self.model.config.output_hidden_states:
            self.logger.add_log("INFO", "llm", "Manager_LLM", "_ensure_hidden_states_enabled", "Enabling hidden states for embedding generation.")
            self.model.config.output_hidden_states = True

    def _compile_model(self, model):
        if hasattr(torch, "compile"):
            self.logger.add_log("INFO", "llm", "Manager_LLM", "_compile_model", "Compiling model with torch.compile for speed improvements...")
            model = torch.compile(model)
        return model

    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0) -> str:
        try:
            with torch.inference_mode():
                inputs = self.tokenizer(prompt, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=temperature
                )
                if self.model.config.is_encoder_decoder:
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    prompt_length = inputs["input_ids"].shape[1]
                    new_tokens = outputs[0][prompt_length:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                self.logger.add_log("INFO", "response", "Manager_LLM", "generate_response", f"Generated response successfully: {response}")
                return response
        except Exception as e:
            self.logger.add_log("ERROR", "response", "Manager_LLM", "generate_response", f"Error generating response: {str(e)}")
            return ""
        
    def generate_embedding(self, text: str) -> list:
        try:
            with torch.inference_mode():
                inputs = self.tokenizer(text, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if self.model.config.is_encoder_decoder:
                    encoder_outputs = self.model.get_encoder()(**inputs)
                    hidden_states = encoder_outputs.last_hidden_state
                else:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None
                if hidden_states is None:
                    self.logger.add_log("WARNING", "embedding", "Manager_LLM", "generate_embedding", "No hidden states returned.")
                    return []
                pooled = torch.mean(hidden_states, dim=1)
                embedding_vector = pooled.squeeze(0).tolist()
                self.logger.add_log("INFO", "embedding", "Manager_LLM", "generate_embedding", "Generated embedding successfully.")
                return embedding_vector
        except Exception as e:
            self.logger.add_log("ERROR", "embedding", "Manager_LLM", "generate_embedding", f"Error generating embedding: {str(e)}")
            return []
