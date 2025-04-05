import os
import re
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TextStreamer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_LLM:
    def __init__(self, parent=None, model_name="mistralai/Mistral-7B-v0.1", model_path=None):
        self.parent = parent
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.local_models_dir = "models/llm"
        os.makedirs(self.local_models_dir, exist_ok=True)
        
        # Quantization configuration
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token=secret_key,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model initialization
        self._init_model(model_path)
        self.streamer = TextStreamer(self.tokenizer)
        
    def _init_model(self, model_path):
        """Initialize model with support for existing checkpoints"""
        if model_path and os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=self.quant_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16
            )
            if PeftModel.is_adapter_model(model_path):
                self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print(f"Loading base model {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=secret_key,
                quantization_config=self.quant_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16
            )
            self.model = prepare_model_for_kbit_training(self.model)
            self._apply_qlora()

    def _apply_qlora(self):
        """Apply QLoRA configuration"""
        if not hasattr(self.model, "peft_config"):
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Applied QLoRA to the model.")

    def toggle_mode(self, training=False):
        """Switch between training/inference modes"""
        if training:
            self.model.train()
            self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            torch.cuda.empty_cache()

    def save_model(self, save_path="models/llm/finetuned_llm"):
        """Save model with adapters"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Saved model to {save_path}")

    def load_model(self, load_path):
        """Load model with adapter support"""
        if os.path.isdir(load_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            except Exception as e:
                print(f"Using base tokenizer: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=secret_key)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=secret_key,
                quantization_config=self.quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            adapter_path = os.path.join(load_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                self.model = PeftModel.from_pretrained(self.model, load_path)
            else:
                self._apply_qlora()
        return self

    def fine_tune(self, training_data, output_dir="models/llm/finetuned_llm"):
        """Fine-tuning implementation with original parameters"""
        self.toggle_mode(training=True)
        try:
            train_dataset = Dataset.from_list(training_data)
            
            def tokenize_fn(examples):
                return self.tokenizer(
                    examples["prompt"],
                    padding="max_length",
                    truncation=True,
                    max_length=768,
                    return_tensors="pt"
                )
            
            tokenized_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "response"])
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                save_strategy="epoch",
                logging_dir=f"{output_dir}/logs",
                remove_unused_columns=False,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8)
            )
            
            trainer.train()
            self.save_model(output_dir)
            
        finally:
            self.toggle_mode(training=False)

    def generate_response(self, prompt, max_new_tokens=256, temperature=1.0):
        """Optimized generation with streaming support"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=self.streamer,
                repetition_penalty=1.1
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Maintain original interface methods
    def generate_training_text(self, character_description, situation, face_descriptions, valid_faces, invalid_faces):
        prompt = f"""
            ### Instruction:
            Analyze the Character Description, Character Situation, and Generated Faces. 
            Respond with:
            - A **ranked list of Valid Faces** (most accurate to the situation first).
            - A **list of Invalid Faces**.

            ### Character Description:
            {character_description}

            ### Character Situation:
            {situation}

            ### Generated Faces:
            {face_descriptions}

            ### Response Format:
            Valid Faces: [#, #, #, ...]
            Invalid Faces: [#, #, #, ...]
        """
            
        valid_faces = f"Valid Faces:\n{valid_faces}"
        invalid_faces = f"Invalid Faces:\n{invalid_faces}"

        response = valid_faces + "\n" + invalid_faces
        return response, prompt

    def auto_generate_face_feedback(self, character_description, situation, generated_faces, describe_face_fn):
        face_descriptions = ""
        for i, face in enumerate(generated_faces):
            face_descriptions += f"{i}: {describe_face_fn(face)}\n"
        
        prompt = f"""
            ### Instruction:
            Analyze the Character Description, Character Situation, and Generated Faces. 
            Respond with:
            - A **ranked list of Valid Faces** (most accurate to the situation first).
            - A **list of Invalid Faces**.

            ### Character Description:
            {character_description}

            ### Character Situation:
            {situation}

            ### Generated Faces:
            {face_descriptions}

            ### Response Format:
            Valid Faces: [#, #, #, ...]
            Invalid Faces: [#, #, #, ...]
        """
        
        response = self.generate_response(prompt)
        
        valid_faces = []
        invalid_faces = []
        
        valid_match = re.search(r"Valid\s*Faces\s*:\s*\[([^\]]+)\]", response, re.IGNORECASE)
        invalid_match = re.search(r"Invalid\s*Faces\s*:\s*\[([^\]]+)\]", response, re.IGNORECASE)
        
        if valid_match:
            valid_str = valid_match.group(1)
            valid_indices = [int(x.strip()) for x in valid_str.split(",") if x.strip().isdigit()]
            valid_faces = [generated_faces[i] for i in valid_indices if i < len(generated_faces)]
        
        if invalid_match:
            invalid_str = invalid_match.group(1)
            invalid_indices = [int(x.strip()) for x in invalid_str.split(",") if x.strip().isdigit()]
            invalid_faces = [generated_faces[i] for i in invalid_indices if i < len(generated_faces)]
        
        return valid_faces, invalid_faces, response

if __name__ == "__main__":
    # Original main execution flow preserved
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "../models/llm")
    
    if not os.path.isdir(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        sys.exit(1)
    
    available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not available_models:
        print("No models found")
        sys.exit(1)
    
    print("Available models:")
    for idx, name in enumerate(available_models):
        print(f"{idx}: {name}")
    
    try:
        selection = int(input("Select model: "))
        selected_model = available_models[selection]
    except (ValueError, IndexError):
        print("Invalid selection")
        sys.exit(1)
    
    manager = Manager_LLM()
    manager.load_model(os.path.join(models_dir, selected_model))
    
    print("Chat interface (type 'exit' to quit)")
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = manager.generate_response(prompt)
        print(f"Response: {response}")