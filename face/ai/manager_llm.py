import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_LLM:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        local_models_dir = "models/llm"
        os.makedirs(local_models_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=secret_key)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not (model_path and os.path.isdir(model_path)):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=secret_key,
                quantization_config=self._get_quantization_config(),
                device_map="auto"
            )
            print(f"Loaded pretrained model '{model_name}' from Hugging Face.")
            self._apply_qlora() 

    def _get_quantization_config(self):
        """Sets up 4-bit quantization for memory efficiency."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )

    def _apply_qlora(self):
        """Applies QLoRA to the model only if it has not been applied already."""
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
        else:
            print("QLoRA already applied. Skipping reapplication.")

    def _load_lora_adapter(self, model_path):
        """Loads LoRA adapters if they exist."""
        adapter_path = os.path.join(model_path, "adapter_model.bin")
        if os.path.exists(adapter_path):
            if not isinstance(self.model, PeftModel): 
                self.model = PeftModel.from_pretrained(self.model, model_path)
                print(f"Loaded LoRA adapter from {model_path}")
            else:
                print("LoRA adapter already applied. Skipping reapplication.")
        else:
            print(f"No LoRA adapter found in {model_path}. Using base model.")

    def save_model(self, save_path="models/llm/finetuned_llm"):
        """Saves the fine-tuned LLM model with LoRA adapters."""
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path) 
        self.tokenizer.save_pretrained(save_path)
        
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_path)
            print(f"Saved fine-tuned LLM with LoRA adapters to {save_path}")
        else:
            print(f"Warning: No LoRA adapters found to save. Only base model saved.")

    def load_model(self, load_path):
        """Loads a saved fine-tuned LLM with LoRA adapters if available."""
        if os.path.isdir(load_path):
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path, 
                token=secret_key,
                quantization_config=self._get_quantization_config(),
                device_map="auto"
            )


            adapter_path = os.path.join(load_path, "adapter_model.bin")
            if os.path.exists(adapter_path):
                self.model = PeftModel.from_pretrained(self.model, load_path)
                print(f"Loaded fine-tuned LLM with LoRA adapters from {load_path}")
            else:
                print(f"No LoRA adapter found in {load_path}. Using base model.")
        else:
            print("No saved LLM model found! Using default.")
        
        return self

    def fine_tune(self, training_data, output_dir="models/llm/finetuned_llm", num_train_epochs=1, batch_size=10):
        """Fine-tunes the LLM based on human rankings at the end of each epoch."""
        
        train_dataset = Dataset.from_list(training_data)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["prompt"], 
                text_target=examples["response"], 
                padding="max_length",
                truncation=True,
                max_length=512
            )

        tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            remove_unused_columns=False,  
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=self.tokenizer,  
            data_collator=data_collator,
        )

        trainer.train()
        self.save_model(output_dir)

    def generate_response(self, prompt, max_new_tokens=128, temperature=1.0):
        """Generates a ranking response from the LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=temperature
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
