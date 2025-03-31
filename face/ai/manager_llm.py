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
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_LLM:
    def __init__(self, parent, model_name="mistralai/Mistral-7B-v0.1", model_path=None):
        self.parent = parent
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        local_models_dir = "models/llm"
        os.makedirs(local_models_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=secret_key)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if model_path is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
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

    def save_model(self, save_path="models/llm/finetuned_llm"):
        """Saves the fine-tuned LLM model with LoRA adapters."""
        os.makedirs(save_path, exist_ok=True)
        
        self.tokenizer.save_pretrained(save_path)
        
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_path)
            print(f"Saved fine-tuned LLM with LoRA adapters to {save_path}")
        else:
            print(f"Warning: No LoRA adapters found to save. Only base model saved.")

    def load_model(self, load_path):
        """Loads a saved fine-tuned LLM with LoRA adapters if available."""
        if os.path.isdir(load_path):
            print(f"Loading base model ({self.model_name}) and tokenizer...")

            # Always load tokenizer from the checkpoint
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            except Exception as e:
                print(f"Tokenizer load failed from {load_path}, falling back to base. Error: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=secret_key)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load the base model first
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=secret_key,
                quantization_config=self._get_quantization_config(),
                device_map="auto"
            )

            # Apply LoRA adapter if found
            adapter_path = os.path.join(load_path, "adapter_model.safetensors")
            config_path = os.path.join(load_path, "adapter_config.json")

            if os.path.exists(adapter_path) and os.path.exists(config_path):
                print(f"Loading LoRA adapter from {load_path}...")
                self.model = PeftModel.from_pretrained(self.model, load_path, is_trainable=True)
                print("Successfully loaded LoRA adapter.")
            else:
                print("Warning: LoRA adapter files not found, applying fresh QLoRA adapter.")
                self._apply_qlora()
        else:
            print("No saved LLM model directory found! Using default base model.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=secret_key)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=secret_key,
                quantization_config=self._get_quantization_config(),
                device_map="auto"
            )
            self._apply_qlora()

        return self

    def fine_tune(self, training_data, output_dir="models/llm/finetuned_llm"):
        """Fine-tunes the LLM based on human rankings at the end of each epoch."""
        self.model.config.use_cache = False
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
            num_train_epochs=1,
            per_device_train_batch_size=1,
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

        train_result = trainer.train()
        final_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
        if final_loss is not None:
            self.parent.manager_loss.store_loss(final_loss)
            self.parent.manager_loss.end_epoch(trainer.state.epoch)
            print(f"Final loss: {final_loss}")

        self.save_model(output_dir)

    def generate_response(self, prompt, max_new_tokens=512, temperature=1.0):
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


    def auto_generate_face_feedback(self, character_description, situation, generated_faces, describe_face_fn):
        # Build a textual description of the faces using the provided description function.
        face_descriptions = ""
        for i, face in enumerate(generated_faces):
            face_descriptions += f"{i}: {describe_face_fn(face)}\n"
        
        # Construct the prompt using the training data format.
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
        
        # Generate a response from the fine-tuned LLM using the constructed prompt.
        # Here, we assume generate_training_text accepts a single prompt string.
        print(f"prompt: {prompt}")
        response = self.generate_response(prompt)
        print(f"response: {response}")
        
        # Parse the LLM response to extract valid and invalid face indices.
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
    # Determine the absolute path of the models directory relative to this file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "../models/llm")
    
    # List all subdirectories (each assumed to be a fine-tuned model)
    if not os.path.isdir(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        sys.exit(1)
    
    available_models = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]
    
    if not available_models:
        print("No fine-tuned models found in the directory.")
        sys.exit(1)
    
    print("Available fine-tuned models:")
    for idx, model_name in enumerate(available_models):
        print(f"{idx}: {model_name}")
    
    # Prompt user to select a model
    try:
        selection = int(input("Select a model by index: "))
        selected_model = available_models[selection]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)
    
    selected_model_path = os.path.join(models_dir, selected_model)
    print(f"Loading model from: {selected_model_path}")
    
    # Create a Manager_LLM instance (parent can be None if not used in this test)
    manager = Manager_LLM(parent=None)
    manager.load_model(selected_model_path)
    
    # Main loop to prompt the model.
    print("Enter a prompt for the model. Type 'exit' to quit.")
    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == "exit":
            break
        
        # Generate response with some new tokens
        response = manager.generate_response(prompt)
        print("Response:", response)