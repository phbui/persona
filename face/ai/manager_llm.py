import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('hf_key')


class Manager_LLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        local_models_dir = "models/llm"
        os.makedirs(local_models_dir, exist_ok=True)

        if model_path and os.path.isdir(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"Loaded finetuned model from {model_path}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=secret_key)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=secret_key)
            print(f"Loaded pretrained model '{model_name}' from Hugging Face.")

        self.model.to(self.device)

    def save_model(self, save_path="models/llm/finetuned_llm"):
        """Saves the fine-tuned LLM model to disk."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Saved LLM to {save_path}")

    def load_model(self, load_path):
        """Loads a saved fine-tuned LLM."""
        if os.path.isdir(load_path):
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModelForCausalLM.from_pretrained(load_path)
            print(f"Loaded fine-tuned LLM from {load_path}")
        else:
            print("No saved LLM model found! Using default.")

    def fine_tune(self, rankings, output_dir="models/llm/finetuned_llm", num_train_epochs=3, batch_size=8):
        """Fine-tunes the LLM based on human rankings."""
        train_dataset = self.prepare_llm_training_data(rankings)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        self.save_model(output_dir)

    def prepare_llm_training_data(self, rankings):
        """Converts human rankings into a training dataset for fine-tuning."""
        formatted_data = []
        for situation, ranked_faces in rankings.items():
            ranked_text = "\n".join([f"{i+1}: {face}" for i, face in enumerate(ranked_faces)])
            prompt = f"Rank the following facial expressions for the situation: {situation}\n\n{ranked_text}\n\nRanking: "
            ranking_output = ", ".join(str(i+1) for i in range(len(ranked_faces)))
            formatted_data.append({"input_text": prompt, "target_text": ranking_output})

        return Dataset.from_dict(formatted_data)

    def generate_response(self, prompt, max_new_tokens=256, temperature=1.0):
        """Generates a ranking response from the LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
