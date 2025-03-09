import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_LLM:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        local_models_dir = os.path.join(os.path.dirname(__file__), 'models')
        local_model_path = os.path.join(local_models_dir, model_name)
        
        if os.path.isdir(local_model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path)
            print(f"Loaded '{model_name}' from local directory.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=secret_key)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=secret_key)
            print(f"Loaded '{model_name}' from Hugging Face repository.")
        
        self.model.to(self.device)


    def fine_tune(self, train_dataset, eval_dataset=None, output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
