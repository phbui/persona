import os
import re
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
    BitsAndBytesConfig,
    TextStreamer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from dotenv import load_dotenv
from copy import deepcopy
import traceback


load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_LLM:
    def __init__(self, parent=None, model_name="meta-llama/Meta-Llama-3-8B-Instruct", training=True):
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
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True 
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
        self._init_model(training)
        self.streamer = TextStreamer(self.tokenizer)
        
    def _init_model(self, training=True):
        """Initialize model with support for existing checkpoints"""
        print(f"Loading base model {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=secret_key,
            quantization_config=self.quant_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )
        if training:
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

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable params: {trainable} / {total}")

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

            self.model = prepare_model_for_kbit_training(self.model)
            
            adapter_path = os.path.join(load_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                self.model = PeftModel.from_pretrained(self.model, load_path)
            else:
                self._apply_qlora()
        return self

    def fine_tune(
        self,
        training_data,
        output_dir: str = "models/llm/finetuned_llm",
        num_epochs: int = 10,
        val_split: float = 0.1,
    ):
        """Fine-tune the LLM on HF comparisons and monitor validation loss."""
        self.toggle_mode(training=True)
        best_val = None

        try:
            print("Starting fine-tuning …")

            raw_ds   = Dataset.from_list(training_data)
            ds_split = raw_ds.train_test_split(test_size=val_split, seed=42)
            train_ds, val_ds = ds_split["train"], ds_split["test"]

            def tokenize_fn(examples):
                full_prompts = [
                    f"<s>[INST] {p.strip()} [/INST] {r.strip()}</s>"
                    for p, r in zip(examples["prompt"], examples["response"])
                ]
                return self.tokenizer(
                    full_prompts,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )

            train_tok = train_ds.map(tokenize_fn, batched=True,
                                    remove_columns=["prompt", "response"])
            val_tok   = val_ds.map(tokenize_fn, batched=True,
                                remove_columns=["prompt", "response"])

            train_tok.set_format("torch", columns=["input_ids", "attention_mask"])
            val_tok.set_format("torch", columns=["input_ids", "attention_mask"])

            training_args = TrainingArguments(
                output_dir               = output_dir,
                num_train_epochs         = num_epochs,
                evaluation_strategy      = "epoch",
                save_strategy            = "epoch",
                load_best_model_at_end   = True,
                metric_for_best_model    = "eval_loss",
                greater_is_better        = False,
                remove_unused_columns    = True,
                per_device_train_batch_size = 2,     
                dataloader_num_workers   = 4,          
                dataloader_pin_memory    = True,        
                gradient_accumulation_steps = 1,     
                fp16                     = True,                          
            )

            trainer = Trainer(
                model         = self.model,
                args          = training_args,
                train_dataset = train_tok,
                eval_dataset  = val_tok,
                data_collator = DataCollatorForLanguageModeling(
                                    self.tokenizer,
                                    pad_to_multiple_of=8,
                                    mlm=False),
                callbacks     = [EarlyStoppingCallback(
                                    early_stopping_patience=2)]
            )

            train_output = trainer.train()
            best_val = train_output.metrics.get("eval_loss")
            print(f"train_output")

            self.save_model(output_dir)

        except Exception as e:
            print("Trainer crashed:", e)
            traceback.print_exc()

        finally:
            self.toggle_mode(training=False)
            return best_val

    def generate_response(self, prompt, max_new_tokens=72, temperature=0.3, top_p=0.9):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,              
                temperature=temperature,      
                top_p=top_p,                 
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate_training_text(
        self,
        character_description: str,
        situation: str,
        face_descriptions: str,
        ranked_face_indices: list[int],
        invalid_face_indices: list[int],
    ):
        user_msg = f"""
        Analyze the Character Description, Character Situation, and 5 Generated Faces.

        Output **exactly two lines**:
        Valid Faces: [a, b, c, d, e]      # all valid faces, best → worst
        Invalid Faces: [i, j]              # any indices that are invalid

        ### Character Description:
        {character_description}

        ### Character Situation:
        {situation}

        ### Generated Faces:
        {face_descriptions}
        """.strip()

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant that always follows the output format exactly.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_msg}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "Valid Faces: ["
        )

        ranked = ", ".join(map(str, ranked_face_indices))
        invalid = ", ".join(map(str, invalid_face_indices))
        response = f"{ranked}]\nInvalid Faces: [{invalid}]<|eot_id|>"

        return response, prompt

    def extract_faces_from_response(self, response: str, generated_faces: list):
        response = response[-100:].strip()

        def grab(label: str) -> list[int]:
            pat = rf"{label}\s*Faces\s*:\s*\[?([0-9,\s]*)\]?"
            matches = list(re.finditer(pat, response, flags=re.I))
            if not matches:
                return []
            last = matches[-1]
            return [int(x) for x in re.findall(r"\d+", last.group(1))]

        valid_idx   = grab("Valid")          # ranked, best → worst (may be empty)
        parsed_inv  = grab("Invalid")        # any indices the model explicitly called invalid

        all_idx     = set(range(len(generated_faces)))
        invalid_idx = sorted(all_idx - set(valid_idx))         # default fill-in
        if parsed_inv:                                         # if model provided any
            invalid_idx = sorted(set(parsed_inv) | set(invalid_idx))

        valid_faces   = [generated_faces[i] for i in valid_idx   if 0 <= i < len(generated_faces)]
        invalid_faces = [generated_faces[i] for i in invalid_idx if 0 <= i < len(generated_faces)]
        
        return valid_faces, invalid_faces, valid_idx

    def auto_generate_face_feedback(self, character_description, situation, generated_faces, describe_face_fn):
        face_descriptions = ""
        for i, face in enumerate(generated_faces):
            face_descriptions += f"{i}: {describe_face_fn(face['aus'])}\n"
        
        user_msg = f"""
            Analyze the Character Description, Character Situation, and 5 Generated Faces.

            Output **exactly two lines**:
            Valid Faces: [a, b, c, d, e]      # all valid faces, best → worst
            Invalid Faces: [i, j]              # any indices that are invalid

            ### Character Description:
            {character_description}

            ### Character Situation:
            {situation}

            ### Generated Faces:
            {face_descriptions}
            """.strip()

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant that always follows the output format exactly.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_msg}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "Valid Faces: ["
        )

        response = self.generate_response(prompt)
        valid_faces, invalid_faces, _ = self.extract_faces_from_response(response, generated_faces)

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
    
    manager = Manager_LLM(training=False)
    manager.load_model(os.path.join(models_dir, selected_model))
    
    print("Chat interface (type 'exit' to quit)")
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = manager.generate_response(prompt)
        print(f"Response: {response}")