import os
import torch
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('hf_key')

class Manager_Analysis():
    def __init__(self, task_type, model):
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(task_type, model=model, token=secret_key, device=device)

    def analyze(self, text, **kwargs):
        return self.pipeline(text, **kwargs)
