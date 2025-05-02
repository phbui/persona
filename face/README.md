Project Overview:

- ai:
  Houses the classes that run the AI

- data:
  Houses the inputs and outputs of the system

- models:
  Houses the trained RL model and fine-tuned PPO model

- ui:
  Houses the UI code for human feedback training, automated fine-tuned LLM training, and exploitation (called novel generation)

How to Run:

- Requirements:
  Atleast 16 GB VRAM
  Preferably 32 GB VRAM NVIDIA GPU

- Libraries:
  pip install -r requimrents.txt

- Secret Keys:
  Create a .env file at the root of the project with a Hugging Face key (hf_key=...)

- Running:
  pyhton app.py
  (The different UI interactions are guided by the wizard)
