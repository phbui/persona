# Persona: Automatic NPC Generation Tool

**Persona** is a work-in-progress automatic NPC generation tool divided into two complementary modules: **Face** and **Speech**. These modules collaboratively produce immersive dialogues and expressive facial interactions for NPCs.

---

## Face: Human-Feedback Driven Facial Expression Generator

The **Face** module uses reinforcement learning with human feedback (RLHF) and a fine-tuned LLM to generate expressive, character-specific facial expressions via Facial Action Units (AUs).

### Paper
[LLMs Can Be Judgy Too](https://www.are.na/block/36403350)

### Features
- **Human-in-the-Loop Training**: Initial training guided by direct human feedback.
- **Automated Feedback**: Further training automated via fine-tuned LLM.
- **Facial AU Representation**: Adheres to Facial Action Coding System (FACS) standards.
- **GUI-based Workflow**: Visualizes faces, gathers human feedback, and automates evaluation.

### Setup and Usage
1. **Install Dependencies**:  
   `pip install -r requirements.txt`
2. **Setup Environment (.env)**:
   - Create `.env` in `persona/src`
   - Add: `hf_key=YOUR_HF_KEY`   
3. **Run GUI**:  
   `python face/app.py`
4. **Training Workflow**:
   - **Human Feedback**: Validate and rank facial expressions through GUI.
   - **Automated Mode**: Use fine-tuned LLM for automated evaluation and ranking.
   
---

## (WIP) Speech: RL-Driven NPC Dialogue System

The **Speech** module generates immersive NPC dialogues, dynamically updating an NPC's mental state based on player interactions, context, and emotional cues. It combines retrieval augmented generation (RAG) large language models (LLMs) with reinforcement learning (RL) techniques.

### Features
- **Dynamic Mental State**: Real-time updates based on player interactions.
- **Multi-modal Inputs**: Utilizes embeddings (SentenceTransformer) and emotional analysis.
- **RL-Driven Adaptation**: Employs Proximal Policy Optimization (PPO) for adaptive, consistent character responses.
- **LLM Integration**: Natural, contextually appropriate dialogues.

### Quick Start
1. **Install Dependencies**:  
   `pip install -r requirements.txt`
2. **Setup Environment (.env)**:
   - Create `.env` in `persona/src`
   - Add: `hf_key=YOUR_HF_KEY`
3. **Run Dialogue System**:  
   `python speech/app.py`

### Configuration
- **Model Parameters**: Modify settings in `src/ai/rl.py`
- **Character Personas**: JSON files located in `src/game/player/personas`
