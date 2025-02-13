# Persona: A RL NPC Dialogue System

A reinforcement learning (RL) driven dialogue system for generating immersive NPC responses. This system dynamically updates an NPC's internal mental state based on player input, dialogue embeddings, and emotion analysis. It uses large language models (LLMs) for response generation and multiple reward signals to guide the RL policy.

## Features

- **Dynamic Mental State:** Continuously updates based on player input and context.
- **Multi-modal Input:** Combines dialogue embeddings (via SentenceTransformer) and emotion analysis.
- **RL-Driven Adaptation:** Uses Proximal Policy Optimization (PPO) to learn optimal state updates for character-consistent responses.
- **LLM Integration:** Generates natural, character-driven dialogue responses.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up `.env`:**
   Place an `.env` in `persona/src`.
   Add a `hf_key` field.

3. **Run System:**
   ```bash
   python persona/app.y
   ```

5. **Usage:**
The system starts the game loop in a separate thread and opens a persistent chat interface for the user. The NPC's dialogue adapts in real-time using RL and LLM feedback.

## Configuration

- **Model Parameters:** Adjust input dimensions, action dimensions, and other hyperparameters in `src/ai/rl.py`.
- **Persona:** Define NPC backstory, goals, and mental state in a JSON file under `src/game/player/personas`.
