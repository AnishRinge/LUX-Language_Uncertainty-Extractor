# LUX: Language Uncertainty Extractor

## Project Overview
LUX (Language Uncertainty Extractor) is a research project focused on **pre-generation hallucination-risk prediction** for open-weight Large Language Models (LLMs).

The goal is to predict the likelihood of a model hallucinating *before* it actually generates a response, by combining observable prompt-level features with the model's internal hidden representations.

## Key Principles
- **Research Project**: This repository follows a strict research methodology.
- **Pre-Generation Prediction**: Risk assessment occurs before the answer is generated.
- **Frozen Target LLM**: The target model (Qwen3 family) remains frozen.
- **Separate Predictor**: The hallucination risk predictor is a separate model from the target LLM.
- **Feature Fusion**: Predictor inputs include:
    - **Prompt Features**: Linguistic, syntactic, complexity, entity, and semantic features extracted from the input prompt.
    - **Hidden States**: Internal representations from the frozen target LLM.
- **No Generation Features**: Generated answers are **NOT** used as features for the predictor. They are only used to construct the empirical ground-truth target during training.

## Research Question
"Can combining observable prompt-level features with the target LLM's internal pre-generation hidden representations improve hallucination-risk prediction compared with using either information source alone?"

## Status
Current Phase: **Checkpoint 1 (Specification and Setup)**
- Repository structure established.
- Research decisions and constraints codified.
- Minimal environment configured.

**Checkpoint 2 has not started.**

## Repository Structure
- `configs/`: Machine-readable research configurations.
- `data/`: Raw and processed data, features, generations, and targets.
- `src/`: Source code for data processing, feature extraction, model interaction, and evaluation.
- `experiments/`: Experiment tracking and results.
- `RESEARCH_DECISIONS.md`: Detailed log of confirmed and unresolved research decisions.

## Installation
1. Create a virtual environment: `python -m venv .venv`
2. Activate the virtual environment: `.venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

## Running
To verify the current status:
```bash
python run_pipeline.py
```

