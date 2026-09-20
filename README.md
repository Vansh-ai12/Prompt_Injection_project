# Prompt Injection Defense System

A lightweight 3-layer defense pipeline for cybersecurity copilots to detect and prevent prompt injection attacks.

## Architecture

### Layer 1: Input Classifier (Trained Model)
- **Model**: distilbert-base-uncased (66M params) - fast training on Colab T4 GPU
- **Classes**: benign, direct_injection, indirect_injection, jailbreak
- **Training**: Google Colab notebook (`notebooks/train_classifier.ipynb`)
- **Inference**: CPU-only locally (no GPU required)
- **Training time**: ~30-45 minutes on Colab T4 GPU

### Layer 2: Canary Token Leak Detector (Pure Code)
- Injects random canaries into system prompts
- Scans outputs for canary leaks
- No ML required - pure string matching

### Layer 3: Tool-Call Intent Auditor (LLM Wrapper)
- **Model**: llama-3.1-8b-instant (fast, low-cost via Groq)
- Multi-step aware (looks at N=3 recent tool calls)
- Returns ALLOW/BLOCK/ESCALATE with reason

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Classifier (Google Colab)
1. Open `notebooks/train_classifier.ipynb` in Google Colab
2. Run cells top-to-bottom (uses free T4 GPU, ~30-45 min)
3. Download the trained model zip file
4. Extract to `models/classifier/`

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Groq API key
```

### 4. Run the API
```bash
uvicorn src.main:app --reload
```

## Project Structure
```
Prompt_Injection_project/
├── data/              # datasets
├── models/            # trained model files
├── src/               # source code (flat structure)
│   ├── layer1_classifier.py
│   ├── layer2_canary.py
│   ├── layer3_auditor.py
│   ├── main.py
│   └── config.py
├── tests/             # test scripts
├── logs/              # runtime logs
├── notebooks/         # Colab training notebook
└── requirements.txt
```

## Research Paper Contribution
- **ML**: Lightweight distilbert classifier trained on Colab
- **Systems**: 3-layer defense pipeline architecture
- **Experimental**: Evaluation metrics via `/simulate_attack` endpoint

## Key Design Decisions
- **Lightweight Model**: distilbert-base-uncased (66M) for fast training
- **Small Dataset**: ~500-1000 examples for initial working version
- **Fast Groq Model**: llama-3.1-8b-instant for speed and low cost
- **CPU Inference**: No GPU required for local pipeline
- **Simple Structure**: Flat src/ with 5 files, no over-engineering
