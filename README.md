# Prompt Injection Defense System

A 3-layer defense pipeline for cybersecurity copilots to detect and prevent prompt injection attacks.

## Architecture

### Layer 1: Input Classifier (Trained Model)
- Fine-tuned DeBERTa classifier for injection detection
- Classifies messages as: benign / direct_injection / indirect_injection / jailbreak
- **Training**: Done in Google Colab (free tier T4 GPU) - see `src/layer1/train_colab.ipynb`
- **Inference**: Runs on CPU locally (no GPU required)

### Layer 2: Heuristic/Rule Checks (No ML)
- Canary token detection in system prompts
- Delimiter/role-confusion detection
- Simple code-based checks

### Layer 3: Output/Tool-Call Auditor (LLM Wrapper)
- Validates tool calls against user intent
- Uses Groq-hosted LLM as judge
- Policy enforcement before execution

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Layer 1 Classifier (Google Colab)
1. Open `src/layer1/train_colab.ipynb` in Google Colab
2. Run cells top-to-bottom (uses free T4 GPU)
3. Download the trained model zip file
4. Extract to `models/saved/classifier/`

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Groq API key and model path
```

### 4. Run the API
```bash
uvicorn main:app --reload
```

## Research Paper Contribution
- Trained injection classifier (ML contribution)
- Layered defense pipeline architecture (systems contribution)
- Evaluation on attack datasets (experimental contribution)

## Key Design Decisions
- **Colab Training**: Training done in Google Colab to avoid local GPU requirements
- **CPU Inference**: Trained model runs on CPU locally for the defense pipeline
- **Frequent Checkpointing**: Colab notebook saves every 500 steps to handle session disconnections
- **Layer Separation**: Clear distinction between trained model (Layer 1), pure code (Layer 2), and LLM wrapper (Layer 3)
