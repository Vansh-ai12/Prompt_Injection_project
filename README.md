# Prompt Injection Defense System

A 3-layer defense pipeline for cybersecurity copilots to detect and prevent prompt injection attacks.

## Architecture

### Layer 1: Input Classifier (Trained Model)
- Fine-tuned DeBERTa classifier for injection detection
- Classifies messages as: benign / direct-injection / indirect-injection / jailbreak
- Fast, runs on local GPU

### Layer 2: Heuristic/Rule Checks (No ML)
- Canary token detection in system prompts
- Delimiter/role-confusion detection
- Simple code-based checks

### Layer 3: Output/Tool-Call Auditor (LLM Wrapper)
- Validates tool calls against user intent
- Uses Groq-hosted LLM as judge
- Policy enforcement before execution

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Research Paper Contribution
- Trained injection classifier (ML contribution)
- Layered defense pipeline architecture (systems contribution)
- Evaluation on attack datasets (experimental contribution)
