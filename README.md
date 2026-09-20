# Prompt Injection Defense System

A lightweight 3-layer defense pipeline for cybersecurity copilots to detect and prevent prompt injection attacks.

## Quick Start

```bash
python setup.py && python run_simulation.py
```

This single command handles everything: virtual environment, dependencies, model download, and simulation.

## Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Step-by-step setup instructions
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and data flow explanation

## Architecture

### Layer 1: Input Classifier (Trained Model)
- **Model:** microsoft/deberta-v3-base (280M params) - downloads automatically or train in Colab
- **Classes:** benign, direct_injection, indirect_injection, jailbreak
- **Training:** Google Colab notebook (`notebooks/train_classifier.ipynb`)
- **Inference:** CPU-only locally (no GPU required)
- **Training time:** ~45-60 minutes on Colab T4 GPU

### Layer 2: Canary Token Leak Detector (Pure Code)
- Injects random canaries into system prompts
- Scans outputs for canary leaks
- No ML required - pure string matching

### Layer 3: Tool-Call Intent Auditor (LLM Wrapper)
- **Model:** llama-3.1-8b-instant (fast, low-cost via Groq)
- Multi-step aware (looks at N=3 recent tool calls)
- Returns ALLOW/BLOCK/ESCALATE with reason

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
├── setup.py           # Automated setup script
├── run_simulation.py  # Single-command simulation
└── requirements.txt
```

## Research Paper Contribution

- **ML:** DEBERTa-v3-base classifier trained on Colab
- **Systems:** 3-layer defense pipeline architecture
- **Experimental:** Evaluation metrics via simulation script

## Key Design Decisions

- **DEBERTa Model:** microsoft/deberta-v3-base (280M) for better accuracy
- **Auto-Download:** Model downloads automatically via setup script
- **Fast Groq Model:** llama-3.1-8b-instant for speed and low cost
- **CPU Inference:** No GPU required for local pipeline
- **Simple Structure:** Flat src/ with 5 files, no over-engineering

## API Endpoints

- `POST /defend` - Single message through 3-layer defense pipeline
- `POST /simulate_attack` - Batch testing for research evaluation
- `GET /health` - Health check
- `GET /` - System information

## Testing

```bash
# Run single-command simulation
python run_simulation.py

# Start API server
uvicorn src.main:app --reload

# Run comprehensive test suite
python tests/test_simulation.py
```

## License

MIT License
