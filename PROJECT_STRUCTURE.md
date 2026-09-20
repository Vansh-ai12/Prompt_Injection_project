# Project Structure

```
Prompt_Injection_project/
├── data/
│   ├── raw/                      # Downloaded datasets (deepset, JailbreakBench, synthetic)
│   └── processed/                # Preprocessed training/validation/test splits
├── models/
│   └── saved/                    # Final trained models (downloaded from Colab)
│       └── classifier/           # DeBERTa classifier trained in Colab
├── src/
│   ├── layer1/                   # Input Classifier (Trained Model)
│   │   ├── train_colab.ipynb     # Google Colab training notebook (T4 GPU)
│   │   ├── generate_synthetic_colab.ipynb  # Colab synthetic data generation
│   │   ├── evaluate.py           # Evaluation script (precision/recall/F1)
│   │   ├── inference.py          # classify_input(text) function (CPU inference)
│   │   └── generate_synthetic.py # Local synthetic data generation script
│   ├── layer2/                   # Canary Token Leak Detector (Pure Code)
│   │   ├── canary_manager.py     # Inject/detect canary tokens
│   │   └── leak_detector.py      # Scan outputs for leaks
│   ├── layer3/                   # Tool-Call Intent Auditor (LLM Wrapper)
│   │   ├── auditor.py            # Groq API wrapper for intent checking
│   │   └── prompts.py            # System prompts for the auditor
│   ├── common/                   # Shared utilities
│   │   ├── config.py             # Configuration management
│   │   ├── logger.py             # Logging setup
│   │   └── database.py           # Supabase connection
│   └── main.py                   # FastAPI app with /defend and /simulate_attack endpoints
├── logs/                         # Request/verdict logs
├── tests/                        # Test scripts
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # Project documentation
```

## Layer Responsibilities

### Layer 1: Input Classifier (TRAINED MODEL)
- **Model**: DeBERTa-v3-base fine-tuned for 4-class classification
- **Classes**: benign, direct_injection, indirect_injection, jailbreak
- **Output**: {label, confidence (0-100)}
- **Training**: Done in Google Colab (free tier T4 GPU) via `train_colab.ipynb`
- **Inference**: Runs on CPU locally (no GPU required)
- **Data**: Uses deepset/prompt-injections, JailbreakBench, synthetic Groq-generated data

### Layer 2: Canary Token Leak Detector (PURE CODE)
- **Function**: Inject random canaries into system prompts, scan outputs for leaks
- **No ML**: Pure string matching and logging
- **Logs**: Full context of any detected leak

### Layer 3: Tool-Call Intent Auditor (LLM WRAPPER)
- **Model**: Groq-hosted Llama 3.1/3.3
- **Function**: Multi-step aware intent checking (looks at N=3 recent tool calls)
- **Output**: ALLOW/BLOCK/ESCALATE + one-line reason
- **Context**: User task, tool call history, proposed action

## Orchestration Flow

1. Receive message → Layer 1 classification
2. If confidence > threshold → BLOCK and log
3. If tool call proposed → Layer 3 audit (with history)
4. If BLOCK/ESCALATE → Block or escalate
5. Layer 2 continuously scans all outputs for canary leaks
6. Everything logged to Supabase with timestamp, layer, verdict, latency

## Key Design Decisions

### Colab Training
- Training done in Google Colab free tier (T4 GPU, ~16GB VRAM)
- No local GPU required for training
- Frequent checkpointing (every 500 steps) to handle Colab session disconnections
- Model saved to Google Drive, then downloaded for local use

### CPU Inference
- Trained model runs on CPU locally for the defense pipeline
- Handles both regular and LoRA/PEFT models
- No GPU required for inference

### Data Generation
- Synthetic data can be generated in Colab or locally
- Uses Groq API for generating diverse injection examples
- ~500 examples across 5 attack styles
