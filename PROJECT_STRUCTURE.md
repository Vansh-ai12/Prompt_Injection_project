# Project Structure

```
Prompt_Injection_project/
├── data/
│   ├── raw/                      # Downloaded datasets (deepset, JailbreakBench, synthetic)
│   └── processed/                # Preprocessed training/validation/test splits
├── models/
│   ├── checkpoints/              # Training checkpoints during fine-tuning
│   └── saved/                    # Final trained models (classifier, etc.)
├── src/
│   ├── layer1/                   # Input Classifier (Trained Model)
│   │   ├── train.py              # Fine-tuning script for DeBERTa
│   │   ├── evaluate.py           # Evaluation script (precision/recall/F1)
│   │   ├── inference.py          # classify_input(text) function
│   │   └── generate_synthetic.py # Script to generate synthetic injections via Groq
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
- **Training**: Uses deepset/prompt-injections, JailbreakBench, synthetic Groq-generated data

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
