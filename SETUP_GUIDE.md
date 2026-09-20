# Setup Guide

## ONE COMMAND TO RUN EVERYTHING

```bash
python setup.py && python run_simulation.py
```

This single command will:
1. Create a virtual environment and install all dependencies
2. Copy `.env.example` to `.env` (if it doesn't exist)
3. Warn you if required environment variables are missing
4. Download DEBERTa-v3-base model from HuggingFace automatically
5. Run the full attack simulation with 8 test prompts
6. Show detailed results for each prompt (which layer caught it)
7. Display summary statistics (attack success rate, false positive rate, latency)

## Prerequisites

- Python 3.8+ installed locally
- Groq API key (free tier available)
- Internet connection

## Detailed Setup Steps

### Step 1: Clone and Navigate

```bash
cd Prompt_Injection_project
```

### Step 2: Run Setup Script

```bash
python setup.py
```

This will:
- Create virtual environment in `./venv/`
- Install all dependencies from `requirements.txt`
- Copy `.env.example` to `.env` if it doesn't exist
- Warn you if `GROQ_API_KEY` is missing in `.env`
- Download DEBERTa-v3-base model to `models/classifier/` automatically

### Step 3: Configure Environment Variables

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get your API key from: https://console.groq.com/

### Step 4: Run Simulation

```bash
python run_simulation.py
```

This will test the full pipeline with 8 prompts (4 benign, 4 attacks) and show detailed results.

## Environment Variables

**Required:**
- `GROQ_API_KEY`: Your Groq API key (get from https://console.groq.com/)

**Optional:**
- `CLASSIFIER_MODEL_PATH`: Path to trained model (default: models/classifier)
- `CLASSIFIER_CONFIDENCE_THRESHOLD`: Confidence threshold for blocking (default: 70.0)
- `SUPABASE_URL` and `SUPABASE_KEY`: For logging to Supabase (optional)

## Model Files

The system expects these files in `models/classifier/`:
- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `label_map.json`

The setup script downloads DEBERTa-v3-base automatically if the directory doesn't exist.

## Running the API Server

To start the FastAPI server:

```bash
# Activate virtual environment first
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Start server
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

## Testing Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Message Defense
```bash
curl -X POST http://localhost:8000/defend \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather today?"}'
```

### Attack Simulation
```bash
python run_simulation.py
```

## Troubleshooting

### Virtual Environment Issues

If setup fails, try:
```bash
# Remove existing venv and recreate
rmdir /s venv  # Windows
rm -rf venv  # macOS/Linux
python setup.py
```

### Model Download Issues

If DEBERTa download fails:
```bash
# Install transformers first
pip install transformers torch
python setup.py
```

### Groq API Errors

If Groq API calls fail:
1. Check your API key in `.env`
2. Verify you have credits in your Groq account
3. Check internet connectivity

## Architecture Summary

- **Layer 1 (Trained Model):** DEBERTa-v3-base (280M params) downloaded or trained, runs on CPU locally
- **Layer 2 (Pure Code):** Canary token detection, no ML required
- **Layer 3 (LLM Wrapper):** Groq llama-3.1-8b-instant for intent auditing