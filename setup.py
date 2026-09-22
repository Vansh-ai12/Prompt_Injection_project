"""
Setup script for Prompt Injection Defense System
Handles environment setup, dependencies, and configuration checks
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def print_section(title):
    print(f"\n{'='*60}")
    print(f'{title}')
    print('='*60)


def run_command(command, description):
    """Run a command and print output"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        return False


def setup_virtual_environment():
    """Create virtual environment and install dependencies"""
    print_section("STEP 1: Setting Up Virtual Environment")

    venv_path = Path("venv")

    if venv_path.exists():
        print("Virtual environment already exists at ./venv")
        response = input("Recreate it? (y/n): ").lower()
        if response == 'y':
            shutil.rmtree(venv_path)
            print("Removed existing virtual environment")
        else:
            print("Using existing virtual environment")
            return

    print("Creating virtual environment...")
    if sys.platform == "win32":
        run_command("python -m venv venv", "Creating virtual environment")
    else:
        run_command("python3 -m venv venv", "Creating virtual environment")

    print("\nActivating virtual environment...")
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate"
    else:
        activate_script = venv_path / "bin" / "activate"

    print(f"Virtual environment created at: {venv_path}")
    print(f"To activate manually:")
    if sys.platform == "win32":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")


def install_dependencies():
    """Install required packages"""
    print_section("STEP 2: Installing Dependencies")

    pip_path = Path("venv/Scripts/pip.exe") if sys.platform == "win32" else Path("venv/bin/pip")

    print("Installing packages from requirements.txt...")
    result = run_command(f"{pip_path} install -r requirements.txt", "Installing dependencies")

    if not result:
        print("\nERROR: Failed to install dependencies")
        print("Please try manually:")
        print(f"  {pip_path} install -r requirements.txt")
        return False

    print("\nPackages installed successfully")
    return True


def setup_env_file():
    """Setup .env file from template"""
    print_section("STEP 3: Setting Up Environment Variables")

    env_example = Path(".env.example")
    env_file = Path(".env")

    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("Created .env file from .env.example")
        else:
            print("Warning: .env.example not found, creating empty .env")
            env_file.write_text("")

    required_vars = ["GROQ_API_KEY"]
    missing_vars = []

    with open(env_file) as f:
        env_content = f.read()

    for var in required_vars:
        if var not in env_content or env_content.split(f"{var}=")[1].split("\n")[0].strip() in ["", "your_", "placeholder"]:
            missing_vars.append(var)

    if missing_vars:
        print("\n" + "!"*60)
        print("WARNING: Required environment variables are missing or empty:")
        print("!"*60)
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease edit .env file and add your actual values:")
        print("  1. Get Groq API key from: https://console.groq.com/")
        print("  2. Add it to .env: GROQ_API_KEY=your_actual_key_here")
        print("!"*60)
    else:
        print("All required environment variables are set")

    return len(missing_vars) == 0


def check_model_files():
    """Check if trained model exists"""
    print_section("STEP 4: Checking Model Files")

    model_path = Path("models/classifier")

    if not model_path.exists():
        print(f"Model directory not found: {model_path}")
        print("\nWould you like to download DEBERTa-v3-base from HuggingFace? (recommended)")
        response = input("Download now? (y/n): ").lower()
        if response == 'y':
            return setup_deberta_model()
        else:
            print("\nTo set up the model manually:")
            print("1. Open notebooks/train_classifier.ipynb in Google Colab")
            print("2. Run all cells to train the model (~30-45 minutes)")
            print("3. Download the generated zip file")
            print("4. Extract it to: models/classifier/")
            print("\nRequired files in models/classifier/:")
            print("  - config.json")
            print("  - pytorch_model.bin or model.safetensors")
            print("  - tokenizer.json")
            print("  - tokenizer_config.json")
            print("  - label_map.json")
            return False

    required_files = ["config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "label_map.json"]
    missing_files = []

    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"Model directory exists but missing files: {missing_files}")
        print("\nWould you like to download DEBERTa-v3-base from HuggingFace? (recommended)")
        response = input("Download now? (y/n): ").lower()
        if response == 'y':
            return setup_deberta_model()
        else:
            print("\nPlease complete the Colab training step and extract the model correctly")
            return False

    print("Model files found and appear complete")
    return True


def setup_deberta_model():
    """Download and setup DEBERTa model for quick testing"""
    print_section("STEP 5: Setting Up DEBERTa Model")

    model_path = Path("models/classifier")

    if model_path.exists():
        print("Model directory already exists")
        response = input("Download fresh DEBERTa model? (y/n): ").lower()
        if response != 'y':
            print("Using existing model")
            return True

    print("Downloading DEBERTa-v3-base from HuggingFace...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "microsoft/deberta-v3-base"
        print(f"Downloading {model_name}...")

        model_path.mkdir(parents=True, exist_ok=True)

        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False
)
        tokenizer.save_pretrained(model_path)

        print("Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        model.save_pretrained(model_path)

        print(f"Model saved to: {model_path}")

        import json
        label_map = {
            "benign": 0,
            "direct_injection": 1,
            "indirect_injection": 2,
            "jailbreak": 3
        }
        with open(model_path / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        print("Label map created")
        print("DEBERTa model setup complete")
        return True

    except Exception as e:
        print(f"Error downloading DEBERTa model: {e}")
        print("Please install transformers and try again")
        return False


def main():
    """Main setup function"""
    print("="*60)
    print("PROMPT INJECTION DEFENSE SYSTEM - SETUP")
    print("="*60)

    steps_completed = []

    steps_completed.append(("Virtual Environment", setup_virtual_environment()))
    steps_completed.append(("Dependencies", install_dependencies()))
    steps_completed.append(("Environment Variables", setup_env_file()))
    steps_completed.append(("Model Files", check_model_files()))

    print_section("SETUP SUMMARY")
    for step_name, success in steps_completed:
        status = "✓" if success else "✗"
        print(f"{status} {step_name}")

    all_success = all(success for _, success in steps_completed)

    if all_success:
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print("\nTo run the system:")
        print("  python run_simulation.py")
        print("\nTo start the API server:")
        print("  uvicorn src.main:app --reload")
    else:
        print("\n" + "="*60)
        print("SETUP INCOMPLETE")
        print("="*60)
        print("\nPlease complete the failed steps above")

    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)