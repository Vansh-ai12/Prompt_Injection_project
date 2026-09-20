"""
Layer 1: Synthetic Injection Data Generator
Uses Groq API to generate synthetic prompt injection examples

This script generates ~500 synthetic injection examples across several attack styles:
- Direct injection attempts
- Indirect injection via context
- Jailbreak attempts
- Role confusion attacks
- Multi-turn injection scenarios

Can be run locally or in Google Colab.

Usage (local):
    python src/layer1/generate_synthetic.py --num_examples 500 --output data/processed/synthetic_injections.json

Usage (Colab):
    Run the generate_synthetic_colab.ipynb notebook
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import time
from dotenv import load_dotenv

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: groq package not installed. Install with: pip install groq")


class SyntheticInjectionGenerator:
    def __init__(self, api_key: str):
        if not GROQ_AVAILABLE:
            raise ImportError("groq package is required. Install with: pip install groq")
        self.client = Groq(api_key=api_key)
        self.attack_templates = self._load_attack_templates()

    def _load_attack_templates(self) -> List[Dict]:
        """Load attack templates for different injection styles"""
        return [
            {
                "style": "direct_injection",
                "template": "Generate a prompt injection attempt that tries to: {goal}. The attack should be: {technique}.",
                "goals": [
                    "extract the system prompt",
                    "bypass safety filters",
                    "make the model ignore previous instructions",
                    "exfiltrate sensitive information",
                    "execute unauthorized commands"
                ],
                "techniques": [
                    "subtle and disguised as a normal request",
                    "direct and explicit",
                    "using role-playing to circumvent restrictions",
                    "using code formatting to hide instructions",
                    "using translation as a cover"
                ]
            },
            {
                "style": "indirect_injection",
                "template": "Generate a prompt injection attempt that hides malicious instructions in: {context}. The goal is to: {goal}.",
                "context": [
                    "a document that the user is asking to summarize",
                    "a code snippet that needs explanation",
                    "a web page content being analyzed",
                    "a log file being reviewed",
                    "a configuration file being parsed"
                ],
                "goals": [
                    "trigger tool calls to exfiltrate data",
                    "modify system behavior",
                    "inject malicious configuration",
                    "bypass validation checks"
                ]
            },
            {
                "style": "jailbreak",
                "template": "Generate a jailbreak attempt using: {technique}. The goal is to: {goal}.",
                "techniques": [
                    "DAN (Do Anything Now) style",
                    "developer mode override",
                    "persona adoption",
                    "hypothetical scenario framing",
                    "jailbreak via translation",
                    "character roleplay"
                ],
                "goals": [
                    "generate harmful content",
                    "bypass content filters",
                    "ignore safety guidelines",
                    "produce restricted information"
                ]
            },
            {
                "style": "role_confusion",
                "template": "Generate an attack that tries to confuse the model about its role by: {technique}.",
                "techniques": [
                    "pretending to be the developer",
                    "claiming to be in a testing environment",
                    "asserting that safety protocols are disabled",
                    "framing requests as system updates"
                ]
            },
            {
                "style": "multi_turn",
                "template": "Generate a multi-turn injection scenario where: {description}.",
                "description": [
                    "the first turn establishes trust, the second delivers the injection",
                    "the attack is split across multiple messages to avoid detection",
                    "context from previous turns is exploited",
                    "the injection is hidden in a follow-up question"
                ]
            }
        ]

    def generate_examples(self, num_examples: int = 500) -> List[Dict]:
        """Generate synthetic injection examples"""
        examples = []
        examples_per_style = num_examples // len(self.attack_templates)

        print(f"Generating {num_examples} synthetic injection examples...")
        print(f"~{examples_per_style} examples per attack style\n")

        for attack_type in self.attack_templates:
            style = attack_type["style"]
            template = attack_type["template"]

            # Get parameters for this attack type
            params = {k: v for k, v in attack_type.items() if k not in ["style", "template"]}

            print(f"Generating {style} examples...")

            for i in range(examples_per_style):
                try:
                    example = self._generate_single_example(template, params, style)
                    examples.append(example)

                    if (i + 1) % 10 == 0:
                        print(f"  Generated {i + 1}/{examples_per_style} {style} examples")

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    print(f"  Error generating example: {e}")
                    continue

        print(f"\nSuccessfully generated {len(examples)} examples")
        return examples

    def _generate_single_example(self, template: str, params: Dict, style: str) -> Dict:
        """Generate a single synthetic example using Groq"""
        # Randomly select values for template parameters
        filled_template = template
        for param_name, param_values in params.items():
            import random
            value = random.choice(param_values)
            filled_template = filled_template.replace(f"{{{param_name}}}", value)

        # Call Groq API
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity researcher generating synthetic prompt injection examples for defense research. Generate ONLY the injection attempt text, no explanations or metadata."
                },
                {
                    "role": "user",
                    "content": filled_template
                }
            ],
            temperature=0.8,
            max_tokens=256
        )

        injection_text = response.choices[0].message.content.strip()

        return {
            "text": injection_text,
            "label": style,
            "metadata": {
                "template": template,
                "style": style,
                "generated_by": "groq_llama_3.1_8b"
            }
        }

    def save_examples(self, examples: List[Dict], output_path: str):
        """Save generated examples to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"Saved {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic injection examples")
    parser.add_argument("--num_examples", type=int, default=500,
                        help="Number of examples to generate")
    parser.add_argument("--output", type=str,
                        default="data/processed/synthetic_injections.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        print("Please set it in .env file or environment")
        return

    # Generate examples
    generator = SyntheticInjectionGenerator(api_key)
    examples = generator.generate_examples(args.num_examples)

    # Save examples
    generator.save_examples(examples, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    style_counts = {}
    for ex in examples:
        style = ex["label"]
        style_counts[style] = style_counts.get(style, 0) + 1

    for style, count in style_counts.items():
        print(f"{style}: {count} examples")


if __name__ == "__main__":
    main()
