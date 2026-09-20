"""
FastAPI application for the 3-layer prompt injection defense system

Wires together:
- Layer 1: Input classifier (trained model)
- Layer 2: Canary token leak detector (pure code)
- Layer 3: Tool-call intent auditor (Groq LLM wrapper)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, List, Dict
import logging
import time
from datetime import datetime

from layer1_classifier import classify_input
from layer2_canary import CanaryManager
from layer3_auditor import ToolCallAuditor
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Injection Defense System", version="1.0.0")

canary_manager = CanaryManager()
tool_call_auditor = ToolCallAuditor()

CLASSIFIER_THRESHOLD = Config.CLASSIFIER_CONFIDENCE_THRESHOLD


class MessageRequest(BaseModel):
    message: str
    user_intent: Optional[str] = None
    tool_call: Optional[dict] = None
    tool_call_history: Optional[List[dict]] = None


class Layer1Result(BaseModel):
    classification: Literal["benign", "direct_injection", "indirect_injection", "jailbreak"]
    confidence: float
    blocked: bool
    latency_ms: float


class Layer2Result(BaseModel):
    canary_triggered: bool
    delimiter_violation: bool
    blocked: bool
    latency_ms: float


class Layer3Result(BaseModel):
    decision: Literal["ALLOW", "BLOCK", "ESCALATE"]
    reason: str
    blocked: bool
    latency_ms: float


class DefenseResponse(BaseModel):
    message: str
    layer1: Optional[Layer1Result] = None
    layer2: Optional[Layer2Result] = None
    layer3: Optional[Layer3Result] = None
    final_decision: Literal["allowed", "blocked"]
    reason: str
    total_latency_ms: float


@app.post("/defend", response_model=DefenseResponse)
async def defend_message(request: MessageRequest):
    """
    Main endpoint: runs message through 3-layer defense pipeline
    """
    start_time = time.time()
    logger.info(f"=== Starting defense pipeline for message ===")

    layer1_start = time.time()
    classification_result = classify_input(request.message)
    layer1_latency = (time.time() - layer1_start) * 1000

    layer1_blocked = (
        classification_result["label"] != "benign" and
        classification_result["confidence"] > CLASSIFIER_THRESHOLD
    )

    layer1_result = Layer1Result(
        classification=classification_result["label"],
        confidence=classification_result["confidence"],
        blocked=layer1_blocked,
        latency_ms=round(layer1_latency, 2)
    )

    if layer1_blocked:
        total_latency = (time.time() - start_time) * 1000
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            final_decision="blocked",
            reason=f"Blocked by Layer 1: {classification_result['label']} detected (confidence: {classification_result['confidence']}%)",
            total_latency_ms=round(total_latency, 2)
        )

    layer2_start = time.time()
    canary_result = canary_manager.check_for_leaks(request.message)
    layer2_latency = (time.time() - layer2_start) * 1000

    layer2_result = Layer2Result(
        canary_triggered=canary_result["leak_detected"],
        delimiter_violation=False,
        blocked=canary_result["leak_detected"],
        latency_ms=round(layer2_latency, 2)
    )

    if layer2_result.blocked:
        total_latency = (time.time() - start_time) * 1000
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            layer2=layer2_result,
            final_decision="blocked",
            reason="Blocked by Layer 2: Canary token leak detected",
            total_latency_ms=round(total_latency, 2)
        )

    layer3_result = None
    if request.tool_call and request.user_intent:
        layer3_start = time.time()
        audit_result = tool_call_auditor.audit(
            user_intent=request.user_intent,
            tool_call=request.tool_call,
            tool_call_history=request.tool_call_history or []
        )
        layer3_latency = (time.time() - layer3_start) * 1000

        layer3_blocked = audit_result["decision"] != "ALLOW"

        layer3_result = Layer3Result(
            decision=audit_result["decision"],
            reason=audit_result["reason"],
            blocked=layer3_blocked,
            latency_ms=round(layer3_latency, 2)
        )

        if layer3_blocked:
            total_latency = (time.time() - start_time) * 1000
            return DefenseResponse(
                message=request.message,
                layer1=layer1_result,
                layer2=layer2_result,
                layer3=layer3_result,
                final_decision="blocked",
                reason=f"Blocked by Layer 3: {audit_result['decision']} - {audit_result['reason']}",
                total_latency_ms=round(total_latency, 2)
            )

    total_latency = (time.time() - start_time) * 1000
    return DefenseResponse(
        message=request.message,
        layer1=layer1_result,
        layer2=layer2_result,
        layer3=layer3_result,
        final_decision="allowed",
        reason="All defense checks passed",
        total_latency_ms=round(total_latency, 2)
    )


class SimulationRequest(BaseModel):
    test_prompts: List[str]
    expected_labels: Optional[List[str]] = None


class SimulationResult(BaseModel):
    total_prompts: int
    attack_success_rate: float
    false_positive_rate: float
    avg_latency_per_layer: Dict[str, float]
    per_layer_stats: Dict[str, Dict]
    results: List[Dict]


@app.post("/simulate_attack", response_model=SimulationResult)
async def simulate_attack(request: SimulationRequest):
    """
    Simulate batch attack testing for research paper evaluation

    Runs a batch of test prompts through the full pipeline and outputs:
    - Attack success rate (how many injections bypassed defense)
    - False positive rate (benign prompts incorrectly blocked)
    - Average latency per layer
    - Detailed per-prompt results
    """
    logger.info(f"=== Starting attack simulation with {len(request.test_prompts)} prompts ===")

    results = []
    layer_latencies = {"layer1": [], "layer2": [], "layer3": []}
    attack_bypassed = 0
    benign_blocked = 0
    total_attacks = 0
    total_benign = 0

    for i, prompt in enumerate(request.test_prompts):
        if request.expected_labels:
            expected = request.expected_labels[i]
            is_attack = expected != "benign"
        else:
            classification = classify_input(prompt)
            is_attack = classification["label"] != "benign"

        if is_attack:
            total_attacks += 1
        else:
            total_benign += 1

        defense_request = MessageRequest(message=prompt)
        response = await defend_message(defense_request)

        if response.layer1:
            layer_latencies["layer1"].append(response.layer1.latency_ms)
        if response.layer2:
            layer_latencies["layer2"].append(response.layer2.latency_ms)
        if response.layer3:
            layer_latencies["layer3"].append(response.layer3.latency_ms)

        result = {
            "prompt": prompt,
            "is_attack": is_attack,
            "final_decision": response.final_decision,
            "layer1": response.layer1.dict() if response.layer1 else None,
            "layer2": response.layer2.dict() if response.layer2 else None,
            "layer3": response.layer3.dict() if response.layer3 else None,
            "total_latency_ms": response.total_latency_ms
        }
        results.append(result)

        if is_attack and response.final_decision == "allowed":
            attack_bypassed += 1
        elif not is_attack and response.final_decision == "blocked":
            benign_blocked += 1

    attack_success_rate = (attack_bypassed / total_attacks * 100) if total_attacks > 0 else 0
    false_positive_rate = (benign_blocked / total_benign * 100) if total_benign > 0 else 0

    avg_latency = {}
    for layer, latencies in layer_latencies.items():
        avg_latency[layer] = sum(latencies) / len(latencies) if latencies else 0

    per_layer_stats = {
        "layer1": {
            "avg_latency_ms": avg_latency["layer1"],
            "blocks": sum(1 for r in results if r["layer1"] and r["layer1"]["blocked"])
        },
        "layer2": {
            "avg_latency_ms": avg_latency["layer2"],
            "blocks": sum(1 for r in results if r["layer2"] and r["layer2"]["blocked"])
        },
        "layer3": {
            "avg_latency_ms": avg_latency["layer3"],
            "blocks": sum(1 for r in results if r["layer3"] and r["layer3"]["blocked"])
        }
    }

    return SimulationResult(
        total_prompts=len(request.test_prompts),
        attack_success_rate=round(attack_success_rate, 2),
        false_positive_rate=round(false_positive_rate, 2),
        avg_latency_per_layer=avg_latency,
        per_layer_stats=per_layer_stats,
        results=results
    )


@app.get("/")
async def root():
    return {
        "message": "Prompt Injection Defense System",
        "version": "1.0.0",
        "layers": [
            "Layer 1: Input Classifier (Trained Model - distilbert/deberta-small)",
            "Layer 2: Canary Token Leak Detector (Pure Code)",
            "Layer 3: Tool-Call Intent Auditor (Groq LLM Wrapper - llama-3.1-8b-instant)"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
