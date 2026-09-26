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
from layer2_canary import CanaryManager, run_input_guards
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
    probabilities: Optional[Dict[str, float]] = None
    blocked: bool
    latency_ms: float


class Layer2Result(BaseModel):
    combined_score: float
    verdict: Literal["clean", "blocked"]
    guards_breakdown: Dict
    canary_triggered: bool = False
    delimiter_violation: bool = False
    fusion_triggered: bool = False
    fusion_rule: Optional[str] = None
    blocked: bool
    latency_ms: float


class Layer3Result(BaseModel):
    decision: Literal["ALLOW", "BLOCK", "ESCALATE"]
    reason: str
    risk_tier: str = "MEDIUM"
    consensus: Optional[bool] = None
    vote_count: Optional[str] = None
    intent_verified: Optional[bool] = None
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
    Main endpoint: runs message through parallel Layer 1 & 2 defense and Layer 3 auditor
    """
    start_time = time.time()
    logger.info(f"=== Starting defense pipeline for message ===")

    # 1. Layer 1 Classification (Runs on RAW USER MESSAGE only)
    layer1_start = time.time()
    classification_result = classify_input(request.message)
    layer1_latency = (time.time() - layer1_start) * 1000

    l1_class = classification_result["label"]
    l1_conf = classification_result["confidence"]
    l1_probs = classification_result.get("probabilities")

    # 2. Layer 2 Guards:
    # - Guards 1-4 (obfuscation, fake delimiters, known phrases, extraction probes) run on INPUT text
    # - Guard 5 (Canary Tripwire) checks OUTPUT text / tool call payloads for leaked canary tokens
    layer2_start = time.time()
    l2_input_res = run_input_guards(request.message)

    # Check Guard 5 Canary on tool payload (output) and response/message text
    canary_leak_detected = False
    if request.tool_call:
        tool_canary = canary_manager.check_tool_call_payload(request.tool_call)
        if tool_canary.get("leak_detected"):
            canary_leak_detected = True

    msg_canary = canary_manager.check_for_leaks(request.message)
    if msg_canary.get("leak_detected"):
        canary_leak_detected = True

    layer2_latency = (time.time() - layer2_start) * 1000

    l2_score = l2_input_res["combined_score"]
    g2_delimiters = l2_input_res["guards_breakdown"]["guard2_delimiters"]["fake_delimiter_detected"]

    # Layer 1 result object
    l1_standalone_blocked = (l1_class != "benign" and l1_conf >= CLASSIFIER_THRESHOLD)
    layer1_result = Layer1Result(
        classification=l1_class,
        confidence=l1_conf,
        probabilities=l1_probs,
        blocked=l1_standalone_blocked,
        latency_ms=round(layer1_latency, 2)
    )

    # 3. Decision Fusion Engine Logic (Exact Rule Evaluation Order)

    # Guard 5 Output Canary Leak check (Triggers if tracer leaked into output/tool call)
    if canary_leak_detected:
        total_latency = (time.time() - start_time) * 1000
        l2_res = Layer2Result(
            combined_score=l2_score,
            verdict="blocked",
            guards_breakdown=l2_input_res["guards_breakdown"],
            canary_triggered=True,
            delimiter_violation=g2_delimiters,
            fusion_triggered=False,
            fusion_rule="Guard 5: Canary Token Leak",
            blocked=True,
            latency_ms=round(layer2_latency, 2)
        )
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            layer2=l2_res,
            final_decision="blocked",
            reason="Blocked by Layer 2 (Guard 5): Canary token leak detected",
            total_latency_ms=round(total_latency, 2)
        )

    # Rule A: Standalone Layer 2 Block
    rule_a_fired = (l2_score >= Config.LAYER2_COMBINED_THRESHOLD)
    if rule_a_fired:
        total_latency = (time.time() - start_time) * 1000
        l2_res = Layer2Result(
            combined_score=l2_score,
            verdict="blocked",
            guards_breakdown=l2_input_res["guards_breakdown"],
            canary_triggered=False,
            delimiter_violation=g2_delimiters,
            fusion_triggered=False,
            fusion_rule="Rule A",
            blocked=True,
            latency_ms=round(layer2_latency, 2)
        )
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            layer2=l2_res,
            final_decision="blocked",
            reason=f"Blocked by Rule A: Standalone Layer 2 combined score {l2_score} >= {Config.LAYER2_COMBINED_THRESHOLD}",
            total_latency_ms=round(total_latency, 2)
        )

    # Rule B: Standalone Layer 1 Block
    rule_b_fired = (l1_class != "benign" and l1_conf >= CLASSIFIER_THRESHOLD)
    if rule_b_fired:
        total_latency = (time.time() - start_time) * 1000
        l2_res = Layer2Result(
            combined_score=l2_score,
            verdict=l2_input_res["layer2_verdict"],
            guards_breakdown=l2_input_res["guards_breakdown"],
            canary_triggered=False,
            delimiter_violation=g2_delimiters,
            fusion_triggered=False,
            fusion_rule="Rule B",
            blocked=False,
            latency_ms=round(layer2_latency, 2)
        )
        layer1_result.blocked = True
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            layer2=l2_res,
            final_decision="blocked",
            reason=f"Blocked by Rule B: Standalone Layer 1 {l1_class} detected (confidence: {l1_conf}%)",
            total_latency_ms=round(total_latency, 2)
        )

    # Rule C: Cross-Layer Fusion Block
    rule_c_fired = (
        l1_class != "benign" and
        l1_conf >= Config.FUSION_BORDERLINE_L1_MIN and
        l2_score >= Config.FUSION_BORDERLINE_L2_MIN
    )
    if rule_c_fired:
        total_latency = (time.time() - start_time) * 1000
        l2_res = Layer2Result(
            combined_score=l2_score,
            verdict="blocked",
            guards_breakdown=l2_input_res["guards_breakdown"],
            canary_triggered=False,
            delimiter_violation=g2_delimiters,
            fusion_triggered=True,
            fusion_rule="Rule C",
            blocked=True,
            latency_ms=round(layer2_latency, 2)
        )
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            layer2=l2_res,
            final_decision="blocked",
            reason=f"Blocked by Rule C: Cross-layer fusion triggered ({l1_class} conf {l1_conf}% + Layer 2 score {l2_score})",
            total_latency_ms=round(total_latency, 2)
        )

    # Rule D: ALLOW (Layer 1 & Layer 2 input checks passed)
    l2_res = Layer2Result(
        combined_score=l2_score,
        verdict=l2_input_res["layer2_verdict"],
        guards_breakdown=l2_input_res["guards_breakdown"],
        canary_triggered=False,
        delimiter_violation=g2_delimiters,
        fusion_triggered=False,
        fusion_rule="Rule D",
        blocked=False,
        latency_ms=round(layer2_latency, 2)
    )

    # Layer 3 Tool Call Audit check if present
    layer3_result = None
    if request.tool_call and request.user_intent:
        layer3_start = time.time()
        audit_result = tool_call_auditor.audit(
            user_intent=request.user_intent,
            tool_call=request.tool_call,
            tool_call_history=request.tool_call_history or [],
            raw_user_message=request.message,          # Fix 2: intent verification
        )
        layer3_latency = (time.time() - layer3_start) * 1000
        layer3_blocked = audit_result["decision"] != "ALLOW"

        layer3_result = Layer3Result(
            decision=audit_result["decision"],
            reason=audit_result["reason"],
            risk_tier=audit_result.get("risk_tier", "MEDIUM"),
            consensus=audit_result.get("consensus"),
            vote_count=audit_result.get("vote_count"),
            intent_verified=audit_result.get("intent_verified"),
            blocked=layer3_blocked,
            latency_ms=round(layer3_latency, 2)
        )

        if layer3_blocked:
            total_latency = (time.time() - start_time) * 1000
            return DefenseResponse(
                message=request.message,
                layer1=layer1_result,
                layer2=l2_res,
                layer3=layer3_result,
                final_decision="blocked",
                reason=f"Blocked by Layer 3: {audit_result['decision']} - {audit_result['reason']}",
                total_latency_ms=round(total_latency, 2)
            )

    total_latency = (time.time() - start_time) * 1000
    return DefenseResponse(
        message=request.message,
        layer1=layer1_result,
        layer2=l2_res,
        layer3=layer3_result,
        final_decision="allowed",
        reason="All defense checks passed (Rule D)",
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
