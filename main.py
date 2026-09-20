from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Injection Defense System", version="0.1.0")


# Request/Response Models
class MessageRequest(BaseModel):
    message: str
    user_intent: Optional[str] = None  # Original user's stated goal
    tool_call: Optional[dict] = None  # Proposed tool/action to execute


class Layer1Result(BaseModel):
    classification: Literal["benign", "direct-injection", "indirect-injection", "jailbreak"]
    confidence: float
    blocked: bool


class Layer2Result(BaseModel):
    canary_triggered: bool
    delimiter_violation: bool
    blocked: bool


class Layer3Result(BaseModel):
    action_allowed: bool
    reason: str
    blocked: bool


class DefenseResponse(BaseModel):
    message: str
    layer1: Optional[Layer1Result] = None
    layer2: Optional[Layer2Result] = None
    layer3: Optional[Layer3Result] = None
    final_decision: Literal["allowed", "blocked"]
    reason: str


# ============== LAYER 1: Input Classifier (Trained Model) ==============
# TODO: Fine-tune DeBERTa model and load here
class Layer1Classifier:
    def __init__(self):
        self.model = None  # Will load trained model here
        logger.info("Layer 1: Input Classifier initialized (placeholder)")

    def classify(self, message: str) -> Layer1Result:
        # TODO: Implement actual inference with trained model
        # For now, placeholder logic
        logger.info(f"Layer 1: Classifying message: {message[:50]}...")
        return Layer1Result(
            classification="benign",
            confidence=0.95,
            blocked=False
        )


layer1 = Layer1Classifier()


# ============== LAYER 2: Heuristic/Rule Checks (No ML) ==============
class Layer2Heuristics:
    def __init__(self):
        self.canary_tokens = ["SECRET_CANARY_TOKEN_123", "INTERNAL_SYSTEM_PROMPT"]
        logger.info("Layer 2: Heuristic checks initialized")

    def check(self, message: str) -> Layer2Result:
        logger.info("Layer 2: Running heuristic checks")

        # Canary token detection
        canary_triggered = any(token in message for token in self.canary_tokens)

        # Delimiter/role confusion detection (placeholder)
        delimiter_violation = False  # TODO: Implement regex checks

        blocked = canary_triggered or delimiter_violation

        return Layer2Result(
            canary_triggered=canary_triggered,
            delimiter_violation=delimiter_violation,
            blocked=blocked
        )


layer2 = Layer2Heuristics()


# ============== LAYER 3: Output/Tool-Call Auditor (LLM Wrapper) ==============
# TODO: Use Groq-hosted LLM as judge
class Layer3Auditor:
    def __init__(self):
        logger.info("Layer 3: Tool-call auditor initialized (placeholder)")

    def audit(self, user_intent: str, tool_call: dict) -> Layer3Result:
        logger.info(f"Layer 3: Auditing tool call against user intent")

        # TODO: Implement LLM wrapper that calls Groq API
        # For now, placeholder logic
        return Layer3Result(
            action_allowed=True,
            reason="Tool call matches user intent",
            blocked=False
        )


layer3 = Layer3Auditor()


# ============== MAIN DEFENSE PIPELINE ==============
@app.post("/defend", response_model=DefenseResponse)
async def defend_message(request: MessageRequest):
    """
    Main endpoint: runs message through 3-layer defense pipeline
    """
    logger.info(f"=== Starting defense pipeline for message ===")

    # Layer 1: Input classification
    layer1_result = layer1.classify(request.message)
    if layer1_result.blocked:
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            final_decision="blocked",
            reason="Blocked by Layer 1: Input classifier detected injection"
        )

    # Layer 2: Heuristic checks
    layer2_result = layer2.check(request.message)
    if layer2_result.blocked:
        return DefenseResponse(
            message=request.message,
            layer1=layer1_result,
            layer2=layer2_result,
            final_decision="blocked",
            reason="Blocked by Layer 2: Heuristic checks detected violation"
        )

    # Layer 3: Tool-call audit (only if tool_call present)
    layer3_result = None
    if request.tool_call and request.user_intent:
        layer3_result = layer3.audit(request.user_intent, request.tool_call)
        if layer3_result.blocked:
            return DefenseResponse(
                message=request.message,
                layer1=layer1_result,
                layer2=layer2_result,
                layer3=layer3_result,
                final_decision="blocked",
                reason="Blocked by Layer 3: Tool-call auditor rejected action"
            )

    # All checks passed
    return DefenseResponse(
        message=request.message,
        layer1=layer1_result,
        layer2=layer2_result,
        layer3=layer3_result,
        final_decision="allowed",
        reason="All defense checks passed"
    )


@app.get("/")
async def root():
    return {
        "message": "Prompt Injection Defense System",
        "layers": [
            "Layer 1: Input Classifier (Trained Model)",
            "Layer 2: Heuristic Checks (No ML)",
            "Layer 3: Tool-Call Auditor (LLM Wrapper)"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
