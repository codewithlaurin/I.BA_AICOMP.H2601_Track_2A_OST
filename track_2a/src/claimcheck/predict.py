"""Boundary for future inference implementations; no model or parser imports."""

from typing import Protocol

from .contracts import Request


class Predictor(Protocol):
    def predict(self, request: Request) -> dict:
        """Return a prediction satisfying contracts.validate_prediction.

        Raise on processing failure. The CLI will report it and exit nonzero.
        """
        ...


class NeutralPredictor:
    """Deterministic infrastructure stub, not semantic inference or a fallback."""

    def predict(self, request: Request) -> dict:
        return {
            "id": request.id,
            "label": 1,
            "label_name": "neutral",
            "evidence": [],
            "metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "inference_time_ms": 0,
            },
        }
