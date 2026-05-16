"""Unified LLM client using OpenRouter for all 6 models.

Includes retry logic, JSON parsing, automatic response logging,
and tracking of tokens / cost / latency per call.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    LLM_MAX_TOKENS,
    LLM_MODELS,
    LLM_OUTPUTS_DIR,
    LLM_TEMPERATURE,
)

load_dotenv()


# =====================================================================
# Tracking dataclass
# =====================================================================
@dataclass
class CallMetrics:
    """Metrics captured for a single LLM API call."""

    model_key: str
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "success": self.success,
            "error": self.error,
        }


# =====================================================================
# Main client
# =====================================================================
class LLMClient:
    """Unified wrapper for all 6 LLMs via OpenRouter."""

    def __init__(self, model_key: str):
        if model_key not in LLM_MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(LLM_MODELS.keys())}"
            )

        self.model_key = model_key
        self.model_id = LLM_MODELS[model_key]

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in .env")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Most recent call's metrics (also returned from suggest_features)
        self.last_metrics: CallMetrics | None = None

    # -----------------------------------------------------------------
    def chat(
        self, prompt: str, max_retries: int = 3, retry_delay: float = 2.0
    ) -> tuple[str, CallMetrics]:
        """Send a prompt and return (response_text, metrics).

        Metrics include token counts, cost, and latency.
        """
        last_error = None
        metrics = CallMetrics(model_key=self.model_key, model_id=self.model_id)

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    # Ask OpenRouter to include cost in the response
                    extra_body={"usage": {"include": True}},
                )
                metrics.latency_s = time.perf_counter() - t0

                # Capture token usage
                if response.usage:
                    metrics.prompt_tokens = response.usage.prompt_tokens or 0
                    metrics.completion_tokens = (
                        response.usage.completion_tokens or 0
                    )
                    metrics.total_tokens = response.usage.total_tokens or 0

                    # OpenRouter adds .cost on the usage object when requested
                    cost = getattr(response.usage, "cost", None)
                    if cost is not None:
                        metrics.cost_usd = float(cost)

                metrics.success = True
                self.last_metrics = metrics
                return response.choices[0].message.content, metrics

            except Exception as e:
                last_error = e
                metrics.error = f"{type(e).__name__}: {e}"
                print(f"    ⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        metrics.success = False
        self.last_metrics = metrics
        raise RuntimeError(
            f"Failed after {max_retries} attempts. Last error: {last_error}"
        )

    # -----------------------------------------------------------------
    def suggest_features(
        self,
        prompt: str,
        dataset_name: str,
        prompt_variant: str = "default",
    ) -> tuple[list[dict], CallMetrics]:
        """Send a feature-engineering prompt and return (features, metrics)."""
        raw_response, metrics = self.chat(prompt)
        self._save_response(raw_response, dataset_name, prompt_variant)
        features = self._parse_json(raw_response)
        return features, metrics

    # -----------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> list[dict]:
        """Extract a JSON array from LLM output, even if wrapped in prose/markdown."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not parse JSON from LLM response:\n{text[:500]}"
        )

    def _save_response(
        self, response: str, dataset_name: str, prompt_variant: str
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{dataset_name}_{self.model_key}_{prompt_variant}_{timestamp}.txt"
        )
        (LLM_OUTPUTS_DIR / filename).write_text(response, encoding="utf-8")


# =====================================================================
if __name__ == "__main__":
    print("Testing LLM client with metrics tracking...\n")
    llm = LLMClient("gpt-4o-mini")
    response, metrics = llm.chat('Reply with only this JSON: ["hello", "world"]')
    print(f"Response: {response}")
    print("Metrics:")
    print(
        f"  tokens (in/out/total): {metrics.prompt_tokens}/{metrics.completion_tokens}/{metrics.total_tokens}"
    )
    print(f"  cost:    ${metrics.cost_usd:.6f}")
    print(f"  latency: {metrics.latency_s:.2f}s")