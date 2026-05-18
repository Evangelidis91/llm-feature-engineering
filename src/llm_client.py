"""Unified LLM client using OpenRouter for all 10 models.

Why this file exists:
- Every notebook needs to call LLMs. Without this wrapper, each one would
  repeat boilerplate (load API key, set base URL, parse JSON, retry on errors).
- One consistent interface means swapping models is a one-line change.
- Metrics (tokens, cost, latency) are captured automatically per call so we
  can analyze cost/value tradeoffs across the study.

The whole class is ~150 lines. Most of it is error handling and parsing.
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

# Load OPENROUTER_API_KEY from .env at import time
load_dotenv()


# =====================================================================
# Tracking dataclass
# =====================================================================
@dataclass
class CallMetrics:
    """Captures everything we want to know about a single LLM API call.

    Used for the cost / latency analysis that ended up in the README.
    Each field defaults to a safe value so a partial-failure call still
    produces a complete record.
    """

    model_key: str  # human-readable name, e.g. "gpt-4o-mini"
    model_id: str  # OpenRouter ID, e.g. "openai/gpt-4o-mini"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict so it can be appended to a CSV row."""
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
    """Unified wrapper for all 10 LLMs via OpenRouter.

    Usage:
        llm = LLMClient("gpt-4o-mini")
        text, metrics = llm.chat("Hello!")
        features, metrics = llm.suggest_features(prompt, "churn", "zero_shot")
    """

    def __init__(self, model_key: str):
        # Reject unknown model keys early — saves a confusing 404 later
        if model_key not in LLM_MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(LLM_MODELS.keys())}"
            )

        self.model_key = model_key
        self.model_id = LLM_MODELS[model_key]

        # Read API key once at construction (not per call)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in .env")

        # We use the OpenAI Python SDK but point it at OpenRouter — same protocol,
        # different endpoint. This is why we can call any of the 10 LLMs the same way.
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Holds metrics from the most recent call. Useful for capturing cost
        # info even when the higher-level method (e.g. suggest_features) fails
        # AFTER the API call succeeded — see notebook 03 cell 4 for usage.
        self.last_metrics: CallMetrics | None = None

    # -----------------------------------------------------------------
    def chat(
        self, prompt: str, max_retries: int = 3, retry_delay: float = 2.0
    ) -> tuple[str, CallMetrics]:
        """Send a prompt to the LLM and return (response_text, metrics).

        Retries up to `max_retries` times on transient errors (network blips,
        rate limits, OpenRouter 5xx). Backoff doubles each attempt:
        2s → 4s → 6s.

        Saves Qwen 2.5 from a 'NoneType' error during one of our runs — the
        retry logic is what made the study reproducible.
        """
        last_error = None
        metrics = CallMetrics(model_key=self.model_key, model_id=self.model_id)

        for attempt in range(max_retries):
            try:
                # Time the call so we can compare LLM latencies fairly
                t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LLM_TEMPERATURE,  # low for consistent suggestions
                    max_tokens=LLM_MAX_TOKENS,
                    # Tell OpenRouter to include cost data in the usage object.
                    # Without this, response.usage.cost would be missing.
                    extra_body={"usage": {"include": True}},
                )
                metrics.latency_s = time.perf_counter() - t0

                # Capture token counts and cost from OpenRouter's response
                if response.usage:
                    metrics.prompt_tokens = response.usage.prompt_tokens or 0
                    metrics.completion_tokens = (
                        response.usage.completion_tokens or 0
                    )
                    metrics.total_tokens = response.usage.total_tokens or 0

                    # `cost` is an OpenRouter extension on top of the standard usage object
                    cost = getattr(response.usage, "cost", None)
                    if cost is not None:
                        metrics.cost_usd = float(cost)

                metrics.success = True
                self.last_metrics = metrics
                return response.choices[0].message.content, metrics

            except Exception as e:
                # Don't crash the whole batch — log this attempt and try again
                last_error = e
                metrics.error = f"{type(e).__name__}: {e}"
                print(f"    Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        # All retries exhausted — surface the last error to the caller
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
        """Send a feature-engineering prompt and return (parsed_features, metrics).

        Steps:
          1. Call the LLM via chat()
          2. Save the raw response to disk (for reproducibility)
          3. Parse the response as JSON, returning a list of feature dicts

        Each feature dict has keys: name, formula, rationale.
        """
        raw_response, metrics = self.chat(prompt)
        self._save_response(raw_response, dataset_name, prompt_variant)
        features = self._parse_json(raw_response)
        return features, metrics

    # -----------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> list[dict]:
        """Extract a JSON array from messy LLM output.

        Tries 3 strategies, in order of preference:
          1. Strip markdown fences (```json ...```) and parse the cleaned text
          2. Find the first [...] block in the text and parse that
          3. Find the first {...} block (rare — for misbehaved LLMs)

        Raises ValueError if all 3 fail. The ~5% of LLM responses that wrap
        JSON in prose are why this method is so defensive.
        """
        # Strategy 1: try parsing after stripping markdown fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: regex-find the first JSON array anywhere in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3: regex-find the first JSON object (fallback)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # All strategies failed — surface the first 500 chars so the caller can debug
        raise ValueError(
            f"Could not parse JSON from LLM response:\n{text[:500]}"
        )

    def _save_response(
        self, response: str, dataset_name: str, prompt_variant: str
    ) -> None:
        """Write the raw LLM response to disk with a timestamped filename.

        Saved files live in `results/llm_outputs/`. We keep these even though
        the parsed JSON is in `llm_suggestions.json`, because:
          - Reproducibility: anyone can re-parse with a different parser
          - Forensics: when a parse fails, we can see exactly what came back
          - Audit: timestamped files = clear chronology of when the study ran
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"{dataset_name}_{self.model_key}_{prompt_variant}_{timestamp}.txt")
        (LLM_OUTPUTS_DIR / filename).write_text(response, encoding="utf-8")


# =====================================================================
# Quick test — runs only when this file is executed directly
# =====================================================================
if __name__ == "__main__":
    print("Testing LLM client with metrics tracking...\n")
    llm = LLMClient("gpt-4o-mini")
    response, metrics = llm.chat(
        'Reply with only this JSON: ["hello", "world"]'
    )
    print(f"Response: {response}")
    print("Metrics:")
    print(f"  tokens (in/out/total): {metrics.prompt_tokens}/{metrics.completion_tokens}/{metrics.total_tokens}")
    print(f"  cost:    ${metrics.cost_usd:.6f}")
    print(f"  latency: {metrics.latency_s:.2f}s")