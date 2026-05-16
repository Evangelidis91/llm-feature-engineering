"""Unified LLM client using OpenRouter for all 6 models.

Includes retry logic, JSON parsing, and automatic response logging
for reproducibility.
"""
import json
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    LLM_MAX_TOKENS,
    LLM_MODELS,
    LLM_OUTPUTS_DIR,
    LLM_TEMPERATURE,
)

load_dotenv()


class LLMClient:
    """Unified wrapper for all 6 LLMs via OpenRouter."""

    def __init__(self, model_key: str):
        if model_key not in LLM_MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Available: {list(LLM_MODELS.keys())}"
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

    # -----------------------------------------------------------------
    def chat(self, prompt: str, max_retries: int = 3, retry_delay: float = 2.0) -> str:
        """Send a prompt; retry on transient failures."""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}")

    # -----------------------------------------------------------------
    def suggest_features(
            self,
            prompt: str,
            dataset_name: str,
            prompt_variant: str = "default",
    ) -> list[dict]:
        """Send a feature-engineering prompt and parse JSON response."""
        raw_response = self.chat(prompt)
        self._save_response(raw_response, dataset_name, prompt_variant)
        return self._parse_json(raw_response)

    # -----------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> list[dict]:
        """Extract a JSON array from LLM output, even if wrapped in prose/markdown."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
            # Remove closing fence
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        # Try direct parse on cleaned text
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: find first JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: find first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response:\n{text[:500]}")

    def _save_response(self, response: str, dataset_name: str, prompt_variant: str) -> None:
        """Save raw LLM response to disk for reproducibility."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{self.model_key}_{prompt_variant}_{timestamp}.txt"
        (LLM_OUTPUTS_DIR / filename).write_text(response, encoding="utf-8")


# =====================================================================
if __name__ == "__main__":
    print("Testing LLM client...\n")
    llm = LLMClient("gpt-4o-mini")
    response = llm.chat('Reply with only this JSON: ["hello", "world"]')
    print(f"Raw response:  {response}")
    print(f"Parsed JSON:   {llm._parse_json(response)}")
