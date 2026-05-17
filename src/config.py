"""Central configuration for the LLM feature engineering experiments."""
from pathlib import Path

# ----- Paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LLM_OUTPUTS_DIR = RESULTS_DIR / "llm_outputs"

# ----- Reproducibility -----
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ----- Datasets -----
CHURN_FILE = DATA_RAW / "telco_churn.csv"
HOUSING_FILE = DATA_RAW / "ames_housing.csv"
BANK_FILE = DATA_RAW / "bank_marketing.csv"

# ----- LLM Models (OpenRouter IDs) -----
LLM_MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "gemini-flash": "google/gemini-2.0-flash-001",
    "deepseek-v3": "deepseek/deepseek-chat",
    "llama-3.3": "meta-llama/llama-3.3-70b-instruct",
    "qwen-2.5": "qwen/qwen-2.5-72b-instruct",

    # Frontier LLMs
    "gpt-5.5": "openai/gpt-5.5",
    "claude-opus-4.6": "anthropic/claude-opus-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "deepseek-v4": "deepseek/deepseek-v4-pro",
}

# ----- LLM Settings -----
LLM_TEMPERATURE = 0.3  # Low for consistency
LLM_MAX_TOKENS = 4000

# Auto-create result directories
for d in [DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, LLM_OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
