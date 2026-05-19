from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
QUERIES_PATH = DATA_DIR / "queries.csv"
ACTORS_PATH = DATA_DIR / "random_actors.json"
RESULTS_PATH = OUTPUTS_DIR / "results.json"
DEBUG_RESULTS_PATH = OUTPUTS_DIR / "debug_results.json"

VECTOR_WEIGHT = 0.55
GRAPH_WEIGHT = 0.35
STRUCTURED_WEIGHT = 0.10
TOP_K = 5

LOCATION_ALIASES = {
    "blr": "bangalore",
    "bengaluru": "bangalore",
    "bangalore": "bangalore",
    "san francisco": "san_francisco",
    "sf": "san_francisco",
}

QUERY_NORMALIZATIONS = {
    "blr": "bangalore",
    "bengaluru": "bangalore",
    "sf": "san_francisco",
    "san francisco": "san_francisco",
    "ml": "machine_learning",
    "machine learning": "machine_learning",
    "vision stuff": "computer_vision vision_ai",
    "vison stuff": "computer_vision vision_ai",
    "vision ai": "computer_vision vision_ai",
    "computer vision": "computer_vision vision_ai",
}


def load_env_file(path: Path | None = None) -> None:
    env_path = path or (BASE_DIR / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
LOCAL_EMBEDDING_LOCAL_FILES_ONLY = (
    os.getenv("LOCAL_EMBEDDING_LOCAL_FILES_ONLY", "true").strip().lower() == "true"
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
).strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/embeddings"
).strip()
OPENROUTER_EXPLANATIONS_ENABLED = (
    os.getenv("OPENROUTER_EXPLANATIONS_ENABLED", "false").strip().lower() == "true"
)
OPENROUTER_EXPLANATION_MODEL = os.getenv(
    "OPENROUTER_EXPLANATION_MODEL", "anthropic/claude-opus-4.7-fast"
).strip()
OPENROUTER_CHAT_BASE_URL = os.getenv(
    "OPENROUTER_CHAT_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
).strip()
OPENROUTER_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30").strip())
MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "127.0.0.1").strip()
MEMGRAPH_PORT = int(os.getenv("MEMGRAPH_PORT", "7687").strip())
MEMGRAPH_USERNAME = os.getenv("MEMGRAPH_USERNAME", "").strip()
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "").strip()
MEMGRAPH_USE = os.getenv("MEMGRAPH_USE", "true").strip().lower() == "true"
HNSW_SPACE = os.getenv("HNSW_SPACE", "cosine").strip()
HNSW_M = int(os.getenv("HNSW_M", "16").strip())
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200").strip())
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "50").strip())
