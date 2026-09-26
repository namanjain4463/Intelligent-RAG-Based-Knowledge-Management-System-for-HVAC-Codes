"""Central configuration for the v2 runtime (loaded before clients are created)."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(Path(os.environ['HVAC_ENV_FILE']) if os.getenv('HVAC_ENV_FILE') else ROOT / '.env')

def resolve_vector_index(configured=None):
    # Migrate only the known v1 name; preserve explicit custom v2 indexes.
    name = configured if configured is not None else os.getenv('VECTOR_INDEX_NAME', 'hvac_passage_embeddings')
    return 'hvac_passage_embeddings' if name == 'hvac_requirement_embeddings' else name

@dataclass(frozen=True)
class RuntimeSettings:
    model: str = os.getenv('OPENAI_MODEL', 'gpt-5.6-luna')
    embedding_model: str = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
    vector_index: str = resolve_vector_index()
    vector_dimensions: int = int(os.getenv('VECTOR_DIMENSION', '3072'))
    api_timeout: float = 45.0
    max_question_chars: int = 4000
    max_request_bytes: int = 240_000
    max_result_bytes: int = 160_000
    max_output_tokens: int = 3500
    max_input_tokens: int = 100_000
    checkpoint_retention_days: int = 7

SETTINGS = RuntimeSettings()
