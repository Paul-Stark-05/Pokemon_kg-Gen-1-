"""
config.py — Central configuration for the Pokémon Knowledge Graph pipeline.

Adjust these values to control data scope, model selection, and output paths.
"""

from pathlib import Path

# ─── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Data scope ───────────────────────────────────────────────────────────────
GEN1_POKEMON_COUNT = 151          # Gen 1: Bulbasaur (#1) → Mew (#151)
POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"

# ─── NLP model selection ──────────────────────────────────────────────────────
# NER: token-classification model for entity recognition
NER_MODEL_NAME = "dslim/bert-base-NER"

# Relationship Extraction: REBEL extracts (subject, relation, object) triples
RE_MODEL_NAME = "Babelscape/rebel-large"

# ─── LLM for battle simulation ───────────────────────────────────────────────
LLM_MODEL_NAME = "google/flan-t5-large"   # 780M params vs 250M  # local HuggingFace model (CPU-friendly)

# Anthropic API (optional — set USE_ANTHROPIC=True for higher quality)
USE_ANTHROPIC = False
ANTHROPIC_API_KEY = ""                     # paste your key here or set env var
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# ─── RDF / Ontology namespaces ────────────────────────────────────────────────
ONTOLOGY_NS = "http://pokemon-kg.example.org/ontology#"
DATA_NS = "http://pokemon-kg.example.org/data#"
KG_OUTPUT_FILE = OUTPUT_DIR / "pokemon_kg.ttl"

# ─── Battle simulation ───────────────────────────────────────────────────────
BATTLE_LOG_FILE = OUTPUT_DIR / "battle_log.json"
DEFAULT_TEAM_SIZE = 3               # 1v1 or 3v3
MAX_TURNS_PER_BATTLE = 50           # safety cap to avoid infinite loops

# ─── Evaluation ───────────────────────────────────────────────────────────────
GOLD_STANDARD_FILE = DATA_DIR / "gold_standard.json"

# ─── Visualisation ────────────────────────────────────────────────────────────
KG_VIS_OUTPUT = OUTPUT_DIR / "kg_visualisation.png"
KG_VIS_HTML = OUTPUT_DIR / "kg_interactive.html"

# ─── Gen 1 type effectiveness chart (attacker → defender → multiplier) ────────
# Simplified but accurate Gen 1 chart. 1.0 = neutral, 2.0 = super effective,
# 0.5 = not very effective, 0.0 = immune.
TYPE_CHART: dict[str, dict[str, float]] = {
    "normal":   {"rock": 0.5, "ghost": 0.0},
    "fire":     {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0,
                 "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water":    {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0,
                 "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0,
                 "flying": 2.0, "dragon": 0.5},
    "grass":    {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5,
                 "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0,
                 "dragon": 0.5},
    "ice":      {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5,
                 "ground": 2.0, "flying": 2.0, "dragon": 2.0},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5,
                 "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0},
    "poison":   {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5,
                 "ghost": 0.5},
    "ground":   {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0,
                 "flying": 0.0, "bug": 0.5, "rock": 2.0},
    "flying":   {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0,
                 "rock": 0.5},
    "psychic":  {"fighting": 2.0, "poison": 2.0, "psychic": 0.5},
    "bug":      {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5,
                 "flying": 0.5, "psychic": 2.0, "ghost": 0.5},
    "rock":     {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5,
                 "flying": 2.0, "bug": 2.0},
    "ghost":    {"normal": 0.0, "ghost": 2.0, "psychic": 0.0},
    "dragon":   {"dragon": 2.0},
}
