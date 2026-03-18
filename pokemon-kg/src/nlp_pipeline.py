"""
nlp_pipeline.py — Data Extraction & NLP Processing for Pokémon KG.

This module implements a two-stage pipeline:
  Stage 1: Structured data extraction from PokeAPI (reliable base facts).
  Stage 2: NLP enrichment from unstructured text (Bulbapedia flavor text,
           Pokédex descriptions) using HuggingFace transformers for NER
           and relationship extraction.

Output: A list of (subject, predicate, object) triples ready for kg_builder.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Lazy-loaded heavy imports (transformers / torch) ─────────────────────────
# We defer these so the module can be imported quickly for testing / inspection
# without triggering multi-GB model downloads.
_ner_pipeline = None
_re_pipeline = None


def _load_ner():
    """Load the NER pipeline on first use."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline as hf_pipeline
        import config
        log.info(f"Loading NER model: {config.NER_MODEL_NAME}")
        _ner_pipeline = hf_pipeline(
            "token-classification",
            model=config.NER_MODEL_NAME,
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def _load_re():
    """Load the Relationship Extraction (REBEL) pipeline on first use."""
    global _re_pipeline
    if _re_pipeline is None:
        from transformers import pipeline as hf_pipeline
        import config
        log.info(f"Loading RE model: {config.RE_MODEL_NAME}")
        _re_pipeline = hf_pipeline(
            "text2text-generation",
            model=config.RE_MODEL_NAME,
            max_length=256,
            num_beams=3,
        )
    return _re_pipeline


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Triple:
    """A single (subject, predicate, object) fact."""
    subject: str
    predicate: str
    obj: str                     # 'object' is a Python builtin, so we use 'obj'
    confidence: float = 1.0      # 1.0 for structured sources, <1.0 for NLP
    source: str = "pokeapi"      # provenance tag

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)


@dataclass
class PokemonRecord:
    """Intermediate representation of a single Pokémon's extracted data."""
    id: int
    name: str
    types: list[str] = field(default_factory=list)
    moves: list[dict] = field(default_factory=list)   # [{name, type, power, pp, accuracy, damage_class}]
    stats: dict[str, int] = field(default_factory=dict)
    flavor_texts: list[str] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 1 — Structured Data Extraction (PokeAPI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _api_get(url: str, retries: int = 3, backoff: float = 1.0) -> dict:
    """GET with retry + exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            log.warning(f"Retry {attempt+1}/{retries} for {url}: {exc} — waiting {wait:.1f}s")
            time.sleep(wait)


def fetch_pokemon(pokemon_id: int) -> PokemonRecord:
    """Fetch one Pokémon's core data from PokeAPI."""
    import config
    url = f"{config.POKEAPI_BASE_URL}/pokemon/{pokemon_id}"
    data = _api_get(url)
    name = data["name"]

    # Types
    types = [t["type"]["name"] for t in data["types"]]

    # Stats
    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}

    # Moves — limit to level-up moves for Gen 1 scope
    move_names = []
    for m in data["moves"]:
        for vg in m["version_group_details"]:
            if vg["move_learn_method"]["name"] == "level-up" and \
               vg["version_group"]["name"] in ("red-blue", "yellow"):
                move_names.append(m["move"]["name"])
                break

    record = PokemonRecord(
        id=pokemon_id,
        name=name,
        types=types,
        stats=stats,
    )

    # Fetch move details (batched, with rate-limit politeness)
    for mname in move_names[:20]:  # cap to avoid hammering the API
        try:
            mdata = _api_get(f"{config.POKEAPI_BASE_URL}/move/{mname}")
            record.moves.append({
                "name": mdata["name"],
                "type": mdata["type"]["name"],
                "power": mdata.get("power"),            # None for status moves
                "pp": mdata.get("pp", 0),
                "accuracy": mdata.get("accuracy"),
                "damage_class": mdata.get("damage_class", {}).get("name", "status"),
            })
        except Exception as exc:
            log.warning(f"  Skipping move {mname}: {exc}")

    # Flavor text from species endpoint
    try:
        species = _api_get(f"{config.POKEAPI_BASE_URL}/pokemon-species/{pokemon_id}")
        for entry in species.get("flavor_text_entries", []):
            if entry["language"]["name"] == "en":
                text = entry["flavor_text"].replace("\n", " ").replace("\f", " ").strip()
                if text not in record.flavor_texts:
                    record.flavor_texts.append(text)
    except Exception as exc:
        log.warning(f"  Could not fetch species data for #{pokemon_id}: {exc}")

    return record


def extract_structured_triples(record: PokemonRecord) -> list[Triple]:
    """
    Convert a PokemonRecord into structured (subject, predicate, object) triples.
    These come directly from the API — confidence = 1.0.
    """
    triples: list[Triple] = []
    poke = record.name

    # hasType
    for t in record.types:
        triples.append(Triple(poke, "hasType", t, 1.0, "pokeapi"))

    # learnsMove + move metadata
    for m in record.moves:
        triples.append(Triple(poke, "learnsMove", m["name"], 1.0, "pokeapi"))
        triples.append(Triple(m["name"], "hasMoveType", m["type"], 1.0, "pokeapi"))
        if m["power"] is not None:
            triples.append(Triple(m["name"], "basePower", str(m["power"]), 1.0, "pokeapi"))
        if m["accuracy"] is not None:
            triples.append(Triple(m["name"], "accuracy", str(m["accuracy"]), 1.0, "pokeapi"))
        triples.append(Triple(m["name"], "damageClass", m["damage_class"], 1.0, "pokeapi"))

    # Base stats
    for stat_name, stat_val in record.stats.items():
        triples.append(Triple(poke, f"baseStat_{stat_name}", str(stat_val), 1.0, "pokeapi"))

    return triples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 2 — NLP Enrichment (NER + Relation Extraction on flavor text)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Domain vocabulary for entity normalisation / co-reference resolution
POKEMON_NAMES: set[str] = set()      # populated at runtime after Stage 1
MOVE_NAMES: set[str] = set()
TYPE_NAMES = {
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon",
}
STATUS_EFFECTS = {
    "burn", "freeze", "paralysis", "poison", "sleep", "confusion", "flinch",
    "badly poisoned", "leech seed",
}

# Pronoun → entity co-reference map (simple rule-based)
PRONOUN_MAP = {"it", "its", "the pokémon", "this pokémon", "the opponent"}


def normalise_entity(text: str) -> Optional[str]:
    """
    Map a raw NER mention to a canonical entity in our domain vocabulary.
    Returns None if the mention doesn't match any known entity.
    """
    t = text.strip().lower().replace("-", "").replace(" ", "")
    # Check Pokémon names
    for name in POKEMON_NAMES:
        if name.replace("-", "") == t:
            return name
    # Check move names
    for name in MOVE_NAMES:
        if name.replace("-", "") == t:
            return name
    # Check types
    for name in TYPE_NAMES:
        if name == t:
            return name
    # Check status effects
    for name in STATUS_EFFECTS:
        if name.replace(" ", "") == t:
            return name
    return None


def resolve_coreferences(text: str, pokemon_name: str) -> str:
    """
    Simple rule-based co-reference resolution.
    Replaces pronouns like 'it', 'this Pokémon' with the actual Pokémon name
    when we know which Pokémon the flavor text is about.
    """
    resolved = text
    for pronoun in PRONOUN_MAP:
        # Case-insensitive whole-word replacement
        pattern = re.compile(rf"\b{re.escape(pronoun)}\b", re.IGNORECASE)
        resolved = pattern.sub(pokemon_name, resolved)
    return resolved


def run_ner(text: str) -> list[dict]:
    """
    Run Named Entity Recognition on a text string.
    Returns list of {entity, label, start, end, score}.
    """
    pipe = _load_ner()
    raw = pipe(text)
    # Post-filter: keep only entities that match our domain vocabulary
    results = []
    for ent in raw:
        canon = normalise_entity(ent["word"])
        if canon:
            results.append({
                "entity": canon,
                "label": ent["entity_group"],
                "score": round(ent["score"], 4),
                "start": ent["start"],
                "end": ent["end"],
            })
    return results


def _parse_rebel_output(generated_text: str) -> list[Triple]:
    """
    Parse the linearised output of REBEL into structured triples.
    REBEL outputs: '<triplet> subject <subj> relation <obj> object'
    """
    triples = []
    # Split on '<triplet>' markers
    parts = generated_text.strip().split("<triplet>")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            subj_split = part.split("<subj>")
            if len(subj_split) != 2:
                continue
            subject = subj_split[0].strip()
            rest = subj_split[1]
            obj_split = rest.split("<obj>")
            if len(obj_split) != 2:
                continue
            relation = obj_split[0].strip()
            obj = obj_split[1].strip()
            if subject and relation and obj:
                triples.append(Triple(
                    subject=subject.lower(),
                    predicate=relation.lower().replace(" ", "_"),
                    obj=obj.lower(),
                    confidence=0.7,
                    source="rebel_nlp",
                ))
        except (IndexError, ValueError):
            continue
    return triples


def run_relation_extraction(text: str) -> list[Triple]:
    """
    Run REBEL relationship extraction on a text string.
    Returns domain-filtered triples.
    """
    pipe = _load_re()
    output = pipe(text)
    if not output:
        return []

    generated = output[0].get("generated_text", "")
    raw_triples = _parse_rebel_output(generated)

    # Filter: keep triples where subject or object is a known entity
    filtered = []
    for t in raw_triples:
        subj_canon = normalise_entity(t.subject)
        obj_canon = normalise_entity(t.obj)
        if subj_canon or obj_canon:
            filtered.append(Triple(
                subject=subj_canon or t.subject,
                predicate=t.predicate,
                obj=obj_canon or t.obj,
                confidence=t.confidence,
                source=t.source,
            ))
    return filtered


def enrich_with_nlp(record: PokemonRecord) -> list[Triple]:
    """
    Run the full NLP pipeline on a Pokémon's flavor texts.
    Returns additional triples discovered from unstructured text.
    """
    nlp_triples: list[Triple] = []
    for text in record.flavor_texts[:3]:  # limit to avoid excessive compute
        # Co-reference resolution
        resolved = resolve_coreferences(text, record.name)
        # Relationship extraction
        triples = run_relation_extraction(resolved)
        nlp_triples.extend(triples)
    return nlp_triples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TYPE EFFECTIVENESS TRIPLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_type_effectiveness_triples() -> list[Triple]:
    """
    Generate triples from the Gen 1 type effectiveness chart.
    e.g. ('fire', 'superEffectiveAgainst', 'grass')
    """
    import config
    triples = []
    for atk_type, matchups in config.TYPE_CHART.items():
        for def_type, mult in matchups.items():
            if mult >= 2.0:
                triples.append(Triple(atk_type, "superEffectiveAgainst", def_type, 1.0, "type_chart"))
            elif mult == 0.5:
                triples.append(Triple(atk_type, "notVeryEffectiveAgainst", def_type, 1.0, "type_chart"))
            elif mult == 0.0:
                triples.append(Triple(atk_type, "immuneAgainst", def_type, 1.0, "type_chart"))
    return triples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(
    pokemon_count: Optional[int] = None,
    enable_nlp: bool = False,
) -> tuple[list[PokemonRecord], list[Triple]]:
    """
    Execute the full extraction pipeline.

    Args:
        pokemon_count: How many Gen 1 Pokémon to process (default: from config).
        enable_nlp: Whether to run the heavy NLP models (Stage 2). Set to False
                    for fast runs that rely only on structured PokeAPI data.

    Returns:
        (records, all_triples) — raw records and deduplicated triple list.
    """
    import config
    count = pokemon_count or config.GEN1_POKEMON_COUNT

    log.info(f"═══ Starting NLP Pipeline: {count} Pokémon, NLP={'ON' if enable_nlp else 'OFF'} ═══")

    all_records: list[PokemonRecord] = []
    all_triples: list[Triple] = []

    # ── Stage 1: Structured extraction ────────────────────────────────────
    for pid in range(1, count + 1):
        log.info(f"  [{pid:3d}/{count}] Fetching Pokémon…")
        try:
            record = fetch_pokemon(pid)
            all_records.append(record)

            # Populate domain vocabulary for NLP stage
            POKEMON_NAMES.add(record.name)
            for m in record.moves:
                MOVE_NAMES.add(m["name"])

            # Convert to triples
            struct_triples = extract_structured_triples(record)
            record.triples.extend(struct_triples)
            all_triples.extend(struct_triples)

            # Be polite to the API
            time.sleep(0.2)

        except Exception as exc:
            log.error(f"  Failed on #{pid}: {exc}")

    # ── Type effectiveness triples ────────────────────────────────────────
    type_triples = build_type_effectiveness_triples()
    all_triples.extend(type_triples)
    log.info(f"  Added {len(type_triples)} type-effectiveness triples")

    # ── Stage 2: NLP enrichment (optional) ────────────────────────────────
    if enable_nlp:
        log.info("  ── Starting NLP enrichment (REBEL + NER) ──")
        for record in all_records:
            nlp_triples = enrich_with_nlp(record)
            record.triples.extend(nlp_triples)
            all_triples.extend(nlp_triples)
            if nlp_triples:
                log.info(f"    {record.name}: +{len(nlp_triples)} NLP triples")

    # ── Deduplicate ───────────────────────────────────────────────────────
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Triple] = []
    for t in all_triples:
        key = t.as_tuple()
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    log.info(f"═══ Pipeline complete: {len(all_records)} Pokémon, {len(deduped)} unique triples ═══")

    # ── Persist raw triples for downstream modules ────────────────────────
    output_path = config.OUTPUT_DIR / "extracted_triples.json"
    with open(output_path, "w") as f:
        json.dump([asdict(t) for t in deduped], f, indent=2)
    log.info(f"  Saved triples → {output_path}")

    return all_records, deduped


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pokémon KG — NLP Pipeline")
    parser.add_argument("-n", "--count", type=int, default=None,
                        help="Number of Pokémon to fetch (default: config value)")
    parser.add_argument("--nlp", action="store_true",
                        help="Enable heavy NLP enrichment (REBEL + NER)")
    args = parser.parse_args()

    records, triples = run_pipeline(pokemon_count=args.count, enable_nlp=args.nlp)
    print(f"\nDone! {len(records)} Pokémon, {len(triples)} triples extracted.")
