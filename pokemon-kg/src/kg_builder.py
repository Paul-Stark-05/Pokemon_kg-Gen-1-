"""
kg_builder.py — RDF Knowledge Graph Construction for Pokémon KG.

This module:
  1. Defines the Pokémon ontology (classes, properties, constraints).
  2. Ingests triples from nlp_pipeline.py into an rdflib Graph.
  3. Provides SPARQL query helpers used by the battle simulator.
  4. Exports the graph as Turtle (.ttl) or RDF/XML (.xml).

Ontology Overview:
  Classes  → :Pokemon, :Move, :Type, :StatusEffect
  Object Properties → :hasType, :learnsMove, :hasMoveType,
                       :superEffectiveAgainst, :notVeryEffectiveAgainst,
                       :immuneAgainst
  Datatype Properties → :basePower, :accuracy, :damageClass,
                         :baseStat_hp, :baseStat_attack, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Namespaces ───────────────────────────────────────────────────────────────

POKE = Namespace("http://pokemon-kg.example.org/ontology#")
DATA = Namespace("http://pokemon-kg.example.org/data#")


# ─── Ontology Definition ─────────────────────────────────────────────────────

def _build_ontology(g: Graph) -> None:
    """Define the Pokémon KG ontology (TBox) inside the graph."""

    # ── Classes ───────────────────────────────────────────────────────────
    for cls in ("Pokemon", "Move", "Type", "StatusEffect"):
        g.add((POKE[cls], RDF.type, OWL.Class))
        g.add((POKE[cls], RDFS.label, Literal(cls)))

    # ── Object Properties ─────────────────────────────────────────────────
    obj_props = {
        "hasType":                  ("Pokemon", "Type"),
        "learnsMove":               ("Pokemon", "Move"),
        "hasMoveType":              ("Move",    "Type"),
        "superEffectiveAgainst":    ("Type",    "Type"),
        "notVeryEffectiveAgainst":  ("Type",    "Type"),
        "immuneAgainst":            ("Type",    "Type"),
        "causesStatus":             ("Move",    "StatusEffect"),
    }
    for prop, (domain, range_) in obj_props.items():
        uri = POKE[prop]
        g.add((uri, RDF.type,    OWL.ObjectProperty))
        g.add((uri, RDFS.domain, POKE[domain]))
        g.add((uri, RDFS.range,  POKE[range_]))
        g.add((uri, RDFS.label,  Literal(prop)))

    # ── Datatype Properties ───────────────────────────────────────────────
    dt_props = {
        "basePower":       ("Move",    XSD.integer),
        "accuracy":        ("Move",    XSD.integer),
        "damageClass":     ("Move",    XSD.string),
        "baseStat_hp":     ("Pokemon", XSD.integer),
        "baseStat_attack": ("Pokemon", XSD.integer),
        "baseStat_defense":("Pokemon", XSD.integer),
        "baseStat_special-attack":  ("Pokemon", XSD.integer),
        "baseStat_special-defense": ("Pokemon", XSD.integer),
        "baseStat_speed":  ("Pokemon", XSD.integer),
    }
    for prop, (domain, datatype) in dt_props.items():
        uri = POKE[prop]
        g.add((uri, RDF.type,    OWL.DatatypeProperty))
        g.add((uri, RDFS.domain, POKE[domain]))
        g.add((uri, RDFS.range,  datatype))
        g.add((uri, RDFS.label,  Literal(prop)))


# ─── Entity URI helpers ───────────────────────────────────────────────────────

# Known entity categories for class assignment
TYPE_NAMES = {
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon",
}

STATUS_NAMES = {
    "burn", "freeze", "paralysis", "poison", "sleep", "confusion", "flinch",
}


def _uri(name: str) -> URIRef:
    """Create a DATA namespace URI from a name string."""
    safe = name.strip().lower().replace(" ", "_").replace("-", "_")
    return DATA[safe]


def _classify_entity(name: str) -> Optional[URIRef]:
    """Return the OWL class URI for an entity, or None if unknown."""
    n = name.strip().lower()
    if n in TYPE_NAMES:
        return POKE.Type
    if n in STATUS_NAMES:
        return POKE.StatusEffect
    return None  # We classify Pokémon and Moves based on predicates instead


# ─── Predicate mapping ────────────────────────────────────────────────────────

# Map triple predicate strings → (rdflib property URI, is_datatype_property)
PREDICATE_MAP: dict[str, tuple[URIRef, bool]] = {
    "hasType":                  (POKE.hasType,                  False),
    "learnsMove":               (POKE.learnsMove,               False),
    "hasMoveType":              (POKE.hasMoveType,              False),
    "superEffectiveAgainst":    (POKE.superEffectiveAgainst,    False),
    "notVeryEffectiveAgainst":  (POKE.notVeryEffectiveAgainst,  False),
    "immuneAgainst":            (POKE.immuneAgainst,            False),
    "causesStatus":             (POKE.causesStatus,             False),
    "basePower":                (POKE.basePower,                True),
    "accuracy":                 (POKE.accuracy,                 True),
    "damageClass":              (POKE.damageClass,              True),
}

# Stat predicates are dynamic: baseStat_hp, baseStat_attack, etc.
STAT_PREFIX = "baseStat_"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PokemonKnowledgeGraph:
    """
    Manages the construction, querying, and serialisation of the Pokémon KG.
    """

    def __init__(self):
        self.g = Graph()
        self.g.bind("poke", POKE)
        self.g.bind("data", DATA)
        self.g.bind("owl", OWL)
        _build_ontology(self.g)
        self._pokemon_set: set[str] = set()
        self._move_set: set[str] = set()

    # ── Ingestion ─────────────────────────────────────────────────────────

    def add_triples(self, triples: list) -> int:
        """
        Ingest a list of Triple objects (from nlp_pipeline) into the RDF graph.
        Returns the number of triples actually added.
        """
        added = 0
        for t in triples:
            subj = t.subject if isinstance(t, object) and hasattr(t, "subject") else t["subject"]
            pred = t.predicate if hasattr(t, "predicate") else t["predicate"]
            obj = t.obj if hasattr(t, "obj") else t["obj"]

            subj_uri = _uri(subj)
            pred_lower = pred.strip()

            # ── Handle stat predicates ────────────────────────────────────
            if pred_lower.startswith(STAT_PREFIX):
                stat_prop = POKE[pred_lower]
                # Ensure the Pokémon is typed
                self.g.add((subj_uri, RDF.type, POKE.Pokemon))
                self._pokemon_set.add(subj)
                try:
                    self.g.add((subj_uri, stat_prop, Literal(int(obj), datatype=XSD.integer)))
                except (ValueError, TypeError):
                    self.g.add((subj_uri, stat_prop, Literal(obj)))
                added += 1
                continue

            # ── Mapped predicates ─────────────────────────────────────────
            if pred_lower in PREDICATE_MAP:
                prop_uri, is_dt = PREDICATE_MAP[pred_lower]

                if is_dt:
                    # Datatype property → object is a literal
                    try:
                        lit = Literal(int(obj), datatype=XSD.integer)
                    except (ValueError, TypeError):
                        lit = Literal(obj)
                    self.g.add((subj_uri, prop_uri, lit))
                else:
                    # Object property → object is a URI
                    obj_uri = _uri(obj)
                    self.g.add((subj_uri, prop_uri, obj_uri))

                    # Auto-classify entities based on predicate usage
                    if pred_lower in ("hasType", "learnsMove"):
                        self.g.add((subj_uri, RDF.type, POKE.Pokemon))
                        self._pokemon_set.add(subj)
                    if pred_lower == "hasType":
                        self.g.add((obj_uri, RDF.type, POKE.Type))
                    if pred_lower == "learnsMove":
                        self.g.add((obj_uri, RDF.type, POKE.Move))
                        self._move_set.add(obj)
                    if pred_lower == "hasMoveType":
                        self.g.add((subj_uri, RDF.type, POKE.Move))
                        self.g.add((obj_uri, RDF.type, POKE.Type))
                        self._move_set.add(subj)
                    if pred_lower in ("superEffectiveAgainst",
                                       "notVeryEffectiveAgainst",
                                       "immuneAgainst"):
                        self.g.add((subj_uri, RDF.type, POKE.Type))
                        self.g.add((obj_uri, RDF.type, POKE.Type))

                added += 1
            else:
                # Unknown predicate — store as generic annotation
                self.g.add((subj_uri, POKE[pred_lower], Literal(obj)))
                added += 1

        return added

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None, fmt: str = "turtle") -> Path:
        """Export the graph. Supported formats: 'turtle', 'xml', 'n3', 'json-ld'."""
        import config
        if path is None:
            ext = {"turtle": ".ttl", "xml": ".xml", "n3": ".n3", "json-ld": ".jsonld"}
            path = config.OUTPUT_DIR / f"pokemon_kg{ext.get(fmt, '.ttl')}"
        self.g.serialize(destination=str(path), format=fmt)
        log.info(f"Graph saved → {path}  ({len(self.g)} triples, format={fmt})")
        return path

    # ── SPARQL Queries (used by battle_simulator.py) ──────────────────────

    def query(self, sparql: str) -> list[dict]:
        """Run a SPARQL SELECT query and return results as list of dicts."""
        results = []
        for row in self.g.query(sparql):
            results.append({str(var): str(val) for var, val in zip(row.labels, row)})
        return results

    def get_pokemon_types(self, pokemon_name: str) -> list[str]:
        """Return the type(s) of a Pokémon."""
        q = f"""
        SELECT ?type WHERE {{
            data:{pokemon_name.replace('-','_')} poke:hasType ?typeUri .
            ?typeUri rdfs:label ?type .
        }}
        """
        # Try with labels first; fall back to extracting from URI
        results = self.query(q)
        if results:
            return [r["type"] for r in results]

        # Fallback: extract type from URI fragment
        q2 = f"""
        SELECT ?typeUri WHERE {{
            data:{pokemon_name.replace('-','_')} poke:hasType ?typeUri .
        }}
        """
        results = self.query(q2)
        return [r["typeUri"].split("#")[-1].split("/")[-1] for r in results]

    def get_pokemon_moves(self, pokemon_name: str) -> list[dict]:
        """Return moves a Pokémon can learn with metadata."""
        safe_name = pokemon_name.replace("-", "_")
        q = f"""
        SELECT ?moveUri ?power ?acc ?mtype ?dmg WHERE {{
            data:{safe_name} poke:learnsMove ?moveUri .
            OPTIONAL {{ ?moveUri poke:basePower ?power . }}
            OPTIONAL {{ ?moveUri poke:accuracy ?acc . }}
            OPTIONAL {{ ?moveUri poke:hasMoveType ?mtype . }}
            OPTIONAL {{ ?moveUri poke:damageClass ?dmg . }}
        }}
        """
        results = self.query(q)
        moves = []
        for r in results:
            move_name = r.get("moveUri", "").split("#")[-1].split("/")[-1]
            mtype = r.get("mtype", "")
            if mtype:
                mtype = mtype.split("#")[-1].split("/")[-1]
            moves.append({
                "name": move_name,
                "power": int(r["power"]) if r.get("power") and r["power"] != "None" else None,
                "accuracy": int(r["acc"]) if r.get("acc") and r["acc"] != "None" else None,
                "type": mtype,
                "damage_class": r.get("dmg", "status"),
            })
        return moves

    def get_type_effectiveness(self, atk_type: str, def_type: str) -> float:
        """
        Query the KG for type effectiveness multiplier.
        Returns 2.0 (super effective), 0.5 (not very effective),
        0.0 (immune), or 1.0 (neutral).
        """
        safe_atk = atk_type.replace("-", "_")
        safe_def = def_type.replace("-", "_")

        # Check super effective
        q = f"""
        ASK {{ data:{safe_atk} poke:superEffectiveAgainst data:{safe_def} . }}
        """
        if bool(self.g.query(q)):
            return 2.0

        # Check not very effective
        q = f"""
        ASK {{ data:{safe_atk} poke:notVeryEffectiveAgainst data:{safe_def} . }}
        """
        if bool(self.g.query(q)):
            return 0.5

        # Check immune
        q = f"""
        ASK {{ data:{safe_atk} poke:immuneAgainst data:{safe_def} . }}
        """
        if bool(self.g.query(q)):
            return 0.0

        return 1.0  # Neutral

    def get_base_stat(self, pokemon_name: str, stat: str) -> Optional[int]:
        """Get a specific base stat value for a Pokémon."""
        safe_name = pokemon_name.replace("-", "_")
        stat_prop = f"baseStat_{stat}"
        q = f"""
        SELECT ?val WHERE {{
            data:{safe_name} poke:{stat_prop} ?val .
        }}
        """
        results = self.query(q)
        if results:
            try:
                return int(results[0]["val"])
            except (ValueError, KeyError):
                pass
        return None

    def validate_move(self, pokemon_name: str, move_name: str) -> bool:
        """Check if a Pokémon can legally use a move."""
        safe_poke = pokemon_name.replace("-", "_")
        safe_move = move_name.replace("-", "_")
        q = f"""
        ASK {{ data:{safe_poke} poke:learnsMove data:{safe_move} . }}
        """
        return bool(self.g.query(q))

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics about the graph."""
        return {
            "total_triples": len(self.g),
            "pokemon_count": len(self._pokemon_set),
            "move_count": len(self._move_set),
            "type_count": len(TYPE_NAMES),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILDER — Load triples from file and construct the KG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_from_triples_file(path: Optional[Path] = None) -> PokemonKnowledgeGraph:
    """
    Load extracted triples from a JSON file (output of nlp_pipeline)
    and construct the full KG.
    """
    import config
    if path is None:
        path = config.OUTPUT_DIR / "extracted_triples.json"

    log.info(f"Loading triples from {path}…")
    with open(path) as f:
        raw = json.load(f)

    kg = PokemonKnowledgeGraph()
    added = kg.add_triples(raw)
    log.info(f"Ingested {added} triples into RDF graph ({len(kg.g)} total with ontology)")
    return kg


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config

    triples_path = config.OUTPUT_DIR / "extracted_triples.json"
    if not triples_path.exists():
        log.error(f"No triples file found at {triples_path}. Run nlp_pipeline.py first.")
        raise SystemExit(1)

    kg = build_from_triples_file(triples_path)
    out = kg.save()

    # Also save as RDF/XML for compatibility
    kg.save(fmt="xml")

    stats = kg.stats()
    print(f"\n✓ Knowledge Graph built successfully!")
    print(f"  Triples:  {stats['total_triples']}")
    print(f"  Pokémon:  {stats['pokemon_count']}")
    print(f"  Moves:    {stats['move_count']}")
    print(f"  Types:    {stats['type_count']}")
    print(f"  Output:   {out}")
