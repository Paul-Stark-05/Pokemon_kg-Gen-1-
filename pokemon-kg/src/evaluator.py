"""
evaluator.py — Evaluation Framework for the Pokémon Knowledge Graph.

Measures the extracted KG's quality by comparing it against a hand-curated
"Gold Standard" set of triples. Computes:
  - Precision:  |extracted ∩ gold| / |extracted|
  - Recall:     |extracted ∩ gold| / |gold|
  - F1-score:   harmonic mean of Precision and Recall

Also provides per-relation-type breakdowns and battle-specific metrics.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """Container for evaluation results."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class BattleMetrics:
    """Metrics for evaluating battle simulation quality."""
    total_decisions: int = 0
    hallucinated_moves: int = 0       # moves the Pokémon can't learn
    type_errors: int = 0              # wrong super-effective / resist calls
    state_errors: int = 0             # HP / status inconsistencies
    valid_decisions: int = 0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated_moves / max(self.total_decisions, 1)

    @property
    def type_accuracy(self) -> float:
        type_related = self.type_errors + (self.total_decisions - self.type_errors)
        return 1.0 - (self.type_errors / max(type_related, 1))

    @property
    def state_consistency(self) -> float:
        return 1.0 - (self.state_errors / max(self.total_decisions, 1))

    def to_dict(self) -> dict:
        return {
            "total_decisions": self.total_decisions,
            "hallucinated_moves": self.hallucinated_moves,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "type_errors": self.type_errors,
            "type_accuracy": round(self.type_accuracy, 4),
            "state_errors": self.state_errors,
            "state_consistency": round(self.state_consistency, 4),
            "valid_decisions": self.valid_decisions,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRIPLE EVALUATION (KG vs Gold Standard)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalise_triple(s: str, p: str, o: str) -> tuple[str, str, str]:
    """Normalise a triple for comparison (lowercase, strip whitespace, underscores)."""
    def _norm(x: str) -> str:
        return x.strip().lower().replace("-", "_").replace(" ", "_")
    return (_norm(s), _norm(p), _norm(o))


def load_gold_standard(path: Optional[Path] = None) -> list[tuple[str, str, str]]:
    """
    Load the gold standard triples from JSON.
    Expected format: [{"subject": "...", "predicate": "...", "object": "..."}, ...]
    """
    import config
    if path is None:
        path = config.GOLD_STANDARD_FILE

    with open(path) as f:
        raw = json.load(f)

    return [
        _normalise_triple(t["subject"], t["predicate"], t["object"])
        for t in raw
    ]


def load_extracted_triples(path: Optional[Path] = None) -> list[tuple[str, str, str]]:
    """Load the extracted triples from the pipeline output."""
    import config
    if path is None:
        path = config.OUTPUT_DIR / "extracted_triples.json"

    with open(path) as f:
        raw = json.load(f)

    return [
        _normalise_triple(t["subject"], t["predicate"], t["obj"])
        for t in raw
    ]


def evaluate_triples(
    extracted: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> EvalMetrics:
    """
    Compute Precision, Recall, and F1 for extracted triples vs gold standard.
    Uses exact triple matching after normalisation.
    """
    extracted_set = set(extracted)
    gold_set = set(gold)

    tp = len(extracted_set & gold_set)
    fp = len(extracted_set - gold_set)
    fn = len(gold_set - extracted_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-10)

    return EvalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def evaluate_by_relation(
    extracted: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> dict[str, EvalMetrics]:
    """
    Break down evaluation by relation type for finer-grained analysis.
    Returns {relation_name: EvalMetrics}.
    """
    # Group triples by predicate
    ext_by_rel: dict[str, set] = defaultdict(set)
    gold_by_rel: dict[str, set] = defaultdict(set)

    for s, p, o in extracted:
        ext_by_rel[p].add((s, p, o))
    for s, p, o in gold:
        gold_by_rel[p].add((s, p, o))

    all_relations = set(ext_by_rel.keys()) | set(gold_by_rel.keys())
    results = {}

    for rel in sorted(all_relations):
        ext_set = ext_by_rel.get(rel, set())
        gold_set = gold_by_rel.get(rel, set())
        tp = len(ext_set & gold_set)
        fp = len(ext_set - gold_set)
        fn = len(gold_set - ext_set)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-10)
        results[rel] = EvalMetrics(precision, recall, f1, tp, fp, fn)

    return results


def find_missing_triples(
    extracted: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Return gold-standard triples that were NOT extracted (false negatives)."""
    return list(set(gold) - set(extracted))


def find_spurious_triples(
    extracted: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Return extracted triples that are NOT in the gold standard (false positives)."""
    return list(set(extracted) - set(gold))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATTLE EVALUATION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_battle_decision(
    decision: dict,
    kg,      # PokemonKnowledgeGraph instance
    pokemon_name: str,
    opponent_types: list[str],
) -> dict[str, bool]:
    """
    Validate a single battle decision from an LLM agent.

    Args:
        decision: {"move": "flamethrower", "reasoning": "..."} from the LLM
        kg: The PokemonKnowledgeGraph to check against
        pokemon_name: The active Pokémon making the decision
        opponent_types: The opponent Pokémon's types

    Returns:
        {"move_legal": bool, "type_correct": bool, "reasoning_sound": bool}
    """
    move_name = decision.get("move", "").strip().lower().replace(" ", "-")

    # Check if the move is legal (Pokémon actually learns it)
    move_legal = kg.validate_move(pokemon_name, move_name)

    # Check type reasoning
    move_info = None
    moves = kg.get_pokemon_moves(pokemon_name)
    for m in moves:
        if m["name"] == move_name:
            move_info = m
            break

    type_correct = True
    if move_info and move_info.get("type"):
        for opp_type in opponent_types:
            eff = kg.get_type_effectiveness(move_info["type"], opp_type)
            reasoning = decision.get("reasoning", "").lower()
            # Flag if the LLM claims super-effective but it's not (or vice versa)
            if "super effective" in reasoning and eff < 2.0:
                type_correct = False
            if "not very effective" in reasoning and eff >= 1.0:
                type_correct = False

    return {
        "move_legal": move_legal,
        "type_correct": type_correct,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REPORT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_evaluation_report(
    overall: EvalMetrics,
    by_relation: dict[str, EvalMetrics],
    battle_vanilla: Optional[BattleMetrics] = None,
    battle_grounded: Optional[BattleMetrics] = None,
) -> str:
    """
    Pretty-print a full evaluation report. Returns the report as a string.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    lines = []
    lines.append("=" * 70)
    lines.append("  POKÉMON KNOWLEDGE GRAPH — EVALUATION REPORT")
    lines.append("=" * 70)

    # Overall KG metrics
    lines.append("\n── KG Extraction Quality (vs Gold Standard) ──")
    lines.append(f"  Precision:        {overall.precision:.4f}")
    lines.append(f"  Recall:           {overall.recall:.4f}")
    lines.append(f"  F1-score:         {overall.f1:.4f}")
    lines.append(f"  True Positives:   {overall.true_positives}")
    lines.append(f"  False Positives:  {overall.false_positives}")
    lines.append(f"  False Negatives:  {overall.false_negatives}")

    # Per-relation breakdown
    lines.append("\n── Per-Relation Breakdown ──")
    if tabulate:
        table_data = [
            [rel, m.precision, m.recall, m.f1, m.true_positives, m.false_positives, m.false_negatives]
            for rel, m in by_relation.items()
        ]
        lines.append(tabulate(
            table_data,
            headers=["Relation", "Precision", "Recall", "F1", "TP", "FP", "FN"],
            floatfmt=".3f",
            tablefmt="simple",
        ))
    else:
        for rel, m in by_relation.items():
            lines.append(f"  {rel:30s}  P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1:.3f}")

    # Battle metrics (if provided)
    if battle_vanilla or battle_grounded:
        lines.append("\n── Battle Simulation Metrics ──")
        if battle_vanilla:
            lines.append("\n  Agent A (Vanilla LLM):")
            for k, v in battle_vanilla.to_dict().items():
                lines.append(f"    {k:25s}  {v}")
        if battle_grounded:
            lines.append("\n  Agent B (KG-Grounded LLM):")
            for k, v in battle_grounded.to_dict().items():
                lines.append(f"    {k:25s}  {v}")
        if battle_vanilla and battle_grounded:
            lines.append("\n  ── Improvement Summary ──")
            h_imp = battle_vanilla.hallucination_rate - battle_grounded.hallucination_rate
            t_imp = battle_grounded.type_accuracy - battle_vanilla.type_accuracy
            s_imp = battle_grounded.state_consistency - battle_vanilla.state_consistency
            lines.append(f"    Hallucination reduction:    {h_imp:+.4f}")
            lines.append(f"    Type accuracy improvement:  {t_imp:+.4f}")
            lines.append(f"    State consistency gain:     {s_imp:+.4f}")

    lines.append("\n" + "=" * 70)
    report = "\n".join(lines)
    return report


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config

    # Load data
    gold = load_gold_standard()
    extracted = load_extracted_triples()

    log.info(f"Gold standard: {len(gold)} triples")
    log.info(f"Extracted:     {len(extracted)} triples")

    # Evaluate
    overall = evaluate_triples(extracted, gold)
    by_rel = evaluate_by_relation(extracted, gold)

    # Print report
    report = print_evaluation_report(overall, by_rel)
    print(report)

    # Save report
    report_path = config.OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report saved → {report_path}")

    # Save JSON metrics
    metrics_path = config.OUTPUT_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "overall": overall.to_dict(),
            "by_relation": {k: v.to_dict() for k, v in by_rel.items()},
        }, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")
