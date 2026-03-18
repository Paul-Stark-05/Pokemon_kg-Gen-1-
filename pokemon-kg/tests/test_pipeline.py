"""
test_pipeline.py — Unit tests for the Pokémon KG pipeline.

Run with: python -m pytest tests/test_pipeline.py -v
"""

import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.kg_builder import PokemonKnowledgeGraph, POKE, DATA
from src.evaluator import (
    evaluate_triples, evaluate_by_relation,
    find_missing_triples, find_spurious_triples,
    EvalMetrics,
)
from src.nlp_pipeline import (
    Triple, extract_structured_triples, PokemonRecord,
    build_type_effectiveness_triples, resolve_coreferences,
    normalise_entity, TYPE_NAMES, POKEMON_NAMES,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP PIPELINE TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNLPPipeline:
    """Tests for nlp_pipeline.py functions."""

    def test_extract_structured_triples_basic(self):
        """Verify that structured extraction produces correct triples."""
        record = PokemonRecord(
            id=4, name="charmander",
            types=["fire"],
            moves=[{"name": "ember", "type": "fire", "power": 40,
                     "accuracy": 100, "damage_class": "special"}],
            stats={"hp": 39, "attack": 52},
        )
        triples = extract_structured_triples(record)
        tuples = [t.as_tuple() for t in triples]

        assert ("charmander", "hasType", "fire") in tuples
        assert ("charmander", "learnsMove", "ember") in tuples
        assert ("ember", "hasMoveType", "fire") in tuples
        assert ("ember", "basePower", "40") in tuples

    def test_extract_structured_triples_multi_type(self):
        """Pokémon with two types should produce two hasType triples."""
        record = PokemonRecord(
            id=6, name="charizard",
            types=["fire", "flying"],
            moves=[], stats={},
        )
        triples = extract_structured_triples(record)
        tuples = [t.as_tuple() for t in triples]

        assert ("charizard", "hasType", "fire") in tuples
        assert ("charizard", "hasType", "flying") in tuples

    def test_type_effectiveness_triples(self):
        """Verify type chart produces expected matchups."""
        triples = build_type_effectiveness_triples()
        tuples = [t.as_tuple() for t in triples]

        assert ("fire", "superEffectiveAgainst", "grass") in tuples
        assert ("water", "superEffectiveAgainst", "fire") in tuples
        assert ("electric", "immuneAgainst", "ground") in tuples
        assert ("normal", "immuneAgainst", "ghost") in tuples

    def test_coreference_resolution(self):
        """Test pronoun replacement in flavor text."""
        text = "It uses its tail to attack opponents."
        resolved = resolve_coreferences(text, "pikachu")
        assert "pikachu" in resolved.lower()
        assert "it " not in resolved.lower().split("pikachu")[0]  # 'it' should be replaced

    def test_normalise_entity_types(self):
        """Known types should normalise correctly."""
        assert normalise_entity("Fire") == "fire"
        assert normalise_entity("WATER") == "water"
        assert normalise_entity("grass") == "grass"
        assert normalise_entity("nonsense") is None

    def test_normalise_entity_pokemon(self):
        """Pokémon names should normalise when in vocabulary."""
        POKEMON_NAMES.add("pikachu")
        POKEMON_NAMES.add("charizard")
        assert normalise_entity("Pikachu") == "pikachu"
        assert normalise_entity("CHARIZARD") == "charizard"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KG BUILDER TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestKGBuilder:
    """Tests for kg_builder.py functions."""

    def _make_kg(self) -> PokemonKnowledgeGraph:
        """Build a small test KG."""
        kg = PokemonKnowledgeGraph()
        triples = [
            {"subject": "pikachu", "predicate": "hasType", "obj": "electric"},
            {"subject": "pikachu", "predicate": "learnsMove", "obj": "thunder-shock"},
            {"subject": "thunder-shock", "predicate": "hasMoveType", "obj": "electric"},
            {"subject": "thunder-shock", "predicate": "basePower", "obj": "40"},
            {"subject": "electric", "predicate": "superEffectiveAgainst", "obj": "water"},
            {"subject": "electric", "predicate": "immuneAgainst", "obj": "ground"},
            {"subject": "pikachu", "predicate": "baseStat_hp", "obj": "35"},
            {"subject": "pikachu", "predicate": "baseStat_attack", "obj": "55"},
        ]
        kg.add_triples(triples)
        return kg

    def test_add_triples(self):
        """Graph should contain the added triples."""
        kg = self._make_kg()
        assert len(kg.g) > 0

    def test_get_pokemon_types(self):
        """Should return correct types for a Pokémon."""
        kg = self._make_kg()
        types = kg.get_pokemon_types("pikachu")
        assert "electric" in types

    def test_get_pokemon_moves(self):
        """Should return move data including power."""
        kg = self._make_kg()
        moves = kg.get_pokemon_moves("pikachu")
        names = [m["name"] for m in moves]
        assert "thunder_shock" in names or "thunder-shock" in [n.replace("_", "-") for n in names]

    def test_type_effectiveness_super(self):
        """Electric vs Water should be 2.0."""
        kg = self._make_kg()
        eff = kg.get_type_effectiveness("electric", "water")
        assert eff == 2.0

    def test_type_effectiveness_immune(self):
        """Electric vs Ground should be 0.0."""
        kg = self._make_kg()
        eff = kg.get_type_effectiveness("electric", "ground")
        assert eff == 0.0

    def test_type_effectiveness_neutral(self):
        """Electric vs Fire should be 1.0 (neutral)."""
        kg = self._make_kg()
        eff = kg.get_type_effectiveness("electric", "fire")
        assert eff == 1.0

    def test_validate_move_legal(self):
        """Pikachu should know thunder-shock."""
        kg = self._make_kg()
        assert kg.validate_move("pikachu", "thunder-shock") is True

    def test_validate_move_illegal(self):
        """Pikachu should NOT know flamethrower."""
        kg = self._make_kg()
        assert kg.validate_move("pikachu", "flamethrower") is False

    def test_get_base_stat(self):
        """Should return the correct base stat."""
        kg = self._make_kg()
        hp = kg.get_base_stat("pikachu", "hp")
        assert hp == 35

    def test_serialise_turtle(self, tmp_path):
        """Graph should serialise to Turtle format without errors."""
        kg = self._make_kg()
        out = kg.save(tmp_path / "test.ttl", fmt="turtle")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_stats(self):
        """Stats dict should have expected keys."""
        kg = self._make_kg()
        s = kg.stats()
        assert "total_triples" in s
        assert "pokemon_count" in s
        assert s["pokemon_count"] >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATOR TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEvaluator:
    """Tests for evaluator.py functions."""

    def test_perfect_score(self):
        """Identical sets should give P=R=F1=1.0."""
        triples = [("a", "b", "c"), ("d", "e", "f")]
        m = evaluate_triples(triples, triples)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_no_overlap(self):
        """Disjoint sets should give P=R=F1=0."""
        extracted = [("a", "b", "c")]
        gold = [("x", "y", "z")]
        m = evaluate_triples(extracted, gold)
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_partial_overlap(self):
        """Verify P, R, F1 for partial overlap."""
        extracted = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
        gold = [("a", "b", "c"), ("d", "e", "f"), ("j", "k", "l")]
        m = evaluate_triples(extracted, gold)
        # TP=2, FP=1, FN=1
        assert m.true_positives == 2
        assert m.false_positives == 1
        assert m.false_negatives == 1
        assert abs(m.precision - 2/3) < 1e-6
        assert abs(m.recall - 2/3) < 1e-6

    def test_evaluate_by_relation(self):
        """Per-relation breakdown should separate different predicates."""
        extracted = [("a", "rel1", "b"), ("c", "rel2", "d")]
        gold = [("a", "rel1", "b"), ("c", "rel2", "x")]
        by_rel = evaluate_by_relation(extracted, gold)
        assert "rel1" in by_rel
        assert "rel2" in by_rel
        assert by_rel["rel1"].f1 == 1.0      # perfect match
        assert by_rel["rel2"].f1 < 1.0       # mismatch on object

    def test_find_missing_and_spurious(self):
        """Missing = FN, Spurious = FP."""
        extracted = [("a", "b", "c"), ("extra", "e", "f")]
        gold = [("a", "b", "c"), ("missing", "m", "n")]
        missing = find_missing_triples(extracted, gold)
        spurious = find_spurious_triples(extracted, gold)
        assert ("missing", "m", "n") in missing
        assert ("extra", "e", "f") in spurious


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATTLE SIMULATOR TESTS (no LLM calls — unit-test the engine)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBattleEngine:
    """Tests for battle_simulator.py damage calculation and helpers."""

    def test_calculate_damage_basic(self):
        from src.battle_simulator import BattlePokemon, calculate_damage

        attacker = BattlePokemon(
            name="pikachu", types=["electric"],
            moves=[{"name": "thunder-shock", "type": "electric", "power": 40,
                     "accuracy": 100, "damage_class": "special"}],
            max_hp=120, current_hp=120,
            attack=55, defense=40, special_attack=50,
            special_defense=50, speed=90,
        )
        defender = BattlePokemon(
            name="squirtle", types=["water"],
            moves=[], max_hp=130, current_hp=130,
            attack=48, defense=65, special_attack=50,
            special_defense=64, speed=43,
        )
        move = attacker.moves[0]
        # Electric vs Water = 2.0x super effective
        dmg = calculate_damage(attacker, defender, move, 2.0)
        assert dmg > 0
        assert isinstance(dmg, int)

    def test_status_move_zero_damage(self):
        from src.battle_simulator import BattlePokemon, calculate_damage

        attacker = BattlePokemon(
            name="pikachu", types=["electric"], moves=[],
            max_hp=120, current_hp=120,
            attack=55, defense=40, special_attack=50,
            special_defense=50, speed=90,
        )
        defender = BattlePokemon(
            name="squirtle", types=["water"], moves=[],
            max_hp=130, current_hp=130,
            attack=48, defense=65, special_attack=50,
            special_defense=64, speed=43,
        )
        status_move = {"name": "growl", "type": "normal", "power": None,
                       "accuracy": 100, "damage_class": "status"}
        dmg = calculate_damage(attacker, defender, status_move, 1.0)
        assert dmg == 0

    def test_take_damage_and_faint(self):
        from src.battle_simulator import BattlePokemon

        poke = BattlePokemon(
            name="test", types=["normal"], moves=[],
            max_hp=100, current_hp=100,
            attack=50, defense=50, special_attack=50,
            special_defense=50, speed=50,
        )
        poke.take_damage(60)
        assert poke.current_hp == 40
        assert not poke.is_fainted

        poke.take_damage(60)
        assert poke.current_hp == 0
        assert poke.is_fainted

    def test_parse_llm_decision(self):
        from src.battle_simulator import _parse_llm_decision

        raw = '{"move": "thunder-shock", "reasoning": "it is super effective"}'
        decision = _parse_llm_decision(raw, ["thunder-shock", "quick-attack"])
        assert decision["move"] == "thunder-shock"

    def test_parse_llm_decision_malformed(self):
        from src.battle_simulator import _parse_llm_decision

        raw = "I think pikachu should use thunder-shock because water is weak"
        decision = _parse_llm_decision(raw, ["thunder-shock", "quick-attack"])
        assert decision["move"] == "thunder-shock"

    def test_parse_llm_decision_fallback(self):
        from src.battle_simulator import _parse_llm_decision

        raw = "gibberish with no move names"
        decision = _parse_llm_decision(raw, ["tackle", "growl"])
        assert decision["move"] == "tackle"  # first available move


# ─── Run tests ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
