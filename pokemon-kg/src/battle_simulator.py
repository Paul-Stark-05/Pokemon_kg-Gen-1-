"""
battle_simulator.py — LLM Testing Arena: Vanilla vs KG-Grounded Battle.

This module sets up a text-based Pokémon battle simulation that compares:
  Agent A (Vanilla):   Prompts the LLM with the battle state directly.
  Agent B (Grounded):  Queries the Knowledge Graph via SPARQL, injects
                       retrieved facts into the LLM prompt (RAG pattern).

Both agents play against a deterministic opponent (or each other) and every
decision is logged and validated for:
  - Illegal moves (hallucinations)
  - Incorrect type-effectiveness reasoning
  - Game-state consistency errors
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Lazy-loaded LLM
_llm_model = None
_llm_tokenizer = None


def _load_llm():
    """Load the HuggingFace model and tokenizer directly (not via pipeline)."""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        import config
        if config.USE_ANTHROPIC and config.ANTHROPIC_API_KEY:
            log.info("Using Anthropic API for LLM inference")
            return None, None

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = config.LLM_MODEL_NAME
        log.info(f"Loading LLM: {model_name}")
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        log.info(f"LLM loaded successfully: {model_name}")

    return _llm_model, _llm_tokenizer


def _call_llm(prompt: str) -> str:
    """Call the LLM (either local or Anthropic API)."""
    import config
    if config.USE_ANTHROPIC and config.ANTHROPIC_API_KEY:
        return _call_anthropic(prompt)

    model, tokenizer = _load_llm()
    if model is None or tokenizer is None:
        log.error("LLM model not loaded")
        return '{"move": "tackle", "reasoning": "model not loaded"}'

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log.info(f"    LLM raw output: {result[:120]}...")
        return result.strip()
    except Exception as exc:
        log.error(f"    LLM inference error: {exc}")
        return '{"move": "tackle", "reasoning": "inference error"}'


def _call_anthropic(prompt: str) -> str:
    """Call the Anthropic API."""
    import config
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except ImportError:
        log.error("anthropic package not installed. pip install anthropic")
        return '{"move": "tackle", "reasoning": "fallback"}'
    except Exception as exc:
        log.error(f"Anthropic API error: {exc}")
        return '{"move": "tackle", "reasoning": "fallback"}'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATTLE STATE DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BattlePokemon:
    """A Pokemon instance in battle, with current HP and status."""
    name: str
    types: list[str]
    moves: list[dict]       # [{name, type, power, accuracy, damage_class}]
    max_hp: int
    current_hp: int
    attack: int
    defense: int
    special_attack: int
    special_defense: int
    speed: int
    status: Optional[str] = None    # burn, freeze, paralysis, poison, sleep
    is_fainted: bool = False

    def take_damage(self, dmg: int):
        self.current_hp = max(0, self.current_hp - dmg)
        if self.current_hp == 0:
            self.is_fainted = True

    def hp_fraction(self) -> str:
        return f"{self.current_hp}/{self.max_hp}"

    def to_state_dict(self) -> dict:
        return {
            "name": self.name,
            "types": self.types,
            "hp": self.hp_fraction(),
            "status": self.status,
            "moves": [m["name"] for m in self.moves],
        }


@dataclass
class TurnLog:
    """Log of one turn in a battle."""
    turn: int
    agent: str                # "vanilla" or "grounded"
    pokemon: str
    opponent: str
    decision: dict            # raw LLM output parsed
    move_used: str
    damage_dealt: int
    was_legal: bool           # did the Pokemon actually know this move?
    type_correct: bool        # was type reasoning accurate?
    kg_context: Optional[str] = None  # injected KG facts (grounded only)
    raw_llm_output: str = ""


@dataclass
class BattleResult:
    """Full result of a battle simulation."""
    winner: Optional[str]     # "vanilla", "grounded", or "draw"
    turns: list[TurnLog] = field(default_factory=list)
    vanilla_hallucinations: int = 0
    grounded_hallucinations: int = 0
    vanilla_type_errors: int = 0
    grounded_type_errors: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DAMAGE CALCULATION (Simplified Gen 1 formula)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_damage(
    attacker: BattlePokemon,
    defender: BattlePokemon,
    move: dict,
    type_effectiveness: float,
) -> int:
    """
    Simplified Gen 1 damage formula:
      damage = ((2*50/5 + 2) * power * A/D) / 50 + 2) * STAB * type * random
    where level is fixed at 50 for this PoC.
    """
    power = move.get("power")
    if power is None or power == 0:
        return 0  # Status move — no direct damage

    # Use physical or special stats based on damage class
    if move.get("damage_class") == "special":
        atk_stat = attacker.special_attack
        def_stat = defender.special_defense
    else:
        atk_stat = attacker.attack
        def_stat = defender.defense

    # Prevent division by zero
    def_stat = max(def_stat, 1)

    level = 50
    base = ((2 * level / 5 + 2) * power * atk_stat / def_stat) / 50 + 2

    # STAB (Same Type Attack Bonus): 1.5x if move type matches attacker's type
    stab = 1.5 if move.get("type") in attacker.types else 1.0

    # Type effectiveness
    eff = type_effectiveness

    # Random factor (0.85-1.0)
    rand_factor = random.uniform(0.85, 1.0)

    damage = int(base * stab * eff * rand_factor)
    return max(1, damage)  # Minimum 1 damage for non-status moves


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM AGENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_vanilla_prompt(
    active: BattlePokemon,
    opponent: BattlePokemon,
) -> str:
    """
    Agent A (Vanilla): prompt the LLM with battle state only.
    No knowledge graph facts injected.
    """
    moves_str = ", ".join(m["name"] for m in active.moves)
    return (
        f"You are playing a Pokemon battle. "
        f"Your Pokemon is {active.name} (type: {', '.join(active.types)}, HP: {active.hp_fraction()}). "
        f"Your available moves are: {moves_str}. "
        f"The opponent is {opponent.name} (type: {', '.join(opponent.types)}, HP: {opponent.hp_fraction()}). "
        f"Which single move should you use? Answer with just the move name and a short reason. "
        f"Format: MOVE: <move_name> REASON: <why>"
    )


def _build_grounded_prompt(
    active: BattlePokemon,
    opponent: BattlePokemon,
    kg_context: str,
) -> str:
    """
    Agent B (KG-Grounded): inject retrieved KG facts into the prompt.
    """
    moves_str = ", ".join(m["name"] for m in active.moves)
    return (
        f"You are playing a Pokemon battle. Use the VERIFIED FACTS below to make your decision. "
        f"FACTS: {kg_context} "
        f"Your Pokemon is {active.name} (type: {', '.join(active.types)}, HP: {active.hp_fraction()}). "
        f"Your available moves are: {moves_str}. "
        f"The opponent is {opponent.name} (type: {', '.join(opponent.types)}, HP: {opponent.hp_fraction()}). "
        f"Which single move should you use? Pick the best move based on the facts. "
        f"Format: MOVE: <move_name> REASON: <why>"
    )


def _retrieve_kg_context(
    kg,  # PokemonKnowledgeGraph
    active: BattlePokemon,
    opponent: BattlePokemon,
) -> str:
    """
    Query the KG via SPARQL to build a context block for the grounded agent.
    ONLY uses the 4 moves assigned to this BattlePokemon (not all KG moves).
    Retrieves: move details, type effectiveness, matchup analysis.
    """
    lines = []

    # Use the battle pokemon's actual 4-move set, not all KG moves
    moves = active.moves

    # 1. Move details (from the battle moveset, enriched with KG data)
    lines.append(f"{active.name}'s 4 battle moves:")
    for m in moves:
        move_type = m.get("type", "normal")
        parts = [f"  - {m['name']}"]
        parts.append(f"type={move_type}")
        if m.get("power"):
            parts.append(f"power={m['power']}")
        if m.get("accuracy"):
            parts.append(f"accuracy={m['accuracy']}%")
        if m.get("damage_class"):
            parts.append(f"class={m['damage_class']}")
        lines.append(", ".join(parts))

    # 2. Type effectiveness for each move against opponent
    lines.append(f"\nType matchup vs {opponent.name} ({', '.join(opponent.types)}):")
    for m in moves:
        move_type = m.get("type", "normal")
        if m.get("power") and m["power"] > 0:
            total_eff = 1.0
            for opp_type in opponent.types:
                total_eff *= kg.get_type_effectiveness(move_type, opp_type)
            if total_eff >= 2.0:
                label = "SUPER EFFECTIVE (2x damage)"
            elif total_eff == 0.0:
                label = "IMMUNE (0 damage) - DO NOT USE"
            elif total_eff < 1.0:
                label = f"not very effective ({total_eff}x damage)"
            else:
                label = "neutral (1x)"
            lines.append(f"  {m['name']} ({move_type}) vs {', '.join(opponent.types)}: {label}")

    # 3. Explicit recommendation — best move from the 4 available
    best_move = None
    best_score = -1
    for m in moves:
        if not m.get("power") or m["power"] == 0:
            continue
        total_eff = 1.0
        move_type = m.get("type", "normal")
        for opp_type in opponent.types:
            total_eff *= kg.get_type_effectiveness(move_type, opp_type)
        stab = 1.5 if move_type in active.types else 1.0
        score = m["power"] * total_eff * stab
        if score > best_score:
            best_score = score
            best_move = m

    if best_move:
        lines.append(f"\nBEST MOVE: {best_move['name']} (effective power: {best_score:.0f})")

    return "\n".join(lines)


def _parse_llm_decision(raw: str, available_moves: list[str]) -> dict:
    """
    Parse the LLM's response into a structured decision.
    Handles multiple output formats gracefully with fallbacks.
    """
    raw_lower = raw.lower().strip()

    # Try JSON format first
    json_match = re.search(r'\{[^}]+\}', raw)
    if json_match:
        try:
            decision = json.loads(json_match.group())
            if "move" in decision:
                decision["move"] = decision["move"].strip().lower().replace(" ", "-").replace("_", "-")
                return decision
        except json.JSONDecodeError:
            pass

    # Try "MOVE: xxx REASON: yyy" format
    move_match = re.search(r'MOVE:\s*([a-zA-Z_\- ]+?)(?:\s+REASON:|\s*$)', raw, re.IGNORECASE)
    if move_match:
        move_name = move_match.group(1).strip().lower().replace(" ", "-").replace("_", "-")
        reason_match = re.search(r'REASON:\s*(.+)', raw, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else raw
        # Check if parsed move matches any available move
        for avail in available_moves:
            avail_norm = avail.replace("_", "-")
            if move_name == avail_norm or move_name.replace("-", "") == avail_norm.replace("-", ""):
                return {"move": avail_norm, "reasoning": reason}

    # Fallback: look for any available move name mentioned in the text
    for move in available_moves:
        move_norm = move.replace("_", "-")
        move_spaced = move.replace("_", " ").replace("-", " ")
        if move_norm in raw_lower or move_spaced in raw_lower or move.replace("-", "") in raw_lower.replace("-", "").replace("_", "").replace(" ", ""):
            return {"move": move_norm, "reasoning": raw}

    # Last resort: pick the first available move
    fallback = available_moves[0].replace("_", "-") if available_moves else "struggle"
    return {"move": fallback, "reasoning": f"[PARSE FAILURE] Original: {raw[:200]}"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POKEMON FACTORY (from KG data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_battle_pokemon(name: str, kg) -> BattlePokemon:
    """
    Create a BattlePokemon from KG data. Falls back to defaults if data
    is missing (defensive coding for PoC robustness).
    """
    types = kg.get_pokemon_types(name)
    if not types:
        types = ["normal"]
        log.warning(f"No types found for {name}, defaulting to [normal]")

    moves_raw = kg.get_pokemon_moves(name)
    # Pick up to 4 damaging moves (prefer higher power), plus fill with status
    damaging = [m for m in moves_raw if m.get("power") and m["power"] > 0]
    damaging.sort(key=lambda m: m["power"] or 0, reverse=True)
    status = [m for m in moves_raw if not m.get("power") or m["power"] == 0]
    selected = damaging[:4]
    if len(selected) < 4:
        selected.extend(status[:4 - len(selected)])

    # HARD CAP: exactly 4 moves maximum, like the real games
    selected = selected[:4]

    if not selected:
        # Absolute fallback
        selected = [{"name": "tackle", "type": "normal", "power": 40,
                      "accuracy": 100, "damage_class": "physical"}]
        log.warning(f"No moves found for {name}, defaulting to [tackle]")

    log.info(f"    {name}'s battle moveset ({len(selected)} moves): "
             f"{[m['name'] for m in selected]}")

    # Stats from KG
    def _stat(s: str, default: int = 50) -> int:
        val = kg.get_base_stat(name, s)
        return val if val is not None else default

    hp_base = _stat("hp", 60)
    # Scale HP to a usable battle value (level 50 approximation)
    max_hp = hp_base * 2 + 110  # simplified: (base*2*50/100) + 50 + 10

    return BattlePokemon(
        name=name,
        types=types,
        moves=selected,
        max_hp=max_hp,
        current_hp=max_hp,
        attack=_stat("attack", 50),
        defense=_stat("defense", 50),
        special_attack=_stat("special-attack", 50),
        special_defense=_stat("special-defense", 50),
        speed=_stat("speed", 50),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATTLE ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_single_battle(
    agent_type: str,           # "vanilla" or "grounded"
    player_pokemon: BattlePokemon,
    opponent_pokemon: BattlePokemon,
    kg=None,                   # Required for grounded agent
) -> list[TurnLog]:
    """
    Run a 1v1 battle between an LLM agent and a deterministic opponent.
    The opponent always picks its highest-power move (simple heuristic).
    Returns the turn log.
    """
    import config
    turns: list[TurnLog] = []
    turn_num = 0

    while (not player_pokemon.is_fainted and
           not opponent_pokemon.is_fainted and
           turn_num < config.MAX_TURNS_PER_BATTLE):

        turn_num += 1
        log.info(f"  Turn {turn_num}: {player_pokemon.name} ({player_pokemon.hp_fraction()}) "
                 f"vs {opponent_pokemon.name} ({opponent_pokemon.hp_fraction()})")

        # ── Player's turn (LLM decision) ─────────────────────────────────
        available_moves = [m["name"] for m in player_pokemon.moves]
        kg_context = None

        if agent_type == "grounded" and kg is not None:
            kg_context = _retrieve_kg_context(kg, player_pokemon, opponent_pokemon)
            prompt = _build_grounded_prompt(player_pokemon, opponent_pokemon, kg_context)
        else:
            prompt = _build_vanilla_prompt(player_pokemon, opponent_pokemon)

        raw_output = _call_llm(prompt)
        decision = _parse_llm_decision(raw_output, available_moves)
        chosen_move_name = decision["move"]

        # Find the move data
        chosen_move = None
        for m in player_pokemon.moves:
            m_norm = m["name"].replace("_", "-")
            if m_norm == chosen_move_name or m["name"] == chosen_move_name:
                chosen_move = m
                break

        # Check legality
        was_legal = chosen_move is not None
        if not was_legal:
            # Hallucination! The LLM picked a move the Pokemon doesn't know.
            log.warning(f"    HALLUCINATION: {player_pokemon.name} doesn't know '{chosen_move_name}'")
            # Fall back to first available move
            chosen_move = player_pokemon.moves[0]
            chosen_move_name = chosen_move["name"]

        # Calculate type effectiveness
        type_mult = 1.0
        if chosen_move.get("type"):
            for opp_type in opponent_pokemon.types:
                if kg:
                    type_mult *= kg.get_type_effectiveness(chosen_move["type"], opp_type)
                else:
                    # Without KG, use config type chart directly
                    import config as cfg
                    chart = cfg.TYPE_CHART.get(chosen_move["type"], {})
                    type_mult *= chart.get(opp_type, 1.0)

        # Calculate and apply damage
        damage = calculate_damage(player_pokemon, opponent_pokemon, chosen_move, type_mult)
        opponent_pokemon.take_damage(damage)

        # Validate type reasoning
        type_correct = True
        reasoning = decision.get("reasoning", "").lower()
        if "super effective" in reasoning and type_mult < 2.0:
            type_correct = False
        if "not very effective" in reasoning and type_mult >= 1.0:
            type_correct = False

        turns.append(TurnLog(
            turn=turn_num,
            agent=agent_type,
            pokemon=player_pokemon.name,
            opponent=opponent_pokemon.name,
            decision=decision,
            move_used=chosen_move_name,
            damage_dealt=damage,
            was_legal=was_legal,
            type_correct=type_correct,
            kg_context=kg_context,
            raw_llm_output=raw_output,
        ))

        log.info(f"    {player_pokemon.name} used {chosen_move_name}! "
                 f"({damage} dmg, eff={type_mult}x) "
                 f"{'OK' if was_legal else 'ILLEGAL'}")

        if opponent_pokemon.is_fainted:
            log.info(f"    {opponent_pokemon.name} fainted!")
            break

        # ── Opponent's turn (deterministic — highest power move) ──────────
        opp_moves_with_power = [m for m in opponent_pokemon.moves if m.get("power") and m["power"] > 0]
        if opp_moves_with_power:
            opp_move = max(opp_moves_with_power, key=lambda m: m["power"])
        else:
            opp_move = opponent_pokemon.moves[0] if opponent_pokemon.moves else \
                       {"name": "struggle", "type": "normal", "power": 50,
                        "accuracy": 100, "damage_class": "physical"}

        opp_type_mult = 1.0
        if opp_move.get("type"):
            for ptype in player_pokemon.types:
                if kg:
                    opp_type_mult *= kg.get_type_effectiveness(opp_move["type"], ptype)
                else:
                    import config as cfg
                    chart = cfg.TYPE_CHART.get(opp_move["type"], {})
                    opp_type_mult *= chart.get(ptype, 1.0)

        opp_damage = calculate_damage(opponent_pokemon, player_pokemon, opp_move, opp_type_mult)
        player_pokemon.take_damage(opp_damage)

        log.info(f"    {opponent_pokemon.name} used {opp_move['name']}! ({opp_damage} dmg)")

        if player_pokemon.is_fainted:
            log.info(f"    {player_pokemon.name} fainted!")

    return turns


def run_comparative_battle(
    pokemon_name: str,
    opponent_name: str,
    kg,
) -> dict:
    """
    Run the same battle scenario twice:
      1. Agent A (Vanilla) — no KG context
      2. Agent B (Grounded) — with KG context

    Returns a comparison report.
    """
    log.info(f"\n{'='*60}")
    log.info(f"  BATTLE: {pokemon_name} vs {opponent_name}")
    log.info(f"{'='*60}")

    # ── Run Vanilla Battle ────────────────────────────────────────────────
    log.info(f"\n-- Agent A (Vanilla LLM) --")
    player_v = create_battle_pokemon(pokemon_name, kg)
    opponent_v = create_battle_pokemon(opponent_name, kg)
    vanilla_log = run_single_battle("vanilla", player_v, opponent_v, kg=None)

    # ── Run Grounded Battle ───────────────────────────────────────────────
    log.info(f"\n-- Agent B (KG-Grounded LLM) --")
    player_g = create_battle_pokemon(pokemon_name, kg)
    opponent_g = create_battle_pokemon(opponent_name, kg)
    grounded_log = run_single_battle("grounded", player_g, opponent_g, kg=kg)

    # ── Aggregate metrics ─────────────────────────────────────────────────
    def _count(log_entries, field):
        return sum(1 for t in log_entries if not getattr(t, field, True))

    result = {
        "matchup": f"{pokemon_name} vs {opponent_name}",
        "vanilla": {
            "turns": len(vanilla_log),
            "won": opponent_v.is_fainted and not player_v.is_fainted,
            "hallucinations": _count(vanilla_log, "was_legal"),
            "type_errors": _count(vanilla_log, "type_correct"),
            "final_hp": player_v.hp_fraction(),
        },
        "grounded": {
            "turns": len(grounded_log),
            "won": opponent_g.is_fainted and not player_g.is_fainted,
            "hallucinations": _count(grounded_log, "was_legal"),
            "type_errors": _count(grounded_log, "type_correct"),
            "final_hp": player_g.hp_fraction(),
        },
        "turn_logs": {
            "vanilla": [asdict(t) for t in vanilla_log],
            "grounded": [asdict(t) for t in grounded_log],
        },
    }

    # Print summary
    log.info(f"\n{'-'*60}")
    log.info(f"  RESULT SUMMARY")
    log.info(f"{'-'*60}")
    log.info(f"  Vanilla:  {'WON' if result['vanilla']['won'] else 'LOST'} "
             f"in {result['vanilla']['turns']} turns | "
             f"Hallucinations: {result['vanilla']['hallucinations']} | "
             f"Type errors: {result['vanilla']['type_errors']}")
    log.info(f"  Grounded: {'WON' if result['grounded']['won'] else 'LOST'} "
             f"in {result['grounded']['turns']} turns | "
             f"Hallucinations: {result['grounded']['hallucinations']} | "
             f"Type errors: {result['grounded']['type_errors']}")

    return result


def run_tournament(
    matchups: list[tuple[str, str]],
    kg,
) -> list[dict]:
    """
    Run a series of comparative battles and aggregate results.
    Each matchup is (player_pokemon_name, opponent_pokemon_name).
    """
    all_results = []
    for player, opponent in matchups:
        try:
            result = run_comparative_battle(player, opponent, kg)
            all_results.append(result)
        except Exception as exc:
            log.error(f"Battle {player} vs {opponent} failed: {exc}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "matchup": f"{player} vs {opponent}",
                "error": str(exc),
            })

    # Print tournament summary
    log.info(f"\n{'='*60}")
    log.info(f"  TOURNAMENT SUMMARY ({len(all_results)} battles)")
    log.info(f"{'='*60}")

    v_hall = sum(r.get("vanilla", {}).get("hallucinations", 0) for r in all_results if "vanilla" in r)
    g_hall = sum(r.get("grounded", {}).get("hallucinations", 0) for r in all_results if "grounded" in r)
    v_terr = sum(r.get("vanilla", {}).get("type_errors", 0) for r in all_results if "vanilla" in r)
    g_terr = sum(r.get("grounded", {}).get("type_errors", 0) for r in all_results if "grounded" in r)

    log.info(f"  Total Hallucinations -- Vanilla: {v_hall}, Grounded: {g_hall}")
    log.info(f"  Total Type Errors    -- Vanilla: {v_terr}, Grounded: {g_terr}")
    if v_hall > 0:
        reduction = ((v_hall - g_hall) / v_hall) * 100
        log.info(f"  Hallucination Reduction: {reduction:.1f}%")

    return all_results


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import config
    from src.kg_builder import build_from_triples_file

    parser = argparse.ArgumentParser(description="Pokemon KG -- Battle Simulator")
    parser.add_argument("--matchups", type=str, default="charizard:blastoise,pikachu:geodude,venusaur:arcanine",
                        help="Comma-separated matchups in format 'player:opponent'")
    args = parser.parse_args()

    # Load the Knowledge Graph
    kg = build_from_triples_file()

    # Parse matchups
    matchups = []
    for pair in args.matchups.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            matchups.append((parts[0].strip(), parts[1].strip()))
        else:
            log.warning(f"Invalid matchup format: {pair}")

    if not matchups:
        matchups = [("charizard", "blastoise"), ("pikachu", "geodude"), ("venusaur", "arcanine")]

    # Run tournament
    results = run_tournament(matchups, kg)

    # Save results
    output_path = config.BATTLE_LOG_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nBattle log saved -> {output_path}")
