# Pokemon_kg - Gen 1
# Pokémon Knowledge Graph — LLM Grounding PoC

 **Can a Knowledge Graph reduce hallucinations in a text-based Pokémon battle?**
 This proof-of-concept builds an RDF Knowledge Graph from Gen 1 Pokémon data,
 then pits a KG-grounded LLM agent against a vanilla LLM agent to find out.
## Results

### Headline: +108% damage uplift with KG grounding

Across 3 deliberately type-disadvantaged matchups, the KG-grounded agent dealt **more than double** the damage of the vanilla agent by making smarter move choices informed by verified Knowledge Graph facts.

| Metric | Vanilla LLM | KG-Grounded LLM | Improvement |
|---|---|---|---|
| **Avg damage / turn** | 14 | 29 | **+108%** |
| **Total damage dealt** | 176 | 356 | **+102%** |
| **Optimal move selection** | 0 / 3 battles | 3 / 3 battles | **100% accuracy** |

### Battle-by-battle breakdown

#### ⚔️ Charizard vs Blastoise — *+92% damage*

<p align="center">
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/6.png" width="80" alt="Charizard">
  &nbsp;&nbsp;&nbsp;&nbsp;⚔️&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/9.png" width="80" alt="Blastoise">
</p>

| | Vanilla | KG-Grounded |
|---|---|---|
| **Move chosen** | Ember (fire) | Slash (normal) |
| **Type effectiveness** | 0.5× (resisted) | 1× (neutral) |
| **Avg damage / turn** | 13 | 26 |
| **Total damage** | 67 | 128 |

The vanilla agent repeatedly used Ember — a fire move against a water-type, dealing half damage. The grounded agent's SPARQL query revealed that fire moves are resisted by water and correctly switched to Slash, a neutral-type move with 70 base power.

#### ⚡ Pikachu vs Geodude — *Immunity detected*

<p align="center">
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/25.png" width="80" alt="Pikachu">
  &nbsp;&nbsp;&nbsp;&nbsp;⚔️&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/74.png" width="80" alt="Geodude">
</p>

| | Vanilla | KG-Grounded |
|---|---|---|
| **Move chosen** | Thunder (electric) | Slam (normal) |
| **Type effectiveness** | 0× (immune!) | 0.5× (resisted) |
| **Avg damage / turn** | 1 | 10 |
| **Total damage** | 1 | 10 |

The showcase matchup. Vanilla picked Thunder — electric against ground is **completely immune** (0× damage, floored to 1 by the game engine). The KG context explicitly flagged `"IMMUNE (0 damage) - DO NOT USE"` for all electric moves. The grounded agent selected Slam instead — the only move that could deal any real damage. Without the Knowledge Graph, the LLM had no way to know ground-types are immune to electric.

#### 🌿 Venusaur vs Arcanine — *+161% damage*

<p align="center">
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/3.png" width="80" alt="Venusaur">
  &nbsp;&nbsp;&nbsp;&nbsp;⚔️&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/59.png" width="80" alt="Arcanine">
</p>

| | Vanilla | KG-Grounded |
|---|---|---|
| **Move chosen** | Tackle (normal) | Solar Beam (grass) |
| **Type effectiveness** | 1× (neutral) | 0.5× (resisted) but STAB |
| **Avg damage / turn** | 18 | 47 |
| **Total damage** | 108 | 284 |

Vanilla defaulted to Tackle (40 base power, neutral). Grounded chose Solar Beam (120 base power, grass-type). Despite the 0.5× type resistance against fire, the raw power plus 1.5× STAB bonus made Solar Beam deal 2.6× more damage. The KG's effective-power formula `(power × type_effectiveness × STAB)` correctly identified Solar Beam as optimal.

### Key findings

Charizard vs Blastoise
+92% dmg
Vanilla repeatedly chose Ember (fire vs water = 0.5×, ~13 dmg/turn). The KG-grounded agent identified fire moves as resisted and switched to Slash (normal, neutral 1×, ~26 dmg/turn). The Knowledge Graph's type effectiveness query directly prevented the suboptimal fire-move choice.

Pikachu vs Geodude
Immunity detected
The most dramatic case. Vanilla picked Thunder (electric vs ground = immune, floor damage of 1). The grounded agent's KG context explicitly flagged "IMMUNE — DO NOT USE" for all electric moves and selected Slam instead (10 dmg). Without the KG, the LLM had no way to know ground-types are immune to electric.

Venusaur vs Arcanine
+161% dmg
Vanilla defaulted to Tackle (normal, 40 power, ~18 dmg/turn). Grounded chose Solar Beam (grass, 120 power). Despite 0.5× type resistance, the high base power + 1.5× STAB bonus gave ~47 dmg/turn. The KG's effective-power calculation (power × effectiveness × STAB) correctly identified Solar Beam as the optimal choice.

### Interactive dashboard

For the full interactive dashboard with Chart.js charts, Gen 3 sprites, and turn-by-turn analysis:

```bash
# Open locally in your browser
open output/dashboard.html
```
## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline end-to-end

```bash
# Step 1 — Extract data & build KG
python -m src.nlp_pipeline          # pulls Gen 1 data, runs NLP
python -m src.kg_builder            # constructs RDF graph → output/pokemon_kg.ttl

# Step 2 — Evaluate KG quality
python -m src.evaluator             # compares KG vs gold standard

# Step 3 — Visualise (for your GitHub README image)
python -m src.visualize_kg          # → output/kg_visualisation.png

# Step 4 — Run the battle arena
python -m src.battle_simulator      # Vanilla vs KG-Grounded LLM battle
```

### 3. Visualise the Knowledge Graph

After building the KG, run:
```bash
python -m src.visualize_kg
```
This produces `output/kg_visualisation.png` — a network diagram of the
ontology and a sample subgraph you can embed in your GitHub repo.

## Configuration

Edit `config.py` to change:

| Variable | Default | Description |
|---|---|---|
| `GEN1_POKEMON_COUNT` | 151 | Number of Pokémon to fetch |
| `LLM_MODEL_NAME` | `google/flan-t5-large` | HuggingFace model for battle agents |
| `NER_MODEL_NAME` | `dslim/bert-base-NER` | NER model for entity extraction |
| `RE_MODEL_NAME` | `Babelscape/rebel-large` | Relationship extraction model |
| `ANTHROPIC_API_KEY` | `""` | Optional — use Claude API instead of local LLM |
| `USE_ANTHROPIC` | `False` | Toggle Anthropic API usage |

## LLM Options

The battle simulator supports two backends:

1. **Local HuggingFace model** (default): Uses `google/flan-t5-base`. Free,
   runs on CPU, but smaller capacity.
2. **Anthropic Claude API**: Set `USE_ANTHROPIC=True` and provide your API key
   in `config.py` for much higher quality battle reasoning.

## Key Design Decisions

- **PokeAPI as primary source**: Structured JSON endpoint gives reliable
  base data; Bulbapedia scraping is available as a secondary enrichment path.
- **REBEL-large for RE**: Pre-trained on >200 relation types; we post-filter
  to our ontology relations.
- **rdflib for RDF**: Pure-Python, no external DB needed for PoC scope.
- **Lightweight battle engine**: Simplified damage formula (Gen 1 style) keeps
  the focus on LLM grounding, not a full game engine.

## Evaluation Metrics

The evaluator compares extracted KG triples against `data/gold_standard.json`:

- **Precision** — fraction of extracted triples that are correct
- **Recall** — fraction of gold-standard triples that were extracted
- **F1-score** — harmonic mean of Precision and Recall

Battle-specific metrics:
- **Hallucination rate** — illegal moves / non-existent abilities cited
- **Type-accuracy** — correct super-effective / not-very-effective calls
- **Game-state consistency** — HP / status tracking correctness

## License

MIT — see LICENSE for details.
