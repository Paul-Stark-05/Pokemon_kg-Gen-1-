# Pokémon Knowledge Graph — LLM Grounding PoC

> **Can a Knowledge Graph reduce hallucinations in a text-based Pokémon battle?**
> This proof-of-concept builds an RDF Knowledge Graph from Gen 1 Pokémon data,
> then pits a KG-grounded LLM agent against a vanilla LLM agent to find out.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE OVERVIEW                            │
│                                                                     │
│  PokeAPI / Bulbapedia ──► NLP Pipeline ──► RDF Knowledge Graph      │
│        (raw data)        (nlp_pipeline)      (kg_builder)           │
│                                                    │                │
│                                        ┌───────────┴──────────┐    │
│                                        ▼                      ▼    │
│                                   SPARQL queries        Gold Std   │
│                                        │                 compare   │
│                                        ▼                      ▼    │
│                               Battle Simulator         Evaluator   │
│                              ┌────────┴────────┐     (P / R / F1)  │
│                              ▼                 ▼                    │
│                        Agent A             Agent B                  │
│                       (Vanilla)          (KG-Grounded)              │
│                              └────────┬────────┘                   │
│                                       ▼                            │
│                              Battle Log + Report                   │
│                         (hallucination tracking)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Layout

```
pokemon-kg/
├── README.md
├── requirements.txt
├── config.py                  # Central configuration & constants
├── src/
│   ├── __init__.py
│   ├── nlp_pipeline.py        # Data extraction + NLP (NER, RE)
│   ├── kg_builder.py          # RDF graph construction + ontology
│   ├── evaluator.py           # Precision / Recall / F1 framework
│   ├── battle_simulator.py    # LLM arena (vanilla vs grounded)
│   └── visualize_kg.py        # Graph visualisation for README images
├── data/
│   └── gold_standard.json     # Ground-truth triples for evaluation
├── output/                    # Generated .ttl, logs, images
└── tests/
    └── test_pipeline.py
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
| `LLM_MODEL_NAME` | `google/flan-t5-base` | HuggingFace model for battle agents |
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
