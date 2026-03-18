"""
visualize_kg.py — Knowledge Graph Visualisation.

Produces:
  1. A static PNG network diagram (for GitHub README embedding).
  2. An interactive HTML visualisation (using pyvis).

Uses networkx for layout and matplotlib for rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from rdflib import RDF, RDFS, OWL

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Colour palette for node types ───────────────────────────────────────────

NODE_COLORS = {
    "Pokemon":       "#FF6B6B",   # coral red
    "Move":          "#4ECDC4",   # teal
    "Type":          "#FFE66D",   # yellow
    "StatusEffect":  "#A855F7",   # purple
    "Unknown":       "#C0C0C0",   # grey
}

EDGE_COLORS = {
    "hasType":                  "#FF9F43",
    "learnsMove":               "#54A0FF",
    "hasMoveType":              "#5F27CD",
    "superEffectiveAgainst":    "#EE5A24",
    "notVeryEffectiveAgainst":  "#78E08F",
    "immuneAgainst":            "#6D6875",
    "basePower":                "#B0B0B0",
}


def _extract_label(uri: str) -> str:
    """Extract the readable label from a URI."""
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.split("/")[-1]
    return str(uri)


def build_networkx_graph(
    kg,                    # PokemonKnowledgeGraph
    max_nodes: int = 80,
    pokemon_limit: int = 6,
) -> nx.DiGraph:
    """
    Convert a subset of the RDF graph into a NetworkX DiGraph for visualisation.

    Args:
        kg: The PokemonKnowledgeGraph instance.
        max_nodes: Hard cap on nodes to keep the diagram readable.
        pokemon_limit: Max Pokémon to include (their moves and types follow).
    """
    from src.kg_builder import POKE, DATA

    G = nx.DiGraph()

    # Collect a representative subset of Pokémon
    pokemon_query = f"""
    SELECT DISTINCT ?p WHERE {{
        ?p a poke:Pokemon .
    }} LIMIT {pokemon_limit}
    """
    pokemon_uris = set()
    for row in kg.g.query(pokemon_query, initNs={"poke": POKE}):
        pokemon_uris.add(str(row[0]))

    # For each Pokémon, add its immediate neighbourhood
    nodes_added = set()

    for poke_uri in pokemon_uris:
        poke_label = _extract_label(poke_uri)
        if len(nodes_added) >= max_nodes:
            break

        G.add_node(poke_label, node_type="Pokemon")
        nodes_added.add(poke_label)

        # Outgoing edges from this Pokémon
        for _, p, o in kg.g.triples((None, None, None)):
            s_str = str(_)
            if s_str != poke_uri:
                continue
            p_label = _extract_label(str(p))
            o_label = _extract_label(str(o))

            # Skip RDF/OWL meta-triples
            if p_label in ("type", "label") or "www.w3.org" in str(p):
                continue

            if len(nodes_added) >= max_nodes:
                break

            # Determine object node type
            obj_type = "Unknown"
            if (_, RDF.type, POKE.Move) in kg.g or "Move" in str(o):
                obj_type = "Move"
            elif (_, RDF.type, POKE.Type) in kg.g or o_label in (
                "fire", "water", "grass", "electric", "normal", "ice",
                "fighting", "poison", "ground", "flying", "psychic",
                "bug", "rock", "ghost", "dragon",
            ):
                obj_type = "Type"

            # Check if object is a literal (datatype property)
            from rdflib import Literal as RDFLiteral
            if isinstance(o, RDFLiteral):
                continue  # Skip datatype properties for cleaner viz

            G.add_node(o_label, node_type=obj_type)
            nodes_added.add(o_label)
            edge_color = EDGE_COLORS.get(p_label, "#888888")
            G.add_edge(poke_label, o_label, label=p_label, color=edge_color)

    # Add type effectiveness edges (subset)
    type_query = """
    SELECT ?atk ?rel ?def WHERE {
        ?atk poke:superEffectiveAgainst ?def .
        BIND("superEffectiveAgainst" AS ?rel)
    } LIMIT 15
    """
    for row in kg.g.query(type_query, initNs={"poke": POKE}):
        atk = _extract_label(str(row[0]))
        rel = str(row[1])
        def_ = _extract_label(str(row[2]))
        G.add_node(atk, node_type="Type")
        G.add_node(def_, node_type="Type")
        G.add_edge(atk, def_, label="superEffective", color=EDGE_COLORS.get("superEffectiveAgainst", "#EE5A24"))

    return G


def render_static_png(
    G: nx.DiGraph,
    output_path: Optional[Path] = None,
    figsize: tuple = (20, 14),
) -> Path:
    """
    Render the NetworkX graph as a static PNG image.
    """
    import config
    if output_path is None:
        output_path = config.KG_VIS_OUTPUT

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Draw edges
    edge_colors = [G[u][v].get("color", "#555555") for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        alpha=0.6,
        width=1.2,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw edge labels
    edge_labels = {(u, v): G[u][v].get("label", "") for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=6, font_color="#aaaaaa", ax=ax,
        label_pos=0.5, alpha=0.7,
    )

    # Draw nodes by type
    for node_type, color in NODE_COLORS.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, ax=ax,
                node_color=color, node_size=800,
                edgecolors="#ffffff", linewidths=1.5,
                alpha=0.9,
            )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7, font_color="#ffffff",
        font_weight="bold",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=t)
        for t, c in NODE_COLORS.items()
        if t != "Unknown"
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        facecolor="#16213e", edgecolor="#555555",
        labelcolor="#ffffff", fontsize=10,
        title="Node Types", title_fontsize=11,
    )

    ax.set_title(
        "Pokémon Knowledge Graph (Gen 1 Subset)",
        fontsize=16, fontweight="bold", color="#ffffff", pad=20,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)

    log.info(f"Static visualisation saved → {output_path}")
    return output_path


def render_interactive_html(
    G: nx.DiGraph,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Render an interactive HTML visualisation using pyvis.
    """
    import config
    if output_path is None:
        output_path = config.KG_VIS_HTML

    try:
        from pyvis.network import Network
    except ImportError:
        log.warning("pyvis not installed — skipping interactive visualisation")
        return output_path

    net = Network(
        height="800px", width="100%",
        bgcolor="#1a1a2e", font_color="#ffffff",
        directed=True,
    )
    net.barnes_hut(gravity=-5000, spring_length=200)

    # Add nodes
    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "Unknown")
        color = NODE_COLORS.get(node_type, "#C0C0C0")
        net.add_node(
            node, label=node, color=color,
            title=f"{node} ({node_type})",
            size=20 if node_type == "Pokemon" else 15,
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        color = data.get("color", "#888888")
        net.add_edge(u, v, label=label, color=color, title=label)

    net.save_graph(str(output_path))
    log.info(f"Interactive visualisation saved → {output_path}")
    return output_path


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config
    from src.kg_builder import build_from_triples_file

    triples_path = config.OUTPUT_DIR / "extracted_triples.json"
    if not triples_path.exists():
        log.error(f"No triples file found at {triples_path}. Run nlp_pipeline.py first.")
        raise SystemExit(1)

    log.info("Loading Knowledge Graph…")
    kg = build_from_triples_file(triples_path)

    log.info("Building visualisation graph…")
    G = build_networkx_graph(kg, max_nodes=80, pokemon_limit=6)
    log.info(f"Vis graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    render_static_png(G)
    render_interactive_html(G)

    print(f"\n✓ Visualisations generated!")
    print(f"  Static PNG:    {config.KG_VIS_OUTPUT}")
    print(f"  Interactive:   {config.KG_VIS_HTML}")
