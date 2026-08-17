#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_GRAPH = Path(__file__).resolve().parent / "RegNetwork" / "graph_human_core.graphml"


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_selected_genes(path: str | Path) -> list[str]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Selected gene list not found: {input_path}")

    gene_frame = pd.read_csv(input_path, header=None)
    if gene_frame.empty:
        raise ValueError(f"Selected gene list is empty: {input_path}")

    genes = [str(value).strip() for value in gene_frame.iloc[:, 0].tolist() if str(value).strip()]
    if genes and genes[0].lower() == "gene":
        genes = genes[1:]
    if not genes:
        raise ValueError(f"No valid genes were found in {input_path}")
    return genes


def to_igraph(graph_nx: nx.DiGraph, weight_attribute: str):
    from igraph import Graph

    graph_ig = Graph(directed=True)
    node_names = list(graph_nx.nodes)
    graph_ig.add_vertices(node_names)

    edges: list[tuple[str, str]] = []
    weights: list[float] = []
    for source, target, attributes in graph_nx.edges(data=True):
        edges.append((source, target))
        weights.append(float(attributes.get(weight_attribute, 1.0)))

    if edges:
        graph_ig.add_edges(edges)
        graph_ig.es[weight_attribute] = weights
    return graph_ig


def topological_gene_order(
    graph_path: str | Path,
    selected_genes: list[str],
    weight_attribute: str,
) -> tuple[list[str], int]:
    full_graph = nx.read_graphml(Path(graph_path).expanduser().resolve())
    selected_set = set(selected_genes)
    filtered_graph = full_graph.subgraph(node for node in full_graph.nodes if node in selected_set).copy()

    graph_ig = to_igraph(filtered_graph, weight_attribute)
    removed_edges = 0
    if len(graph_ig.es) > 0:
        feedback_arc_set = graph_ig.feedback_arc_set(weights=graph_ig.es[weight_attribute])
        removed_edges = len(feedback_arc_set)
        if removed_edges > 0:
            graph_ig.delete_edges(feedback_arc_set)

    order_indices = graph_ig.topological_sorting(mode="OUT")
    ordered_genes_in_graph = [graph_ig.vs[index]["name"] for index in order_indices]
    ordered_gene_set = set(ordered_genes_in_graph)
    tail_genes = [gene for gene in selected_genes if gene not in ordered_gene_set]
    return ordered_genes_in_graph + tail_genes, removed_edges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topologically order a selected gene list with a GRN graph.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help="GraphML file containing the regulatory network.",
    )
    parser.add_argument(
        "--selected-genes",
        type=Path,
        required=True,
        help="CSV/TXT file containing the selected genes to be ordered.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file for the ordered genes.",
    )
    parser.add_argument(
        "--weight-attribute",
        type=str,
        default="confidence",
        help="Edge attribute used when breaking cycles.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)

    selected_genes = load_selected_genes(args.selected_genes)
    ordered_genes, removed_edges = topological_gene_order(
        graph_path=args.graph,
        selected_genes=selected_genes,
        weight_attribute=args.weight_attribute,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": ordered_genes}).to_csv(output_path, index=False, header=False)

    print(f"Input selected genes: {len(selected_genes)}")
    print(f"Output ordered genes: {len(ordered_genes)}")
    print(f"Removed feedback edges: {removed_edges}")
    print(f"First 10 genes: {ordered_genes[:10]}")
    print(f"Saved ordered genes to: {output_path}")


if __name__ == "__main__":
    main()
