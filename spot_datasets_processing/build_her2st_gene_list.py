#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from spot_datasets_processing.common import (
        Her2stPaths,
        build_logger,
        compute_common_genes,
        default_dataset_root,
        ensure_directory,
        load_aligned_adata,
        read_slide_names,
        write_lines,
    )
except ImportError:
    from common import (
        Her2stPaths,
        build_logger,
        compute_common_genes,
        default_dataset_root,
        ensure_directory,
        load_aligned_adata,
        read_slide_names,
        write_lines,
    )


def compute_union_hvg(
    paths: Her2stPaths,
    slide_names: Sequence[str],
    common_genes: Sequence[str],
    top_hvgs: int,
    exclude_prefixes: Sequence[str],
    logger,
) -> list[str]:
    import scanpy as sc

    union_hvg: set[str] = set()
    excluded_prefixes = tuple(exclude_prefixes)

    for slide_name in slide_names:
        adata = load_aligned_adata(paths.st_dir, slide_name, common_genes)
        sc.pp.filter_cells(adata, min_genes=1)
        sc.pp.filter_genes(adata, min_cells=1)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=top_hvgs)

        hvg_genes = set(adata.var_names[adata.var["highly_variable"]])
        union_hvg.update(hvg_genes)
        logger.info("%s union_hvg_size=%d", slide_name, len(union_hvg))

    filtered = sorted(
        gene for gene in union_hvg if not gene.startswith(excluded_prefixes)
    )
    logger.info("Union HVG size after prefix filtering=%d", len(filtered))
    return filtered


def accumulate_gene_statistics(
    paths: Her2stPaths,
    slide_names: Sequence[str],
    common_genes: Sequence[str],
    target_genes: Sequence[str],
    logger,
) -> pd.DataFrame:
    gene_names = list(target_genes)
    sum_expr = np.zeros(len(gene_names), dtype=np.float64)
    sumsq_expr = np.zeros(len(gene_names), dtype=np.float64)
    total_spots = 0

    for slide_name in slide_names:
        adata = load_aligned_adata(paths.st_dir, slide_name, common_genes)
        matrix = adata[:, gene_names].X

        if sparse.issparse(matrix):
            matrix = matrix.astype(np.float64)
            sum_expr += np.asarray(matrix.sum(axis=0)).ravel()
            sumsq_expr += np.asarray(matrix.power(2).sum(axis=0)).ravel()
        else:
            matrix = np.asarray(matrix, dtype=np.float64)
            sum_expr += matrix.sum(axis=0)
            sumsq_expr += np.square(matrix).sum(axis=0)

        total_spots += adata.n_obs
        logger.info(
            "%s aggregated spots=%d cumulative_spots=%d",
            slide_name,
            adata.n_obs,
            total_spots,
        )

    if total_spots == 0:
        raise ValueError("No spots found while aggregating gene statistics.")

    mean_expr = sum_expr / total_spots
    if total_spots == 1:
        std_expr = np.zeros_like(mean_expr)
    else:
        variance = (sumsq_expr - (sum_expr ** 2) / total_spots) / (total_spots - 1)
        variance = np.maximum(variance, 0.0)
        std_expr = np.sqrt(variance)

    return pd.DataFrame(
        {
            "gene": gene_names,
            "mean": mean_expr,
            "std": std_expr,
        }
    )


def build_gene_ranking(
    stats_df: pd.DataFrame,
    target_gene_count: int,
) -> tuple[list[str], pd.DataFrame]:
    mean_order = stats_df.sort_values(
        by=["mean", "gene"],
        ascending=[False, True],
    )["gene"].tolist()
    std_order = stats_df.sort_values(
        by=["std", "gene"],
        ascending=[False, True],
    )["gene"].tolist()

    rank_mean = {gene: index for index, gene in enumerate(mean_order)}
    rank_std = {gene: index for index, gene in enumerate(std_order)}

    ranking_df = stats_df.copy()
    ranking_df["mean_rank"] = ranking_df["gene"].map(rank_mean)
    ranking_df["std_rank"] = ranking_df["gene"].map(rank_std)
    ranking_df["combined_score"] = ranking_df["mean_rank"] + ranking_df["std_rank"]
    ranking_df = ranking_df.sort_values(
        by=["combined_score", "gene"],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected_count = min(target_gene_count, len(ranking_df))
    selected_genes = sorted(ranking_df.head(selected_count)["gene"].tolist())
    ranking_df["selected"] = ranking_df["gene"].isin(selected_genes)
    return selected_genes, ranking_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the HER2ST selected gene list."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help="HER2ST dataset root containing st/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory used to store the selected gene list and ranking table.",
    )
    parser.add_argument(
        "--slides",
        nargs="*",
        default=None,
        help="Optional slide names, for example: SPA119 SPA120.",
    )
    parser.add_argument(
        "--slides-file",
        type=Path,
        default=None,
        help="Optional text file containing one slide name per line.",
    )
    parser.add_argument(
        "--top-hvgs",
        type=int,
        default=2000,
        help="Number of highly variable genes computed per slide.",
    )
    parser.add_argument(
        "--target-genes",
        type=int,
        default=200,
        help="Number of genes retained in the final selected list.",
    )
    parser.add_argument(
        "--exclude-prefixes",
        nargs="*",
        default=["MT", "mt", "RPS", "RPL"],
        help="Gene prefixes removed from the union HVG set.",
    )
    parser.add_argument(
        "--gene-list-filename",
        type=str,
        default=None,
        help=(
            "Output filename for the selected gene list. When omitted, the canonical "
            "name genes_selected_top<N>.txt is used."
        ),
    )
    parser.add_argument(
        "--slides-filename",
        type=str,
        default="samples_her2st.txt",
        help="Output filename for the HER2ST sample identifier list.",
    )
    parser.add_argument(
        "--ranking-filename",
        type=str,
        default=None,
        help=(
            "Output filename for the gene selection statistics. When omitted, the "
            "canonical name genes_selection_stats_top<N>.csv is used."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = build_logger(args.verbose)

    paths = Her2stPaths.from_dataset_root(args.dataset_root)
    output_root = ensure_directory(args.output_root or (paths.dataset_root / "derived_features"))
    slide_names = read_slide_names(args.slides, args.slides_file)
    common_genes = compute_common_genes(paths.st_dir, slide_names, logger)
    if not common_genes:
        raise ValueError("No common genes were found across the requested slides.")
    logger.info("Aligned all slides to %d common genes", len(common_genes))

    union_hvg = compute_union_hvg(
        paths=paths,
        slide_names=slide_names,
        common_genes=common_genes,
        top_hvgs=args.top_hvgs,
        exclude_prefixes=args.exclude_prefixes,
        logger=logger,
    )
    if not union_hvg:
        raise ValueError("Union HVG set is empty after filtering.")

    stats_df = accumulate_gene_statistics(
        paths=paths,
        slide_names=slide_names,
        common_genes=common_genes,
        target_genes=union_hvg,
        logger=logger,
    )
    selected_genes, ranking_df = build_gene_ranking(
        stats_df=stats_df,
        target_gene_count=args.target_genes,
    )

    selected_count = len(selected_genes)
    gene_list_filename = args.gene_list_filename or f"genes_selected_top{selected_count}.txt"
    ranking_filename = args.ranking_filename or f"genes_selection_stats_top{selected_count}.csv"

    write_lines(output_root / gene_list_filename, selected_genes)
    write_lines(output_root / args.slides_filename, slide_names)
    ranking_df.to_csv(output_root / ranking_filename, index=False)

    logger.info("Selected gene count=%d", len(selected_genes))
    logger.info("Saved gene list to %s", output_root / gene_list_filename)
    logger.info("Saved slide list to %s", output_root / args.slides_filename)
    logger.info("Saved ranking table to %s", output_root / ranking_filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
