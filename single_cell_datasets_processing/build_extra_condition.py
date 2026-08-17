"""Build auxiliary single-cell condition features.

This script creates `extra_cond.npy` for each split under a fold directory by
concatenating:

1. One-hot cell-type labels.
2. kNN neighborhood cell-type composition features.

The output rows are aligned with `expression.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def build_type_mapping(cell_type_frame: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Return sorted cell-type categories and an index mapping."""

    categories = (
        cell_type_frame["cell_type"].dropna().astype(str).unique().tolist()
    )
    categories = sorted(categories)
    if "Unknown" not in categories:
        categories.append("Unknown")
    return categories, {cell_type: index for index, cell_type in enumerate(categories)}


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""

    encoded = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(indices.shape[0]), indices] = 1.0
    return encoded


def compute_knn_composition(
    coords: np.ndarray,
    labels_idx: np.ndarray,
    num_types: int,
    k_neighbors: int,
) -> np.ndarray:
    """Compute normalized neighborhood cell-type composition for each cell."""

    num_cells = coords.shape[0]
    neighborhood = np.zeros((num_cells, num_types), dtype=np.float32)

    valid_mask = ~np.isnan(coords).any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        print("[WARN] No valid coordinates found. Neighborhood features will be all zeros.")
        return neighborhood
    if len(valid_indices) < 2:
        print("[WARN] Fewer than two valid coordinates found. Neighborhood features will be all zeros.")
        return neighborhood

    effective_k = min(k_neighbors, len(valid_indices))
    if effective_k < 1:
        print("[WARN] Effective k is smaller than 1. Neighborhood features will be all zeros.")
        return neighborhood

    print(f"[INFO] Using k={effective_k} over {len(valid_indices)} cells with valid coordinates.")
    coords_valid = coords[valid_indices]
    neighbors = NearestNeighbors(n_neighbors=effective_k, algorithm="ball_tree").fit(coords_valid)
    _, indices = neighbors.kneighbors(coords_valid)

    for local_index, global_index in enumerate(valid_indices):
        neighbor_globals = valid_indices[indices[local_index]]
        neighbor_globals = neighbor_globals[neighbor_globals != global_index]
        if neighbor_globals.size == 0:
            continue

        neighbor_labels = labels_idx[neighbor_globals]
        counts = np.bincount(neighbor_labels, minlength=num_types).astype(np.float32)
        total = counts.sum()
        if total > 0:
            neighborhood[global_index] = counts / total

    return neighborhood


def process_split(
    split_dir: Path,
    cell_type_frame: pd.DataFrame,
    type_to_index: dict[str, int],
    num_types: int,
    k_neighbors: int,
) -> None:
    """Build `extra_cond.npy` for one split directory."""

    expression_path = split_dir / "expression.csv"
    coords_path = split_dir / "coords.npy"

    print(f"\n[INFO] Processing split: {split_dir}")
    if not expression_path.exists():
        print(f"[WARN] Missing {expression_path}. Skipping.")
        return
    if not coords_path.exists():
        print(f"[WARN] Missing {coords_path}. Skipping.")
        return

    expression_frame = pd.read_csv(expression_path, index_col=0)
    cell_ids = expression_frame.index.astype(str).tolist()
    coords = np.load(coords_path)
    if coords.shape[0] != len(cell_ids):
        raise ValueError(
            f"Coordinate rows ({coords.shape[0]}) do not match expression rows ({len(cell_ids)})."
        )

    cell_type_series = cell_type_frame.reindex(cell_ids)["cell_type"]
    cell_type_series = cell_type_series.fillna("Unknown").astype(str)
    missing = int((cell_type_series == "Unknown").sum())
    if missing > 0:
        print(f"[INFO] Assigned 'Unknown' to {missing} cells with missing annotations.")

    labels_idx = np.array(
        [type_to_index.get(cell_type, type_to_index["Unknown"]) for cell_type in cell_type_series.values],
        dtype=np.int64,
    )
    cell_type_onehot = one_hot(labels_idx, num_types)
    neighborhood = compute_knn_composition(coords, labels_idx, num_types, k_neighbors)

    extra_condition = np.concatenate([cell_type_onehot, neighborhood], axis=1).astype(np.float32)
    output_path = split_dir / "extra_cond.npy"
    np.save(output_path, extra_condition)
    print(f"[INFO] Saved extra condition tensor with shape {extra_condition.shape} to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build extra single-cell condition features from cell types and spatial neighborhoods."
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        required=True,
        help="Fold directory containing `train/` and `val/` split folders.",
    )
    parser.add_argument(
        "--cell-type-csv",
        type=Path,
        required=True,
        help="CSV file with at least `cell_id` and `cell_type` columns.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=37,
        help="Number of nearest neighbors used when computing neighborhood composition.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cell_type_frame = pd.read_csv(args.cell_type_csv)
    if "cell_id" not in cell_type_frame.columns or "cell_type" not in cell_type_frame.columns:
        raise ValueError("The cell-type CSV must contain `cell_id` and `cell_type` columns.")
    cell_type_frame["cell_id"] = cell_type_frame["cell_id"].astype(str)
    cell_type_frame = cell_type_frame.set_index("cell_id")

    categories, type_to_index = build_type_mapping(cell_type_frame)
    print(f"[INFO] Loaded {len(categories)} cell types (including 'Unknown').")
    print(categories)

    for split_name in ("train", "val"):
        process_split(
            split_dir=args.fold_dir / split_name,
            cell_type_frame=cell_type_frame,
            type_to_index=type_to_index,
            num_types=len(categories),
            k_neighbors=args.k_neighbors,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
