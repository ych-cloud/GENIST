"""Match H&E nuclei instances to Xenium nuclei instances.

The script computes overlap between two segmentation masks with the same shape:

- `seg_hist`: nuclei segmented from the H&E image.
- `seg_xenium`: nuclei segmented from the Xenium / DAPI image.

For each H&E nucleus, the Xenium nucleus with the largest overlap is selected,
then filtered by a minimum overlap fraction. The filtered mapping is applied to
the Xenium cell-by-gene matrix to produce a histology-aligned expression table.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tifffile
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    tifffile = None


seg_hist = None
seg_xenium = None


def require_runtime_dependencies() -> None:
    if tifffile is None:
        raise ModuleNotFoundError("Missing runtime dependency: tifffile")


def init_worker(seg_hist_arr: np.ndarray, seg_xenium_arr: np.ndarray) -> None:
    global seg_hist, seg_xenium
    seg_hist = seg_hist_arr
    seg_xenium = seg_xenium_arr


def process_nucleus(nucleus_id: int) -> tuple[list[int], list[int], list[int], list[int]]:
    nucleus_coords = np.where(seg_hist == nucleus_id)
    overlapping_cells = seg_xenium[nucleus_coords[0], nucleus_coords[1]]
    unique_cells, counts = np.unique(overlapping_cells, return_counts=True)
    nonzero_indices = np.nonzero(unique_cells)[0]
    if len(nonzero_indices) == 0:
        return [], [], [], []

    matched_ids = list(unique_cells[nonzero_indices])
    matched_counts = counts[unique_cells != 0]
    hist_sizes = [len(nucleus_coords[0])] * len(nonzero_indices)
    hist_ids = [nucleus_id] * len(nonzero_indices)
    return hist_ids, matched_ids, matched_counts.tolist(), hist_sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match H&E nuclei to Xenium nuclei and filter the expression matrix accordingly."
    )
    parser.add_argument(
        "--fp-seg-hist",
        type=str,
        default="he_image_nuclei_seg_microns.tif",
        help="H&E nuclei segmentation file.",
    )
    parser.add_argument(
        "--fp-seg-xenium",
        type=str,
        default="xenium_nuclei.tif",
        help="Xenium nuclei segmentation file relative to `--dir-output` unless absolute.",
    )
    parser.add_argument(
        "--fp-cgm",
        type=str,
        default="cell_gene_matrix.csv",
        help="Cell-by-gene matrix relative to `--dir-output` unless absolute.",
    )
    parser.add_argument(
        "--fp-out-matched-nuclei",
        type=str,
        default="matched_nuclei.csv",
        help="Output CSV filename for raw and filtered nucleus matches.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=28,
        help="Maximum number of CPU processes to use.",
    )
    parser.add_argument(
        "--min-overlap-fraction",
        type=float,
        default=0.5,
        help="Minimum overlap ratio required to keep a nucleus match.",
    )
    parser.add_argument(
        "--dir-output",
        type=Path,
        default=Path("processed"),
        help="Directory containing the segmentation masks and expression matrix.",
    )
    return parser


def resolve_in_output_dir(root_dir: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root_dir / candidate).resolve()


def main() -> None:
    args = build_parser().parse_args()
    require_runtime_dependencies()
    output_dir = args.dir_output.expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    seg_xenium_path = resolve_in_output_dir(output_dir, args.fp_seg_xenium)
    seg_hist_path = resolve_in_output_dir(output_dir, args.fp_seg_hist)
    expression_path = resolve_in_output_dir(output_dir, args.fp_cgm)

    seg_xenium_local = tifffile.imread(seg_xenium_path)
    seg_hist_local = tifffile.imread(seg_hist_path)
    if seg_xenium_local.shape != seg_hist_local.shape:
        raise ValueError(
            "The H&E and Xenium segmentations must have the same shape. "
            "Use the micron-resolution outputs for both inputs."
        )

    nucleus_ids = np.unique(seg_hist_local)
    num_cpus = min(max(cpu_count() - 2, 1), args.n_processes)
    print(f"[INFO] Processing nuclei with {num_cpus} CPU workers.")

    with Pool(
        processes=num_cpus,
        initializer=init_worker,
        initargs=(seg_hist_local, seg_xenium_local),
    ) as pool:
        results = list(tqdm(pool.imap(process_nucleus, nucleus_ids), total=len(nucleus_ids)))

    hist_ids: list[int] = []
    xenium_ids: list[int] = []
    overlaps: list[int] = []
    hist_sizes: list[int] = []
    for orig_ids, match_ids, overlap_counts, orig_sizes in results:
        hist_ids.extend(orig_ids)
        xenium_ids.extend(match_ids)
        overlaps.extend(overlap_counts)
        hist_sizes.extend(orig_sizes)

    combined = np.vstack((hist_ids, xenium_ids, overlaps, hist_sizes)).T
    matches = pd.DataFrame(
        combined,
        columns=["id_histology", "id_xenium", "overlap", "size_pix_histology"],
    )
    max_overlap_indices = matches.groupby("id_histology")["overlap"].idxmax()
    matches = matches.loc[max_overlap_indices].copy()
    matches = matches[matches["id_histology"] != 0]

    raw_match_path = resolve_in_output_dir(output_dir, args.fp_out_matched_nuclei)
    matches.to_csv(raw_match_path, index=False)

    filtered_matches = matches[
        matches["overlap"] >= args.min_overlap_fraction * matches["size_pix_histology"]
    ].copy()
    filtered_matches.sort_values(by=["id_xenium", "overlap"], ascending=[True, False], inplace=True)
    filtered_matches.drop_duplicates(subset="id_xenium", keep="first", inplace=True)
    filtered_matches.sort_values(by="id_histology", inplace=True)

    filtered_match_path = raw_match_path.with_name(f"{raw_match_path.stem}_filtered.csv")
    filtered_matches.to_csv(filtered_match_path, index=False)

    mapping = dict(zip(filtered_matches["id_xenium"], filtered_matches["id_histology"]))
    expression_frame = pd.read_csv(expression_path, index_col=0)
    if "cell_id" not in expression_frame.columns:
        expression_frame["cell_id"] = expression_frame.index

    expression_frame["cell_id"] = expression_frame["cell_id"].map(mapping)
    expression_frame.set_index("cell_id", inplace=True, drop=False)
    expression_frame.index.name = None
    expression_frame.dropna(inplace=True)
    expression_frame.drop("cell_id", axis=1, inplace=True)
    expression_frame.index = expression_frame.index.astype(int)

    filtered_expression_path = expression_path.with_name(f"{expression_path.stem}_filtered.csv")
    expression_frame.to_csv(filtered_expression_path)

    print(f"[INFO] Saved raw nucleus matches to {raw_match_path}")
    print(f"[INFO] Saved filtered nucleus matches to {filtered_match_path}")
    print(f"[INFO] Saved filtered expression matrix to {filtered_expression_path}")


if __name__ == "__main__":
    main()
