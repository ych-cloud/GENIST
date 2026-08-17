"""Build a Xenium cell-by-gene expression matrix from a feature-matrix export."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
from pathlib import Path

import natsort
import pandas as pd
from scipy.io import mmread


def ungzip_if_needed(path: str | Path) -> tuple[Path, bool]:
    """Extract a `.gz` file next to itself when needed.

    Returns the usable file path and a flag indicating whether this call created
    a temporary decompressed file.
    """

    path = Path(path).expanduser().resolve()
    if path.suffix != ".gz":
        return path, False

    output_path = path.with_suffix("")
    if not output_path.exists():
        with gzip.open(path, "rb") as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        return output_path, True
    return output_path, False


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a TSV file without headers."""

    file_path = Path(path).expanduser().resolve()
    print(f"[INFO] Loading {file_path}")
    frame = pd.read_csv(file_path, sep="\t", index_col=False, header=None)
    frame.reset_index(drop=True, inplace=True)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a filtered Xenium cell-by-gene matrix from `barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.mtx.gz`."
    )
    parser.add_argument(
        "--dir-feature-matrix",
        type=Path,
        required=True,
        help="Directory containing the Xenium cell feature-matrix files.",
    )
    parser.add_argument(
        "--fp-barcodes",
        type=str,
        default="barcodes.tsv.gz",
        help="Barcode filename inside the feature-matrix directory.",
    )
    parser.add_argument(
        "--fp-features",
        type=str,
        default="features.tsv.gz",
        help="Feature filename inside the feature-matrix directory.",
    )
    parser.add_argument(
        "--fp-matrix",
        type=str,
        default="matrix.mtx.gz",
        help="Sparse matrix filename inside the feature-matrix directory.",
    )
    parser.add_argument(
        "--fp-genes",
        type=str,
        default="genes.txt",
        help="Output filename for the filtered and sorted gene list.",
    )
    parser.add_argument(
        "--dir-output",
        type=Path,
        default=Path("processed"),
        help="Directory used to save the expression matrix and gene list.",
    )
    parser.add_argument(
        "--fp-out-matrix",
        type=str,
        default="cell_gene_matrix.csv",
        help="Output filename for the cell-by-gene expression matrix.",
    )
    parser.add_argument(
        "--delete-intermediate-files",
        action="store_true",
        help="Delete the decompressed temporary files after export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.dir_output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix_dir = args.dir_feature_matrix.expanduser().resolve()
    if feature_matrix_dir.suffix:
        if str(feature_matrix_dir).endswith(".tar.gz"):
            raise SystemExit("Please extract `cell_feature_matrix.tar.gz` before running this script.")
        raise SystemExit("`--dir-feature-matrix` must point to a directory, not a file.")

    barcodes_path, delete_barcodes = ungzip_if_needed(feature_matrix_dir / args.fp_barcodes)
    features_path, delete_features = ungzip_if_needed(feature_matrix_dir / args.fp_features)
    matrix_path, delete_matrix = ungzip_if_needed(feature_matrix_dir / args.fp_matrix)

    barcodes = read_tsv(barcodes_path)
    features = read_tsv(features_path)
    matrix = mmread(matrix_path).toarray()

    map_cell_ids = all(isinstance(item, str) for item in barcodes[0].unique().tolist())
    cell_id_mapping = None
    if map_cell_ids:
        mapping_path = output_dir / "xenium_cell_ids_dict.csv"
        cell_id_mapping = pd.read_csv(mapping_path, index_col=0)

    feature_names = features.loc[features.iloc[:, 2] == "Gene Expression", 1].tolist()
    transcripts_to_filter = [
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword_",
        "BLANK_",
        "Blank-",
        "NegPrb",
        "Unassigned",
    ]
    valid_genes = [
        gene_name
        for gene_name in feature_names
        if not any(substr in gene_name for substr in transcripts_to_filter)
    ]
    valid_genes = natsort.natsorted(valid_genes)

    gene_list_path = output_dir / args.fp_genes
    with gene_list_path.open("w", encoding="utf-8", newline="\n") as handle:
        for gene_name in valid_genes:
            handle.write(f"{gene_name}\n")

    feature_order = [feature_names.index(gene_name) for gene_name in valid_genes]
    matrix_filtered = matrix[feature_order, :].T
    expression_frame = pd.DataFrame(matrix_filtered, index=barcodes[0], columns=valid_genes)

    if map_cell_ids:
        cell_id_dict = cell_id_mapping["cell_id_num"].to_dict()
        expression_frame = expression_frame[expression_frame.index.isin(cell_id_dict.keys())]
        expression_frame.index = expression_frame.index.map(cell_id_dict)

    output_matrix_path = output_dir / args.fp_out_matrix
    expression_frame.to_csv(output_matrix_path)
    print(f"[INFO] Saved expression matrix to {output_matrix_path}")
    print(f"[INFO] Saved gene list to {gene_list_path}")

    if args.delete_intermediate_files:
        for temp_path, should_delete in (
            (barcodes_path, delete_barcodes),
            (features_path, delete_features),
            (matrix_path, delete_matrix),
        ):
            if should_delete and temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    main()
