from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from .common import DEFAULT_TOPK_VALUES, build_logger, ensure_directory, normalize_mode, save_arguments, write_json
from .data import load_ground_truth


SPOT_ID_PATTERN = re.compile(r"(\d+)x(\d+)")


def safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    if x.std() <= eps or y.std() <= eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def concordance_correlation(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute Lin's concordance correlation coefficient independently per gene."""
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if ground_truth.ndim != 2 or prediction.shape != ground_truth.shape:
        raise ValueError(
            "CCC inputs must have matching shape [N, G]; "
            f"received {ground_truth.shape} and {prediction.shape}."
        )

    mean_ground_truth = ground_truth.mean(axis=0)
    mean_prediction = prediction.mean(axis=0)
    centered_ground_truth = ground_truth - mean_ground_truth
    centered_prediction = prediction - mean_prediction
    variance_ground_truth = np.mean(centered_ground_truth * centered_ground_truth, axis=0)
    variance_prediction = np.mean(centered_prediction * centered_prediction, axis=0)
    covariance = np.mean(centered_ground_truth * centered_prediction, axis=0)
    denominator = (
        variance_ground_truth
        + variance_prediction
        + (mean_ground_truth - mean_prediction) ** 2
    )
    return np.divide(
        2 * covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0,
    )


def safe_ssim_2d(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute 2D SSIM for a display-only spatial comparison."""
    image_a = np.asarray(image_a, dtype=np.float64)
    image_b = np.asarray(image_b, dtype=np.float64)
    if image_a.ndim != 2 or image_b.ndim != 2 or image_a.shape != image_b.shape:
        return float("nan")

    finite = np.concatenate([image_a.ravel(), image_b.ravel()])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    if data_max <= data_min + 1e-12:
        return 1.0

    min_side = min(image_a.shape)
    if min_side < 3:
        return float("nan")

    try:
        kwargs = {
            "data_range": data_max - data_min,
            "gaussian_weights": True,
            "sigma": 1.5,
            "use_sample_covariance": False,
            "channel_axis": None,
        }
        if min_side < 11:
            win_size = min_side if min_side % 2 == 1 else min_side - 1
            if win_size < 3:
                return float("nan")
            kwargs["win_size"] = win_size
        return float(structural_similarity(image_a, image_b, **kwargs))
    except Exception:
        return float("nan")


def build_spot_grid(entity_ids: list[str], values: np.ndarray, fill_value: float) -> np.ndarray:
    coordinates: list[tuple[int, int]] = []
    invalid_ids: list[str] = []
    for entity_id in entity_ids:
        match = SPOT_ID_PATTERN.search(str(entity_id))
        if match is None:
            invalid_ids.append(str(entity_id))
            continue
        x_coord, y_coord = match.groups()
        coordinates.append((int(x_coord), int(y_coord)))

    if invalid_ids:
        raise ValueError(
            "Spatial SSIM requires spot IDs containing '<x>x<y>' coordinates. "
            f"Unrecognized examples: {invalid_ids[:10]}"
        )

    x_values = [coord[0] for coord in coordinates]
    y_values = [coord[1] for coord in coordinates]
    x_min, y_min = min(x_values), min(y_values)
    grid = np.full(
        (max(y_values) - y_min + 1, max(x_values) - x_min + 1),
        fill_value,
        dtype=np.float64,
    )
    for (x_coord, y_coord), value in zip(coordinates, values):
        grid[y_coord - y_min, x_coord - x_min] = value
    return grid


def compute_spot_gene_ssim(
    entity_ids: list[str],
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Compute per-gene SSIM for spot grids for visualisation only.

    This helper is intentionally not called by the main evaluation entry point.
    SSIM is not part of the reported spot PCC/RMSE or single-cell CCC metrics.
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if ground_truth.ndim != 2 or prediction.shape != ground_truth.shape:
        raise ValueError(
            "Spot SSIM inputs must have matching shape [N, G]; "
            f"received {ground_truth.shape} and {prediction.shape}."
        )

    _, num_genes = ground_truth.shape
    gene_ssim = np.full(num_genes, np.nan, dtype=np.float64)
    for gene_index in range(num_genes):
        gt_values = ground_truth[:, gene_index]
        pred_values = prediction[:, gene_index]
        finite_values = np.concatenate([gt_values, pred_values])
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            continue
        fill_value = float(np.min(finite_values))
        gt_grid = build_spot_grid(entity_ids, gt_values, fill_value)
        pred_grid = build_spot_grid(entity_ids, pred_values, fill_value)
        gene_ssim[gene_index] = safe_ssim_2d(gt_grid, pred_grid)
    return gene_ssim


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def topk_mean_desc(values: np.ndarray, top_k: int) -> float:
    filtered = np.asarray(values, dtype=np.float64)
    filtered = filtered[np.isfinite(filtered)]
    if filtered.size == 0:
        return float("nan")
    filtered.sort()
    filtered = filtered[::-1]
    return float(filtered[: min(top_k, filtered.size)].mean())


def load_prediction_tensor(sample_file: str | Path) -> torch.Tensor:
    prediction = torch.load(Path(sample_file).expanduser().resolve(), map_location="cpu")
    if prediction.ndim == 3 and prediction.shape[1] == 1:
        prediction = prediction.squeeze(1)
    if prediction.ndim != 2:
        raise ValueError(
            "Prediction tensor must have shape [N, G] or [N, 1, G]. "
            f"Received shape: {tuple(prediction.shape)}"
        )
    return prediction.float()


def infer_repetitions(num_predictions: int, num_entities: int, explicit_repetitions: int | None) -> int:
    if explicit_repetitions is not None and explicit_repetitions > 0:
        if num_predictions != num_entities * explicit_repetitions:
            raise ValueError(
                "Prediction rows do not match the requested samples-per-condition: "
                f"{num_predictions} vs {num_entities} * {explicit_repetitions}"
            )
        return explicit_repetitions

    if num_predictions % num_entities != 0:
        raise ValueError(
            f"Prediction rows ({num_predictions}) are not divisible by the number of entities ({num_entities})."
        )
    return num_predictions // num_entities


def default_output_dir(sample_file: str | Path) -> Path:
    sample_path = Path(sample_file).expanduser().resolve()
    if sample_path.parent.name == "predictions":
        evaluation_root = sample_path.parent.parent / "evaluations"
    else:
        evaluation_root = sample_path.parent / "evaluations"
    return ensure_directory(evaluation_root / sample_path.stem)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GENIST sampling outputs against ground-truth expression.")
    parser.add_argument("--mode", type=str, required=True, help="Evaluation mode: spot or single_cell.")
    parser.add_argument("--prediction-file", "--sample-file", dest="sample_file", type=Path, required=True, help="Generated expression prediction tensor.")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=Path, default=Path("datasets/her2st"), help="Dataset root directory.")
    parser.add_argument("--feature-root", "--processed-dir", "--processed_dir", dest="feature_root", type=Path, default=None, help="Optional override for the derived feature directory.")
    parser.add_argument("--gene-order-file", "--gene-list", "--gene_list_filename", dest="gene_list", type=str, default=None, help="Ordered gene file filename or path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where evaluation artefacts are stored.")
    parser.add_argument("--samples-per-condition", type=int, default=0, help="Optional override for repetitions per condition.")
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOPK_VALUES),
        help="Spot-mode top-k PCC cutoffs; ignored in single-cell mode.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    parser.add_argument("--slide-out", "--slide_out", dest="slide_out", type=str, default="", help="Held-out spot slide for spot mode.")
    parser.add_argument("--fold", type=int, default=0, help="Single-cell fold index.")
    parser.add_argument("--split", type=str, default="val", help="Single-cell split name.")
    parser.add_argument("--slide-name", type=str, default="", help="Single-cell slide/sample name.")
    return parser


def maybe_save_histogram(
    values: np.ndarray,
    output_path: Path,
    xlabel: str,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(valid, bins=50)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Gene count")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.mode = normalize_mode(args.mode)

    output_dir = ensure_directory(args.output_dir or default_output_dir(args.sample_file))
    logger = build_logger(output_dir, args.verbose, log_filename="evaluation.log")

    ground_truth = load_ground_truth(args)
    prediction_tensor = load_prediction_tensor(args.sample_file)

    num_entities, num_genes = ground_truth.matrix.shape
    if prediction_tensor.shape[1] != num_genes:
        raise ValueError(
            f"Prediction gene dimension ({prediction_tensor.shape[1]}) does not match ground truth ({num_genes})."
        )

    explicit_repetitions = args.samples_per_condition if args.samples_per_condition > 0 else None
    repetitions = infer_repetitions(prediction_tensor.shape[0], num_entities, explicit_repetitions)
    prediction_mean = prediction_tensor.view(num_entities, repetitions, num_genes).mean(dim=1).numpy()

    ground_truth_matrix = ground_truth.matrix.astype(np.float64)
    prediction_matrix = prediction_mean.astype(np.float64)
    pred_variance = np.var(prediction_matrix, axis=0)
    gt_variance = np.var(ground_truth_matrix, axis=0)

    metrics = {
        "mode": args.mode,
        "sample_file": str(Path(args.sample_file).expanduser().resolve()),
        "num_entities": int(num_entities),
        "num_genes": int(num_genes),
        "samples_per_condition": int(repetitions),
        "primary_metrics": "gene_level_pcc_rmse" if args.mode == "spot" else "gene_level_ccc",
    }

    if args.mode == "spot":
        gene_pcc = np.array(
            [
                safe_corr(ground_truth_matrix[:, gene_index], prediction_matrix[:, gene_index])
                for gene_index in range(num_genes)
            ],
            dtype=np.float64,
        )
        gene_rmse = np.sqrt(np.mean((ground_truth_matrix - prediction_matrix) ** 2, axis=0))
        metrics.update(
            {
                "pcc_mean": finite_mean(gene_pcc),
                "pcc_median": finite_median(gene_pcc),
                "rmse_mean": finite_mean(gene_rmse),
                "rmse_median": finite_median(gene_rmse),
                "num_valid_pcc_genes": int(np.isfinite(gene_pcc).sum()),
                "num_valid_rmse_genes": int(np.isfinite(gene_rmse).sum()),
            }
        )
        for top_k in sorted(set(args.top_k)):
            metrics[f"pcc_top_{top_k}"] = topk_mean_desc(gene_pcc, top_k)

        gene_metrics_frame = pd.DataFrame(
            {
                "gene": ground_truth.gene_names,
                "pcc": gene_pcc,
                "rmse": gene_rmse,
                "gt_var": gt_variance,
                "pred_var": pred_variance,
            }
        ).sort_values(by=["pcc", "gene"], ascending=[False, True])
        histogram_specs = [
            (
                gene_pcc,
                "gene_pcc_distribution.png",
                "Pearson correlation",
                "GENIST gene-wise PCC distribution",
            ),
            (
                gene_rmse,
                "gene_rmse_distribution.png",
                "RMSE",
                "GENIST gene-wise RMSE distribution",
            ),
        ]
        top_genes_filename = "top_pcc_genes.csv"
    else:
        gene_ccc = concordance_correlation(ground_truth_matrix, prediction_matrix)
        metrics.update(
            {
                "ccc_mean": finite_mean(gene_ccc),
                "ccc_median": finite_median(gene_ccc),
                "num_valid_ccc_genes": int(np.isfinite(gene_ccc).sum()),
            }
        )
        gene_metrics_frame = pd.DataFrame(
            {
                "gene": ground_truth.gene_names,
                "ccc": gene_ccc,
                "gt_var": gt_variance,
                "pred_var": pred_variance,
            }
        ).sort_values(by=["ccc", "gene"], ascending=[False, True])
        histogram_specs = [
            (
                gene_ccc,
                "gene_ccc_distribution.png",
                "Concordance correlation coefficient",
                "GENIST gene-wise CCC distribution",
            ),
        ]
        top_genes_filename = "top_ccc_genes.csv"

    gene_metrics_frame.to_csv(output_dir / "gene_level_metrics.csv", index=False, encoding="utf-8-sig")

    zero_gt_frame = gene_metrics_frame.loc[gene_metrics_frame["gt_var"] <= 1e-8]
    zero_pred_frame = gene_metrics_frame.loc[gene_metrics_frame["pred_var"] <= 1e-8]
    if not zero_gt_frame.empty:
        zero_gt_frame.to_csv(output_dir / "zero_variance_ground_truth_genes.csv", index=False, encoding="utf-8-sig")
    if not zero_pred_frame.empty:
        zero_pred_frame.to_csv(output_dir / "zero_variance_prediction_genes.csv", index=False, encoding="utf-8-sig")

    gene_metrics_frame.head(min(10, len(gene_metrics_frame))).to_csv(
        output_dir / top_genes_filename,
        index=False,
        encoding="utf-8-sig",
    )

    for values, filename, xlabel, title in histogram_specs:
        maybe_save_histogram(values, output_dir / filename, xlabel=xlabel, title=title)

    write_json(output_dir / "summary_metrics.json", metrics)
    save_arguments(output_dir / "evaluation_config.json", args, extra={"output_dir": str(output_dir)})

    logger.info("Evaluation complete. Key metrics:")
    for key, value in metrics.items():
        if key.startswith("pcc_top_") or key.endswith(("_mean", "_median")):
            logger.info("%s = %.6f", key, value)


if __name__ == "__main__":
    main()
