from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from .common import DEFAULT_TOPK_VALUES, build_logger, ensure_directory, normalize_mode, save_arguments, write_json
from .data import load_ground_truth, resolve_feature_root


SPOT_ID_PATTERN = re.compile(r"(\d+)x(\d+)")


def safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    if x.std() <= eps or y.std() <= eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_ssim_2d(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute spatial SSIM with the Wang/skimage Gaussian-window settings."""
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


def build_spatial_grid_index(coordinates: np.ndarray, grid_size: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"Single-cell coordinates must have shape [N, 2]; received {coordinates.shape}.")
    if coordinates.shape[0] < 3 or not np.isfinite(coordinates[:, :2]).all():
        raise ValueError("Spatial SSIM requires at least three entities with finite coordinates.")

    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    x_index = np.floor((x_values - x_min) / x_span * (grid_size - 1)).astype(np.int32)
    y_index = np.floor((y_values - y_min) / y_span * (grid_size - 1)).astype(np.int32)
    x_index = np.clip(x_index, 0, grid_size - 1)
    y_index = np.clip(y_index, 0, grid_size - 1)
    flat_index = y_index * grid_size + x_index

    occupancy = np.bincount(flat_index, minlength=grid_size * grid_size).reshape(grid_size, grid_size)
    occupied_rows = np.where(occupancy.any(axis=1))[0]
    occupied_columns = np.where(occupancy.any(axis=0))[0]
    crop = (
        slice(int(occupied_rows[0]), int(occupied_rows[-1]) + 1),
        slice(int(occupied_columns[0]), int(occupied_columns[-1]) + 1),
    )
    return flat_index, crop


def rasterize_gene_to_grid(values: np.ndarray, flat_index: np.ndarray, grid_size: int) -> np.ndarray:
    values = np.asarray(values)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    valid_values = values[valid].astype(np.float64, copy=False)
    sums = np.bincount(flat_index[valid], weights=valid_values, minlength=grid_size * grid_size)
    counts = np.bincount(flat_index[valid], minlength=grid_size * grid_size)
    grid = np.zeros(grid_size * grid_size, dtype=np.float32)
    occupied = counts > 0
    grid[occupied] = (sums[occupied] / counts[occupied]).astype(np.float32, copy=False)
    return grid.reshape(grid_size, grid_size)


def compute_gene_ssim(
    args,
    entity_ids: list[str],
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    num_entities, num_genes = ground_truth.shape
    gene_ssim = np.full(num_genes, np.nan, dtype=np.float64)

    if args.mode == "spot":
        for gene_index in range(num_genes):
            gt_values = ground_truth[:, gene_index]
            pred_values = prediction[:, gene_index]
            fill_value = float(min(np.min(gt_values), np.min(pred_values)))
            gt_grid = build_spot_grid(entity_ids, gt_values, fill_value)
            pred_grid = build_spot_grid(entity_ids, pred_values, fill_value)
            gene_ssim[gene_index] = safe_ssim_2d(gt_grid, pred_grid)
        return gene_ssim

    processed_dir = resolve_feature_root(args.data_path, args.feature_root)
    default_coordinates_path = processed_dir / f"fold_{args.fold}" / args.split / "coords.npy"
    coordinates_path = (args.coords_file or default_coordinates_path).expanduser().resolve()
    if not coordinates_path.exists():
        raise FileNotFoundError(
            "Spatial SSIM requires single-cell coordinates. "
            f"Missing coordinate file: {coordinates_path}"
        )
    coordinates = np.load(coordinates_path)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            f"Single-cell coordinates must have shape [N, 2]; received {coordinates.shape}."
        )
    if coordinates.shape[0] != num_entities:
        raise ValueError(
            "Coordinate rows do not match evaluation entities: "
            f"{coordinates.shape[0]} vs {num_entities}."
        )

    valid_coordinates = np.isfinite(coordinates[:, :2]).all(axis=1)
    coordinates = coordinates[valid_coordinates]
    ground_truth = ground_truth[valid_coordinates]
    prediction = prediction[valid_coordinates]
    flat_index, crop = build_spatial_grid_index(coordinates, args.ssim_grid_size)
    for gene_index in range(num_genes):
        gt_grid = rasterize_gene_to_grid(ground_truth[:, gene_index], flat_index, args.ssim_grid_size)[crop]
        pred_grid = rasterize_gene_to_grid(prediction[:, gene_index], flat_index, args.ssim_grid_size)[crop]
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
    parser.add_argument("--top-k", nargs="+", type=int, default=list(DEFAULT_TOPK_VALUES), help="Top-k PCC cutoffs.")
    parser.add_argument("--coords-file", type=Path, default=None, help="Optional single-cell coordinate .npy override.")
    parser.add_argument("--ssim-grid-size", type=int, default=128, help="Single-cell raster size used for spatial SSIM.")
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
    if args.ssim_grid_size < 16:
        raise ValueError("--ssim-grid-size must be at least 16.")

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
    gene_pcc = np.array(
        [safe_corr(ground_truth_matrix[:, gene_index], prediction_matrix[:, gene_index]) for gene_index in range(num_genes)],
        dtype=np.float64,
    )
    gene_rmse = np.sqrt(np.mean((ground_truth_matrix - prediction_matrix) ** 2, axis=0))
    gene_ssim = compute_gene_ssim(
        args=args,
        entity_ids=ground_truth.entity_ids,
        ground_truth=ground_truth_matrix,
        prediction=prediction_matrix,
    )

    pred_variance = np.var(prediction_matrix, axis=0)
    gt_variance = np.var(ground_truth_matrix, axis=0)

    metrics = {
        "mode": args.mode,
        "sample_file": str(Path(args.sample_file).expanduser().resolve()),
        "num_entities": int(num_entities),
        "num_genes": int(num_genes),
        "samples_per_condition": int(repetitions),
        "ssim_dimension": "2d_spatial",
        "ssim_grid_size": int(args.ssim_grid_size) if args.mode == "single_cell" else None,
        "pcc_mean": finite_mean(gene_pcc),
        "pcc_median": finite_median(gene_pcc),
        "rmse_mean": finite_mean(gene_rmse),
        "rmse_median": finite_median(gene_rmse),
        "ssim_mean": finite_mean(gene_ssim),
        "ssim_median": finite_median(gene_ssim),
        "num_valid_pcc_genes": int(np.isfinite(gene_pcc).sum()),
        "num_valid_rmse_genes": int(np.isfinite(gene_rmse).sum()),
        "num_valid_ssim_genes": int(np.isfinite(gene_ssim).sum()),
    }
    for top_k in sorted(set(args.top_k)):
        metrics[f"pcc_top_{top_k}"] = topk_mean_desc(gene_pcc, top_k)

    gene_metrics_frame = pd.DataFrame(
        {
            "gene": ground_truth.gene_names,
            "pcc": gene_pcc,
            "rmse": gene_rmse,
            "ssim": gene_ssim,
            "gt_var": gt_variance,
            "pred_var": pred_variance,
        }
    ).sort_values(by=["pcc", "gene"], ascending=[False, True])
    gene_metrics_frame.to_csv(output_dir / "gene_level_metrics.csv", index=False, encoding="utf-8-sig")

    zero_gt_frame = gene_metrics_frame.loc[gene_metrics_frame["gt_var"] <= 1e-8]
    zero_pred_frame = gene_metrics_frame.loc[gene_metrics_frame["pred_var"] <= 1e-8]
    if not zero_gt_frame.empty:
        zero_gt_frame.to_csv(output_dir / "zero_variance_ground_truth_genes.csv", index=False, encoding="utf-8-sig")
    if not zero_pred_frame.empty:
        zero_pred_frame.to_csv(output_dir / "zero_variance_prediction_genes.csv", index=False, encoding="utf-8-sig")

    gene_metrics_frame.head(min(10, len(gene_metrics_frame))).to_csv(
        output_dir / "top_pcc_genes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    maybe_save_histogram(
        gene_pcc,
        output_dir / "gene_pcc_distribution.png",
        xlabel="Pearson correlation",
        title="GENIST gene-wise PCC distribution",
    )
    maybe_save_histogram(
        gene_rmse,
        output_dir / "gene_rmse_distribution.png",
        xlabel="RMSE",
        title="GENIST gene-wise RMSE distribution",
    )
    maybe_save_histogram(
        gene_ssim,
        output_dir / "gene_ssim_distribution.png",
        xlabel="SSIM",
        title="GENIST gene-wise SSIM distribution",
    )
    write_json(output_dir / "summary_metrics.json", metrics)
    save_arguments(output_dir / "evaluation_config.json", args, extra={"output_dir": str(output_dir)})

    logger.info("Evaluation complete. Key metrics:")
    for key, value in metrics.items():
        if key.startswith("pcc_top_") or key in {
            "pcc_mean",
            "pcc_median",
            "rmse_mean",
            "rmse_median",
            "ssim_mean",
            "ssim_median",
        }:
            logger.info("%s = %.6f", key, value)


if __name__ == "__main__":
    main()
