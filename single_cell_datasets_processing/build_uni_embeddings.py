"""Build UNI embeddings for single-cell patches and create fold splits.

This script:

1. Loads single-cell RGB patches and binary masks.
2. Extracts UNI token-level local+global features.
3. Saves baseline and augmented embeddings.
4. Aligns the embeddings with the expression matrix.
5. Builds spatial train/validation folds based on the y-coordinate of each cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from tqdm import tqdm

try:
    import tifffile
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    tifffile = None


AUGMENT_TRANSFORMS = (
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)


def require_runtime_dependencies() -> None:
    if tifffile is None:
        raise ModuleNotFoundError("Missing runtime dependency: tifffile")


def import_uni_encoder():
    try:
        from uni import get_encoder
    except ImportError as exc:
        raise ModuleNotFoundError(
            "UNI is an optional external dependency. Install the official package with "
            "`pip install git+https://github.com/mahmoodlab/UNI.git` and request access "
            "to MahmoodLab/UNI on Hugging Face."
        ) from exc
    return get_encoder


def load_uni_encoder(device: torch.device):
    get_encoder = import_uni_encoder()
    model, base_transform = get_encoder(enc_name="uni", device=device)
    model.eval()
    return model, base_transform


def prepare_uni_input(patch: Image.Image, base_transform) -> torch.Tensor:
    tensor = base_transform(patch)
    if not torch.is_tensor(tensor):
        tensor = transforms.ToTensor()(tensor)
    return tensor.float().unsqueeze(0)


@torch.no_grad()
def uni_forward_tokens(model, inputs: torch.Tensor) -> torch.Tensor:
    """Return all UNI tokens, including the CLS token."""

    return model.forward_features(inputs)


@torch.no_grad()
def get_uni_local_global_batch(
    patches: list[Image.Image],
    masks: list[Image.Image],
    model,
    base_transform,
    device: torch.device,
    img_size: int = 224,
    patch_size: int = 16,
) -> torch.Tensor:
    """Extract local+global UNI features for a batch of patches."""

    if len(patches) != len(masks):
        raise ValueError("The number of patches and masks must match.")

    patch_tensors = []
    mask_tensors = []
    for patch, mask in zip(patches, masks):
        patch_tensors.append(prepare_uni_input(patch, base_transform))
        mask_resized = mask.resize((img_size, img_size), Image.Resampling.NEAREST)
        mask_tensors.append(transforms.ToTensor()(mask_resized).unsqueeze(0))

    patch_batch = torch.cat(patch_tensors, dim=0).to(device)
    mask_batch = torch.cat(mask_tensors, dim=0).to(device)
    tokens = uni_forward_tokens(model, patch_batch)
    if tokens.ndim != 3:
        raise RuntimeError(f"Unexpected UNI token shape: {tokens.shape}")

    cls_tokens = tokens[:, 0, :]
    patch_tokens = tokens[:, 1:, :]

    pooled_masks = F.avg_pool2d(mask_batch, kernel_size=patch_size, stride=patch_size)
    pooled_masks = pooled_masks.view(mask_batch.shape[0], -1)
    weights = (pooled_masks > 0.5).float()
    sums = weights.sum(dim=1, keepdim=True)

    zero_mask = sums.squeeze(1) == 0
    if zero_mask.any():
        max_indices = torch.argmax(pooled_masks[zero_mask], dim=1)
        weights[zero_mask] = 0.0
        weights[zero_mask, max_indices] = 1.0
        sums[zero_mask] = 1.0

    normalized_weights = weights / (sums + 1e-8)
    local_tokens = (normalized_weights.unsqueeze(-1) * patch_tokens).sum(dim=1)
    return torch.cat([local_tokens, cls_tokens], dim=1).cpu()


def load_or_compute_coords(
    cell_ids: Sequence[int],
    segmentation_path: str | Path,
    cache_path: str | Path,
) -> tuple[np.ndarray, int]:
    """Load cached cell coordinates or compute them from the segmentation."""

    cache_path = Path(cache_path).expanduser().resolve()
    if cache_path.exists():
        cache = np.load(cache_path)
        cached_ids = cache["ids"]
        cached_coords = cache["coords"]
        seg_height = int(cache["seg_height"])
        id_to_coord = {int(cell_id): coord for cell_id, coord in zip(cached_ids, cached_coords)}
        coords = np.array([id_to_coord.get(int(cell_id), (np.nan, np.nan)) for cell_id in cell_ids], dtype=np.float32)
        print(f"[INFO] Loaded cached coordinates from {cache_path}")
        return coords, seg_height

    segmentation = tifffile.imread(segmentation_path)
    unique_ids = np.unique(segmentation)
    unique_ids = unique_ids[unique_ids != 0]
    centers = ndimage.center_of_mass(
        np.ones_like(segmentation, dtype=np.uint8),
        labels=segmentation,
        index=unique_ids,
    )

    coords = []
    for center in centers:
        if isinstance(center, tuple) and not any(np.isnan(value) for value in center):
            coords.append((float(center[0]), float(center[1])))
        else:
            coords.append((np.nan, np.nan))
    coords_array = np.array(coords, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        ids=unique_ids.astype(np.int64),
        coords=coords_array,
        seg_height=int(segmentation.shape[0]),
    )
    print(f"[INFO] Cached coordinates to {cache_path}")

    id_to_coord = {int(cell_id): coord for cell_id, coord in zip(unique_ids, coords_array)}
    ordered_coords = np.array([id_to_coord.get(int(cell_id), (np.nan, np.nan)) for cell_id in cell_ids], dtype=np.float32)
    return ordered_coords, int(segmentation.shape[0])


def save_text_list(path: str | Path, values: Sequence[str]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(f"{value}\n")


def validate_divisions(divisions: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    validated: list[tuple[float, float]] = []
    for division in divisions:
        if len(division) != 2:
            raise ValueError(f"Each region division must contain exactly two values, got: {division}")
        start, end = float(division[0]), float(division[1])
        if start > end:
            raise ValueError(f"Region division start must be <= end, got: {division}")
        validated.append((start, end))
    return validated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract UNI local+global embeddings for single-cell patches and build fold splits."
    )
    parser.add_argument(
        "--patch-metadata",
        type=Path,
        default=Path("processed/single_cell_patches/patch_metadata.csv"),
        help="Patch metadata CSV with `cell_id`, `patch_path`, and `mask_path` columns.",
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=Path("processed/single_cell_patches"),
        help="Root directory that contains the `patches/` and `masks/` folders.",
    )
    parser.add_argument(
        "--expr-csv",
        type=Path,
        default=Path("processed/cell_gene_matrix_filtered.csv"),
        help="Filtered cell-by-gene expression matrix aligned to the selected cells.",
    )
    parser.add_argument(
        "--fp-seg-hist",
        type=Path,
        default=Path("processed/he_image_nuclei_seg_microns.tif"),
        help="Micron-resolution H&E nuclei segmentation used to compute spatial folds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("derived_features"),
        help="Output directory used to store embeddings and fold splits.",
    )
    parser.add_argument(
        "--slide-name",
        type=str,
        default="sample",
        help="Slide or sample name used when saving embedding tensors.",
    )
    parser.add_argument(
        "--gene-list-filename",
        type=str,
        default="genes_selected.txt",
        help="Canonical filename for the generated gene list.",
    )
    parser.add_argument(
        "--sample-list-filename",
        type=str,
        default="samples_single_cell.txt",
        help="Canonical filename for the generated sample list.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional Hugging Face token used when downloading UNI weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device. Falls back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=7,
        help="Number of transpose-based augmentations saved per cell. Must be between 0 and 7.",
    )
    parser.add_argument(
        "--region-divisions",
        type=str,
        default="[[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]",
        help="JSON list of y-coordinate intervals used to define validation folds.",
    )
    parser.add_argument(
        "--batch-cells",
        type=int,
        default=16,
        help="Number of cells processed per forward batch before augmentation expansion.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_runtime_dependencies()
    if args.num_aug < 0 or args.num_aug > len(AUGMENT_TRANSFORMS):
        raise ValueError(f"`--num-aug` must be between 0 and {len(AUGMENT_TRANSFORMS)}.")

    output_dir = args.output_dir.expanduser().resolve()
    for subdir in (
        "embeddings/single_cell/uni/original",
        "embeddings/single_cell/uni/augmented",
    ):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.hf_token:
        from huggingface_hub import login

        login(token=args.hf_token)

    print("[INFO] Loading UNI encoder.")
    uni_model, uni_transform = load_uni_encoder(device)

    metadata = pd.read_csv(args.patch_metadata).sort_values("cell_id").reset_index(drop=True)
    metadata["cell_id"] = metadata["cell_id"].astype(int)
    patch_paths = [args.patch_root / rel_path for rel_path in metadata["patch_path"]]
    mask_paths = [args.patch_root / rel_path for rel_path in metadata["mask_path"]]
    cell_ids = metadata["cell_id"].tolist()

    coords_cache = args.patch_root / "coords_cache.npz"
    coords, seg_height = load_or_compute_coords(cell_ids, args.fp_seg_hist, coords_cache)
    norm_y = coords[:, 0] / seg_height

    print("[INFO] Extracting UNI embeddings.")
    batch_cells = max(1, int(args.batch_cells))
    base_chunks: list[torch.Tensor] = []
    aug_chunks: list[torch.Tensor] = []

    selected_transforms = AUGMENT_TRANSFORMS[: args.num_aug]
    for start in tqdm(range(0, len(patch_paths), batch_cells), desc="Batches"):
        end = min(start + batch_cells, len(patch_paths))
        batch_patches = []
        batch_masks = []
        for patch_path, mask_path in zip(patch_paths[start:end], mask_paths[start:end]):
            batch_patches.append(Image.open(patch_path).convert("RGB"))
            batch_masks.append(Image.open(mask_path).convert("L"))

        expanded_patches: list[Image.Image] = []
        expanded_masks: list[Image.Image] = []
        for patch, mask in zip(batch_patches, batch_masks):
            expanded_patches.append(patch)
            expanded_masks.append(mask)
            for transform in selected_transforms:
                expanded_patches.append(patch.transpose(transform))
                expanded_masks.append(mask.transpose(transform))

        features = get_uni_local_global_batch(
            expanded_patches,
            expanded_masks,
            uni_model,
            uni_transform,
            device,
        )
        features = features.view(end - start, 1 + args.num_aug, -1)
        base_chunks.append(features[:, 0, :])
        aug_chunks.append(features[:, 1:, :])

    uni_tensor = torch.cat(base_chunks, dim=0)
    uni_aug_tensor = torch.cat(aug_chunks, dim=0)

    uni_path = output_dir / "embeddings/single_cell/uni/original" / f"{args.slide_name}.pt"
    uni_aug_path = output_dir / "embeddings/single_cell/uni/augmented" / f"{args.slide_name}.pt"
    torch.save(uni_tensor, uni_path)
    torch.save(uni_aug_tensor, uni_aug_path)

    expression_frame = pd.read_csv(args.expr_csv, index_col=0)
    if "scClassify_label" in expression_frame.columns:
        expression_frame = expression_frame.drop(columns=["scClassify_label"])
    expression_frame.index = expression_frame.index.astype(int)
    missing_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in expression_frame.index]
    if missing_cell_ids:
        raise ValueError(
            f"{len(missing_cell_ids)} cells from patch metadata are missing in the expression matrix. "
            f"Examples: {missing_cell_ids[:10]}"
        )
    expression_frame = expression_frame.loc[cell_ids]

    gene_list = expression_frame.columns.astype(str).tolist()
    save_text_list(output_dir / args.gene_list_filename, gene_list)
    save_text_list(output_dir / args.sample_list_filename, [args.slide_name])

    divisions = validate_divisions(json.loads(args.region_divisions))

    def save_split(split_dir: Path, indices: np.ndarray) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        for subdir in (
            "embeddings/single_cell/uni/original",
            "embeddings/single_cell/uni/augmented",
        ):
            (split_dir / subdir).mkdir(parents=True, exist_ok=True)

        torch.save(
            uni_tensor[indices],
            split_dir / "embeddings/single_cell/uni/original" / f"{args.slide_name}.pt",
        )
        torch.save(
            uni_aug_tensor[indices],
            split_dir / "embeddings/single_cell/uni/augmented" / f"{args.slide_name}.pt",
        )
        split_expression = expression_frame.iloc[indices]
        split_expression.to_csv(split_dir / "expression.csv")
        np.save(split_dir / "cell_ids.npy", split_expression.index.values)
        np.save(split_dir / "coords.npy", coords[indices])
        metadata.iloc[indices].to_csv(split_dir / "patch_metadata.csv", index=False)

    for fold_index, (start, end) in enumerate(divisions, start=1):
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_text_list(fold_dir / args.gene_list_filename, gene_list)
        save_text_list(fold_dir / args.sample_list_filename, [args.slide_name])

        if fold_index == len(divisions):
            val_mask = (norm_y >= start) & (norm_y <= end)
        else:
            val_mask = (norm_y >= start) & (norm_y < end)
        val_mask &= np.isfinite(norm_y)

        val_indices = np.sort(np.where(val_mask)[0])
        train_indices = np.sort(np.where(~val_mask)[0])
        save_split(fold_dir / "train", train_indices)
        save_split(fold_dir / "val", val_indices)

        print(
            f"[INFO] Fold {fold_index}: train={len(train_indices)} cells, "
            f"val={len(val_indices)} cells, region=[{start}, {end}]"
        )

    print(f"[INFO] Saved UNI embeddings to {uni_path}")
    print(f"[INFO] Saved augmented UNI embeddings to {uni_aug_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
