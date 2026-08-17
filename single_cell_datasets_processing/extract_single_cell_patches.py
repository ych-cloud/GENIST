"""Extract single-cell image patches and binary masks.

Given:

- A full-resolution H&E image.
- A micron-resolution nuclei segmentation aligned to the H&E image.
- A CSV that identifies the cells to keep.

This script extracts one square RGB patch and one binary mask per selected cell.
It stores the patch assets under `patches/` and `masks/`, and writes the
corresponding metadata to `patch_metadata.csv`.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    cv2 = None

try:
    import tifffile
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    tifffile = None


HIST = None
SEG = None
H = W = 0
HS = WS = 0
SCALE_Y = 1.0
SCALE_X = 1.0
PATCH_SIZE = 224
BBOX_MARGIN_FRAC = 0.5
PATCH_DIR = ""
MASK_DIR = ""


def require_runtime_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if tifffile is None:
        missing.append("tifffile")
    if missing:
        raise ModuleNotFoundError(
            "Missing runtime dependencies: " + ", ".join(missing)
        )


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Return an `H x W x 3` RGB image."""

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image.astype(np.uint8)


def init_worker(
    hist: np.ndarray,
    seg: np.ndarray,
    image_height: int,
    image_width: int,
    seg_height: int,
    seg_width: int,
    scale_y: float,
    scale_x: float,
    patch_size: int,
    bbox_margin_frac: float,
    patch_dir: str,
    mask_dir: str,
) -> None:
    """Populate process-local globals used by worker functions."""

    global HIST, SEG, H, W, HS, WS, SCALE_Y, SCALE_X
    global PATCH_SIZE, BBOX_MARGIN_FRAC, PATCH_DIR, MASK_DIR

    HIST = hist
    SEG = seg
    H, W = image_height, image_width
    HS, WS = seg_height, seg_width
    SCALE_Y, SCALE_X = scale_y, scale_x
    PATCH_SIZE = patch_size
    BBOX_MARGIN_FRAC = bbox_margin_frac
    PATCH_DIR = patch_dir
    MASK_DIR = mask_dir


def process_cell_id(cell_id: int) -> dict[str, object] | None:
    """Extract one patch and one mask for a single cell."""

    if SEG is None:
        return None

    ys, xs = np.where(SEG == cell_id)
    if ys.size == 0:
        return None

    y_min_seg = ys.min()
    y_max_seg = ys.max() + 1
    x_min_seg = xs.min()
    x_max_seg = xs.max() + 1

    y1 = int(np.floor(y_min_seg * SCALE_Y))
    y2 = int(np.ceil(y_max_seg * SCALE_Y))
    x1 = int(np.floor(x_min_seg * SCALE_X))
    x2 = int(np.ceil(x_max_seg * SCALE_X))

    y1 = max(y1, 0)
    y2 = min(y2, H)
    x1 = max(x1, 0)
    x2 = min(x2, W)
    if y2 <= y1 or x2 <= x1:
        return None

    bbox_height = y2 - y1
    bbox_width = x2 - x1
    side = max(bbox_height, bbox_width)
    margin = int(round(side * BBOX_MARGIN_FRAC))

    center_y = (y1 + y2) // 2
    center_x = (x1 + x2) // 2
    half = side // 2 + margin

    patch_y1 = max(center_y - half, 0)
    patch_y2 = min(center_y + half, H)
    patch_x1 = max(center_x - half, 0)
    patch_x2 = min(center_x + half, W)
    patch = HIST[patch_y1:patch_y2, patch_x1:patch_x2, :]

    seg_y1 = max(int(np.floor(patch_y1 / SCALE_Y)), 0)
    seg_y2 = min(int(np.ceil(patch_y2 / SCALE_Y)), HS)
    seg_x1 = max(int(np.floor(patch_x1 / SCALE_X)), 0)
    seg_x2 = min(int(np.ceil(patch_x2 / SCALE_X)), WS)
    seg_region = SEG[seg_y1:seg_y2, seg_x1:seg_x2]
    mask_patch = (seg_region == cell_id).astype(np.uint8) * 255

    patch_resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)

    patch_name = f"cell_{cell_id:07d}.png"
    mask_name = f"cell_{cell_id:07d}_mask.png"

    cv2.imwrite(os.path.join(PATCH_DIR, patch_name), cv2.cvtColor(patch_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask_resized)

    return {
        "cell_id": int(cell_id),
        "patch_path": os.path.join("patches", patch_name),
        "mask_path": os.path.join("masks", mask_name),
        "cy_hist": int(center_y),
        "cx_hist": int(center_x),
    }


def load_cell_ids(csv_path: str | Path) -> np.ndarray:
    """Read cell IDs from a CSV file using either a `cell_id` column or the index."""

    frame = pd.read_csv(csv_path, index_col=0)
    if "cell_id" in frame.columns:
        cell_ids = frame["cell_id"].astype(int).to_numpy()
    else:
        cell_ids = frame.index.astype(int).to_numpy()
    cell_ids = np.unique(cell_ids)
    return np.sort(cell_ids[cell_ids != 0])


def extract_single_cell_patches(
    fp_hist: str | Path,
    fp_seg_hist: str | Path,
    fp_cells_csv: str | Path,
    output_dir: str | Path,
    patch_size: int = 224,
    bbox_margin_frac: float = 0.5,
    num_workers: int | None = None,
) -> None:
    """Run the full single-cell patch extraction pipeline."""

    output_dir = Path(output_dir).expanduser().resolve()
    patch_dir = output_dir / "patches"
    mask_dir = output_dir / "masks"
    patch_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    hist = ensure_rgb(tifffile.imread(fp_hist))
    seg = tifffile.imread(fp_seg_hist).astype(np.uint32)

    image_height, image_width, _ = hist.shape
    seg_height, seg_width = seg.shape
    scale_y = image_height / seg_height
    scale_x = image_width / seg_width

    cell_ids = load_cell_ids(fp_cells_csv)
    print(f"[INFO] Extracting patches for {len(cell_ids)} cells.")

    if num_workers is None or num_workers <= 0:
        num_workers = cpu_count()
    print(f"[INFO] Using {num_workers} worker processes.")

    metadata_rows: list[dict[str, object]] = []
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            hist,
            seg,
            image_height,
            image_width,
            seg_height,
            seg_width,
            scale_y,
            scale_x,
            patch_size,
            bbox_margin_frac,
            str(patch_dir),
            str(mask_dir),
        ),
    ) as pool:
        for metadata in tqdm(
            pool.imap_unordered(process_cell_id, cell_ids),
            total=len(cell_ids),
            desc="Extracting patches",
        ):
            if metadata is not None:
                metadata_rows.append(metadata)

    metadata_frame = pd.DataFrame(metadata_rows).sort_values("cell_id").reset_index(drop=True)
    metadata_path = output_dir / "patch_metadata.csv"
    metadata_frame.to_csv(metadata_path, index=False)
    print(f"[INFO] Saved patch metadata to {metadata_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract square RGB patches and binary masks for matched single cells."
    )
    parser.add_argument(
        "--fp-hist",
        type=Path,
        required=True,
        help="Full-resolution registered H&E image.",
    )
    parser.add_argument(
        "--fp-seg-hist",
        type=Path,
        required=True,
        help="Micron-resolution H&E nuclei segmentation aligned to the H&E image.",
    )
    parser.add_argument(
        "--fp-cells-csv",
        type=Path,
        required=True,
        help="CSV listing the cells to extract. Accepts a `cell_id` column or uses the index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/single_cell_patches"),
        help="Directory used to store patches, masks, and metadata.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Output patch size after resizing.",
    )
    parser.add_argument(
        "--bbox-margin-frac",
        type=float,
        default=0.5,
        help="Extra context margin relative to the bounding-box side length.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes. Use 0 or a negative value to use all CPU cores.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_runtime_dependencies()
    extract_single_cell_patches(
        fp_hist=args.fp_hist,
        fp_seg_hist=args.fp_seg_hist,
        fp_cells_csv=args.fp_cells_csv,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        bbox_margin_frac=args.bbox_margin_frac,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
