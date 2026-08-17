"""Rasterize Xenium nucleus polygons into a segmentation mask."""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path

import natsort
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


MICRONS_PER_PIXEL = 0.2125


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


def process_patch(
    width_start: int,
    width_end: int,
    boundaries: pd.DataFrame,
    cell_ids: list[int],
    image_width: int,
    image_height: int,
    output_dir: str,
) -> None:
    """Rasterize one vertical strip of polygons."""

    patch = np.zeros((image_height, width_end - width_start), dtype=np.uint32)
    boundaries_chunk = boundaries[(boundaries["vertex_x"] >= width_start) & (boundaries["vertex_x"] < width_end)]
    cells_chunk = set(boundaries_chunk["cell_id"].tolist())
    cells_to_draw = list(set(cell_ids) & cells_chunk)
    boundaries_cells = boundaries[boundaries["cell_id"].isin(cells_to_draw)]

    for cell_id in tqdm(cells_to_draw, total=len(cells_to_draw)):
        cell_vertices = boundaries_cells[boundaries_cells["cell_id"] == cell_id]
        if (cell_vertices["vertex_x"] >= width_end).any() or (cell_vertices["vertex_x"] < width_start).any():
            continue

        cell_vertices = cell_vertices.assign(vertex_x=cell_vertices["vertex_x"] - width_start)
        cell_vertices.loc[cell_vertices["vertex_x"] >= width_end, "vertex_x"] = width_end - 1
        cell_vertices.loc[cell_vertices["vertex_y"] >= image_height, "vertex_y"] = image_height - 1

        coords = [list(vertex) for vertex in zip(cell_vertices["vertex_x"], cell_vertices["vertex_y"])]
        polygon = np.array([coords], dtype=np.int32)
        mask = np.zeros((image_height, width_end - width_start), dtype=np.uint8)
        mask = cv2.fillPoly(mask, polygon, 255)
        patch = np.where(mask > 0, cell_id, patch)

    tifffile.imwrite(Path(output_dir) / f"xenium_nuclei_{width_start}.tif", patch, photometric="minisblack")


def get_n_processes(requested_processes: int | None) -> int:
    if requested_processes is None or requested_processes <= 0:
        return max(mp.cpu_count() - 2, 1)
    return min(requested_processes, mp.cpu_count())


def swap_columns(frame: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    columns = frame.columns.tolist()
    index1, index2 = columns.index(column1), columns.index(column2)
    columns[index1], columns[index2] = columns[index2], columns[index1]
    frame.columns = columns
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rasterize Xenium nucleus boundaries into a full segmentation mask."
    )
    parser.add_argument(
        "--fp-boundaries",
        type=Path,
        required=True,
        help="Compressed Xenium nucleus boundary CSV.",
    )
    parser.add_argument(
        "--fp-he-img",
        type=Path,
        required=True,
        help="Registered full-resolution H&E image used only to determine output size.",
    )
    parser.add_argument(
        "--fp-out-nuclei-seg",
        type=str,
        default="xenium_nuclei.tif",
        help="Output filename for the rasterized Xenium nucleus segmentation.",
    )
    parser.add_argument(
        "--fp-ids-out",
        type=str,
        default="cell_ids_xenium.csv",
        help="Output filename for the Xenium string-to-integer cell-ID mapping.",
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=0.2,
        help="Width of each processing strip as a fraction of the smaller image dimension.",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=0.1,
        help="Overlap ratio between adjacent processing strips.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=28,
        help="Maximum number of CPU workers. Use 0 or a negative value to use all cores.",
    )
    parser.add_argument(
        "--dir-output",
        type=Path,
        default=Path("processed"),
        help="Directory used to store intermediate crops and final outputs.",
    )
    parser.add_argument(
        "--delete-intermediate-files",
        action="store_true",
        help="Delete the intermediate strip files after the full mask is assembled.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_runtime_dependencies()
    output_dir = args.dir_output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    boundaries = pd.read_csv(args.fp_boundaries, index_col=None, compression="gzip")
    original_cell_ids = boundaries["cell_id"].unique().tolist()

    if all(isinstance(item, str) for item in original_cell_ids):
        numeric_cell_ids = list(range(1, len(original_cell_ids) + 1))
        cell_id_frame = pd.DataFrame(numeric_cell_ids, index=original_cell_ids, columns=["cell_id_num"])
        mapping_path = output_dir / args.fp_ids_out
        cell_id_frame.to_csv(mapping_path)
        boundaries = boundaries.merge(cell_id_frame, how="left", left_on="cell_id", right_index=True)
        boundaries = swap_columns(boundaries, "cell_id", "cell_id_num")
        cell_ids = numeric_cell_ids
    else:
        cell_ids = original_cell_ids.copy()

    he_image = tifffile.imread(args.fp_he_img)
    if he_image.shape[-1] == 3:
        image_height, image_width = he_image.shape[0], he_image.shape[1]
    elif he_image.shape[0] == 3:
        image_height, image_width = he_image.shape[1], he_image.shape[2]
    else:
        raise SystemExit("The H&E image must be RGB with the channel dimension first or last.")

    strip_size = int(round(args.crop_fraction * min(image_height, image_width)))
    overlap = int(round(args.overlap_fraction * strip_size))
    width_starts = list(np.arange(0, image_width - strip_size, strip_size - overlap))
    width_starts.append(image_width - strip_size)

    boundaries = boundaries.assign(vertex_x=boundaries["vertex_x"] / MICRONS_PER_PIXEL)
    boundaries = boundaries.assign(vertex_y=boundaries["vertex_y"] / MICRONS_PER_PIXEL)
    boundaries["vertex_x"] = boundaries["vertex_x"].round(0).astype(int)
    boundaries["vertex_y"] = boundaries["vertex_y"].round(0).astype(int)

    num_processes = get_n_processes(args.n_processes)
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(
            process_patch,
            [
                (width_start, width_start + strip_size, boundaries, cell_ids, image_width, image_height, str(output_dir))
                for width_start in width_starts
            ],
        )

    output_path = output_dir / args.fp_out_nuclei_seg
    section_paths = natsort.natsorted(glob.glob(str(output_dir / "xenium_nuclei_*.tif")))
    full_mask = np.zeros((image_height, image_width), dtype=np.uint32)

    print("[INFO] Combining strips into the full-resolution segmentation.")
    for section_path in tqdm(section_paths):
        section = tifffile.imread(section_path)
        width_start = int(Path(section_path).stem.split("_")[-1])
        width_end = width_start + section.shape[1]
        full_mask[:, width_start:width_end] = np.where(section > 0, section, full_mask[:, width_start:width_end])

    resized_height = int(round(image_height * MICRONS_PER_PIXEL))
    resized_width = int(round(image_width * MICRONS_PER_PIXEL))
    full_mask = cv2.resize(
        full_mask.astype(np.float32),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint32)

    tifffile.imwrite(output_path, full_mask, photometric="minisblack")
    print(f"[INFO] Saved Xenium nucleus segmentation to {output_path}")
    if all(isinstance(item, str) for item in original_cell_ids):
        print(f"[INFO] Saved Xenium cell-id mapping to {output_dir / args.fp_ids_out}")
    print(f"[INFO] Total nuclei: {len(np.unique(full_mask)) - 1}")

    if args.delete_intermediate_files:
        for section_path in section_paths:
            os.remove(section_path)


if __name__ == "__main__":
    main()
