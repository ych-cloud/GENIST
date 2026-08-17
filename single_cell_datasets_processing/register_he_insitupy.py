"""Register an H&E image to Xenium coordinates with InSituPy.

This script wraps the standard InSituPy workflow:

1. Load a Xenium run directory.
2. Register an external H&E image to a Xenium template image, such as `nuclei`.
3. Optionally save the registered image back to the Xenium object layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register an H&E image to Xenium coordinates with InSituPy."
    )
    parser.add_argument(
        "--xenium-dir",
        type=Path,
        required=True,
        help="Xenium output directory readable by `insitupy.io.read_xenium`.",
    )
    parser.add_argument(
        "--he-image",
        type=Path,
        required=True,
        help="Path to the H&E image that should be registered.",
    )
    parser.add_argument(
        "--template-image-name",
        type=str,
        default="nuclei",
        help="Xenium image key used as the registration template.",
    )
    parser.add_argument(
        "--axes-image",
        type=str,
        default="YXS",
        help="Axis order of the H&E image. Typical RGB TIFF layout is `YXS`.",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        default="HE",
        help="Channel name passed to InSituPy for the registered H&E image.",
    )
    parser.add_argument(
        "--axes-template",
        type=str,
        default="YX",
        help="Axis order of the Xenium template image.",
    )
    parser.add_argument(
        "--save-registered-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save the registered image through InSituPy.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from insitupy.io import read_xenium
    from insitupy.tools import register_images

    xenium_dir = args.xenium_dir.expanduser().resolve()
    he_image = args.he_image.expanduser().resolve()

    print(f"[INFO] Loading Xenium data from {xenium_dir}")
    xenium_data = read_xenium(xenium_dir)
    print(xenium_data)
    print("[INFO] Available Xenium images:", xenium_data.images)

    print(f"[INFO] Registering H&E image {he_image}")
    register_images(
        data=xenium_data,
        image_to_be_registered=he_image,
        axes_image=args.axes_image,
        channel_names=args.channel_names,
        template_image_name=args.template_image_name,
        axes_template=args.axes_template,
        save_registered_images=args.save_registered_images,
    )
    print("[INFO] Registration finished.")


if __name__ == "__main__":
    main()
