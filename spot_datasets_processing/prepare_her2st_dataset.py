#!/usr/bin/env python
from __future__ import annotations

from typing import Sequence


GENERAL_HELP = """usage: prepare_her2st_dataset.py {download,embeddings,genes} [args]

HER2ST preprocessing entrypoint.

commands:
  download     Download the HEST-1k subset used by GENIST.
  embeddings   Extract slide-level spot embeddings from HER2ST WSIs.
  genes        Build the HER2ST selected gene list.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or [])

    if not args or args[0] in {"-h", "--help"}:
        print(GENERAL_HELP)
        return 0

    command, remaining = args[0], args[1:]

    if command == "download":
        try:
            from download_hest1k import main as download_main
        except ImportError:
            from .download_hest1k import main as download_main
        return download_main(remaining)

    if command == "embeddings":
        try:
            from extract_her2st_embeddings import main as embeddings_main
        except ImportError:
            from .extract_her2st_embeddings import main as embeddings_main
        return embeddings_main(remaining)

    if command == "genes":
        try:
            from build_her2st_gene_list import main as gene_list_main
        except ImportError:
            from .build_her2st_gene_list import main as gene_list_main
        return gene_list_main(remaining)

    print(GENERAL_HELP)
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
