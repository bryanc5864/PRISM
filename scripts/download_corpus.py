#!/usr/bin/env python
"""Download pre-training corpus datasets.

Usage:
    python scripts/download_corpus.py --tier 1
    python scripts/download_corpus.py --tier 2
    python scripts/download_corpus.py --tier 3
    python scripts/download_corpus.py --tier all
    python scripts/download_corpus.py --dataset replogle_k562_gwps
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.data.corpus.download import download_dataset, download_tier, download_all
from prism.data.corpus.registry import DATASET_REGISTRY
from prism.data.corpus.builder import CorpusBuilder


def main():
    parser = argparse.ArgumentParser(description="Download corpus datasets")
    parser.add_argument("--tier", type=str, default="all", help="Tier to download (1/2/3/all)")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset ID")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--status", action="store_true", help="Show status only")
    args = parser.parse_args()

    if args.status:
        builder = CorpusBuilder()
        builder.status()
        return

    if args.dataset:
        path = download_dataset(args.dataset, force=args.force)
        print(f"Downloaded: {path}")
        return

    if args.tier == "all":
        paths = download_all(force=args.force)
    else:
        tier = int(args.tier)
        paths = download_tier(tier, force=args.force)

    print(f"\nCompleted: {len(paths)} datasets downloaded")
    for did, path in paths.items():
        fsize = os.path.getsize(path) / (1024**3) if os.path.exists(path) else 0
        print(f"  {did}: {fsize:.2f} GB -> {path}")


if __name__ == "__main__":
    main()
