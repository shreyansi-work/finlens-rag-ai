"""Ingest every PDF in data/raw/ (incremental: unchanged files are skipped).

Run:  python scripts/ingest.py [--strategy recursive] [--rebuild]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finlens.config import RAW_DIR, settings  # noqa: E402
from finlens.ingestion.pipeline import ingest_folder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=settings.chunk_strategy,
                    choices=["fixed", "recursive", "structure", "semantic"])
    ap.add_argument("--folder", default=str(RAW_DIR))
    ap.add_argument("--rebuild", action="store_true", help="delete the database and start fresh")
    args = ap.parse_args()

    s = settings.with_(chunk_strategy=args.strategy)
    if args.rebuild:
        for p in Path(s.db_path).parent.glob(Path(s.db_path).stem + ".*"):
            p.unlink()
    print(f"Ingesting {args.folder} (chunking: {s.chunk_strategy})")
    stats = ingest_folder(Path(args.folder), s)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
