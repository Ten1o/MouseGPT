#!/usr/bin/env python3
"""Validate a session or rank its frames by a natural-language query."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "4_NLUI"))


def validate_session(path):
    """Check the stored vectors and configured model without a provider request."""
    import pandas as pd
    from mousegpt_config import load_config
    from mousegpt_embeddings import require_compatible_embeddings
    from runtime_config import validate_frame_ids
    from tools_function.similarity import validate_vectors

    config = load_config("EMBEDDING", require_key=False)
    with pd.HDFStore(path, mode="r") as store:
        descriptions = store.get("descriptions")
        features = store.get("features_des")
        ids = validate_frame_ids(descriptions["id"], features.index)
        matrix = validate_vectors(features.to_numpy(), name="Stored embeddings")
        require_compatible_embeddings(path, store, config, matrix.shape[1])
        has_linkage = "/linkage_matrix" in store.keys()
    return {"frames": len(ids), "dimensions": matrix.shape[1],
            "embedding_model": config.model, "has_linkage": has_linkage}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--query", help="Behavior to search for, such as 'mouse walking'")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, help="New CSV path; existing files are never overwritten")
    parser.add_argument("--validate-only", action="store_true", help="Validate locally without a key or an API call")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.validate_only and (not args.query or not args.query.strip()):
        parser.error("Supply --query or use --validate-only")
    if not args.session.is_file():
        parser.error(f"Session not found: {args.session}")
    if not args.validate_only and args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new filename")
    try:
        summary = validate_session(args.session)
        if args.validate_only:
            print(json.dumps({"status": "validated", **summary}, indent=2))
            return 0
        from tools_function.search import search
        ids, scores = search(args.session, args.query)
        rows = [{"frame_id": frame_id, "similarity": float(score)}
                for frame_id, score in zip(ids[:args.top_k], scores[:args.top_k])]
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["frame_id", "similarity"])
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps({"query": args.query, "results": rows}, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Session error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
