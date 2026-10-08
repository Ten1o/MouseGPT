"""Select cohort records from one or more image JSONL datasets."""

import argparse
import json
import os
from pathlib import Path
import re
import tempfile


def extract_records(inputs, output, groups=None):
    output = Path(output)
    inputs = [Path(path) for path in inputs]
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("The output must not overwrite an input dataset")
    selected = set(groups) if groups else None
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         delete=False) as target:
            temporary = Path(target.name)
            for source in inputs:
                with source.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        images = record.get("images")
                        if (not isinstance(images, list) or not images
                                or any(not isinstance(image, str) or not image for image in images)):
                            raise ValueError(f"{source}:{line_number}: expected a nonempty images list")
                        cohorts = set(re.findall(r"(?<![A-Za-z0-9])C\d+(?=M\d+|[^A-Za-z0-9]|$)", images[0]))
                        if selected is None or selected.intersection(cohorts):
                            record["images"] = [image if Path(image).is_absolute() else
                                                os.path.relpath((source.parent / image).resolve(), output.parent.resolve())
                                                for image in images]
                            target.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Source JSONL datasets")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", nargs="+", help="Cohort IDs such as C3 C4 C5; omit to include all records")
    args = parser.parse_args()
    print(f"Wrote {extract_records(args.inputs, args.output, args.groups)} records to {args.output}")


if __name__ == "__main__":
    main()
