"""Analyze explicitly selected cohort summaries with optional contextual text."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousegpt_config import load_config, sdk_client


def load_cohorts(path):
    groups = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("knowledge_group", "test_group"):
        rows = groups.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key} must be a nonempty list of [session ID, label] pairs.")
        if any(not isinstance(row, list) or len(row) != 2 or not all(isinstance(value, str) and value.strip() for value in row) for row in rows):
            raise ValueError(f"Invalid rows in {key}.")
        ids = [row[0] for row in rows]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids):
            raise ValueError(f"Duplicate or unsafe session IDs in {key}.")
    if set(row[0] for row in groups["knowledge_group"]) & set(row[0] for row in groups["test_group"]):
        raise ValueError("Knowledge and test sessions must be disjoint.")
    return groups


def build_instructions(context=None):
    instructions = """You analyze supplied mouse behavioral summaries and predict treatment labels.
Compute group means and standard deviations from the knowledge group only. Use those
training statistics to standardize both knowledge and test features, explicitly handle
zero-variance features, and calculate Euclidean distances. Do not use test labels to fit
or select the classifier. Show calculations and distinguish computed results from interpretation.
Do not invent statistical tests, p-values, or causal findings from narrative context.
"""
    if context is not None:
        if not context.strip():
            raise ValueError("The selected context report is empty.")
        instructions += "\nUser-supplied contextual narrative (not independently established statistical evidence):\n" + context
    return instructions


def request_prediction(client, instructions, prompt, file_id, *, model="gpt-4o", timeout=600):
    """Analyze the attached CSV using Responses Code Interpreter with a finite timeout."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Prediction timeout must be finite and positive.")
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        tools=[{"type": "code_interpreter", "container": {
            "type": "auto", "file_ids": [file_id],
        }}],
        timeout=timeout,
    )
    if response.status != "completed":
        raise RuntimeError(f"Analysis response did not complete: {response.status}")
    if not isinstance(response.output_text, str) or not response.output_text.strip():
        raise RuntimeError("Completed analysis returned no text.")
    return response


def main(args):
    if not math.isfinite(args.max_wait) or args.max_wait <= 0:
        raise ValueError("--max_wait must be positive.")
    csv = Path(args.data_csv)
    if not csv.is_file():
        raise FileNotFoundError(csv)
    groups = load_cohorts(args.groups_json)
    context_path = Path(args.context_report) if args.context_report else None
    context = context_path.read_text(encoding="utf-8") if context_path else None
    instructions = build_instructions(context)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config("PHENOTYPE", overrides={"model": args.model} if args.model else None)
    client = sdk_client(config, max_retries=0, timeout=60)
    with csv.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="user_data")
    provenance = {
        "data_csv": str(csv), "data_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "context_report": str(context_path) if context_path else None,
        "context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest() if context_path else None,
        "groups": groups, "model": config.model,
    }
    (output / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for session, label in groups["test_group"]:
        prompt = f"Knowledge group labels: {groups['knowledge_group']}\nPredict treatment for session {session} using the CSV. Show feature standardization, distances, and the predicted treatment and dose."
        response = request_prediction(
            client, instructions, prompt, uploaded.id, model=config.model, timeout=args.max_wait,
        )
        (output / f"{session}.txt").write_text(
            f"Test session: {session}\nHeld-out label (not supplied in the prediction prompt): {label}\n\n" + response.output_text, encoding="utf-8",
        )
        print(f"Processed {session}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", required=True, help="Cohort feature CSV uploaded for analysis.")
    parser.add_argument("--groups_json", required=True, help="Explicit knowledge/test session IDs and labels.")
    parser.add_argument("--context_report", help="Optional contextual narrative; no report is loaded by default.")
    parser.add_argument("--output", default="output/phenotype")
    parser.add_argument("--model", help="Optional model override from .env.local")
    parser.add_argument("--max_wait", type=float, default=600, help="Prediction HTTP request timeout in seconds")
    main(parser.parse_args())
