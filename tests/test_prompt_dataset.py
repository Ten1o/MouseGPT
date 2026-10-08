"""Regression tests for portable JSONL filtering and safe output handling."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "2_VLM/prompt_test/extract_jsonl.py"
SPEC = importlib.util.spec_from_file_location("extract_prompt_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExtractRecordsTests(unittest.TestCase):
    def test_cohort_filter_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset.jsonl"
            records = [{"images": [f"images/240507_C{cohort}M1/00000012.jpg"],
                        "query": "Describe posture", "response": "Example"} for cohort in (1, 10)]
            source.write_text("\n".join(json.dumps(record) for record in records))
            output = root / "subset/records.jsonl"
            self.assertEqual(MODULE.extract_records([source], output, ["C1"]), 1)
            selected = json.loads(output.read_text())
            self.assertEqual(selected["query"], records[0]["query"])
            self.assertEqual((output.parent / selected["images"][0]).resolve(),
                             (source.parent / records[0]["images"][0]).resolve())

    def test_invalid_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.jsonl", root / "output.jsonl"
            source.write_text('{"images": ["C3/image.jpg"]}\n{"images": []}\n')
            output.write_text("existing results\n")
            with self.assertRaisesRegex(ValueError, "images list"):
                MODULE.extract_records([source], output)
            self.assertEqual(output.read_text(), "existing results\n")
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_input_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text('{"images": ["C3/image.jpg"]}\n')
            with self.assertRaisesRegex(ValueError, "overwrite an input"):
                MODULE.extract_records([source], source)


if __name__ == "__main__":
    unittest.main()
