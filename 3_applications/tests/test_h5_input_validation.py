"""Input validation before HDF5 writing; no HDF5 engine or API calls required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

missing = [name for name in ("numpy", "pandas", "tqdm") if importlib.util.find_spec(name) is None]
packing = None
if not missing:
    spec = importlib.util.spec_from_file_location("create_h5", Path(__file__).resolve().parents[1] / "create_h5.py")
    packing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packing)


@unittest.skipIf(bool(missing), f"Requires analysis dependencies: {', '.join(missing)}")
class H5InputValidationTest(unittest.TestCase):
    def test_malformed_embedding_is_preserved_and_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "00000008_des.csv"
            file.write_text("1,2,3\n")
            original = file.read_bytes()
            with self.assertRaises(ValueError):
                packing.load_data_matrix([str(file)])
            self.assertEqual(file.read_bytes(), original)

    def test_missing_description_never_silently_drops_a_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "00000008.json"
            file.write_text('{"choices": []}')
            original = file.read_bytes()
            with self.assertRaises(ValueError):
                packing.load_description_content([str(file)])
            self.assertEqual(file.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
