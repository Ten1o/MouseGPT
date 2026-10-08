import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("assets", Path(__file__).parents[1] / "scripts/download_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def test_builder_rejects_incomplete_source_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "source"
            data = source / "2_VLM/prompt_test/data"
            data.mkdir(parents=True)
            (data / "test_data.jsonl").write_text(json.dumps({"images": ["a.jpg"]}) + "\n")
            result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "scripts/build_example_bundles.py"),
                "--source", str(source), "--output", str(folder / "out"),
                "--manifest", str(folder / "manifest.json")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Expected 10 VLM records", result.stderr)
            self.assertFalse((folder / "out").exists())

    def bundle(self, folder, name="imgs/frame.jpg"):
        path = folder / "sample.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(name, b"sample")
        info = {"id": "demo", "repo_id": "owner/data", "revision": "fixed",
                "sha256": assets.sha256(path), "file_count": 1, "uncompressed_bytes": 6}
        return path, info

    def test_install_verify_and_detect_changed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            archive, info = self.bundle(folder)
            target = folder / "data"
            self.assertEqual(assets.install_archive(archive, target, info), 1)
            self.assertEqual(assets.verify_install(target, info), 1)
            (target / "imgs/frame.jpg").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "modified"):
                assets.verify_install(target, info)

    def test_checksum_failure_leaves_no_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            archive, info = self.bundle(folder)
            info["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum"):
                assets.install_archive(archive, folder / "data", info)
            self.assertFalse((folder / "data").exists())

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            archive, info = self.bundle(folder, "../outside.txt")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                assets.install_archive(archive, folder / "data", info)
            self.assertFalse((folder / "outside.txt").exists())

    def test_existing_user_folder_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            archive, info = self.bundle(folder)
            target = folder / "data"
            target.mkdir()
            (target / "user.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "receipt"):
                assets.install_archive(archive, target, info)
            self.assertEqual((target / "user.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
