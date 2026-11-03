"""Reject environments validated against a different application contract."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from environment_access import access
from environment_manager import EnvironmentManager, PROFILES, atomic_json
from environment_validation import ROOT, application_fingerprint, validation_is_current


class ValidationFreshnessTests(unittest.TestCase):
    def test_legacy_pass_is_not_enough_to_activate_or_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            interpreter = folder / "python"
            interpreter.touch()
            record = {"id":"old", "profile":"agent", "python":str(interpreter),
                      "validation":{"passed":True, "checks":["Old miniature agent"]}}
            manager = EnvironmentManager(folder)
            registry = {"environments":[record], "active":{"agent":"old"}}
            atomic_json(manager.registry, registry)
            self.assertFalse(manager.status()["environments"][0]["validation_current"])
            self.assertEqual(access("agent-launch", registry, {"agent":str(interpreter)}, PROFILES)["reason"],
                             "stale_validation")
            with self.assertRaisesRegex(ValueError, "Validate this environment again"):
                manager.activate("old")
            self.assertEqual(manager.read_registry(), registry)

    def test_source_and_dependency_changes_invalidate_the_agent_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = ["ui/environment_probe.py", "ui/environment_validation.py", "requirements-api.txt",
                     "requirements-assets.txt", "4_NLUI/requirements.txt", "mousegpt_config.py",
                     "mousegpt_embeddings.py"]
            paths += [str(path.relative_to(ROOT)) for pattern in ("4_NLUI/*.py", "4_NLUI/tools_function/*.py")
                      for path in ROOT.glob(pattern)]
            for name in paths:
                dest = folder / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            initial = application_fingerprint("agent", folder)
            self.assertEqual(initial, application_fingerprint("agent"))
            requirements = folder / "4_NLUI/requirements.txt"
            requirements.write_text(requirements.read_text() + "\nnew-library==1.0\n")
            self.assertNotEqual(initial, application_fingerprint("agent", folder))
            shutil.copyfile(ROOT / "4_NLUI/requirements.txt", requirements)
            code = folder / "4_NLUI/agent.py"
            code.write_text(code.read_text() + "\n# changed application\n")
            self.assertNotEqual(initial, application_fingerprint("agent", folder))

    def test_current_reports_are_profile_specific(self):
        report = {"passed":True, "application_fingerprint":application_fingerprint("agent")}
        self.assertTrue(validation_is_current("agent", report))
        self.assertFalse(validation_is_current("core", report))
        self.assertFalse(validation_is_current("agent", {**report, "passed":False}))

    def test_all_agent_platforms_share_the_current_application_dependencies(self):
        directory = ROOT / "ui/environments"
        matrix = json.loads((directory / "platform-matrix.json").read_text())
        recipes = [item for item in matrix["recipes"] if item["profile"] == "agent"]
        self.assertEqual(len(recipes), 5)
        for item in recipes:
            with self.subTest(platform=item["platform"]):
                contents = (directory / item["lock"]).read_bytes()
                self.assertEqual(hashlib.sha256(contents).hexdigest(), item["lock_sha256"])
                self.assertEqual(item["python"], "3.12.11")
                for required in (b"gradio==6.31.0", b"langchain-classic==1.0.8",
                                 b"langchain-openai==1.7.3", b"openai==3.26.1"):
                    self.assertIn(required, contents)


if __name__ == "__main__":
    unittest.main()
