# Copyright 2026 ReMind authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_help_without_environment(self):
        for name in ["run_grpo.sh", "run_grpo_remind.sh"]:
            p = subprocess.run(["bash", str(ROOT / "examples/remind" / name), "--help"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("--dry-run", p.stdout)

    def test_missing_required_path_fails(self):
        env = {k: v for k, v in os.environ.items() if k not in {"MODEL_PATH", "TRAIN_DATA", "VAL_DATA"}}
        p = subprocess.run(["bash", str(ROOT / "examples/remind/run_grpo.sh")], env=env, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("MODEL_PATH", p.stderr)

    def test_launchers_forward_overrides_and_preserve_spaces(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake-python"
            fake.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = dict(os.environ, MODEL_PATH="/model with spaces", TRAIN_DATA="/train file.json", VAL_DATA="/val.json", PYTHON_BIN=str(fake))
            for name, config in [("run_grpo.sh", "vanilla_grpo"), ("run_grpo_remind.sh", "grpo_remind")]:
                p = subprocess.run(["bash", str(ROOT / "examples/remind" / name), "trainer.max_steps=7"], cwd=d, env=env, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                args = json.loads(p.stdout)
                self.assertEqual(args[:2], ["-m", "verl.trainer.main"])
                self.assertIn("worker.actor.model.model_path=/model with spaces", args)
                self.assertIn("data.train_files=/train file.json", args)
                self.assertEqual(args[-1], "trainer.max_steps=7")
                self.assertTrue(any(x.endswith(f"/{config}.yaml") for x in args))
                if name == "run_grpo_remind.sh":
                    self.assertIn("algorithm.review_queue_enabled=True", args)

    def test_dry_run_does_not_execute_python(self):
        env = dict(os.environ, MODEL_PATH="model", TRAIN_DATA="train", VAL_DATA="val", PYTHON_BIN="/does/not/exist")
        p = subprocess.run(["bash", str(ROOT / "examples/remind/run_grpo_remind.sh"), "--dry-run"], env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("verl.trainer.main", p.stdout)


if __name__ == "__main__":
    unittest.main()
