#!/usr/bin/env python3
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_solution_gfrun_regressions.py")
SPEC = importlib.util.spec_from_file_location("solution_gate", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class SolutionGfrunRegressionTest(unittest.TestCase):
    def test_inventory_is_exactly_twenty_one_unique_cases(self):
        self.assertEqual(21, len(gate.CASES))
        self.assertEqual(21, len({case.name for case in gate.CASES}))

    def test_nonzero_process_status_never_passes(self):
        ok, reason = gate.classify_gfrun(
            9, "Success to Reach the End of Benchmark! R2 = 0\n")
        self.assertFalse(ok)
        self.assertEqual("gfrun_exit_9", reason)

    def test_assertion_is_reported_separately(self):
        ok, reason = gate.classify_gfrun(
            1, "illegal instruction: ASSERTION FAILED: bad mask\n")
        self.assertFalse(ok)
        self.assertEqual("assertion", reason)

    def test_nonzero_finisher_never_passes(self):
        ok, reason = gate.classify_gfrun(
            0, "Success to Reach the End of Benchmark! R2 = 1\n")
        self.assertFalse(ok)
        self.assertEqual("nonzero_finisher_1", reason)

    def test_incorrect_finisher_spelling_never_passes(self):
        ok, reason = gate.classify_gfrun(
            0, "Reach End Benchmark! R2 = 0\n")
        self.assertFalse(ok)
        self.assertEqual("missing_finisher", reason)

    def test_missing_elf_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                gate.find_fresh_elf(Path(directory))

    def test_multiple_elfs_are_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.elf").write_bytes(b"one")
            (root / "two.elf").write_bytes(b"two")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                gate.find_fresh_elf(root)

    def test_timeout_never_passes(self):
        ok, reason = gate.classify_gfrun(
            0, "Success to Reach the End of Benchmark! R2 = 0\n", timed_out=True)
        self.assertFalse(ok)
        self.assertEqual("timeout", reason)

    def test_log_is_bounded_and_keeps_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "bounded.log"
            rc, timed_out = gate.run_logged(
                [sys.executable, "-c", "print('A' * 10000); print('END-MARKER')"],
                cwd=root, env=os.environ.copy(), log=log, timeout=5,
                max_log_bytes=1024)
            self.assertEqual(0, rc)
            self.assertFalse(timed_out)
            text = log.read_text()
            self.assertIn("output bytes omitted by bounded log", text)
            self.assertIn("END-MARKER", text)
            self.assertLess(log.stat().st_size, 2048)


if __name__ == "__main__":
    unittest.main()
