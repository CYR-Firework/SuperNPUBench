#!/usr/bin/env python3
"""Native checker must not compose evidence across unrelated functions."""

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "check_disassembly", HERE / "check_disassembly.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckerFunctionScopeTest(unittest.TestCase):
    def test_exact_extractors_do_not_select_prefix_clone(self):
        disassembly = """
00000000 <kernel.clone>:
 CLONE_ONLY
00000010 <kernel>:
 EXACT_ONLY
"""
        ir = """
define void @kernel.clone() {
clone.entry:
  ret void
}
define void @kernel() {
exact.entry:
  ret void
}
"""
        self.assertIn(
            "EXACT_ONLY", CHECKER.function_body_exact(disassembly, "kernel"))
        self.assertNotIn(
            "CLONE_ONLY", CHECKER.function_body_exact(disassembly, "kernel"))
        self.assertIn("exact.entry", CHECKER.ir_function_body_exact(ir, "kernel"))
        self.assertNotIn(
            "clone.entry", CHECKER.ir_function_body_exact(ir, "kernel"))

    def test_indexed_snapshot_rejects_corrupt_tail_helper(self):
        disassembly = """
00000000 <transpose_f16_element_tile_full>:
 BSTART.TLSU MGATHER, FP16
 B.IOR ExecMaskPresent
 BSTART.TLSU TMOV, FP16
 B.ASSEMBLE
 B.SUBVIEW
 BSTART.TEPL TLEA, U32
 BSTART.TLSU TSTORE, FP16
 B.IOR ExecMaskPresent
00000010 <transpose_f16_element_tile_tail>:
 BSTART.TLSU MGATHER, FP16
 B.IOR ExecMaskPresent
 BSTART.TLSU TMOV, FP16
 B.ASSEMBLE
 B.SUBVIEW
 BSTART.TEPL TLEA, U32
"""
        ir = """
define void @transpose_f16_element_tile_full() {
  %v = call <64 x half> @llvm.linx.experimental.ew.mgather()
  ret void
}
define void @transpose_f16_element_tile_tail() {
  %v = call <64 x half> @llvm.linx.experimental.ew.mgather()
  ret void
}
"""
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_indexed_snapshot(disassembly, ir)

    def test_inline_broadcast_rejects_split_function_evidence(self):
        disassembly = """
00000000 <broadcast_s32_element_tile_load>:
 BSTART.TLSU MGATHER, U32
 BSTART.TEPL TLEA, U32
 B.ASSEMBLE
 B.IOT ->t<2KB>
00000010 <broadcast_s32_element_tile_store>:
 BSTART.TLSU TSTORE, S32
"""
        ir = """
define void @broadcast_s32_element_tile_load() { ret void }
define void @broadcast_s32_element_tile_store() { ret void }
"""
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_broadcast_inline(disassembly, ir)

    def test_fourpe_broadcast_rejects_split_function_evidence(self):
        disassembly = """
00000000 <unrelated_load>:
 BSTART.TLSU TLOAD, FP32
 C.B.DIMI 128, ->lb0
 C.B.DIMI 128, ->lb2
 B.IOT ->t<512B>
00000010 <unrelated_store>:
 BSTART.TLSU TSTORE, FP32
"""
        ir = "define void @unrelated(<128 x i32> %x) { ret void }\n"
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_broadcast_fourpe(disassembly, ir)

    def test_leaf_checker_rejects_split_function_evidence(self):
        disassembly = """
00000000 <process_gather_tile_load>:
 BSTART.TLSU MGATHER, FP32
 B.IOR ExecMaskPresent
 BSTART.TEPL TLEA, U32
00000010 <process_gather_tile_store>:
 BSTART.TLSU MSCATTER, FP32
 B.IOR ExecMaskPresent
"""
        ir = """
define void @process_gather_tile_load() { ret void }
define void @process_gather_tile_store() { ret void }
"""
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_leaf_gather_scatter(
                disassembly, "gather_v2_f32_0_original", ir)


if __name__ == "__main__":
    unittest.main()
