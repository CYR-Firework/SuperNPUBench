#!/usr/bin/env python3
"""Corruption tests for paired-atomic and exact FP32 observation contracts."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MASK32 = 0xFFFFFFFF


class AtomicPairTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.golden = self.root / "golden"
        subprocess.run(
            ["python3", str(HERE / "generate_case.py"), "--case",
             "generic_atomic_cfg_i32", "--out", str(self.golden)], check=True,
        )
        manifest = json.loads((self.golden / "manifest.json").read_text())
        names_and_sizes = [
            ("generic_atomic_cfg_i32_indices", 512),
            ("generic_atomic_cfg_i32_deltas", 512),
            ("generic_atomic_cfg_i32_gates", 512),
            ("generic_atomic_cfg_i32_histogram", 128),
            ("generic_atomic_cfg_i32_first", 512),
            ("generic_atomic_cfg_i32_second", 512),
            ("generic_atomic_cfg_i32_combined", 512),
            ("generic_atomic_cfg_i32_poison", 512),
            ("generic_atomic_cfg_i32_status", 32),
        ]
        address = 0x1000
        self.symbols: dict[str, tuple[int, int]] = {}
        for name, size in names_and_sizes:
            self.symbols[name] = (address, size)
            address += (size + 31) & ~31
        self.memory = bytearray(address - 0x1000)
        for item in manifest["segments"]:
            if item["kind"] == "exact":
                self._store(item["symbol"], (self.golden / item["file"]).read_bytes())

        indices = list(struct.unpack("<128i", (self.golden / "indices_s32.bin").read_bytes()))
        first_adds = self._read_u32("first_adds_u32.bin")
        second_adds = self._read_u32("second_adds_u32.bin")
        initial = self._read_u32("initial_histogram_u32.bin")
        first, second, combined = [11] * 128, [13] * 128, [17] * 128
        for bin_index in range(32):
            current = initial[bin_index]
            elements = [
                element for element in reversed(range(125))
                if indices[element] + 16 == bin_index
            ]
            # Reverse source element order and interleave two elements as
            # first(A), first(B), second(B), second(A). This is deliberately
            # different from serialized element order while retaining the
            # required source order inside each element.
            for begin in range(0, len(elements), 2):
                pair = elements[begin:begin + 2]
                old_first: dict[int, int] = {}
                for element in pair:
                    old_first[element] = current
                    current = (current + first_adds[element]) & MASK32
                old_second: dict[int, int] = {}
                for element in reversed(pair):
                    old_second[element] = current
                    current = (current + second_adds[element]) & MASK32
                for element in pair:
                    first[element] = (old_first[element] + 11) & MASK32
                    second[element] = (old_second[element] + 13) & MASK32
                    combined[element] = (
                        (old_first[element] ^
                         ((old_second[element] + 0x13579BDF) & MASK32)) + 17
                    ) & MASK32
        self._store("generic_atomic_cfg_i32_first", struct.pack("<128I", *first))
        self._store("generic_atomic_cfg_i32_second", struct.pack("<128I", *second))
        self._store("generic_atomic_cfg_i32_combined", struct.pack("<128I", *combined))
        self.symbol_file = self.root / "symbols.txt"
        self.symbol_file.write_text("".join(
            f"{address:016x} g O .data {size:08x} {name}\n"
            for name, (address, size) in self.symbols.items()
        ))
        self.dump = self.root / "memory.bin"
        self.dump.write_bytes(self.memory)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _read_u32(self, filename: str) -> list[int]:
        data = (self.golden / filename).read_bytes()
        return list(struct.unpack(f"<{len(data) // 4}I", data))

    def _store(self, symbol: str, data: bytes) -> None:
        address, size = self.symbols[symbol]
        self.assertEqual(len(data), size)
        offset = address - 0x1000
        self.memory[offset:offset + size] = data

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(HERE / "compare_memory.py"),
             "--symbols", str(self.symbol_file), "--golden", str(self.golden),
             "--dump", str(self.dump), "--dump-base", "0x1000"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_accepts_source_order_without_global_element_order(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("order-independent paired atomic trace", result.stdout)

    def test_rejects_first_before_second_inversion(self) -> None:
        first_address, _ = self.symbols["generic_atomic_cfg_i32_first"]
        second_address, _ = self.symbols["generic_atomic_cfg_i32_second"]
        combined_address, _ = self.symbols["generic_atomic_cfg_i32_combined"]
        element = 0
        first_stored = struct.unpack_from(
            "<I", self.memory, first_address - 0x1000 + element * 4
        )[0]
        second_offset = second_address - 0x1000 + element * 4
        second_stored = struct.unpack_from("<I", self.memory, second_offset)[0]
        old_first = (first_stored - 11) & MASK32
        old_second = (second_stored - 13) & MASK32
        struct.pack_into(
            "<I", self.memory, first_address - 0x1000 + element * 4,
            (old_second + 11) & MASK32,
        )
        struct.pack_into(
            "<I", self.memory, second_offset, (old_first + 13) & MASK32
        )
        old_first, old_second = old_second, old_first
        combined = (
            (old_first ^ ((old_second + 0x13579BDF) & MASK32)) + 17
        ) & MASK32
        struct.pack_into(
            "<I", self.memory, combined_address - 0x1000 + element * 4, combined
        )
        self.dump.write_bytes(self.memory)
        result = self._run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("cannot form a modulo-2^32 atomic order", result.stdout)

    def test_rejects_inactive_seed_corruption(self) -> None:
        first_address, _ = self.symbols["generic_atomic_cfg_i32_first"]
        struct.pack_into(
            "<I", self.memory, first_address - 0x1000 + 127 * 4, 12
        )
        self.dump.write_bytes(self.memory)
        result = self._run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("inactive element 127 did not retain", result.stdout)


class OriginalF32BitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture(self, strong: bool) -> tuple[Path, Path, bytearray]:
        golden = self.root / ("strong" if strong else "original")
        subprocess.run(["python3", str(HERE / "generate_case.py"), "--case",
                        "elementwise_if_for_strong" if strong else "elementwise_if_for",
                        "--out", str(golden)], check=True)
        manifest = json.loads((golden / "manifest.json").read_text())
        memory = bytearray()
        symbols = self.root / "symbols.txt"
        symbol_lines = []
        for item in manifest["segments"]:
            data = (golden / item["file"]).read_bytes()
            symbol_lines.append(f"{0x1000 + len(memory):016x} g O .data {len(data):08x} {item['symbol']}\n")
            memory.extend(data)
        symbols.write_text("".join(symbol_lines))
        return golden, symbols, memory

    def _compare(self, golden: Path, symbols: Path, memory: bytearray) -> subprocess.CompletedProcess[str]:
        dump = self.root / "memory.bin"
        dump.write_bytes(memory)
        return subprocess.run(["python3", str(HERE / "compare_memory.py"),
                               "--symbols", str(symbols), "--golden", str(golden),
                               "--dump", str(dump), "--dump-base", "0x1000"],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_original_retains_three_512_byte_zero_buffers(self) -> None:
        golden, symbols, memory = self._fixture(False)
        self.assertEqual(len(memory), 1536)
        self.assertEqual(memory, bytes(1536))
        result = self._compare(golden, symbols, memory)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_strong_rawbits_and_guards_pass(self) -> None:
        golden, symbols, memory = self._fixture(True)
        self.assertEqual(len(memory), 1664)
        self.assertEqual(memory[1024:1088], bytes([0x5A]) * 64)
        self.assertEqual(memory[-64:], bytes([0x5A]) * 64)
        self.assertEqual(struct.unpack_from("<I", memory, 1088 + 4)[0], 0x80000000)
        result = self._compare(golden, symbols, memory)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_rejects_signed_zero_normalization(self) -> None:
        golden, symbols, memory = self._fixture(True)
        struct.pack_into("<I", memory, 1088 + 4, 0)
        result = self._compare(golden, symbols, memory)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("expected=0x80000000", result.stdout)

    def test_rejects_input_or_output_guard_corruption(self) -> None:
        golden, symbols, memory = self._fixture(True)
        for offset, symbol in ((0, "lhs_storage"), (512, "rhs_storage"),
                               (1024, "out_storage"), (len(memory) - 1, "out_storage")):
            with self.subTest(offset=offset):
                corrupted = memory.copy()
                corrupted[offset] ^= 1
                result = self._compare(golden, symbols, corrupted)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(symbol + ": mismatch", result.stdout)

    def test_rejects_short_output_extent(self) -> None:
        golden, symbols, memory = self._fixture(True)
        result = self._compare(golden, symbols, memory[:-1])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("dump does not cover out_storage", result.stdout)


if __name__ == "__main__":
    unittest.main()
