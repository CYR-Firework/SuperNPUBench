#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gfrun_dump import MAX_DUMP_BYTES, assemble_dump, dump_specs, plan_dump


class DumpPlanTest(unittest.TestCase):
    def test_existing_single_range_and_exact_boundary(self):
        for size in (1, 4, MAX_DUMP_BYTES):
            chunks = plan_dump(f"0x1000:{size}", Path("out.mem"))
            self.assertEqual(chunks, [{"address": 0x1000, "size": size,
                                       "path": "out.mem"}])
            self.assertEqual(dump_specs(chunks), [f"0x1000:{size:#x}:out.mem"])

    def test_full_original_range_is_contiguous_and_bounded(self):
        address, size = 0x14000, 0x13890f8
        chunks = plan_dump(f"{address}:{size}", Path("gfrun.mem"))
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk["address"], address)
            self.assertLessEqual(chunk["size"], MAX_DUMP_BYTES)
            address += chunk["size"]
        self.assertEqual(sum(c["size"] for c in chunks), size)
        self.assertEqual(len({c["path"] for c in chunks}), len(chunks))

    def test_rejects_invalid_ranges(self):
        for value in ("0:0", "-1:1", "0:-1", "0xffffffffffffffff:1", "bad:1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan_dump(value, Path("out.mem"))

    def test_assembly_preserves_bytes_and_all_source_chunks(self):
        with tempfile.TemporaryDirectory() as directory, patch("gfrun_dump.MAX_DUMP_BYTES", 3):
            output = Path(directory) / "gfrun.mem"
            chunks = plan_dump("0x1000:8", output)
            expected = bytes(range(8))
            for chunk in chunks:
                start = chunk["address"] - 0x1000
                Path(chunk["path"]).write_bytes(expected[start:start + chunk["size"]])
            assemble_dump(chunks, output)
            self.assertEqual(output.read_bytes(), expected)
            self.assertTrue(all(Path(c["path"]).exists() for c in chunks))

    def test_missing_short_or_long_chunk_never_publishes_result(self):
        for length in (None, 1, 3):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as directory, patch("gfrun_dump.MAX_DUMP_BYTES", 2):
                output = Path(directory) / "gfrun.mem"
                chunks = plan_dump("0:4", output)
                Path(chunks[0]["path"]).write_bytes(b"ab")
                if length is not None:
                    Path(chunks[1]["path"]).write_bytes(b"x" * length)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    assemble_dump(chunks, output)
                self.assertFalse(output.exists())

    def test_stale_temporary_is_not_overwritten_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory, patch("gfrun_dump.MAX_DUMP_BYTES", 2):
            output = Path(directory) / "gfrun.mem"
            chunks = plan_dump("0:4", output)
            for chunk in chunks:
                Path(chunk["path"]).write_bytes(b"ab")
            temporary = output.with_name(output.name + ".assembling")
            temporary.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                assemble_dump(chunks, output)
            self.assertEqual(temporary.read_bytes(), b"preserve")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
