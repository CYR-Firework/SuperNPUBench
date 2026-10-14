#!/usr/bin/env python3
import struct
import unittest
from qli_topk_fixture import compare_storage, profiles, scores, storage_golden


class QLITopKGoldenTest(unittest.TestCase):
    def test_original_dimensions_and_positive_unique_scores(self):
        self.assertEqual(profiles(), ((64, 128, 128), (4, 2048, 512),
                                      (4, 8192, 512), (4, 2080, 512)))
        for sq, skv, k in profiles():
            bits = scores(sq, skv)
            for row in range(sq):
                values = bits[row * skv:(row + 1) * skv]
                self.assertEqual(len(set(values)), skv)
                self.assertTrue(all(0x3f000000 <= value < 0x3f800000 for value in values))
            self.assertEqual(len(storage_golden(sq, skv, k)),
                             sq * k + 2048 + sq * skv + 256 + 32)

    def test_unordered_indices_accepted(self):
        expected = storage_golden(1, 128, 8)
        actual = expected[:8][::-1] + expected[8:]
        pack = lambda words: struct.pack(f'<{len(words)}I', *words)
        compare_storage(pack(actual), pack(expected), 1, 128, 8)

    def test_duplicate_index_and_scratch_corruption_rejected(self):
        expected = storage_golden(1, 128, 8)
        pack = lambda words: struct.pack(f'<{len(words)}I', *words)
        for changed in (1, 8, len(expected) - 1):
            actual = list(expected)
            actual[changed] = actual[0] if changed == 1 else actual[changed] ^ 1
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                compare_storage(pack(actual), pack(expected), 1, 128, 8)


if __name__ == '__main__':
    unittest.main()
