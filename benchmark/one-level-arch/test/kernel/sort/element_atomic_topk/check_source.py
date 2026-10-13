#!/usr/bin/env python3
"""Keep TopK arithmetic expressed in scalar C++ under element pragmas."""

import re
from pathlib import Path


def validate(source: str) -> None:
    # Ignore comments so documentation can name the APIs being replaced.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    if re.search(r"\b(?:TADDS|TSHRS|TANDS)\s*\(", code):
        raise ValueError("TopK must express element arithmetic as scalar C++")
    if len(re.findall(r"^\s*#pragma pto element for\s*$", code, re.M)) != 2:
        raise ValueError("TopK must retain both element histogram loops")


def main() -> None:
    kernel = (Path(__file__).resolve().parents[4]
              / "kernels/single_thread/sort/element_atomic_topk.hpp")
    source = kernel.read_text(encoding="utf-8")
    validate(source)
    mutations = [source + f"\n{opcode}(values, view, 0);"
                 for opcode in ("TADDS", "TSHRS", "TANDS")]
    mutations.append(source.replace("#pragma pto element for", "", 1))
    for mutation in mutations:
        try:
            validate(mutation)
        except ValueError:
            continue
        raise AssertionError("source regression checker accepted a mutation")
    print("TopK scalar source check and mutation canaries PASS")


if __name__ == "__main__":
    main()
