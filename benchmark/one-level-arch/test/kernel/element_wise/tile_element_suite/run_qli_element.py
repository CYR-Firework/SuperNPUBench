#!/usr/bin/env python3
"""Compile and compare QLI element diagnostics on gfrun (not whole-app acceptance)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_memory import load_symbols
from check_disassembly import function_body
import qli_key_fixture
import qli_histogram_fixture

OPTIONS = ('enable-all-vector-as-tilereg=true', 'linxv5-enable-HL-Inst-Opt=true',
           'linxv5-enable-dim-opt=true', 'linxv5-enable-ldst-bridge=false',
           'linxv5-enable-continuous-mem-opt=true', 'linxv5-enable-tile-clock-hand=false',
           'linxv5-enable-simt-clock-hand=true', 'enable-misched=false',
           'linxv5-enable-generic-element=true')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_native(text, kind, profiles):
    for capacity, valid in sorted({(p[0], p[1]) for p in profiles}):
        if valid == 0:
            continue
        name = f'{"make_key" if kind == "key" else "histogram"}ILj{capacity}ELj{valid}E'
        body = function_body(text, name)
        assert 'MGATHER, U32' in body, name
        assert 'MSCATTER, U32' in body, name
        assert 0 <= body.rfind('B.ASSEMBLE') < body.index('MSCATTER, U32'), name
        if valid < capacity:
            for block in re.split(r'BSTART\.', body)[1:]:
                if block.startswith('TLSU\tMGATHER, U32'):
                    assert 'ExecMaskPresent' in block, name
        if kind == 'histogram':
            assert 'MGATHER.ADD, U32' in body, name
    assert 'THISTOGRAM' not in text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('key', 'histogram'), required=True)
    for option in ('compiler-bin', 'sysroot', 'resource-dir', 'api-include', 'gfrun', 'out'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    suite = Path(__file__).resolve().parent
    bench = suite.parents[3]
    fixture = qli_key_fixture if args.kind == 'key' else qli_histogram_fixture
    cases = list(fixture.profiles())
    fixture.generate(out / 'fixture')
    tools = args.compiler_bin.resolve()
    binaries = [tools / name for name in ('clang++', 'llc', 'ld.lld', 'llvm-objdump')]
    binaries.append(args.gfrun.resolve())
    before = {str(path): sha(path) for path in binaries}
    inputs = [suite / f'qli_{args.kind}.cpp', Path(fixture.__file__),
              bench / 'kernels/solution/qli/qli_element.hpp']
    before.update({str(path): sha(path) for path in inputs})
    target = ['--target=linx64v5-unknown-linux-musl', '--sysroot=' + str(args.sysroot),
              '-resource-dir=' + str(args.resource_dir)]
    commands = []

    def run(command, output):
        commands.append(list(map(str, command)))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with output.open('w') as log:
            subprocess.run(commands[-1], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)

    compile_flags = ['-c', '-mlxbc', '-fenable-matrix', '-O2', '-std=c++20',
                     '-fno-vectorize', '-fno-slp-vectorize', '-Werror=unknown-pragmas',
                     '-D__linx', '-DENABLE_TENSOR_INSTR', '-I' + str(args.api_include),
                     '-I' + str(bench / 'kernels'), '-I' + str(bench / 'test/common/src'),
                     '-I' + str(out / 'fixture')]
    for option in OPTIONS:
        compile_flags += ['-mllvm', '-' + option]
    run([tools / 'clang++', *target, *compile_flags, '-S', '-emit-llvm', inputs[0],
         '-o', out / 'kernel.ll'], out / 'frontend.log')
    run([tools / 'llc', '-mtriple=linx64v5-unknown-linux-musl', '-O2', '-filetype=obj',
         *['-' + option for option in OPTIONS], out / 'kernel.ll', '-o', out / 'kernel.o'],
        out / 'backend.log')
    run([tools / 'clang++', *target, '-nostartfiles', '-fuse-ld=lld',
         bench / 'test/common/_start.s', out / 'kernel.o', '-o', out / 'kernel.elf'],
        out / 'link.log')
    run([tools / 'llvm-objdump', '-d', '--disassembler-options=no-tile-macros',
         out / 'kernel.elf'], out / 'kernel.diss')
    check_native((out / 'kernel.diss').read_text(), args.kind, cases)
    run([tools / 'llvm-objdump', '-t', out / 'kernel.elf'], out / 'symbols.txt')
    symbols = load_symbols(out / 'symbols.txt')
    expected = {p.stem: p.read_bytes() for p in (out / 'fixture/golden').glob('*.bin')}
    for name, raw in expected.items():
        assert symbols[name][1] == len(raw), name
    for stem in json.loads((out / 'fixture/buffers.json').read_text()):
        before_guard, body, after_guard = [symbols[stem + tail]
                                          for tail in ('_before', '', '_after')]
        assert before_guard[0] + before_guard[1] == body[0], stem
        assert body[0] + body[1] == after_guard[0], stem
    start = min(symbols[name][0] for name in expected)
    end = max(symbols[name][0] + len(raw) for name, raw in expected.items())
    run([args.gfrun, '--pto059-execution-mask-dev', '-s', 'softcore.multiThreadNum=1',
         '--dump-force', '--dump-memory', f'{start:#x}:{end-start:#x}:{out}/memory.bin',
         '-f', out / 'kernel.elf'], out / 'gfrun.log')
    assert 'Suaccelss to Reach the End of Benchmark! R2 = 0' in (out / 'gfrun.log').read_text()
    memory = (out / 'memory.bin').read_bytes()
    assert len(memory) == end - start
    for name, raw in expected.items():
        offset = symbols[name][0] - start
        actual = memory[offset:offset + len(raw)]
        if actual != raw:
            first = next(i for i, (a, b) in enumerate(zip(actual, raw)) if a != b)
            raise AssertionError(f'{name}: first byte mismatch at {first}')
    assert all(sha(Path(path)) == digest for path, digest in before.items()), 'inputs changed during run'
    report = {'kind': args.kind, 'profiles': len(cases), 'segments': len(expected),
              'bytes': sum(map(len, expected.values())), 'pass': True,
              'scope': 'diagnostic only; not frozen whole-application acceptance',
              'input_sha256': before, 'elf_sha256': sha(out / 'kernel.elf')}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{args.kind}: {len(cases)} profiles, {len(expected)} exact segments PASS')


if __name__ == '__main__':
    main()
