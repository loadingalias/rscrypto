"""Keep the generated P-384 assembly reproducible.

  p384.py check      fail if any generated `asm!` block differs from its generator (default)
  p384.py write      rewrite the generated blocks in place
  p384.py simulate   execute every generated kernel on edge and random inputs against Python
                     integers (`--cases N`, default 3000)

Generated: the x86-64 multiply, square, small multiple, add, sub, and in-place doubling in
`src/auth/p384_x86_64.rs`, and the AArch64 multiply, square, and small multiple in
`src/auth/p384_aarch64.rs`. Everything else in those files is written by hand.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p384_aarch64  # noqa: E402
import p384_simulate  # noqa: E402
import p384_x86_64  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'src/auth/p384_x86_64.rs': p384_x86_64.KERNELS,
    'src/auth/p384_aarch64.rs': p384_aarch64.KERNELS,
}
TEMPLATE_LINE = re.compile(r'^(\s*)"((?:[^"\\]|\\.)*)",$')


def template_span(text, function):
    """Line range of the template strings that open `function`'s `asm!` block."""
    lines = text.split('\n')
    start = next(i for i, line in enumerate(lines) if f'fn {function}(' in line)
    end = next(i for i in range(start, len(lines)) if lines[i] == '}')
    opening = next(i for i in range(start, end) if lines[i].strip() == 'core::arch::asm!(')
    first = opening + 1
    last = first
    while TEMPLATE_LINE.match(lines[last]):
        last += 1
    if last == first:
        raise ValueError(f'{function}: no asm! template lines')
    return lines, first, last


def regenerate(text, kernels):
    """Return `text` with every generated block replaced, and the names that changed."""
    stale = []
    for function, generator in kernels.items():
        lines, first, last = template_span(text, function)
        indent = TEMPLATE_LINE.match(lines[first]).group(1)
        generated = [f'{indent}"{line}",' for line in generator()[0]]
        if lines[first:last] != generated:
            stale.append(function)
            lines[first:last] = generated
        text = '\n'.join(lines)
    return text, stale


def regenerate_double(text):
    head = text[:text.index(p384_x86_64.POINT_DOUBLE_MARKER)]
    new = head + p384_x86_64.point_double_rust()
    return new, ([] if new == text else ['point_double_bmi2_adx'])


def sources():
    for relative, kernels in SOURCES.items():
        path = ROOT / relative
        text = path.read_text()
        new, stale = regenerate(text, kernels)
        if relative.endswith('p384_x86_64.rs'):
            new, stale_double = regenerate_double(new)
            stale += stale_double
        yield path, text, new, stale


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', choices=('check', 'write', 'simulate'), default='check')
    parser.add_argument('--cases', type=int, default=3000)
    args = parser.parse_args()

    if args.mode == 'simulate':
        failures = p384_simulate.simulate(p384_x86_64, p384_aarch64, args.cases)
        for failure in failures[:20]:
            print(f'mismatch: {failure}', file=sys.stderr)
        if failures:
            return 1
        print(f'P-384 generated kernels match Python integers ({args.cases} cases per kernel)')
        return 0

    stale_total = []
    for path, text, new, stale in sources():
        relative = path.relative_to(ROOT)
        if args.mode == 'write' and new != text:
            path.write_text(new)
            print(f'rewrote {relative}: {", ".join(stale)}')
        stale_total += [f'{relative}: {name}' for name in stale]
    if args.mode == 'check' and stale_total:
        for entry in stale_total:
            print(f'stale generated assembly: {entry}', file=sys.stderr)
        print('Run `scripts/lib/python.sh scripts/asm/p384.py write` after changing a generator.', file=sys.stderr)
        return 1
    if args.mode == 'check':
        print('P-384 generated assembly matches its generators')
    return 0


if __name__ == '__main__':
    sys.exit(main())
