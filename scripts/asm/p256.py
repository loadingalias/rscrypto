"""Keep the generated P-256 verification assembly reproducible.

  p256.py check      fail if any generated `asm!` block differs from its generator (default)
  p256.py write      rewrite the generated blocks in place
  p256.py simulate   execute every generated kernel on edge and random inputs against Python
                     integers (`--cases N`, default 3000)

Generated: the x86-64 multiply, square, add, and sub in `src/auth/ecdsa_p256_verify.rs`.
Everything else in that file is written by hand. The x86-64 simulator is the one that checks the
P-384 kernels.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generated import check_or_write  # noqa: E402

import p256_x86_64  # noqa: E402
import p384  # noqa: E402
import p384_simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'src/auth/ecdsa_p256_verify.rs'
MASK = (1 << 64) - 1
P = 2**256 - 2**224 + 2**192 + 2**96 - 1
R_INVERSE = pow(2**256, -1, P)
SCRATCH = [f'w{i}' for i in range(8)] + ['lo', 'hi', 'rdx', 'a', 'b', 't', 'cs', 'm', 't1', 't3', 'd0', 'd1',
                                          'd2', 'd3']


def limbs(value):
    return [(value >> (64 * i)) & MASK for i in range(4)]


def number(values):
    return sum(v << (64 * i) for i, v in enumerate(values))


def operands(rng, cases):
    """Canonical edge values, values with all-ones or all-zero limbs, and random field elements."""
    edges = [0, 1, 2, P - 1, P - 2, 0xFFFF_FFFF, 0xFFFF_FFFF_0000_0001 << 192, 1 << 255, P - (1 << 200),
             2**256 - 2**224, (P - 1) // 2, number([MASK, 0, MASK, 0]), number([0, MASK, 0, MASK])]
    return [edge % P for edge in edges] + [rng.randrange(P) for _ in range(cases)]


def simulate(generators, cases, seed=256):
    """Return a list of failure strings: every kernel against Python integers."""
    rng = random.Random(seed)
    values = operands(rng, cases)
    failures = []

    def run(name, regs, memory, expected):
        lines, out = generators[name]()
        p384_simulate.run_x86(lines, regs, memory)
        if number([regs[placeholder.strip('{}')] for placeholder in out]) != expected:
            failures.append(f'{name}({memory})')

    def scratch():
        return {name: rng.getrandbits(64) for name in SCRATCH}

    for index, left in enumerate(values):
        right = values[(index * 7 + 3) % len(values)]
        run('montgomery_mul_bmi2_adx', scratch(), {'a': limbs(left), 'b': limbs(right)}, left * right * R_INVERSE % P)
        run('montgomery_square_bmi2_adx', scratch(), {'a': limbs(left)}, left * left * R_INVERSE % P)
        for name, expected in (('add_mod', (left + right) % P), ('sub_mod', (left - right) % P)):
            regs = scratch()
            regs.update({f'd{j}': limb for j, limb in enumerate(limbs(left))})
            run(name, regs, {'right': limbs(right)}, expected)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', choices=('check', 'write', 'simulate'), default='check')
    parser.add_argument('--cases', type=int, default=3000)
    args = parser.parse_args()

    if args.mode == 'simulate':
        failures = simulate(p256_x86_64.KERNELS, args.cases)
        for failure in failures[:20]:
            print(f'mismatch: {failure}', file=sys.stderr)
        if failures:
            return 1
        print(f'P-256 generated kernels match Python integers ({args.cases} cases per kernel)')
        return 0

    path = ROOT / SOURCE
    text = path.read_text(encoding='utf-8')
    new, stale = p384.regenerate(text, p256_x86_64.KERNELS)
    return check_or_write(args.mode, ROOT, 'scripts/asm/p256.py', [(path, text, new, stale)],
                          'P-256 generated assembly matches its generator')


if __name__ == '__main__':
    sys.exit(main())
