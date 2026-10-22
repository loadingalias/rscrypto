"""Keep the generated paired RSA assembly reproducible.

  rsa_pair.py check      fail if a generated `asm!` block differs from its generator (default)
  rsa_pair.py write      rewrite the generated blocks in place
  rsa_pair.py simulate   execute both kernels on edge and random inputs against Python integers
                         (`--cases N` per shape, default 24)

Generated: `mul_kernel` and `square_kernel` in `src/auth/rsa_aarch64_pair.rs`. Everything else in
that file is written by hand.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p384  # noqa: E402
import rsa_pair_aarch64  # noqa: E402
import rsa_pair_simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'src/auth/rsa_aarch64_pair.rs'


def sources():
    path = ROOT / SOURCE
    text = path.read_text()
    new, stale = p384.regenerate(text, rsa_pair_aarch64.KERNELS)
    yield path, text, new, stale


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', choices=('check', 'write', 'simulate'), default='check')
    parser.add_argument('--cases', type=int, default=24)
    args = parser.parse_args()

    if args.mode == 'simulate':
        failures = rsa_pair_simulate.check_kernels(rsa_pair_aarch64.KERNELS, args.cases, random.Random(1))
        for failure in failures[:20]:
            print(f'mismatch: {failure}', file=sys.stderr)
        if failures:
            return 1
        print(f'Paired RSA kernels match Python integers ({args.cases} cases per shape)')
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
        print('Run `scripts/lib/python.sh scripts/asm/rsa_pair.py write` after changing a generator.', file=sys.stderr)
        return 1
    if args.mode == 'check':
        print('Paired RSA assembly matches its generator')
    return 0


if __name__ == '__main__':
    sys.exit(main())
