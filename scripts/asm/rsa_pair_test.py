"""Regression and negative controls for the paired RSA assembly generator and simulator."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p384  # noqa: E402
import rsa_pair  # noqa: E402
import rsa_pair_aarch64  # noqa: E402
import rsa_pair_simulate  # noqa: E402

CASES = 4
SHAPES = [(640, 24), (1024, 40)]


def check_committed_source_is_current():
    for path, _, _, stale in rsa_pair.sources():
        assert not stale, f'{path}: {stale}'


def check_detects_edited_template():
    text = (rsa_pair.ROOT / rsa_pair.SOURCE).read_text(encoding='utf-8')
    lines, first, _ = p384.template_span(text, 'square_kernel')
    lines[first + 3] = lines[first + 3].replace('x3', 'x1')
    _, stale = p384.regenerate('\n'.join(lines), rsa_pair_aarch64.KERNELS)
    assert stale == ['square_kernel'], stale


def mutated(generator, predicate, edit):
    """Apply `edit` to the first template line that satisfies `predicate`."""
    def wrapper():
        lines = list(generator()[0])
        index = next(i for i, line in enumerate(lines) if predicate(line))
        lines[index] = edit(lines[index])
        return (lines,)
    return wrapper


MUTATIONS = {
    # A high-half product read from the low half.
    'umlal2 as umlal': (lambda line: line.startswith('umlal2 '),
                        lambda line: line.replace('umlal2', 'umlal').replace('.4s', '.2s')),
    # The triangle's carry from x_k into column k + 1.
    'dropped column carry': (lambda line: line.startswith('add x13, x13, x9, lsr #28'),
                             lambda line: 'add x13, x13, xzr'),
    # The nonzero-low-bits correction of the same carry.
    'dropped carry rounding': (lambda line: line.startswith('cinc x24'),
                               lambda line: 'mov x24, x24'),
    # The final pass's digit mask.
    'unmasked output digit': (lambda line: line.startswith('and x12, x9,'),
                              lambda line: 'mov x12, x9'),
    # An entering column read one slot too far, past the column buffer on the last pass.
    'column read past its slot': (lambda line: line.startswith('ldr q8, [x0, #112]'),
                                  lambda line: line.replace('#112', '#128')),
}


def check_simulator_detects_each_mutation():
    for name, generator in rsa_pair_aarch64.KERNELS.items():
        for label, (predicate, edit) in MUTATIONS.items():
            kernels = {name: mutated(generator, predicate, edit)}
            failures = rsa_pair_simulate.check_kernels(kernels, CASES, random.Random(1), SHAPES)
            assert any(f'{name}(' in failure for failure in failures), (name, label)


def check_simulator_accepts_both_kernels():
    failures = rsa_pair_simulate.check_kernels(rsa_pair_aarch64.KERNELS, CASES, random.Random(2), SHAPES)
    assert not failures, failures[:5]


def main():
    for check in (check_committed_source_is_current, check_detects_edited_template,
                  check_simulator_accepts_both_kernels, check_simulator_detects_each_mutation):
        check()
    print('paired RSA assembly checks passed')


if __name__ == '__main__':
    main()
