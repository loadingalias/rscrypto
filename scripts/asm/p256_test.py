"""Regression and negative controls for the P-256 verification assembly generator and simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p256  # noqa: E402
import p256_x86_64  # noqa: E402
import p384  # noqa: E402

CASES = 40


def check_committed_source_is_current():
    text = (p256.ROOT / p256.SOURCE).read_text(encoding='utf-8')
    assert not p384.regenerate(text, p256_x86_64.KERNELS)[1]


def check_detects_edited_template():
    text = (p256.ROOT / p256.SOURCE).read_text(encoding='utf-8')
    lines, first, _ = p384.template_span(text, 'montgomery_mul_bmi2_adx')
    lines[first] = lines[first].replace('mov rdx', 'mov rax')
    assert p384.regenerate('\n'.join(lines), p256_x86_64.KERNELS)[1] == ['montgomery_mul_bmi2_adx']


def mutated(generator, index):
    """Replace the `index`-th carry-propagating add with a carry-dropping one."""
    def wrapper():
        lines, out = generator()
        positions = [i for i, line in enumerate(lines) if line.startswith('adc ')]
        position = positions[index]
        return lines[:position] + ['add ' + lines[position][4:]] + lines[position + 1:], out
    return wrapper


def check_simulator_detects_each_mutation():
    """Every kernel fails when its first or last carry is dropped; the last one is the final correction."""
    for name, generator in p256_x86_64.KERNELS.items():
        for index in (0, -1):
            kernels = {**p256_x86_64.KERNELS, name: mutated(generator, index)}
            assert any(failure.startswith(f'{name}(') for failure in p256.simulate(kernels, CASES)), (name, index)


def check_current_generator_simulates_clean():
    assert not p256.simulate(p256_x86_64.KERNELS, CASES)


def main():
    check_committed_source_is_current()
    check_detects_edited_template()
    check_simulator_detects_each_mutation()
    check_current_generator_simulates_clean()
    print('P-256 assembly generator regressions passed')


if __name__ == '__main__':
    main()
