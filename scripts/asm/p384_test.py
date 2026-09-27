"""Regression and negative controls for the P-384 assembly generators and simulators."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p384  # noqa: E402
import p384_aarch64  # noqa: E402
import p384_simulate  # noqa: E402
import p384_x86_64  # noqa: E402

CASES = 40


def check_committed_source_is_current():
    for path, _, _, stale in p384.sources():
        assert not stale, f'{path}: {stale}'


def check_detects_edited_template():
    relative = 'src/auth/p384_x86_64.rs'
    text = (p384.ROOT / relative).read_text()
    lines, first, _ = p384.template_span(text, 'montgomery_mul_bmi2_adx')
    lines[first] = lines[first].replace('mov rdx', 'mov rax')
    _, stale = p384.regenerate('\n'.join(lines), p384.SOURCES[relative])
    assert stale == ['montgomery_mul_bmi2_adx'], stale
    edited = text.replace('"add rsp, ', '"add rsp, 8 + ', 1)
    assert p384.regenerate_double(edited)[1] == ['point_double_bmi2_adx']


def mutated(generator):
    """Replace the first carry-propagating add with a carry-dropping one."""
    def wrapper():
        lines, out = generator()
        index = next(i for i, line in enumerate(lines) if line.startswith(('adc ', 'adcs ')))
        op, _, rest = lines[index].partition(' ')
        return lines[:index] + [f"{'add' if op == 'adc' else 'adds'} {rest}"] + lines[index + 1:], out
    return wrapper


def check_simulator_detects_each_mutation():
    for name, generator in p384_x86_64.KERNELS.items():
        kernels = {**p384_x86_64.KERNELS, name: mutated(generator)}
        failures = p384_simulate.check_x86(kernels, CASES, random.Random(1))
        assert any(f'x86 {name}(' in failure for failure in failures), name
    for name, generator in p384_aarch64.KERNELS.items():
        kernels = {**p384_aarch64.KERNELS, name: mutated(generator)}
        failures = p384_simulate.check_aarch64(kernels, CASES, random.Random(1))
        assert any(f'aarch64 {name}(' in failure for failure in failures), name

    def broken_double():
        lines, frame = p384_x86_64.point_double()
        index = next(i for i, line in enumerate(lines) if line.startswith('adc '))
        return lines[:index] + ['add ' + lines[index][4:]] + lines[index + 1:], frame
    assert p384_simulate.check_x86_double(broken_double, CASES, random.Random(1))


def check_simulator_reaches_final_correction():
    """Dropping the carry of the final `+ (2^384 - p)` correction needs constructed operands."""
    def replaced(generator, target, replacement):
        def wrapper():
            lines, out = generator()
            assert target in lines, target
            return [replacement if line == target else line for line in lines], out
        return wrapper

    for name in ('montgomery_mul_bmi2_adx', 'montgomery_square_bmi2_adx'):
        kernels = {**p384_x86_64.KERNELS, name: replaced(p384_x86_64.KERNELS[name], 'adc {hi}, 1', 'add {hi}, 1')}
        assert any(f'x86 {name}(' in f for f in p384_simulate.check_x86(kernels, CASES, random.Random(2))), name
    for name, operand in (('montgomery_mul', 'b0'), ('montgomery_square', 'a0')):
        target = f'adcs {{w8}}, {{w2}}, {{{operand}}}'
        kernels = {**p384_aarch64.KERNELS,
                   name: replaced(p384_aarch64.KERNELS[name], target, target.replace('adcs', 'adds'))}
        assert any(f'aarch64 {name}(' in f for f in p384_simulate.check_aarch64(kernels, CASES, random.Random(2))), name


def check_unmodeled_instruction_is_rejected():
    for run, line in ((lambda l: p384_simulate.run_x86(l, {'rax': 0}, {}), 'rol rax, 1'),
                      (lambda l: p384_simulate.run_aarch64(l, {'x': 0}), 'ror {x}, {x}, #1')):
        try:
            run([line])
        except ValueError:
            continue
        raise AssertionError(f'{line!r} was accepted')


def check_current_generators_simulate_clean():
    assert not p384_simulate.simulate(p384_x86_64, p384_aarch64, CASES)


def main():
    check_committed_source_is_current()
    check_detects_edited_template()
    check_simulator_detects_each_mutation()
    check_simulator_reaches_final_correction()
    check_unmodeled_instruction_is_rejected()
    check_current_generators_simulate_clean()
    print('P-384 assembly generator regressions passed')


if __name__ == '__main__':
    main()
