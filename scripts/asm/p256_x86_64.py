"""Generate the x86-64 P-256 verification field assembly in `src/auth/ecdsa_p256_verify.rs`.

Each kernel function returns the `asm!` template lines (Intel syntax with Rust placeholders) and
the placeholders that hold its four output limbs. `scripts/asm/p256.py` compares the output with
the committed source, rewrites it, or simulates it.

Multiplication and squaring need BMI2 `MULX` and the ADX `ADCX`/`ADOX` pair; addition and
subtraction need only the x86-64 baseline. With `p = 2^256 - 2^224 + 2^192 + 2^96 - 1`,
`-p^-1 mod 2^64 = 1`, so each Montgomery step's quotient `m` is the low limb, and `m·p` adds
`m << 32` and `m >> 32` one and two limbs up and the 128-bit `m·p[3]` three limbs up, where
`p[3] = 2^64 - 2^32 + 1`. Each step keeps its carry out as 0 or -1 in the limb that the step
frees, and the next step adds it to its `m·p[3]` high word, which is at most `2^64 - 2^32`.
"""

W = [f'{{w{i}}}' for i in range(8)]
P1 = 0x0000_0000_FFFF_FFFF
P3 = 0xFFFF_FFFF_0000_0001


def _operand(pointer, limb):
    return f'qword ptr [{{{pointer}}} + {8 * limb}]' if limb else f'qword ptr [{{{pointer}}}]'


def _low32(placeholder):
    """The 32-bit form of a register placeholder: `{x}` becomes `{x:e}`."""
    return placeholder[:-1] + ':e}'


def _add_back(lines, value, mask, low_half, top):
    """Add `p & mask` to `value`. Its limb 0 is the mask, limb 1 the mask's low half, limb 2 zero,
    and limb 3 the negated low half."""
    lines += [f'mov {_low32(low_half)}, {_low32(mask)}', f'xor {top}, {top}', f'sub {top}, {low_half}',
              f'add {value[0]}, {mask}', f'adc {value[1]}, {low_half}', f'adc {value[2]}, 0',
              f'adc {value[3]}, {top}']
    return value


def _reduce(lines, low, high):
    """Reduce the 512-bit `W[0..8]` in place; return the four limbs of the canonical result."""
    for k in range(4):
        m = W[k]
        lines += [f'mov {{lo}}, {m}', 'shl {lo}, 32', f'mov {{hi}}, {m}', 'shr {hi}, 32',
                  f'mov rdx, {P3}', f'mulx {high}, {low}, {m}']
        if k:
            lines.append(f'sub {high}, {W[k - 1]}')
        lines += [f'add {W[k + 1]}, {{lo}}', f'adc {W[k + 2]}, {{hi}}', f'adc {W[k + 3]}, {low}',
                  f'adc {W[k + 4]}, {high}', f'sbb {m}, {m}']
    # The value `W[4..8] + carry·2^256`, with `carry = -W[3]`, is below 2p: subtract p, and add it
    # back when the subtraction borrows past the carry.
    lines += [f'mov {{lo}}, {P3}', f'mov {{hi:e}}, {P1}',
              f'sub {W[4]}, -1', f'sbb {W[5]}, {{hi}}', f'sbb {W[6]}, 0', f'sbb {W[7]}, {{lo}}',
              f'sbb {low}, {low}', f'sub {low}, {W[3]}']
    return _add_back(lines, W[4:8], low, high, '{lo}')


def montgomery_mul_bmi2_adx():
    """`a·b·2^-256 mod p`: four MULX rows with ADCX/ADOX carry chains, then the reduction."""
    lines = ['mov rdx, qword ptr [{a}]', f'mulx {W[1]}, {W[0]}, {_operand("b", 0)}']
    for j in range(1, 4):
        lines += [f'mulx {W[j + 1]}, {{lo}}, {_operand("b", j)}', f"{'add' if j == 1 else 'adc'} {W[j]}, {{lo}}"]
    lines.append(f'adc {W[4]}, 0')
    for i in range(1, 4):
        # The XOR zeroes the new top limb and clears CF and OF for the two chains.
        lines += [f'mov rdx, {_operand("a", i)}', f'xor {W[i + 4]}, {W[i + 4]}']
        for j in range(4):
            lines += [f'mulx {{hi}}, {{lo}}, {_operand("b", j)}', f'adcx {W[i + j]}, {{lo}}',
                      f'adox {W[i + j + 1]}, {{hi}}']
        lines.append(f'adc {W[i + 4]}, 0')
    return lines, _reduce(lines, '{a}', '{b}')


def montgomery_square_bmi2_adx():
    """`a²·2^-256 mod p`: six cross products, doubled through CF, plus four squares, then the
    reduction."""
    def a(limb):
        return _operand('a', limb)

    lines = [f'mov rdx, {a(0)}', f'mulx {W[2]}, {W[1]}, {a(1)}',
             f'mulx {W[3]}, {{lo}}, {a(2)}', f'add {W[2]}, {{lo}}',
             f'mulx {W[4]}, {{lo}}, {a(3)}', f'adc {W[3]}, {{lo}}', f'adc {W[4]}, 0',
             f'mov rdx, {a(1)}', f'xor {W[5]}, {W[5]}',
             f'mulx {{hi}}, {{lo}}, {a(2)}', f'adcx {W[3]}, {{lo}}', f'adox {W[4]}, {{hi}}',
             f'mulx {{hi}}, {{lo}}, {a(3)}', f'adcx {W[4]}, {{lo}}', f'adox {W[5]}, {{hi}}', f'adc {W[5]}, 0',
             f'mov rdx, {a(2)}', f'mulx {W[6]}, {{lo}}, {a(3)}', f'add {W[5]}, {{lo}}', f'adc {W[6]}, 0',
             f'add {W[1]}, {W[1]}', *[f'adc {W[k]}, {W[k]}' for k in range(2, 7)],
             # MOV and MULX keep the flags, so the doubling's carry reaches the top limb and the squares.
             f'mov {_low32(W[7])}, 0', f'adc {W[7]}, 0',
             f'mov rdx, {a(0)}', f'mulx {{hi}}, {W[0]}, rdx', f'add {W[1]}, {{hi}}']
    for j in range(1, 4):
        lines += [f'mov rdx, {a(j)}', 'mulx {hi}, {lo}, rdx', f'adc {W[2 * j]}, {{lo}}',
                  f'adc {W[2 * j + 1]}, {{hi}}']
    return lines, _reduce(lines, '{a}', '{t}')


def add_mod():
    """`left + right mod p`: the sum and its carry, then `p` subtracted unless that borrows past the
    carry."""
    lines = ['add {d0}, qword ptr [{right}]', 'adc {d1}, qword ptr [{right} + 8]',
             'adc {d2}, qword ptr [{right} + 16]', 'adc {d3}, qword ptr [{right} + 24]',
             'mov {cs:e}, 0', 'adc {cs}, 0', f'mov {{t3}}, {P3}', f'mov {{t1:e}}, {P1}',
             'sub {d0}, -1', 'sbb {d1}, {t1}', 'sbb {d2}, 0', 'sbb {d3}, {t3}', 'sbb {m}, {m}', 'add {m}, {cs}']
    return lines, _add_back(lines, ['{d0}', '{d1}', '{d2}', '{d3}'], '{m}', '{t1}', '{t3}')


def sub_mod():
    """`left - right mod p`: the difference, then `p` added back when it borrows."""
    lines = ['sub {d0}, qword ptr [{right}]', 'sbb {d1}, qword ptr [{right} + 8]',
             'sbb {d2}, qword ptr [{right} + 16]', 'sbb {d3}, qword ptr [{right} + 24]', 'sbb {m}, {m}']
    return lines, _add_back(lines, ['{d0}', '{d1}', '{d2}', '{d3}'], '{m}', '{t1}', '{t3}')


KERNELS = {
    'montgomery_mul_bmi2_adx': montgomery_mul_bmi2_adx,
    'montgomery_square_bmi2_adx': montgomery_square_bmi2_adx,
    'add_mod': add_mod,
    'sub_mod': sub_mod,
}
