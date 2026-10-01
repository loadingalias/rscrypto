"""Generate the x86-64 BMI2/ADX P-384 assembly in `src/auth/p384_x86_64.rs`.

Each kernel function returns the `asm!` template lines (Intel syntax with Rust placeholders)
and the placeholders that hold its six output limbs. `point_double_rust` renders the whole
in-place doubling function that ends the source file. `scripts/asm/p384.py` compares the
output with the committed source or rewrites it.
"""

import re

W = [f'{{w{i}}}' for i in range(7)]


def _mul_rows():
    """768-bit product: six MULX rows with ADCX/ADOX carry chains through `{buf}`."""
    lines = []

    def reg(k):
        return W[k % 7]

    def right(j):
        return f'qword ptr [{{b}} + {8 * j}]'

    lines.append('mov rdx, qword ptr [{a}]')
    lines.append(f'mulx {reg(1)}, {reg(0)}, {right(0)}')
    for j in range(1, 6):
        lines.append(f'mulx {reg(j + 1)}, {{lo}}, {right(j)}')
        lines.append(f"{'add' if j == 1 else 'adc'} {reg(j)}, {{lo}}")
    lines.append(f'adc {reg(6)}, 0')
    lines.append(f'mov qword ptr [{{buf}}], {reg(0)}')
    for i in range(1, 6):
        lines.append(f'mov rdx, qword ptr [{{a}} + {8 * i}]')
        lines.append(f'xor {reg(i + 6)}, {reg(i + 6)}')  # zero the new top limb, clear CF and OF
        for j in range(6):
            lines.append(f'mulx {{hi}}, {{lo}}, {right(j)}')
            lines.append(f'adcx {reg(i + j)}, {{lo}}')
            lines.append(f'adox {reg(i + j + 1)}, {{hi}}')
        lines.append(f'adc {reg(i + 6)}, 0')
        lines.append(f'mov qword ptr [{{buf}} + {8 * i}], {reg(i)}')
    for k in range(6, 12):
        lines.append(f'mov qword ptr [{{buf}} + {8 * k}], {reg(k)}')
    low = [reg(k) for k in range(6, 12)]
    for k in range(6):
        lines.append(f'mov {low[k]}, qword ptr [{{buf}} + {8 * k}]')
    return lines, low, reg(5)


def _square_rows():
    """15 cross products in five MULX rows, then doubling through CF and diagonals through OF."""
    lines = []
    reg = {}
    free = W[:]

    def operand(j):
        return f'qword ptr [{{a}} + {8 * j}]'

    def take(k):
        reg[k] = free.pop(0)
        return reg[k]

    def release(k):
        free.append(reg.pop(k))

    lines.append(f'mov rdx, {operand(0)}')
    take(1)
    take(2)
    lines.append(f'mulx {reg[2]}, {reg[1]}, {operand(1)}')
    for j in range(2, 6):
        take(j + 1)
        lines.append(f'mulx {reg[j + 1]}, {{lo}}, {operand(j)}')
        lines.append(f"{'add' if j == 2 else 'adc'} {reg[j]}, {{lo}}")
    lines.append(f'adc {reg[6]}, 0')
    for k in (1, 2):
        lines.append(f'mov qword ptr [{{buf}} + {8 * k}], {reg[k]}')
        release(k)
    for i in range(1, 5):
        lines.append(f'mov rdx, {operand(i)}')
        top = i + 6
        take(top)
        lines.append(f'xor {reg[top]}, {reg[top]}')
        for j in range(i + 1, 6):
            lines.append(f'mulx {{hi}}, {{lo}}, {operand(j)}')
            lines.append(f'adcx {reg[i + j]}, {{lo}}')
            lines.append(f'adox {reg[i + j + 1]}, {{hi}}')
        lines.append(f'adc {reg[top]}, 0')
        for k in (2 * i + 1, 2 * i + 2):
            lines.append(f'mov qword ptr [{{buf}} + {8 * k}], {reg[k]}')
            release(k)
    assert not reg, reg
    low, borrow = W[:6], W[6]
    lines.append(f'xor {borrow}, {borrow}')  # clears CF and OF; the register doubles as zero
    for i in range(6):
        lines.append(f'mov rdx, {operand(i)}')
        lines.append('mulx {hi}, {lo}, rdx')
        for k, source in ((2 * i, '{lo}'), (2 * i + 1, '{hi}')):
            target = low[k] if k < 6 else '{b}'
            if k in (0, 11):
                lines.append(f'mov {target}, 0')
            else:
                lines.append(f'mov {target}, qword ptr [{{buf}} + {8 * k}]')
            lines.append(f'adcx {target}, {target}')
            lines.append(f'adox {target}, {source}')
            if k >= 6:
                lines.append(f'mov qword ptr [{{buf}} + {8 * k}], {target}')
    return lines, low, borrow


def _reduce(lines, low, borrow):
    """Six Montgomery steps on the low half with a deferred borrow; quotient in RDX."""
    h0, k1, k2 = '{lo}', '{hi}', '{a}'
    for step in range(6):
        window = [low[(step + j) % 6] for j in range(6)]
        first, last = step == 0, step == 5
        lines += [f'mov rdx, {window[0]}', 'shl rdx, 32', f'add rdx, {window[0]}',
                  f'mulx {h0}, {k1}, qword ptr [rip + {{consts}}]',
                  f'mulx {k2}, {k1}, qword ptr [rip + {{consts}} + 8]',
                  f'add {k1}, {h0}',
                  f'adc {k2}, rdx',
                  f'mov {h0}, 0',
                  f"adc {h0}, {'0' if first else borrow}",
                  f'sub {window[1]}, {k1}', f'sbb {window[2]}, {k2}', f'sbb {window[3]}, {h0}']
        if last:
            lines += [f'sbb {window[4]}, 0', f'sbb {window[5]}, 0', 'sbb rdx, 0', f'mov {window[0]}, rdx']
        else:
            lines += [f'mov {borrow}, 0', f'adc {borrow}, 0', f'mov {window[0]}, rdx']


def _finish(lines, low, borrow):
    """Add the high half, then add `c = 2^384 - p` and keep the sum if it carried."""
    lines.append(f'add {low[0]}, qword ptr [{{buf}} + 48]')
    for j in range(1, 6):
        lines.append(f'adc {low[j]}, qword ptr [{{buf}} + {48 + 8 * j}]')
    lines += [f'mov {borrow}, 0', f'adc {borrow}, 0']
    scratch = ['rdx', '{lo}', '{hi}', '{a}', '{b}', '{buf}']
    addend = ['qword ptr [rip + {consts}]', 'qword ptr [rip + {consts} + 8]', '1', '0', '0', '0']
    for j in range(6):
        lines.append(f'mov {scratch[j]}, {low[j]}')
        lines.append(f"{'add' if j == 0 else 'adc'} {scratch[j]}, {addend[j]}")
    lines += [f'adc {borrow}, 0', f'test {borrow}, {borrow}']
    for j in range(6):
        lines.append(f'cmovnz {low[j]}, {scratch[j]}')


def montgomery_mul():
    """`montgomery_mul_bmi2_adx`: `left * right * 2^-384 mod p`."""
    lines, low, borrow = _mul_rows()
    _reduce(lines, low, borrow)
    _finish(lines, low, borrow)
    return lines, low


def montgomery_square():
    """`montgomery_square_bmi2_adx`: `value^2 * 2^-384 mod p`."""
    lines, low, borrow = _square_rows()
    _reduce(lines, low, borrow)
    _finish(lines, low, borrow)
    return lines, low


def mul_small():
    """`mul_small_bmi2`: `value * k mod p` for `k <= 8`, value limbs in `{d0}`..`{d5}`."""
    d = [f'{{d{i}}}' for i in range(6)]
    lines = [f'mulx {{h}}, {d[0]}, {d[0]}']
    for j in range(1, 6):
        high = '{t}' if j % 2 else '{h}'
        previous = '{h}' if j % 2 else '{t}'
        lines.append(f'mulx {high}, {d[j]}, {d[j]}')
        lines.append(f"{'add' if j == 1 else 'adc'} {d[j]}, {previous}")
    top = '{t}'  # the high half of limb 5 lands in {t}
    lines.append(f'adc {top}, 0')
    # Fold top * (2^384 - p) = top * [c0, c1, 1].
    lines += [f'mov rdx, {top}',
              'mulx {h}, {m0}, qword ptr [rip + {consts}]',
              'mulx {cv}, {m1}, qword ptr [rip + {consts} + 8]',
              'add {m1}, {h}',
              f'add {d[0]}, {{m0}}', f'adc {d[1]}, {{m1}}', f'adc {d[2]}, {top}',
              f'adc {d[3]}, 0', f'adc {d[4]}, 0', f'adc {d[5]}, 0',
              'mov {cv}, 0', 'adc {cv}, 0']
    # Subtract p; add it back when the value was below p (mask = carry - borrow).
    lines.append(f'sub {d[0]}, qword ptr [rip + {{modulus}}]')
    lines += [f'sbb {d[j]}, qword ptr [rip + {{modulus}} + {8 * j}]' for j in range(1, 6)]
    lines += ['sbb {m}, {m}', 'add {m}, {cv}',
              'mov {h:e}, {m:e}', 'mov {t}, {m}', 'sub {t}, {h}', 'mov {m0}, {m}', 'and {m0}, -2',
              f'add {d[0]}, {{h}}', f'adc {d[1]}, {{t}}', f'adc {d[2]}, {{m0}}',
              f'adc {d[3]}, {{m}}', f'adc {d[4]}, {{m}}', f'adc {d[5]}, {{m}}']
    return lines, d


def _add_back(lines, d):
    """Add `p & mask` where `{m}` is all ones or zero."""
    lines += ['mov {t0:e}, {m:e}', 'mov {t1}, {m}', 'sub {t1}, {t0}', 'mov {t2}, {m}', 'and {t2}, -2',
              f'add {d[0]}, {{t0}}', f'adc {d[1]}, {{t1}}', f'adc {d[2]}, {{t2}}',
              f'adc {d[3]}, {{m}}', f'adc {d[4]}, {{m}}', f'adc {d[5]}, {{m}}']


def add_mod():
    """`add_mod`: `left + right mod p`, left limbs in `{d0}`..`{d5}`, right at `{right}`."""
    d = [f'{{d{i}}}' for i in range(6)]
    lines = [f'add {d[0]}, qword ptr [{{right}}]']
    lines += [f'adc {d[j]}, qword ptr [{{right}} + {8 * j}]' for j in range(1, 6)]
    lines += ['mov {cs:e}, 0', 'adc {cs}, 0']
    lines.append(f'sub {d[0]}, qword ptr [rip + {{modulus}}]')
    lines += [f'sbb {d[j]}, qword ptr [rip + {{modulus}} + {8 * j}]' for j in range(1, 6)]
    lines += ['sbb {m}, {m}', 'add {m}, {cs}']
    _add_back(lines, d)
    return lines, d


def sub_mod():
    """`sub_mod`: `left - right mod p`, left limbs in `{d0}`..`{d5}`, right at `{right}`."""
    d = [f'{{d{i}}}' for i in range(6)]
    lines = [f'sub {d[0]}, qword ptr [{{right}}]']
    lines += [f'sbb {d[j]}, qword ptr [{{right}} + {8 * j}]' for j in range(1, 6)]
    lines.append('sbb {m}, {m}')
    _add_back(lines, d)
    return lines, d


# In-place doubling. The point (X, Y, Z) is 18 limbs at {p}; temporaries live in a frame
# the block reserves below RSP.
POINT_LIMBS = 18
X, Y, Z = 0, 6, 12
TEMPORARIES = ['delta', 'gamma', 't1', 't2', 'x2p', 'beta', 'x4p', 'd', 'dx2', 'yz', 'yz2', 'g2', 'zt']
OFFSET = {name: POINT_LIMBS + 6 * i for i, name in enumerate(TEMPORARIES)}


def _memory(offset):
    if offset < POINT_LIMBS:
        return f'qword ptr [{{p}} + {8 * offset}]'
    return f'qword ptr [rsp + {8 * (offset - POINT_LIMBS)}]'


def _finish_into(lines, low, carry, destination):
    """Store `carry:low mod p` for `carry:low < 2p`.

    The unreduced limbs go to `destination` first. Adding `c = 2^384 - p` carries out of
    bit 384 exactly when the value is at least `p`; otherwise CMOVZ reloads the stored
    limbs. The serial chain is seven additions and one select, half that of subtracting
    `p` and adding it back under a mask.
    """
    for j in range(6):
        lines.append(f'mov {_memory(destination + j)}, {low[j]}')
    addend = ['qword ptr [rip + {consts}]', 'qword ptr [rip + {consts} + 8]', '1', '0', '0', '0']
    for j in range(6):
        lines.append(f"{'add' if j == 0 else 'adc'} {low[j]}, {addend[j]}")
    lines.append(f'adc {carry}, 0')  # ZF is set exactly when the value was below p
    for j in range(6):
        lines.append(f'cmovz {low[j]}, {_memory(destination + j)}')
    for j in range(6):
        lines.append(f'mov {_memory(destination + j)}, {low[j]}')


def _reduce_low(lines, low, borrow):
    h0, k1, k2 = '{lo}', '{hi}', '{s1}'
    for step in range(6):
        window = [low[(step + j) % 6] for j in range(6)]
        first, last = step == 0, step == 5
        lines += [f'mov rdx, {window[0]}', 'shl rdx, 32', f'add rdx, {window[0]}',
                  f'mulx {h0}, {k1}, qword ptr [rip + {{consts}}]',
                  f'mulx {k2}, {k1}, qword ptr [rip + {{consts}} + 8]',
                  f'add {k1}, {h0}', f'adc {k2}, rdx', f'mov {h0}, 0',
                  f"adc {h0}, {'0' if first else borrow}",
                  f'sub {window[1]}, {k1}', f'sbb {window[2]}, {k2}', f'sbb {window[3]}, {h0}']
        if last:
            lines += [f'sbb {window[4]}, 0', f'sbb {window[5]}, 0', 'sbb rdx, 0', f'mov {window[0]}, rdx']
        else:
            lines += [f'mov {borrow}, 0', f'adc {borrow}, 0', f'mov {window[0]}, rdx']


def _finish_high(lines, low, borrow, destination, buffer):
    lines.append(f'add {low[0]}, {_memory(buffer + 6)}')
    for j in range(1, 6):
        lines.append(f'adc {low[j]}, {_memory(buffer + 6 + j)}')
    lines += [f'mov {borrow}, 0', f'adc {borrow}, 0']
    _finish_into(lines, low, borrow, destination)


def _square_into(lines, destination, source, buffer):
    reg = {}
    free = W[:]

    def operand(j):
        return _memory(source + j)

    def take(k):
        reg[k] = free.pop(0)
        return reg[k]

    def release(k):
        free.append(reg.pop(k))

    lines.append(f'mov rdx, {operand(0)}')
    take(1)
    take(2)
    lines.append(f'mulx {reg[2]}, {reg[1]}, {operand(1)}')
    for j in range(2, 6):
        take(j + 1)
        lines.append(f'mulx {reg[j + 1]}, {{lo}}, {operand(j)}')
        lines.append(f"{'add' if j == 2 else 'adc'} {reg[j]}, {{lo}}")
    lines.append(f'adc {reg[6]}, 0')
    for k in (1, 2):
        lines.append(f'mov {_memory(buffer + k)}, {reg[k]}')
        release(k)
    for i in range(1, 5):
        lines.append(f'mov rdx, {operand(i)}')
        top = i + 6
        take(top)
        lines.append(f'xor {reg[top]}, {reg[top]}')
        for j in range(i + 1, 6):
            lines.append(f'mulx {{hi}}, {{lo}}, {operand(j)}')
            lines.append(f'adcx {reg[i + j]}, {{lo}}')
            lines.append(f'adox {reg[i + j + 1]}, {{hi}}')
        lines.append(f'adc {reg[top]}, 0')
        for k in (2 * i + 1, 2 * i + 2):
            lines.append(f'mov {_memory(buffer + k)}, {reg[k]}')
            release(k)
    low, borrow = W[:6], W[6]
    lines.append(f'xor {borrow}, {borrow}')
    for i in range(6):
        lines.append(f'mov rdx, {operand(i)}')
        lines.append('mulx {hi}, {lo}, rdx')
        for k, part in ((2 * i, '{lo}'), (2 * i + 1, '{hi}')):
            target = low[k] if k < 6 else '{s1}'
            lines.append(f'mov {target}, 0' if k in (0, 11) else f'mov {target}, {_memory(buffer + k)}')
            lines.append(f'adcx {target}, {target}')
            lines.append(f'adox {target}, {part}')
            if k >= 6:
                lines.append(f'mov {_memory(buffer + k)}, {target}')
    _reduce_low(lines, low, borrow)
    _finish_high(lines, low, borrow, destination, buffer)


def _interleaved_mul_into(lines, destination, a, b):
    """Interleaved Montgomery multiplication: row i, then reduce limb i.

    The accumulator window holds limbs i..i+7 in registers T[k % 8]. Reduction step i
    subtracts (k1, k2, h0 + borrow) from limbs i+1..i+3 with the borrow out deferred to
    step i+1, and adds q at limb i+6. No product buffer.
    """
    window = W[:7] + ['{s2}']

    def reg(k):
        return window[k % 8]

    borrow = '{s1}'

    def right(j):
        return _memory(b + j)

    for i in range(6):
        lines.append(f'mov rdx, {_memory(a + i)}')
        if i == 0:
            lines.append(f'mulx {reg(1)}, {reg(0)}, {right(0)}')
            for j in range(1, 6):
                lines.append(f'mulx {reg(j + 1)}, {{lo}}, {right(j)}')
                lines.append(f"{'add' if j == 1 else 'adc'} {reg(j)}, {{lo}}")
            lines.append(f'adc {reg(6)}, 0')
            lines.append(f'xor {reg(7)}, {reg(7)}')
        else:
            lines.append(f'xor {reg(i + 7)}, {reg(i + 7)}')
            for j in range(6):
                lines.append(f'mulx {{hi}}, {{lo}}, {right(j)}')
                lines.append(f'adcx {reg(i + j)}, {{lo}}')
                lines.append(f'adox {reg(i + j + 1)}, {{hi}}')
            lines += ['mov {lo}, 0', f'adcx {reg(i + 6)}, {{lo}}', f'adox {reg(i + 7)}, {{lo}}', f'adc {reg(i + 7)}, 0']
        t0 = reg(i)
        lines += [f'mov rdx, {t0}', 'shl rdx, 32', f'add rdx, {t0}',
                  f'mulx {t0}, {{lo}}, qword ptr [rip + {{consts}}]',
                  f'mulx {{hi}}, {{lo}}, qword ptr [rip + {{consts}} + 8]',
                  f'add {{lo}}, {t0}', f'adc {{hi}}, rdx', f'mov {t0}, 0',
                  f"adc {t0}, {'0' if i == 0 else borrow}",
                  f'sub {reg(i + 1)}, {{lo}}', f'sbb {reg(i + 2)}, {{hi}}', f'sbb {reg(i + 3)}, {t0}']
        if i < 5:
            lines += [f'mov {borrow}, 0', f'adc {borrow}, 0',
                      f'add {reg(i + 6)}, rdx', f'adc {reg(i + 7)}, 0']
        else:
            lines += [f'sbb {reg(9)}, 0', f'sbb {reg(10)}, 0', f'sbb {reg(11)}, 0', f'sbb {reg(12)}, 0',
                      f'add {reg(11)}, rdx', f'adc {reg(12)}, 0']
    low = [reg(k) for k in range(6, 12)]
    _finish_into(lines, low, reg(12), destination)


def _add_mod_into(lines, destination, a, b):
    d = W[:6]
    for j in range(6):
        lines.append(f'mov {d[j]}, {_memory(a + j)}')
    lines.append(f'add {d[0]}, {_memory(b)}')
    for j in range(1, 6):
        lines.append(f'adc {d[j]}, {_memory(b + j)}')
    lines += ['mov {w6}, 0', 'adc {w6}, 0']
    _finish_into(lines, d, '{w6}', destination)


def _sub_mod_into(lines, destination, a, b):
    d = W[:6]
    for j in range(6):
        lines.append(f'mov {d[j]}, {_memory(a + j)}')
    lines.append(f'sub {d[0]}, {_memory(b)}')
    for j in range(1, 6):
        lines.append(f'sbb {d[j]}, {_memory(b + j)}')
    mask, t0, t1, t2 = '{w6}', 'rdx', '{lo}', '{hi}'
    lines += [f'sbb {mask}, {mask}', f'mov {t0}, {mask}', f'shr {t0}, 32', f'mov {t1}, {mask}', f'sub {t1}, {t0}',
              f'mov {t2}, {mask}', f'and {t2}, -2',
              f'add {d[0]}, {t0}', f'adc {d[1]}, {t1}', f'adc {d[2]}, {t2}',
              f'adc {d[3]}, {mask}', f'adc {d[4]}, {mask}', f'adc {d[5]}, {mask}']
    for j in range(6):
        lines.append(f'mov {_memory(destination + j)}, {d[j]}')


def _linear_combination_into(lines, destination, a, k1, b, k2):
    """destination = k1 * a - k2 * b mod p for canonical a, b and 1 <= k1, k2 <= 12."""
    d = W[:6]
    for j in range(6):
        lines.append(f'mov {d[j]}, qword ptr [rip + {{modulus}} + {8 * j}]')
    lines.append(f'sub {d[0]}, {_memory(b)}')
    for j in range(1, 6):
        lines.append(f'sbb {d[j]}, {_memory(b + j)}')
    if k2 != 1:
        lines.append(f'mov rdx, {k2}')
        lines.append(f'mulx {{hi}}, {d[0]}, {d[0]}')
        for j in range(1, 6):
            high = '{lo}' if j % 2 else '{hi}'
            previous = '{hi}' if j % 2 else '{lo}'
            lines.append(f'mulx {high}, {d[j]}, {d[j]}')
            lines.append(f"{'add' if j == 1 else 'adc'} {d[j]}, {previous}")
        lines.append('mov {w6}, {lo}')
        lines.append('adc {w6}, 0')
    else:
        lines.append('mov {w6}, 0')
    lines.append(f'mov rdx, {k1}')
    lines.append('xor {s2:e}, {s2:e}')
    for j in range(6):
        lines.append(f'mulx {{hi}}, {{lo}}, {_memory(a + j)}')
        lines.append(f'adcx {d[j]}, {{lo}}')
        lines.append(f"adox {d[j + 1] if j < 5 else '{w6}'}, {{hi}}")
    lines.append('adcx {w6}, {s2}')
    lines += ['mov rdx, {w6}',
              'mulx {hi}, {lo}, qword ptr [rip + {consts}]',
              'mulx {s1}, {s2}, qword ptr [rip + {consts} + 8]',
              'add {s2}, {hi}',
              f'add {d[0]}, {{lo}}', f'adc {d[1]}, {{s2}}', f'adc {d[2]}, {{w6}}',
              f'adc {d[3]}, 0', f'adc {d[4]}, 0', f'adc {d[5]}, 0',
              'mov {w6}, 0', 'adc {w6}, 0']
    _finish_into(lines, d, '{w6}', destination)


def point_double():
    """`point_double_bmi2_adx` body and frame size.

    Algebra (a = -3, 3M + 5S): delta = Z^2, gamma = Y^2, x2p = (X - delta)(X + delta),
    beta = X * gamma, d = 12 beta - 9 x2p^2, X3 = 4 beta - d, Y3 = 3 d x2p - 8 gamma^2,
    Z3 = (Y + Z)^2 - gamma - delta. Z3 is finished early and Y3 is the tail, so the next
    doubling's Z3^2 can overlap this block's last kernels.
    """
    lines = []
    o = OFFSET
    next_buffer = [POINT_LIMBS + 6 * len(TEMPORARIES)]

    def buffer():
        start = next_buffer[0]
        next_buffer[0] += 12
        return start

    _square_into(lines, o['delta'], Z, buffer())
    _square_into(lines, o['gamma'], Y, buffer())
    _add_mod_into(lines, o['yz'], Y, Z)  # last read of Y and Z
    _sub_mod_into(lines, o['t1'], X, o['delta'])
    _add_mod_into(lines, o['t2'], X, o['delta'])
    _square_into(lines, o['yz2'], o['yz'], buffer())
    _interleaved_mul_into(lines, o['x2p'], o['t1'], o['t2'])
    _interleaved_mul_into(lines, o['beta'], X, o['gamma'])  # last read of X
    _sub_mod_into(lines, o['zt'], o['yz2'], o['gamma'])
    _sub_mod_into(lines, Z, o['zt'], o['delta'])  # Z3
    _square_into(lines, o['x4p'], o['x2p'], buffer())
    _linear_combination_into(lines, o['d'], o['beta'], 12, o['x4p'], 9)
    _square_into(lines, o['g2'], o['gamma'], buffer())
    _linear_combination_into(lines, X, o['beta'], 4, o['d'], 1)  # X3
    _interleaved_mul_into(lines, o['dx2'], o['d'], o['x2p'])
    _linear_combination_into(lines, Y, o['dx2'], 3, o['g2'], 8)  # Y3
    frame = 8 * (next_buffer[0] - POINT_LIMBS)
    return [f'sub rsp, {frame}'] + lines + [f'add rsp, {frame}'], frame


POINT_DOUBLE_MARKER = '/// Double a Jacobian point in place'

POINT_DOUBLE_DOC = """/// Double a Jacobian point in place: `(X, Y, Z)` becomes `2 * (X, Y, Z)` for
/// canonical Montgomery coordinates below `p`.
///
/// This fuses the shared `a = -3` doubling (3M + 5S) into one block with
/// `d = 12 * beta - 9 * x2p^2`, `X3 = 4 * beta - d`, and
/// `Y3 = 3 * d * x2p - 8 * gamma^2`, where `x2p = (X - Z^2)(X + Z^2)` and
/// `beta = X * Y^2`. These are the same polynomials as the portable formula,
/// so the canonical results are identical, including infinity (`Z = 0`).
///
/// Squaring matches `montgomery_square_bmi2_adx`. Multiplication interleaves
/// the reduction with the product rows: after row `i`, limb `i` sets the
/// quotient, `(k1, k2, h0)` is subtracted from limbs `i + 1..=i + 3` with the
/// borrow deferred to the next step, and the quotient is added at limb
/// `i + 6`, so the accumulator stays in an eight-register window.
/// `k1 * a - k2 * b` terms are formed as `k1 * a + k2 * (p - b) < 2^389`, the
/// top limb is folded through `c = 2^384 - p`. Squares, products, additions,
/// and linear combinations finish by storing their unreduced limbs, adding `c`,
/// and reloading the stored limbs with CMOVZ when the sum does not carry out
/// of bit 384.
///
/// The block writes Z3 early and Y3 last, so a following doubling can start
/// `Z3^2` while this one finishes. X, Y, and Z are last read before X3, Y3,
/// and Z3 are written.
///
/// # Safety
///
/// The CPU must support BMI2 and ADX."""

POINT_DOUBLE_SAFETY = """  // SAFETY: The caller guarantees BMI2 and ADX. The block reads and writes
  // the 144-byte, 8-byte-aligned `point`.
  // It also reads `REDUCTION_CONSTANTS` and `MODULUS`. Every access uses a
  // fixed offset. It reserves a {frame}-byte frame by subtracting from RSP,
  // which `nostack` is omitted to permit, addresses the frame only below the
  // original RSP, and restores RSP before exit. Every register it writes is a
  // declared output, and flags are clobbered by default."""

SCRATCH_REGISTERS = ['lo', 'hi', 's1', 's2', 'w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']


def point_double_rust():
    """Render the complete `point_double_bmi2_adx` function that ends the source file."""

    lines, frame = point_double()
    used = set(re.findall(r'\{(\w+)(?::e)?\}', '\n'.join(lines)))
    out = [POINT_DOUBLE_DOC, '#[inline(never)]', 'pub(super) unsafe fn point_double_bmi2_adx(point: &mut [u64; 18]) {',
           POINT_DOUBLE_SAFETY.format(frame=frame), '  unsafe {', '    core::arch::asm!(']
    out += [f'      "{line}",' for line in lines]
    out.append('      p = in(reg) point.as_mut_ptr(),')
    for symbol, target in (('consts', 'REDUCTION_CONSTANTS'), ('modulus', 'MODULUS')):
        if symbol in used:
            out.append(f'      {symbol} = sym {target},')
    out.append('      out("rdx") _,')
    out += [f'      {register} = out(reg) _,' for register in SCRATCH_REGISTERS if register in used]
    out += ['    );', '  }', '}']
    return '\n'.join(out) + '\n'


KERNELS = {
    'montgomery_mul_bmi2_adx': montgomery_mul,
    'montgomery_square_bmi2_adx': montgomery_square,
    'mul_small_bmi2': mul_small,
    'add_mod': add_mod,
    'sub_mod': sub_mod,
}
