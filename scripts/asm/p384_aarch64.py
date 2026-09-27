"""Generate the AArch64 P-384 assembly in `src/auth/p384_aarch64.rs`.

Each kernel function returns the `asm!` template lines and the placeholders that hold its six
output limbs. `scripts/asm/p384.py` compares the output with the committed source or rewrites it.
`add_mod`, `sub_mod`, `select_comb`, and the divstep loop are written by hand.
"""

import re


def _reduce_deferred(window, first, last):
    """One Montgomery step with the borrow deferred to the next step.

    q lives in window[0], which becomes the new top limb. The pending borrow {b} belongs
    at window[3] of this step (old window[4] of the previous step).
    """
    lines = [f'add {window[0]}, {window[0]}, {window[0]}, lsl #32',
             f'umulh {{h0}}, {window[0]}, {{c0}}',
             f'mul {{k1}}, {window[0]}, {{c1}}',
             f'umulh {{k2}}, {window[0]}, {{c1}}',
             'adds {k1}, {k1}, {h0}',
             f'adcs {{k2}}, {{k2}}, {window[0]}',
             'cset {h0}, hs' if first else 'adc {h0}, {b}, xzr',
             f'subs {window[1]}, {window[1]}, {{k1}}',
             f'sbcs {window[2]}, {window[2]}, {{k2}}',
             f'sbcs {window[3]}, {window[3]}, {{h0}}']
    if last:
        lines += [f'sbcs {window[4]}, {window[4]}, xzr', f'sbcs {window[5]}, {window[5]}, xzr',
                  f'sbc {window[0]}, {window[0]}, xzr']
    else:
        lines.append('cset {b}, lo')
    return lines


def _constants():
    return ['mov {c1:w}, #-1', 'neg {c0}, {c1}']


def _final(low, carry, out, scratch, c0, one):
    """low + carry * 2^384 < 2p: add 2^384 - p = [c0, c1, 1, 0, 0, 0] and select."""
    lines = [f'adds {scratch[0]}, {low[0]}, {c0}',
             f'adcs {scratch[1]}, {low[1]}, {{c1}}',
             f'mov {one}, #1',
             f'adcs {scratch[2]}, {low[2]}, {one}',
             f'adcs {scratch[3]}, {low[3]}, xzr',
             f'adcs {scratch[4]}, {low[4]}, xzr',
             f'adcs {scratch[5]}, {low[5]}, xzr',
             f'adc {carry}, {carry}, xzr',
             f'cmp {carry}, #0']
    lines += [f'csel {out[j]}, {scratch[j]}, {low[j]}, ne' for j in range(6)]
    return lines


def _reduce_and_finish(lines, w, out):
    lines += _constants()
    for k in range(6):
        window = [w[(k + j) % 6] for j in range(6)]
        lines += _reduce_deferred(window, k == 0, k == 5)
    lines.append(f'adds {w[0]}, {w[0]}, {w[6]}')
    lines += [f'adcs {w[j]}, {w[j]}, {w[j + 6]}' for j in range(1, 6)]
    lines.append('cset {x}, hs')
    lines += _final(w[:6], '{x}', out, ['{w6}', '{w7}', '{w8}', '{w9}', '{w10}', '{w11}'], '{c0}', '{q}')


def _rename(lines, mapping):
    """Map temporaries onto output registers that are dead at that point, keeping modifiers."""
    pattern = re.compile(r'\{(\w+)(:\w+)?\}')

    def replace(match):
        target = mapping.get(match.group(1))
        return f'{{{target}{match.group(2) or ""}}}' if target else match.group(0)

    return [pattern.sub(replace, line) for line in lines]


def montgomery_square():
    """`montgomery_square`: value in and result out of `{a0}`..`{a5}`."""
    a = [f'{{a{i}}}' for i in range(6)]
    w = [f'{{w{i}}}' for i in range(12)]
    lines = [f'mul {w[j]}, {a[0]}, {a[j]}' for j in range(1, 6)]
    lines += [f'umulh {{x}}, {a[0]}, {a[1]}', f'adds {w[2]}, {w[2]}, {{x}}']
    for j in range(2, 5):
        lines += [f'umulh {{x}}, {a[0]}, {a[j]}', f'adcs {w[j + 1]}, {w[j + 1]}, {{x}}']
    lines += [f'umulh {{x}}, {a[0]}, {a[5]}', f'adc {w[6]}, {{x}}, xzr']
    for i in range(1, 5):
        for n, j in enumerate(range(i + 1, 6)):
            lines += [f'mul {{x}}, {a[i]}, {a[j]}', f"{'adds' if n == 0 else 'adcs'} {w[i + j]}, {w[i + j]}, {{x}}"]
        lines.append(f'adc {w[i + 6]}, xzr, xzr')
        for n, j in enumerate(range(i + 1, 6)):
            lines.append(f'umulh {{x}}, {a[i]}, {a[j]}')
            if j == 5:
                lines.append(f"{'add' if n == 0 else 'adc'} {w[i + 6]}, {w[i + 6]}, {{x}}")
            else:
                lines.append(f"{'adds' if n == 0 else 'adcs'} {w[i + j + 1]}, {w[i + j + 1]}, {{x}}")
    lines.append(f'adds {w[1]}, {w[1]}, {w[1]}')
    lines += [f'adcs {w[j]}, {w[j]}, {w[j]}' for j in range(2, 11)]
    lines.append(f'adc {w[11]}, xzr, xzr')
    lines += [f'mul {w[0]}, {a[0]}, {a[0]}', f'umulh {{x}}, {a[0]}, {a[0]}', f'adds {w[1]}, {w[1]}, {{x}}']
    for i in range(1, 6):
        lines += [f'mul {{x}}, {a[i]}, {a[i]}', f'adcs {w[2 * i]}, {w[2 * i]}, {{x}}',
                  f'umulh {{x}}, {a[i]}, {a[i]}', f"{'adc' if i == 5 else 'adcs'} {w[2 * i + 1]}, {w[2 * i + 1]}, {{x}}"]
    _reduce_and_finish(lines, w, a)
    # The inputs are dead after the diagonals, so the temporaries use the output registers.
    return _rename(lines, {'q': 'a0', 'b': 'a0', 'k1': 'a1', 'k2': 'a2', 'h0': 'a3', 'c0': 'a4', 'c1': 'a5'}), a


def montgomery_mul():
    """`montgomery_mul`: `{l0}`..`{l5}` times `{b0}`..`{b5}`; result out of `{b0}`..`{b5}`."""
    b = [f'{{b{i}}}' for i in range(6)]
    w = [f'{{w{i}}}' for i in range(12)]
    lines = []
    for i in range(6):
        left = f'{{l{i}}}'
        if i == 0:
            lines += [f'mul {w[j]}, {left}, {b[j]}' for j in range(6)]
            lines += [f'umulh {{x}}, {left}, {b[0]}', f'adds {w[1]}, {w[1]}, {{x}}']
            for j in range(1, 5):
                lines += [f'umulh {{x}}, {left}, {b[j]}', f'adcs {w[j + 1]}, {w[j + 1]}, {{x}}']
            lines += [f'umulh {{x}}, {left}, {b[5]}', f'adc {w[6]}, {{x}}, xzr']
        else:
            for j in range(6):
                lines += [f'mul {{x}}, {left}, {b[j]}', f"{'adds' if j == 0 else 'adcs'} {w[i + j]}, {w[i + j]}, {{x}}"]
            lines.append(f'adc {w[i + 6]}, xzr, xzr')
            for j in range(6):
                lines.append(f'umulh {{x}}, {left}, {b[j]}')
                lines.append(f"{'adds' if j == 0 else ('adc' if j == 5 else 'adcs')} {w[i + j + 1]}, {w[i + j + 1]}, {{x}}")
    _reduce_and_finish(lines, w, b)
    # The right operand is dead after the product rows.
    return _rename(lines, {'q': 'b0', 'b': 'b0', 'k1': 'b1', 'k2': 'b2', 'h0': 'b3', 'c0': 'b4', 'c1': 'b5'}), b


def mul_small():
    """`mul_small`: `value * k mod p` for `k <= 8`; value in and result out of `{a0}`..`{a5}`."""
    a = [f'{{a{i}}}' for i in range(6)]
    t = [f'{{t{i}}}' for i in range(6)]
    lines = [f'mul {t[0]}, {a[0]}, {{k}}', f'umulh {{h}}, {a[0]}, {{k}}']
    for j in range(1, 6):
        lines += [f'mul {t[j]}, {a[j]}, {{k}}', f"{'adds' if j == 1 else 'adcs'} {t[j]}, {t[j]}, {{h}}",
                  f'umulh {{h}}, {a[j]}, {{k}}']
    lines.append('adc {top}, {h}, xzr')
    # Fold top * (2^384 - p) = top * [c0, c1, 1].
    lines += ['mov {c1:w}, #-1', 'neg {c0}, {c1}',
              'mul {h}, {top}, {c0}', 'umulh {cv}, {top}, {c0}', 'mul {m1}, {top}, {c1}', 'add {m1}, {m1}, {cv}',
              f'adds {t[0]}, {t[0]}, {{h}}', f'adcs {t[1]}, {t[1]}, {{m1}}', f'adcs {t[2]}, {t[2]}, {{top}}',
              f'adcs {t[3]}, {t[3]}, xzr', f'adcs {t[4]}, {t[4]}, xzr', f'adcs {t[5]}, {t[5]}, xzr', 'cset {cv}, hs',
              f'adds {a[0]}, {t[0]}, {{c0}}', f'adcs {a[1]}, {t[1]}, {{c1}}', 'mov {h}, #1', f'adcs {a[2]}, {t[2]}, {{h}}',
              f'adcs {a[3]}, {t[3]}, xzr', f'adcs {a[4]}, {t[4]}, xzr', f'adcs {a[5]}, {t[5]}, xzr',
              'adc {cv}, {cv}, xzr', 'cmp {cv}, #0']
    lines += [f'csel {a[j]}, {a[j]}, {t[j]}, ne' for j in range(6)]
    return lines, a


KERNELS = {
    'montgomery_mul': montgomery_mul,
    'montgomery_square': montgomery_square,
    'mul_small': mul_small,
}
