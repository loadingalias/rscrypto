"""Generate the AArch64 paired RSA Montgomery kernels in `src/auth/rsa_aarch64_pair.rs`.

The kernels run the p and q almost-Montgomery products of a CRT pair in the two 64-bit lanes of
the vector unit. A paired value has `D = 8·P` radix-2^28 digits; digit `j` is the eight-byte word
`p_j | q_j << 32`, stored twice in a 16-byte slot, so one `LDR Q` loads a multiplier that both
`UMLAL` (low half) and `UMLAL2` (high half) can use. Products of 28-bit digits fit 56 bits, so
64-bit column sums of up to 255 products cannot overflow and no carry moves until a column is
complete.

Structure, for operands `a` (packed eight digits at a time into registers) and `b` (streamed):

- Pass -1 adds product strip 0 (`a[0..8]·b`) into fresh columns.
- Pass s (0 <= s <= P - 2) adds product strip s + 1 and reduction strip s (`m[8s..8s + 8]·n`) in
  one sweep over the columns. Each step streams one digit of `b` and one of `n`, adds eight
  products into the window `A` and eight into `B`, flushes the oldest column to `T`, and loads
  the entering column into `B`. Two windows let each accumulator take one `UMLAL` per step, which
  the three-cycle `UMLAL` latency needs.
- The scalar unit computes the next eight multipliers (the "triangle") from the first eight
  columns each pass flushes, spread over the pass's first sixteen steps.
- The final pass adds reduction strip P - 1 and turns columns D..2D-1 into normalized digits
  with a scalar carry chain.

Squaring streams `a` itself against the doubled a-block, so each cross product appears once:
pass s skips product strip s + 1 until its diagonal block.

`check_kernels` in `rsa_pair_simulate.py` runs both kernels against Python integers.
"""

W = 8
MASK28 = (1 << 28) - 1

# Vector registers.
VA = [16, 17, 18, 19]  # a-block: packed digit pairs [a_2i, a_2i+1]
VM = [20, 21, 22, 23]  # m-block, packed
A = list(range(0, 8))  # product window
B = list(range(8, 16))  # reduction window
VS = 24  # replicated stream digit (b, or a when squaring)
VN = 25  # replicated modulus digit
VECTOR_REGISTERS = 26

# General registers: column pointer, stream pointer, modulus stream, a-block pointer (the output
# pointer in the final pass), parameter block, modulus base, loop counter, pass counter.
TP, SP, NP, AP, MB, NB, CNT, PASS = 'x0', 'x1', 'x2', 'x3', 'x5', 'x6', 'x7', 'x8'
MM = CNT  # multiplier pair for the store; the counter is free while the triangle runs
DM = 'x4'  # shape word or output digit pair
# Triangle registers per lane. H holds h_k = m_k·n_1 + floor(m_k·n_0 / 2^28) between digits and
# the carry into the next block between triangles, and the running carry in the final pass.
LANE_REGS = [
    dict(X='x9', MS='x10', H='x11', M='x12', P='x13', QA='x14', QB='x15', K='x16', N01='x17'),
    dict(X='x20', MS='x21', H='x22', M='x23', P='x24', QA='x25', QB='x26', K='x27', N01='x28'),
]
INPUT_REGISTERS = [TP, SP, NP, AP, MB, NB]
CLOBBERED = [DM, CNT, PASS] + [r for lane in LANE_REGS for r in lane.values()]

# Parameter block at MB, in bytes: the m-block (8 pairs), K = n'·2^36 per lane, N01 = n_0 + n_1·2^28
# per lane, the output pointer, P, 16·(D - 1), and 16·D.
K_SLOT, N01_SLOT, OUT_SLOT = 64, 80, 96
BLOCKS_SLOT, BACK_SLOT, STRIDE_SLOT = 104, 112, 120
PARAM_BYTES = 128


def _w(reg):
    return 'w' + reg[1:]


def _mac(dst, fresh, pair, r, stream):
    """dst += digit r of a packed block times the replicated stream digit."""
    op, arrangement = ('umlal', '.2s') if r % 2 == 0 else ('umlal2', '.4s')
    if fresh:
        op = op.replace('umlal', 'umull')
    return f'{op} v{dst}.2d, v{pair}{arrangement}, v{stream}{arrangement}'


class _Emitter:
    def __init__(self):
        self.lines = []
        self.label = 10

    def emit(self, *lines):
        self.lines.extend(lines)

    def new_label(self):
        self.label += 1
        return self.label


class _Window:
    """Per slot, whether the A and B accumulators hold a value."""

    def __init__(self):
        self.a = [False] * W
        self.b = [False] * W

    def snapshot(self):
        return (tuple(self.a), tuple(self.b))

    def restore(self, snapshot):
        self.a, self.b = list(snapshot[0]), list(snapshot[1])


def triangle_digit(k, c):
    """Scalar code for multiplier k of the next m-block, both lanes, placed after step c >= k + 1.

    Column j of the block is flushed by step j, so after step c it sits at [TP, #16·(j - c - 1)].
    Digit k forms x_k = P'_k + h_(k-1), then m_k·2^36 = x_k·K mod 2^64 and h_k = umulh(m_k·2^36,
    N01). It prepares P'_(k+1) = T_(k+1) + sum(m_r·n_(k+1-r), r <= k - 1) + (x_k >> 28) +
    [x_k mod 2^28 != 0]: because x_k + m_k·n_0 is a multiple of 2^28, its high part is
    (x_k >> 28) + floor(m_k·n_0 / 2^28) + [x_k mod 2^28 != 0], and h_k supplies the middle term.
    The chain from x_k to x_(k+1) is one MUL, one UMULH and one ADD.
    """
    p, q = LANE_REGS
    lanes = (p, q)

    def col(j):
        return 16 * (j - c - 1)

    out = []
    if k == 0:
        out += [
            f"ldp {p['K']}, {q['K']}, [{MB}, #{K_SLOT}]",
            f"ldp {p['N01']}, {q['N01']}, [{MB}, #{N01_SLOT}]",
            f"ldp {p['P']}, {q['P']}, [{TP}, #{col(0)}]",
        ]
    # For k = 0, H holds the carry into the block.
    out += [f"add {g['X']}, {g['P']}, {g['H']}" for g in lanes]
    out += [f"mul {g['MS']}, {g['X']}, {g['K']}" for g in lanes]
    if k < W - 1:
        out.append(f"ldp {p['P']}, {q['P']}, [{TP}, #{col(k + 1)}]")
        for r in range(k - 1):
            out += [
                f"ldp {_w(p['QA'])}, {_w(q['QA'])}, [{MB}, #{8 * r}]",
                f"ldp {_w(p['QB'])}, {_w(q['QB'])}, [{NB}, #{16 * (k + 1 - r)}]",
            ]
            out += [f"madd {g['P']}, {g['QA']}, {g['QB']}, {g['P']}" for g in lanes]
        if k >= 1:
            # m_(k-1)·n_2 from the register, before M takes m_k.
            out.append(f"ldp {_w(p['QB'])}, {_w(q['QB'])}, [{NB}, #32]")
            out += [f"madd {g['P']}, {g['M']}, {g['QB']}, {g['P']}" for g in lanes]
        out += [f"umulh {g['H']}, {g['MS']}, {g['N01']}" for g in lanes]
    out += [f"lsr {g['M']}, {g['MS']}, #36" for g in lanes]
    out += [f"orr {MM}, {p['M']}, {q['M']}, lsl #32", f'str {MM}, [{MB}, #{8 * k}]']
    if k < W - 1:
        for g in lanes:
            out += [
                f"add {g['P']}, {g['P']}, {g['X']}, lsr #28",
                f"tst {g['X']}, #{MASK28:#x}",
                f"cinc {g['P']}, {g['P']}, ne",
            ]
    else:
        # The carry out of the block, (x_7 + m_7·n_0) >> 28.
        out.append(f"ldp {_w(p['QB'])}, {_w(q['QB'])}, [{NB}]")
        for g in lanes:
            out += [f"madd {g['H']}, {g['M']}, {g['QB']}, {g['X']}", f"lsr {g['H']}, {g['H']}, #28"]
    return out


def _step(e, win, i, product, reduction, flush, enter):
    """One step at slot pattern i = c mod W.

    product: None, 'full', or 'diag' (squaring's diagonal block, where stream digit i of the block
    pairs with block digits r < i and squares itself); reduction: None, 'full', or ('pre', t),
    the reduction contributions of n_t (1 <= t < W) to columns at or above the pass's first;
    flush: None, 'plain' (to T), or 'norm' (to T and the output digits); enter: None, 'load', or
    'fresh' for the column that takes slot i after the flush.
    """
    lines = []
    contrib_a = []
    contrib_b = []
    if product is not None:
        lines.append(f'ldr q{VS}, [{SP}], #16')
        for r in range(W):
            slot = (i + r) % W
            if product == 'full' or r < i:
                contrib_a.append((slot, 'pair', r))
            elif r == i:
                contrib_a.append((slot, 'square', r))
    if reduction is not None:
        lines.append(f'ldr q{VN}, [{NP}], #16')
        first = 0 if reduction == 'full' else W - reduction[1]
        contrib_b = [((i + r) % W, r) for r in range(first, W)]
    # A step with both kinds sends the product to A and the reduction to B. A single-kind step
    # alternates by parity, so each register's chain advances every other step.
    single = None if (contrib_a and contrib_b) else ('a' if i % 2 == 0 else 'b')
    for slot, kind, r in contrib_a:
        regs, state = (A, win.a) if (single or 'a') == 'a' else (B, win.b)
        fresh = not state[slot]
        if kind == 'pair':
            lines.append(_mac(regs[slot], fresh, VA[r // 2], r, VS))
        else:
            lines.append(f"{'umull' if fresh else 'umlal'} v{regs[slot]}.2d, v{VS}.2s, v{VS}.2s")
        state[slot] = True
    for slot, r in contrib_b:
        regs, state = (A, win.a) if (single or 'b') == 'a' else (B, win.b)
        lines.append(_mac(regs[slot], not state[slot], VM[r // 2], r, VN))
        state[slot] = True
    slot = i % W
    if flush is not None:
        if win.a[slot] and win.b[slot]:
            lines.append(f'add v{B[slot]}.2d, v{B[slot]}.2d, v{A[slot]}.2d')
        elif not (win.a[slot] or win.b[slot]):
            raise ValueError('flush of an empty column')
        source = B[slot] if win.b[slot] else A[slot]
        lines.append(f'str q{source}, [{TP}], #16')
        if flush == 'norm':
            # Add the running carry, keep 28 bits, and write the replicated digit pair.
            p, q = LANE_REGS
            lines.append(f"ldp {p['X']}, {q['X']}, [{TP}, #-16]")
            for g in (p, q):
                lines += [
                    f"add {g['X']}, {g['X']}, {g['H']}",
                    f"and {g['M']}, {g['X']}, #{MASK28:#x}",
                    f"lsr {g['H']}, {g['X']}, #28",
                ]
            lines += [f"orr {DM}, {p['M']}, {q['M']}, lsl #32", f'stp {DM}, {DM}, [{AP}], #16']
        win.a[slot] = False
        win.b[slot] = False
        if enter == 'load':
            lines.append(f'ldr q{B[slot]}, [{TP}, #{16 * (W - 1)}]')
            win.b[slot] = True
    e.emit(*lines)


def _block(e, win, product, reduction, flush, enter, side=None, loop_body=False):
    before = win.snapshot()
    for i in range(W):
        _step(e, win, i, product, reduction, flush, enter)
        if side:
            e.emit(*side[i])
    if loop_body and win.snapshot() != before:
        raise ValueError('a loop body changes the window state')


def _counted(e, minus, body):
    """Run body P - minus times, skipped when that is zero."""
    skip = e.new_label()
    e.emit(f'ldr {CNT}, [{MB}, #{BLOCKS_SLOT}]', f'subs {CNT}, {CNT}, #{minus}', f'b.eq {skip}f')
    top = e.new_label()
    e.emit(f'{top}:')
    body()
    e.emit(f'subs {CNT}, {CNT}, #1', f'b.ne {top}b', f'{skip}:')


def _back(e, reg, slot):
    e.emit(f'ldr {DM}, [{MB}, #{slot}]', f'sub {reg}, {reg}, {DM}')


def _load_window(e, win, loaded):
    if loaded:
        for i in range(0, W, 2):
            e.emit(f'ldp q{B[i]}, q{B[i + 1]}, [{TP}, #{16 * i}]')
    win.a = [False] * W
    win.b = [loaded] * W


def _load_m_block(e):
    e.emit(f'ldp q{VM[0]}, q{VM[1]}, [{MB}]', f'ldp q{VM[2]}, q{VM[3]}, [{MB}, #32]')


def _load_a_block(e, double):
    # LD2 needs consecutive registers. Each load also writes a duplicate into the next register,
    # which the following load, or the m-block load for the last, overwrites.
    for i in range(4):
        e.emit(f'ld2 {{{{v{VA[i]}.2d, v{VA[i] + 1}.2d}}}}, [{AP}], #32')
    if double:
        for i in range(4):
            e.emit(f'shl v{VA[i]}.4s, v{VA[i]}.4s, #1')


def _pre_steps(e, win):
    for t in range(1, W):
        _step(e, win, t, None, ('pre', t), None, None)


def _drain(e, win):
    """Flush the W - 1 product-only columns after the last stream digit, then zero the next."""
    for i in range(W - 1):
        _step(e, win, i, None, None, 'plain', None)
    e.emit(f'stp xzr, xzr, [{TP}]')


def _triangle_schedule():
    """Side code for the steps of blocks 0 and 1: digit k after step 2k + 1."""
    side = [[] for _ in range(2 * W)]
    for k in range(W):
        side[2 * k + 1] += triangle_digit(k, 2 * k + 1)
    return side


def kernel(square):
    """Template lines of the multiply (`square` false) or square kernel for any P >= 3."""
    e = _Emitter()
    win = _Window()
    side = _triangle_schedule()
    p, q = LANE_REGS
    e.emit(f"mov {p['H']}, xzr", f"mov {q['H']}, xzr")
    # Pass -1: product strip 0 into fresh columns, and m-block 0.
    if square:
        e.emit(f'mov {SP}, {AP}')
    _load_a_block(e, square)
    _load_window(e, win, False)
    _block(e, win, 'diag' if square else 'full', None, 'plain', 'fresh', side=side[:W])
    _block(e, win, 'full', None, 'plain', 'fresh', side=side[W:])
    _counted(e, 2, lambda: _block(e, win, 'full', None, 'plain', 'fresh', loop_body=True))
    _drain(e, win)
    _back(e, TP, BACK_SLOT)
    if not square:
        _back(e, SP, STRIDE_SLOT)
    # Passes s = 0..P-2.
    e.emit(f'mov {PASS}, xzr')
    pass_top = e.new_label()
    e.emit(f'{pass_top}:')
    if square:
        e.emit(f'mov {SP}, {AP}')
    _load_a_block(e, square)
    _load_m_block(e)
    _load_window(e, win, True)
    _pre_steps(e, win)
    if not square:
        _block(e, win, 'full', 'full', 'plain', 'load', side=side[:W])
        _block(e, win, 'full', 'full', 'plain', 'load', side=side[W:])
        _counted(e, 3, lambda: _block(e, win, 'full', 'full', 'plain', 'load', loop_body=True))
        _block(e, win, 'full', None, 'plain', 'fresh')
        _drain(e, win)
    else:
        # Blocks 0..s only reduce; block s + 1 holds the diagonal; later blocks do both.
        _block(e, win, None, 'full', 'plain', 'load', side=side[:W])
        after_first = win.snapshot()
        later = e.new_label()
        done = e.new_label()
        e.emit(f'cbnz {PASS}, {later}f')
        _block(e, win, 'diag', 'full', 'plain', 'load', side=side[W:])
        _counted(e, 3, lambda: _block(e, win, 'full', 'full', 'plain', 'load', loop_body=True))
        _block(e, win, 'full', None, 'plain', 'fresh')
        end_state = win.snapshot()
        _drain(e, win)
        e.emit(f'b {done}f', f'{later}:')
        win.restore(after_first)
        _block(e, win, None, 'full', 'plain', 'load', side=side[W:])
        skip = e.new_label()
        e.emit(f'subs {CNT}, {PASS}, #1', f'b.eq {skip}f')
        top = e.new_label()
        e.emit(f'{top}:')
        _block(e, win, None, 'full', 'plain', 'load', loop_body=True)
        e.emit(f'subs {CNT}, {CNT}, #1', f'b.ne {top}b', f'{skip}:')
        before_diagonal = win.snapshot()
        last = e.new_label()
        e.emit(f'ldr {DM}, [{MB}, #{BLOCKS_SLOT}]', f'sub {DM}, {DM}, #2', f'cmp {PASS}, {DM}', f'b.eq {last}f')
        _block(e, win, 'diag', 'full', 'plain', 'load')
        skip = e.new_label()
        e.emit(f'ldr {CNT}, [{MB}, #{BLOCKS_SLOT}]', f'sub {CNT}, {CNT}, #3', f'subs {CNT}, {CNT}, {PASS}',
               f'b.eq {skip}f')
        top = e.new_label()
        e.emit(f'{top}:')
        _block(e, win, 'full', 'full', 'plain', 'load', loop_body=True)
        e.emit(f'subs {CNT}, {CNT}, #1', f'b.ne {top}b', f'{skip}:')
        _block(e, win, 'full', None, 'plain', 'fresh')
        if win.snapshot() != end_state:
            raise ValueError('squaring paths disagree before the drain')
        _drain(e, win)
        e.emit(f'b {done}f', f'{last}:')
        # s = P - 2: the diagonal block is the last block, after the reduction ends.
        win.restore(before_diagonal)
        _block(e, win, 'diag', None, 'plain', 'fresh')
        _drain(e, win)
        e.emit(f'{done}:')
    _back(e, TP, BACK_SLOT)
    _back(e, NP, BACK_SLOT)
    if not square:
        _back(e, SP, STRIDE_SLOT)
    e.emit(f'add {PASS}, {PASS}, #1', f'ldr {DM}, [{MB}, #{BLOCKS_SLOT}]', f'sub {DM}, {DM}, #1',
           f'cmp {PASS}, {DM}', f'b.ne {pass_top}b')
    # Final pass: reduction strip P - 1; columns D..2D-1 become the output digits.
    _load_m_block(e)
    e.emit(f'ldr {AP}, [{MB}, #{OUT_SLOT}]')
    _load_window(e, win, True)
    _pre_steps(e, win)
    _counted(e, 1, lambda: _block(e, win, None, 'full', 'norm', 'load', loop_body=True))
    for i in range(W):
        _step(e, win, i, None, None, 'norm', None)
    return e.lines


def mul_kernel():
    return (kernel(False),)


def square_kernel():
    return (kernel(True),)


KERNELS = {
    'mul_kernel': mul_kernel,
    'square_kernel': square_kernel,
}
