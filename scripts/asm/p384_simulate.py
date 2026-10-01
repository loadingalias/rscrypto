"""Check the generated P-384 kernels against Python integers with bit-exact simulators.

The x86-64 simulator models RDX, CF, OF (separately, for ADCX/ADOX), and ZF; memory operands
name their base placeholder. The AArch64 simulator models C and Z. Both raise on any
instruction they do not model, so an unsupported instruction cannot pass silently.
This is independent evidence for the committed text: `p384.py check` proves the source equals
the generator output, and this proves the generator output computes the specified values.
"""

import random
import re

MASK = (1 << 64) - 1
P = 2**384 - 2**128 - 2**96 + 2**32 - 1
R = 2**384
R_INVERSE = pow(R, -1, P)
REDUCTION_CONSTANTS = [0xFFFFFFFF00000001, 0xFFFFFFFF]


def limbs(value, count=6):
    return [(value >> (64 * i)) & MASK for i in range(count)]


def number(values):
    return sum(v << (64 * i) for i, v in enumerate(values))


def _signed(value):
    return value - (1 << 64) if value >> 63 else value


def run_x86(lines, regs, memory):
    """Execute Intel-syntax template lines; `regs` and `memory` are updated in place.

    Flags follow the architecture for every modeled instruction: ADD/ADC/SUB/SBB write CF,
    OF, and ZF; ADCX writes only CF and ADOX only OF; logic operations and TEST clear CF and
    OF; shifts write CF and leave OF undefined for counts above one. Reading an undefined
    flag raises, so a chain that depends on clobbered flags cannot pass.
    """
    flags = {'cf': 0, 'of': 0, 'zf': 0}
    address = re.compile(r'qword ptr \[(\w+)(?: \+ (\d+))?\]')
    line = ''

    def flag(name):
        if flags[name] is None:
            raise ValueError(f'reads undefined {name.upper()}: {line}')
        return flags[name]

    def read(operand):
        if operand.startswith('qword ptr'):
            match = address.fullmatch(operand)
            return memory[match.group(1)][int(match.group(2) or 0) // 8]
        if re.fullmatch(r'-?\d+', operand):
            return int(operand) & MASK
        if operand.endswith(':e'):
            return regs[operand[:-2]] & 0xFFFFFFFF
        return regs[operand]

    def write(operand, value):
        if operand.startswith('qword ptr'):
            match = address.fullmatch(operand)
            memory[match.group(1)][int(match.group(2) or 0) // 8] = value & MASK
        elif operand.endswith(':e'):
            regs[operand[:-2]] = value & 0xFFFFFFFF  # 32-bit writes zero-extend
        else:
            regs[operand] = value & MASK

    for line in lines:
        text = line.replace('{', '').replace('}', '').replace('rip + ', '')
        op, _, rest = text.partition(' ')
        ops = [o.strip() for o in re.split(r',(?![^\[]*\])', rest)]
        if op == 'mov':
            write(ops[0], read(ops[1]))
        elif op == 'mulx':
            product = regs['rdx'] * read(ops[2])
            write(ops[1], product)
            write(ops[0], product >> 64)
        elif op in ('xor', 'and', 'or'):
            left, right = read(ops[0]), read(ops[1])
            value = {'xor': left ^ right, 'and': left & right, 'or': left | right}[op]
            write(ops[0], value)
            flags.update(cf=0, of=0, zf=int(value & MASK == 0))
        elif op in ('add', 'adc'):
            left, right = read(ops[0]), read(ops[1])
            carry_in = flag('cf') if op == 'adc' else 0
            total = left + right + carry_in
            result = total & MASK
            flags.update(cf=total >> 64, of=int(_signed(left) + _signed(right) + carry_in != _signed(result)),
                         zf=int(result == 0))
            write(ops[0], result)
        elif op == 'adcx':
            total = read(ops[0]) + read(ops[1]) + flag('cf')
            flags['cf'] = total >> 64
            write(ops[0], total)
        elif op == 'adox':
            total = read(ops[0]) + read(ops[1]) + flag('of')
            flags['of'] = total >> 64
            write(ops[0], total)
        elif op in ('sub', 'sbb'):
            left, right = read(ops[0]), read(ops[1])
            borrow_in = flag('cf') if op == 'sbb' else 0
            total = left - right - borrow_in
            result = total & MASK
            flags.update(cf=int(total < 0), of=int(_signed(left) - _signed(right) - borrow_in != _signed(result)),
                         zf=int(result == 0))
            write(ops[0], result)
        elif op in ('shl', 'shr'):
            original, count = read(ops[0]), int(ops[1])
            assert 0 < count < 64, line
            if op == 'shl':
                result = original << count & MASK
                carry = original >> (64 - count) & 1
                overflow = (result >> 63) ^ carry
            else:
                result = original >> count
                carry = original >> (count - 1) & 1
                overflow = original >> 63
            flags.update(cf=carry, of=overflow if count == 1 else None, zf=int(result == 0))
            write(ops[0], result)
        elif op == 'test':
            flags.update(cf=0, of=0, zf=int(read(ops[0]) & read(ops[1]) == 0))
        elif op in ('cmovz', 'cmovnz'):
            value = read(ops[1])  # the source operand is read whether or not it moves
            if flag('zf') == (op == 'cmovz'):
                write(ops[0], value)
        else:
            raise ValueError(f'unmodeled x86 instruction: {line}')


def run_aarch64(lines, regs):
    """Execute A64 template lines on 64-bit registers; `regs` is updated in place."""
    flags = {'c': 0, 'z': 0}

    def value(operand):
        if operand == 'xzr':
            return 0
        if operand.startswith('#'):
            return int(operand[1:], 0) & MASK
        return regs[operand.replace(':w', '')]

    for line in lines:
        op, _, rest = line.partition(' ')
        parts = [p.strip().replace('{', '').replace('}', '') for p in rest.split(',')]
        target, ops = parts[0], parts[1:]
        narrow = target.endswith(':w')
        target = target.replace(':w', '')
        if op == 'mul':
            regs[target] = value(ops[0]) * value(ops[1]) & MASK
        elif op == 'umulh':
            regs[target] = value(ops[0]) * value(ops[1]) >> 64
        elif op in ('add', 'adds'):
            right = value(ops[1])
            if len(ops) > 2:
                assert ops[2].startswith('lsl #'), line
                right = right << int(ops[2][5:]) & MASK
            total = value(ops[0]) + right
            if op == 'adds':
                flags.update(c=total >> 64, z=int(total & MASK == 0))
            regs[target] = total & MASK
        elif op in ('adc', 'adcs'):
            total = value(ops[0]) + value(ops[1]) + flags['c']
            if op == 'adcs':
                flags.update(c=total >> 64, z=int(total & MASK == 0))
            regs[target] = total & MASK
        elif op == 'sub':
            regs[target] = (value(ops[0]) - value(ops[1])) & MASK
        elif op in ('subs', 'cmp'):
            left, right = (value(target), value(ops[0])) if op == 'cmp' else (value(ops[0]), value(ops[1]))
            flags.update(c=int(left >= right), z=int(left == right))
            if op == 'subs':
                regs[target] = (left - right) & MASK
        elif op in ('sbc', 'sbcs'):
            total = value(ops[0]) - value(ops[1]) - (1 - flags['c'])
            if op == 'sbcs':
                flags.update(c=int(total >= 0), z=int(total & MASK == 0))
            regs[target] = total & MASK
        elif op == 'cset':
            regs[target] = {'hs': flags['c'], 'lo': 1 - flags['c']}[ops[0]]
        elif op == 'mov':
            regs[target] = int(ops[0][1:]) & (0xFFFFFFFF if narrow else MASK)
        elif op == 'neg':
            regs[target] = -value(ops[0]) & MASK
        elif op == 'csel':
            assert ops[2] == 'ne', line
            regs[target] = value(ops[0]) if not flags['z'] else value(ops[1])
        else:
            raise ValueError(f'unmodeled AArch64 instruction: {line}')


def montgomery(a, b):
    return a * b * R_INVERSE % P


BOUNDARY_LIMBS = [0, 1, 2, MASK, MASK - 1, 1 << 63, (1 << 63) - 1, 0xFFFFFFFF, 0xFFFFFFFF00000000]


def operands(rng, count):
    """Canonical edges, values near p, boundary-limb values, then uniform values below p.

    Boundary limbs (0, 1, all ones, 2^63, and 32-bit halves) drive carry and borrow chains
    that uniform limbs almost never reach.
    """
    edges = [0, 1, 2, P - 1, P - 2, R % P, R * R % P, 2**383 % P, (P - 1) // 2, 2**64 - 1, 2**256 % P]
    values = edges + [P - 1 - rng.randrange(2**64) for _ in range(count // 10)]
    while len(values) < len(edges) + count // 10 + count // 3:
        value = number([rng.choice(BOUNDARY_LIMBS) if rng.random() < 0.8 else rng.getrandbits(64) for _ in range(6)])
        if value < P:
            values.append(value)
    return values + [rng.randrange(P) for _ in range(count)]


def small_multiple_cases(values, rng):
    """(value, k) pairs for `value * k`, including carries random limbs almost never reach.

    Adding the previous limb's high word h < k to `limb * k mod 2^64` carries only when that
    low word is within k of 2^64, which uniform limbs hit with probability about k / 2^64.
    For odd k, limbs equal to `(2^64 - 1) / k mod 2^64` make every low word all ones, so any
    nonzero incoming high word carries. For even k the carry is impossible.
    """
    cases = [(value, 1 + index % 8) for index, value in enumerate(values)]
    for k in (3, 5, 7):
        limb = MASK * pow(k, -1, 1 << 64) & MASK
        for _ in range(4):
            crafted = [MASK, limb, limb, limb, limb, limb]  # the top limb stays below p's
            for hole in range(1, 6):
                variant = crafted[:]
                variant[hole] = rng.getrandbits(64)
                cases.append((number(variant) % P, k))
            cases.append((number(crafted) % P, k))
    # Products whose limbs 1..5 are all ones before the fold, so folding `top * (2^384 - p)`
    # carries through every limb: v = T / k for T = top * 2^384 + (2^384 - 2^64) + low.
    for k in range(2, 9):
        for top in range(1, k - 1):
            base = top * R + (R - (1 << 64))
            low = (-base) % k
            if low < (1 << 64) and (base + low) // k < P:
                cases.append(((base + low) // k, k))
    # Products p + r with small r: the folded value lies in [p, 2^384), so adding
    # 2^384 - p in the final correction carries through limbs 1..5.
    for k in range(2, 9):
        for r in (0, 1, 2, 1 << 32, 1 << 64, 1 << 100):
            r += (-(P + r)) % k
            cases.append(((P + r) // k, k))
    return cases


def _square_root(value):
    """A square root modulo p (p = 3 mod 4), or None."""
    root = pow(value, (P + 1) // 4, P)
    return root if root * root % P == value else None


def multiplication_cases(values, rng):
    """Operand pairs, including pairs whose Montgomery product is small.

    The final step adds `c = 2^384 - p` to the unreduced `T < 2p` and keeps the sum when it
    carries. Carries through limbs 3..5 need `T` in `[2^384 - c, 2^384)`, which uniform
    operands hit with probability about 2^-256. A small result `r` with `T = r + p` lands
    there, and `b = r * R / a` makes `a * b * R^-1 = r` for any chosen `a`.
    """
    pairs = [(value, values[(index * 7 + 3) % len(values)]) for index, value in enumerate(values)]
    for _ in range(max(8, len(values) // 10)):
        a = rng.randrange(1, P)
        result = rng.getrandbits(rng.choice((1, 32, 64, 100)))
        pairs.append((a, result * R * pow(a, -1, P) % P))
    return pairs


def square_cases(values, rng):
    """Square inputs, including inputs whose Montgomery square is small (see above)."""
    squares = list(values)
    while len(squares) < len(values) + max(8, len(values) // 10):
        root = _square_root(rng.getrandbits(rng.choice((1, 32, 64, 100))) * R % P)
        if root is not None:
            squares.append(root)
    return squares


def _random_registers(rng, names):
    return {name: rng.getrandbits(64) for name in names}


X86_SCRATCH = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'lo', 'hi', 'rdx', 'a', 'b', 'buf', 'h', 't', 'm', 'm0',
               'm1', 'cv', 'cs', 't0', 't1', 't2', 's1', 's2']
AARCH64_SCRATCH = [f'w{i}' for i in range(12)] + ['x', 'q', 'b', 'k1', 'k2', 'h0', 'c0', 'c1', 'h', 'top', 'cv',
                                                   'm1'] + [f't{i}' for i in range(6)]


def check_x86(generators, cases, rng):
    """Return a list of failure strings for the x86-64 kernels."""
    failures = []
    values = operands(rng, cases)
    memory_constants = {'consts': REDUCTION_CONSTANTS, 'modulus': limbs(P)}

    def outputs(regs, names):
        return number([regs[n.strip('{}')] for n in names])

    def montgomery_kernel(name, left, right):
        lines, out = generators[name]()
        regs = _random_registers(rng, X86_SCRATCH)
        memory = {'a': limbs(left), 'b': limbs(right), 'buf': [rng.getrandbits(64) for _ in range(12)],
                  **memory_constants}
        run_x86(lines, regs, memory)
        if outputs(regs, out) != montgomery(left, right):
            failures.append(f'x86 {name}({left:#x}, {right:#x})')

    for left, right in multiplication_cases(values, rng):
        montgomery_kernel('montgomery_mul_bmi2_adx', left, right)
    for value in square_cases(values, rng):
        montgomery_kernel('montgomery_square_bmi2_adx', value, value)
    for index, left in enumerate(values):
        right = values[(index * 7 + 3) % len(values)]
        for name, expected in (('add_mod', (left + right) % P), ('sub_mod', (left - right) % P)):
            lines, out = generators[name]()
            regs = _random_registers(rng, X86_SCRATCH)
            regs.update({f'd{j}': limb for j, limb in enumerate(limbs(left))})
            memory = {'right': limbs(right), **memory_constants}
            run_x86(lines, regs, memory)
            if outputs(regs, out) != expected:
                failures.append(f'x86 {name}({left:#x}, {right:#x})')
    lines, out = generators['mul_small_bmi2']()
    for value, k in small_multiple_cases(values, rng):
        regs = _random_registers(rng, X86_SCRATCH)
        regs.update({f'd{j}': limb for j, limb in enumerate(limbs(value))}, rdx=k)
        run_x86(lines, regs, dict(memory_constants))
        if outputs(regs, out) != value * k % P:
            failures.append(f'x86 mul_small_bmi2({value:#x}, k={k})')
    return failures


def check_x86_double(point_double, cases, rng):
    """The fused doubling must equal the portable a = -3 formula on canonical coordinates."""
    lines, frame = point_double()
    failures = []
    values = operands(rng, cases)

    def reference(x, y, z):
        delta, gamma = montgomery(z, z), montgomery(y, y)
        beta = montgomery(x, gamma)
        alpha = 3 * montgomery((x - delta) % P, (x + delta) % P) % P
        x3 = (montgomery(alpha, alpha) - 8 * beta) % P
        z3 = (montgomery((y + z) % P, (y + z) % P) - gamma - delta) % P
        y3 = (montgomery(alpha, (4 * beta - x3) % P) - 8 * montgomery(gamma, gamma)) % P
        return x3, y3, z3

    triples = [(a, b, c) for a in values[:11] for b in values[:11] for c in (0, 1, P - 1)]
    triples += [tuple(rng.choice(values) for _ in range(3)) for _ in range(cases)]
    for x, y, z in triples:
        memory = {'p': limbs(x) + limbs(y) + limbs(z), 'rsp': [rng.getrandbits(64) for _ in range(frame // 8)],
                  'consts': REDUCTION_CONSTANTS, 'modulus': limbs(P)}
        regs = _random_registers(rng, ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'lo', 'hi', 's1', 's2', 'rdx'])
        regs['rsp'] = 0
        run_x86(lines, regs, memory)
        got = number(memory['p'][0:6]), number(memory['p'][6:12]), number(memory['p'][12:18])
        if regs['rsp'] != 0 or got != reference(x, y, z):
            failures.append(f'x86 point_double_bmi2_adx({x:#x}, {y:#x}, {z:#x})')
    return failures


def check_aarch64(generators, cases, rng):
    """Return a list of failure strings for the AArch64 kernels."""
    failures = []
    values = operands(rng, cases)
    lines, out = generators['montgomery_mul']()
    for left, right in multiplication_cases(values, rng):
        regs = _random_registers(rng, AARCH64_SCRATCH)
        regs.update({f'l{j}': v for j, v in enumerate(limbs(left))}, **{f'b{j}': v for j, v in enumerate(limbs(right))})
        run_aarch64(lines, regs)
        if number([regs[n.strip('{}')] for n in out]) != montgomery(left, right):
            failures.append(f'aarch64 montgomery_mul({left:#x}, {right:#x})')
    lines, out = generators['montgomery_square']()
    for value in square_cases(values, rng):
        regs = _random_registers(rng, AARCH64_SCRATCH)
        regs.update({f'a{j}': v for j, v in enumerate(limbs(value))})
        run_aarch64(lines, regs)
        if number([regs[n.strip('{}')] for n in out]) != montgomery(value, value):
            failures.append(f'aarch64 montgomery_square({value:#x})')
    lines, out = generators['mul_small']()
    for value, k in small_multiple_cases(values, rng):
        regs = _random_registers(rng, AARCH64_SCRATCH)
        regs.update({f'a{j}': v for j, v in enumerate(limbs(value))}, k=k)
        run_aarch64(lines, regs)
        if number([regs[n.strip('{}')] for n in out]) != value * k % P:
            failures.append(f'aarch64 mul_small({value:#x}, k={k})')
    return failures


def simulate(x86, aarch64, cases, seed=384):
    rng = random.Random(seed)
    return (check_x86(x86.KERNELS, cases, rng) + check_x86_double(x86.point_double, cases // 10, rng)
            + check_aarch64(aarch64.KERNELS, cases, rng))
