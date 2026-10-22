"""Check the generated paired RSA kernels against Python integers in a bit-exact AArch64 model.

The model covers exactly the instructions the generator emits and raises on anything else, so an
unsupported instruction cannot pass silently. Memory is a set of bounds-checked regions laid out
as the Rust caller lays them out: an access outside its region, or a write to an input, fails.
Registers and the column buffer start with random contents, so a kernel that reads state it
never wrote cannot pass.

The reference is digit-serial almost-Montgomery reduction with radix 2^28: for k = 0..D-1,
m_k = ((x >> 28·k) mod 2^28)·n' mod 2^28 and x += m_k·n·2^(28·k), starting from x = a·b. The
kernels must produce exactly x >> 28·D, in canonical digits, for each lane.
"""

import random
import re

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
MASK28 = (1 << 28) - 1


class Memory:
    def __init__(self):
        self.regions = []
        self.next = 0x10000

    def alloc(self, name, data, writable):
        base = self.next
        self.next += ((len(data) + 0xFFF) & ~0xFFF) + 0x1000
        self.regions.append((base, bytearray(data), name, writable))
        return base

    def _find(self, addr, size, write):
        for base, buf, name, writable in self.regions:
            if base <= addr and addr + size <= base + len(buf):
                if write and not writable:
                    raise MemoryError(f'write to {name}')
                return buf, addr - base
        raise MemoryError(f'{size}-byte access at {addr:#x} outside every region')

    def read(self, addr, size):
        buf, off = self._find(addr, size, False)
        return int.from_bytes(buf[off:off + size], 'little')

    def write(self, addr, size, value):
        buf, off = self._find(addr, size, True)
        buf[off:off + size] = (value & ((1 << (8 * size)) - 1)).to_bytes(size, 'little')

    def region(self, base):
        return next(buf for b, buf, _, _ in self.regions if b == base)


def _lanes(value, bits):
    mask = (1 << bits) - 1
    return [(value >> (bits * i)) & mask for i in range(128 // bits)]


def _pack(values, bits):
    mask = (1 << bits) - 1
    return sum((v & mask) << (bits * i) for i, v in enumerate(values))


class Cpu:
    def __init__(self, memory, rng):
        self.x = {f'x{i}': rng.getrandbits(64) for i in range(31)}
        self.v = [rng.getrandbits(128) for _ in range(32)]
        self.memory = memory
        self.z = rng.getrandbits(1)
        self.executed = 0

    def get(self, name):
        if name in ('xzr', 'wzr'):
            return 0
        if name.startswith('w'):
            return self.x['x' + name[1:]] & M32
        return self.x[name]

    def set(self, name, value):
        if name in ('xzr', 'wzr'):
            return
        if name.startswith('w'):
            self.x['x' + name[1:]] = value & M32
        else:
            self.x[name] = value & M64

    @staticmethod
    def imm(token):
        if not token.startswith('#'):
            raise ValueError(f'expected an immediate: {token}')
        return int(token[1:], 0)

    @staticmethod
    def vreg(token):
        match = re.fullmatch(r'v(\d+)\.(2d|2s|4s)', token)
        if not match:
            raise ValueError(f'expected a vector register: {token}')
        return int(match.group(1)), match.group(2)

    def address(self, token, post):
        parts = [part.strip() for part in token[1:-1].split(',')]
        addr = (self.get(parts[0]) + (self.imm(parts[1]) if len(parts) > 1 else 0)) & M64
        if post is not None:
            self.set(parts[0], self.get(parts[0]) + post)
        return addr

    def run(self, lines, limit=50_000_000):
        code = []
        labels = {}
        for line in lines:
            line = line.replace('{{', '{').replace('}}', '}')
            match = re.fullmatch(r'(\d+):', line)
            if match:
                labels.setdefault(int(match.group(1)), []).append(len(code))
            else:
                code.append(line)
        pc = 0
        while pc < len(code):
            self.executed += 1
            if self.executed > limit:
                raise RuntimeError('instruction limit')
            target = self.execute(code[pc], labels, pc)
            pc = pc + 1 if target is None else target

    @staticmethod
    def operands(text):
        out, depth, current = [], 0, ''
        for ch in text:
            depth += ch in '[{'
            depth -= ch in ']}'
            if ch == ',' and depth == 0:
                out.append(current.strip())
                current = ''
            else:
                current += ch
        if current.strip():
            out.append(current.strip())
        return out

    def execute(self, line, labels, pc):
        op, _, rest = line.partition(' ')
        handler = getattr(self, 'op_' + op.replace('.', '_'), None)
        if handler is None:
            raise NotImplementedError(line)
        return handler(self.operands(rest), labels, pc)

    def _target(self, token, labels, pc):
        match = re.fullmatch(r'(\d+)([bf])', token)
        candidates = labels[int(match.group(1))]
        if match.group(2) == 'b':
            return max(c for c in candidates if c <= pc)
        return min(c for c in candidates if c > pc)

    # Scalar.
    def op_mov(self, o, *_):
        self.set(o[0], self.imm(o[1]) if o[1].startswith('#') else self.get(o[1]))

    def _shifted(self, o):
        value = self.get(o[2])
        if len(o) == 4:
            kind, amount = o[3].split()
            value = {'lsr': value >> self.imm(amount), 'lsl': (value << self.imm(amount)) & M64}[kind]
        return value

    def op_add(self, o, *_):
        if o[0].startswith('v'):
            d, arrangement = self.vreg(o[0])
            n, _ = self.vreg(o[1])
            m, _ = self.vreg(o[2])
            if arrangement != '2d':
                raise NotImplementedError(o)
            self.v[d] = _pack([a + b for a, b in zip(_lanes(self.v[n], 64), _lanes(self.v[m], 64))], 64)
            return
        addend = self.imm(o[2]) if o[2].startswith('#') else self._shifted(o)
        self.set(o[0], self.get(o[1]) + addend)

    def op_sub(self, o, *_):
        self.set(o[0], self.get(o[1]) - (self.imm(o[2]) if o[2].startswith('#') else self.get(o[2])))

    def op_subs(self, o, *_):
        value = (self.get(o[1]) - (self.imm(o[2]) if o[2].startswith('#') else self.get(o[2]))) & M64
        self.z = int(value == 0)
        self.set(o[0], value)

    def op_cmp(self, o, *_):
        self.z = int(self.get(o[0]) == (self.imm(o[1]) if o[1].startswith('#') else self.get(o[1])))

    def op_tst(self, o, *_):
        self.z = int(self.get(o[0]) & self.imm(o[1]) == 0)

    def op_cinc(self, o, *_):
        if o[2] != 'ne':
            raise NotImplementedError(o)
        self.set(o[0], self.get(o[1]) + (0 if self.z else 1))

    def op_mul(self, o, *_):
        self.set(o[0], self.get(o[1]) * self.get(o[2]))

    def op_umulh(self, o, *_):
        self.set(o[0], (self.get(o[1]) * self.get(o[2])) >> 64)

    def op_madd(self, o, *_):
        self.set(o[0], self.get(o[1]) * self.get(o[2]) + self.get(o[3]))

    def op_and(self, o, *_):
        self.set(o[0], self.get(o[1]) & self.imm(o[2]))

    def op_orr(self, o, *_):
        self.set(o[0], self.get(o[1]) | self._shifted(o))

    def op_lsr(self, o, *_):
        self.set(o[0], self.get(o[1]) >> self.imm(o[2]))

    def op_b_ne(self, o, labels, pc):
        return None if self.z else self._target(o[0], labels, pc)

    def op_b_eq(self, o, labels, pc):
        return self._target(o[0], labels, pc) if self.z else None

    def op_b(self, o, labels, pc):
        return self._target(o[0], labels, pc)

    def op_cbnz(self, o, labels, pc):
        return self._target(o[1], labels, pc) if self.get(o[0]) else None

    # Memory.
    def op_ldr(self, o, *_):
        addr = self.address(o[1], self.imm(o[2]) if len(o) == 3 else None)
        reg = o[0]
        if reg.startswith('q'):
            self.v[int(reg[1:])] = self.memory.read(addr, 16)
        else:
            self.set(reg, self.memory.read(addr, 8 if reg.startswith('x') else 4))

    def op_str(self, o, *_):
        addr = self.address(o[1], self.imm(o[2]) if len(o) == 3 else None)
        reg = o[0]
        if reg.startswith('q'):
            self.memory.write(addr, 16, self.v[int(reg[1:])])
        else:
            self.memory.write(addr, 8 if reg.startswith('x') else 4, self.get(reg))

    def op_ldp(self, o, *_):
        addr = self.address(o[2], self.imm(o[3]) if len(o) == 4 else None)
        size = {'q': 16, 'x': 8, 'w': 4}[o[0][0]]
        for k, reg in enumerate(o[:2]):
            value = self.memory.read(addr + size * k, size)
            if reg.startswith('q'):
                self.v[int(reg[1:])] = value
            else:
                self.set(reg, value)

    def op_stp(self, o, *_):
        addr = self.address(o[2], self.imm(o[3]) if len(o) == 4 else None)
        for k, reg in enumerate(o[:2]):
            if not (reg == 'xzr' or reg.startswith('x')):
                raise NotImplementedError(o)
            self.memory.write(addr + 8 * k, 8, self.get(reg))

    def op_ld2(self, o, *_):
        first, second = [self.vreg(t.strip()) for t in o[0].strip('{}').split(',')]
        if first[1] != '2d' or second[1] != '2d' or second[0] != first[0] + 1:
            raise ValueError(f'LD2 needs consecutive .2d registers: {o}')
        addr = self.address(o[1], self.imm(o[2]) if len(o) == 3 else None)
        words = [self.memory.read(addr + 8 * k, 8) for k in range(4)]
        self.v[first[0]] = _pack([words[0], words[2]], 64)
        self.v[second[0]] = _pack([words[1], words[3]], 64)

    # Vector arithmetic.
    def _long(self, o, accumulate, high):
        d, _ = self.vreg(o[0])
        n, n_arrangement = self.vreg(o[1])
        m, m_arrangement = self.vreg(o[2])
        if n_arrangement != m_arrangement or n_arrangement != ('4s' if high else '2s'):
            raise ValueError(f'arrangement: {o}')
        base = 2 if high else 0
        products = [_lanes(self.v[n], 32)[base + k] * _lanes(self.v[m], 32)[base + k] for k in range(2)]
        old = _lanes(self.v[d], 64) if accumulate else [0, 0]
        self.v[d] = _pack([old[k] + products[k] for k in range(2)], 64)

    def op_umlal(self, o, *_):
        self._long(o, True, False)

    def op_umlal2(self, o, *_):
        self._long(o, True, True)

    def op_umull(self, o, *_):
        self._long(o, False, False)

    def op_umull2(self, o, *_):
        self._long(o, False, True)

    def op_shl(self, o, *_):
        d, arrangement = self.vreg(o[0])
        n, _ = self.vreg(o[1])
        if arrangement != '4s':
            raise NotImplementedError(o)
        self.v[d] = _pack([x << self.imm(o[2]) for x in _lanes(self.v[n], 32)], 32)


def digits(value, count):
    out = [(value >> (28 * k)) & MASK28 for k in range(count)]
    if value >> (28 * count):
        raise ValueError('value does not fit the digits')
    return out


def replicated(p_value, q_value, count):
    """Paired value as the Rust caller stores it: each digit pair twice per 16 bytes."""
    out = bytearray()
    for dp, dq in zip(digits(p_value, count), digits(q_value, count)):
        out += (dp | dq << 32).to_bytes(8, 'little') * 2
    return bytes(out)


def reference(a, b, m, count):
    nprime = -pow(m, -1, 1 << 28) % (1 << 28)
    x = a * b
    for k in range(count):
        x += ((((x >> (28 * k)) & MASK28) * nprime) & MASK28) * m << (28 * k)
    if x & ((1 << (28 * count)) - 1):
        raise AssertionError('reference reduction left low digits')
    return x >> (28 * count)


def run(lines, count, moduli, a_values, b_values, rng, in_place=False):
    """Run one kernel; b_values is None for squaring. Returns the output values per lane."""
    memory = Memory()
    t = memory.alloc('T', bytes(rng.getrandbits(8) for _ in range(64 * count)), True)
    n = memory.alloc('n', replicated(*moduli, count), False)
    a = memory.alloc('a', replicated(*a_values, count), in_place)
    b = memory.alloc('b', replicated(*b_values, count), False) if b_values else None
    out = a if in_place else memory.alloc('out', bytes(16 * count), True)
    params = bytearray(rng.getrandbits(8) for _ in range(128))
    for lane, m in enumerate(moduli):
        nprime = -pow(m, -1, 1 << 28) % (1 << 28)
        params[64 + 8 * lane:72 + 8 * lane] = (nprime << 36).to_bytes(8, 'little')
        params[80 + 8 * lane:88 + 8 * lane] = ((m & MASK28) | ((m >> 28) & MASK28) << 28).to_bytes(8, 'little')
    params[96:104] = out.to_bytes(8, 'little')
    params[104:112] = (count // 8).to_bytes(8, 'little')
    params[112:120] = (16 * (count - 1)).to_bytes(8, 'little')
    params[120:128] = (16 * count).to_bytes(8, 'little')
    mb = memory.alloc('params', bytes(params), True)
    cpu = Cpu(memory, rng)
    cpu.x.update({'x0': t, 'x1': b if b is not None else rng.getrandbits(64), 'x2': n + 16, 'x3': a,
                  'x5': mb, 'x6': n})
    cpu.run(lines)
    buf = memory.region(out)
    values = [0, 0]
    for k in range(count):
        low = int.from_bytes(buf[16 * k:16 * k + 8], 'little')
        if low != int.from_bytes(buf[16 * k + 8:16 * k + 16], 'little'):
            raise AssertionError(f'output digit {k} is not replicated')
        for lane in range(2):
            digit = (low >> (32 * lane)) & M32
            if digit > MASK28:
                raise AssertionError(f'output digit {k} is not canonical')
            values[lane] |= digit << (28 * k)
    return values


# Digit counts of the supported prime widths (1024, 1536, 2048 bits), the smallest shape, and a
# shape between.
SHAPES = [(1024, 40), (1536, 56), (2048, 80), (640, 24), (1536, 64)]


def _modulus(rng, bits):
    return rng.getrandbits(bits) | (1 << (bits - 1)) | 1


def check_kernels(kernels, cases, rng, shapes=SHAPES):
    """Return a description of every mismatch between the kernels and the reference."""
    failures = []
    for name, generator in kernels.items():
        lines = generator()[0]
        square = name == 'square_kernel'
        for bits, count in shapes:
            for case in range(cases):
                moduli = [_modulus(rng, bits), _modulus(rng, bits)]
                if case == 0:
                    moduli = [(1 << bits) - 1, (1 << (bits - 1)) + 1]
                # Odd cases pair edge operands (0, 1, 2m - 1, m - 1, m) across both kernels' inputs
                # and lanes; even cases are random below 2m.
                edges = [lambda m: 0, lambda m: 1, lambda m: 2 * m - 1, lambda m: m - 1, lambda m: m]
                if case % 2:
                    index = case // 2
                    a_values = [edges[(index + lane) % 5](m) for lane, m in enumerate(moduli)]
                    b_values = [edges[(index // 5 + 2 * lane) % 5](m) for lane, m in enumerate(moduli)]
                else:
                    a_values = [rng.randrange(2 * m) for m in moduli]
                    b_values = [rng.randrange(2 * m) for m in moduli]
                if square:
                    b_values = list(a_values)
                in_place = case % 3 == 1
                try:
                    got = run(lines, count, moduli, a_values, None if square else b_values, rng, in_place)
                except (MemoryError, AssertionError, ValueError, NotImplementedError, RuntimeError) as error:
                    failures.append(f'{name}(D={count}, case {case}): {error}')
                    continue
                want = [reference(a, b, m, count) for a, b, m in zip(a_values, b_values, moduli)]
                if got != want:
                    failures.append(f'{name}(D={count}, case {case}): result differs')
                if any(w >= 2 * m for w, m in zip(want, moduli)):
                    failures.append(f'{name}(D={count}, case {case}): reference above 2m')
    return failures


if __name__ == '__main__':
    import rsa_pair_aarch64

    problems = check_kernels(rsa_pair_aarch64.KERNELS, 8, random.Random(1))
    print('\n'.join(problems) or 'ok')
