//! The CRC engine that every CRC width shares.
//!
//! A [`Spec`] describes one CRC in the Rocksoft model: width (8 to 64 bits), polynomial, bit order,
//! initial value, and final XOR. `const fn` code derives every table and folding constant from it.
//!
//! # One 64-bit domain
//!
//! An `n`-bit CRC over `G(x)` equals a 64-bit CRC over `G'(x) = G(x) * x^(64 - n)`, with the result in
//! the top `n` bits. The folding engines compute only 64-bit CRCs over `G'`, so one engine per
//! architecture serves every width. Only the constants change.
//!
//! - MSB-first: the 64-bit state holds the CRC register in its top `n` bits. A 16-byte block is the
//!   polynomial with the first byte at the highest degree, so the engines reverse the byte order of
//!   each block.
//! - Reflected (LSB-first): the state holds the reflected register in its low `n` bits. A
//!   little-endian load of a block is the bit reversal of its polynomial, so the engines load
//!   blocks unchanged.
//!
//! # Words
//!
//! Portable code keeps the register in the smallest word that holds it ([`Word`]): `u16`, `u32`, or
//! `u64`. A reflected register sits in the low bits of the word, an MSB-first register in the top
//! bits. Tables use this word domain.

pub(in crate::checksum) mod api;
pub(in crate::checksum) mod dispatch;
pub(in crate::checksum) mod portable;

#[cfg(target_arch = "aarch64")]
pub(in crate::checksum) mod aarch64;
#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
pub(in crate::checksum) mod power;
#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
pub(in crate::checksum) mod riscv64;
#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
pub(in crate::checksum) mod s390x;
#[cfg(all(
  any(target_arch = "powerpc64", target_arch = "s390x"),
  not(feature = "portable-only")
))]
pub(in crate::checksum) mod vector;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(in crate::checksum) mod x86_64;

/// One CRC algorithm in the Rocksoft model, with equal input and output reflection.
#[derive(Clone, Copy, Debug)]
pub(in crate::checksum) struct Spec {
  /// Width in bits, from 8 through 64.
  pub width: u32,
  /// Generator polynomial without the `x^width` term, in normal (MSB-first) form.
  pub poly: u64,
  /// `true` when input and output are reflected (LSB-first).
  pub reflected: bool,
  /// Initial register value, in the specification's form.
  pub init: u64,
  /// Final XOR value, in the specification's form.
  pub xorout: u64,
}

impl Spec {
  /// Returns the specification, after a check of its parameters.
  #[must_use]
  pub(in crate::checksum) const fn new(width: u32, poly: u64, reflected: bool, init: u64, xorout: u64) -> Self {
    assert!(width >= 8 && width <= 64, "CRC width must be 8 through 64");
    assert!(poly & 1 == 1, "a CRC polynomial has a constant term");
    assert!(poly & !mask(width) == 0 && init & !mask(width) == 0 && xorout & !mask(width) == 0);
    Self {
      width,
      poly,
      reflected,
      init,
      xorout,
    }
  }

  /// The low 64 coefficients of `G'(x) = G(x) * x^(64 - width)`.
  #[must_use]
  const fn poly64(&self) -> u64 {
    self.poly << (64u32.strict_sub(self.width))
  }

  /// Converts a register value in the specification's form to the 64-bit engine domain.
  #[must_use]
  pub(in crate::checksum) const fn domain(&self, register: u64) -> u64 {
    if self.reflected {
      register
    } else {
      register << (64u32.strict_sub(self.width))
    }
  }

  /// Converts a 64-bit engine-domain value to a register value in the specification's form.
  #[must_use]
  pub(in crate::checksum) const fn register(&self, value: u64) -> u64 {
    if self.reflected {
      value
    } else {
      value >> (64u32.strict_sub(self.width))
    }
  }
}

/// Returns a mask of the low `bits` bits.
#[must_use]
pub(in crate::checksum) const fn mask(bits: u32) -> u64 {
  if bits >= 64 { !0 } else { (1u64 << bits).strict_sub(1) }
}

// GF(2) polynomial arithmetic, normal form, modulo G'(x) = x^64 + g(x).

/// Carryless product of two polynomials of degree below 64.
#[must_use]
pub(in crate::checksum) const fn clmul(a: u64, b: u64) -> u128 {
  let mut product = 0u128;
  let mut i = 0u32;
  while i < 64 {
    if (a >> i) & 1 == 1 {
      product ^= (b as u128) << i;
    }
    i = i.strict_add(1);
  }
  product
}

/// Reduces a polynomial of degree below 128 modulo `x^64 + g`.
#[must_use]
const fn reduce(mut value: u128, g: u64) -> u64 {
  let modulus = (1u128 << 64) | g as u128;
  let mut i = 127u32;
  while i >= 64 {
    if (value >> i) & 1 == 1 {
      value ^= modulus << (i.strict_sub(64));
    }
    i = i.strict_sub(1);
  }
  low64(value)
}

/// The low 64 bits of a 128-bit value.
#[must_use]
pub(in crate::checksum) const fn low64(value: u128) -> u64 {
  let b = value.to_le_bytes();
  u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// The low 32 bits of a 64-bit value.
#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2")))]
#[must_use]
pub(in crate::checksum) const fn low32(value: u64) -> u32 {
  let b = value.to_le_bytes();
  u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The high 64 bits of a 128-bit value.
#[must_use]
pub(in crate::checksum) const fn high64(value: u128) -> u64 {
  low64(value >> 64)
}

/// `a * b mod (x^64 + g)`.
#[must_use]
pub(in crate::checksum) const fn mulmod(a: u64, b: u64, g: u64) -> u64 {
  reduce(clmul(a, b), g)
}

/// `x^e mod (x^64 + g)`, by square and multiply.
#[must_use]
const fn xpow(e: u64, g: u64) -> u64 {
  let mut result = 1u64;
  let mut base = 2u64;
  let mut e = e;
  while e > 0 {
    if e & 1 == 1 {
      result = mulmod(result, base, g);
    }
    base = mulmod(base, base, g);
    e >>= 1;
  }
  result
}

#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(
    any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
    not(feature = "portable-only")
  )
))]
/// The low 64 coefficients of `floor(x^128 / (x^64 + g))`.
#[must_use]
const fn barrett_quotient(g: u64) -> u64 {
  // x^128 minus x^64 * G' leaves g * x^64; continue the long division from degree 127.
  let mut rem = (g as u128) << 64;
  let mut quotient = 0u64;
  let modulus = (1u128 << 64) | g as u128;
  let mut i = 127u32;
  while i >= 64 {
    if (rem >> i) & 1 == 1 {
      quotient |= 1u64 << (i.strict_sub(64));
      rem ^= modulus << (i.strict_sub(64));
    }
    i = i.strict_sub(1);
  }
  quotient
}

#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(
    any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
    not(feature = "portable-only")
  )
))]
/// The inverse of an odd polynomial modulo `z^64`, as a power series.
#[must_use]
const fn inverse_mod_z64(p: u64) -> u64 {
  let mut inverse = 1u64;
  let mut k = 1u32;
  while k < 64 {
    // Coefficient k of p * inverse must be zero.
    let mut sum = 0u64;
    let mut i = 1u32;
    while i <= k {
      sum ^= ((p >> i) & 1) & ((inverse >> (k.strict_sub(i))) & 1);
      i = i.strict_add(1);
    }
    inverse |= sum << k;
    k = k.strict_add(1);
  }
  inverse
}

#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(
    any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
    not(feature = "portable-only")
  )
))]
/// Constants for the carryless-multiply folding engines.
///
/// A 16-byte lane is a 128-bit value: `lo` is its low 64 bits, `hi` its high 64 bits. A fold by `d`
/// bits computes `clmul(lane.lo, key[0]) ^ clmul(lane.hi, key[1])`, which every engine uses for
/// both bit orders.
#[derive(Clone, Copy, Debug)]
pub(in crate::checksum) struct Keys {
  /// `fold[m - 1]` folds a lane by `16 * m` bytes.
  pub fold: [[u64; 2]; 32],
  /// `tail[r]` folds a lane by `r` bytes; `tail[8]` also starts the final reduction.
  pub tail: [[u64; 2]; 16],
  /// Barrett constant: the quotient (MSB-first) or the inverse of `poly` (reflected).
  pub mu: u64,
  /// Barrett polynomial: `g` (MSB-first) or the reversed `G'` without its top bit (reflected).
  pub poly: u64,
  /// All ones when a reflected reduction needs the `z^64` term of the reversed `G'` (width 64).
  pub top: u64,
}

#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(
    any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
    not(feature = "portable-only")
  )
))]
impl Keys {
  /// Derives the keys of `spec`.
  #[must_use]
  pub(in crate::checksum) const fn new(spec: &Spec) -> Self {
    let g = spec.poly64();
    let mut fold = [[0u64; 2]; 32];
    let mut m = 0usize;
    while m < 32 {
      fold[m] = fold_key(spec, ((m as u64).strict_add(1)).strict_mul(128));
      m = m.strict_add(1);
    }
    let mut tail = [[0u64; 2]; 16];
    let mut r = 1usize;
    while r < 16 {
      tail[r] = fold_key(spec, (r as u64).strict_mul(8));
      r = r.strict_add(1);
    }
    if spec.reflected {
      let poly = (g.reverse_bits() << 1) | 1;
      Self {
        fold,
        tail,
        mu: inverse_mod_z64(poly),
        poly,
        top: if spec.width == 64 { !0 } else { 0 },
      }
    } else {
      Self {
        fold,
        tail,
        mu: barrett_quotient(g),
        poly: g,
        top: 0,
      }
    }
  }
}

#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(
    any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
    not(feature = "portable-only")
  )
))]
/// The key pair that folds a lane by `d` bits (`d >= 8`).
///
/// MSB-first: `lane = hi * x^64 + lo`, so `lane * x^d = hi * x^(d+64) + lo * x^d`. Reflected: the
/// low half holds the high-degree coefficients, and a carryless product of reversed operands is
/// the reversed product times `x`, so the exponents drop by one.
#[must_use]
const fn fold_key(spec: &Spec, d: u64) -> [u64; 2] {
  let g = spec.poly64();
  if spec.reflected {
    [
      xpow(d.strict_add(63), g).reverse_bits(),
      xpow(d.strict_sub(1), g).reverse_bits(),
    ]
  } else {
    [xpow(d, g), xpow(d.strict_add(64), g)]
  }
}

/// Multipliers for `combine`: entry `k` multiplies an engine-domain value by `x^(8 * 2^k) mod G`.
///
/// An entry is unaligned (degree below `width`), so a product with an aligned value stays aligned.
/// MSB-first, the product is exact. Reflected, a carryless product of bit-reversed operands carries
/// an extra factor `x`, so the reflected entry holds `x^(8 * 2^k - 1) mod G`, bit-reversed.
#[must_use]
pub(in crate::checksum) const fn combine_powers(spec: &Spec) -> [u64; 64] {
  let g = spec.poly64();
  let shift = 64u32.strict_sub(spec.width);
  // `x^-1 mod G`: G has a constant term, so `x * (G + 1) / x = G + 1 = 1 (mod G)`.
  let inverse_x = (1u64 << (spec.width.strict_sub(1))) | (spec.poly >> 1);
  let mut powers = [0u64; 64];
  // Aligned `x^8 mod G`.
  let mut power = xpow(8u64.strict_add(shift as u64), g);
  let mut k = 0usize;
  while k < 64 {
    powers[k] = if spec.reflected {
      (mulmod(power, inverse_x, g) >> shift).reverse_bits()
    } else {
      power >> shift
    };
    power = mulmod(power, power >> shift, g);
    k = k.strict_add(1);
  }
  powers
}

/// Software carryless product, four bits of `a` per step.
#[must_use]
fn clmul_soft(a: u64, b: u64) -> u128 {
  let mut multiples = [0u128; 16];
  let mut i = 1usize;
  while i < 16 {
    multiples[i] = if i & 1 == 0 {
      multiples[i >> 1] << 1
    } else {
      multiples[i.strict_sub(1)] ^ u128::from(b)
    };
    i = i.strict_add(1);
  }
  let mut product = 0u128;
  let mut shift = 64u32;
  while shift > 0 {
    shift = shift.strict_sub(4);
    product = (product << 4) ^ multiples[usize::from((a >> shift).to_le_bytes()[0] & 0x0F)];
  }
  product
}

#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "x86_64", target_feature = "sse2"),
  all(
    any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
    not(feature = "portable-only")
  )
))]
/// `MASK[r..r + 16]` keeps the last `r` bytes of a 16-byte block, for the engines' partial blocks.
pub(in crate::checksum) const MASK: [u8; 32] = {
  let mut mask = [0u8; 32];
  let mut i = 16usize;
  while i < 32 {
    mask[i] = 0xFF;
    i = i.strict_add(1);
  }
  mask
};

/// Tables per CRC for slice-by-16.
pub(in crate::checksum) const SLICES: usize = 16;

/// Everything one CRC needs at run time, derived from its [`Spec`] at compile time.
///
/// `W` is the portable register word.
pub(in crate::checksum) struct Crc<W: 'static> {
  pub spec: Spec,
  pub table: [[W; 256]; SLICES],
  #[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(
      any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
      not(feature = "portable-only")
    )
  ))]
  pub keys: Keys,
  pub powers: [u64; 64],
}

/// Which CRC-32 instruction computes a CRC, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
  not(feature = "crc32"),
  expect(dead_code, reason = "only the CRC-32 types name an instruction")
)]
pub(in crate::checksum) enum Hw {
  /// No instruction.
  None,
  /// CRC-32/ISO-HDLC: AArch64 `CRC32X`.
  Crc32,
  /// CRC-32C: x86-64 SSE4.2 `CRC32`, AArch64 `CRC32CX`.
  Crc32c,
}

/// One shipped CRC, with its static data.
pub(in crate::checksum) trait Algo: 'static {
  /// Portable register word.
  type W: Word;
  /// Static data.
  const CRC: &'static Crc<Self::W>;
  /// CRC-32 instruction support.
  const HW: Hw;
}

/// Builds the static data of one CRC.
///
/// `$word` is `u16`, `u32`, or `u64`.
macro_rules! crc_data {
  ($word:ty, $spec:expr) => {{
    const CRC_SPEC: $crate::checksum::crc::Spec = $spec;
    const WIDE: [[u64; 256]; $crate::checksum::crc::SLICES] =
      $crate::checksum::crc::wide_table::<{ $crate::checksum::crc::SLICES }>(&CRC_SPEC, <$word>::BITS);
    $crate::checksum::crc::Crc::<$word> {
      spec: CRC_SPEC,
      table: {
        let mut table: [[$word; 256]; $crate::checksum::crc::SLICES] = [[0; 256]; $crate::checksum::crc::SLICES];
        let mut k = 0usize;
        while k < $crate::checksum::crc::SLICES {
          let mut i = 0usize;
          while i < 256 {
            table[k][i] = match WIDE[k][i]
              .to_le_bytes()
              .first_chunk::<{ core::mem::size_of::<$word>() }>()
            {
              Some(low) => <$word>::from_le_bytes(*low),
              None => 0,
            };
            i = i.strict_add(1);
          }
          k = k.strict_add(1);
        }
        table
      },
      #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(
          any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x"),
          not(feature = "portable-only")
        )
      ))]
      keys: $crate::checksum::crc::Keys::new(&CRC_SPEC),
      powers: $crate::checksum::crc::combine_powers(&CRC_SPEC),
    }
  }};
}
pub(in crate::checksum) use crc_data;

/// Slice-by-N tables in a `bits`-wide word domain, as 64-bit entries.
///
/// `table[k][b]` is the register after byte `b` and then `k` zero bytes, from a zero register.
#[must_use]
pub(in crate::checksum) const fn wide_table<const N: usize>(spec: &Spec, bits: u32) -> [[u64; 256]; N] {
  let word_mask = mask(bits);
  let shift = bits.strict_sub(spec.width);
  let mut table = [[0u64; 256]; N];
  let mut b = 0usize;
  while b < 256 {
    let mut reg: u64;
    if spec.reflected {
      let poly = spec.poly.reverse_bits() >> (64u32.strict_sub(spec.width));
      reg = b as u64;
      let mut i = 0u32;
      while i < 8 {
        reg = if reg & 1 == 1 { (reg >> 1) ^ poly } else { reg >> 1 };
        i = i.strict_add(1);
      }
    } else {
      let poly = spec.poly << shift;
      let top = 1u64 << (bits.strict_sub(1));
      reg = (b as u64) << (bits.strict_sub(8));
      let mut i = 0u32;
      while i < 8 {
        reg = if reg & top != 0 {
          ((reg << 1) ^ poly) & word_mask
        } else {
          (reg << 1) & word_mask
        };
        i = i.strict_add(1);
      }
    }
    table[0][b] = reg;
    b = b.strict_add(1);
  }
  let mut k = 1usize;
  while k < N {
    let mut b = 0usize;
    while b < 256 {
      let prev = table[k.strict_sub(1)][b];
      table[k][b] = if spec.reflected {
        table[0][(prev & 0xFF) as usize] ^ (prev >> 8)
      } else {
        table[0][((prev >> (bits.strict_sub(8))) & 0xFF) as usize] ^ ((prev << 8) & word_mask)
      };
      b = b.strict_add(1);
    }
    k = k.strict_add(1);
  }
  table
}

/// A portable CRC register word.
pub(in crate::checksum) trait Word:
  Copy + Eq + core::fmt::Debug + core::ops::BitXor<Output = Self> + 'static
{
  /// Width in bits.
  const BITS: u32;
  /// Width in bytes.
  const BYTES: usize;
  /// Zero-extends the word.
  fn widen(self) -> u64;
  /// Keeps the low `BITS` bits.
  fn narrow(value: u64) -> Self;
  /// The word shifted down by one byte.
  fn shift_down(self) -> Self;
  /// The word shifted up by one byte, dropping its top byte.
  fn shift_up(self) -> Self;
}

macro_rules! impl_word {
  ($($word:ty),*) => {$(
    impl Word for $word {
      const BITS: u32 = <$word>::BITS;
      const BYTES: usize = core::mem::size_of::<$word>();

      #[inline(always)]
      fn widen(self) -> u64 {
        u64::from(self)
      }

      #[inline(always)]
      fn narrow(value: u64) -> Self {
        match value.to_le_bytes().first_chunk::<{ core::mem::size_of::<$word>() }>() {
          Some(low) => <$word>::from_le_bytes(*low),
          None => 0,
        }
      }

      #[inline(always)]
      fn shift_down(self) -> Self {
        self >> 8
      }

      #[inline(always)]
      fn shift_up(self) -> Self {
        self << 8
      }
    }
  )*};
}
impl_word!(u16, u32, u64);

impl<W: Word> Crc<W> {
  /// Converts a word-domain register to the 64-bit engine domain.
  #[inline(always)]
  pub(in crate::checksum) fn word_to_domain(&self, word: W) -> u64 {
    if self.spec.reflected {
      word.widen()
    } else {
      word.widen() << (64u32.strict_sub(W::BITS))
    }
  }

  /// Converts a 64-bit engine-domain value to the word domain.
  #[inline(always)]
  pub(in crate::checksum) fn domain_to_word(&self, value: u64) -> W {
    if self.spec.reflected {
      W::narrow(value)
    } else {
      W::narrow(value >> (64u32.strict_sub(W::BITS)))
    }
  }

  /// The word-domain register for a register value in the specification's form.
  #[inline(always)]
  pub(in crate::checksum) fn register_to_word(&self, register: u64) -> W {
    self.domain_to_word(self.spec.domain(register & mask(self.spec.width)))
  }

  /// The register value in the specification's form for a word-domain register.
  #[inline(always)]
  pub(in crate::checksum) fn word_to_register(&self, word: W) -> u64 {
    self.spec.register(self.word_to_domain(word))
  }

  /// `crc(A || B)` from `crc(A)`, `crc(B)`, and the length of `B`, as finalized values.
  ///
  /// `crc(A || B) = (crc(A) ^ xorout ^ init) * x^(8 * len(B)) ^ crc(B)`, modulo `G`. `shift`
  /// multiplies an engine-domain value by `x^(8 * len(B))`.
  #[must_use]
  pub(in crate::checksum) fn combine(
    &self,
    crc_a: u64,
    crc_b: u64,
    len_b: usize,
    shift: impl Fn(u64, usize) -> u64,
  ) -> u64 {
    let spec = &self.spec;
    let width_mask = mask(spec.width);
    if len_b == 0 {
      return crc_a & width_mask;
    }
    let value = spec.domain((crc_a ^ spec.xorout ^ spec.init) & width_mask);
    spec.register(shift(value, len_b)) ^ (crc_b & width_mask)
  }

  /// One portable multiplication by a power entry: a software carryless product, then the
  /// slice-by-8 tables reduce its high-degree half.
  #[must_use]
  pub(in crate::checksum) fn multiply_portable(&self, value: u64, power: u64) -> u64 {
    let product = clmul_soft(value, power);
    let (top, rest) = if self.spec.reflected {
      (low64(product).to_le_bytes(), high64(product))
    } else {
      (high64(product).to_be_bytes(), low64(product))
    };
    self.word_to_domain(portable::reduce8(self, &top)) ^ rest
  }
}

/// `value * x^(8 * len)` in the engine domain, with `multiply` applying one power entry.
#[inline(always)]
pub(in crate::checksum) fn shift_by(
  mut value: u64,
  len: usize,
  powers: &[u64; 64],
  multiply: impl Fn(u64, u64) -> u64,
) -> u64 {
  let mut rest = len as u64;
  let mut k = 0usize;
  while rest != 0 {
    if rest & 1 == 1 {
      value = multiply(value, powers[k]);
    }
    rest >>= 1;
    k = k.strict_add(1);
  }
  value
}

/// Bitwise reference CRC: the register after `data`, from `register`, both in the specification's
/// form.
///
/// This follows the definition one bit at a time and shares no table or constant with the fast
/// paths, so it is their test oracle.
#[must_use]
pub(in crate::checksum) const fn reference(spec: &Spec, register: u64, data: &[u8]) -> u64 {
  let width_mask = mask(spec.width);
  let mut reg = register & width_mask;
  let mut i = 0usize;
  if spec.reflected {
    let poly = spec.poly.reverse_bits() >> (64u32.strict_sub(spec.width));
    while i < data.len() {
      reg ^= data[i] as u64;
      let mut bit = 0u32;
      while bit < 8 {
        reg = if reg & 1 == 1 { (reg >> 1) ^ poly } else { reg >> 1 };
        bit = bit.strict_add(1);
      }
      i = i.strict_add(1);
    }
  } else {
    let top = 1u64 << (spec.width.strict_sub(1));
    while i < data.len() {
      reg ^= (data[i] as u64) << (spec.width.strict_sub(8));
      let mut bit = 0u32;
      while bit < 8 {
        reg = if reg & top != 0 {
          ((reg << 1) ^ spec.poly) & width_mask
        } else {
          (reg << 1) & width_mask
        };
        bit = bit.strict_add(1);
      }
      i = i.strict_add(1);
    }
  }
  reg
}

#[cfg(test)]
mod tests;
