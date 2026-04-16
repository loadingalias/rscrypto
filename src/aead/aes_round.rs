//! Shared portable AES round helpers for AEGIS and Hamburg vperm constants.

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
const BLOCK_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// Hamburg vperm lookup tables (tower-field GF(2^4) AES S-box decomposition)
// ---------------------------------------------------------------------------
//
// These tables implement the Hamburg technique (CHES 2009) for computing the
// AES S-box using only 4-bit nibble lookups via byte-shuffle instructions
// (PSHUFB, VPERM, vrgather). All operations are register-to-register with
// no secret-dependent memory access — constant-time by construction.
//
// Reference: Hamburg, "Accelerating AES with Vector Permute Instructions"
// https://www.shiftleft.org/papers/vector_aes/vector_aes.pdf
//
// Table sources: OpenSSL Lk_ipt, Lk_inv, Lk_sbo (vpaes-x86_64.pl).
// Validated for all 256 input values in aegis256.rs tests.

/// Input transform, low nibble: maps AES GF(2^8) basis to tower-field basis.
#[rustfmt::skip]
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const VPERM_IPT_LO: [u8; 16] = [
  0x00, 0x70, 0x2A, 0x5A, 0x98, 0xE8, 0xB2, 0xC2,
  0x08, 0x78, 0x22, 0x52, 0x90, 0xE0, 0xBA, 0xCA,
];

/// Input transform, high nibble: maps AES GF(2^8) basis to tower-field basis.
#[rustfmt::skip]
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const VPERM_IPT_HI: [u8; 16] = [
  0x00, 0x4D, 0x7C, 0x31, 0x7D, 0x30, 0x01, 0x4C,
  0x81, 0xCC, 0xFD, 0xB0, 0xFC, 0xB1, 0x80, 0xCD,
];

/// GF(2^4) inverse table, low nibble. Entry 0 is `0x80` (infinity sentinel).
#[rustfmt::skip]
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const VPERM_INV_LO: [u8; 16] = [
  0x80, 0x01, 0x08, 0x0D, 0x0F, 0x06, 0x05, 0x0E,
  0x02, 0x0C, 0x0B, 0x0A, 0x09, 0x03, 0x07, 0x04,
];

/// GF(2^4) inverse table, high nibble. Entry 0 is `0x80` (infinity sentinel).
#[rustfmt::skip]
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const VPERM_INV_HI: [u8; 16] = [
  0x80, 0x07, 0x0B, 0x0F, 0x06, 0x0A, 0x04, 0x01,
  0x09, 0x08, 0x05, 0x02, 0x0C, 0x0E, 0x0D, 0x03,
];

/// SubBytes output table, upper component (inverse-isomorphism + AES affine,
/// no MixColumns). `sbou[io] ^ sbot[jo] = AES_SBOX[x] ^ 0x63`.
#[rustfmt::skip]
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const VPERM_SBOU: [u8; 16] = [
  0x00, 0xC7, 0xBD, 0x6F, 0x17, 0x6D, 0xD2, 0xD0,
  0x78, 0xA8, 0x02, 0xC5, 0x7A, 0xBF, 0xAA, 0x15,
];

/// SubBytes output table, lower component (inverse-isomorphism + AES affine,
/// no MixColumns). `sbou[io] ^ sbot[jo] = AES_SBOX[x] ^ 0x63`.
#[rustfmt::skip]
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const VPERM_SBOT: [u8; 16] = [
  0x00, 0x6A, 0xBB, 0x5F, 0xA5, 0x74, 0xE4, 0xCF,
  0xFA, 0x35, 0x2B, 0x41, 0xD1, 0x90, 0x1E, 0x8E,
];

/// AES ShiftRows permutation: output byte `i` reads from input byte `SR[i]`.
#[rustfmt::skip]
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const VPERM_SR: [u8; 16] = [
  0x00, 0x05, 0x0A, 0x0F, 0x04, 0x09, 0x0E, 0x03,
  0x08, 0x0D, 0x02, 0x07, 0x0C, 0x01, 0x06, 0x0B,
];

/// AES affine constant: Hamburg vperm S-box omits this; XOR after SubBytes.
#[cfg(any(test, target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const AES_AFFINE: u8 = 0x63;

/// AES MixColumns reduction constant: `xtime(b) = (b<<1) ^ ((b>>7) * 0x1B)`.
#[cfg(any(target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const XTIME_REDUCE: u8 = 0x1B;

/// Nibble mask: extract low 4 bits of each byte.
#[cfg(target_arch = "s390x")]
pub(crate) const NIBBLE_MASK: u8 = 0x0F;

/// MixColumns column-rotate-by-1 permutation: rotates each 4-byte column by 1.
/// `[b1,b2,b3,b0, b5,b6,b7,b4, b9,b10,b11,b8, b13,b14,b15,b12]`
#[rustfmt::skip]
#[cfg(any(target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const MC_ROT1: [u8; 16] = [
  0x01, 0x02, 0x03, 0x00, 0x05, 0x06, 0x07, 0x04,
  0x09, 0x0A, 0x0B, 0x08, 0x0D, 0x0E, 0x0F, 0x0C,
];

/// MixColumns column-rotate-by-2 permutation: rotates each 4-byte column by 2.
/// `[b2,b3,b0,b1, b6,b7,b4,b5, b10,b11,b8,b9, b14,b15,b12,b13]`
#[rustfmt::skip]
#[cfg(any(target_arch = "s390x", target_arch = "riscv64"))]
pub(crate) const MC_ROT2: [u8; 16] = [
  0x02, 0x03, 0x00, 0x01, 0x06, 0x07, 0x04, 0x05,
  0x0A, 0x0B, 0x08, 0x09, 0x0E, 0x0F, 0x0C, 0x0D,
];

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn gf256_mul(a: u8, b: u8) -> u8 {
  let a = a as u16;
  let b = b as u16;

  let mut prod: u16 = 0;
  prod ^= a.wrapping_mul(b & 1);
  prod ^= (a << 1).wrapping_mul((b >> 1) & 1);
  prod ^= (a << 2).wrapping_mul((b >> 2) & 1);
  prod ^= (a << 3).wrapping_mul((b >> 3) & 1);
  prod ^= (a << 4).wrapping_mul((b >> 4) & 1);
  prod ^= (a << 5).wrapping_mul((b >> 5) & 1);
  prod ^= (a << 6).wrapping_mul((b >> 6) & 1);
  prod ^= (a << 7).wrapping_mul((b >> 7) & 1);

  prod ^= (prod >> 14).wrapping_mul(0x11b << 6);
  prod ^= (prod >> 13).wrapping_mul(0x11b << 5);
  prod ^= (prod >> 12).wrapping_mul(0x11b << 4);
  prod ^= (prod >> 11).wrapping_mul(0x11b << 3);
  prod ^= (prod >> 10).wrapping_mul(0x11b << 2);
  prod ^= (prod >> 9).wrapping_mul(0x11b << 1);
  prod ^= (prod >> 8).wrapping_mul(0x11b);

  prod as u8
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn gf256_sq(x: u8) -> u8 {
  gf256_mul(x, x)
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn gf256_inv(x: u8) -> u8 {
  let x2 = gf256_sq(x);
  let x3 = gf256_mul(x2, x);
  let x6 = gf256_sq(x3);
  let x12 = gf256_sq(x6);
  let x14 = gf256_mul(x12, x2);
  let x15 = gf256_mul(x14, x);
  let x30 = gf256_sq(x15);
  let x60 = gf256_sq(x30);
  let x62 = gf256_mul(x60, x2);
  let x63 = gf256_mul(x62, x);
  let x126 = gf256_sq(x63);
  let x252 = gf256_sq(x126);
  gf256_mul(x252, x2)
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn sbox(x: u8) -> u8 {
  let inv = gf256_inv(x);
  let r = inv ^ inv.rotate_left(1) ^ inv.rotate_left(2) ^ inv.rotate_left(3) ^ inv.rotate_left(4);
  r ^ 0x63
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn col_byte(col: u32, row: usize) -> u8 {
  (col >> (24u32.strict_sub((row as u32).strict_mul(8)))) as u8
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn xtime(x: u8) -> u8 {
  let hi = (x >> 7) & 1;
  (x << 1) ^ (hi.wrapping_mul(0x1b))
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn mix_column(col: [u8; 4]) -> u32 {
  let [b0, b1, b2, b3] = col;

  let r0 = xtime(b0) ^ xtime(b1) ^ b1 ^ b2 ^ b3;
  let r1 = b0 ^ xtime(b1) ^ xtime(b2) ^ b2 ^ b3;
  let r2 = b0 ^ b1 ^ xtime(b2) ^ xtime(b3) ^ b3;
  let r3 = xtime(b0) ^ b0 ^ b1 ^ b2 ^ xtime(b3);

  (r0 as u32) << 24 | (r1 as u32) << 16 | (r2 as u32) << 8 | r3 as u32
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline(always)]
const fn aes_round(s0: u32, s1: u32, s2: u32, s3: u32) -> (u32, u32, u32, u32) {
  let sr0 = [
    sbox(col_byte(s0, 0)),
    sbox(col_byte(s1, 1)),
    sbox(col_byte(s2, 2)),
    sbox(col_byte(s3, 3)),
  ];
  let sr1 = [
    sbox(col_byte(s1, 0)),
    sbox(col_byte(s2, 1)),
    sbox(col_byte(s3, 2)),
    sbox(col_byte(s0, 3)),
  ];
  let sr2 = [
    sbox(col_byte(s2, 0)),
    sbox(col_byte(s3, 1)),
    sbox(col_byte(s0, 2)),
    sbox(col_byte(s1, 3)),
  ];
  let sr3 = [
    sbox(col_byte(s3, 0)),
    sbox(col_byte(s0, 1)),
    sbox(col_byte(s1, 2)),
    sbox(col_byte(s2, 3)),
  ];

  (mix_column(sr0), mix_column(sr1), mix_column(sr2), mix_column(sr3))
}

#[cfg(any(test, not(any(target_arch = "s390x", target_arch = "riscv64"))))]
#[inline]
pub(crate) fn aes_enc_round_portable(block: &[u8; BLOCK_SIZE], round_key: &[u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
  let s0 = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
  let s1 = u32::from_be_bytes([block[4], block[5], block[6], block[7]]);
  let s2 = u32::from_be_bytes([block[8], block[9], block[10], block[11]]);
  let s3 = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);

  let (r0, r1, r2, r3) = aes_round(s0, s1, s2, s3);

  let k0 = u32::from_be_bytes([round_key[0], round_key[1], round_key[2], round_key[3]]);
  let k1 = u32::from_be_bytes([round_key[4], round_key[5], round_key[6], round_key[7]]);
  let k2 = u32::from_be_bytes([round_key[8], round_key[9], round_key[10], round_key[11]]);
  let k3 = u32::from_be_bytes([round_key[12], round_key[13], round_key[14], round_key[15]]);

  let mut out = [0u8; BLOCK_SIZE];
  out[0..4].copy_from_slice(&(r0 ^ k0).to_be_bytes());
  out[4..8].copy_from_slice(&(r1 ^ k1).to_be_bytes());
  out[8..12].copy_from_slice(&(r2 ^ k2).to_be_bytes());
  out[12..16].copy_from_slice(&(r3 ^ k3).to_be_bytes());
  out
}
