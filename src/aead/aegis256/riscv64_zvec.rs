use super::{BLOCK_SIZE, C0, C1, KEY_SIZE, NONCE_SIZE, TAG_SIZE};

// ── T-table AES round ────────────────────────────────────────────────────
//
// T-tables fuse SubBytes + MixColumns into a single 32-bit lookup per
// input byte. T0 is the base table; T1-T3 are byte rotations of T0.
// ShiftRows is handled by indexing into different columns of the state.

// ── AES S-box (FIPS 197) ──

/// Full AES S-box lookup table (256 entries).
#[rustfmt::skip]
const SBOX: [u8; 256] = [
  0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
  0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
  0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
  0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
  0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
  0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
  0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
  0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
  0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
  0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
  0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
  0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
  0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
  0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
  0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
  0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
];

// ── T-table generation ──

/// xtime: multiply by 2 in GF(2^8) with AES polynomial.
const fn xt(b: u8) -> u8 {
  let r = (b as u16) << 1;
  (r ^ (if r & 0x100 != 0 { 0x1B } else { 0 })) as u8
}

/// Generate T0: T0[b] = column [2*S(b), S(b), S(b), 3*S(b)] as big-endian u32.
/// T1/T2/T3 are byte rotations of T0.
const fn generate_t0() -> [u32; 256] {
  let mut t = [0u32; 256];
  let mut i = 0;
  while i < 256 {
    let s = SBOX[i];
    let s2 = xt(s);
    let s3 = s2 ^ s;
    t[i] = (s2 as u32) << 24 | (s as u32) << 16 | (s as u32) << 8 | s3 as u32;
    i += 1;
  }
  t
}

const fn generate_t1() -> [u32; 256] {
  let t0 = generate_t0();
  let mut t = [0u32; 256];
  let mut i = 0;
  while i < 256 {
    t[i] = t0[i].rotate_right(8);
    i += 1;
  }
  t
}

const fn generate_t2() -> [u32; 256] {
  let t0 = generate_t0();
  let mut t = [0u32; 256];
  let mut i = 0;
  while i < 256 {
    t[i] = t0[i].rotate_right(16);
    i += 1;
  }
  t
}

const fn generate_t3() -> [u32; 256] {
  let t0 = generate_t0();
  let mut t = [0u32; 256];
  let mut i = 0;
  while i < 256 {
    t[i] = t0[i].rotate_right(24);
    i += 1;
  }
  t
}

static T0: [u32; 256] = generate_t0();
static T1: [u32; 256] = generate_t1();
static T2: [u32; 256] = generate_t2();
static T3: [u32; 256] = generate_t3();

// ── Block helpers ──────────────────────────────────────────────────────

type Block = [u8; BLOCK_SIZE];

#[inline(always)]
fn xor_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  let mut i = 0;
  while i < BLOCK_SIZE {
    out[i] = a[i] ^ b[i];
    i = i.strict_add(1);
  }
  out
}

#[inline(always)]
fn and_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  let mut i = 0;
  while i < BLOCK_SIZE {
    out[i] = a[i] & b[i];
    i = i.strict_add(1);
  }
  out
}

// ── T-table AES round ──────────────────────────────────────────────────

/// Single AES round via T-tables: SubBytes + ShiftRows + MixColumns + AddRoundKey.
///
/// T-tables fuse SubBytes + MixColumns into a single 32-bit lookup per byte.
/// ShiftRows is handled by indexing into the correct column positions.
#[inline(always)]
fn aes_round(block: &Block, round_key: &Block) -> Block {
  // Load state as 4 big-endian column words.
  let s0 = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
  let s1 = u32::from_be_bytes([block[4], block[5], block[6], block[7]]);
  let s2 = u32::from_be_bytes([block[8], block[9], block[10], block[11]]);
  let s3 = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);

  // Load round key as 4 big-endian column words.
  let k0 = u32::from_be_bytes([round_key[0], round_key[1], round_key[2], round_key[3]]);
  let k1 = u32::from_be_bytes([round_key[4], round_key[5], round_key[6], round_key[7]]);
  let k2 = u32::from_be_bytes([round_key[8], round_key[9], round_key[10], round_key[11]]);
  let k3 = u32::from_be_bytes([round_key[12], round_key[13], round_key[14], round_key[15]]);

  // T-table lookups with ShiftRows baked into the column indexing.
  // Column j uses bytes from rows (0,j), (1,j+1), (2,j+2), (3,j+3) mod 4.
  // In big-endian column words: byte 0 = row 0 (bits 31:24), byte 3 = row 3 (bits 7:0).
  let c0 = T0[(s0 >> 24) as usize]
    ^ T1[((s1 >> 16) & 0xFF) as usize]
    ^ T2[((s2 >> 8) & 0xFF) as usize]
    ^ T3[(s3 & 0xFF) as usize]
    ^ k0;

  let c1 = T0[(s1 >> 24) as usize]
    ^ T1[((s2 >> 16) & 0xFF) as usize]
    ^ T2[((s3 >> 8) & 0xFF) as usize]
    ^ T3[(s0 & 0xFF) as usize]
    ^ k1;

  let c2 = T0[(s2 >> 24) as usize]
    ^ T1[((s3 >> 16) & 0xFF) as usize]
    ^ T2[((s0 >> 8) & 0xFF) as usize]
    ^ T3[(s1 & 0xFF) as usize]
    ^ k2;

  let c3 = T0[(s3 >> 24) as usize]
    ^ T1[((s0 >> 16) & 0xFF) as usize]
    ^ T2[((s1 >> 8) & 0xFF) as usize]
    ^ T3[(s2 & 0xFF) as usize]
    ^ k3;

  let mut out = [0u8; BLOCK_SIZE];
  out[0..4].copy_from_slice(&c0.to_be_bytes());
  out[4..8].copy_from_slice(&c1.to_be_bytes());
  out[8..12].copy_from_slice(&c2.to_be_bytes());
  out[12..16].copy_from_slice(&c3.to_be_bytes());
  out
}

// ── AEGIS-256 state operations ──────────────────────────────────────────

type State = [Block; 6];

#[inline(always)]
fn update(s: &mut State, m: &Block) {
  let tmp = s[5];
  s[5] = aes_round(&s[4], &s[5]);
  s[4] = aes_round(&s[3], &s[4]);
  s[3] = aes_round(&s[2], &s[3]);
  s[2] = aes_round(&s[1], &s[2]);
  s[1] = aes_round(&s[0], &s[1]);
  s[0] = xor_block(&aes_round(&tmp, &s[0]), m);
}

#[inline(always)]
fn keystream(s: &State) -> Block {
  xor_block(&xor_block(&s[1], &s[4]), &xor_block(&s[5], &and_block(&s[2], &s[3])))
}

// ── Fused encrypt/decrypt ───────────────────────────────────────────────

pub(super) fn encrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  let (kh0, kh1) = super::split_halves(key);
  let (nh0, nh1) = super::split_halves(nonce);
  let k0_xor_n0 = xor_block(kh0, nh0);
  let k1_xor_n1 = xor_block(kh1, nh1);
  let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

  for _ in 0..4 {
    update(&mut s, kh0);
    update(&mut s, kh1);
    update(&mut s, &k0_xor_n0);
    update(&mut s, &k1_xor_n1);
  }

  // ── aad ──
  let mut offset = 0usize;
  while offset.strict_add(BLOCK_SIZE) <= aad.len() {
    let mut tmp = [0u8; BLOCK_SIZE];
    tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
    update(&mut s, &tmp);
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < aad.len() {
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
    update(&mut s, &pad);
  }

  // ── encrypt ──
  let msg_len = buffer.len();
  let len = buffer.len();
  offset = 0;
  while offset.strict_add(BLOCK_SIZE) <= len {
    let z = keystream(&s);
    let mut xi = [0u8; BLOCK_SIZE];
    xi.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
    update(&mut s, &xi);
    buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xor_block(&xi, &z));
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < len {
    let z = keystream(&s);
    let tail_len = len.strict_sub(offset);
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    update(&mut s, &pad);
    let ct = xor_block(&pad, &z);
    buffer[offset..].copy_from_slice(&ct[..tail_len]);
  }

  // ── finalize ──
  let ad_bits = (aad.len() as u64).strict_mul(8);
  let msg_bits = (msg_len as u64).strict_mul(8);
  let mut len_bytes = [0u8; BLOCK_SIZE];
  len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
  len_bytes[8..].copy_from_slice(&msg_bits.to_le_bytes());
  let t = xor_block(&s[3], &len_bytes);
  for _ in 0..7 {
    update(&mut s, &t);
  }
  xor_block(
    &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
    &xor_block(&s[4], &s[5]),
  )
}

pub(super) fn decrypt_fused(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  let (kh0, kh1) = super::split_halves(key);
  let (nh0, nh1) = super::split_halves(nonce);
  let k0_xor_n0 = xor_block(kh0, nh0);
  let k1_xor_n1 = xor_block(kh1, nh1);
  let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

  for _ in 0..4 {
    update(&mut s, kh0);
    update(&mut s, kh1);
    update(&mut s, &k0_xor_n0);
    update(&mut s, &k1_xor_n1);
  }

  // ── aad ──
  let mut offset = 0usize;
  while offset.strict_add(BLOCK_SIZE) <= aad.len() {
    let mut tmp = [0u8; BLOCK_SIZE];
    tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
    update(&mut s, &tmp);
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < aad.len() {
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
    update(&mut s, &pad);
  }

  // ── decrypt ──
  let ct_len = buffer.len();
  let len = buffer.len();
  offset = 0;
  while offset.strict_add(BLOCK_SIZE) <= len {
    let z = keystream(&s);
    let mut ci = [0u8; BLOCK_SIZE];
    ci.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
    let xi = xor_block(&ci, &z);
    update(&mut s, &xi);
    buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xi);
    offset = offset.strict_add(BLOCK_SIZE);
  }
  if offset < len {
    let z = keystream(&s);
    let tail_len = len.strict_sub(offset);
    let mut pad = [0u8; BLOCK_SIZE];
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    let mut pt_pad = [0u8; BLOCK_SIZE];
    for i in 0..tail_len {
      pt_pad[i] = pad[i] ^ z[i];
    }
    update(&mut s, &pt_pad);
    buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
  }

  // ── finalize ──
  let ad_bits = (aad.len() as u64).strict_mul(8);
  let ct_bits = (ct_len as u64).strict_mul(8);
  let mut len_bytes = [0u8; BLOCK_SIZE];
  len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
  len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
  let t = xor_block(&s[3], &len_bytes);
  for _ in 0..7 {
    update(&mut s, &t);
  }
  xor_block(
    &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
    &xor_block(&s[4], &s[5]),
  )
}
