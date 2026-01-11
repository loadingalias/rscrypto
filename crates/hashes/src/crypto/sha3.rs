//! SHA-3 (FIPS 202) - SHA3-256 and SHA3-512.
//!
//! Portable, `no_std`, pure Rust Keccak-f[1600] sponge.

#![allow(clippy::indexing_slicing)] // Keccak state is fixed-size; indexing is audited

use traits::Digest;

const KECCAKF_ROUNDS: usize = 24;

// Round constants.
const RC: [u64; KECCAKF_ROUNDS] = [
  0x0000_0000_0000_0001,
  0x0000_0000_0000_8082,
  0x8000_0000_0000_808a,
  0x8000_0000_8000_8000,
  0x0000_0000_0000_808b,
  0x0000_0000_8000_0001,
  0x8000_0000_8000_8081,
  0x8000_0000_0000_8009,
  0x0000_0000_0000_008a,
  0x0000_0000_0000_0088,
  0x0000_0000_8000_8009,
  0x0000_0000_8000_000a,
  0x0000_0000_8000_808b,
  0x8000_0000_0000_008b,
  0x8000_0000_0000_8089,
  0x8000_0000_0000_8003,
  0x8000_0000_0000_8002,
  0x8000_0000_0000_0080,
  0x0000_0000_0000_800a,
  0x8000_0000_8000_000a,
  0x8000_0000_8000_8081,
  0x8000_0000_0000_8080,
  0x0000_0000_8000_0001,
  0x8000_0000_8000_8008,
];

#[inline(always)]
fn keccakf(state: &mut [u64; 25]) {
  let mut a0 = state[0];
  let mut a1 = state[1];
  let mut a2 = state[2];
  let mut a3 = state[3];
  let mut a4 = state[4];
  let mut a5 = state[5];
  let mut a6 = state[6];
  let mut a7 = state[7];
  let mut a8 = state[8];
  let mut a9 = state[9];
  let mut a10 = state[10];
  let mut a11 = state[11];
  let mut a12 = state[12];
  let mut a13 = state[13];
  let mut a14 = state[14];
  let mut a15 = state[15];
  let mut a16 = state[16];
  let mut a17 = state[17];
  let mut a18 = state[18];
  let mut a19 = state[19];
  let mut a20 = state[20];
  let mut a21 = state[21];
  let mut a22 = state[22];
  let mut a23 = state[23];
  let mut a24 = state[24];

  for &rc in &RC {
    // θ
    let c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;
    let c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;
    let c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;
    let c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;
    let c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;

    let d0 = c4 ^ c1.rotate_left(1);
    let d1 = c0 ^ c2.rotate_left(1);
    let d2 = c1 ^ c3.rotate_left(1);
    let d3 = c2 ^ c4.rotate_left(1);
    let d4 = c3 ^ c0.rotate_left(1);

    a0 ^= d0;
    a5 ^= d0;
    a10 ^= d0;
    a15 ^= d0;
    a20 ^= d0;

    a1 ^= d1;
    a6 ^= d1;
    a11 ^= d1;
    a16 ^= d1;
    a21 ^= d1;

    a2 ^= d2;
    a7 ^= d2;
    a12 ^= d2;
    a17 ^= d2;
    a22 ^= d2;

    a3 ^= d3;
    a8 ^= d3;
    a13 ^= d3;
    a18 ^= d3;
    a23 ^= d3;

    a4 ^= d4;
    a9 ^= d4;
    a14 ^= d4;
    a19 ^= d4;
    a24 ^= d4;

    // ρ + π
    let b0 = a0;
    let b10 = a1.rotate_left(1);
    let b20 = a2.rotate_left(62);
    let b5 = a3.rotate_left(28);
    let b15 = a4.rotate_left(27);

    let b16 = a5.rotate_left(36);
    let b1 = a6.rotate_left(44);
    let b11 = a7.rotate_left(6);
    let b21 = a8.rotate_left(55);
    let b6 = a9.rotate_left(20);

    let b7 = a10.rotate_left(3);
    let b17 = a11.rotate_left(10);
    let b2 = a12.rotate_left(43);
    let b12 = a13.rotate_left(25);
    let b22 = a14.rotate_left(39);

    let b23 = a15.rotate_left(41);
    let b8 = a16.rotate_left(45);
    let b18 = a17.rotate_left(15);
    let b3 = a18.rotate_left(21);
    let b13 = a19.rotate_left(8);

    let b14 = a20.rotate_left(18);
    let b24 = a21.rotate_left(2);
    let b9 = a22.rotate_left(61);
    let b19 = a23.rotate_left(56);
    let b4 = a24.rotate_left(14);

    // χ
    a0 = b0 ^ ((!b1) & b2);
    a1 = b1 ^ ((!b2) & b3);
    a2 = b2 ^ ((!b3) & b4);
    a3 = b3 ^ ((!b4) & b0);
    a4 = b4 ^ ((!b0) & b1);

    a5 = b5 ^ ((!b6) & b7);
    a6 = b6 ^ ((!b7) & b8);
    a7 = b7 ^ ((!b8) & b9);
    a8 = b8 ^ ((!b9) & b5);
    a9 = b9 ^ ((!b5) & b6);

    a10 = b10 ^ ((!b11) & b12);
    a11 = b11 ^ ((!b12) & b13);
    a12 = b12 ^ ((!b13) & b14);
    a13 = b13 ^ ((!b14) & b10);
    a14 = b14 ^ ((!b10) & b11);

    a15 = b15 ^ ((!b16) & b17);
    a16 = b16 ^ ((!b17) & b18);
    a17 = b17 ^ ((!b18) & b19);
    a18 = b18 ^ ((!b19) & b15);
    a19 = b19 ^ ((!b15) & b16);

    a20 = b20 ^ ((!b21) & b22);
    a21 = b21 ^ ((!b22) & b23);
    a22 = b22 ^ ((!b23) & b24);
    a23 = b23 ^ ((!b24) & b20);
    a24 = b24 ^ ((!b20) & b21);

    // ι
    a0 ^= rc;
  }

  state[0] = a0;
  state[1] = a1;
  state[2] = a2;
  state[3] = a3;
  state[4] = a4;
  state[5] = a5;
  state[6] = a6;
  state[7] = a7;
  state[8] = a8;
  state[9] = a9;
  state[10] = a10;
  state[11] = a11;
  state[12] = a12;
  state[13] = a13;
  state[14] = a14;
  state[15] = a15;
  state[16] = a16;
  state[17] = a17;
  state[18] = a18;
  state[19] = a19;
  state[20] = a20;
  state[21] = a21;
  state[22] = a22;
  state[23] = a23;
  state[24] = a24;
}

#[derive(Clone)]
struct Sha3Core<const RATE: usize> {
  state: [u64; 25],
  buf: [u8; RATE],
  buf_len: usize,
}

impl<const RATE: usize> Default for Sha3Core<RATE> {
  #[inline]
  fn default() -> Self {
    Self {
      state: [0u64; 25],
      buf: [0u8; RATE],
      buf_len: 0,
    }
  }
}

impl<const RATE: usize> Sha3Core<RATE> {
  #[inline(always)]
  fn absorb_block(state: &mut [u64; 25], block: &[u8; RATE]) {
    debug_assert_eq!(RATE % 8, 0);
    let lanes = RATE / 8;
    let (chunks, _) = block.as_chunks::<8>();
    for (lane, chunk) in state[..lanes].iter_mut().zip(chunks.iter()) {
      *lane ^= u64::from_le_bytes(*chunk);
    }
    keccakf(state);
  }

  fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

    if self.buf_len != 0 {
      let take = core::cmp::min(RATE - self.buf_len, data.len());
      self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
      self.buf_len += take;
      data = &data[take..];

      if self.buf_len == RATE {
        let state = &mut self.state;
        let block = &self.buf;
        Self::absorb_block(state, block);
        self.buf_len = 0;
      }
    }

    let state = &mut self.state;
    let (blocks, rest) = data.as_chunks::<RATE>();
    for block in blocks {
      Self::absorb_block(state, block);
    }
    data = rest;

    if !data.is_empty() {
      self.buf[..data.len()].copy_from_slice(data);
      self.buf_len = data.len();
    }
  }

  #[inline(always)]
  fn finalize_state(&self) -> [u64; 25] {
    let mut state = self.state;
    let mut buf = self.buf;
    let buf_len = self.buf_len;

    // Ensure padding happens over a zero-padded block.
    buf[buf_len..].fill(0);

    // SHA-3 padding: domain separator 0x06, then pad10*1 with final 0x80.
    buf[buf_len] ^= 0x06;
    buf[RATE - 1] ^= 0x80;

    // Absorb final padded block.
    let lanes = RATE / 8;
    let (chunks, _) = buf.as_chunks::<8>();
    for (lane, chunk) in state[..lanes].iter_mut().zip(chunks.iter()) {
      *lane ^= u64::from_le_bytes(*chunk);
    }
    keccakf(&mut state);

    state
  }

  #[inline(always)]
  fn finalize_into_fixed<const OUT: usize>(&self, out: &mut [u8; OUT]) {
    debug_assert!(OUT <= RATE);
    let state = self.finalize_state();

    for i in 0..(OUT + 7) / 8 {
      let bytes = state[i].to_le_bytes();
      let start = i * 8;
      let end = core::cmp::min(start + 8, OUT);
      out[start..end].copy_from_slice(&bytes[..end - start]);
    }
  }
}

/// SHA3-256.
#[derive(Clone, Default)]
pub struct Sha3_256 {
  core: Sha3Core<136>,
}

impl Digest for Sha3_256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 32];
    self.core.finalize_into_fixed(&mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

/// SHA3-512.
#[derive(Clone, Default)]
pub struct Sha3_512 {
  core: Sha3Core<72>,
}

impl Digest for Sha3_512 {
  const OUTPUT_SIZE: usize = 64;
  type Output = [u8; 64];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 64];
    self.core.finalize_into_fixed(&mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

#[cfg(test)]
mod tests {
  use traits::Digest;

  use super::{Sha3_256, Sha3_512};

  fn hex(bytes: &[u8]) -> alloc::string::String {
    use alloc::string::String;
    use core::fmt::Write;
    let mut s = String::new();
    for &b in bytes {
      write!(&mut s, "{:02x}", b).unwrap();
    }
    s
  }

  extern crate alloc;

  #[test]
  fn sha3_256_vectors() {
    assert_eq!(
      hex(&Sha3_256::digest(b"")),
      "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
    );
    assert_eq!(
      hex(&Sha3_256::digest(b"abc")),
      "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
    );
  }

  #[test]
  fn sha3_512_vectors() {
    assert_eq!(
      hex(&Sha3_512::digest(b"")),
      "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
    );
    assert_eq!(
      hex(&Sha3_512::digest(b"abc")),
      "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0"
    );
  }
}
