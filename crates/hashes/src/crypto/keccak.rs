//! Keccak-f[1600] sponge core (internal).
//!
//! This module intentionally exposes only the minimum surface needed by SHA-3,
//! SHAKE, and SP800-185 derived constructions.

#![allow(clippy::indexing_slicing)] // Keccak state is fixed-size; indexing is audited

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

  macro_rules! round {
    ($rc:expr) => {{
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
      a0 ^= $rc;
    }};
  }

  round!(RC[0]);
  round!(RC[1]);
  round!(RC[2]);
  round!(RC[3]);
  round!(RC[4]);
  round!(RC[5]);
  round!(RC[6]);
  round!(RC[7]);
  round!(RC[8]);
  round!(RC[9]);
  round!(RC[10]);
  round!(RC[11]);
  round!(RC[12]);
  round!(RC[13]);
  round!(RC[14]);
  round!(RC[15]);
  round!(RC[16]);
  round!(RC[17]);
  round!(RC[18]);
  round!(RC[19]);
  round!(RC[20]);
  round!(RC[21]);
  round!(RC[22]);
  round!(RC[23]);

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
pub(crate) struct KeccakCore<const RATE: usize> {
  state: [u64; 25],
  buf: [u8; RATE],
  buf_len: usize,
}

impl<const RATE: usize> Default for KeccakCore<RATE> {
  #[inline]
  fn default() -> Self {
    Self {
      state: [0u64; 25],
      buf: [0u8; RATE],
      buf_len: 0,
    }
  }
}

impl<const RATE: usize> KeccakCore<RATE> {
  #[inline(always)]
  fn absorb_block(state: &mut [u64; 25], block: &[u8; RATE]) {
    debug_assert_eq!(RATE % 8, 0);
    let lanes = RATE / 8;
    let mut i = 0usize;
    let ptr = block.as_ptr() as *const u64;
    while i < lanes {
      // SAFETY: `RATE % 8 == 0` and `i < lanes == RATE / 8`, so this reads within `block`;
      // `read_unaligned` supports the 1-byte alignment of `[u8; RATE]`.
      let v = unsafe { core::ptr::read_unaligned(ptr.add(i)) };
      state[i] ^= u64::from_le(v);
      i += 1;
    }
    keccakf(state);
  }

  pub(crate) fn update(&mut self, mut data: &[u8]) {
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
  fn finalize_state(&self, ds: u8) -> [u64; 25] {
    let mut state = self.state;
    let mut buf = self.buf;
    let buf_len = self.buf_len;

    // Ensure padding happens over a zero-padded block.
    buf[buf_len..].fill(0);

    // Domain separator, then pad10*1 with final 0x80.
    buf[buf_len] ^= ds;
    buf[RATE - 1] ^= 0x80;

    Self::absorb_block(&mut state, &buf);
    state
  }

  pub(crate) fn finalize_into_fixed<const OUT: usize>(&self, ds: u8, out: &mut [u8; OUT]) {
    debug_assert!(OUT <= RATE);
    let state = self.finalize_state(ds);

    for (i, &word) in state.iter().enumerate().take(OUT.div_ceil(8)) {
      let bytes = word.to_le_bytes();
      let start = i * 8;
      let end = core::cmp::min(start + 8, OUT);
      out[start..end].copy_from_slice(&bytes[..end - start]);
    }
  }

  pub(crate) fn finalize_xof(&self, ds: u8) -> KeccakXof<RATE> {
    let state = self.finalize_state(ds);
    let mut buf = [0u8; RATE];
    KeccakXof::<RATE>::fill_buf(&state, &mut buf);
    KeccakXof { state, buf, pos: 0 }
  }

  pub(crate) fn finalize_xof_into(&self, ds: u8, mut out: &mut [u8]) {
    let mut state = self.finalize_state(ds);

    debug_assert_eq!(RATE % 8, 0);
    let lanes = RATE / 8;
    if out.len() <= RATE {
      let (chunks, rem) = out.as_chunks_mut::<8>();
      let mut i = 0usize;
      while i < chunks.len() {
        chunks[i] = state[i].to_le_bytes();
        i += 1;
      }
      if !rem.is_empty() {
        let bytes = state[chunks.len()].to_le_bytes();
        rem.copy_from_slice(&bytes[..rem.len()]);
      }
      return;
    }

    while !out.is_empty() {
      let mut lane = 0usize;
      while lane < lanes && !out.is_empty() {
        let bytes = state[lane].to_le_bytes();
        let take = core::cmp::min(8, out.len());
        out[..take].copy_from_slice(&bytes[..take]);
        out = &mut out[take..];
        lane += 1;
      }

      if !out.is_empty() {
        keccakf(&mut state);
      }
    }
  }
}

#[derive(Clone)]
pub(crate) struct KeccakXof<const RATE: usize> {
  state: [u64; 25],
  buf: [u8; RATE],
  pos: usize,
}

impl<const RATE: usize> KeccakXof<RATE> {
  #[inline(always)]
  fn fill_buf(state: &[u64; 25], out: &mut [u8; RATE]) {
    debug_assert_eq!(RATE % 8, 0);
    let lanes = RATE / 8;
    let mut i = 0usize;
    while i < lanes {
      let bytes = state[i].to_le_bytes();
      out[i * 8..i * 8 + 8].copy_from_slice(&bytes);
      i += 1;
    }
  }

  pub(crate) fn squeeze_into(&mut self, mut out: &mut [u8]) {
    while !out.is_empty() {
      if self.pos == RATE {
        keccakf(&mut self.state);
        Self::fill_buf(&self.state, &mut self.buf);
        self.pos = 0;
      }

      let take = core::cmp::min(RATE - self.pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
      self.pos += take;
      out = &mut out[take..];
    }
  }
}
