//! Keccak-f[1600] sponge core (internal).
//!
//! This module intentionally exposes only the minimum surface needed by SHA-3,
//! SHAKE, and SP800-185 derived constructions.

#![allow(clippy::indexing_slicing)] // Keccak state is fixed-size; indexing is audited

use core::marker::PhantomData;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[doc(hidden)]
pub(crate) mod dispatch;
#[doc(hidden)]
pub(crate) mod dispatch_tables;
pub(crate) mod kernels;
#[cfg(target_arch = "s390x")]
pub(crate) mod s390x;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_avx512;

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

// `#[inline]` (not `always`) — the function body is large and must not be
// duplicated at every call site. See the loop comment below.
#[inline]
pub(crate) fn keccakf_portable(state: &mut [u64; 25]) {
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

  // Loop over 24 rounds. Keeping the outer loop NOT unrolled is critical:
  // the round body is ~110 scalar ops, so full unrolling produces ~10-15 KB
  // of machine code and causes severe I-cache pressure. A compact loop body
  // fits comfortably in L1i and lets the branch predictor handle the back-edge.
  for &rc in &RC {
    round!(rc);
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

// ---------------------------------------------------------------------------
// Permuter trait + platform-specific implementations
// ---------------------------------------------------------------------------
//
// Direct-call permuters replace the old function-pointer dispatch. Each
// platform gets a concrete `Permuter` that calls the best kernel directly,
// allowing LLVM to inline the permutation into the absorb loop. This is the
// single most impactful change for SHA-3 throughput.
//
// - x86_64 / generic: `InlinePermuter` → `keccakf_portable` (the AVX-512 and AVX2 χ-only SIMD
//   kernels added pack/extract overhead that exceeded the VPTERNLOG savings; pure scalar is
//   faster).
// - aarch64: `Aarch64Permuter` → SHA3 CE when available (branch-based, not
//   function pointer), portable fallback. The 2-state interleaved kernel is
//   preserved for `digest_pair`.
// - s390x: `S390xPermuter` → portable permutation + KIMD batch-absorb.

pub(crate) trait Permuter: Copy {
  fn permute(self, state: &mut [u64; 25], len_hint: usize);

  /// Permute two independent states in parallel.
  /// Default: two sequential single-state permutations.
  #[inline(always)]
  fn permute_x2(self, state_a: &mut [u64; 25], state_b: &mut [u64; 25], len_hint: usize) {
    self.permute(state_a, len_hint);
    self.permute(state_b, len_hint);
  }

  /// Try to batch-absorb complete rate-sized blocks via hardware instruction
  /// (e.g., s390x KIMD). Returns `true` if the hardware path was taken, in
  /// which case the caller must NOT perform the per-block XOR + permute loop.
  ///
  /// Default: returns `false` (fall back to per-block XOR + permute).
  #[inline(always)]
  fn try_absorb_blocks(self, _state: &mut [u64; 25], _blocks: &[u8], _rate: usize) -> bool {
    false
  }
}

/// Direct-call permuter using the portable scalar kernel. No function pointer
/// indirection — LLVM can inline `keccakf_portable` into the absorb loop.
#[derive(Clone, Copy, Default)]
pub(crate) struct InlinePermuter;

impl Permuter for InlinePermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 25], _len_hint: usize) {
    keccakf_portable(state);
  }
}

/// aarch64 permuter: portable for single-state (the 1-state SHA3 CE kernel
/// wastes 50% of the NEON width), SHA3 CE for 2-state interleaved.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) struct Aarch64Permuter {
  has_sha3: bool,
}

#[cfg(target_arch = "aarch64")]
impl Default for Aarch64Permuter {
  #[inline]
  fn default() -> Self {
    Self {
      has_sha3: {
        use crate::platform::caps::aarch64 as aarch64_caps;
        crate::hashes::util::dispatch_caps().has(aarch64_caps::SHA3)
      },
    }
  }
}

#[cfg(target_arch = "aarch64")]
impl Permuter for Aarch64Permuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 25], _len_hint: usize) {
    // Use SHA3 CE when available. The branch is always-taken and perfectly
    // predicted after warmup. Even though the 1-state kernel wastes 50% of
    // the NEON width, some microarchitectures (Apple M-series) have fast
    // enough SHA3 CE that it still wins over portable scalar.
    if self.has_sha3 {
      aarch64::keccakf_aarch64_sha3(state);
    } else {
      keccakf_portable(state);
    }
  }

  #[inline(always)]
  fn permute_x2(self, state_a: &mut [u64; 25], state_b: &mut [u64; 25], len_hint: usize) {
    if self.has_sha3 {
      // The 2-state kernel uses both NEON lanes meaningfully (state_a in
      // lane 0, state_b in lane 1), achieving ~2× aggregate throughput.
      aarch64::keccakf_aarch64_sha3_x2(state_a, state_b);
    } else {
      self.permute(state_a, len_hint);
      self.permute(state_b, len_hint);
    }
  }
}

/// s390x permuter: portable permutation + CPACF KIMD batch-absorb.
#[cfg(target_arch = "s390x")]
#[derive(Clone, Copy)]
pub(crate) struct S390xPermuter {
  has_kimd: bool,
}

#[cfg(target_arch = "s390x")]
impl Default for S390xPermuter {
  #[inline]
  fn default() -> Self {
    Self {
      has_kimd: {
        use crate::platform::caps::s390x as s390x_caps;
        crate::hashes::util::dispatch_caps().has(s390x_caps::MSA8)
      },
    }
  }
}

#[cfg(target_arch = "s390x")]
impl Permuter for S390xPermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 25], _len_hint: usize) {
    keccakf_portable(state);
  }

  #[inline(always)]
  fn try_absorb_blocks(self, state: &mut [u64; 25], blocks: &[u8], rate: usize) -> bool {
    if self.has_kimd
      && let Some(fc) = s390x::kimd_fc_for_rate(rate)
    {
      // SAFETY: MSA8 verified at construction time (has_kimd flag).
      unsafe {
        s390x::absorb_blocks_kimd(state, blocks, fc);
      }
      return true;
    }
    false
  }
}

// ---------------------------------------------------------------------------
// Platform-specific permuter selection
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub(crate) type PlatformPermuter = Aarch64Permuter;

#[cfg(target_arch = "s390x")]
pub(crate) type PlatformPermuter = S390xPermuter;

#[cfg(not(any(target_arch = "aarch64", target_arch = "s390x")))]
pub(crate) type PlatformPermuter = InlinePermuter;

pub(crate) type KeccakCore<const RATE: usize> = KeccakCoreImpl<RATE, PlatformPermuter>;

#[cfg(any(test, feature = "std"))]
pub(crate) type KeccakCorePortable<const RATE: usize> = KeccakCoreImpl<RATE, InlinePermuter>;

pub(crate) type KeccakXof<const RATE: usize> = KeccakXofImpl<RATE, PlatformPermuter>;

// ---------------------------------------------------------------------------
// Sponge core
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct KeccakCoreImpl<const RATE: usize, P: Permuter> {
  state: [u64; 25],
  buf: [u8; RATE],
  buf_len: usize,
  bytes_hint: usize,
  permuter: P,
  _p: PhantomData<P>,
}

impl<const RATE: usize> Default for KeccakCoreImpl<RATE, PlatformPermuter> {
  #[inline]
  fn default() -> Self {
    Self {
      state: [0u64; 25],
      buf: [0u8; RATE],
      buf_len: 0,
      bytes_hint: 0,
      permuter: PlatformPermuter::default(),
      _p: PhantomData,
    }
  }
}

// On aarch64/s390x, `PlatformPermuter` is NOT `InlinePermuter`, so the
// portable-reference type alias needs its own `Default`. On other targets
// `PlatformPermuter = InlinePermuter` and the impl above already covers it.
#[cfg(all(any(test, feature = "std"), any(target_arch = "aarch64", target_arch = "s390x")))]
impl<const RATE: usize> Default for KeccakCoreImpl<RATE, InlinePermuter> {
  #[inline]
  fn default() -> Self {
    Self {
      state: [0u64; 25],
      buf: [0u8; RATE],
      buf_len: 0,
      bytes_hint: 0,
      permuter: InlinePermuter,
      _p: PhantomData,
    }
  }
}

impl<const RATE: usize, P: Permuter> Drop for KeccakCoreImpl<RATE, P> {
  fn drop(&mut self) {
    for word in self.state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.buf);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl<const RATE: usize, P: Permuter> KeccakCoreImpl<RATE, P> {
  #[inline(always)]
  fn absorb_block(permuter: P, len_hint: usize, state: &mut [u64; 25], block: &[u8; RATE]) {
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
    permuter.permute(state, len_hint);
  }

  pub(crate) fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

    self.bytes_hint = self.bytes_hint.wrapping_add(data.len());
    let permuter = self.permuter;
    let len_hint = self.bytes_hint;
    if self.buf_len != 0 {
      let take = core::cmp::min(RATE - self.buf_len, data.len());
      self.buf[self.buf_len..self.buf_len.strict_add(take)].copy_from_slice(&data[..take]);
      self.buf_len = self.buf_len.strict_add(take);
      data = &data[take..];

      if self.buf_len == RATE {
        let state = &mut self.state;
        let block = &self.buf;
        Self::absorb_block(permuter, len_hint, state, block);
        self.buf_len = 0;
      }
    }

    let state = &mut self.state;
    let (blocks, rest) = data.as_chunks::<RATE>();
    if !blocks.is_empty() {
      let block_bytes = &data[..blocks.len().strict_mul(RATE)];
      if !permuter.try_absorb_blocks(state, block_bytes, RATE) {
        for block in blocks {
          Self::absorb_block(permuter, len_hint, state, block);
        }
      }
    }
    data = rest;

    if !data.is_empty() {
      self.buf[..data.len()].copy_from_slice(data);
      self.buf_len = data.len();
    }
  }

  #[inline(always)]
  fn finalize_state(&self, ds: u8) -> [u64; 25] {
    let permuter = self.permuter;
    let len_hint = self.bytes_hint;
    let mut state = self.state;
    let mut buf = self.buf;
    let buf_len = self.buf_len;

    debug_assert!(buf_len < RATE, "buf_len={} should be < RATE={}", buf_len, RATE);

    // Ensure padding happens over a zero-padded block.
    buf[buf_len..].fill(0);

    // Domain separator, then pad10*1 with final 0x80.
    // SAFETY: buf_len < RATE is guaranteed by the assertion above.
    buf[buf_len] ^= ds;
    buf[RATE - 1] ^= 0x80;

    Self::absorb_block(permuter, len_hint, &mut state, &buf);
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

  pub(crate) fn finalize_xof(&self, ds: u8) -> KeccakXofImpl<RATE, P> {
    let permuter = self.permuter;
    let state = self.finalize_state(ds);
    let mut buf = [0u8; RATE];
    KeccakXofImpl::<RATE, P>::fill_buf(&state, &mut buf);
    KeccakXofImpl {
      state,
      buf,
      pos: 0,
      bytes_hint: self.bytes_hint,
      bytes_out: 0,
      permuter,
      _p: PhantomData,
    }
  }

  pub(crate) fn finalize_xof_into(&self, ds: u8, mut out: &mut [u8]) {
    let permuter = self.permuter;
    let len_hint = self.bytes_hint.wrapping_add(out.len());
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
        permuter.permute(&mut state, len_hint);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2-state parallel oneshot (digest_pair)
// ---------------------------------------------------------------------------

/// XOR a RATE-sized block into the Keccak state.
#[inline(always)]
fn xor_block_into<const RATE: usize>(state: &mut [u64; 25], block: &[u8; RATE]) {
  debug_assert_eq!(RATE % 8, 0);
  let lanes = RATE / 8;
  let ptr = block.as_ptr() as *const u64;
  let mut i = 0usize;
  while i < lanes {
    // SAFETY: `RATE % 8 == 0` and `i < lanes == RATE / 8`, so this reads within `block`;
    // `read_unaligned` supports the 1-byte alignment of `[u8; RATE]`.
    let v = unsafe { core::ptr::read_unaligned(ptr.add(i)) };
    state[i] ^= u64::from_le(v);
    i += 1;
  }
}

/// Pad and absorb the final block for a Keccak sponge.
#[inline(always)]
fn finalize_pad_absorb<const RATE: usize>(
  state: &mut [u64; 25],
  remainder: &[u8],
  ds: u8,
  permuter: PlatformPermuter,
  len_hint: usize,
) {
  let mut buf = [0u8; RATE];
  buf[..remainder.len()].copy_from_slice(remainder);
  buf[remainder.len()] ^= ds;
  buf[RATE - 1] ^= 0x80;
  xor_block_into::<RATE>(state, &buf);
  permuter.permute(state, len_hint);
}

/// Extract fixed-size output from a Keccak state.
#[inline(always)]
fn extract_output<const OUT: usize>(state: &[u64; 25], out: &mut [u8; OUT]) {
  for (i, &word) in state.iter().enumerate().take(OUT.div_ceil(8)) {
    let bytes = word.to_le_bytes();
    let start = i.strict_mul(8);
    let end = core::cmp::min(start.strict_add(8), OUT);
    out[start..end].copy_from_slice(&bytes[..end.strict_sub(start)]);
  }
}

/// Hash two independent messages in parallel using 2-state interleaved
/// permutation (aarch64 SHA3 CE) or sequential fallback.
///
/// On aarch64 with SHA3 CE, this achieves ~2× the aggregate throughput of
/// two sequential hash computations.
pub(crate) fn oneshot_pair<const RATE: usize, const OUT: usize>(
  ds: u8,
  data_a: &[u8],
  data_b: &[u8],
) -> ([u8; OUT], [u8; OUT]) {
  debug_assert!(OUT <= RATE);
  let permuter = PlatformPermuter::default();
  let mut state_a = [0u64; 25];
  let mut state_b = [0u64; 25];
  let len_hint = core::cmp::max(data_a.len(), data_b.len());

  // Split into complete blocks + remainder.
  let (blocks_a, rest_a) = data_a.as_chunks::<RATE>();
  let (blocks_b, rest_b) = data_b.as_chunks::<RATE>();
  let min_blocks = core::cmp::min(blocks_a.len(), blocks_b.len());

  // Process paired blocks using 2-state interleaved permutation.
  for i in 0..min_blocks {
    xor_block_into::<RATE>(&mut state_a, &blocks_a[i]);
    xor_block_into::<RATE>(&mut state_b, &blocks_b[i]);
    permuter.permute_x2(&mut state_a, &mut state_b, len_hint);
  }

  // Process remaining blocks of the longer input with single-state.
  for block in &blocks_a[min_blocks..] {
    xor_block_into::<RATE>(&mut state_a, block);
    permuter.permute(&mut state_a, len_hint);
  }
  for block in &blocks_b[min_blocks..] {
    xor_block_into::<RATE>(&mut state_b, block);
    permuter.permute(&mut state_b, len_hint);
  }

  // Finalize both states (pad + final absorb + extract).
  finalize_pad_absorb::<RATE>(&mut state_a, rest_a, ds, permuter, len_hint);
  finalize_pad_absorb::<RATE>(&mut state_b, rest_b, ds, permuter, len_hint);

  let mut out_a = [0u8; OUT];
  let mut out_b = [0u8; OUT];
  extract_output::<OUT>(&state_a, &mut out_a);
  extract_output::<OUT>(&state_b, &mut out_b);

  // Zeroize state.
  for word in state_a.iter_mut().chain(state_b.iter_mut()) {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

  (out_a, out_b)
}

#[derive(Clone)]
pub(crate) struct KeccakXofImpl<const RATE: usize, P: Permuter> {
  state: [u64; 25],
  buf: [u8; RATE],
  pos: usize,
  bytes_hint: usize,
  bytes_out: usize,
  permuter: P,
  _p: PhantomData<P>,
}

impl<const RATE: usize, P: Permuter> Drop for KeccakXofImpl<RATE, P> {
  fn drop(&mut self) {
    for word in self.state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.buf);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl<const RATE: usize, P: Permuter> KeccakXofImpl<RATE, P> {
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
        let permuter = self.permuter;
        let len_hint = self.bytes_hint.wrapping_add(self.bytes_out);
        permuter.permute(&mut self.state, len_hint);
        Self::fill_buf(&self.state, &mut self.buf);
        self.pos = 0;
      }

      let take = core::cmp::min(RATE - self.pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.pos..self.pos.strict_add(take)]);
      self.bytes_out = self.bytes_out.wrapping_add(take);
      self.pos = self.pos.strict_add(take);
      out = &mut out[take..];
    }
  }
}

#[cfg(feature = "std")]
pub(crate) mod kernel_test;
