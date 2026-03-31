//! Ascon hash and XOF (NIST LWC).
//!
//! Portable, `no_std`, pure Rust implementation.

#![allow(clippy::indexing_slicing)] // Fixed-size state + sponge buffering

use core::fmt;

use crate::{
  hashes::crypto::dispatch_util::SizeClassDispatch,
  traits::{Digest, Xof},
};

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[doc(hidden)]
pub(crate) mod dispatch;
#[doc(hidden)]
pub(crate) mod dispatch_tables;
pub(crate) mod kernels;
#[cfg(target_arch = "x86_64")]
mod x86_64_avx2;
#[cfg(target_arch = "x86_64")]
mod x86_64_avx512;

const RATE: usize = 8;

trait Permuter: Copy {
  fn permute(self, state: &mut [u64; 5], len_hint: usize);
}

#[derive(Clone, Copy)]
struct DispatchPermuter {
  dispatch: SizeClassDispatch<fn(&mut [u64; 5])>,
}

impl Default for DispatchPermuter {
  #[inline]
  fn default() -> Self {
    Self {
      dispatch: dispatch::permute_dispatch(),
    }
  }
}

impl Permuter for DispatchPermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 5], len_hint: usize) {
    (self.dispatch.select(len_hint))(state);
  }
}

#[cfg(any(test, feature = "std"))]
#[derive(Clone, Copy)]
struct FixedPermuter {
  permute: fn(&mut [u64; 5]),
}

#[cfg(any(test, feature = "std"))]
impl Permuter for FixedPermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 5], _len_hint: usize) {
    (self.permute)(state);
  }
}

// Ascon permutation round constants (12 rounds).
const RC: [u64; 12] = [0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B];

// Domain-specific IVs (from the Ascon hash/XOF specification).
const HASH256_IV: [u64; 5] = [
  0x9b1e_5494_e934_d681,
  0x4bc3_a01e_3337_51d2,
  0xae65_396c_6b34_b81a,
  0x3c7f_d4a4_d56a_4db3,
  0x1a5c_4649_06c5_976d,
];

const XOF128_IV: [u64; 5] = [
  0xda82_ce76_8d94_47eb,
  0xcc7c_e6c7_5f1e_f969,
  0xe750_8fd7_8008_5631,
  0x0ee0_ea53_416b_58cc,
  0xe054_7524_db6f_0bde,
];

const CXOF128_IV: [u64; 5] = [0x0000_0800_00cc_0004, 0, 0, 0, 0];

/// Ascon-CXOF128 customization strings are limited to 256 bytes by SP 800-232.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsconCxofCustomizationError;

impl AsconCxofCustomizationError {
  /// Construct a new customization-length error.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self
  }
}

impl Default for AsconCxofCustomizationError {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl fmt::Display for AsconCxofCustomizationError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("Ascon-CXOF128 customization exceeds 256 bytes")
  }
}

impl core::error::Error for AsconCxofCustomizationError {}

#[inline(always)]
const fn pad(n: usize) -> u64 {
  // Produce the padding mask used by the reference construction:
  // XOR `pad(len)` into state[0], with state interpreted little-endian.
  0x01_u64 << (8 * n)
}

#[inline(always)]
pub(crate) fn permute_12_portable(s: &mut [u64; 5]) {
  for &c in &RC {
    round(s, c);
  }
}

/// 6-round Ascon permutation (PB) used by Ascon-AEAD128.
#[inline(always)]
pub(crate) fn permute_6_portable(s: &mut [u64; 5]) {
  // PB uses the last 6 round constants (rounds 6..12).
  for &c in &RC[6..] {
    round(s, c);
  }
}

#[inline(always)]
fn round(s: &mut [u64; 5], c: u64) {
  let mut x0 = s[0];
  let mut x1 = s[1];
  let mut x2 = s[2];
  let mut x3 = s[3];
  let mut x4 = s[4];

  // Add round constant.
  x2 ^= c;

  // Substitution layer.
  x0 ^= x4;
  x4 ^= x3;
  x2 ^= x1;

  let t0 = (!x0) & x1;
  let t1 = (!x1) & x2;
  let t2 = (!x2) & x3;
  let t3 = (!x3) & x4;
  let t4 = (!x4) & x0;

  x0 ^= t1;
  x1 ^= t2;
  x2 ^= t3;
  x3 ^= t4;
  x4 ^= t0;

  x1 ^= x0;
  x0 ^= x4;
  x3 ^= x2;
  x2 = !x2;

  // Linear diffusion layer.
  x0 ^= x0.rotate_right(19) ^ x0.rotate_right(28);
  x1 ^= x1.rotate_right(61) ^ x1.rotate_right(39);
  x2 ^= x2.rotate_right(1) ^ x2.rotate_right(6);
  x3 ^= x3.rotate_right(10) ^ x3.rotate_right(17);
  x4 ^= x4.rotate_right(7) ^ x4.rotate_right(41);

  s[0] = x0;
  s[1] = x1;
  s[2] = x2;
  s[3] = x3;
  s[4] = x4;
}

#[derive(Clone)]
struct Sponge<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64> {
  state: [u64; 5],
  buf: [u8; RATE],
  buf_len: usize,
  bytes_hint: usize,
  permuter: P,
}

impl<P: Permuter + Default, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64> Default
  for Sponge<P, IV0, IV1, IV2, IV3, IV4>
{
  #[inline]
  fn default() -> Self {
    Self {
      state: [IV0, IV1, IV2, IV3, IV4],
      buf: [0u8; RATE],
      buf_len: 0,
      bytes_hint: 0,
      permuter: P::default(),
    }
  }
}

impl<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64> Drop
  for Sponge<P, IV0, IV1, IV2, IV3, IV4>
{
  fn drop(&mut self) {
    for word in self.state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.buf);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64>
  Sponge<P, IV0, IV1, IV2, IV3, IV4>
{
  #[inline(always)]
  fn absorb_block(&mut self, block: &[u8; RATE]) {
    self.state[0] ^= u64::from_le_bytes(*block);
    let permuter = self.permuter;
    let len_hint = self.bytes_hint;
    permuter.permute(&mut self.state, len_hint);
  }

  fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

    // Kernel selection only uses this as an advisory size hint, so wrapping is intentional.
    self.bytes_hint = self.bytes_hint.wrapping_add(data.len());
    if self.buf_len != 0 {
      let take = core::cmp::min(RATE - self.buf_len, data.len());
      self.buf[self.buf_len..self.buf_len.strict_add(take)].copy_from_slice(&data[..take]);
      self.buf_len = self.buf_len.strict_add(take);
      data = &data[take..];

      if self.buf_len == RATE {
        let block = self.buf;
        self.absorb_block(&block);
        self.buf_len = 0;
      }
    }

    let (blocks, rest) = data.as_chunks::<RATE>();
    for block in blocks {
      self.absorb_block(block);
    }
    data = rest;

    if !data.is_empty() {
      self.buf[..data.len()].copy_from_slice(data);
      self.buf_len = data.len();
    }
  }

  fn finalize_state(&self) -> [u64; 5] {
    let mut st = self.state;
    let permuter = self.permuter;
    let len_hint = self.bytes_hint;
    let last = &self.buf[..self.buf_len];

    let mut tmp = [0u8; RATE];
    tmp[..last.len()].copy_from_slice(last);
    st[0] ^= u64::from_le_bytes(tmp);
    st[0] ^= pad(last.len());
    permuter.permute(&mut st, len_hint);

    st
  }

  fn finalize_state_with_hint(&self) -> ([u64; 5], usize) {
    (self.finalize_state(), self.bytes_hint)
  }
}

#[cfg(any(test, feature = "std"))]
#[inline]
fn fixed_sponge<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64>(
  permuter: P,
) -> Sponge<P, IV0, IV1, IV2, IV3, IV4> {
  Sponge {
    state: [IV0, IV1, IV2, IV3, IV4],
    buf: [0u8; RATE],
    buf_len: 0,
    bytes_hint: 0,
    permuter,
  }
}

#[cfg(any(test, feature = "std"))]
#[inline(always)]
const fn init_states<const N: usize>(iv: [u64; 5]) -> [[u64; 5]; N] {
  [iv; N]
}

#[cfg(any(test, feature = "std"))]
#[inline(always)]
fn permute_12_many_portable<const N: usize>(states: &mut [[u64; 5]; N]) {
  for state in states {
    permute_12_portable(state);
  }
}

#[cfg(any(test, feature = "std"))]
fn absorb_equal_len_group<const N: usize>(
  states: &mut [[u64; 5]; N],
  inputs: &[&[u8]],
  permute_many: fn(&mut [[u64; 5]; N]),
) {
  debug_assert_eq!(inputs.len(), N);
  let len = inputs[0].len();
  let full_bytes = len / RATE * RATE;

  for off in (0..full_bytes).step_by(RATE) {
    for (state, input) in states.iter_mut().zip(inputs.iter().copied()) {
      let mut block = [0u8; RATE];
      block.copy_from_slice(&input[off..off + RATE]);
      state[0] ^= u64::from_le_bytes(block);
    }
    permute_many(states);
  }

  let tail_len = len - full_bytes;
  for (state, input) in states.iter_mut().zip(inputs.iter().copied()) {
    let mut block = [0u8; RATE];
    block[..tail_len].copy_from_slice(&input[full_bytes..]);
    state[0] ^= u64::from_le_bytes(block);
    state[0] ^= pad(tail_len);
  }
  permute_many(states);
}

#[cfg(any(test, feature = "std"))]
fn squeeze_hash256_group<const N: usize>(
  states: &mut [[u64; 5]; N],
  outputs: &mut [[u8; 32]],
  permute_many: fn(&mut [[u64; 5]; N]),
) {
  debug_assert_eq!(outputs.len(), N);
  let mut off = 0usize;
  while off < 24 {
    for (state, output) in states.iter().zip(outputs.iter_mut()) {
      output[off..off + RATE].copy_from_slice(&state[0].to_le_bytes());
    }
    permute_many(states);
    off += RATE;
  }
  for (state, output) in states.iter().zip(outputs.iter_mut()) {
    output[24..32].copy_from_slice(&state[0].to_le_bytes());
  }
}

#[cfg(any(test, feature = "std"))]
fn squeeze_xof_group<const N: usize>(
  states: &mut [[u64; 5]; N],
  out_len: usize,
  outputs: &mut [u8],
  permute_many: fn(&mut [[u64; 5]; N]),
) {
  debug_assert_eq!(outputs.len(), N * out_len);
  let mut produced = 0usize;
  while produced < out_len {
    let take = core::cmp::min(RATE, out_len - produced);
    for (state, output) in states.iter().zip(outputs.chunks_exact_mut(out_len)) {
      output[produced..produced + take].copy_from_slice(&state[0].to_le_bytes()[..take]);
    }
    produced += take;
    if produced < out_len {
      permute_many(states);
    }
  }
}

#[cfg(any(test, feature = "std"))]
#[inline]
fn inputs_have_equal_len(inputs: &[&[u8]]) -> bool {
  inputs
    .split_first()
    .is_none_or(|(first, rest)| rest.iter().all(|input| input.len() == first.len()))
}

#[cfg(any(test, feature = "std"))]
fn digest_many_equal_len_group<const N: usize>(
  inputs: &[&[u8]],
  outputs: &mut [[u8; 32]],
  iv: [u64; 5],
  permute_many: fn(&mut [[u64; 5]; N]),
) {
  debug_assert_eq!(inputs.len(), N);
  debug_assert_eq!(outputs.len(), N);
  let mut states = init_states::<N>(iv);
  absorb_equal_len_group(&mut states, inputs, permute_many);
  squeeze_hash256_group(&mut states, outputs, permute_many);
}

#[cfg(any(test, feature = "std"))]
fn xof_many_equal_len_group<const N: usize>(
  inputs: &[&[u8]],
  out_len: usize,
  outputs: &mut [u8],
  iv: [u64; 5],
  permute_many: fn(&mut [[u64; 5]; N]),
) {
  debug_assert_eq!(inputs.len(), N);
  debug_assert_eq!(outputs.len(), N * out_len);
  let mut states = init_states::<N>(iv);
  absorb_equal_len_group(&mut states, inputs, permute_many);
  squeeze_xof_group(&mut states, out_len, outputs, permute_many);
}

/// Ascon-Hash256.
#[derive(Clone, Default)]
pub struct AsconHash256 {
  sponge: Sponge<
    DispatchPermuter,
    { HASH256_IV[0] },
    { HASH256_IV[1] },
    { HASH256_IV[2] },
    { HASH256_IV[3] },
    { HASH256_IV[4] },
  >,
}

impl AsconHash256 {
  #[inline]
  #[cfg(any(test, feature = "std"))]
  fn batch_kernel_id_for_inputs(inputs: &[&[u8]]) -> kernels::AsconPermute12KernelId {
    inputs
      .first()
      .map_or(kernels::AsconPermute12KernelId::Portable, |first| {
        let caps = crate::platform::caps();
        let table = dispatch_tables::select_runtime_table(caps);
        let candidate = table.kernel_for_len(first.len());
        if caps.has(kernels::required_caps(candidate)) {
          candidate
        } else {
          kernels::AsconPermute12KernelId::Portable
        }
      })
  }

  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_with_kernel(kid: kernels::AsconPermute12KernelId, data: &[u8]) -> [u8; 32] {
    let permute = kernels::permute_fn(kid);
    let mut sponge: Sponge<
      FixedPermuter,
      { HASH256_IV[0] },
      { HASH256_IV[1] },
      { HASH256_IV[2] },
      { HASH256_IV[3] },
      { HASH256_IV[4] },
    > = fixed_sponge(FixedPermuter { permute });
    sponge.update(data);

    let mut st = sponge.finalize_state();
    let mut out = [0u8; 32];
    let mut off = 0usize;
    while off < 24 {
      out[off..off + 8].copy_from_slice(&st[0].to_le_bytes());
      permute(&mut st);
      off += 8;
    }
    out[24..32].copy_from_slice(&st[0].to_le_bytes());
    out
  }

  /// Multi-message one-shot hashing using an explicitly selected kernel.
  ///
  /// `outputs.len()` must equal `inputs.len()`.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  #[doc(hidden)]
  pub fn digest_many_with_kernel(kid: kernels::AsconPermute12KernelId, inputs: &[&[u8]], outputs: &mut [[u8; 32]]) {
    assert_eq!(inputs.len(), outputs.len(), "input/output batch length mismatch");

    let degree = kernels::simd_degree(kid);
    if degree == 1 || inputs.len() < degree || !inputs_have_equal_len(inputs) {
      for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
        *output = Self::digest_with_kernel(kid, input);
      }
      return;
    }

    match kid {
      kernels::AsconPermute12KernelId::Portable => {
        let mut input_groups = inputs.chunks_exact(1);
        let mut output_groups = outputs.chunks_exact_mut(1);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          digest_many_equal_len_group::<1>(group_inputs, group_outputs, HASH256_IV, permute_12_many_portable::<1>);
        }
      }
      #[cfg(target_arch = "aarch64")]
      kernels::AsconPermute12KernelId::Aarch64Neon => {
        let mut input_groups = inputs.chunks_exact(2);
        let mut output_groups = outputs.chunks_exact_mut(2);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          digest_many_equal_len_group::<2>(
            group_inputs,
            group_outputs,
            HASH256_IV,
            aarch64::permute_12_aarch64_neon_x2,
          );
        }
        for (input, output) in input_groups
          .remainder()
          .iter()
          .zip(output_groups.into_remainder().iter_mut())
        {
          *output = Self::digest_with_kernel(kid, input);
        }
      }
      #[cfg(target_arch = "x86_64")]
      kernels::AsconPermute12KernelId::X86Avx2 => {
        let mut input_groups = inputs.chunks_exact(4);
        let mut output_groups = outputs.chunks_exact_mut(4);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          digest_many_equal_len_group::<4>(
            group_inputs,
            group_outputs,
            HASH256_IV,
            x86_64_avx2::permute_12_x86_avx2_x4,
          );
        }
        for (input, output) in input_groups
          .remainder()
          .iter()
          .zip(output_groups.into_remainder().iter_mut())
        {
          *output = Self::digest_with_kernel(kid, input);
        }
      }
      #[cfg(target_arch = "x86_64")]
      kernels::AsconPermute12KernelId::X86Avx512 => {
        let mut input_groups = inputs.chunks_exact(8);
        let mut output_groups = outputs.chunks_exact_mut(8);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          digest_many_equal_len_group::<8>(
            group_inputs,
            group_outputs,
            HASH256_IV,
            x86_64_avx512::permute_12_x86_avx512_x8,
          );
        }
        for (input, output) in input_groups
          .remainder()
          .iter()
          .zip(output_groups.into_remainder().iter_mut())
        {
          *output = Self::digest_with_kernel(kid, input);
        }
      }
    }
  }

  /// Multi-message one-shot hashing with automatic kernel selection.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  #[doc(hidden)]
  pub fn digest_many(inputs: &[&[u8]], outputs: &mut [[u8; 32]]) {
    Self::digest_many_with_kernel(Self::batch_kernel_id_for_inputs(inputs), inputs, outputs);
  }
}

impl core::fmt::Debug for AsconHash256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("AsconHash256").finish_non_exhaustive()
  }
}

impl Digest for AsconHash256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.sponge.update(data);
  }

  fn finalize(&self) -> Self::Output {
    let (mut st, hint) = self.sponge.finalize_state_with_hint();

    let mut out = [0u8; 32];
    let mut off = 0usize;
    while off < 24 {
      out[off..off + 8].copy_from_slice(&st[0].to_le_bytes());
      dispatch::permute_12_for_len(&mut st, hint.wrapping_add(off));
      off += 8;
    }
    out[24..32].copy_from_slice(&st[0].to_le_bytes());
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

/// Ascon-XOF128 hasher.
#[derive(Clone, Default)]
pub struct AsconXof {
  sponge:
    Sponge<DispatchPermuter, { XOF128_IV[0] }, { XOF128_IV[1] }, { XOF128_IV[2] }, { XOF128_IV[3] }, { XOF128_IV[4] }>,
}

impl fmt::Debug for AsconXof {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("AsconXof").finish_non_exhaustive()
  }
}

impl AsconXof {
  #[inline]
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  #[must_use]
  pub fn xof(data: &[u8]) -> AsconXofReader {
    let mut h = Self::new();
    h.update(data);
    h.finalize_xof()
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.sponge.update(data);
  }

  #[inline]
  pub fn reset(&mut self) {
    *self = Self::default();
  }

  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> AsconXofReader {
    let (state, hint) = self.sponge.finalize_state_with_hint();
    AsconXofReader {
      state,
      buf: [0u8; RATE],
      pos: RATE,
      hint,
      bytes_out: 0,
    }
  }

  #[inline]
  /// Convenience one-shot XOF output for callers that only need bytes.
  ///
  /// The canonical one-shot API is [`Self::xof`]. Use [`Self::new`],
  /// [`Self::update`], and [`Self::finalize_xof`] for streaming.
  pub fn hash_into(data: &[u8], out: &mut [u8]) {
    Self::xof(data).squeeze(out);
  }

  #[inline]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn hash_into_with_kernel(kid: kernels::AsconPermute12KernelId, data: &[u8], mut out: &mut [u8]) {
    let permute = kernels::permute_fn(kid);
    let mut sponge: Sponge<
      FixedPermuter,
      { XOF128_IV[0] },
      { XOF128_IV[1] },
      { XOF128_IV[2] },
      { XOF128_IV[3] },
      { XOF128_IV[4] },
    > = fixed_sponge(FixedPermuter { permute });
    sponge.update(data);
    let mut state = sponge.finalize_state();

    let mut buf = [0u8; RATE];
    let mut pos = RATE;
    while !out.is_empty() {
      if pos == RATE {
        buf = state[0].to_le_bytes();
        permute(&mut state);
        pos = 0;
      }

      let take = core::cmp::min(RATE - pos, out.len());
      out[..take].copy_from_slice(&buf[pos..pos + take]);
      out = &mut out[take..];
      pos += take;
    }
  }

  /// Multi-message XOF using an explicitly selected kernel.
  ///
  /// `outputs` is a flat buffer laid out as `inputs.len()` adjacent outputs of
  /// `out_len` bytes each.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  #[doc(hidden)]
  pub fn hash_many_into_with_kernel(
    kid: kernels::AsconPermute12KernelId,
    inputs: &[&[u8]],
    out_len: usize,
    outputs: &mut [u8],
  ) {
    assert_eq!(
      outputs.len(),
      inputs.len() * out_len,
      "input/output batch length mismatch"
    );

    let degree = kernels::simd_degree(kid);
    if degree == 1 || inputs.len() < degree || !inputs_have_equal_len(inputs) {
      for (index, input) in inputs.iter().enumerate() {
        let base = index * out_len;
        Self::hash_into_with_kernel(kid, input, &mut outputs[base..base + out_len]);
      }
      return;
    }

    match kid {
      kernels::AsconPermute12KernelId::Portable => {
        let mut input_groups = inputs.chunks_exact(1);
        let mut output_groups = outputs.chunks_exact_mut(out_len);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          xof_many_equal_len_group::<1>(
            group_inputs,
            out_len,
            group_outputs,
            XOF128_IV,
            permute_12_many_portable::<1>,
          );
        }
      }
      #[cfg(target_arch = "aarch64")]
      kernels::AsconPermute12KernelId::Aarch64Neon => {
        let mut input_groups = inputs.chunks_exact(2);
        let mut output_groups = outputs.chunks_exact_mut(2 * out_len);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          xof_many_equal_len_group::<2>(
            group_inputs,
            out_len,
            group_outputs,
            XOF128_IV,
            aarch64::permute_12_aarch64_neon_x2,
          );
        }
        for (input, output) in input_groups
          .remainder()
          .iter()
          .zip(output_groups.into_remainder().chunks_exact_mut(out_len))
        {
          Self::hash_into_with_kernel(kid, input, output);
        }
      }
      #[cfg(target_arch = "x86_64")]
      kernels::AsconPermute12KernelId::X86Avx2 => {
        let mut input_groups = inputs.chunks_exact(4);
        let mut output_groups = outputs.chunks_exact_mut(4 * out_len);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          xof_many_equal_len_group::<4>(
            group_inputs,
            out_len,
            group_outputs,
            XOF128_IV,
            x86_64_avx2::permute_12_x86_avx2_x4,
          );
        }
        for (input, output) in input_groups
          .remainder()
          .iter()
          .zip(output_groups.into_remainder().chunks_exact_mut(out_len))
        {
          Self::hash_into_with_kernel(kid, input, output);
        }
      }
      #[cfg(target_arch = "x86_64")]
      kernels::AsconPermute12KernelId::X86Avx512 => {
        let mut input_groups = inputs.chunks_exact(8);
        let mut output_groups = outputs.chunks_exact_mut(8 * out_len);
        for (group_inputs, group_outputs) in input_groups.by_ref().zip(output_groups.by_ref()) {
          xof_many_equal_len_group::<8>(
            group_inputs,
            out_len,
            group_outputs,
            XOF128_IV,
            x86_64_avx512::permute_12_x86_avx512_x8,
          );
        }
        for (input, output) in input_groups
          .remainder()
          .iter()
          .zip(output_groups.into_remainder().chunks_exact_mut(out_len))
        {
          Self::hash_into_with_kernel(kid, input, output);
        }
      }
    }
  }

  /// Multi-message XOF with automatic kernel selection.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  #[doc(hidden)]
  pub fn hash_many_into(inputs: &[&[u8]], out_len: usize, outputs: &mut [u8]) {
    let kid = AsconHash256::batch_kernel_id_for_inputs(inputs);
    Self::hash_many_into_with_kernel(kid, inputs, out_len, outputs);
  }
}

/// Ascon-XOF128 reader.
#[derive(Clone)]
pub struct AsconXofReader {
  state: [u64; 5],
  buf: [u8; RATE],
  pos: usize,
  hint: usize,
  bytes_out: usize,
}

impl fmt::Debug for AsconXofReader {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("AsconXofReader").finish_non_exhaustive()
  }
}

impl Drop for AsconXofReader {
  fn drop(&mut self) {
    for word in self.state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.buf);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl AsconXofReader {
  #[inline(always)]
  fn refill(&mut self) {
    self.buf = self.state[0].to_le_bytes();
    dispatch::permute_12_for_len(&mut self.state, self.hint.wrapping_add(self.bytes_out));
    self.pos = 0;
  }
}

impl Xof for AsconXofReader {
  fn squeeze(&mut self, mut out: &mut [u8]) {
    while !out.is_empty() {
      if self.pos == RATE {
        self.refill();
      }

      let take = core::cmp::min(RATE - self.pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.pos..self.pos.strict_add(take)]);
      self.bytes_out = self.bytes_out.wrapping_add(take);
      self.pos = self.pos.strict_add(take);
      out = &mut out[take..];
    }
  }
}

/// Spec-precise alias for [`AsconXof`].
pub type AsconXof128 = AsconXof;

/// Spec-precise alias for [`AsconXofReader`].
pub type AsconXof128Xof = AsconXofReader;

/// Ascon-CXOF128 hasher with explicit customization.
#[derive(Clone)]
pub struct AsconCxof128 {
  sponge: Sponge<
    DispatchPermuter,
    { CXOF128_IV[0] },
    { CXOF128_IV[1] },
    { CXOF128_IV[2] },
    { CXOF128_IV[3] },
    { CXOF128_IV[4] },
  >,
  initial_sponge: Sponge<
    DispatchPermuter,
    { CXOF128_IV[0] },
    { CXOF128_IV[1] },
    { CXOF128_IV[2] },
    { CXOF128_IV[3] },
    { CXOF128_IV[4] },
  >,
}

impl fmt::Debug for AsconCxof128 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("AsconCxof128").finish_non_exhaustive()
  }
}

impl AsconCxof128 {
  /// Maximum customization length in bytes.
  pub const MAX_CUSTOMIZATION_LEN: usize = 256;

  /// Construct a customized Ascon-CXOF128 state.
  pub fn new(customization: &[u8]) -> Result<Self, AsconCxofCustomizationError> {
    if customization.len() > Self::MAX_CUSTOMIZATION_LEN {
      return Err(AsconCxofCustomizationError::new());
    }

    let mut sponge: Sponge<
      DispatchPermuter,
      { CXOF128_IV[0] },
      { CXOF128_IV[1] },
      { CXOF128_IV[2] },
      { CXOF128_IV[3] },
      { CXOF128_IV[4] },
    > = Sponge::default();
    let permuter = sponge.permuter;
    permuter.permute(&mut sponge.state, 0);

    let customization_bits = match u64::try_from(customization.len()) {
      Ok(value) => value.strict_mul(8),
      Err(_) => panic!("customization length exceeds u64"),
    };
    sponge.absorb_block(&customization_bits.to_le_bytes());

    let (blocks, rest) = customization.as_chunks::<RATE>();
    for block in blocks {
      sponge.absorb_block(block);
    }

    let mut final_block = [0u8; RATE];
    final_block[..rest.len()].copy_from_slice(rest);
    final_block[rest.len()] = 0x01;
    sponge.absorb_block(&final_block);

    Ok(Self {
      initial_sponge: sponge.clone(),
      sponge,
    })
  }

  /// Compute a one-shot Ascon-CXOF128 reader.
  pub fn xof(customization: &[u8], data: &[u8]) -> Result<AsconCxof128Reader, AsconCxofCustomizationError> {
    let mut hasher = Self::new(customization)?;
    hasher.update(data);
    Ok(hasher.finalize_xof())
  }

  /// Absorb message data.
  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.sponge.update(data);
  }

  /// Reset back to the post-customization initial state.
  #[inline]
  pub fn reset(&mut self) {
    self.sponge = self.initial_sponge.clone();
  }

  /// Finalize into an extendable-output reader.
  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> AsconCxof128Reader {
    let (state, hint) = self.sponge.finalize_state_with_hint();
    AsconCxof128Reader {
      state,
      buf: [0u8; RATE],
      pos: RATE,
      hint,
      bytes_out: 0,
    }
  }

  /// Fill `out` in one shot.
  pub fn hash_into(customization: &[u8], data: &[u8], out: &mut [u8]) -> Result<(), AsconCxofCustomizationError> {
    let mut reader = Self::xof(customization, data)?;
    reader.squeeze(out);
    Ok(())
  }
}

/// Ascon-CXOF128 output reader.
pub type AsconCxof128Reader = AsconXofReader;
