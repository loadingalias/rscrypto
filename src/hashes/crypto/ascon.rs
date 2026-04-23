//! Ascon hash and XOF (NIST LWC).
//!
//! Portable, `no_std`, pure Rust implementation.

#![allow(clippy::indexing_slicing)] // Fixed-size state + sponge buffering

use core::fmt;

use crate::{
  backend::ascon::permute_12_portable,
  traits::{Digest, Xof},
};

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[doc(hidden)]
pub(crate) mod dispatch;
#[doc(hidden)]
pub(crate) mod dispatch_tables;
#[cfg(test)]
mod kernel_test;
pub(crate) mod kernels;
#[cfg(target_arch = "x86_64")]
mod x86_64_avx2;
#[cfg(target_arch = "x86_64")]
mod x86_64_avx512;

const RATE: usize = 8;

trait Permuter: Copy {
  fn permute(self, state: &mut [u64; 5], len_hint: usize);
}

/// Direct-call permuter using the portable scalar kernel.
///
/// No function pointer indirection — LLVM can inline `permute_12_portable`
/// into the absorb/squeeze loops, enabling register allocation across the
/// entire sponge operation and instruction scheduling across round boundaries.
#[derive(Clone, Copy, Default)]
struct InlinePermuter;

impl Permuter for InlinePermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 5], _len_hint: usize) {
    permute_12_portable(state);
  }
}

#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct DispatchPermuter {
  dispatch: crate::hashes::crypto::dispatch_util::SizeClassDispatch<fn(&mut [u64; 5])>,
}

#[cfg(any(test, feature = "std"))]
impl Default for DispatchPermuter {
  #[inline]
  fn default() -> Self {
    Self {
      dispatch: dispatch::permute_dispatch(),
    }
  }
}

#[cfg(any(test, feature = "std"))]
impl Permuter for DispatchPermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 5], len_hint: usize) {
    (self.dispatch.select(len_hint))(state);
  }
}

// Ascon permutation round constants (12 rounds).
// Used by SIMD kernels; the shared portable permutation inlines the constants.
#[allow(dead_code)]
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

define_unit_error! {
  /// Ascon-CXOF128 customization strings are limited to 256 bytes by SP 800-232.
  pub struct AsconCxofCustomizationError;
  "Ascon-CXOF128 customization exceeds 256 bytes"
}

#[inline(always)]
const fn pad(n: usize) -> u64 {
  // Produce the padding mask used by the reference construction:
  // XOR `pad(len)` into state[0], with state interpreted little-endian.
  0x01_u64 << (8 * n)
}

#[derive(Clone)]
struct Sponge<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64> {
  state: [u64; 5],
  buf: [u8; RATE],
  buf_len: usize,
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
      permuter: P::default(),
    }
  }
}

impl<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64>
  Sponge<P, IV0, IV1, IV2, IV3, IV4>
{
  #[inline(always)]
  fn absorb_block(&mut self, block: &[u8; RATE]) {
    self.state[0] ^= u64::from_le_bytes(*block);
    self.permuter.permute(&mut self.state, 0);
  }

  #[inline(always)]
  fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

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

  #[inline(always)]
  fn finalize_state(&self) -> [u64; 5] {
    let mut st = self.state;
    let last = &self.buf[..self.buf_len];

    let mut tmp = [0u8; RATE];
    tmp[..last.len()].copy_from_slice(last);
    st[0] ^= u64::from_le_bytes(tmp);
    st[0] ^= pad(last.len());
    self.permuter.permute(&mut st, 0);

    st
  }
}

#[inline(always)]
fn finalize_state_one_shot(mut state: [u64; 5], data: &[u8], mut permute: impl FnMut(&mut [u64; 5])) -> [u64; 5] {
  let (blocks, tail) = data.as_chunks::<RATE>();
  for block in blocks {
    state[0] ^= u64::from_le_bytes(*block);
    permute(&mut state);
  }

  let mut final_block = [0u8; RATE];
  final_block[..tail.len()].copy_from_slice(tail);
  state[0] ^= u64::from_le_bytes(final_block);
  state[0] ^= pad(tail.len());
  permute(&mut state);
  state
}

#[inline(always)]
fn squeeze_hash256(mut state: [u64; 5], mut permute: impl FnMut(&mut [u64; 5])) -> [u8; 32] {
  let mut out = [0u8; 32];
  let mut off = 0usize;
  while off < 24 {
    out[off..off + 8].copy_from_slice(&state[0].to_le_bytes());
    permute(&mut state);
    off += 8;
  }
  out[24..32].copy_from_slice(&state[0].to_le_bytes());
  out
}

#[inline(always)]
fn squeeze_xof_into(mut state: [u64; 5], out: &mut [u8], mut permute: impl FnMut(&mut [u64; 5])) {
  let mut buf = state[0].to_le_bytes();
  let (blocks, tail) = out.as_chunks_mut::<RATE>();
  let block_count = blocks.len();
  for (index, block) in blocks.iter_mut().enumerate() {
    block.copy_from_slice(&buf);
    if index + 1 < block_count || !tail.is_empty() {
      permute(&mut state);
      buf = state[0].to_le_bytes();
    }
  }
  if !tail.is_empty() {
    tail.copy_from_slice(&buf[..tail.len()]);
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
    InlinePermuter,
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
  fn batch_kernel_id_for_count(count: usize) -> kernels::AsconPermute12KernelId {
    dispatch::batch_kernel_id_for_count(count)
  }

  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_with_kernel(kid: kernels::AsconPermute12KernelId, data: &[u8]) -> [u8; 32] {
    let permute = kernels::permute_fn(kid);
    let state = finalize_state_one_shot(HASH256_IV, data, permute);
    squeeze_hash256(state, permute)
  }

  /// Multi-message one-shot hashing using an explicitly selected kernel.
  ///
  /// `outputs.len()` must equal `inputs.len()`.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_many_with_kernel(
    mut kid: kernels::AsconPermute12KernelId,
    inputs: &[&[u8]],
    outputs: &mut [[u8; 32]],
  ) {
    assert_eq!(inputs.len(), outputs.len(), "input/output batch length mismatch");
    if inputs.is_empty() {
      return;
    }

    kid = dispatch::batch_fallback_kernel_id(kid, inputs.len());
    let degree = kernels::simd_degree(kid);
    if degree == 1 || !inputs_have_equal_len(inputs) {
      let scalar_kid = dispatch::scalar_kernel_id();
      for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
        *output = Self::digest_with_kernel(scalar_kid, input);
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
        let rem_inputs = input_groups.remainder();
        let rem_outputs = output_groups.into_remainder();
        if !rem_inputs.is_empty() {
          Self::digest_many_with_kernel(
            dispatch::batch_fallback_kernel_id(kid, rem_inputs.len()),
            rem_inputs,
            rem_outputs,
          );
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
        let rem_inputs = input_groups.remainder();
        let rem_outputs = output_groups.into_remainder();
        if !rem_inputs.is_empty() {
          Self::digest_many_with_kernel(
            dispatch::batch_fallback_kernel_id(kid, rem_inputs.len()),
            rem_inputs,
            rem_outputs,
          );
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
        let rem_inputs = input_groups.remainder();
        let rem_outputs = output_groups.into_remainder();
        if !rem_inputs.is_empty() {
          Self::digest_many_with_kernel(
            dispatch::batch_fallback_kernel_id(kid, rem_inputs.len()),
            rem_inputs,
            rem_outputs,
          );
        }
      }
    }
  }

  /// Hashes many messages into adjacent fixed-size outputs.
  ///
  /// This is the batched companion to [`Digest::digest`].
  /// Equal-length inputs let the implementation use the widest available
  /// permutation backend; mixed lengths automatically fall back to per-message
  /// hashing.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  pub fn digest_many(inputs: &[&[u8]], outputs: &mut [[u8; 32]]) {
    assert_eq!(inputs.len(), outputs.len(), "input/output batch length mismatch");

    let mut start = 0usize;
    while start < inputs.len() {
      let len = inputs[start].len();
      let mut end = start + 1;
      while end < inputs.len() && inputs[end].len() == len {
        end += 1;
      }

      let kid = Self::batch_kernel_id_for_count(end - start);
      Self::digest_many_with_kernel(kid, &inputs[start..end], &mut outputs[start..end]);
      start = end;
    }
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

  #[inline(always)]
  fn finalize(&self) -> Self::Output {
    squeeze_hash256(self.sponge.finalize_state(), permute_12_portable)
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
    Sponge<InlinePermuter, { XOF128_IV[0] }, { XOF128_IV[1] }, { XOF128_IV[2] }, { XOF128_IV[3] }, { XOF128_IV[4] }>,
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
    let state = finalize_state_one_shot(XOF128_IV, data, permute_12_portable);
    AsconXofReader::from_state(state)
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
    AsconXofReader::from_state(self.sponge.finalize_state())
  }

  #[inline]
  /// Convenience one-shot XOF output for callers that only need bytes.
  ///
  /// The canonical one-shot API is [`Self::xof`]. Use [`Self::new`],
  /// [`Self::update`], and [`Self::finalize_xof`] for streaming.
  pub fn hash_into(data: &[u8], out: &mut [u8]) {
    let state = finalize_state_one_shot(XOF128_IV, data, permute_12_portable);
    squeeze_xof_into(state, out, permute_12_portable);
  }

  #[inline]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn hash_into_with_kernel(kid: kernels::AsconPermute12KernelId, data: &[u8], out: &mut [u8]) {
    let permute = kernels::permute_fn(kid);
    let state = finalize_state_one_shot(XOF128_IV, data, permute);
    squeeze_xof_into(state, out, permute);
  }

  /// Multi-message XOF using an explicitly selected kernel.
  ///
  /// `outputs` is a flat buffer laid out as `inputs.len()` adjacent outputs of
  /// `out_len` bytes each.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn hash_many_into_with_kernel(
    mut kid: kernels::AsconPermute12KernelId,
    inputs: &[&[u8]],
    out_len: usize,
    outputs: &mut [u8],
  ) {
    assert_eq!(
      outputs.len(),
      inputs.len() * out_len,
      "input/output batch length mismatch"
    );
    if inputs.is_empty() {
      return;
    }

    kid = dispatch::batch_fallback_kernel_id(kid, inputs.len());
    let degree = kernels::simd_degree(kid);
    if degree == 1 || !inputs_have_equal_len(inputs) {
      let scalar_kid = dispatch::scalar_kernel_id();
      for (index, input) in inputs.iter().enumerate() {
        let base = index * out_len;
        Self::hash_into_with_kernel(scalar_kid, input, &mut outputs[base..base + out_len]);
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
        let rem_inputs = input_groups.remainder();
        let rem_outputs = output_groups.into_remainder();
        if !rem_inputs.is_empty() {
          Self::hash_many_into_with_kernel(
            dispatch::batch_fallback_kernel_id(kid, rem_inputs.len()),
            rem_inputs,
            out_len,
            rem_outputs,
          );
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
        let rem_inputs = input_groups.remainder();
        let rem_outputs = output_groups.into_remainder();
        if !rem_inputs.is_empty() {
          Self::hash_many_into_with_kernel(
            dispatch::batch_fallback_kernel_id(kid, rem_inputs.len()),
            rem_inputs,
            out_len,
            rem_outputs,
          );
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
        let rem_inputs = input_groups.remainder();
        let rem_outputs = output_groups.into_remainder();
        if !rem_inputs.is_empty() {
          Self::hash_many_into_with_kernel(
            dispatch::batch_fallback_kernel_id(kid, rem_inputs.len()),
            rem_inputs,
            out_len,
            rem_outputs,
          );
        }
      }
    }
  }

  /// Squeezes equal-sized XOF outputs for many messages into a flat output buffer.
  ///
  /// `outputs` must be exactly `inputs.len() * out_len` bytes long. Equal-length
  /// inputs allow batched permutation backends; mixed lengths automatically fall
  /// back to the scalar per-message path.
  #[inline]
  #[cfg(any(test, feature = "std"))]
  pub fn hash_many_into(inputs: &[&[u8]], out_len: usize, outputs: &mut [u8]) {
    assert_eq!(
      outputs.len(),
      inputs.len() * out_len,
      "input/output batch length mismatch"
    );

    let mut start = 0usize;
    while start < inputs.len() {
      let len = inputs[start].len();
      let mut end = start + 1;
      while end < inputs.len() && inputs[end].len() == len {
        end += 1;
      }

      let kid = AsconHash256::batch_kernel_id_for_count(end - start);
      let start_byte = start * out_len;
      let end_byte = end * out_len;
      Self::hash_many_into_with_kernel(kid, &inputs[start..end], out_len, &mut outputs[start_byte..end_byte]);
      start = end;
    }
  }
}

/// Ascon-XOF128 reader.
#[derive(Clone)]
pub struct AsconXofReader {
  state: [u64; 5],
  buf: [u8; RATE],
  pos: usize,
}

impl fmt::Debug for AsconXofReader {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("AsconXofReader").finish_non_exhaustive()
  }
}

impl AsconXofReader {
  #[inline(always)]
  fn from_state(state: [u64; 5]) -> Self {
    Self {
      buf: state[0].to_le_bytes(),
      state,
      pos: 0,
    }
  }

  #[inline(always)]
  fn advance(&mut self) {
    permute_12_portable(&mut self.state);
    self.buf = self.state[0].to_le_bytes();
    self.pos = 0;
  }
}

impl Xof for AsconXofReader {
  #[inline(always)]
  fn squeeze(&mut self, mut out: &mut [u8]) {
    if self.pos < RATE && !out.is_empty() {
      let take = core::cmp::min(RATE - self.pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.pos..self.pos.strict_add(take)]);
      self.pos = self.pos.strict_add(take);
      out = &mut out[take..];
    }

    let (blocks, tail) = out.as_chunks_mut::<RATE>();
    for block in blocks {
      if self.pos == RATE {
        self.advance();
      }
      block.copy_from_slice(&self.buf);
      self.pos = RATE;
    }

    if !tail.is_empty() {
      if self.pos == RATE {
        self.advance();
      }
      tail.copy_from_slice(&self.buf[..tail.len()]);
      self.pos = tail.len();
    }
  }
}

impl_xof_read!(AsconXofReader);

/// Spec-precise alias for [`AsconXof`].
pub type AsconXof128 = AsconXof;

/// Spec-precise alias for [`AsconXofReader`].
pub type AsconXof128Reader = AsconXofReader;

/// Ascon-CXOF128 state with explicit customization.
///
/// Standardized in NIST SP 800-232.
///
/// # Examples
///
/// ```
/// use rscrypto::{AsconCxof128, Xof};
///
/// let mut reader = AsconCxof128::xof(b"domain", b"abc")?;
/// let mut out = [0u8; 32];
/// reader.squeeze(&mut out);
///
/// assert_ne!(out, [0u8; 32]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// [`AsconCxof128::new`], [`AsconCxof128::xof`], and [`AsconCxof128::hash_into`]
/// return [`AsconCxofCustomizationError`] when `customization` exceeds
/// [`AsconCxof128::MAX_CUSTOMIZATION_LEN`].
#[derive(Clone)]
pub struct AsconCxof128 {
  sponge: Sponge<
    InlinePermuter,
    { CXOF128_IV[0] },
    { CXOF128_IV[1] },
    { CXOF128_IV[2] },
    { CXOF128_IV[3] },
    { CXOF128_IV[4] },
  >,
  initial_sponge: Sponge<
    InlinePermuter,
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
  ///
  /// # Errors
  ///
  /// Returns [`AsconCxofCustomizationError`] when `customization` exceeds
  /// [`MAX_CUSTOMIZATION_LEN`](Self::MAX_CUSTOMIZATION_LEN).
  pub fn new(customization: &[u8]) -> Result<Self, AsconCxofCustomizationError> {
    if customization.len() > Self::MAX_CUSTOMIZATION_LEN {
      return Err(AsconCxofCustomizationError::new());
    }

    let mut sponge: Sponge<
      InlinePermuter,
      { CXOF128_IV[0] },
      { CXOF128_IV[1] },
      { CXOF128_IV[2] },
      { CXOF128_IV[3] },
      { CXOF128_IV[4] },
    > = Sponge::default();
    permute_12_portable(&mut sponge.state);

    let customization_bits = crate::bytes_to_bits_saturating(customization.len());
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
  ///
  /// # Errors
  ///
  /// Returns [`AsconCxofCustomizationError`] when `customization` exceeds
  /// [`MAX_CUSTOMIZATION_LEN`](Self::MAX_CUSTOMIZATION_LEN).
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
    AsconXofReader::from_state(self.sponge.finalize_state())
  }

  /// Fill `out` in one shot.
  ///
  /// # Errors
  ///
  /// Returns [`AsconCxofCustomizationError`] when `customization` exceeds
  /// [`MAX_CUSTOMIZATION_LEN`](Self::MAX_CUSTOMIZATION_LEN).
  pub fn hash_into(customization: &[u8], data: &[u8], out: &mut [u8]) -> Result<(), AsconCxofCustomizationError> {
    let mut reader = Self::xof(customization, data)?;
    reader.squeeze(out);
    Ok(())
  }
}

/// Ascon-CXOF128 output reader.
pub type AsconCxof128Reader = AsconXofReader;
