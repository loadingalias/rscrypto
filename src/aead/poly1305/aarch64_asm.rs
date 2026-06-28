//! rscrypto-owned aarch64 Poly1305 par4 assembly.
//!
//! This module is diagnostic-only while the owned assembly competes against the
//! current Rust NEON backend. It mirrors the existing four-block Poly1305 AEAD
//! updater: Rust handles padding/cached tails; assembly handles aligned par4
//! accumulation over public-length 64-byte groups.

#![allow(unsafe_code)]

use core::arch::global_asm;

use super::{LIMB_MASK, State};
use crate::{aead::AeadByteLengths, traits::ct};

#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_poly1305_par4_aarch64_apple_darwin.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_poly1305_par4_aarch64_linux.s"));

unsafe extern "C" {
  #[cfg(target_os = "macos")]
  fn rscrypto_poly1305_accumulate4_aarch64_apple_darwin(
    h: *mut u32,
    powers: *const Powers,
    input: *const u8,
    groups: usize,
  );

  #[cfg(target_os = "linux")]
  fn rscrypto_poly1305_accumulate4_aarch64_linux(h: *mut u32, powers: *const Powers, input: *const u8, groups: usize);
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Powers {
  r1: [u32; 5],
  r2: [u32; 5],
  r3: [u32; 5],
  r4: [u32; 5],
}

impl Powers {
  #[inline(always)]
  fn new(r1: [u32; 5]) -> Self {
    let r2 = mul_mod(r1, r1);
    let r3 = mul_mod(r2, r1);
    let r4 = mul_mod(r2, r2);
    Self { r1, r2, r3, r4 }
  }
}

pub(crate) struct AeadPar4Asm {
  state: State,
  powers: Powers,
  cached: [[u8; 16]; 4],
  num_cached: usize,
}

impl AeadPar4Asm {
  #[inline]
  pub(crate) fn new(key: &[u8; 32]) -> Self {
    let state = State::new(key);
    let powers = Powers::new(state.r);
    Self {
      state,
      powers,
      cached: [[0u8; 16]; 4],
      num_cached: 0,
    }
  }

  #[inline]
  pub(crate) fn update_padded_segment(&mut self, segment: &[u8]) {
    let mut offset = 0usize;
    while self.num_cached != 0 && offset.strict_add(16) <= segment.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&segment[offset..offset.strict_add(16)]);
      self.push(block);
      offset = offset.strict_add(16);
    }

    if self.num_cached == 0 {
      let group_len = segment.len().strict_sub(offset).strict_div(64).strict_mul(64);
      let group_end = offset.strict_add(group_len);
      self.update_groups(&segment[offset..group_end]);
      offset = group_end;
    }

    while offset.strict_add(16) <= segment.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&segment[offset..offset.strict_add(16)]);
      self.push(block);
      offset = offset.strict_add(16);
    }

    let rem = &segment[offset..];
    if !rem.is_empty() {
      let mut block = [0u8; 16];
      block[..rem.len()].copy_from_slice(rem);
      self.push(block);
    }
  }

  #[inline]
  pub(crate) fn finalize(mut self, lengths: AeadByteLengths) -> [u8; 16] {
    self.push(lengths.to_le_bytes_block());

    for block in self.cached.iter().take(self.num_cached) {
      super::aarch64_neon::compute_block(&mut self.state, block, false);
    }

    let tag = self.state.clone().finalize();
    ct::zeroize(self.cached.as_flattened_mut());
    tag
  }

  #[inline(always)]
  fn push(&mut self, block: [u8; 16]) {
    self.cached[self.num_cached] = block;
    self.num_cached = self.num_cached.strict_add(1);
    if self.num_cached == 4 {
      let cached = self.cached;
      self.update_groups(cached.as_flattened());
      self.num_cached = 0;
    }
  }

  #[inline]
  fn update_groups(&mut self, groups: &[u8]) {
    debug_assert_eq!(groups.len() % 64, 0);
    let group_count = groups.len() / 64;
    if group_count == 0 {
      return;
    }

    // SAFETY: aarch64 Poly1305 par4 accumulation because:
    // 1. This module is compiled only for aarch64 macOS/Linux, where Advanced SIMD is baseline.
    // 2. `self.state.h` is a fixed five-limb Poly1305 accumulator writable for the assembly ABI.
    // 3. `self.powers` is `#[repr(C)]` and stores r, r^2, r^3, r^4 as contiguous five-limb arrays.
    // 4. `groups` is checked to contain `group_count` complete public-length 64-byte groups.
    // 5. Branches and memory addresses in the assembly depend only on public group count.
    unsafe {
      #[cfg(target_os = "macos")]
      rscrypto_poly1305_accumulate4_aarch64_apple_darwin(
        self.state.h.as_mut_ptr(),
        &self.powers,
        groups.as_ptr(),
        group_count,
      );
      #[cfg(target_os = "linux")]
      rscrypto_poly1305_accumulate4_aarch64_linux(
        self.state.h.as_mut_ptr(),
        &self.powers,
        groups.as_ptr(),
        group_count,
      );
    }
  }
}

#[inline(always)]
fn mul_mod(a: [u32; 5], b: [u32; 5]) -> [u32; 5] {
  reduce_unreduced(mul_unreduced(a, b))
}

#[inline(always)]
fn mul_unreduced(a: [u32; 5], b: [u32; 5]) -> [u64; 5] {
  let b1_5 = b[1] * 5;
  let b2_5 = b[2] * 5;
  let b3_5 = b[3] * 5;
  let b4_5 = b[4] * 5;

  [
    (u64::from(a[0]) * u64::from(b[0]))
      + (u64::from(a[1]) * u64::from(b4_5))
      + (u64::from(a[2]) * u64::from(b3_5))
      + (u64::from(a[3]) * u64::from(b2_5))
      + (u64::from(a[4]) * u64::from(b1_5)),
    (u64::from(a[0]) * u64::from(b[1]))
      + (u64::from(a[1]) * u64::from(b[0]))
      + (u64::from(a[2]) * u64::from(b4_5))
      + (u64::from(a[3]) * u64::from(b3_5))
      + (u64::from(a[4]) * u64::from(b2_5)),
    (u64::from(a[0]) * u64::from(b[2]))
      + (u64::from(a[1]) * u64::from(b[1]))
      + (u64::from(a[2]) * u64::from(b[0]))
      + (u64::from(a[3]) * u64::from(b4_5))
      + (u64::from(a[4]) * u64::from(b3_5)),
    (u64::from(a[0]) * u64::from(b[3]))
      + (u64::from(a[1]) * u64::from(b[2]))
      + (u64::from(a[2]) * u64::from(b[1]))
      + (u64::from(a[3]) * u64::from(b[0]))
      + (u64::from(a[4]) * u64::from(b4_5)),
    (u64::from(a[0]) * u64::from(b[4]))
      + (u64::from(a[1]) * u64::from(b[3]))
      + (u64::from(a[2]) * u64::from(b[2]))
      + (u64::from(a[3]) * u64::from(b[1]))
      + (u64::from(a[4]) * u64::from(b[0])),
  ]
}

#[inline(always)]
fn reduce_unreduced(mut d: [u64; 5]) -> [u32; 5] {
  let mut c = d[0] >> 26;
  let mut h0 = d[0] & u64::from(LIMB_MASK);
  d[1] = d[1].wrapping_add(c);

  c = d[1] >> 26;
  let h1_base = d[1] & u64::from(LIMB_MASK);
  d[2] = d[2].wrapping_add(c);

  c = d[2] >> 26;
  let h2 = (d[2] as u32) & LIMB_MASK;
  d[3] = d[3].wrapping_add(c);

  c = d[3] >> 26;
  let h3 = (d[3] as u32) & LIMB_MASK;
  d[4] = d[4].wrapping_add(c);

  c = d[4] >> 26;
  let h4 = (d[4] as u32) & LIMB_MASK;
  h0 = h0.wrapping_add(c * 5);

  c = h0 >> 26;
  h0 &= u64::from(LIMB_MASK);
  let h1 = h1_base.wrapping_add(c);

  [h0 as u32, h1 as u32, h2, h3, h4]
}
