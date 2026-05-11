//! Linux x86-64 external AES-GCM kernels.

#![allow(unsafe_code)]

use core::arch::global_asm;

global_asm!(include_str!("asm/rscrypto_aes_gcm_x86_64_linux.s"));

#[repr(C)]
#[allow(dead_code)]
pub(super) struct AesGcmX86State {
  acc_lo: u64,
  acc_hi: u64,
  pub(super) ctr: u32,
  _pad: u32,
  pub(super) processed: usize,
}

#[allow(dead_code)]
impl AesGcmX86State {
  #[inline]
  pub(super) fn new(acc: u128, ctr: u32) -> Self {
    Self {
      acc_lo: acc as u64,
      acc_hi: (acc >> 64) as u64,
      ctr,
      _pad: 0,
      processed: 0,
    }
  }

  #[inline]
  pub(super) fn acc(&self) -> u128 {
    (self.acc_lo as u128) | ((self.acc_hi as u128) << 64)
  }
}

unsafe extern "C" {
  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_seal_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_open_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_seal_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_open_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_seal_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_open_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_seal_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_open_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_seal_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_open_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_seal_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_open_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_seal_8x_vaes256_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcm_open_8x_vaes256_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_seal_8x_vaes256_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcm_open_8x_vaes256_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmX86State,
  );

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes128_gcmsiv_ctr_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
  ) -> usize;

  #[allow(dead_code)]
  pub(super) fn rscrypto_aes256_gcmsiv_ctr_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
  ) -> usize;
}
