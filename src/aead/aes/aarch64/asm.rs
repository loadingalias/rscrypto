//! AArch64 external AES-GCM kernels.

#![allow(unsafe_code)]

use core::arch::global_asm;

global_asm!(include_str!("asm/rscrypto_aes_gcm_aarch64_apple_darwin.s"));

#[repr(C)]
pub(super) struct AesGcmAarch64State {
  acc_lo: u64,
  acc_hi: u64,
  pub(super) ctr: u32,
  _pad: u32,
  pub(super) processed: usize,
}

impl AesGcmAarch64State {
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
  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes128_gcm_seal_8x_aarch64")]
  pub(super) fn rscrypto_aes128_gcm_seal_8x_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmAarch64State,
  );

  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes128_gcm_open_8x_aarch64")]
  pub(super) fn rscrypto_aes128_gcm_open_8x_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmAarch64State,
  );

  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes128_gcm_seal_16x_eor3_aarch64")]
  pub(super) fn rscrypto_aes128_gcm_seal_16x_eor3_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_16: *const u128,
    h_powers_rev_16_mid: *const u128,
    h_powers_rev_16_pair: *const u128,
    state: *mut AesGcmAarch64State,
  );

  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes128_gcm_open_16x_eor3_aarch64")]
  pub(super) fn rscrypto_aes128_gcm_open_16x_eor3_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_16: *const u128,
    h_powers_rev_16_mid: *const u128,
    h_powers_rev_16_pair: *const u128,
    state: *mut AesGcmAarch64State,
  );

  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes256_gcm_seal_8x_aarch64")]
  pub(super) fn rscrypto_aes256_gcm_seal_8x_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmAarch64State,
  );

  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes256_gcm_open_8x_aarch64")]
  pub(super) fn rscrypto_aes256_gcm_open_8x_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_8: *const u128,
    state: *mut AesGcmAarch64State,
  );

  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes256_gcm_seal_16x_eor3_aarch64")]
  pub(super) fn rscrypto_aes256_gcm_seal_16x_eor3_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_16: *const u128,
    h_powers_rev_16_mid: *const u128,
    h_powers_rev_16_pair: *const u128,
    state: *mut AesGcmAarch64State,
  );

  #[cfg_attr(target_os = "linux", link_name = "_rscrypto_aes256_gcm_open_16x_eor3_aarch64")]
  pub(super) fn rscrypto_aes256_gcm_open_16x_eor3_aarch64(
    round_keys: *const u8,
    iv_prefix: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_16: *const u128,
    h_powers_rev_16_mid: *const u128,
    h_powers_rev_16_pair: *const u128,
    state: *mut AesGcmAarch64State,
  );
}
