//! AArch64 rscrypto-owned AES-GCM assembly kernels.

use core::arch::global_asm;

global_asm!(include_str!("asm/rscrypto_aes_gcm_aarch64_apple_darwin.s"));

#[repr(C)]
pub(super) struct AesGcmAarch64State {
  acc_lo: u64,
  acc_hi: u64,
  pub(super) ctr: u32,
  _pad: u32,
  pub(super) processed: usize,
  input: *const u8,
  input_spans: *const *const u8,
}

const _: () = {
  assert!(core::mem::offset_of!(AesGcmAarch64State, acc_lo) == 0);
  assert!(core::mem::offset_of!(AesGcmAarch64State, ctr) == 16);
  assert!(core::mem::offset_of!(AesGcmAarch64State, processed) == 24);
  assert!(core::mem::offset_of!(AesGcmAarch64State, input) == 32);
  assert!(core::mem::offset_of!(AesGcmAarch64State, input_spans) == 40);
};

impl AesGcmAarch64State {
  #[inline]
  pub(super) fn new(acc: u128, ctr: u32, input: *const u8) -> Self {
    Self {
      acc_lo: u64::try_from(acc & u128::from(u64::MAX)).expect("masked accumulator half fits u64"),
      acc_hi: u64::try_from(acc >> 64).expect("shifted accumulator half fits u64"),
      ctr,
      _pad: 0,
      processed: 0,
      input,
      input_spans: core::ptr::null(),
    }
  }

  #[inline]
  pub(super) fn new_gather(acc: u128, ctr: u32, input: *const u8, input_spans: *const *const u8) -> Self {
    let mut state = Self::new(acc, ctr, input);
    state.input_spans = input_spans;
    state
  }

  #[inline]
  pub(super) fn acc(&self) -> u128 {
    (self.acc_lo as u128) | ((self.acc_hi as u128) << 64)
  }
}

impl Drop for AesGcmAarch64State {
  fn drop(&mut self) {
    crate::traits::ct::zeroize_words_no_fence(core::slice::from_mut(&mut self.acc_lo));
    crate::traits::ct::zeroize_words(core::slice::from_mut(&mut self.acc_hi));
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
