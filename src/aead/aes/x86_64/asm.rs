//! Linux x86-64 rscrypto-owned AES-GCM assembly kernels.

use core::arch::global_asm;

global_asm!(include_str!("asm/rscrypto_aes_gcm_x86_64_linux.s"));

#[repr(C)]
pub(super) struct AesGcmX86State {
  acc_lo: u64,
  acc_hi: u64,
  pub(super) ctr: u32,
  _pad: u32,
  pub(super) processed: usize,
}

impl AesGcmX86State {
  #[inline]
  pub(super) fn new(acc: u128, ctr: u32) -> Self {
    Self {
      acc_lo: u64::try_from(acc & u128::from(u64::MAX)).expect("masked accumulator half fits u64"),
      acc_hi: u64::try_from(acc >> 64).expect("shifted accumulator half fits u64"),
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
  pub(super) fn rscrypto_aes128_gcm_seal_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes128_gcm_open_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes128_gcm_seal_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes128_gcm_open_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes128_gcm_seal_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes128_gcm_open_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes256_gcm_seal_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes256_gcm_open_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_32: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes256_gcm_seal_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes256_gcm_open_64x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_64: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes256_gcm_seal_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes256_gcm_open_128x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
    h_powers_rev_128: *const u128,
    state: *mut AesGcmX86State,
  );

  pub(super) fn rscrypto_aes128_gcmsiv_ctr_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
  ) -> usize;

  pub(super) fn rscrypto_aes256_gcmsiv_ctr_16x_vaes512_x86_64_linux(
    round_keys: *const u8,
    initial_counter: *const u8,
    data: *mut u8,
    len: usize,
  ) -> usize;
}
