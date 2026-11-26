use super::{BLOCK_LEN, kernels::Sha512KernelId};
pub(crate) use crate::hashes::crypto::sha512::dispatch::compress_dispatch;
#[cfg(feature = "diag")]
pub(crate) use crate::hashes::crypto::sha512::dispatch::kernel_name_for_len;

define_sha_digest! {
  compress_fn_ty: Sha512KernelId,
  output_len: 48,
  word: u64,
  total_bits_ty: u128,
  length_offset: 112,
  h0: super::H0,
}
