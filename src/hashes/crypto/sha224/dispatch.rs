use super::{BLOCK_LEN, kernels::CompressBlocksFn};
pub(crate) use crate::hashes::crypto::sha256::dispatch::compress_dispatch;
#[cfg(feature = "diag")]
pub(crate) use crate::hashes::crypto::sha256::dispatch::kernel_name_for_len;

define_sha_digest! {
  compress_fn_ty: CompressBlocksFn,
  output_len: 28,
  word: u32,
  total_bits_ty: u64,
  length_offset: 56,
  h0: super::H0,
}
