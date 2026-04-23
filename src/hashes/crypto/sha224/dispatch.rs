use super::{
  BLOCK_LEN,
  kernels::{CompressBlocksFn, Sha224KernelId, compress_blocks_fn, required_caps},
};

define_sha_family_dispatch! {
  kernel_id: Sha224KernelId,
  compress_fn_ty: CompressBlocksFn,
  portable_kernel: Sha224KernelId::Portable,
  compress_fn: compress_blocks_fn,
  required_caps: required_caps,
  runtime_table: super::dispatch_tables::select_runtime_table,
  output_len: 28,
  word_bytes: 4,
  total_bits_ty: u64,
  length_offset: 56,
  h0: super::H0,
  compile_time: {
    hw: crate::hashes::crypto::sha256::kernels::COMPILE_TIME_HW,
    name: crate::hashes::crypto::sha256::kernels::COMPILE_TIME_NAME,
    best: crate::hashes::crypto::sha256::kernels::compile_time_best(),
  },
}
