use super::{
  BLOCK_LEN,
  kernels::{CompressBlocksFn, Sha512KernelId, compress_blocks_fn, required_caps},
};

define_sha_family_dispatch! {
  kernel_id: Sha512KernelId,
  compress_fn_ty: CompressBlocksFn,
  portable_kernel: Sha512KernelId::Portable,
  compress_fn: compress_blocks_fn,
  required_caps: required_caps,
  runtime_table: super::dispatch_tables::select_runtime_table,
  output_len: 64,
  word_bytes: 8,
  total_bits_ty: u128,
  length_offset: 112,
  h0: super::H0,
  compile_time: {
    hw: false,
    name: "portable",
    best: compress_blocks_fn(Sha512KernelId::Portable),
  },
}
