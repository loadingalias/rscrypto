use super::{
  BLOCK_LEN,
  kernels::{Sha512KernelId, required_caps},
};

define_sha_family_dispatch! {
  kernel_id: Sha512KernelId,
  compress_fn_ty: Sha512KernelId,
  portable_kernel: Sha512KernelId::Portable,
  compress_fn: core::convert::identity,
  required_caps: required_caps,
  runtime_kernel: super::dispatch_policy::select_runtime_kernel,
  compile_time: {
    hw: false,
    name: "portable",
    best: Sha512KernelId::Portable,
  },
}

define_sha_digest! {
  compress_fn_ty: Sha512KernelId,
  output_len: 64,
  word: u64,
  total_bits_ty: u128,
  length_offset: 112,
  h0: super::H0,
}
