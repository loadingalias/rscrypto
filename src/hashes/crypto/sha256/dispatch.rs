use super::{
  BLOCK_LEN,
  kernels::{Sha256KernelId, required_caps},
};

define_sha_family_dispatch! {
  kernel_id: Sha256KernelId,
  compress_fn_ty: Sha256KernelId,
  portable_kernel: Sha256KernelId::Portable,
  compress_fn: core::convert::identity,
  required_caps: required_caps,
  runtime_kernel: super::dispatch_policy::select_runtime_kernel,
  compile_time: {
    hw: super::kernels::COMPILE_TIME_HW,
    name: super::kernels::COMPILE_TIME_NAME,
    best: super::kernels::compile_time_best(),
  },
}

define_sha_digest! {
  compress_fn_ty: Sha256KernelId,
  output_len: 32,
  word: u32,
  total_bits_ty: u64,
  length_offset: 56,
  h0: super::H0,
}
