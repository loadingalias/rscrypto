// SHA-224 uses SHA-256's compression function (FIPS 180-4 SS6.3).
// Only H0 and output truncation differ.
#[cfg(test)]
pub(crate) use crate::hashes::crypto::sha256::kernels::ALL;
pub(crate) use crate::hashes::crypto::sha256::kernels::{
  CompressBlocksFn, Sha256KernelId as Sha224KernelId, compress_blocks_fn, required_caps,
};
