#[cfg(test)]
pub(crate) use crate::hashes::crypto::sha512::kernels::{ALL, required_caps};
pub(crate) use crate::hashes::crypto::sha512::kernels::{
  CompressBlocksFn, Sha512KernelId as Sha384KernelId, compress_blocks_fn,
};
