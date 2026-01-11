//! XXH3 (**NOT CRYPTO**).
//!
//! Staged implementation: API surface first, portable kernel next.

use traits::FastHash;

#[derive(Clone, Default)]
pub struct Xxh3_64;

#[derive(Clone, Default)]
pub struct Xxh3_128;

impl FastHash for Xxh3_64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = u64;

  fn hash_with_seed(_seed: Self::Seed, _data: &[u8]) -> Self::Output {
    todo!("XXH3 implementation staged-in next");
  }
}

impl FastHash for Xxh3_128 {
  const OUTPUT_SIZE: usize = 16;
  type Output = u128;
  type Seed = u64;

  fn hash_with_seed(_seed: Self::Seed, _data: &[u8]) -> Self::Output {
    todo!("XXH3 implementation staged-in next");
  }
}
