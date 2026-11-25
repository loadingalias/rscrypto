#![cfg(feature = "sha3")]

use proptest::prelude::*;
use rscrypto::{hashes::crypto::Shake256 as Shake, traits::Xof as _};

fn oracle_shake() -> tiny_keccak::Shake {
  tiny_keccak::Shake::v256()
}

const RATE: usize = 136;

include!("common/shake_differential.rs");
