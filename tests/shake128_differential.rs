#![cfg(feature = "sha3")]

use proptest::prelude::*;
use rscrypto::{hashes::crypto::Shake128 as Shake, traits::Xof as _};

fn oracle_shake() -> tiny_keccak::Shake {
  tiny_keccak::Shake::v128()
}

const RATE: usize = 168;

include!("common/shake_differential.rs");
