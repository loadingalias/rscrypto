//! Integration oracle tests for AES-128-GCM-SIV.
//!
//! Validates rscrypto's AES-128-GCM-SIV against the RustCrypto `aes-gcm-siv`
//! crate across multiple input sizes and boundary conditions.
//!
//! # Coverage
//!
//! 1. Encrypt/decrypt round-trip matches RustCrypto oracle
//! 2. Empty plaintext, empty AAD, both empty
//! 3. Block boundary sizes (16-byte AES blocks)
//! 4. Large multi-block inputs
//! 5. AAD-size sweep across the 4-block POLYVAL wide boundary
//! 6. Tag forgery / ciphertext / AAD tampering rejection

#![cfg(feature = "aead")]

use aes_gcm_siv::{Aes128GcmSiv as Oracle, KeyInit, Nonce as OracleNonce, aead::AeadInOut};
use rscrypto::{
  Aes128GcmSiv as Cipher, Aes128GcmSivKey as Key, Aes128GcmSivTag as Tag,
  aead::{Nonce96, expert::AeadWithNonce},
};

const KEY_BYTES: usize = 16;
const ORACLE_AAD: &[u8] = b"rscrypto-aes128-gcm-siv-oracle";

include!("common/aes_gcmsiv_oracle.rs");
