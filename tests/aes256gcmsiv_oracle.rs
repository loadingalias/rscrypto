//! Integration oracle tests for AES-256-GCM-SIV.
//!
//! Validates rscrypto's AES-256-GCM-SIV against the RustCrypto `aes-gcm-siv`
//! crate across multiple input sizes and boundary conditions.
//!
//! # Coverage
//!
//! 1. Encrypt/decrypt round-trip matches RustCrypto oracle
//! 2. Empty plaintext, empty AAD, both empty
//! 3. Block boundary sizes (16-byte AES blocks)
//! 4. Large multi-block inputs
//! 5. Tag forgery / ciphertext / AAD tampering rejection

#![cfg(feature = "aead")]

use aes_gcm_siv::{Aes256GcmSiv as Oracle, KeyInit, Nonce as OracleNonce, aead::AeadInOut};
use rscrypto::{
  Aes256GcmSiv as Cipher, Aes256GcmSivKey as Key, Aes256GcmSivTag as Tag,
  aead::{Nonce96, expert::AeadWithNonce},
};

const KEY_BYTES: usize = 32;
const ORACLE_AAD: &[u8] = b"rscrypto-aes-gcm-siv-oracle";

include!("common/aes_gcmsiv_oracle.rs");
