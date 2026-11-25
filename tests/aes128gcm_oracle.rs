//! Integration oracle tests for AES-128-GCM.
//!
//! Validates rscrypto's AES-128-GCM against the RustCrypto `aes-gcm` crate
//! across multiple input sizes, AAD patterns, and boundary conditions.
//!
//! # Coverage
//!
//! 1. Encrypt/decrypt round-trip matches RustCrypto oracle
//! 2. Empty plaintext with AAD
//! 3. Empty AAD with plaintext
//! 4. Large multi-block inputs
//! 5. Tag forgery rejection

#![cfg(feature = "aead")]

use aes_gcm::{
  Aes128Gcm as Oracle,
  aead::{AeadInOut, KeyInit, array::Array},
};
use rscrypto::{
  Aes128Gcm as Cipher, Aes128GcmKey as Key, Aes128GcmTag as Tag,
  aead::{Nonce96, expert::AeadWithNonce},
};

const KEY_BYTES: usize = 16;
const ORACLE_AAD: &[u8] = b"rscrypto-aes128-gcm-oracle";
const SHORT_NONCE: u8 = 0x18;
const SHORT_AAD: u8 = 0x42;
const SHORT_MESSAGE: u8 = 0x81;
const LARGE_KEY: u8 = 0x5D;
const LARGE_AAD: u8 = 0xAC;
const LARGE_NONCE: u8 = 0x71;
const LARGE_MESSAGE: u8 = 0x33;

include!("common/aes_gcm_oracle.rs");
