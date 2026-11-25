//! Integration oracle tests for AES-256-GCM.
//!
//! Validates rscrypto's AES-256-GCM against the RustCrypto `aes-gcm` crate
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
  Aes256Gcm as Oracle,
  aead::{AeadInOut, KeyInit, array::Array},
};
use rscrypto::{
  Aes256Gcm as Cipher, Aes256GcmKey as Key, Aes256GcmTag as Tag,
  aead::{Nonce96, expert::AeadWithNonce},
};

const KEY_BYTES: usize = 32;
const ORACLE_AAD: &[u8] = b"rscrypto-aes-gcm-oracle";
const SHORT_NONCE: u8 = 0x28;
const SHORT_AAD: u8 = 0x52;
const SHORT_MESSAGE: u8 = 0x91;
const LARGE_KEY: u8 = 0x6D;
const LARGE_AAD: u8 = 0xBC;
const LARGE_NONCE: u8 = 0x81;
const LARGE_MESSAGE: u8 = 0x43;

include!("common/aes_gcm_oracle.rs");
