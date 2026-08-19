#![cfg(all(feature = "blake2b", feature = "ed25519", feature = "x25519"))]

use dryoc::classic::{
  crypto_core::{crypto_scalarmult, crypto_scalarmult_base},
  crypto_generichash::crypto_generichash,
  crypto_sign::{crypto_sign_detached, crypto_sign_seed_keypair, crypto_sign_verify_detached},
};
use rscrypto::{Blake2b256, Blake2b512, Blake2bKey, Ed25519SecretKey, X25519SecretKey};

const DATA: &[u8] = b"dryoc migration equivalence data";
const KEY_32: [u8; 32] = [0x42; 32];
const KEY_64: [u8; 64] = [0x24; 64];

#[test]
fn test_dryoc_blake2b_migration_examples_are_byte_equivalent() {
  let mut dryoc_b256 = [0u8; 32];
  crypto_generichash(&mut dryoc_b256, DATA, None).expect("dryoc BLAKE2b-256 migration hash must succeed");
  assert_eq!(Blake2b256::digest(DATA), dryoc_b256);

  let mut dryoc_b512 = [0u8; 64];
  crypto_generichash(&mut dryoc_b512, DATA, None).expect("dryoc BLAKE2b-512 migration hash must succeed");
  assert_eq!(Blake2b512::digest(DATA), dryoc_b512);

  let mut dryoc_keyed_b256 = [0u8; 32];
  crypto_generichash(&mut dryoc_keyed_b256, DATA, Some(&KEY_32))
    .expect("dryoc keyed BLAKE2b-256 migration hash must succeed");
  assert_eq!(
    Blake2b256::keyed_digest(
      Blake2bKey::new(&KEY_32).expect("32-byte BLAKE2b migration key must be valid"),
      DATA,
    ),
    dryoc_keyed_b256
  );

  let mut dryoc_keyed_b512 = [0u8; 64];
  crypto_generichash(&mut dryoc_keyed_b512, DATA, Some(&KEY_64))
    .expect("dryoc keyed BLAKE2b-512 migration hash must succeed");
  assert_eq!(
    Blake2b512::keyed_digest(
      Blake2bKey::new(&KEY_64).expect("64-byte BLAKE2b migration key must be valid"),
      DATA,
    ),
    dryoc_keyed_b512
  );
}

#[test]
fn test_dryoc_ed25519_migration_examples_are_byte_equivalent() {
  let seed = [0x13; 32];
  let (dryoc_public, dryoc_secret) = crypto_sign_seed_keypair(&seed);
  let mut dryoc_signature = [0u8; 64];
  crypto_sign_detached(&mut dryoc_signature, DATA, &dryoc_secret)
    .expect("dryoc Ed25519 migration signing must succeed");

  let ours = Ed25519SecretKey::from_bytes(seed);
  let ours_public = ours.public_key();
  let ours_signature = ours.sign(DATA);

  assert_eq!(ours_public.as_bytes(), &dryoc_public);
  assert_eq!(ours_signature.as_bytes(), &dryoc_signature);

  crypto_sign_verify_detached(ours_signature.as_bytes(), DATA, &dryoc_public)
    .expect("dryoc must verify the rscrypto Ed25519 migration signature");
  ours_public
    .verify(DATA, &ours_signature)
    .expect("rscrypto must verify its Ed25519 migration signature");
}

#[test]
fn test_dryoc_x25519_migration_examples_are_byte_equivalent() {
  let alice_bytes = [0x18; 32];
  let bob_bytes = [0x34; 32];

  let ours_alice = X25519SecretKey::from_bytes(alice_bytes);
  let ours_bob_public = X25519SecretKey::from_bytes(bob_bytes).public_key();
  let ours_shared = ours_alice
    .diffie_hellman(&ours_bob_public)
    .expect("rscrypto X25519 migration exchange must produce a nonzero secret");

  let mut dryoc_bob_public = [0u8; 32];
  crypto_scalarmult_base(&mut dryoc_bob_public, &bob_bytes);
  assert_eq!(ours_bob_public.as_bytes(), &dryoc_bob_public);

  let mut dryoc_shared = [0u8; 32];
  crypto_scalarmult(&mut dryoc_shared, &alice_bytes, &dryoc_bob_public)
    .expect("dryoc X25519 migration exchange must produce a nonzero secret");
  assert_eq!(ours_shared.as_bytes(), &dryoc_shared);
}
