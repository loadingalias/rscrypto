#![cfg(feature = "p256-ecdh")]

use rscrypto::{P256EphemeralSecret, P256PublicKey};

#[path = "common/allocations.rs"]
mod allocations;
use allocations::measure_allocations;

fn secret(bytes: [u8; 32]) -> P256EphemeralSecret {
  P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("test scalar must be valid")
}

#[test]
fn complete_p256_ecdh_operation_is_allocation_free() {
  let peer_bytes = secret([0x24; 32]).public_key().to_sec1_bytes();
  let allocations = measure_allocations(|| {
    let local = secret([0x42; 32]);
    let local_public = local.public_key();
    let peer = P256PublicKey::from_sec1_bytes(&peer_bytes).expect("peer key must parse");
    let shared = local.diffie_hellman(&peer);
    core::hint::black_box((local_public, shared));
  });

  assert_eq!(
    allocations, 0,
    "P-256 ECDH generation, derivation, parsing, and agreement must not allocate"
  );
}

#[test]
fn p256_public_key_storage_footprint_is_bounded() {
  assert_eq!(core::mem::size_of::<P256EphemeralSecret>(), 32);
  assert_eq!(core::mem::size_of::<rscrypto::P256SharedSecret>(), 32);
  assert!(
    core::mem::size_of::<P256PublicKey>() <= 144,
    "P-256 public key exceeded the reviewed footprint"
  );
}
