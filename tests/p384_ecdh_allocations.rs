#![cfg(feature = "p384-ecdh")]

use rscrypto::{P384EphemeralSecret, P384PublicKey};

#[path = "common/allocations.rs"]
mod allocations;
use allocations::measure_allocations;

fn secret(bytes: [u8; 48]) -> P384EphemeralSecret {
  P384EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("test scalar must be valid")
}

#[test]
fn complete_p384_ecdh_operation_is_allocation_free() {
  let peer_bytes = secret([0x24; 48]).public_key().to_sec1_bytes();
  let allocations = measure_allocations(|| {
    let local = secret([0x42; 48]);
    let local_public = local.public_key();
    let peer = P384PublicKey::from_sec1_bytes(&peer_bytes).expect("peer key must parse");
    let shared = local.diffie_hellman(&peer);
    core::hint::black_box((local_public, shared));
  });

  assert_eq!(
    allocations, 0,
    "P-384 ECDH generation, derivation, parsing, and agreement must not allocate"
  );
}

#[test]
fn p384_public_key_storage_footprint_is_bounded() {
  assert_eq!(core::mem::size_of::<P384EphemeralSecret>(), 48);
  assert_eq!(core::mem::size_of::<rscrypto::P384SharedSecret>(), 48);
  assert!(
    core::mem::size_of::<P384PublicKey>() <= 200,
    "P-384 public key exceeded the reviewed footprint"
  );
}
