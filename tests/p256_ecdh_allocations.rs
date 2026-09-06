#![cfg(feature = "p256-ecdh")]

use core::{
  alloc::{GlobalAlloc, Layout},
  cell::Cell,
};
use std::alloc::System;

use rscrypto::{P256EphemeralSecret, P256PublicKey};

struct CountingAllocator;

const MEASUREMENT_DISABLED: usize = usize::MAX;

std::thread_local! {
  static ALLOCATIONS: Cell<usize> = const { Cell::new(MEASUREMENT_DISABLED) };
}

fn record_allocation() {
  let _counter_result = ALLOCATIONS.try_with(|allocations| {
    let count = allocations.get();
    if count != MEASUREMENT_DISABLED {
      allocations.set(count.strict_add(1));
    }
  });
}

// SAFETY: This allocator delegates every memory operation to `System` with the
// original pointer and layout. The thread-local counter observes allocation
// calls without changing the delegated allocation contract.
unsafe impl GlobalAlloc for CountingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    record_allocation();
    // SAFETY: `layout` is forwarded unchanged to the system allocator.
    unsafe { System.alloc(layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    record_allocation();
    // SAFETY: `layout` is forwarded unchanged to the system allocator.
    unsafe { System.alloc_zeroed(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: `ptr` and `layout` came from the delegated system allocator.
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    record_allocation();
    // SAFETY: The original pointer, layout, and requested size are forwarded
    // unchanged to the system allocator.
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

struct Measurement;

impl Drop for Measurement {
  fn drop(&mut self) {
    let _counter_result = ALLOCATIONS.try_with(|allocations| allocations.set(MEASUREMENT_DISABLED));
  }
}

fn measure_allocations(operation: impl FnOnce()) -> usize {
  ALLOCATIONS.with(|allocations| {
    assert_eq!(
      allocations.replace(0),
      MEASUREMENT_DISABLED,
      "allocation measurements must not be nested"
    );
  });
  let measurement = Measurement;
  operation();
  let count = ALLOCATIONS.with(|allocations| allocations.replace(MEASUREMENT_DISABLED));
  drop(measurement);
  count
}

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
