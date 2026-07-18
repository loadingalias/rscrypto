//! Public password-record behavior and hostile-PHC regression tests.

#![cfg(all(
  feature = "argon2",
  feature = "scrypt",
  feature = "phc-strings",
  feature = "getrandom"
))]

use core::{
  alloc::{GlobalAlloc, Layout},
  cell::Cell,
};
use std::alloc::System;

use rscrypto::{Argon2Params, Argon2idPassword, PasswordStatus, ScryptParams, ScryptPassword};

struct TrackingAllocator;

thread_local! {
  static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
  static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
  TRACK_ALLOCATIONS.with(|tracking| {
    if tracking.get() {
      ALLOCATION_COUNT.with(|count| count.set(count.get().strict_add(1)));
    }
  });
}

// SAFETY: TrackingAllocator delegates every operation to System with the
// exact pointer and layout supplied by the caller. It adds only thread-local
// accounting after successful allocation and never changes allocator results.
unsafe impl GlobalAlloc for TrackingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    // SAFETY:
    // 1. layout is supplied by the GlobalAlloc caller.
    // 2. System is the backing allocator for every pointer returned here.
    // 3. The returned pointer is forwarded unchanged.
    let pointer = unsafe { System.alloc(layout) };
    if !pointer.is_null() {
      record_allocation();
    }
    pointer
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    // SAFETY:
    // 1. layout is supplied by the GlobalAlloc caller.
    // 2. System is the backing allocator for every pointer returned here.
    // 3. The returned pointer is forwarded unchanged.
    let pointer = unsafe { System.alloc_zeroed(layout) };
    if !pointer.is_null() {
      record_allocation();
    }
    pointer
  }

  unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
    // SAFETY:
    // 1. Every pointer returned by this allocator comes from System.
    // 2. The caller must provide the original allocation layout.
    // 3. Neither pointer nor layout is changed before delegation.
    unsafe { System.dealloc(pointer, layout) };
  }

  unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY:
    // 1. Every pointer returned by this allocator comes from System.
    // 2. The caller supplies its current layout and requested new size.
    // 3. All arguments and the returned pointer are forwarded unchanged.
    let replacement = unsafe { System.realloc(pointer, layout, new_size) };
    if !replacement.is_null() {
      record_allocation();
    }
    replacement
  }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn allocations_during<T>(operation: impl FnOnce() -> T) -> (T, usize) {
  TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
  ALLOCATION_COUNT.with(|count| count.set(0));
  TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
  let result = operation();
  TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
  let allocations = ALLOCATION_COUNT.with(Cell::get);
  (result, allocations)
}

#[test]
fn generated_password_records_are_canonical_and_self_verifying() {
  let argon2 = Argon2idPassword::new(Argon2Params::new(32, 2, 1).unwrap()).unwrap();
  let argon2_record = argon2.hash_password(b"correct horse battery staple").unwrap();
  assert!(argon2_record.starts_with("$argon2id$v=19$m=32,t=2,p=1$"));
  assert_eq!(
    argon2.verify_password(b"correct horse battery staple", &argon2_record),
    Ok(PasswordStatus::Current)
  );
  assert!(argon2.verify_password(b"wrong", &argon2_record).is_err());

  let scrypt = ScryptPassword::new(ScryptParams::new(4, 1, 1).unwrap()).unwrap();
  let scrypt_record = scrypt.hash_password(b"correct horse battery staple").unwrap();
  assert!(scrypt_record.starts_with("$scrypt$ln=4,r=1,p=1$"));
  assert_eq!(
    scrypt.verify_password(b"correct horse battery staple", &scrypt_record),
    Ok(PasswordStatus::Current)
  );
  assert!(scrypt.verify_password(b"wrong", &scrypt_record).is_err());
}

#[test]
fn verifier_reports_accepted_stale_profiles() {
  let old_argon2 = Argon2idPassword::new(Argon2Params::new(32, 2, 1).unwrap()).unwrap();
  let argon2_record = old_argon2.hash_password(b"password").unwrap();
  let current_argon2 = Argon2idPassword::new(Argon2Params::new(40, 2, 1).unwrap()).unwrap();
  assert_eq!(
    current_argon2.verify_password(b"password", &argon2_record),
    Ok(PasswordStatus::NeedsRehash)
  );

  let old_scrypt = ScryptPassword::new(ScryptParams::new(4, 1, 1).unwrap()).unwrap();
  let scrypt_record = old_scrypt.hash_password(b"password").unwrap();
  let current_scrypt = ScryptPassword::new(ScryptParams::new(5, 1, 1).unwrap()).unwrap();
  assert_eq!(
    current_scrypt.verify_password(b"password", &scrypt_record),
    Ok(PasswordStatus::NeedsRehash)
  );
}

#[test]
fn every_rejected_phc_class_allocates_nothing() {
  let argon2 = Argon2idPassword::new(Argon2Params::new(32, 2, 1).unwrap()).unwrap();
  let oversized = "x".repeat(1_025);
  let argon2_rejections = [
    oversized.as_str(),
    "not-a-phc-record",
    "$scrypt$ln=4,r=1,p=1$*$*",
    "$argon2id$v=19$t=2,m=32,p=1$*$*",
    "$argon2id$v=19$m=32,m=2,p=1$*$*",
    "$argon2id$v=19$m=42949672960,t=2,p=1$*$*",
    "$argon2id$v=19$m=40,t=2,p=1$*$*",
    "$argon2id$v=19$m=32,t=2,p=1$**********************$*******************************************",
  ];
  for encoded in argon2_rejections {
    let (result, allocations) = allocations_during(|| argon2.verify_password(b"password", encoded));
    assert!(result.is_err(), "rejected Argon2 PHC: {encoded}");
    assert_eq!(allocations, 0, "rejected Argon2 PHC allocated: {encoded}");
  }

  let scrypt = ScryptPassword::new(ScryptParams::new(4, 1, 1).unwrap()).unwrap();
  let scrypt_rejections = [
    oversized.as_str(),
    "not-a-phc-record",
    "$argon2id$v=19$m=32,t=2,p=1$*$*",
    "$scrypt$r=1,ln=4,p=1$*$*",
    "$scrypt$ln=4,ln=1,p=1$*$*",
    "$scrypt$ln=42949672960,r=1,p=1$*$*",
    "$scrypt$ln=5,r=1,p=1$*$*",
    "$scrypt$ln=4,r=1,p=1$**********************$*******************************************",
  ];
  for encoded in scrypt_rejections {
    let (result, allocations) = allocations_during(|| scrypt.verify_password(b"password", encoded));
    assert!(result.is_err(), "rejected scrypt PHC: {encoded}");
    assert_eq!(allocations, 0, "rejected scrypt PHC allocated: {encoded}");
  }
}

#[test]
fn public_verifiers_reject_noncanonical_and_cross_algorithm_records() {
  let argon2 = Argon2idPassword::new(Argon2Params::new(32, 2, 1).unwrap()).unwrap();
  let scrypt = ScryptPassword::new(ScryptParams::new(4, 1, 1).unwrap()).unwrap();
  let valid_argon2 = argon2.hash_password(b"password").unwrap();
  let valid_scrypt = scrypt.hash_password(b"password").unwrap();

  assert!(argon2.verify_password(b"password", &valid_scrypt).is_err());
  assert!(scrypt.verify_password(b"password", &valid_argon2).is_err());
  assert!(
    argon2
      .verify_password(
        b"password",
        "$argon2id$v=19$t=2,m=32,p=1$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
      )
      .is_err()
  );
  assert!(
    scrypt
      .verify_password(
        b"password",
        "$scrypt$r=1,ln=4,p=1$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
      )
      .is_err()
  );
}
