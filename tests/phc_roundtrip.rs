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
  let argon2_params = Argon2Params::new(32, 2, 1).expect("Argon2 test parameters must be valid");
  let argon2 = Argon2idPassword::new(argon2_params).expect("Argon2 password service must accept valid parameters");
  let argon2_record = argon2
    .hash_password(b"correct horse battery staple")
    .expect("Argon2 password hashing must succeed");
  assert!(argon2_record.starts_with("$argon2id$v=19$m=32,t=2,p=1$"));
  assert_eq!(
    argon2.verify_password(b"correct horse battery staple", &argon2_record),
    Ok(PasswordStatus::Current)
  );
  argon2
    .verify_password(b"wrong", &argon2_record)
    .expect_err("Argon2 verification must reject the wrong password");

  let scrypt_params = ScryptParams::new(4, 1, 1).expect("scrypt test parameters must be valid");
  let scrypt = ScryptPassword::new(scrypt_params).expect("scrypt password service must accept valid parameters");
  let scrypt_record = scrypt
    .hash_password(b"correct horse battery staple")
    .expect("scrypt password hashing must succeed");
  assert!(scrypt_record.starts_with("$scrypt$ln=4,r=1,p=1$"));
  assert_eq!(
    scrypt.verify_password(b"correct horse battery staple", &scrypt_record),
    Ok(PasswordStatus::Current)
  );
  scrypt
    .verify_password(b"wrong", &scrypt_record)
    .expect_err("scrypt verification must reject the wrong password");
}

#[test]
fn verifier_reports_accepted_stale_profiles() {
  let old_argon2_params = Argon2Params::new(32, 2, 1).expect("old Argon2 profile must be valid");
  let old_argon2 =
    Argon2idPassword::new(old_argon2_params).expect("old Argon2 password service must accept its profile");
  let argon2_record = old_argon2
    .hash_password(b"password")
    .expect("old Argon2 profile must produce a password record");
  let current_argon2_params = Argon2Params::new(40, 2, 1).expect("current Argon2 profile must be valid");
  let current_argon2 =
    Argon2idPassword::new(current_argon2_params).expect("current Argon2 password service must accept its profile");
  assert_eq!(
    current_argon2.verify_password(b"password", &argon2_record),
    Ok(PasswordStatus::NeedsRehash)
  );

  let old_scrypt_params = ScryptParams::new(4, 1, 1).expect("old scrypt profile must be valid");
  let old_scrypt = ScryptPassword::new(old_scrypt_params).expect("old scrypt password service must accept its profile");
  let scrypt_record = old_scrypt
    .hash_password(b"password")
    .expect("old scrypt profile must produce a password record");
  let current_scrypt_params = ScryptParams::new(5, 1, 1).expect("current scrypt profile must be valid");
  let current_scrypt =
    ScryptPassword::new(current_scrypt_params).expect("current scrypt password service must accept its profile");
  assert_eq!(
    current_scrypt.verify_password(b"password", &scrypt_record),
    Ok(PasswordStatus::NeedsRehash)
  );
}

#[test]
fn every_rejected_phc_class_allocates_nothing() {
  let argon2_params = Argon2Params::new(32, 2, 1).expect("Argon2 rejection-test parameters must be valid");
  let argon2 =
    Argon2idPassword::new(argon2_params).expect("Argon2 password service must accept rejection-test parameters");
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
    result.expect_err("known-invalid Argon2 PHC record must be rejected");
    assert_eq!(allocations, 0, "rejected Argon2 PHC allocated: {encoded}");
  }

  let scrypt_params = ScryptParams::new(4, 1, 1).expect("scrypt rejection-test parameters must be valid");
  let scrypt =
    ScryptPassword::new(scrypt_params).expect("scrypt password service must accept rejection-test parameters");
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
    result.expect_err("known-invalid scrypt PHC record must be rejected");
    assert_eq!(allocations, 0, "rejected scrypt PHC allocated: {encoded}");
  }
}

#[test]
fn public_verifiers_reject_noncanonical_and_cross_algorithm_records() {
  let argon2_params = Argon2Params::new(32, 2, 1).expect("Argon2 cross-algorithm parameters must be valid");
  let argon2 =
    Argon2idPassword::new(argon2_params).expect("Argon2 password service must accept cross-algorithm parameters");
  let scrypt_params = ScryptParams::new(4, 1, 1).expect("scrypt cross-algorithm parameters must be valid");
  let scrypt =
    ScryptPassword::new(scrypt_params).expect("scrypt password service must accept cross-algorithm parameters");
  let valid_argon2 = argon2
    .hash_password(b"password")
    .expect("Argon2 must produce a cross-algorithm fixture");
  let valid_scrypt = scrypt
    .hash_password(b"password")
    .expect("scrypt must produce a cross-algorithm fixture");

  argon2
    .verify_password(b"password", &valid_scrypt)
    .expect_err("Argon2 must reject a scrypt record");
  scrypt
    .verify_password(b"password", &valid_argon2)
    .expect_err("scrypt must reject an Argon2 record");
  argon2
    .verify_password(
      b"password",
      "$argon2id$v=19$t=2,m=32,p=1$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    )
    .expect_err("Argon2 must reject noncanonical parameter order");
  scrypt
    .verify_password(
      b"password",
      "$scrypt$r=1,ln=4,p=1$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    )
    .expect_err("scrypt must reject noncanonical parameter order");
}
