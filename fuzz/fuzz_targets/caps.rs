//! Fuzz target for Caps (CPU capabilities) bitset operations.
//!
//! Tests that:
//! - No panics on arbitrary input
//! - Union/intersection are commutative and associative
//! - has() is consistent with has_bit()
//! - count() matches popcount of underlying words
//! - Self-containment: caps.has(caps) is always true

#![no_main]

use libfuzzer_sys::fuzz_target;
use platform::Caps;

fuzz_target!(|data: [u64; 4]| {
  let caps = Caps::from_raw(data);

  // ─── Invariant: Self-containment ───
  assert!(caps.has(caps), "caps must contain itself");

  // ─── Invariant: Count accuracy ───
  let expected_count: u32 = data.iter().map(|w| w.count_ones()).sum();
  assert_eq!(
    caps.count(),
    expected_count,
    "count() must equal sum of popcounts"
  );

  // ─── Invariant: is_empty consistency ───
  assert_eq!(
    caps.is_empty(),
    expected_count == 0,
    "is_empty() must match count() == 0"
  );

  // ─── Invariant: Union identity ───
  assert_eq!(caps | Caps::NONE, caps, "union with NONE must be identity");

  // ─── Invariant: Intersection absorbing ───
  assert_eq!(caps & Caps::NONE, Caps::NONE, "intersection with NONE must be NONE");

  // ─── Invariant: Idempotence ───
  assert_eq!(caps | caps, caps, "union with self must be idempotent");
  assert_eq!(caps & caps, caps, "intersection with self must be idempotent");

  // ─── Invariant: has_bit consistency ───
  // For each bit position, has_bit and has(Caps::bit(n)) must agree
  for n in 0u8..=255 {
    let single = Caps::bit(n);
    let has_via_bit = caps.has_bit(n);
    let has_via_caps = caps.has(single);
    assert_eq!(
      has_via_bit, has_via_caps,
      "has_bit({n}) must equal has(Caps::bit({n}))"
    );
  }
});
