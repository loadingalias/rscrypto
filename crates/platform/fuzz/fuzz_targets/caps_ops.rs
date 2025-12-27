//! Fuzz target for Caps binary operations (union, intersection).
//!
//! Tests algebraic properties:
//! - Commutativity: a | b == b | a, a & b == b & a
//! - Associativity: (a | b) | c == a | (b | c)
//! - Distributivity: a & (b | c) == (a & b) | (a & c)
//! - Subset relationships after operations

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use platform::Caps;

#[derive(Arbitrary, Debug)]
struct Input {
  a: [u64; 4],
  b: [u64; 4],
  c: [u64; 4],
}

fuzz_target!(|input: Input| {
  let a = Caps::from_raw(input.a);
  let b = Caps::from_raw(input.b);
  let c = Caps::from_raw(input.c);

  // ─── Commutativity ───
  assert_eq!(a | b, b | a, "union must be commutative");
  assert_eq!(a & b, b & a, "intersection must be commutative");

  // ─── Associativity ───
  assert_eq!((a | b) | c, a | (b | c), "union must be associative");
  assert_eq!((a & b) & c, a & (b & c), "intersection must be associative");

  // ─── Distributivity ───
  assert_eq!(
    a & (b | c),
    (a & b) | (a & c),
    "intersection must distribute over union"
  );

  // ─── Subset relationships after union ───
  let ab = a | b;
  assert!(ab.has(a), "union must contain first operand");
  assert!(ab.has(b), "union must contain second operand");

  // ─── Subset relationships after intersection ───
  let ab_inter = a & b;
  assert!(a.has(ab_inter), "first operand must contain intersection");
  assert!(b.has(ab_inter), "second operand must contain intersection");

  // ─── Count bounds ───
  let union = a | b;
  let intersection = a & b;
  assert!(
    union.count() >= a.count().max(b.count()),
    "union count must be >= max of operand counts"
  );
  assert!(
    intersection.count() <= a.count().min(b.count()),
    "intersection count must be <= min of operand counts"
  );

  // ─── Absorption laws ───
  // a | (a & b) == a
  assert_eq!(a | (a & b), a, "absorption law 1 failed");
  // a & (a | b) == a
  assert_eq!(a & (a | b), a, "absorption law 2 failed");
});
