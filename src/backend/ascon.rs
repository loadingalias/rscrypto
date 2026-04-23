//! Shared Ascon permutation kernels used by hash and AEAD surfaces.

// Single round on five local state words.
macro_rules! ascon_round {
  ($x0:ident, $x1:ident, $x2:ident, $x3:ident, $x4:ident, $c:literal) => {
    let t0 = $x0 ^ $x4;
    let t2 = $x2 ^ $x1 ^ $c;
    let t4 = $x4 ^ $x3;

    let s0 = t0 ^ ((!$x1) & t2);
    let s1 = $x1 ^ ((!t2) & $x3);
    let s2 = t2 ^ ((!$x3) & t4);
    let s3 = $x3 ^ ((!t4) & t0);
    let s4 = t4 ^ ((!t0) & $x1);

    let s1 = s1 ^ s0;
    let s3 = s3 ^ s2;
    let s0 = s0 ^ s4;

    let l0 = s0 ^ s0.rotate_right(9);
    let l1 = s1 ^ s1.rotate_right(22);
    let l2 = s2 ^ s2.rotate_right(5);
    let l3 = s3 ^ s3.rotate_right(7);
    let l4 = s4 ^ s4.rotate_right(34);

    $x0 = s0 ^ l0.rotate_right(19);
    $x1 = s1 ^ l1.rotate_right(39);
    $x2 = !(s2 ^ l2.rotate_right(1));
    $x3 = s3 ^ l3.rotate_right(10);
    $x4 = s4 ^ l4.rotate_right(7);
  };
}

/// Ascon-p[12]: full 12-round permutation (portable, scalar).
///
/// Fully unrolled so the compiler can keep all five 64-bit state words in
/// registers across the entire permutation and freely schedule instructions.
#[inline(always)]
pub(crate) fn permute_12_portable(s: &mut [u64; 5]) {
  let mut x0 = s[0];
  let mut x1 = s[1];
  let mut x2 = s[2];
  let mut x3 = s[3];
  let mut x4 = s[4];

  ascon_round!(x0, x1, x2, x3, x4, 0xF0);
  ascon_round!(x0, x1, x2, x3, x4, 0xE1);
  ascon_round!(x0, x1, x2, x3, x4, 0xD2);
  ascon_round!(x0, x1, x2, x3, x4, 0xC3);
  ascon_round!(x0, x1, x2, x3, x4, 0xB4);
  ascon_round!(x0, x1, x2, x3, x4, 0xA5);
  ascon_round!(x0, x1, x2, x3, x4, 0x96);
  ascon_round!(x0, x1, x2, x3, x4, 0x87);
  ascon_round!(x0, x1, x2, x3, x4, 0x78);
  ascon_round!(x0, x1, x2, x3, x4, 0x69);
  ascon_round!(x0, x1, x2, x3, x4, 0x5A);
  ascon_round!(x0, x1, x2, x3, x4, 0x4B);

  s[0] = x0;
  s[1] = x1;
  s[2] = x2;
  s[3] = x3;
  s[4] = x4;
}

/// Ascon-p[8]: 8-round permutation (PB) used by Ascon-AEAD128.
///
/// Fully unrolled for the same register-allocation benefits as
/// [`permute_12_portable`].
#[cfg(any(feature = "ascon-aead", test))]
#[cfg_attr(test, allow(dead_code))]
#[inline(always)]
pub(crate) fn permute_8_portable(s: &mut [u64; 5]) {
  let mut x0 = s[0];
  let mut x1 = s[1];
  let mut x2 = s[2];
  let mut x3 = s[3];
  let mut x4 = s[4];

  ascon_round!(x0, x1, x2, x3, x4, 0xB4);
  ascon_round!(x0, x1, x2, x3, x4, 0xA5);
  ascon_round!(x0, x1, x2, x3, x4, 0x96);
  ascon_round!(x0, x1, x2, x3, x4, 0x87);
  ascon_round!(x0, x1, x2, x3, x4, 0x78);
  ascon_round!(x0, x1, x2, x3, x4, 0x69);
  ascon_round!(x0, x1, x2, x3, x4, 0x5A);
  ascon_round!(x0, x1, x2, x3, x4, 0x4B);

  s[0] = x0;
  s[1] = x1;
  s[2] = x2;
  s[3] = x3;
  s[4] = x4;
}
