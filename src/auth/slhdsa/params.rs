//! FIPS 205 Table 2 parameter sets and the values derived from them.

/// One FIPS 205 parameter set.
///
/// The core carries `n` as a const generic; this value repeats it so the
/// derived lengths are computed in one place.
#[derive(Clone, Copy)]
pub(crate) struct Params {
  pub(crate) n: usize,
  /// Total hypertree height h.
  pub(crate) h: u32,
  /// Hypertree layers d.
  pub(crate) d: u32,
  /// XMSS tree height h' = h / d.
  pub(crate) hp: u32,
  /// FORS tree height a.
  pub(crate) a: u32,
  /// FORS trees k.
  pub(crate) k: usize,
  /// Bits per WOTS+ chain digit, lg_w.
  pub(crate) lg_w: u32,
  /// WOTS+ message digits, len1 (Equation 5.2).
  pub(crate) len1: usize,
  /// WOTS+ checksum digits, len2 (Algorithm 1).
  pub(crate) len2: usize,
  /// WOTS+ chains, len = len1 + len2.
  pub(crate) len: usize,
  /// Message digest bytes signed by FORS, ceil(k * a / 8).
  pub(crate) md_bytes: usize,
  /// Digest bytes selecting the XMSS tree, ceil((h - h') / 8).
  pub(crate) tree_bytes: usize,
  /// Digest bytes selecting the leaf, ceil(h' / 8).
  pub(crate) leaf_bytes: usize,
  /// H_msg output bytes m.
  pub(crate) m: usize,
  /// Signature bytes, (1 + k(1 + a) + h + d * len) * n.
  pub(crate) signature_len: usize,
}

/// Largest XMSS or FORS tree height among the supported sets; sizes the
/// tree-hash node stack.
pub(crate) const MAX_TREE_HEIGHT: usize = 14;
/// Largest WOTS+ chain count among the supported sets.
pub(crate) const MAX_WOTS_LEN: usize = 67;
/// Largest FORS tree count among the supported sets.
pub(crate) const MAX_FORS_TREES: usize = 35;
/// Largest H_msg output among the supported sets.
pub(crate) const MAX_DIGEST: usize = 49;

impl Params {
  /// Derive a parameter set from FIPS 205's free parameters and check that it
  /// fits the core's fixed capacities.
  const fn new(n: usize, h: u32, d: u32, a: u32, k: usize, lg_w: u32) -> Self {
    assert!(h.is_multiple_of(d), "the hypertree height divides into layers");
    let hp = h.strict_div(d);
    assert!(lg_w >= 1 && lg_w <= 8, "a WOTS+ digit fits in a byte");
    assert!(d <= 255, "a layer address fits the compressed address byte");
    assert!(h.strict_sub(hp) <= 64, "an XMSS tree index fits the address");
    assert!(
      hp as usize <= MAX_TREE_HEIGHT && a as usize <= MAX_TREE_HEIGHT,
      "trees fit the node stack"
    );
    assert!(k <= MAX_FORS_TREES, "FORS roots fit their buffer");
    assert!((k << a) <= 1 << 32, "FORS tree indices fit the 4-byte address word");

    let len1 = (8usize.strict_mul(n)).div_ceil(lg_w as usize);
    let len2 = gen_len2(len1, lg_w);
    let len = len1.strict_add(len2);
    assert!(len <= MAX_WOTS_LEN, "WOTS+ chains fit their buffer");
    assert!(
      (len2.strict_mul(lg_w as usize)).div_ceil(8) <= 4,
      "the WOTS+ checksum fits a u32"
    );

    let md_bytes = (k.strict_mul(a as usize)).div_ceil(8);
    let tree_bytes = (h.strict_sub(hp) as usize).div_ceil(8);
    let leaf_bytes = (hp as usize).div_ceil(8);
    let m = md_bytes.strict_add(tree_bytes).strict_add(leaf_bytes);
    assert!(m <= MAX_DIGEST, "the message digest fits its buffer");

    let nodes = 1usize
      .strict_add(k.strict_mul(1usize.strict_add(a as usize)))
      .strict_add(h as usize)
      .strict_add((d as usize).strict_mul(len));
    Self {
      n,
      h,
      d,
      hp,
      a,
      k,
      lg_w,
      len1,
      len2,
      len,
      md_bytes,
      tree_bytes,
      leaf_bytes,
      m,
      signature_len: nodes.strict_mul(n),
    }
  }

  /// Bytes in one XMSS signature: a WOTS+ signature and an authentication path.
  pub(crate) const fn xmss_nodes(&self) -> usize {
    self.len.strict_add(self.hp as usize)
  }

  /// Nodes in a FORS signature: k secret values with their authentication paths.
  pub(crate) const fn fors_nodes(&self) -> usize {
    self.k.strict_mul(1usize.strict_add(self.a as usize))
  }
}

/// len2 (Algorithm 1): checksum digits for `len1` digits of `lg_w` bits.
const fn gen_len2(len1: usize, lg_w: u32) -> usize {
  let w = 1usize << lg_w;
  let max_checksum = len1.strict_mul(w.strict_sub(1));
  let mut len2 = 1usize;
  let mut capacity = w;
  while capacity <= max_checksum {
    len2 = len2.strict_add(1);
    capacity = capacity.strict_mul(w);
  }
  len2
}

/// Check a derived set against FIPS 205 Table 2.
#[expect(clippy::too_many_arguments, reason = "one argument per FIPS 205 Table 2 column")]
const fn table_2(n: usize, h: u32, d: u32, hp: u32, a: u32, k: usize, lg_w: u32, m: usize, signature: usize) -> Params {
  let params = Params::new(n, h, d, a, k, lg_w);
  assert!(
    params.hp == hp && params.m == m && params.signature_len == signature,
    "FIPS 205 Table 2"
  );
  params
}

pub(crate) const P128S: Params = table_2(16, 63, 7, 9, 12, 14, 4, 30, 7_856);
pub(crate) const P128F: Params = table_2(16, 66, 22, 3, 6, 33, 4, 34, 17_088);
pub(crate) const P192S: Params = table_2(24, 63, 7, 9, 14, 17, 4, 39, 16_224);
pub(crate) const P192F: Params = table_2(24, 66, 22, 3, 8, 33, 4, 42, 35_664);
pub(crate) const P256S: Params = table_2(32, 64, 8, 8, 14, 22, 4, 47, 29_792);
pub(crate) const P256F: Params = table_2(32, 68, 17, 4, 9, 35, 4, 49, 49_856);
