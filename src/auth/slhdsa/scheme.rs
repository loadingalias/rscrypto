//! FIPS 205 Sections 4-9: WOTS+, XMSS, the hypertree, FORS, and the internal
//! key generation, signing, and verification functions, generic over n and
//! the hash suite.
//!
//! A signature is handled as a sequence of n-byte nodes. Trees are computed by
//! an iterative tree hash over a fixed node stack, which yields each node once
//! together with the root and one leaf's authentication path; Algorithms 9 and
//! 15 specify the same nodes recursively. Signing takes each XMSS root from
//! its tree hash and the FORS public key from its roots, where Algorithms 12
//! and 19 recompute both from the new signature. The results are identical.
//!
//! All indices derive from public values: the message digest, which the
//! signature's randomizer determines, and the tree positions it selects.

use super::address::{self, Address};
use super::hash::Suite;
use super::params::{MAX_DIGEST, MAX_FORS_TREES, MAX_TREE_HEIGHT, MAX_WOTS_LEN, Params};
use crate::traits::ct;

/// base_2b (Algorithm 4): the first `out.len()` big-endian `b`-bit digits of `x`.
fn base_2b(x: &[u8], b: u32, out: &mut [u32]) {
  let mut next = 0usize;
  let mut bits = 0u32;
  let mut total = 0u64;
  for digit in out {
    while bits < b {
      total = (total << 8) | u64::from(x[next]);
      next = next.strict_add(1);
      bits = bits.strict_add(8);
    }
    bits = bits.strict_sub(b);
    *digit = u32::try_from((total >> bits) & low_mask(b)).expect("a b-bit digit");
    total &= low_mask(bits);
  }
}

/// `2^bits - 1`, for `bits` at most 64.
const fn low_mask(bits: u32) -> u64 {
  match 1u64.checked_shl(bits) {
    Some(bit) => bit.strict_sub(1),
    None => u64::MAX,
  }
}

/// toInt (Algorithm 2) of at most eight bytes.
fn to_int(bytes: &[u8]) -> u64 {
  bytes.iter().fold(0, |total, &byte| (total << 8) | u64::from(byte))
}

/// WOTS+ chain digits of `message` and its checksum (Algorithm 7, lines 1-7).
fn wots_digits<const N: usize>(p: &Params, message: &[u8; N], digits: &mut [u32; MAX_WOTS_LEN]) {
  let (message_digits, checksum_digits) = digits[..p.len].split_at_mut(p.len1);
  base_2b(message, p.lg_w, message_digits);
  let w_max = (1u32 << p.lg_w).strict_sub(1);
  let checksum = message_digits
    .iter()
    .fold(0u32, |sum, &digit| sum.strict_add(w_max.strict_sub(digit)));
  let checksum_bits = p.len2.strict_mul(p.lg_w as usize);
  let shift = (8usize.strict_sub(checksum_bits % 8)) % 8;
  let encoded = (checksum << shift).to_be_bytes();
  let checksum_bytes = checksum_bits.div_ceil(8);
  base_2b(&encoded[4usize.strict_sub(checksum_bytes)..], p.lg_w, checksum_digits);
}

/// chain (Algorithm 5): apply F `steps` times from chain position `start`.
fn chain<S: Suite<N>, const N: usize>(suite: &S, address: &mut Address, value: &mut [u8; N], start: u32, steps: u32) {
  for position in start..start.strict_add(steps) {
    address.set_hash(position);
    suite.f(address, value);
  }
}

/// wots_pkGen (Algorithm 6) for the key pair at `address` (type WOTS_HASH).
fn wots_public_key<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  sk_seed: &[u8; N],
  address: &Address,
  out: &mut [u8; N],
) {
  let mut sk_address = address.with_type_keeping_key_pair(address::WOTS_PRF);
  let mut chain_address = *address;
  let w_max = (1u32 << p.lg_w).strict_sub(1);
  // Each end starts as a chain's secret value and leaves as its public end.
  let mut ends = [[0u8; N]; MAX_WOTS_LEN];
  for (index, end) in (0u32..).zip(&mut ends[..p.len]) {
    sk_address.set_chain(index);
    suite.prf(&sk_address, sk_seed, end);
    chain_address.set_chain(index);
    chain(suite, &mut chain_address, end, 0, w_max);
  }
  let pk_address = address.with_type_keeping_key_pair(address::WOTS_PK);
  suite.t(&pk_address, &ends[..p.len], out);
}

/// wots_sign (Algorithm 7) of `message` with the key pair at `address`.
fn wots_sign<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  message: &[u8; N],
  sk_seed: &[u8; N],
  address: &Address,
  out: &mut [[u8; N]],
) {
  let mut digits = [0; MAX_WOTS_LEN];
  wots_digits(p, message, &mut digits);
  let mut sk_address = address.with_type_keeping_key_pair(address::WOTS_PRF);
  let mut chain_address = *address;
  // Each node starts as a chain's secret value and leaves at its revealed position.
  for ((index, node), &digit) in (0u32..).zip(out.iter_mut()).zip(&digits[..p.len]) {
    sk_address.set_chain(index);
    suite.prf(&sk_address, sk_seed, node);
    chain_address.set_chain(index);
    chain(suite, &mut chain_address, node, 0, digit);
  }
}

/// wots_pkFromSig (Algorithm 8).
fn wots_public_key_from_signature<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  signature: &[[u8; N]],
  message: &[u8; N],
  address: &Address,
  out: &mut [u8; N],
) {
  let mut digits = [0; MAX_WOTS_LEN];
  wots_digits(p, message, &mut digits);
  let mut chain_address = *address;
  let w_max = (1u32 << p.lg_w).strict_sub(1);
  let mut ends = [[0u8; N]; MAX_WOTS_LEN];
  for ((index, end), (node, &digit)) in (0u32..)
    .zip(&mut ends[..p.len])
    .zip(signature.iter().zip(&digits[..p.len]))
  {
    *end = *node;
    chain_address.set_chain(index);
    chain(suite, &mut chain_address, end, digit, w_max.strict_sub(digit));
  }
  let pk_address = address.with_type_keeping_key_pair(address::WOTS_PK);
  suite.t(&pk_address, &ends[..p.len], out);
}

/// Root of the Merkle tree over the 2^`height` leaves with global indices
/// `first..first + 2^height`, writing the authentication path of leaf
/// `first + target` when `auth` is `Some((target, path))`.
///
/// `node_address` carries the tree's layer, tree, type, and key pair fields;
/// each internal node sets its height and global index (Algorithms 9 and 15,
/// lines 8-11).
fn tree_hash<S: Suite<N>, const N: usize>(
  suite: &S,
  height: u32,
  first: u32,
  mut auth: Option<(u32, &mut [[u8; N]])>,
  node_address: &mut Address,
  mut leaf: impl FnMut(u32, &mut [u8; N]),
  root: &mut [u8; N],
) {
  let mut stack = [[0u8; N]; MAX_TREE_HEIGHT + 1];
  let mut heights = [0u32; MAX_TREE_HEIGHT + 1];
  let mut depth = 0usize;
  for offset in 0..1u32 << height {
    let mut node = [0u8; N];
    leaf(first.strict_add(offset), &mut node);
    let mut node_height = 0u32;
    loop {
      if let Some((target, path)) = auth.as_mut()
        && node_height < height
        && offset >> node_height == (*target >> node_height) ^ 1
      {
        path[node_height as usize] = node;
      }
      if depth == 0 || heights[depth.strict_sub(1)] != node_height {
        break;
      }
      depth = depth.strict_sub(1);
      node_height = node_height.strict_add(1);
      node_address.set_tree_height(node_height);
      node_address.set_tree_index(first.strict_add(offset) >> node_height);
      let right = node;
      suite.h(node_address, &stack[depth], &right, &mut node);
    }
    stack[depth] = node;
    heights[depth] = node_height;
    depth = depth.strict_add(1);
  }
  *root = stack[0];
}

/// Root of the XMSS tree at `address`'s layer and tree address (Algorithm 9
/// with z = h'), writing the authentication path of leaf `auth.0` if requested.
fn xmss_tree<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  sk_seed: &[u8; N],
  address: &Address,
  auth: Option<(u32, &mut [[u8; N]])>,
  root: &mut [u8; N],
) {
  let mut node_address = *address;
  node_address.set_type_and_clear(address::TREE);
  let leaf = |index: u32, node: &mut [u8; N]| {
    let mut leaf_address = *address;
    leaf_address.set_type_and_clear(address::WOTS_HASH);
    leaf_address.set_key_pair(index);
    wots_public_key(p, suite, sk_seed, &leaf_address, node);
  };
  tree_hash(suite, p.hp, 0, auth, &mut node_address, leaf, root);
}

/// xmss_sign (Algorithm 10) of `message` with leaf `index`, writing
/// len + h' nodes to `out`; returns the tree's public root.
fn xmss_sign<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  message: &[u8; N],
  sk_seed: &[u8; N],
  index: u32,
  address: &Address,
  out: &mut [[u8; N]],
) -> [u8; N] {
  let (wots, auth) = out.split_at_mut(p.len);
  let mut root = [0u8; N];
  xmss_tree(p, suite, sk_seed, address, Some((index, auth)), &mut root);
  let mut wots_address = *address;
  wots_address.set_type_and_clear(address::WOTS_HASH);
  wots_address.set_key_pair(index);
  wots_sign(p, suite, message, sk_seed, &wots_address, wots);
  root
}

/// Hash `node`, at global leaf index `index`, up its authentication path
/// (Algorithm 11, lines 6-18; Algorithm 17, lines 8-18).
fn climb<S: Suite<N>, const N: usize>(
  suite: &S,
  address: &mut Address,
  mut index: u32,
  auth: &[[u8; N]],
  node: &mut [u8; N],
) {
  for (height, sibling) in (1u32..).zip(auth) {
    let current = *node;
    let left_child = index & 1 == 0;
    index >>= 1;
    address.set_tree_height(height);
    address.set_tree_index(index);
    if left_child {
      suite.h(address, &current, sibling, node);
    } else {
      suite.h(address, sibling, &current, node);
    }
  }
}

/// xmss_pkFromSig (Algorithm 11).
fn xmss_root_from_signature<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  index: u32,
  signature: &[[u8; N]],
  message: &[u8; N],
  address: &Address,
  root: &mut [u8; N],
) {
  let (wots, auth) = signature.split_at(p.len);
  let mut wots_address = *address;
  wots_address.set_type_and_clear(address::WOTS_HASH);
  wots_address.set_key_pair(index);
  wots_public_key_from_signature(p, suite, wots, message, &wots_address, root);
  let mut node_address = *address;
  node_address.set_type_and_clear(address::TREE);
  climb(suite, &mut node_address, index, auth, root);
}

/// The next layer's leaf and tree indices (Algorithm 12, lines 7-8).
fn parent_position(p: &Params, tree: u64) -> (u32, u64) {
  let leaf = u32::try_from(tree & low_mask(p.hp)).expect("an XMSS leaf index");
  (leaf, tree >> p.hp)
}

/// ht_sign (Algorithm 12), writing h + d * len nodes to `out`.
fn ht_sign<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  message: &[u8; N],
  sk_seed: &[u8; N],
  mut tree: u64,
  mut leaf: u32,
  out: &mut [[u8; N]],
) {
  let mut address = Address::new();
  address.set_tree(tree);
  let (bottom, upper) = out.split_at_mut(p.xmss_nodes());
  let mut root = xmss_sign(p, suite, message, sk_seed, leaf, &address, bottom);
  for (layer, xmss) in (1u8..).zip(upper.chunks_exact_mut(p.xmss_nodes())) {
    (leaf, tree) = parent_position(p, tree);
    address.set_layer(layer);
    address.set_tree(tree);
    root = xmss_sign(p, suite, &root, sk_seed, leaf, &address, xmss);
  }
}

/// ht_verify (Algorithm 13).
fn ht_verify<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  message: &[u8; N],
  signature: &[[u8; N]],
  mut tree: u64,
  mut leaf: u32,
  pk_root: &[u8; N],
) -> bool {
  let mut address = Address::new();
  address.set_tree(tree);
  let mut node = [0u8; N];
  let (bottom, upper) = signature.split_at(p.xmss_nodes());
  xmss_root_from_signature(p, suite, leaf, bottom, message, &address, &mut node);
  for (layer, xmss) in (1u8..).zip(upper.chunks_exact(p.xmss_nodes())) {
    (leaf, tree) = parent_position(p, tree);
    address.set_layer(layer);
    address.set_tree(tree);
    let signed = node;
    xmss_root_from_signature(p, suite, leaf, xmss, &signed, &address, &mut node);
  }
  node == *pk_root
}

/// fors_skGen (Algorithm 14) for global leaf `index`.
fn fors_secret<S: Suite<N>, const N: usize>(
  suite: &S,
  sk_seed: &[u8; N],
  address: &Address,
  index: u32,
  out: &mut [u8; N],
) {
  let mut sk_address = address.with_type_keeping_key_pair(address::FORS_PRF);
  sk_address.set_tree_index(index);
  suite.prf(&sk_address, sk_seed, out);
}

/// fors_sign (Algorithm 16) of `md`, writing k(1 + a) nodes to `out` and the
/// FORS public key to `pk`.
fn fors_sign<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  md: &[u8],
  sk_seed: &[u8; N],
  address: &Address,
  out: &mut [[u8; N]],
  pk: &mut [u8; N],
) {
  let mut indices = [0; MAX_FORS_TREES];
  base_2b(md, p.a, &mut indices[..p.k]);
  let mut roots = [[0u8; N]; MAX_FORS_TREES];
  let trees = (0u32..)
    .zip(out.chunks_exact_mut(1usize.strict_add(p.a as usize)))
    .zip(&indices[..p.k])
    .zip(&mut roots[..p.k]);
  for (((tree, signature), &index), root) in trees {
    let first = tree << p.a;
    let (secret, auth) = signature.split_at_mut(1);
    fors_secret(suite, sk_seed, address, first.strict_add(index), &mut secret[0]);
    let leaf = |leaf: u32, node: &mut [u8; N]| {
      // The leaf holds its secret value until F replaces it (Algorithm 15, lines 2-5).
      fors_secret(suite, sk_seed, address, leaf, node);
      let mut leaf_address = *address;
      leaf_address.set_tree_height(0);
      leaf_address.set_tree_index(leaf);
      suite.f(&leaf_address, node);
    };
    let mut node_address = *address;
    tree_hash(suite, p.a, first, Some((index, auth)), &mut node_address, leaf, root);
  }
  let pk_address = address.with_type_keeping_key_pair(address::FORS_ROOTS);
  suite.t(&pk_address, &roots[..p.k], pk);
}

/// fors_pkFromSig (Algorithm 17).
fn fors_public_key_from_signature<S: Suite<N>, const N: usize>(
  p: &Params,
  suite: &S,
  md: &[u8],
  signature: &[[u8; N]],
  address: &Address,
  pk: &mut [u8; N],
) {
  let mut indices = [0; MAX_FORS_TREES];
  base_2b(md, p.a, &mut indices[..p.k]);
  let mut roots = [[0u8; N]; MAX_FORS_TREES];
  let trees = (0u32..)
    .zip(signature.chunks_exact(1usize.strict_add(p.a as usize)))
    .zip(&indices[..p.k])
    .zip(&mut roots[..p.k]);
  for (((tree, signature), &index), root) in trees {
    let leaf = (tree << p.a).strict_add(index);
    let (secret, auth) = signature.split_at(1);
    let mut node_address = *address;
    node_address.set_tree_height(0);
    node_address.set_tree_index(leaf);
    *root = secret[0];
    suite.f(&node_address, root);
    climb(suite, &mut node_address, leaf, auth, root);
  }
  let pk_address = address.with_type_keeping_key_pair(address::FORS_ROOTS);
  suite.t(&pk_address, &roots[..p.k], pk);
}

/// Split the message digest into the FORS digest and the tree and leaf
/// indices (Algorithm 19, lines 6-10).
fn split_digest<'a>(p: &Params, digest: &'a [u8]) -> (&'a [u8], u64, u32) {
  let (md, rest) = digest.split_at(p.md_bytes);
  let (tree, rest) = rest.split_at(p.tree_bytes);
  let tree = to_int(tree) & low_mask(p.h.strict_sub(p.hp));
  let leaf = to_int(&rest[..p.leaf_bytes]) & low_mask(p.hp);
  (md, tree, u32::try_from(leaf).expect("an XMSS leaf index"))
}

/// The FORS address selected by a digest (Algorithm 19, lines 11-13).
fn fors_address(tree: u64, leaf: u32) -> Address {
  let mut address = Address::new();
  address.set_tree(tree);
  address.set_type_and_clear(address::FORS_TREE);
  address.set_key_pair(leaf);
  address
}

/// The four private-key fields (FIPS 205 Figure 15), borrowed from one key.
pub(crate) struct PrivateKey<'a, const N: usize> {
  pub(crate) sk_seed: &'a [u8; N],
  pub(crate) sk_prf: &'a [u8; N],
  pub(crate) pk_seed: &'a [u8; N],
  pub(crate) pk_root: &'a [u8; N],
}

/// slh_keygen_internal (Algorithm 18): PK.root for `sk_seed` and `pk_seed`.
pub(crate) fn keygen<S: Suite<N>, const N: usize>(
  p: &Params,
  sk_seed: &[u8; N],
  pk_seed: &[u8; N],
  pk_root: &mut [u8; N],
) {
  debug_assert_eq!(p.n, N);
  let suite = S::new(pk_seed);
  let mut address = Address::new();
  address.set_layer(u8::try_from(p.d.strict_sub(1)).expect("a layer address"));
  xmss_tree(p, &suite, sk_seed, &address, None, pk_root);
}

/// slh_sign_internal (Algorithm 19) of the concatenated `message`, using
/// `opt_rand` (addrnd, or PK.seed for the deterministic variant).
pub(crate) fn sign<S: Suite<N>, const N: usize>(
  p: &Params,
  key: &PrivateKey<'_, N>,
  message: &[&[u8]],
  opt_rand: &[u8; N],
  signature: &mut [[u8; N]],
) {
  debug_assert_eq!(p.n, N);
  debug_assert_eq!(signature.len().strict_mul(N), p.signature_len);
  let suite = S::new(key.pk_seed);
  let (r, rest) = signature.split_at_mut(1);
  let r = &mut r[0];
  S::prf_msg(key.sk_prf, opt_rand, message, r);
  let mut digest = [0u8; MAX_DIGEST];
  S::h_msg(r, key.pk_seed, key.pk_root, message, &mut digest[..p.m]);
  let (md, tree, leaf) = split_digest(p, &digest[..p.m]);
  let (fors, ht) = rest.split_at_mut(p.fors_nodes());
  let mut pk_fors = [0u8; N];
  fors_sign(
    p,
    &suite,
    md,
    key.sk_seed,
    &fors_address(tree, leaf),
    fors,
    &mut pk_fors,
  );
  ht_sign(p, &suite, &pk_fors, key.sk_seed, tree, leaf, ht);
  ct::zeroize(&mut digest);
}

/// slh_verify_internal (Algorithm 20) of the concatenated `message`.
pub(crate) fn verify<S: Suite<N>, const N: usize>(
  p: &Params,
  pk_seed: &[u8; N],
  pk_root: &[u8; N],
  message: &[&[u8]],
  signature: &[u8],
) -> bool {
  if signature.len() != p.signature_len {
    return false;
  }
  let (nodes, _) = signature.as_chunks::<N>();
  let Some((r, rest)) = nodes.split_first() else {
    return false;
  };
  let suite = S::new(pk_seed);
  let mut digest = [0u8; MAX_DIGEST];
  S::h_msg(r, pk_seed, pk_root, message, &mut digest[..p.m]);
  let (md, tree, leaf) = split_digest(p, &digest[..p.m]);
  let (fors, ht) = rest.split_at(p.fors_nodes());
  let mut pk_fors = [0u8; N];
  fors_public_key_from_signature(p, &suite, md, fors, &fors_address(tree, leaf), &mut pk_fors);
  ct::zeroize(&mut digest);
  ht_verify(p, &suite, &pk_fors, ht, tree, leaf, pk_root)
}
