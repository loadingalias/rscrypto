#![allow(clippy::indexing_slicing)] // Fixed-size FIPS buffers and public loop indices bound every access.

use crate::{
  auth::mlkem::MlKemError,
  hashes::crypto::{Sha3_256, Sha3_512, Shake128, Shake256},
  traits::{
    Digest, Xof,
    ct::{self},
  },
};

const N: usize = 256;
const K: usize = 3;
const K_U8: u8 = 3;
const Q: u16 = 3329;
const Q_U32: u32 = Q as u32;
const Q_HALF: u32 = Q_U32 / 2;
const Q_DIV_SHIFT: u32 = 36;
const Q_DIV_RECIP: u64 = 20_642_679;
const Q_COMPRESS_DIV_SHIFT: u32 = 33;
const Q_COMPRESS_DIV_RECIP: u64 = 2_580_335;
const INV_NTT_SCALE: u16 = 3303;

const SEED_BYTES: usize = 32;
const HASH_BYTES: usize = 32;
const SHARED_SECRET_BYTES: usize = 32;
const POLY_BYTES: usize = 384;
const DK_PKE_BYTES: usize = POLY_BYTES * K;
const EK_BYTES: usize = POLY_BYTES * K + SEED_BYTES;
const DK_BYTES: usize = DK_PKE_BYTES + EK_BYTES + HASH_BYTES + SEED_BYTES;
const CT_BYTES: usize = 1088;
const DU: usize = 10;
const DV: usize = 4;
const POLY_DU_BYTES: usize = 32 * DU;
const POLY_DV_BYTES: usize = 32 * DV;
const ETA2_RANDOM_BYTES: usize = 64 * 2;

type Poly = [u16; N];
type PolyVec = [Poly; K];

const ZETAS: [u16; 128] = [
  1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746, 296, 2447, 1339, 1476, 3046,
  56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821, 289, 331, 3253, 1756, 1197, 2304, 2277, 2055, 650,
  1977, 2513, 632, 2865, 33, 1320, 1915, 2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474,
  3110, 1227, 910, 17, 2761, 583, 2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050, 1703,
  1651, 2789, 1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641, 1584, 2298, 2037, 3220, 375,
  2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594, 2804, 1092, 403, 1026, 1143, 2150, 2775, 886, 1722,
  1212, 1874, 1029, 2110, 2935, 885, 2154,
];

const GAMMAS: [u16; 128] = [
  17, 3312, 2761, 568, 583, 2746, 2649, 680, 1637, 1692, 723, 2606, 2288, 1041, 1100, 2229, 1409, 1920, 2662, 667,
  3281, 48, 233, 3096, 756, 2573, 2156, 1173, 3015, 314, 3050, 279, 1703, 1626, 1651, 1678, 2789, 540, 1789, 1540,
  1847, 1482, 952, 2377, 1461, 1868, 2687, 642, 939, 2390, 2308, 1021, 2437, 892, 2388, 941, 733, 2596, 2337, 992, 268,
  3061, 641, 2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109, 375, 2954, 2549, 780, 2090, 1239, 1645, 1684, 1063,
  2266, 319, 3010, 2773, 556, 757, 2572, 2099, 1230, 561, 2768, 2466, 863, 2594, 735, 2804, 525, 1092, 2237, 403, 2926,
  1026, 2303, 1143, 2186, 2150, 1179, 2775, 554, 886, 2443, 1722, 1607, 1212, 2117, 1874, 1455, 1029, 2300, 2110, 1219,
  2935, 394, 885, 2444, 2154, 1175,
];

#[inline]
pub(super) fn validate_encapsulation_key(ek: &[u8; EK_BYTES]) -> Result<(), MlKemError> {
  let mut ok = 0xffu8;
  let mut decoded = [0u16; N];
  let mut encoded = [0u8; POLY_BYTES];

  for i in 0..K {
    let start = i.strict_mul(POLY_BYTES);
    let end = start.strict_add(POLY_BYTES);
    byte_decode::<12>(&ek[start..end], &mut decoded);
    byte_encode::<12>(&decoded, &mut encoded);
    ok &= ct_eq_mask(&encoded, &ek[start..end]);
  }

  ct::zeroize_words(&mut decoded);
  ct::zeroize(&mut encoded);

  if ok == 0xff {
    Ok(())
  } else {
    Err(MlKemError::InvalidEncapsulationKey)
  }
}

#[inline]
pub(super) fn validate_decapsulation_key(dk: &[u8; DK_BYTES]) -> Result<(), MlKemError> {
  let ek_start = DK_PKE_BYTES;
  let ek_end = ek_start.strict_add(EK_BYTES);
  let h_start = ek_end;
  let h_end = h_start.strict_add(HASH_BYTES);
  let expected = h(&dk[ek_start..ek_end]);

  if ct_eq_mask(&expected, &dk[h_start..h_end]) == 0xff {
    Ok(())
  } else {
    Err(MlKemError::InvalidDecapsulationKey)
  }
}

pub(super) fn keygen(random: &[u8; 64]) -> ([u8; EK_BYTES], [u8; DK_BYTES]) {
  let mut d = [0u8; SEED_BYTES];
  let mut z = [0u8; SEED_BYTES];
  d.copy_from_slice(&random[..SEED_BYTES]);
  z.copy_from_slice(&random[SEED_BYTES..]);

  let (ek, dk_pke) = pke_keygen(&d);
  let mut dk = [0u8; DK_BYTES];
  dk[..DK_PKE_BYTES].copy_from_slice(&dk_pke);
  dk[DK_PKE_BYTES..DK_PKE_BYTES.strict_add(EK_BYTES)].copy_from_slice(&ek);

  let ek_hash = h(&ek);
  let h_start = DK_PKE_BYTES.strict_add(EK_BYTES);
  dk[h_start..h_start.strict_add(HASH_BYTES)].copy_from_slice(&ek_hash);
  dk[h_start.strict_add(HASH_BYTES)..].copy_from_slice(&z);

  ct::zeroize(&mut d);
  ct::zeroize(&mut z);
  (ek, dk)
}

pub(super) fn encapsulate(ek: &[u8; EK_BYTES], m: &[u8; SEED_BYTES]) -> ([u8; CT_BYTES], [u8; SHARED_SECRET_BYTES]) {
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(m);
  input[SEED_BYTES..].copy_from_slice(&h(ek));

  let expanded = g(&input);
  let mut shared = [0u8; SHARED_SECRET_BYTES];
  let mut r = [0u8; SEED_BYTES];
  shared.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let ciphertext = pke_encrypt(ek, m, &r);
  ct::zeroize(&mut input);
  ct::zeroize(&mut r);
  (ciphertext, shared)
}

pub(super) fn decapsulate(dk: &[u8; DK_BYTES], c: &[u8; CT_BYTES]) -> [u8; SHARED_SECRET_BYTES] {
  let dk_pke = &dk[..DK_PKE_BYTES];
  let ek_start = DK_PKE_BYTES;
  let ek_end = ek_start.strict_add(EK_BYTES);
  let ek = &dk[ek_start..ek_end];
  let h_start = ek_end;
  let h_stored = &dk[h_start..h_start.strict_add(HASH_BYTES)];
  let z = &dk[h_start.strict_add(HASH_BYTES)..];

  let mut m_prime = pke_decrypt(dk_pke, c);
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(&m_prime);
  input[SEED_BYTES..].copy_from_slice(h_stored);

  let expanded = g(&input);
  let mut k_prime = [0u8; SHARED_SECRET_BYTES];
  let mut r_prime = [0u8; SEED_BYTES];
  k_prime.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r_prime.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let mut k_bar = j(z, c);
  let mut c_prime = pke_encrypt(ek, &m_prime, &r_prime);
  let mut match_mask = ct_eq_mask(c, &c_prime);
  let reject_mask = !match_mask;

  let mut shared = [0u8; SHARED_SECRET_BYTES];
  for i in 0..SHARED_SECRET_BYTES {
    shared[i] = (k_prime[i] & match_mask) | (k_bar[i] & reject_mask);
  }

  ct::zeroize(&mut m_prime);
  ct::zeroize(&mut input);
  ct::zeroize(&mut k_prime);
  ct::zeroize(&mut r_prime);
  ct::zeroize(&mut k_bar);
  ct::zeroize(&mut c_prime);
  ct::zeroize(core::slice::from_mut(&mut match_mask));

  shared
}

fn pke_keygen(d: &[u8; SEED_BYTES]) -> ([u8; EK_BYTES], [u8; DK_PKE_BYTES]) {
  let mut seed = [0u8; 33];
  seed[..SEED_BYTES].copy_from_slice(d);
  seed[SEED_BYTES] = K_U8;

  let expanded = g(&seed);
  let mut rho = [0u8; SEED_BYTES];
  let mut sigma = [0u8; SEED_BYTES];
  rho.copy_from_slice(&expanded[..SEED_BYTES]);
  sigma.copy_from_slice(&expanded[SEED_BYTES..]);

  let mut nonce = 0u8;
  let mut s = [[0u16; N]; K];
  let mut e = [[0u16; N]; K];
  for poly in &mut s {
    sample_noise_eta2(&sigma, nonce, poly);
    nonce = nonce.wrapping_add(1);
  }
  for poly in &mut e {
    sample_noise_eta2(&sigma, nonce, poly);
    nonce = nonce.wrapping_add(1);
  }

  let mut s_hat = s;
  let mut e_hat = e;
  for poly in &mut s_hat {
    ntt(poly);
  }
  for poly in &mut e_hat {
    ntt(poly);
  }

  let mut t_hat = [[0u16; N]; K];
  for i in 0..K {
    t_hat[i] = e_hat[i];
    for (j, s_j) in s_hat.iter().enumerate() {
      let mut a_ij = sample_ntt(&rho, j as u8, i as u8);
      let mut product = multiply_ntts(&a_ij, s_j);
      poly_add_assign(&mut t_hat[i], &product);
      zeroize_poly(&mut a_ij);
      zeroize_poly(&mut product);
    }
  }

  let mut ek = [0u8; EK_BYTES];
  for (i, poly) in t_hat.iter().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_encode::<12>(poly, &mut ek[start..start.strict_add(POLY_BYTES)]);
  }
  ek[DK_PKE_BYTES..].copy_from_slice(&rho);

  let mut dk_pke = [0u8; DK_PKE_BYTES];
  for (i, poly) in s_hat.iter().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_encode::<12>(poly, &mut dk_pke[start..start.strict_add(POLY_BYTES)]);
  }

  ct::zeroize(&mut seed);
  ct::zeroize(&mut sigma);
  zeroize_polyvec(&mut s);
  zeroize_polyvec(&mut e);
  zeroize_polyvec(&mut s_hat);
  zeroize_polyvec(&mut e_hat);
  (ek, dk_pke)
}

fn pke_encrypt(ek: &[u8], m: &[u8; SEED_BYTES], r: &[u8; SEED_BYTES]) -> [u8; CT_BYTES] {
  let mut t_hat = [[0u16; N]; K];
  for (i, poly) in t_hat.iter_mut().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_decode::<12>(&ek[start..start.strict_add(POLY_BYTES)], poly);
  }

  let mut rho = [0u8; SEED_BYTES];
  rho.copy_from_slice(&ek[DK_PKE_BYTES..DK_PKE_BYTES.strict_add(SEED_BYTES)]);

  let mut nonce = 0u8;
  let mut y = [[0u16; N]; K];
  let mut e1 = [[0u16; N]; K];
  for poly in &mut y {
    sample_noise_eta2(r, nonce, poly);
    nonce = nonce.wrapping_add(1);
  }
  for poly in &mut e1 {
    sample_noise_eta2(r, nonce, poly);
    nonce = nonce.wrapping_add(1);
  }
  let mut e2 = [0u16; N];
  sample_noise_eta2(r, nonce, &mut e2);

  let mut y_hat = y;
  for poly in &mut y_hat {
    ntt(poly);
  }

  let mut u = [[0u16; N]; K];
  for i in 0..K {
    let mut acc = [0u16; N];
    for (j, y_j) in y_hat.iter().enumerate() {
      let mut a_ji = sample_ntt(&rho, i as u8, j as u8);
      let mut product = multiply_ntts(&a_ji, y_j);
      poly_add_assign(&mut acc, &product);
      zeroize_poly(&mut a_ji);
      zeroize_poly(&mut product);
    }
    inverse_ntt(&mut acc);
    poly_add_assign(&mut acc, &e1[i]);
    u[i] = acc;
  }

  let mut v = [0u16; N];
  for i in 0..K {
    let mut product = multiply_ntts(&t_hat[i], &y_hat[i]);
    poly_add_assign(&mut v, &product);
    zeroize_poly(&mut product);
  }
  inverse_ntt(&mut v);
  poly_add_assign(&mut v, &e2);

  let mut mu = [0u16; N];
  byte_decode::<1>(m, &mut mu);
  decompress_poly_add_assign::<1>(&mu, &mut v);

  let mut ciphertext = [0u8; CT_BYTES];
  let mut compressed = [0u16; N];
  for (i, poly) in u.iter().enumerate() {
    compress_poly::<DU>(poly, &mut compressed);
    let start = i.strict_mul(POLY_DU_BYTES);
    byte_encode::<DU>(&compressed, &mut ciphertext[start..start.strict_add(POLY_DU_BYTES)]);
  }

  compress_poly::<DV>(&v, &mut compressed);
  byte_encode::<DV>(&compressed, &mut ciphertext[POLY_DU_BYTES.strict_mul(K)..]);

  zeroize_polyvec(&mut t_hat);
  ct::zeroize(&mut rho);
  zeroize_polyvec(&mut y);
  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  zeroize_poly(&mut v);
  zeroize_poly(&mut mu);
  zeroize_poly(&mut compressed);
  ciphertext
}

fn pke_decrypt(dk_pke: &[u8], c: &[u8; CT_BYTES]) -> [u8; SEED_BYTES] {
  let mut u = [[0u16; N]; K];
  let mut decoded = [0u16; N];
  for (i, poly) in u.iter_mut().enumerate() {
    let start = i.strict_mul(POLY_DU_BYTES);
    byte_decode::<DU>(&c[start..start.strict_add(POLY_DU_BYTES)], &mut decoded);
    decompress_poly::<DU>(&decoded, poly);
  }

  let c2_start = POLY_DU_BYTES.strict_mul(K);
  let mut v_prime = [0u16; N];
  byte_decode::<DV>(&c[c2_start..c2_start.strict_add(POLY_DV_BYTES)], &mut decoded);
  decompress_poly::<DV>(&decoded, &mut v_prime);

  let mut s_hat = [[0u16; N]; K];
  for (i, poly) in s_hat.iter_mut().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_decode::<12>(&dk_pke[start..start.strict_add(POLY_BYTES)], poly);
  }

  for poly in &mut u {
    ntt(poly);
  }

  let mut acc = [0u16; N];
  for i in 0..K {
    let mut product = multiply_ntts(&s_hat[i], &u[i]);
    poly_add_assign(&mut acc, &product);
    zeroize_poly(&mut product);
  }
  inverse_ntt(&mut acc);

  let mut w = [0u16; N];
  for i in 0..N {
    w[i] = sub_mod(v_prime[i], acc[i]);
  }

  let mut compressed = [0u16; N];
  compress_poly::<1>(&w, &mut compressed);
  let mut message = [0u8; SEED_BYTES];
  byte_encode::<1>(&compressed, &mut message);

  zeroize_polyvec(&mut u);
  zeroize_poly(&mut decoded);
  zeroize_poly(&mut v_prime);
  zeroize_polyvec(&mut s_hat);
  zeroize_poly(&mut acc);
  zeroize_poly(&mut w);
  zeroize_poly(&mut compressed);
  message
}

fn sample_ntt(rho: &[u8; SEED_BYTES], j: u8, i: u8) -> Poly {
  let mut xof = Shake128::new();
  xof.update(rho);
  xof.update(&[j, i]);
  let mut reader = xof.finalize_xof();

  let mut out = [0u16; N];
  let mut filled = 0usize;
  while filled < N {
    let mut buf = [0u8; 3];
    reader.squeeze(&mut buf);

    let d1 = u16::from(buf[0]) | (u16::from(buf[1] & 0x0f) << 8);
    let d2 = (u16::from(buf[1]) >> 4) | (u16::from(buf[2]) << 4);

    if d1 < Q {
      out[filled] = d1;
      filled = filled.strict_add(1);
    }
    if d2 < Q && filled < N {
      out[filled] = d2;
      filled = filled.strict_add(1);
    }
  }
  out
}

fn sample_noise_eta2(seed: &[u8; SEED_BYTES], nonce: u8, out: &mut Poly) {
  let mut buf = [0u8; ETA2_RANDOM_BYTES];
  prf_eta2(seed, nonce, &mut buf);
  sample_poly_cbd_eta2(&buf, out);
  ct::zeroize(&mut buf);
}

fn sample_poly_cbd_eta2(input: &[u8; ETA2_RANDOM_BYTES], out: &mut Poly) {
  for (i, byte) in input.iter().copied().enumerate() {
    let x0 = (byte & 1).strict_add((byte >> 1) & 1);
    let y0 = ((byte >> 2) & 1).strict_add((byte >> 3) & 1);
    let x1 = ((byte >> 4) & 1).strict_add((byte >> 5) & 1);
    let y1 = ((byte >> 6) & 1).strict_add((byte >> 7) & 1);
    out[i.strict_mul(2)] = small_signed_to_mod_q(i16::from(x0) - i16::from(y0));
    out[i.strict_mul(2).strict_add(1)] = small_signed_to_mod_q(i16::from(x1) - i16::from(y1));
  }
}

fn ntt(poly: &mut Poly) {
  let mut zeta_index = 1usize;
  let mut len = 128usize;
  while len >= 2 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS[zeta_index];
      zeta_index = zeta_index.strict_add(1);
      for j in start..start.strict_add(len) {
        let t = mul_mod(zeta, poly[j.strict_add(len)]);
        let u = poly[j];
        poly[j.strict_add(len)] = sub_mod(u, t);
        poly[j] = add_mod(u, t);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len >>= 1;
  }
}

fn inverse_ntt(poly: &mut Poly) {
  let mut zeta_index = 127usize;
  let mut len = 2usize;
  while len <= 128 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS[zeta_index];
      zeta_index = zeta_index.strict_sub(1);
      for j in start..start.strict_add(len) {
        let t = poly[j];
        let u = poly[j.strict_add(len)];
        poly[j] = add_mod(t, u);
        poly[j.strict_add(len)] = mul_mod(zeta, sub_mod(u, t));
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len <<= 1;
  }

  for coeff in poly {
    *coeff = mul_mod(*coeff, INV_NTT_SCALE);
  }
}

fn multiply_ntts(a: &Poly, b: &Poly) -> Poly {
  let mut out = [0u16; N];
  for (i, &gamma) in GAMMAS.iter().enumerate() {
    let j = i.strict_mul(2);
    let (c0, c1) = base_case_multiply(a[j], a[j.strict_add(1)], b[j], b[j.strict_add(1)], gamma);
    out[j] = c0;
    out[j.strict_add(1)] = c1;
  }
  out
}

#[inline]
fn base_case_multiply(a0: u16, a1: u16, b0: u16, b1: u16, gamma: u16) -> (u16, u16) {
  let a0b0 = mul_mod(a0, b0);
  let a1b1_gamma = mul_mod(mul_mod(a1, b1), gamma);
  let c0 = add_mod(a0b0, a1b1_gamma);
  let c1 = add_mod(mul_mod(a0, b1), mul_mod(a1, b0));
  (c0, c1)
}

#[inline]
fn poly_add_assign(lhs: &mut Poly, rhs: &Poly) {
  for i in 0..N {
    lhs[i] = add_mod(lhs[i], rhs[i]);
  }
}

fn compress_poly<const D: usize>(input: &Poly, out: &mut Poly) {
  for i in 0..N {
    let numerator = (u32::from(input[i]) << D) + Q_HALF;
    out[i] = (div_q_compress_u32(numerator) & ((1u32 << D) - 1)) as u16;
  }
}

fn decompress_poly<const D: usize>(input: &Poly, out: &mut Poly) {
  for i in 0..N {
    out[i] = decompress_value::<D>(input[i]);
  }
}

fn decompress_poly_add_assign<const D: usize>(input: &Poly, out: &mut Poly) {
  for i in 0..N {
    out[i] = add_mod(out[i], decompress_value::<D>(input[i]));
  }
}

#[inline]
fn decompress_value<const D: usize>(value: u16) -> u16 {
  (((Q_U32 * u32::from(value)) + (1u32 << (D - 1))) >> D) as u16
}

fn byte_encode<const D: usize>(input: &Poly, out: &mut [u8]) {
  debug_assert_eq!(out.len(), 32 * D);

  match D {
    1 => byte_encode_1(input, out),
    4 => byte_encode_4(input, out),
    10 => byte_encode_10(input, out),
    12 => byte_encode_12(input, out),
    _ => unreachable!("unsupported ML-KEM byte encoding width"),
  }
}

fn byte_decode<const D: usize>(input: &[u8], out: &mut Poly) {
  debug_assert_eq!(input.len(), 32 * D);

  match D {
    1 => byte_decode_1(input, out),
    4 => byte_decode_4(input, out),
    10 => byte_decode_10(input, out),
    12 => byte_decode_12(input, out),
    _ => unreachable!("unsupported ML-KEM byte decoding width"),
  }
}

fn byte_encode_1(input: &Poly, out: &mut [u8]) {
  for (i, byte) in out.iter_mut().enumerate() {
    let start = i.strict_mul(8);
    let mut packed = 0u8;
    for bit in 0..8 {
      packed |= ((input[start.strict_add(bit)] & 1) as u8) << bit;
    }
    *byte = packed;
  }
}

fn byte_decode_1(input: &[u8], out: &mut Poly) {
  for (i, byte) in input.iter().copied().enumerate() {
    let start = i.strict_mul(8);
    for bit in 0..8 {
      out[start.strict_add(bit)] = u16::from((byte >> bit) & 1);
    }
  }
}

fn byte_encode_4(input: &Poly, out: &mut [u8]) {
  for (i, byte) in out.iter_mut().enumerate() {
    let j = i.strict_mul(2);
    *byte = ((input[j] & 0x0f) | ((input[j.strict_add(1)] & 0x0f) << 4)) as u8;
  }
}

fn byte_decode_4(input: &[u8], out: &mut Poly) {
  for (i, byte) in input.iter().copied().enumerate() {
    let j = i.strict_mul(2);
    out[j] = u16::from(byte & 0x0f);
    out[j.strict_add(1)] = u16::from(byte >> 4);
  }
}

fn byte_encode_10(input: &Poly, out: &mut [u8]) {
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(5);
    let t0 = input[j] & 0x03ff;
    let t1 = input[j.strict_add(1)] & 0x03ff;
    let t2 = input[j.strict_add(2)] & 0x03ff;
    let t3 = input[j.strict_add(3)] & 0x03ff;

    out[k] = t0 as u8;
    out[k.strict_add(1)] = ((t0 >> 8) | (t1 << 2)) as u8;
    out[k.strict_add(2)] = ((t1 >> 6) | (t2 << 4)) as u8;
    out[k.strict_add(3)] = ((t2 >> 4) | (t3 << 6)) as u8;
    out[k.strict_add(4)] = (t3 >> 2) as u8;
  }
}

fn byte_decode_10(input: &[u8], out: &mut Poly) {
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(5);
    let b0 = u16::from(input[k]);
    let b1 = u16::from(input[k.strict_add(1)]);
    let b2 = u16::from(input[k.strict_add(2)]);
    let b3 = u16::from(input[k.strict_add(3)]);
    let b4 = u16::from(input[k.strict_add(4)]);

    out[j] = b0 | ((b1 & 0x03) << 8);
    out[j.strict_add(1)] = (b1 >> 2) | ((b2 & 0x0f) << 6);
    out[j.strict_add(2)] = (b2 >> 4) | ((b3 & 0x3f) << 4);
    out[j.strict_add(3)] = (b3 >> 6) | (b4 << 2);
  }
}

fn byte_encode_12(input: &Poly, out: &mut [u8]) {
  for i in 0usize..128 {
    let j = i.strict_mul(2);
    let k = i.strict_mul(3);
    let t0 = input[j];
    let t1 = input[j.strict_add(1)];

    out[k] = t0 as u8;
    out[k.strict_add(1)] = ((t0 >> 8) | (t1 << 4)) as u8;
    out[k.strict_add(2)] = (t1 >> 4) as u8;
  }
}

fn byte_decode_12(input: &[u8], out: &mut Poly) {
  for i in 0usize..128 {
    let j = i.strict_mul(2);
    let k = i.strict_mul(3);
    let b0 = u16::from(input[k]);
    let b1 = u16::from(input[k.strict_add(1)]);
    let b2 = u16::from(input[k.strict_add(2)]);

    out[j] = sub_if_ge_q(b0 | ((b1 & 0x0f) << 8));
    out[j.strict_add(1)] = sub_if_ge_q((b1 >> 4) | (b2 << 4));
  }
}

#[inline]
fn add_mod(a: u16, b: u16) -> u16 {
  let sum = u32::from(a) + u32::from(b);
  let reduced = sum.wrapping_sub(Q_U32);
  let borrow = reduced >> 31;
  reduced.wrapping_add(borrow * Q_U32) as u16
}

#[inline]
fn sub_mod(a: u16, b: u16) -> u16 {
  let diff = u32::from(a).wrapping_sub(u32::from(b));
  let borrow = diff >> 31;
  diff.wrapping_add(borrow * Q_U32) as u16
}

#[inline]
fn sub_if_ge_q(value: u16) -> u16 {
  let reduced = u32::from(value).wrapping_sub(Q_U32);
  let borrow = reduced >> 31;
  reduced.wrapping_add(borrow * Q_U32) as u16
}

#[inline]
fn mul_mod(a: u16, b: u16) -> u16 {
  reduce_u32(u32::from(a) * u32::from(b))
}

#[inline]
fn reduce_u32(value: u32) -> u16 {
  let quotient = div_q_u32(value);
  value.wrapping_sub(quotient * Q_U32) as u16
}

#[inline]
fn div_q_u32(value: u32) -> u32 {
  ((u64::from(value) * Q_DIV_RECIP) >> Q_DIV_SHIFT) as u32
}

#[inline]
fn div_q_compress_u32(value: u32) -> u32 {
  ((u64::from(value) * Q_COMPRESS_DIV_RECIP) >> Q_COMPRESS_DIV_SHIFT) as u32
}

#[inline]
fn small_signed_to_mod_q(value: i16) -> u16 {
  let value = i32::from(value);
  (value + ((value >> 31) & i32::from(Q))) as u16
}

fn h(input: &[u8]) -> [u8; HASH_BYTES] {
  Sha3_256::digest(input)
}

fn g(input: &[u8]) -> [u8; 64] {
  Sha3_512::digest(input)
}

fn j(z: &[u8], c: &[u8; CT_BYTES]) -> [u8; SHARED_SECRET_BYTES] {
  let mut xof = Shake256::new();
  xof.update(z);
  xof.update(c);
  let mut reader = xof.finalize_xof();
  let mut out = [0u8; SHARED_SECRET_BYTES];
  reader.squeeze(&mut out);
  out
}

fn prf_eta2(seed: &[u8; SEED_BYTES], nonce: u8, out: &mut [u8; ETA2_RANDOM_BYTES]) {
  let mut xof = Shake256::new();
  xof.update(seed);
  xof.update(&[nonce]);
  let mut reader = xof.finalize_xof();
  reader.squeeze(out);
}

fn ct_eq_mask(a: &[u8], b: &[u8]) -> u8 {
  debug_assert_eq!(a.len(), b.len());

  let mut diff = 0u8;
  for i in 0..a.len() {
    diff |= a[i] ^ b[i];
  }

  let diff = u32::from(diff);
  let nonzero = (diff | diff.wrapping_neg()) >> 31;
  0u8.wrapping_sub((nonzero ^ 1) as u8)
}

#[inline]
fn zeroize_poly(poly: &mut Poly) {
  ct::zeroize_words(poly);
}

#[inline]
fn zeroize_polyvec(polyvec: &mut PolyVec) {
  for poly in polyvec {
    zeroize_poly(poly);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn ntt_round_trip_preserves_polynomial() {
    let mut poly = [0u16; N];
    for (i, coeff) in poly.iter_mut().enumerate() {
      *coeff = ((i.strict_mul(17).strict_add(91)) as u16) % Q;
    }
    let original = poly;

    ntt(&mut poly);
    inverse_ntt(&mut poly);

    assert_eq!(poly, original);
  }

  #[test]
  fn byte_encode_decode_round_trips_supported_widths() {
    let mut poly = [0u16; N];
    for (i, coeff) in poly.iter_mut().enumerate() {
      *coeff = ((i.strict_mul(19).strict_add(7)) as u16) % Q;
    }

    let mut encoded_1 = [0u8; 32];
    let mut decoded_1 = [0u16; N];
    byte_encode::<1>(&poly, &mut encoded_1);
    byte_decode::<1>(&encoded_1, &mut decoded_1);
    for i in 0..N {
      assert_eq!(decoded_1[i], poly[i] & 1);
    }

    let mut encoded_4 = [0u8; 128];
    let mut decoded_4 = [0u16; N];
    byte_encode::<4>(&poly, &mut encoded_4);
    byte_decode::<4>(&encoded_4, &mut decoded_4);
    for i in 0..N {
      assert_eq!(decoded_4[i], poly[i] & 0x0f);
    }

    let mut encoded_10 = [0u8; 320];
    let mut decoded_10 = [0u16; N];
    byte_encode::<10>(&poly, &mut encoded_10);
    byte_decode::<10>(&encoded_10, &mut decoded_10);
    for i in 0..N {
      assert_eq!(decoded_10[i], poly[i] & 0x03ff);
    }

    let mut encoded_12 = [0u8; 384];
    let mut decoded_12 = [0u16; N];
    byte_encode::<12>(&poly, &mut encoded_12);
    byte_decode::<12>(&encoded_12, &mut decoded_12);
    assert_eq!(decoded_12, poly);
  }

  #[test]
  fn decompressed_values_recompress_to_original() {
    for d in [1usize, 4, 10] {
      let max = 1u16 << d;
      for y in 0..max {
        let x = ((Q_U32 * u32::from(y)) + (1u32 << (d - 1))) >> d;
        let compressed = (div_q_compress_u32((x << d) + Q_HALF) & ((1u32 << d) - 1)) as u16;
        assert_eq!(compressed, y, "d={d} y={y}");
      }
    }
  }
}
