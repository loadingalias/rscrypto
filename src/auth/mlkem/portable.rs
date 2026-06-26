#![allow(clippy::indexing_slicing)] // Fixed-size FIPS buffers and public loop indices bound every access.

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
mod aarch64;
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
mod s390x;
#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
mod x86_64;

#[cfg(all(
  target_arch = "aarch64",
  not(any(target_os = "macos", target_os = "linux")),
  not(miri),
  not(feature = "portable-only")
))]
use core::arch::aarch64::vgetq_lane_u16;
#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
use core::arch::aarch64::{
  int16x4_t, int16x8_t, int32x4_t, uint16x4_t, uint16x8_t, uint16x8x2_t, vaddq_s16, vaddq_s32, vaddq_u16, vaddq_u32,
  vandq_s16, vandq_u16, vandq_u32, vcgeq_u16, vcgtq_u32, vcombine_s16, vcombine_u16, vcombine_u32, vdup_n_u32,
  vdupq_n_s16, vdupq_n_u16, vdupq_n_u32, vget_high_s16, vget_high_u16, vget_high_u32, vget_lane_u16, vget_low_s16,
  vget_low_u32, vld1_u16, vld1q_s16, vld1q_u16, vld2q_u16, vmovl_u16, vmovn_s32, vmovn_u32, vmovn_u64, vmul_n_s16,
  vmull_n_s16, vmull_s16, vmull_u32, vmulq_n_s16, vmulq_n_u16, vmulq_n_u32, vqdmulhq_n_s16, vreinterpretq_s16_u16,
  vreinterpretq_u16_s16, vshll_n_u16, vshrn_n_s32, vshrq_n_s16, vshrq_n_u32, vshrq_n_u64, vst1_u16, vst1q_u16,
  vst2q_u16, vsub_s16, vsubq_s16, vsubq_u16, vsubq_u32,
};
#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
use core::arch::aarch64::{
  uint32x2_t, uint32x4_t, vadd_s16, vadd_u16, vand_s16, vand_u16, vcge_u16, vcgt_u16, vcgtq_u16, vdup_n_s16,
  vdup_n_u16, vget_low_u16, vreinterpret_s16_u16, vreinterpret_u16_s16, vreinterpret_u32_u16, vreinterpretq_u16_u32,
  vreinterpretq_u32_u16, vset_lane_s16, vshr_n_s16, vsub_u16, vuzp1q_u32, vuzp2q_u32, vzip1_u32, vzip2_u32,
};
#[cfg(all(
  target_arch = "aarch64",
  not(any(target_os = "macos", target_os = "linux")),
  not(miri),
  not(feature = "portable-only")
))]
use core::arch::aarch64::{vget_high_u8, vget_low_u8, vld3q_u8, vmovl_u8, vorrq_u16, vshlq_n_u16, vshrq_n_u16};
#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
use core::arch::x86_64::{
  __m128i, __m256i, _mm_add_epi16, _mm_and_si128, _mm_cmpgt_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_mulhi_epi16,
  _mm_mullo_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi16,
  _mm256_add_epi32, _mm256_and_si256, _mm256_cmpgt_epi32, _mm256_cvtepi16_epi32, _mm256_loadu_si256,
  _mm256_mulhi_epi16, _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi32, _mm256_setzero_si256,
  _mm256_slli_epi32, _mm256_srai_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32,
};

use crate::{
  auth::mlkem::MlKemError,
  hashes::crypto::{Sha3_256, Sha3_512, Shake128, Shake128XofReader, Shake256, Shake256XofReader},
  traits::{
    Digest, Xof,
    ct::{self},
  },
};

const N: usize = 256;
const Q: u16 = 3329;
const Q_U32: u32 = Q as u32;
#[cfg(any(
  all(target_arch = "aarch64", not(miri), not(feature = "portable-only")),
  all(target_arch = "x86_64", not(miri), not(feature = "portable-only"))
))]
const Q_I16: i16 = Q as i16;
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
const Q_I32: i32 = Q as i32;
const Q_HALF: u32 = Q_U32 / 2;
#[cfg(test)]
const Q_DIV_SHIFT: u32 = 36;
#[cfg(test)]
const Q_DIV_RECIP: u64 = 20_642_679;
#[cfg(any(test, not(target_arch = "s390x")))]
const Q_COMPRESS_DIV_SHIFT: i32 = 33;
#[cfg(any(test, not(target_arch = "s390x")))]
const Q_COMPRESS_DIV_RECIP: u64 = 2_580_335;
const Q_MONT_INV_U16: u16 = 62_209;
const MONT_R_SQUARED_MOD_Q: i16 = 1353;
#[cfg(test)]
const INV_NTT_SCALE_MONT: i16 = 512;
const INV_NTT_PRODUCT_SCALE_MONT: i16 = 1441;

const SEED_BYTES: usize = 32;
const HASH_BYTES: usize = 32;
const SHARED_SECRET_BYTES: usize = 32;
const POLY_BYTES: usize = 384;
const SHAKE128_RATE_BYTES: usize = 168;
const ETA2_RANDOM_BYTES: usize = 128;
const ETA3_RANDOM_BYTES: usize = 192;
const SAMPLE_NTT_ACC_CHUNK_COEFFS: usize = 16;

type Poly = [u16; N];
type PolyVec<const K: usize> = [Poly; K];
type PolyMatrix<const K: usize> = [PolyVec<K>; K];

#[derive(Clone)]
pub(super) struct PreparedEncapsulationArithmetic<const K: usize> {
  t_hat: PolyVec<K>,
  #[cfg(any(test, feature = "diag"))]
  rho: [u8; SEED_BYTES],
  a_transpose_hat: PolyMatrix<K>,
}

#[derive(Clone)]
pub(super) struct PreparedDecapsulationArithmetic<const K: usize> {
  s_hat: PolyVec<K>,
  encapsulation: PreparedEncapsulationArithmetic<K>,
}

impl<const K: usize> Drop for PreparedDecapsulationArithmetic<K> {
  fn drop(&mut self) {
    zeroize_polyvec(&mut self.s_hat);
  }
}

const ZETAS_MONT: [i16; 128] = [
  -1044, -758, -359, -1517, 1493, 1422, 287, 202, -171, 622, 1577, 182, 962, -1202, -1474, 1468, 573, -1325, 264, 383,
  -829, 1458, -1602, -130, -681, 1017, 732, 608, -1542, 411, -205, -1571, 1223, 652, -552, 1015, -1293, 1491, -282,
  -1544, 516, -8, -320, -666, -1618, -1162, 126, 1469, -853, -90, -271, 830, 107, -1421, -247, -951, -398, 961, -1508,
  -725, 448, -1065, 677, -1275, -1103, 430, 555, 843, -1251, 871, 1550, 105, 422, 587, 177, -235, -291, -460, 1574,
  1653, -246, 778, 1159, -147, -777, 1483, -602, 1119, -1590, 644, -872, 349, 418, 329, -156, -75, 817, 1097, 603, 610,
  1322, -1285, -1465, 384, -1215, -136, 1218, -1335, -874, 220, -1187, -1659, -1185, -1530, -1278, 794, -1510, -854,
  -870, 478, -108, -308, 996, 991, 958, -1460, 1522, 1628,
];

#[cfg(test)]
const GAMMAS: [u16; 128] = [
  17, 3312, 2761, 568, 583, 2746, 2649, 680, 1637, 1692, 723, 2606, 2288, 1041, 1100, 2229, 1409, 1920, 2662, 667,
  3281, 48, 233, 3096, 756, 2573, 2156, 1173, 3015, 314, 3050, 279, 1703, 1626, 1651, 1678, 2789, 540, 1789, 1540,
  1847, 1482, 952, 2377, 1461, 1868, 2687, 642, 939, 2390, 2308, 1021, 2437, 892, 2388, 941, 733, 2596, 2337, 992, 268,
  3061, 641, 2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109, 375, 2954, 2549, 780, 2090, 1239, 1645, 1684, 1063,
  2266, 319, 3010, 2773, 556, 757, 2572, 2099, 1230, 561, 2768, 2466, 863, 2594, 735, 2804, 525, 1092, 2237, 403, 2926,
  1026, 2303, 1143, 2186, 2150, 1179, 2775, 554, 886, 2443, 1722, 1607, 1212, 2117, 1874, 1455, 1029, 2300, 2110, 1219,
  2935, 394, 885, 2444, 2154, 1175,
];

const GAMMAS_MONT: [i16; 128] = [
  -1103, 1103, 430, -430, 555, -555, 843, -843, -1251, 1251, 871, -871, 1550, -1550, 105, -105, 422, -422, 587, -587,
  177, -177, -235, 235, -291, 291, -460, 460, 1574, -1574, 1653, -1653, -246, 246, 778, -778, 1159, -1159, -147, 147,
  -777, 777, 1483, -1483, -602, 602, 1119, -1119, -1590, 1590, 644, -644, -872, 872, 349, -349, 418, -418, 329, -329,
  -156, 156, -75, 75, 817, -817, 1097, -1097, 603, -603, 610, -610, 1322, -1322, -1285, 1285, -1465, 1465, 384, -384,
  -1215, 1215, -136, 136, 1218, -1218, -1335, 1335, -874, 874, 220, -220, -1187, 1187, -1659, 1659, -1185, 1185, -1530,
  1530, -1278, 1278, 794, -794, -1510, 1510, -854, 854, -870, 870, 478, -478, -108, 108, -308, 308, 996, -996, 991,
  -991, 958, -958, -1460, 1460, 1522, -1522, 1628, -1628,
];

#[inline]
pub(super) fn validate_encapsulation_key<const K: usize, const EK_BYTES: usize>(
  ek: &[u8; EK_BYTES],
) -> Result<(), MlKemError> {
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

  if ok == 0xff {
    Ok(())
  } else {
    Err(MlKemError::InvalidEncapsulationKey)
  }
}

#[inline]
pub(super) fn validate_decapsulation_key<const DK_PKE_BYTES: usize, const EK_BYTES: usize, const DK_BYTES: usize>(
  dk: &[u8; DK_BYTES],
) -> Result<(), MlKemError> {
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

#[inline]
pub(super) fn encapsulation_key_hash<const EK_BYTES: usize>(ek: &[u8; EK_BYTES]) -> [u8; HASH_BYTES] {
  h(ek)
}

pub(super) fn validate_and_prepare_encapsulation_key<const K: usize, const EK_BYTES: usize>(
  ek: &[u8; EK_BYTES],
) -> Result<PreparedEncapsulationArithmetic<K>, MlKemError> {
  let mut ok = 0xffu8;
  let mut arithmetic = prepare_encapsulation_key::<K, EK_BYTES>(ek);
  let mut encoded = [0u8; POLY_BYTES];

  for i in 0..K {
    let start = i.strict_mul(POLY_BYTES);
    let end = start.strict_add(POLY_BYTES);
    byte_encode::<12>(&arithmetic.t_hat[i], &mut encoded);
    ok &= ct_eq_mask(&encoded, &ek[start..end]);
  }

  if ok == 0xff {
    Ok(arithmetic)
  } else {
    zeroize_polyvec(&mut arithmetic.t_hat);
    Err(MlKemError::InvalidEncapsulationKey)
  }
}

pub(super) fn validate_and_prepare_decapsulation_key<
  const K: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
  const DK_BYTES: usize,
>(
  dk: &[u8; DK_BYTES],
) -> Result<PreparedDecapsulationArithmetic<K>, MlKemError> {
  let ek_start = DK_PKE_BYTES;
  let ek_end = ek_start.strict_add(EK_BYTES);
  let h_start = ek_end;
  let h_end = h_start.strict_add(HASH_BYTES);
  let expected = h(&dk[ek_start..ek_end]);

  if ct_eq_mask(&expected, &dk[h_start..h_end]) != 0xff {
    return Err(MlKemError::InvalidDecapsulationKey);
  }

  let ek = match <&[u8; EK_BYTES]>::try_from(&dk[ek_start..ek_end]) {
    Ok(ek) => ek,
    Err(_) => unreachable!("ML-KEM decapsulation key layout must include an encapsulation key"),
  };
  Ok(PreparedDecapsulationArithmetic {
    s_hat: prepare_decapsulation_key::<K, DK_PKE_BYTES, DK_BYTES>(dk),
    encapsulation: prepare_encapsulation_key::<K, EK_BYTES>(ek),
  })
}

fn prepare_encapsulation_key<const K: usize, const EK_BYTES: usize>(
  ek: &[u8; EK_BYTES],
) -> PreparedEncapsulationArithmetic<K> {
  let mut t_hat = [[0u16; N]; K];
  for (i, poly) in t_hat.iter_mut().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_decode::<12>(&ek[start..start.strict_add(POLY_BYTES)], poly);
  }

  let mut rho = [0u8; SEED_BYTES];
  let rho_start = K.strict_mul(POLY_BYTES);
  rho.copy_from_slice(&ek[rho_start..rho_start.strict_add(SEED_BYTES)]);

  let a_transpose_hat = prepare_matrix_transpose::<K>(&rho);

  PreparedEncapsulationArithmetic {
    t_hat,
    #[cfg(any(test, feature = "diag"))]
    rho,
    a_transpose_hat,
  }
}

fn prepare_matrix_transpose<const K: usize>(rho: &[u8; SEED_BYTES]) -> PolyMatrix<K> {
  let mut a_transpose_hat = [[[0u16; N]; K]; K];
  for (i, row) in a_transpose_hat.iter_mut().enumerate() {
    for (j, poly) in row.iter_mut().enumerate() {
      sample_ntt_into(rho, i as u8, j as u8, poly);
    }
  }
  a_transpose_hat
}

fn prepare_decapsulation_key<const K: usize, const DK_PKE_BYTES: usize, const DK_BYTES: usize>(
  dk: &[u8; DK_BYTES],
) -> PolyVec<K> {
  prepare_decapsulation_key_slice::<K, DK_PKE_BYTES>(&dk[..DK_PKE_BYTES])
}

fn prepare_decapsulation_key_slice<const K: usize, const DK_PKE_BYTES: usize>(dk_pke: &[u8]) -> PolyVec<K> {
  debug_assert_eq!(dk_pke.len(), DK_PKE_BYTES);
  let mut s_hat = [[0u16; N]; K];
  for (i, poly) in s_hat.iter_mut().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_decode::<12>(&dk_pke[start..start.strict_add(POLY_BYTES)], poly);
  }
  s_hat
}

pub(super) fn keygen<
  const K: usize,
  const K_U8: u8,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
  const DK_BYTES: usize,
>(
  random: &[u8; 64],
) -> ([u8; EK_BYTES], [u8; DK_BYTES]) {
  let mut d = [0u8; SEED_BYTES];
  let mut z = [0u8; SEED_BYTES];
  d.copy_from_slice(&random[..SEED_BYTES]);
  z.copy_from_slice(&random[SEED_BYTES..]);

  let (ek, mut dk_pke) = pke_keygen::<K, K_U8, ETA1_RANDOM_BYTES, DK_PKE_BYTES, EK_BYTES>(&d);
  let mut dk = [0u8; DK_BYTES];
  dk[..DK_PKE_BYTES].copy_from_slice(&dk_pke);
  dk[DK_PKE_BYTES..DK_PKE_BYTES.strict_add(EK_BYTES)].copy_from_slice(&ek);

  let ek_hash = h(&ek);
  let h_start = DK_PKE_BYTES.strict_add(EK_BYTES);
  dk[h_start..h_start.strict_add(HASH_BYTES)].copy_from_slice(&ek_hash);
  dk[h_start.strict_add(HASH_BYTES)..].copy_from_slice(&z);

  ct::zeroize(&mut d);
  ct::zeroize(&mut z);
  ct::zeroize(&mut dk_pke);
  (ek, dk)
}

pub(super) fn encapsulate<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const _DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  ek: &[u8; EK_BYTES],
  m: &[u8; SEED_BYTES],
) -> ([u8; CT_BYTES], [u8; SHARED_SECRET_BYTES]) {
  let ek_hash = encapsulation_key_hash(ek);
  let arithmetic = prepare_encapsulation_key::<K, EK_BYTES>(ek);
  encapsulate_prepared::<K, ETA1_RANDOM_BYTES, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(&arithmetic, &ek_hash, m)
}

pub(super) fn encapsulate_prepared<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  ek: &PreparedEncapsulationArithmetic<K>,
  ek_hash: &[u8; HASH_BYTES],
  m: &[u8; SEED_BYTES],
) -> ([u8; CT_BYTES], [u8; SHARED_SECRET_BYTES]) {
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(m);
  input[SEED_BYTES..].copy_from_slice(ek_hash);

  let mut expanded = g(&input);
  let mut shared = [0u8; SHARED_SECRET_BYTES];
  let mut r = [0u8; SEED_BYTES];
  shared.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let ciphertext =
    pke_encrypt_prepared::<K, ETA1_RANDOM_BYTES, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(ek, m, &r);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut r);
  (ciphertext, shared)
}

pub(super) fn encapsulate_prepared_512(
  ek: &PreparedEncapsulationArithmetic<2>,
  ek_hash: &[u8; HASH_BYTES],
  m: &[u8; SEED_BYTES],
) -> ([u8; 768], [u8; SHARED_SECRET_BYTES]) {
  encapsulate_prepared::<2, 192, 768, 10, 4, 320, 128>(ek, ek_hash, m)
}

pub(super) fn encapsulate_prepared_768(
  ek: &PreparedEncapsulationArithmetic<3>,
  ek_hash: &[u8; HASH_BYTES],
  m: &[u8; SEED_BYTES],
) -> ([u8; 1088], [u8; SHARED_SECRET_BYTES]) {
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(m);
  input[SEED_BYTES..].copy_from_slice(ek_hash);

  let mut expanded = g(&input);
  let mut shared = [0u8; SHARED_SECRET_BYTES];
  let mut r = [0u8; SEED_BYTES];
  shared.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let ciphertext = pke_encrypt_prepared_768(ek, m, &r);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut r);
  (ciphertext, shared)
}

pub(super) fn encapsulate_prepared_1024(
  ek: &PreparedEncapsulationArithmetic<4>,
  ek_hash: &[u8; HASH_BYTES],
  m: &[u8; SEED_BYTES],
) -> ([u8; 1568], [u8; SHARED_SECRET_BYTES]) {
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(m);
  input[SEED_BYTES..].copy_from_slice(ek_hash);

  let mut expanded = g(&input);
  let mut shared = [0u8; SHARED_SECRET_BYTES];
  let mut r = [0u8; SEED_BYTES];
  shared.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let ciphertext = pke_encrypt_prepared_1024(ek, m, &r);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut r);
  (ciphertext, shared)
}

pub(super) fn decapsulate_prepared_512(
  dk: &[u8; 1632],
  prepared: &PreparedDecapsulationArithmetic<2>,
  c: &[u8; 768],
) -> [u8; SHARED_SECRET_BYTES] {
  let h_start = 768usize.strict_add(800);
  let h_stored = &dk[h_start..h_start.strict_add(HASH_BYTES)];
  let z = &dk[h_start.strict_add(HASH_BYTES)..];

  let mut m_prime = pke_decrypt_prepared::<2, 768, 10, 4, 320, 128>(&prepared.s_hat, c);
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(&m_prime);
  input[SEED_BYTES..].copy_from_slice(h_stored);

  let mut expanded = g(&input);
  let mut k_prime = [0u8; SHARED_SECRET_BYTES];
  let mut r_prime = [0u8; SEED_BYTES];
  k_prime.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r_prime.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let mut k_bar = j(z, c);
  let mut match_mask = pke_encrypt_prepared_512_compare(&prepared.encapsulation, &m_prime, &r_prime, c);
  let reject_mask = !match_mask;

  let mut shared = [0u8; SHARED_SECRET_BYTES];
  for i in 0..SHARED_SECRET_BYTES {
    shared[i] = (k_prime[i] & match_mask) | (k_bar[i] & reject_mask);
  }

  ct::zeroize(&mut m_prime);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut k_prime);
  ct::zeroize(&mut r_prime);
  ct::zeroize(&mut k_bar);
  ct::zeroize(core::slice::from_mut(&mut match_mask));

  shared
}

pub(super) fn decapsulate_prepared_768(
  dk: &[u8; 2400],
  prepared: &PreparedDecapsulationArithmetic<3>,
  c: &[u8; 1088],
) -> [u8; SHARED_SECRET_BYTES] {
  let h_start = 1152usize.strict_add(1184);
  let h_stored = &dk[h_start..h_start.strict_add(HASH_BYTES)];
  let z = &dk[h_start.strict_add(HASH_BYTES)..];

  let mut m_prime = pke_decrypt_prepared::<3, 1088, 10, 4, 320, 128>(&prepared.s_hat, c);
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(&m_prime);
  input[SEED_BYTES..].copy_from_slice(h_stored);

  let mut expanded = g(&input);
  let mut k_prime = [0u8; SHARED_SECRET_BYTES];
  let mut r_prime = [0u8; SEED_BYTES];
  k_prime.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r_prime.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let mut k_bar = j(z, c);
  let mut match_mask = pke_encrypt_prepared_768_compare(&prepared.encapsulation, &m_prime, &r_prime, c);
  let reject_mask = !match_mask;

  let mut shared = [0u8; SHARED_SECRET_BYTES];
  for i in 0..SHARED_SECRET_BYTES {
    shared[i] = (k_prime[i] & match_mask) | (k_bar[i] & reject_mask);
  }

  ct::zeroize(&mut m_prime);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut k_prime);
  ct::zeroize(&mut r_prime);
  ct::zeroize(&mut k_bar);
  ct::zeroize(core::slice::from_mut(&mut match_mask));

  shared
}

pub(super) fn decapsulate_prepared_1024(
  dk: &[u8; 3168],
  prepared: &PreparedDecapsulationArithmetic<4>,
  c: &[u8; 1568],
) -> [u8; SHARED_SECRET_BYTES] {
  let h_start = 1536usize.strict_add(1568);
  let h_stored = &dk[h_start..h_start.strict_add(HASH_BYTES)];
  let z = &dk[h_start.strict_add(HASH_BYTES)..];

  let mut m_prime = pke_decrypt_prepared::<4, 1568, 11, 5, 352, 160>(&prepared.s_hat, c);
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(&m_prime);
  input[SEED_BYTES..].copy_from_slice(h_stored);

  let mut expanded = g(&input);
  let mut k_prime = [0u8; SHARED_SECRET_BYTES];
  let mut r_prime = [0u8; SEED_BYTES];
  k_prime.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r_prime.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let mut k_bar = j(z, c);
  let mut match_mask = pke_encrypt_prepared_1024_compare(&prepared.encapsulation, &m_prime, &r_prime, c);
  let reject_mask = !match_mask;

  let mut shared = [0u8; SHARED_SECRET_BYTES];
  for i in 0..SHARED_SECRET_BYTES {
    shared[i] = (k_prime[i] & match_mask) | (k_bar[i] & reject_mask);
  }

  ct::zeroize(&mut m_prime);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut k_prime);
  ct::zeroize(&mut r_prime);
  ct::zeroize(&mut k_bar);
  ct::zeroize(core::slice::from_mut(&mut match_mask));

  shared
}

pub(super) fn decapsulate<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
  const DK_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  dk: &[u8; DK_BYTES],
  c: &[u8; CT_BYTES],
) -> [u8; SHARED_SECRET_BYTES] {
  let dk_pke = &dk[..DK_PKE_BYTES];
  let ek_start = DK_PKE_BYTES;
  let ek_end = ek_start.strict_add(EK_BYTES);
  let ek = match <&[u8; EK_BYTES]>::try_from(&dk[ek_start..ek_end]) {
    Ok(ek) => ek,
    Err(_) => unreachable!("ML-KEM decapsulation key layout must include an encapsulation key"),
  };
  let h_start = ek_end;
  let h_stored = &dk[h_start..h_start.strict_add(HASH_BYTES)];
  let z = &dk[h_start.strict_add(HASH_BYTES)..];

  let mut m_prime = pke_decrypt::<K, DK_PKE_BYTES, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(dk_pke, c);
  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(&m_prime);
  input[SEED_BYTES..].copy_from_slice(h_stored);

  let mut expanded = g(&input);
  let mut k_prime = [0u8; SHARED_SECRET_BYTES];
  let mut r_prime = [0u8; SEED_BYTES];
  k_prime.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r_prime.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let mut k_bar = j(z, c);
  let mut c_prime = pke_encrypt::<K, ETA1_RANDOM_BYTES, EK_BYTES, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(
    ek, &m_prime, &r_prime,
  );
  let mut match_mask = ct_eq_mask(c, &c_prime);
  let reject_mask = !match_mask;

  let mut shared = [0u8; SHARED_SECRET_BYTES];
  for i in 0..SHARED_SECRET_BYTES {
    shared[i] = (k_prime[i] & match_mask) | (k_bar[i] & reject_mask);
  }

  ct::zeroize(&mut m_prime);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut k_prime);
  ct::zeroize(&mut r_prime);
  ct::zeroize(&mut k_bar);
  ct::zeroize(&mut c_prime);
  ct::zeroize(core::slice::from_mut(&mut match_mask));

  shared
}

fn pke_keygen<
  const K: usize,
  const K_U8: u8,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
>(
  d: &[u8; SEED_BYTES],
) -> ([u8; EK_BYTES], [u8; DK_PKE_BYTES]) {
  let mut seed = [0u8; 33];
  seed[..SEED_BYTES].copy_from_slice(d);
  seed[SEED_BYTES] = K_U8;

  let mut expanded = g(&seed);
  let mut rho = [0u8; SEED_BYTES];
  let mut sigma = [0u8; SEED_BYTES];
  rho.copy_from_slice(&expanded[..SEED_BYTES]);
  sigma.copy_from_slice(&expanded[SEED_BYTES..]);

  let keys = pke_keygen_expanded::<K, ETA1_RANDOM_BYTES, DK_PKE_BYTES, EK_BYTES>(&rho, &sigma);
  ct::zeroize(&mut seed);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut sigma);
  keys
}

fn pke_keygen_expanded<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
>(
  rho: &[u8; SEED_BYTES],
  sigma: &[u8; SEED_BYTES],
) -> ([u8; EK_BYTES], [u8; DK_PKE_BYTES]) {
  let mut s_hat = [[0u16; N]; K];
  let mut t_hat = [[0u16; N]; K];
  keygen_sample_noise::<K, ETA1_RANDOM_BYTES>(sigma, &mut s_hat, &mut t_hat);
  keygen_ntt_noise::<K>(&mut s_hat, &mut t_hat);
  keygen_matrix_accumulate::<K>(rho, &s_hat, &mut t_hat);
  keygen_from_product_domain::<K>(&mut t_hat);

  let keys = keygen_encode::<K, DK_PKE_BYTES, EK_BYTES>(rho, &s_hat, &t_hat);
  zeroize_polyvec(&mut s_hat);
  zeroize_polyvec(&mut t_hat);
  keys
}

#[inline(always)]
fn keygen_sample_noise<const K: usize, const ETA1_RANDOM_BYTES: usize>(
  sigma: &[u8; SEED_BYTES],
  s_hat: &mut PolyVec<K>,
  t_hat: &mut PolyVec<K>,
) {
  if K == 2 {
    let (s0, s_tail) = s_hat.split_at_mut(1);
    let (s1, _) = s_tail.split_at_mut(1);
    let (e0, e_tail) = t_hat.split_at_mut(1);
    let (e1, _) = e_tail.split_at_mut(1);
    sample_noise_quad::<ETA1_RANDOM_BYTES>(sigma, 0, &mut s0[0], 1, &mut s1[0], 2, &mut e0[0], 3, &mut e1[0]);
  } else if ETA1_RANDOM_BYTES == ETA2_RANDOM_BYTES && K == 4 {
    let (s0, s_tail) = s_hat.split_at_mut(1);
    let (s1, s_tail) = s_tail.split_at_mut(1);
    let (s2, s3) = s_tail.split_at_mut(1);
    sample_noise_quad::<ETA2_RANDOM_BYTES>(sigma, 0, &mut s0[0], 1, &mut s1[0], 2, &mut s2[0], 3, &mut s3[0]);

    let (e0, e_tail) = t_hat.split_at_mut(1);
    let (e1, e_tail) = e_tail.split_at_mut(1);
    let (e2, e3) = e_tail.split_at_mut(1);
    sample_noise_quad::<ETA2_RANDOM_BYTES>(sigma, 4, &mut e0[0], 5, &mut e1[0], 6, &mut e2[0], 7, &mut e3[0]);
  } else if ETA1_RANDOM_BYTES == ETA2_RANDOM_BYTES && K == 3 {
    let (s0, s_tail) = s_hat.split_at_mut(1);
    let (s1, s2) = s_tail.split_at_mut(1);
    let (e0, e_tail) = t_hat.split_at_mut(1);
    sample_noise_quad::<ETA2_RANDOM_BYTES>(sigma, 0, &mut s0[0], 1, &mut s1[0], 2, &mut s2[0], 3, &mut e0[0]);

    let (e1, e2) = e_tail.split_at_mut(1);
    sample_noise_pair::<ETA2_RANDOM_BYTES>(sigma, 4, &mut e1[0], 5, &mut e2[0]);
  } else {
    let mut nonce = 0u8;
    for poly in s_hat.iter_mut() {
      sample_noise::<ETA1_RANDOM_BYTES>(sigma, nonce, poly);
      nonce = nonce.wrapping_add(1);
    }
    for poly in t_hat.iter_mut() {
      sample_noise::<ETA1_RANDOM_BYTES>(sigma, nonce, poly);
      nonce = nonce.wrapping_add(1);
    }
  }
}

#[inline(always)]
fn keygen_ntt_noise<const K: usize>(s_hat: &mut PolyVec<K>, t_hat: &mut PolyVec<K>) {
  for poly in s_hat.iter_mut() {
    ntt(poly);
  }
  for poly in t_hat.iter_mut() {
    ntt_to_montgomery_product_domain(poly);
  }
}

#[inline(always)]
fn keygen_matrix_accumulate<const K: usize>(rho: &[u8; SEED_BYTES], s_hat: &PolyVec<K>, t_hat: &mut PolyVec<K>) {
  if use_fused_matrix_accumulate::<K>() {
    for (i, t_hat_i) in t_hat.iter_mut().enumerate() {
      if K == 4 {
        sample_ntt_quad_mul_accumulate(
          rho,
          [
            (0, i as u8, &s_hat[0]),
            (1, i as u8, &s_hat[1]),
            (2, i as u8, &s_hat[2]),
            (3, i as u8, &s_hat[3]),
          ],
          t_hat_i,
        );
      } else {
        let mut j = 0usize;
        while j.strict_add(1) < K {
          let next = j.strict_add(1);
          sample_ntt_pair_mul_accumulate(
            rho,
            (j as u8, i as u8, &s_hat[j]),
            (next as u8, i as u8, &s_hat[next]),
            t_hat_i,
          );

          j = j.strict_add(2);
        }
        if j < K {
          sample_ntt_mul_accumulate(rho, j as u8, i as u8, &s_hat[j], t_hat_i);
        }
      }
    }
  } else {
    sample_matrix_ntt_mul_accumulate_materialized::<K>(rho, s_hat, t_hat, false);
  }
}

#[inline(always)]
fn keygen_from_product_domain<const K: usize>(t_hat: &mut PolyVec<K>) {
  for poly in t_hat.iter_mut() {
    poly_from_montgomery_product_domain(poly);
  }
}

#[inline(always)]
fn keygen_encode<const K: usize, const DK_PKE_BYTES: usize, const EK_BYTES: usize>(
  rho: &[u8; SEED_BYTES],
  s_hat: &PolyVec<K>,
  t_hat: &PolyVec<K>,
) -> ([u8; EK_BYTES], [u8; DK_PKE_BYTES]) {
  let mut ek = [0u8; EK_BYTES];
  for (i, poly) in t_hat.iter().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_encode::<12>(poly, &mut ek[start..start.strict_add(POLY_BYTES)]);
  }
  ek[DK_PKE_BYTES..].copy_from_slice(rho);

  let mut dk_pke = [0u8; DK_PKE_BYTES];
  for (i, poly) in s_hat.iter().enumerate() {
    let start = i.strict_mul(POLY_BYTES);
    byte_encode::<12>(poly, &mut dk_pke[start..start.strict_add(POLY_BYTES)]);
  }

  (ek, dk_pke)
}

#[inline]
pub(super) fn keygen_1024(random: &[u8; 64]) -> ([u8; 1568], [u8; 3168]) {
  keygen::<4, 4, 128, 1536, 1568, 3168>(random)
}

#[cfg(feature = "diag")]
pub(super) fn diag_keygen_expand_digest<const K_U8: u8>(d: &[u8; SEED_BYTES]) -> u16 {
  let mut seed = [0u8; 33];
  seed[..SEED_BYTES].copy_from_slice(d);
  seed[SEED_BYTES] = K_U8;

  let mut expanded = g(&seed);
  let digest = diag_fold_bytes(&expanded);
  ct::zeroize(&mut seed);
  ct::zeroize(&mut expanded);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_keygen_noise_sample_digest<const K: usize, const ETA1_RANDOM_BYTES: usize>(
  sigma: &[u8; SEED_BYTES],
) -> u16 {
  let mut s_hat = [[0u16; N]; K];
  let mut t_hat = [[0u16; N]; K];
  keygen_sample_noise::<K, ETA1_RANDOM_BYTES>(sigma, &mut s_hat, &mut t_hat);

  let digest = diag_fold_polyvec(&s_hat) ^ diag_fold_polyvec(&t_hat);
  zeroize_polyvec(&mut s_hat);
  zeroize_polyvec(&mut t_hat);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_keygen_noise_ntt_digest<const K: usize, const ETA1_RANDOM_BYTES: usize>(
  sigma: &[u8; SEED_BYTES],
) -> u16 {
  let mut s_hat = [[0u16; N]; K];
  let mut t_hat = [[0u16; N]; K];
  keygen_sample_noise::<K, ETA1_RANDOM_BYTES>(sigma, &mut s_hat, &mut t_hat);
  keygen_ntt_noise::<K>(&mut s_hat, &mut t_hat);

  let digest = diag_fold_polyvec(&s_hat) ^ diag_fold_polyvec(&t_hat);
  zeroize_polyvec(&mut s_hat);
  zeroize_polyvec(&mut t_hat);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_keygen_matrix_accumulate_digest<const K: usize, const ETA1_RANDOM_BYTES: usize>(
  rho: &[u8; SEED_BYTES],
  sigma: &[u8; SEED_BYTES],
) -> u16 {
  let mut s_hat = [[0u16; N]; K];
  let mut t_hat = [[0u16; N]; K];
  keygen_sample_noise::<K, ETA1_RANDOM_BYTES>(sigma, &mut s_hat, &mut t_hat);
  keygen_ntt_noise::<K>(&mut s_hat, &mut t_hat);
  keygen_matrix_accumulate::<K>(rho, &s_hat, &mut t_hat);

  let digest = diag_fold_polyvec(&t_hat);
  zeroize_polyvec(&mut s_hat);
  zeroize_polyvec(&mut t_hat);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_keygen_from_product_domain_digest<const K: usize, const ETA1_RANDOM_BYTES: usize>(
  rho: &[u8; SEED_BYTES],
  sigma: &[u8; SEED_BYTES],
) -> u16 {
  let mut s_hat = [[0u16; N]; K];
  let mut t_hat = [[0u16; N]; K];
  keygen_sample_noise::<K, ETA1_RANDOM_BYTES>(sigma, &mut s_hat, &mut t_hat);
  keygen_ntt_noise::<K>(&mut s_hat, &mut t_hat);
  keygen_matrix_accumulate::<K>(rho, &s_hat, &mut t_hat);
  keygen_from_product_domain::<K>(&mut t_hat);

  let digest = diag_fold_polyvec(&t_hat);
  zeroize_polyvec(&mut s_hat);
  zeroize_polyvec(&mut t_hat);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_keygen_encode_digest<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
>(
  rho: &[u8; SEED_BYTES],
  sigma: &[u8; SEED_BYTES],
) -> u16 {
  let mut s_hat = [[0u16; N]; K];
  let mut t_hat = [[0u16; N]; K];
  keygen_sample_noise::<K, ETA1_RANDOM_BYTES>(sigma, &mut s_hat, &mut t_hat);
  keygen_ntt_noise::<K>(&mut s_hat, &mut t_hat);
  keygen_matrix_accumulate::<K>(rho, &s_hat, &mut t_hat);
  keygen_from_product_domain::<K>(&mut t_hat);

  let (mut ek, mut dk_pke) = keygen_encode::<K, DK_PKE_BYTES, EK_BYTES>(rho, &s_hat, &t_hat);
  let digest = diag_fold_bytes(&ek) ^ diag_fold_bytes(&dk_pke);
  ct::zeroize(&mut ek);
  ct::zeroize(&mut dk_pke);
  zeroize_polyvec(&mut s_hat);
  zeroize_polyvec(&mut t_hat);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_keygen_secret_noise_digest<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
>(
  rho: &[u8; SEED_BYTES],
  sigma: &[u8; SEED_BYTES],
) -> [u8; HASH_BYTES] {
  let (mut ek, mut dk_pke) = pke_keygen_expanded::<K, ETA1_RANDOM_BYTES, DK_PKE_BYTES, EK_BYTES>(rho, sigma);
  let mut digest = [0u8; HASH_BYTES];

  for (i, byte) in ek.iter().chain(dk_pke.iter()).copied().enumerate() {
    digest[i & (HASH_BYTES - 1)] ^= byte;
  }

  ct::zeroize(&mut ek);
  ct::zeroize(&mut dk_pke);
  digest
}

#[cfg(feature = "diag")]
#[inline(never)]
fn diag_fold_polyvec<const K: usize>(polyvec: &PolyVec<K>) -> u16 {
  let mut acc = 0u16;
  for poly in polyvec {
    acc ^= diag_fold_poly(poly);
  }
  acc
}

#[cfg(feature = "diag")]
pub(super) fn diag_matrix_sample_scalar_digest<const K: usize>(rho: &[u8; SEED_BYTES]) -> u16 {
  let mut digest = 0u16;
  for i in 0..K {
    for j in 0..K {
      let mut poly = [0u16; N];
      sample_ntt_into(rho, j as u8, i as u8, &mut poly);
      digest ^= diag_fold_poly(&poly);
      zeroize_poly(&mut poly);
    }
  }
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_matrix_sample_pair_digest<const K: usize>(rho: &[u8; SEED_BYTES]) -> u16 {
  let mut digest = 0u16;
  let mut entry = 0usize;
  while entry.strict_add(1) < K.strict_mul(K) {
    let (j0, i0) = matrix_sample_coord::<K>(entry);
    let (j1, i1) = matrix_sample_coord::<K>(entry.strict_add(1));
    let mut poly0 = [0u16; N];
    let mut poly1 = [0u16; N];
    sample_ntt_pair_into(rho, j0, i0, j1, i1, &mut poly0, &mut poly1);
    digest ^= diag_fold_poly(&poly0);
    digest ^= diag_fold_poly(&poly1);
    zeroize_poly(&mut poly0);
    zeroize_poly(&mut poly1);
    entry = entry.strict_add(2);
  }

  if entry < K.strict_mul(K) {
    let (j, i) = matrix_sample_coord::<K>(entry);
    let mut poly = [0u16; N];
    sample_ntt_into(rho, j, i, &mut poly);
    digest ^= diag_fold_poly(&poly);
    zeroize_poly(&mut poly);
  }

  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_matrix_sample_quad_digest<const K: usize>(rho: &[u8; SEED_BYTES]) -> u16 {
  let mut digest = 0u16;
  let mut entry = 0usize;
  while entry.strict_add(3) < K.strict_mul(K) {
    let coord0 = matrix_sample_coord::<K>(entry);
    let coord1 = matrix_sample_coord::<K>(entry.strict_add(1));
    let coord2 = matrix_sample_coord::<K>(entry.strict_add(2));
    let coord3 = matrix_sample_coord::<K>(entry.strict_add(3));
    let mut poly0 = [0u16; N];
    let mut poly1 = [0u16; N];
    let mut poly2 = [0u16; N];
    let mut poly3 = [0u16; N];
    sample_ntt_quad_into(
      rho,
      [coord0, coord1, coord2, coord3],
      [&mut poly0, &mut poly1, &mut poly2, &mut poly3],
    );
    digest ^= diag_fold_poly(&poly0);
    digest ^= diag_fold_poly(&poly1);
    digest ^= diag_fold_poly(&poly2);
    digest ^= diag_fold_poly(&poly3);
    zeroize_poly(&mut poly0);
    zeroize_poly(&mut poly1);
    zeroize_poly(&mut poly2);
    zeroize_poly(&mut poly3);
    entry = entry.strict_add(4);
  }

  while entry.strict_add(1) < K.strict_mul(K) {
    let (j0, i0) = matrix_sample_coord::<K>(entry);
    let (j1, i1) = matrix_sample_coord::<K>(entry.strict_add(1));
    let mut poly0 = [0u16; N];
    let mut poly1 = [0u16; N];
    sample_ntt_pair_into(rho, j0, i0, j1, i1, &mut poly0, &mut poly1);
    digest ^= diag_fold_poly(&poly0);
    digest ^= diag_fold_poly(&poly1);
    zeroize_poly(&mut poly0);
    zeroize_poly(&mut poly1);
    entry = entry.strict_add(2);
  }

  if entry < K.strict_mul(K) {
    let (j, i) = matrix_sample_coord::<K>(entry);
    let mut poly = [0u16; N];
    sample_ntt_into(rho, j, i, &mut poly);
    digest ^= diag_fold_poly(&poly);
    zeroize_poly(&mut poly);
  }

  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_ntt_digest(seed: u16) -> u16 {
  let mut poly = diag_poly(seed);
  ntt(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_ntt_input_digest(mut poly: Poly) -> u16 {
  ntt(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_ntt_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::ntt_vector(&mut poly);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the aarch64 NEON NTT kernel.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 targets with baseline Advanced SIMD
/// available.
#[cfg(all(feature = "diag", target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_aarch64_ntt_neon_digest(seed: u16) -> u16 {
  let poly = diag_poly(seed);
  // SAFETY: forwarded from this function's caller contract.
  unsafe { diag_aarch64_ntt_neon_input_digest(poly) }
}

/// Diagnostic digest for the aarch64 NEON NTT kernel.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 targets with baseline Advanced SIMD
/// available.
#[cfg(all(feature = "diag", target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_aarch64_ntt_neon_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct NEON diagnostic call because:
  // 1. The caller guarantees this runs only on an aarch64 CPU with Advanced SIMD.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the NEON kernel contract.
  // 3. This diagnostic root intentionally bypasses production dispatch so benchmark and CT artifacts
  //    can inspect the aarch64 NEON candidate itself.
  // 4. The NEON memory access schedule depends only on public ML-KEM dimensions.
  unsafe {
    ntt_neon(&mut poly);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_inverse_ntt_montgomery_product_digest(seed: u16) -> u16 {
  let mut poly = diag_poly(seed);
  inverse_ntt_montgomery_product(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_inverse_ntt_montgomery_product_input_digest(mut poly: Poly) -> u16 {
  inverse_ntt_montgomery_product(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_inverse_ntt_montgomery_product_add_assign_digest(seed: u16) -> u16 {
  let mut poly = diag_poly(seed);
  let addend = diag_poly(seed.wrapping_add(1));
  inverse_ntt_montgomery_product_add_assign(&mut poly, &addend);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_inverse_ntt_montgomery_product_add_assign_input_digest(mut poly: Poly, addend: Poly) -> u16 {
  inverse_ntt_montgomery_product_add_assign(&mut poly, &addend);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector inverse-NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_inverse_ntt_montgomery_product_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. `INV_NTT_PRODUCT_SCALE_MONT` is the public ML-KEM product-domain scale constant.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::inverse_ntt_vector(&mut poly, INV_NTT_PRODUCT_SCALE_MONT);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_add_assign_digest(seed: u16) -> u16 {
  let a = diag_poly(seed);
  let b = diag_poly(seed.wrapping_add(1));
  let mut acc = diag_poly(seed.wrapping_add(2));
  multiply_ntts_add_assign(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_poly(&mut acc);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_add_assign_input_digest(a: Poly, b: Poly, mut acc: Poly) -> u16 {
  multiply_ntts_add_assign(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_poly(&mut acc);
  digest
}

/// Diagnostic digest for the rscrypto-owned aarch64 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
pub(super) unsafe fn diag_aarch64_multiply_ntts_add_assign_asm_digest(seed: u16) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { aarch64::diag_basemul_accumulate_asm_digest(seed) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
pub(super) unsafe fn diag_aarch64_multiply_ntts_add_assign_asm_input_digest(a: Poly, b: Poly, acc: Poly) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { aarch64::diag_basemul_accumulate_asm_input_digest(a, b, acc) }
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_accumulate_k2_input_digest(
  mut a: PolyVec<2>,
  mut b: PolyVec<2>,
  mut acc: Poly,
) -> u16 {
  multiply_ntts_accumulate(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_accumulate_k2_digest(seed: u16) -> u16 {
  let a = [diag_poly(seed), diag_poly(seed.wrapping_add(1))];
  let b = [diag_poly(seed.wrapping_add(2)), diag_poly(seed.wrapping_add(3))];
  let acc = diag_poly(seed.wrapping_add(4));
  diag_multiply_ntts_accumulate_k2_input_digest(a, b, acc)
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_accumulate_k3_input_digest(
  mut a: PolyVec<3>,
  mut b: PolyVec<3>,
  mut acc: Poly,
) -> u16 {
  multiply_ntts_accumulate(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_accumulate_k3_digest(seed: u16) -> u16 {
  let a = [
    diag_poly(seed),
    diag_poly(seed.wrapping_add(1)),
    diag_poly(seed.wrapping_add(2)),
  ];
  let b = [
    diag_poly(seed.wrapping_add(3)),
    diag_poly(seed.wrapping_add(4)),
    diag_poly(seed.wrapping_add(5)),
  ];
  let acc = diag_poly(seed.wrapping_add(6));
  diag_multiply_ntts_accumulate_k3_input_digest(a, b, acc)
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_accumulate_k4_input_digest(
  mut a: PolyVec<4>,
  mut b: PolyVec<4>,
  mut acc: Poly,
) -> u16 {
  multiply_ntts_accumulate(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_multiply_ntts_accumulate_k4_digest(seed: u16) -> u16 {
  let a = [
    diag_poly(seed),
    diag_poly(seed.wrapping_add(1)),
    diag_poly(seed.wrapping_add(2)),
    diag_poly(seed.wrapping_add(3)),
  ];
  let b = [
    diag_poly(seed.wrapping_add(4)),
    diag_poly(seed.wrapping_add(5)),
    diag_poly(seed.wrapping_add(6)),
    diag_poly(seed.wrapping_add(7)),
  ];
  let acc = diag_poly(seed.wrapping_add(8));
  diag_multiply_ntts_accumulate_k4_input_digest(a, b, acc)
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=3 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
pub(super) unsafe fn diag_aarch64_multiply_ntts_accumulate_k3_asm_digest(seed: u16) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { aarch64::diag_basemul_accumulate_k3_asm_digest(seed) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=4 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
pub(super) unsafe fn diag_aarch64_multiply_ntts_accumulate_k4_asm_digest(seed: u16) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { aarch64::diag_basemul_accumulate_k4_asm_digest(seed) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=3 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
pub(super) unsafe fn diag_aarch64_multiply_ntts_accumulate_k3_asm_input_digest(
  a: PolyVec<3>,
  b: PolyVec<3>,
  acc: Poly,
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { aarch64::diag_basemul_accumulate_k3_asm_input_digest(a, b, acc) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=4 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
pub(super) unsafe fn diag_aarch64_multiply_ntts_accumulate_k4_asm_input_digest(
  a: PolyVec<4>,
  b: PolyVec<4>,
  acc: Poly,
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { aarch64::diag_basemul_accumulate_k4_asm_input_digest(a, b, acc) }
}

#[cfg(feature = "diag")]
pub(super) fn diag_to_montgomery_product_domain_digest(seed: u16) -> u16 {
  let mut poly = diag_poly(seed);
  poly_to_montgomery_product_domain(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_to_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  poly_to_montgomery_product_domain(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_from_montgomery_product_domain_digest(seed: u16) -> u16 {
  let mut poly = diag_poly(seed);
  poly_from_montgomery_product_domain(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_from_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  poly_from_montgomery_product_domain(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector product-domain conversion kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_to_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x product-domain conversion kernel itself.
  unsafe {
    s390x::to_montgomery_product_domain_vector(&mut poly);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector product-domain exit kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_from_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. `MONT_R_SQUARED_MOD_Q` is the public ML-KEM conversion constant.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x product-domain exit kernel itself.
  unsafe {
    s390x::from_montgomery_product_domain_vector(&mut poly);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector base-multiply accumulator kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_multiply_ntts_add_assign_input_digest(a: Poly, b: Poly, mut acc: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `acc`, `a`, and `b` are fixed 256-coefficient polynomials matching the kernel contract.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::multiply_ntts_add_assign_vector(&mut acc, &a, &b);
  }
  let digest = diag_fold_poly(&acc);
  zeroize_poly(&mut acc);
  digest
}

/// Diagnostic digest for the s390x z/Vector k=3 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_multiply_ntts_accumulate_k3_input_digest(
  mut a: PolyVec<3>,
  mut b: PolyVec<3>,
  mut acc: Poly,
) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `acc`, `a`, and `b` contain fixed 256-coefficient polynomials matching the kernel contract.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::multiply_ntts_accumulate_k3_vector(&mut acc, [&a[0], &a[1], &a[2]], [&b[0], &b[1], &b[2]]);
  }
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

/// Diagnostic digest for the s390x z/Vector k=4 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_multiply_ntts_accumulate_k4_input_digest(
  mut a: PolyVec<4>,
  mut b: PolyVec<4>,
  mut acc: Poly,
) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `acc`, `a`, and `b` contain fixed 256-coefficient polynomials matching the kernel contract.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::multiply_ntts_accumulate_k4_vector(&mut acc, [&a[0], &a[1], &a[2], &a[3]], [&b[0], &b[1], &b[2], &b[3]]);
  }
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_compress_decompress_digest(seed: u16) -> u16 {
  let values = [
    seed % Q,
    seed.wrapping_mul(17).wrapping_add(3) % Q,
    seed.wrapping_mul(29).wrapping_add(11) % Q,
    seed.wrapping_mul(43).wrapping_add(19) % Q,
  ];
  let compressed = compress_values_4::<10>(values);
  let decompressed = decompress_values_4::<10>(compressed);
  let compressed_11 = compress_values_4::<11>(values);
  let decompressed_11 = decompress_values_4::<11>(compressed_11);
  let compressed_5 = compress_values_4::<5>(values);
  let decompressed_5 = decompress_values_4::<5>(compressed_5);

  let mut digest = 0u16;
  for value in compressed {
    digest = digest.rotate_left(3) ^ value;
  }
  for value in decompressed {
    digest = digest.rotate_left(5) ^ value;
  }
  for value in compressed_11 {
    digest = digest.rotate_left(7) ^ value;
  }
  for value in decompressed_11 {
    digest = digest.rotate_left(11) ^ value;
  }
  for value in compressed_5 {
    digest = digest.rotate_left(13) ^ value;
  }
  for value in decompressed_5 {
    digest = digest.rotate_left(2) ^ value;
  }
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  let compressed = compress_values_4::<10>(values);
  let decompressed = decompress_values_4::<10>(compressed);
  let compressed_11 = compress_values_4::<11>(values);
  let decompressed_11 = decompress_values_4::<11>(compressed_11);
  let compressed_5 = compress_values_4::<5>(values);
  let decompressed_5 = decompress_values_4::<5>(compressed_5);

  let mut digest = 0u16;
  for value in compressed {
    digest = digest.rotate_left(3) ^ value;
  }
  for value in decompressed {
    digest = digest.rotate_left(5) ^ value;
  }
  for value in compressed_11 {
    digest = digest.rotate_left(7) ^ value;
  }
  for value in decompressed_11 {
    digest = digest.rotate_left(11) ^ value;
  }
  for value in compressed_5 {
    digest = digest.rotate_left(13) ^ value;
  }
  for value in decompressed_5 {
    digest = digest.rotate_left(2) ^ value;
  }
  digest
}

/// Diagnostic digest for the s390x z/Vector compress/decompress kernels.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(super) unsafe fn diag_s390x_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  // SAFETY: Direct z/Vector diagnostic calls because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `values` is exactly the four-coefficient shape required by the vector helpers.
  // 3. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x compression kernels themselves.
  let compressed = unsafe { s390x::compress_values_4::<10>(values) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let decompressed = unsafe { s390x::decompress_values_4::<10>(compressed) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let compressed_11 = unsafe { s390x::compress_values_4::<11>(values) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let decompressed_11 = unsafe { s390x::decompress_values_4::<11>(compressed_11) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let compressed_5 = unsafe { s390x::compress_values_4::<5>(values) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let decompressed_5 = unsafe { s390x::decompress_values_4::<5>(compressed_5) };

  let mut digest = 0u16;
  for value in compressed {
    digest = digest.rotate_left(3) ^ value;
  }
  for value in decompressed {
    digest = digest.rotate_left(5) ^ value;
  }
  for value in compressed_11 {
    digest = digest.rotate_left(7) ^ value;
  }
  for value in decompressed_11 {
    digest = digest.rotate_left(11) ^ value;
  }
  for value in compressed_5 {
    digest = digest.rotate_left(13) ^ value;
  }
  for value in decompressed_5 {
    digest = digest.rotate_left(2) ^ value;
  }
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_decap_decrypt_digest<
  const K: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  prepared: &PreparedDecapsulationArithmetic<K>,
  c: &[u8; CT_BYTES],
) -> u16 {
  let mut message = pke_decrypt_prepared::<K, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(&prepared.s_hat, c);
  let digest = diag_fold_bytes(&message);
  ct::zeroize(&mut message);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_decap_reencrypt_digest<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  prepared: &PreparedDecapsulationArithmetic<K>,
  seed: u8,
) -> u16 {
  let mut m = [0u8; SEED_BYTES];
  let mut r = [0u8; SEED_BYTES];
  for (i, byte) in m.iter_mut().enumerate() {
    *byte = seed.wrapping_add(i as u8);
  }
  for (i, byte) in r.iter_mut().enumerate() {
    *byte = seed.wrapping_add(0x80).wrapping_add(i as u8);
  }

  let mut ciphertext = pke_encrypt_prepared::<K, ETA1_RANDOM_BYTES, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(
    &prepared.encapsulation,
    &m,
    &r,
  );
  let digest = diag_fold_bytes(&ciphertext);
  ct::zeroize(&mut m);
  ct::zeroize(&mut r);
  ct::zeroize(&mut ciphertext);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_decap_reencrypt_ciphertext<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  prepared: &PreparedDecapsulationArithmetic<K>,
  seed: u8,
) -> [u8; CT_BYTES] {
  let mut m = [0u8; SEED_BYTES];
  let mut r = [0u8; SEED_BYTES];
  fill_diag_seed(&mut m, seed);
  fill_diag_seed(&mut r, seed.wrapping_add(0x80));

  let ciphertext = pke_encrypt_prepared::<K, ETA1_RANDOM_BYTES, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(
    &prepared.encapsulation,
    &m,
    &r,
  );
  ct::zeroize(&mut m);
  ct::zeroize(&mut r);
  ciphertext
}

#[cfg(feature = "diag")]
pub(super) fn diag_decap_reencrypt_compare_512_digest(
  prepared: &PreparedDecapsulationArithmetic<2>,
  expected: &[u8; 768],
  seed: u8,
) -> u16 {
  let mut m = [0u8; SEED_BYTES];
  let mut r = [0u8; SEED_BYTES];
  fill_diag_seed(&mut m, seed);
  fill_diag_seed(&mut r, seed.wrapping_add(0x80));

  let mask = pke_encrypt_prepared_512_compare(&prepared.encapsulation, &m, &r, expected);
  let digest = u16::from(mask) ^ u16::from(expected[0]);
  ct::zeroize(&mut m);
  ct::zeroize(&mut r);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_decap_reencrypt_compare_768_digest(
  prepared: &PreparedDecapsulationArithmetic<3>,
  expected: &[u8; 1088],
  seed: u8,
) -> u16 {
  let mut m = [0u8; SEED_BYTES];
  let mut r = [0u8; SEED_BYTES];
  fill_diag_seed(&mut m, seed);
  fill_diag_seed(&mut r, seed.wrapping_add(0x80));

  let mask = pke_encrypt_prepared_768_compare(&prepared.encapsulation, &m, &r, expected);
  let digest = u16::from(mask) ^ u16::from(expected[0]);
  ct::zeroize(&mut m);
  ct::zeroize(&mut r);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_decap_reencrypt_compare_1024_digest(
  prepared: &PreparedDecapsulationArithmetic<4>,
  expected: &[u8; 1568],
  seed: u8,
) -> u16 {
  let mut m = [0u8; SEED_BYTES];
  let mut r = [0u8; SEED_BYTES];
  fill_diag_seed(&mut m, seed);
  fill_diag_seed(&mut r, seed.wrapping_add(0x80));

  let mask = pke_encrypt_prepared_1024_compare(&prepared.encapsulation, &m, &r, expected);
  let digest = u16::from(mask) ^ u16::from(expected[0]);
  ct::zeroize(&mut m);
  ct::zeroize(&mut r);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_decap_hash_select_digest<
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
  const DK_BYTES: usize,
  const CT_BYTES: usize,
>(
  dk: &[u8; DK_BYTES],
  c: &[u8; CT_BYTES],
  seed: u8,
) -> u16 {
  let h_start = DK_PKE_BYTES.strict_add(EK_BYTES);
  let h_stored = &dk[h_start..h_start.strict_add(HASH_BYTES)];
  let z = &dk[h_start.strict_add(HASH_BYTES)..];

  let mut m_prime = [0u8; SEED_BYTES];
  for (i, byte) in m_prime.iter_mut().enumerate() {
    *byte = seed.wrapping_add(i as u8);
  }

  let mut input = [0u8; 64];
  input[..SEED_BYTES].copy_from_slice(&m_prime);
  input[SEED_BYTES..].copy_from_slice(h_stored);

  let mut expanded = g(&input);
  let mut k_prime = [0u8; SHARED_SECRET_BYTES];
  let mut r_prime = [0u8; SEED_BYTES];
  k_prime.copy_from_slice(&expanded[..SHARED_SECRET_BYTES]);
  r_prime.copy_from_slice(&expanded[SHARED_SECRET_BYTES..]);

  let mut k_bar = j(z, c);
  let mut c_prime = *c;
  c_prime[0] ^= seed & 1;
  let mut match_mask = ct_eq_mask(c, &c_prime);
  let reject_mask = !match_mask;

  let mut shared = [0u8; SHARED_SECRET_BYTES];
  for i in 0..SHARED_SECRET_BYTES {
    shared[i] = (k_prime[i] & match_mask) | (k_bar[i] & reject_mask);
  }

  let digest = diag_fold_bytes(&shared) ^ u16::from(r_prime[0]);
  ct::zeroize(&mut m_prime);
  ct::zeroize(&mut input);
  ct::zeroize(&mut expanded);
  ct::zeroize(&mut k_prime);
  ct::zeroize(&mut r_prime);
  ct::zeroize(&mut k_bar);
  ct::zeroize(&mut c_prime);
  ct::zeroize(&mut shared);
  ct::zeroize(core::slice::from_mut(&mut match_mask));
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_pke_noise_ntt_digest<const K: usize, const ETA1_RANDOM_BYTES: usize>(seed: u8) -> u16 {
  let mut r = [0u8; SEED_BYTES];
  fill_diag_seed(&mut r, seed);

  let mut nonce = 0u8;
  let mut y = [[0u16; N]; K];
  let mut e1 = [[0u16; N]; K];
  for poly in &mut y {
    sample_noise::<ETA1_RANDOM_BYTES>(&r, nonce, poly);
    nonce = nonce.wrapping_add(1);
  }
  for poly in &mut e1 {
    sample_noise::<ETA2_RANDOM_BYTES>(&r, nonce, poly);
    nonce = nonce.wrapping_add(1);
  }
  let mut e2 = [0u16; N];
  sample_noise::<ETA2_RANDOM_BYTES>(&r, nonce, &mut e2);

  for poly in &mut y {
    ntt(poly);
  }

  let mut digest = diag_fold_poly(&e2);
  for poly in &y {
    digest ^= diag_fold_poly(poly);
  }
  for poly in &e1 {
    digest ^= diag_fold_poly(poly);
  }

  ct::zeroize(&mut r);
  zeroize_polyvec(&mut y);
  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_pke_matrix_u_digest<const K: usize>(ek: &PreparedEncapsulationArithmetic<K>, seed: u16) -> u16 {
  let mut y_hat = [[0u16; N]; K];
  for (i, poly) in y_hat.iter_mut().enumerate() {
    *poly = diag_poly(seed.wrapping_add(i as u16));
  }

  let mut u = [[0u16; N]; K];
  sample_matrix_ntt_mul_accumulate_materialized::<K>(&ek.rho, &y_hat, &mut u, true);

  let mut digest = 0u16;
  for poly in &u {
    digest ^= diag_fold_poly(poly);
  }

  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_pke_matrix_u_fused_digest<const K: usize>(
  ek: &PreparedEncapsulationArithmetic<K>,
  seed: u16,
) -> u16 {
  let mut y_hat = [[0u16; N]; K];
  for (i, poly) in y_hat.iter_mut().enumerate() {
    *poly = diag_poly(seed.wrapping_add(i as u16));
  }

  let mut u = [[0u16; N]; K];
  for (i, u_i) in u.iter_mut().enumerate() {
    let mut acc = [0u16; N];
    if K == 4 {
      sample_ntt_quad_mul_accumulate(
        &ek.rho,
        [
          (i as u8, 0, &y_hat[0]),
          (i as u8, 1, &y_hat[1]),
          (i as u8, 2, &y_hat[2]),
          (i as u8, 3, &y_hat[3]),
        ],
        &mut acc,
      );
    } else {
      let mut j = 0usize;
      while j.strict_add(1) < K {
        let next = j.strict_add(1);
        sample_ntt_pair_mul_accumulate(
          &ek.rho,
          (i as u8, j as u8, &y_hat[j]),
          (i as u8, next as u8, &y_hat[next]),
          &mut acc,
        );

        j = j.strict_add(2);
      }
      if j < K {
        sample_ntt_mul_accumulate(&ek.rho, i as u8, j as u8, &y_hat[j], &mut acc);
      }
    }
    *u_i = acc;
  }

  let mut digest = 0u16;
  for poly in &u {
    digest ^= diag_fold_poly(poly);
  }

  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_pke_inverse_u_add_digest<const K: usize>(seed: u16) -> u16 {
  let mut u = [[0u16; N]; K];
  let mut e1 = [[0u16; N]; K];
  for i in 0..K {
    u[i] = diag_poly(seed.wrapping_add(i as u16));
    e1[i] = diag_poly(seed.wrapping_add(0x40).wrapping_add(i as u16));
  }

  for i in 0..K {
    inverse_ntt_montgomery_product_add_assign(&mut u[i], &e1[i]);
  }

  let mut digest = 0u16;
  for poly in &u {
    digest ^= diag_fold_poly(poly);
  }

  zeroize_polyvec(&mut u);
  zeroize_polyvec(&mut e1);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_pke_v_digest<const K: usize>(ek: &PreparedEncapsulationArithmetic<K>, seed: u16) -> u16 {
  let mut y_hat = [[0u16; N]; K];
  for (i, poly) in y_hat.iter_mut().enumerate() {
    *poly = diag_poly(seed.wrapping_add(i as u16));
  }

  let mut v = [0u16; N];
  multiply_ntts_accumulate(&mut v, &ek.t_hat, &y_hat);
  inverse_ntt_montgomery_product(&mut v);

  let digest = diag_fold_poly(&v);
  zeroize_polyvec(&mut y_hat);
  zeroize_poly(&mut v);
  digest
}

#[cfg(feature = "diag")]
pub(super) fn diag_pke_encode_digest<
  const K: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
>(
  seed: u16,
) -> u16 {
  let mut u = [[0u16; N]; K];
  for (i, poly) in u.iter_mut().enumerate() {
    *poly = diag_poly(seed.wrapping_add(i as u16));
  }
  let v = diag_poly(seed.wrapping_add(0x80));

  let mut ciphertext = [0u8; CT_BYTES];
  for (i, poly) in u.iter().enumerate() {
    let start = i.strict_mul(POLY_DU_BYTES);
    compress_encode_poly::<DU>(poly, &mut ciphertext[start..start.strict_add(POLY_DU_BYTES)]);
  }
  compress_encode_poly::<DV>(&v, &mut ciphertext[POLY_DU_BYTES.strict_mul(K)..]);

  let digest = diag_fold_bytes(&ciphertext);
  zeroize_polyvec(&mut u);
  ct::zeroize(&mut ciphertext);
  digest
}

#[cfg(feature = "diag")]
#[inline]
fn matrix_sample_coord<const K: usize>(entry: usize) -> (u8, u8) {
  ((entry % K) as u8, (entry / K) as u8)
}

#[cfg(feature = "diag")]
fn fill_diag_seed(out: &mut [u8; SEED_BYTES], seed: u8) {
  for (i, byte) in out.iter_mut().enumerate() {
    *byte = seed.wrapping_add(i as u8);
  }
}

#[cfg(feature = "diag")]
fn diag_poly(seed: u16) -> Poly {
  let mut state = u32::from(seed).wrapping_mul(0x9E37).wrapping_add(0x7F4A_7C15);
  let mut poly = [0u16; N];
  for coeff in &mut poly {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *coeff = (state % u32::from(Q)) as u16;
  }
  poly
}

#[cfg(feature = "diag")]
#[inline(never)]
fn diag_fold_poly(poly: &Poly) -> u16 {
  let mut acc = 0u16;
  for (i, &coeff) in poly.iter().enumerate() {
    acc ^= coeff.wrapping_mul((i as u16).wrapping_add(1));
  }
  acc
}

#[cfg(feature = "diag")]
#[inline(never)]
fn diag_fold_bytes(bytes: &[u8]) -> u16 {
  let mut acc = 0u16;
  for (i, &byte) in bytes.iter().enumerate() {
    acc ^= u16::from(byte).wrapping_mul((i as u16).wrapping_add(1));
  }
  acc
}

fn pke_encrypt<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const EK_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  ek: &[u8; EK_BYTES],
  m: &[u8; SEED_BYTES],
  r: &[u8; SEED_BYTES],
) -> [u8; CT_BYTES] {
  let prepared = prepare_encapsulation_key::<K, EK_BYTES>(ek);
  pke_encrypt_prepared::<K, ETA1_RANDOM_BYTES, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(&prepared, m, r)
}

fn pke_encrypt_prepared_768(
  ek: &PreparedEncapsulationArithmetic<3>,
  m: &[u8; SEED_BYTES],
  r: &[u8; SEED_BYTES],
) -> [u8; 1088] {
  let mut y_hat = [[0u16; N]; 3];
  let mut e1 = [[0u16; N]; 3];
  let [y0, y1, y2] = &mut y_hat;
  let [e10, e11, e12] = &mut e1;
  sample_noise_quad::<128>(r, 0, y0, 1, y1, 2, y2, 3, e10);
  sample_noise_pair::<ETA2_RANDOM_BYTES>(r, 4, e11, 5, e12);
  let mut e2 = [0u16; N];
  sample_noise::<ETA2_RANDOM_BYTES>(r, 6, &mut e2);

  ntt(&mut y_hat[0]);
  ntt(&mut y_hat[1]);
  ntt(&mut y_hat[2]);

  let mut u = [[0u16; N]; 3];
  matrix_ntt_mul_accumulate_prepared_transpose(ek, &y_hat, &mut u);

  inverse_ntt_montgomery_product_add_assign(&mut u[0], &e1[0]);
  inverse_ntt_montgomery_product_add_assign(&mut u[1], &e1[1]);
  inverse_ntt_montgomery_product_add_assign(&mut u[2], &e1[2]);

  let mut v = [0u16; N];
  multiply_ntts_accumulate(&mut v, &ek.t_hat, &y_hat);
  inverse_ntt_montgomery_product_add_assign(&mut v, &e2);
  decompress_message_add_assign(m, &mut v);

  let mut ciphertext = [0u8; 1088];
  compress_encode_poly::<10>(&u[0], &mut ciphertext[0..320]);
  compress_encode_poly::<10>(&u[1], &mut ciphertext[320..640]);
  compress_encode_poly::<10>(&u[2], &mut ciphertext[640..960]);
  compress_encode_poly::<4>(&v, &mut ciphertext[960..]);

  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  zeroize_poly(&mut v);
  ciphertext
}

fn pke_encrypt_prepared_1024(
  ek: &PreparedEncapsulationArithmetic<4>,
  m: &[u8; SEED_BYTES],
  r: &[u8; SEED_BYTES],
) -> [u8; 1568] {
  let mut y_hat = [[0u16; N]; 4];
  let mut e1 = [[0u16; N]; 4];
  let [y0, y1, y2, y3] = &mut y_hat;
  sample_noise_quad::<128>(r, 0, y0, 1, y1, 2, y2, 3, y3);
  let [e10, e11, e12, e13] = &mut e1;
  sample_noise_quad::<ETA2_RANDOM_BYTES>(r, 4, e10, 5, e11, 6, e12, 7, e13);
  let mut e2 = [0u16; N];
  sample_noise::<ETA2_RANDOM_BYTES>(r, 8, &mut e2);

  ntt(&mut y_hat[0]);
  ntt(&mut y_hat[1]);
  ntt(&mut y_hat[2]);
  ntt(&mut y_hat[3]);

  let mut u = [[0u16; N]; 4];
  matrix_ntt_mul_accumulate_prepared_transpose(ek, &y_hat, &mut u);

  inverse_ntt_montgomery_product_add_assign(&mut u[0], &e1[0]);
  inverse_ntt_montgomery_product_add_assign(&mut u[1], &e1[1]);
  inverse_ntt_montgomery_product_add_assign(&mut u[2], &e1[2]);
  inverse_ntt_montgomery_product_add_assign(&mut u[3], &e1[3]);

  let mut v = [0u16; N];
  multiply_ntts_accumulate(&mut v, &ek.t_hat, &y_hat);
  inverse_ntt_montgomery_product_add_assign(&mut v, &e2);
  decompress_message_add_assign(m, &mut v);

  let mut ciphertext = [0u8; 1568];
  compress_encode_poly::<11>(&u[0], &mut ciphertext[0..352]);
  compress_encode_poly::<11>(&u[1], &mut ciphertext[352..704]);
  compress_encode_poly::<11>(&u[2], &mut ciphertext[704..1056]);
  compress_encode_poly::<11>(&u[3], &mut ciphertext[1056..1408]);
  compress_encode_poly::<5>(&v, &mut ciphertext[1408..]);

  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  zeroize_poly(&mut v);
  ciphertext
}

fn pke_encrypt_prepared_512_compare(
  ek: &PreparedEncapsulationArithmetic<2>,
  m: &[u8; SEED_BYTES],
  r: &[u8; SEED_BYTES],
  expected: &[u8; 768],
) -> u8 {
  let mut y_hat = [[0u16; N]; 2];
  let mut e1 = [[0u16; N]; 2];
  let [y0, y1] = &mut y_hat;
  sample_noise_pair::<192>(r, 0, y0, 1, y1);
  let [e10, e11] = &mut e1;
  sample_noise_pair::<ETA2_RANDOM_BYTES>(r, 2, e10, 3, e11);
  let mut e2 = [0u16; N];
  sample_noise::<ETA2_RANDOM_BYTES>(r, 4, &mut e2);

  ntt(&mut y_hat[0]);
  ntt(&mut y_hat[1]);

  let mut u = [[0u16; N]; 2];
  matrix_ntt_mul_accumulate_prepared_transpose(ek, &y_hat, &mut u);

  inverse_ntt_montgomery_product_add_assign(&mut u[0], &e1[0]);
  inverse_ntt_montgomery_product_add_assign(&mut u[1], &e1[1]);

  let mut v = [0u16; N];
  multiply_ntts_accumulate(&mut v, &ek.t_hat, &y_hat);
  inverse_ntt_montgomery_product_add_assign(&mut v, &e2);
  decompress_message_add_assign(m, &mut v);

  let mut mask = 0xffu8;
  mask &= compress_encode_compare_poly::<10, 320>(&u[0], &expected[0..320]);
  mask &= compress_encode_compare_poly::<10, 320>(&u[1], &expected[320..640]);
  mask &= compress_encode_compare_poly::<4, 128>(&v, &expected[640..]);

  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  zeroize_poly(&mut v);
  mask
}

fn pke_encrypt_prepared_768_compare(
  ek: &PreparedEncapsulationArithmetic<3>,
  m: &[u8; SEED_BYTES],
  r: &[u8; SEED_BYTES],
  expected: &[u8; 1088],
) -> u8 {
  let mut y_hat = [[0u16; N]; 3];
  let mut e1 = [[0u16; N]; 3];
  let [y0, y1, y2] = &mut y_hat;
  let [e10, e11, e12] = &mut e1;
  sample_noise_quad::<128>(r, 0, y0, 1, y1, 2, y2, 3, e10);
  sample_noise_pair::<ETA2_RANDOM_BYTES>(r, 4, e11, 5, e12);
  let mut e2 = [0u16; N];
  sample_noise::<ETA2_RANDOM_BYTES>(r, 6, &mut e2);

  ntt(&mut y_hat[0]);
  ntt(&mut y_hat[1]);
  ntt(&mut y_hat[2]);

  let mut u = [[0u16; N]; 3];
  matrix_ntt_mul_accumulate_prepared_transpose(ek, &y_hat, &mut u);

  inverse_ntt_montgomery_product_add_assign(&mut u[0], &e1[0]);
  inverse_ntt_montgomery_product_add_assign(&mut u[1], &e1[1]);
  inverse_ntt_montgomery_product_add_assign(&mut u[2], &e1[2]);

  let mut v = [0u16; N];
  multiply_ntts_accumulate(&mut v, &ek.t_hat, &y_hat);
  inverse_ntt_montgomery_product_add_assign(&mut v, &e2);
  decompress_message_add_assign(m, &mut v);

  let mut mask = 0xffu8;
  mask &= compress_encode_compare_poly::<10, 320>(&u[0], &expected[0..320]);
  mask &= compress_encode_compare_poly::<10, 320>(&u[1], &expected[320..640]);
  mask &= compress_encode_compare_poly::<10, 320>(&u[2], &expected[640..960]);
  mask &= compress_encode_compare_poly::<4, 128>(&v, &expected[960..]);

  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  zeroize_poly(&mut v);
  mask
}

fn pke_encrypt_prepared_1024_compare(
  ek: &PreparedEncapsulationArithmetic<4>,
  m: &[u8; SEED_BYTES],
  r: &[u8; SEED_BYTES],
  expected: &[u8; 1568],
) -> u8 {
  let mut y_hat = [[0u16; N]; 4];
  let mut e1 = [[0u16; N]; 4];
  let [y0, y1, y2, y3] = &mut y_hat;
  sample_noise_quad::<128>(r, 0, y0, 1, y1, 2, y2, 3, y3);
  let [e10, e11, e12, e13] = &mut e1;
  sample_noise_quad::<ETA2_RANDOM_BYTES>(r, 4, e10, 5, e11, 6, e12, 7, e13);
  let mut e2 = [0u16; N];
  sample_noise::<ETA2_RANDOM_BYTES>(r, 8, &mut e2);

  ntt(&mut y_hat[0]);
  ntt(&mut y_hat[1]);
  ntt(&mut y_hat[2]);
  ntt(&mut y_hat[3]);

  let mut u = [[0u16; N]; 4];
  matrix_ntt_mul_accumulate_prepared_transpose(ek, &y_hat, &mut u);

  inverse_ntt_montgomery_product_add_assign(&mut u[0], &e1[0]);
  inverse_ntt_montgomery_product_add_assign(&mut u[1], &e1[1]);
  inverse_ntt_montgomery_product_add_assign(&mut u[2], &e1[2]);
  inverse_ntt_montgomery_product_add_assign(&mut u[3], &e1[3]);

  let mut v = [0u16; N];
  multiply_ntts_accumulate(&mut v, &ek.t_hat, &y_hat);
  inverse_ntt_montgomery_product_add_assign(&mut v, &e2);
  decompress_message_add_assign(m, &mut v);

  let mut mask = 0xffu8;
  mask &= compress_encode_compare_poly::<11, 352>(&u[0], &expected[0..352]);
  mask &= compress_encode_compare_poly::<11, 352>(&u[1], &expected[352..704]);
  mask &= compress_encode_compare_poly::<11, 352>(&u[2], &expected[704..1056]);
  mask &= compress_encode_compare_poly::<11, 352>(&u[3], &expected[1056..1408]);
  mask &= compress_encode_compare_poly::<5, 160>(&v, &expected[1408..]);

  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  zeroize_poly(&mut v);
  mask
}

fn pke_encrypt_prepared<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  ek: &PreparedEncapsulationArithmetic<K>,
  m: &[u8; SEED_BYTES],
  r: &[u8; SEED_BYTES],
) -> [u8; CT_BYTES] {
  let mut y_hat = [[0u16; N]; K];
  let mut e1 = [[0u16; N]; K];
  let mut e2 = [0u16; N];
  if K == 2 {
    let (y0, y_tail) = y_hat.split_at_mut(1);
    let (y1, _) = y_tail.split_at_mut(1);
    sample_noise_pair::<ETA1_RANDOM_BYTES>(r, 0, &mut y0[0], 1, &mut y1[0]);

    let (e10, e1_tail) = e1.split_at_mut(1);
    let (e11, _) = e1_tail.split_at_mut(1);
    sample_noise_pair::<ETA2_RANDOM_BYTES>(r, 2, &mut e10[0], 3, &mut e11[0]);
    sample_noise::<ETA2_RANDOM_BYTES>(r, 4, &mut e2);
  } else {
    let mut nonce = 0u8;
    for poly in &mut y_hat {
      sample_noise::<ETA1_RANDOM_BYTES>(r, nonce, poly);
      nonce = nonce.wrapping_add(1);
    }
    for poly in &mut e1 {
      sample_noise::<ETA2_RANDOM_BYTES>(r, nonce, poly);
      nonce = nonce.wrapping_add(1);
    }
    sample_noise::<ETA2_RANDOM_BYTES>(r, nonce, &mut e2);
  }

  for poly in &mut y_hat {
    ntt(poly);
  }

  let mut u = [[0u16; N]; K];
  matrix_ntt_mul_accumulate_prepared_transpose(ek, &y_hat, &mut u);

  for i in 0..K {
    inverse_ntt_montgomery_product_add_assign(&mut u[i], &e1[i]);
  }

  let mut v = [0u16; N];
  multiply_ntts_accumulate(&mut v, &ek.t_hat, &y_hat);
  inverse_ntt_montgomery_product_add_assign(&mut v, &e2);

  decompress_message_add_assign(m, &mut v);

  let mut ciphertext = [0u8; CT_BYTES];
  for (i, poly) in u.iter().enumerate() {
    let start = i.strict_mul(POLY_DU_BYTES);
    compress_encode_poly::<DU>(poly, &mut ciphertext[start..start.strict_add(POLY_DU_BYTES)]);
  }

  compress_encode_poly::<DV>(&v, &mut ciphertext[POLY_DU_BYTES.strict_mul(K)..]);

  zeroize_polyvec(&mut e1);
  zeroize_poly(&mut e2);
  zeroize_polyvec(&mut y_hat);
  zeroize_polyvec(&mut u);
  zeroize_poly(&mut v);
  ciphertext
}

fn pke_decrypt<
  const K: usize,
  const DK_PKE_BYTES: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  dk_pke: &[u8],
  c: &[u8; CT_BYTES],
) -> [u8; SEED_BYTES] {
  let mut s_hat = prepare_decapsulation_key_slice::<K, DK_PKE_BYTES>(dk_pke);
  let message = pke_decrypt_prepared::<K, CT_BYTES, DU, DV, POLY_DU_BYTES, POLY_DV_BYTES>(&s_hat, c);
  zeroize_polyvec(&mut s_hat);
  message
}

fn pke_decrypt_prepared<
  const K: usize,
  const CT_BYTES: usize,
  const DU: usize,
  const DV: usize,
  const POLY_DU_BYTES: usize,
  const POLY_DV_BYTES: usize,
>(
  s_hat: &PolyVec<K>,
  c: &[u8; CT_BYTES],
) -> [u8; SEED_BYTES] {
  let mut u = [[0u16; N]; K];
  for (i, poly) in u.iter_mut().enumerate() {
    let start = i.strict_mul(POLY_DU_BYTES);
    decode_decompress_poly::<DU>(&c[start..start.strict_add(POLY_DU_BYTES)], poly);
  }

  let c2_start = POLY_DU_BYTES.strict_mul(K);
  let mut v_prime = [0u16; N];
  decode_decompress_poly::<DV>(&c[c2_start..c2_start.strict_add(POLY_DV_BYTES)], &mut v_prime);

  for poly in &mut u {
    ntt(poly);
  }

  let mut acc = [0u16; N];
  multiply_ntts_accumulate(&mut acc, s_hat, &u);
  inverse_ntt_montgomery_product(&mut acc);

  let mut message = [0u8; SEED_BYTES];
  subtract_compress_encode_message(&v_prime, &acc, &mut message);

  zeroize_poly(&mut acc);
  message
}

#[cfg(test)]
fn sample_ntt(rho: &[u8; SEED_BYTES], j: u8, i: u8) -> Poly {
  let mut out = [0u16; N];
  sample_ntt_into(rho, j, i, &mut out);
  out
}

#[cfg(test)]
fn sample_ntt_pair(rho: &[u8; SEED_BYTES], j0: u8, i0: u8, j1: u8, i1: u8) -> (Poly, Poly) {
  let mut out0 = [0u16; N];
  let mut out1 = [0u16; N];
  sample_ntt_pair_into(rho, j0, i0, j1, i1, &mut out0, &mut out1);
  (out0, out1)
}

#[inline(always)]
fn sample_ntt_four_materialized_into(rho: &[u8; SEED_BYTES], lanes: [(u8, u8); 4], out: [&mut Poly; 4]) {
  sample_ntt_quad_into(rho, lanes, out);
}

#[inline(always)]
fn matrix_ntt_mul_accumulate_prepared_transpose<const K: usize>(
  ek: &PreparedEncapsulationArithmetic<K>,
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
) {
  for (row, acc_i) in ek.a_transpose_hat.iter().zip(acc.iter_mut()) {
    multiply_ntts_accumulate(acc_i, row, rhs);
  }
}

fn sample_matrix_ntt_mul_accumulate_materialized<const K: usize>(
  rho: &[u8; SEED_BYTES],
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
  transpose: bool,
) {
  if K == 2 {
    if transpose {
      sample_matrix_ntt_mul_accumulate_materialized_k2_transpose(rho, rhs, acc);
    } else {
      sample_matrix_ntt_mul_accumulate_materialized_k2(rho, rhs, acc);
    }
    return;
  }

  if K == 4 {
    if transpose {
      sample_matrix_ntt_mul_accumulate_materialized_k4_transpose(rho, rhs, acc);
    } else {
      sample_matrix_ntt_mul_accumulate_materialized_k4(rho, rhs, acc);
    }
    return;
  }

  if K == 3 {
    if transpose {
      sample_matrix_ntt_mul_accumulate_materialized_k3_transpose(rho, rhs, acc);
    } else {
      sample_matrix_ntt_mul_accumulate_materialized_k3(rho, rhs, acc);
    }
    return;
  }

  let mut entry = 0usize;
  while entry.strict_add(3) < K.strict_mul(K) {
    let ((j0, i0), dst0, rhs0) = matrix_accumulate_coord::<K>(entry, transpose);
    let ((j1, i1), dst1, rhs1) = matrix_accumulate_coord::<K>(entry.strict_add(1), transpose);
    let ((j2, i2), dst2, rhs2) = matrix_accumulate_coord::<K>(entry.strict_add(2), transpose);
    let ((j3, i3), dst3, rhs3) = matrix_accumulate_coord::<K>(entry.strict_add(3), transpose);
    let mut a0 = [0u16; N];
    let mut a1 = [0u16; N];
    let mut a2 = [0u16; N];
    let mut a3 = [0u16; N];
    sample_ntt_four_materialized_into(
      rho,
      [(j0, i0), (j1, i1), (j2, i2), (j3, i3)],
      [&mut a0, &mut a1, &mut a2, &mut a3],
    );
    multiply_ntts_add_assign(&mut acc[dst0], &a0, &rhs[rhs0]);
    multiply_ntts_add_assign(&mut acc[dst1], &a1, &rhs[rhs1]);
    multiply_ntts_add_assign(&mut acc[dst2], &a2, &rhs[rhs2]);
    multiply_ntts_add_assign(&mut acc[dst3], &a3, &rhs[rhs3]);
    entry = entry.strict_add(4);
  }

  while entry.strict_add(1) < K.strict_mul(K) {
    let ((j0, i0), dst0, rhs0) = matrix_accumulate_coord::<K>(entry, transpose);
    let ((j1, i1), dst1, rhs1) = matrix_accumulate_coord::<K>(entry.strict_add(1), transpose);
    let mut a0 = [0u16; N];
    let mut a1 = [0u16; N];
    sample_ntt_pair_into(rho, j0, i0, j1, i1, &mut a0, &mut a1);
    multiply_ntts_add_assign(&mut acc[dst0], &a0, &rhs[rhs0]);
    multiply_ntts_add_assign(&mut acc[dst1], &a1, &rhs[rhs1]);
    entry = entry.strict_add(2);
  }

  if entry < K.strict_mul(K) {
    let ((j, i), dst, rhs_index) = matrix_accumulate_coord::<K>(entry, transpose);
    let mut a = [0u16; N];
    sample_ntt_into(rho, j, i, &mut a);
    multiply_ntts_add_assign(&mut acc[dst], &a, &rhs[rhs_index]);
  }
}

#[inline(always)]
fn sample_matrix_ntt_mul_accumulate_materialized_k2<const K: usize>(
  rho: &[u8; SEED_BYTES],
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
) {
  debug_assert_eq!(K, 2);

  let mut row0 = [[0u16; N]; 2];
  let (row0_left, row0_right) = row0.split_at_mut(1);
  sample_ntt_pair_into(rho, 0, 0, 1, 0, &mut row0_left[0], &mut row0_right[0]);
  multiply_ntts_accumulate_k2_refs(&mut acc[0], [&row0[0], &row0[1]], [&rhs[0], &rhs[1]]);
  zeroize_polyvec(&mut row0);

  let mut row1 = [[0u16; N]; 2];
  let (row1_left, row1_right) = row1.split_at_mut(1);
  sample_ntt_pair_into(rho, 0, 1, 1, 1, &mut row1_left[0], &mut row1_right[0]);
  multiply_ntts_accumulate_k2_refs(&mut acc[1], [&row1[0], &row1[1]], [&rhs[0], &rhs[1]]);
  zeroize_polyvec(&mut row1);
}

#[inline(always)]
fn sample_matrix_ntt_mul_accumulate_materialized_k2_transpose<const K: usize>(
  rho: &[u8; SEED_BYTES],
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
) {
  debug_assert_eq!(K, 2);

  let mut row0 = [[0u16; N]; 2];
  let (row0_left, row0_right) = row0.split_at_mut(1);
  sample_ntt_pair_into(rho, 0, 0, 0, 1, &mut row0_left[0], &mut row0_right[0]);
  multiply_ntts_accumulate_k2_refs(&mut acc[0], [&row0[0], &row0[1]], [&rhs[0], &rhs[1]]);
  zeroize_polyvec(&mut row0);

  let mut row1 = [[0u16; N]; 2];
  let (row1_left, row1_right) = row1.split_at_mut(1);
  sample_ntt_pair_into(rho, 1, 0, 1, 1, &mut row1_left[0], &mut row1_right[0]);
  multiply_ntts_accumulate_k2_refs(&mut acc[1], [&row1[0], &row1[1]], [&rhs[0], &rhs[1]]);
  zeroize_polyvec(&mut row1);
}

#[inline(always)]
fn sample_matrix_ntt_mul_accumulate_materialized_k3<const K: usize>(
  rho: &[u8; SEED_BYTES],
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
) {
  debug_assert_eq!(K, 3);

  let mut a0 = [0u16; N];
  let mut a1 = [0u16; N];
  let mut a2 = [0u16; N];
  let mut a3 = [0u16; N];
  sample_ntt_four_materialized_into(
    rho,
    [(0, 0), (1, 0), (2, 0), (0, 1)],
    [&mut a0, &mut a1, &mut a2, &mut a3],
  );
  multiply_ntts_accumulate_k3_refs(&mut acc[0], [&a0, &a1, &a2], [&rhs[0], &rhs[1], &rhs[2]]);
  let row1_a0 = a3;

  sample_ntt_four_materialized_into(
    rho,
    [(1, 1), (2, 1), (0, 2), (1, 2)],
    [&mut a0, &mut a1, &mut a2, &mut a3],
  );
  multiply_ntts_accumulate_k3_refs(&mut acc[1], [&row1_a0, &a0, &a1], [&rhs[0], &rhs[1], &rhs[2]]);
  let row2_a0 = a2;
  let row2_a1 = a3;

  sample_ntt_into(rho, 2, 2, &mut a0);
  multiply_ntts_accumulate_k3_refs(&mut acc[2], [&row2_a0, &row2_a1, &a0], [&rhs[0], &rhs[1], &rhs[2]]);
}

#[inline(always)]
fn sample_matrix_ntt_mul_accumulate_materialized_k3_transpose<const K: usize>(
  rho: &[u8; SEED_BYTES],
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
) {
  debug_assert_eq!(K, 3);

  let mut a0 = [0u16; N];
  let mut a1 = [0u16; N];
  let mut a2 = [0u16; N];
  let mut a3 = [0u16; N];
  sample_ntt_four_materialized_into(
    rho,
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [&mut a0, &mut a1, &mut a2, &mut a3],
  );
  multiply_ntts_accumulate_k3_refs(&mut acc[0], [&a0, &a1, &a2], [&rhs[0], &rhs[1], &rhs[2]]);
  let row1_a0 = a3;

  sample_ntt_four_materialized_into(
    rho,
    [(1, 1), (1, 2), (2, 0), (2, 1)],
    [&mut a0, &mut a1, &mut a2, &mut a3],
  );
  multiply_ntts_accumulate_k3_refs(&mut acc[1], [&row1_a0, &a0, &a1], [&rhs[0], &rhs[1], &rhs[2]]);
  let row2_a0 = a2;
  let row2_a1 = a3;

  sample_ntt_into(rho, 2, 2, &mut a0);
  multiply_ntts_accumulate_k3_refs(&mut acc[2], [&row2_a0, &row2_a1, &a0], [&rhs[0], &rhs[1], &rhs[2]]);
}

#[inline(always)]
fn sample_matrix_ntt_mul_accumulate_materialized_k4<const K: usize>(
  rho: &[u8; SEED_BYTES],
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
) {
  debug_assert_eq!(K, 4);

  macro_rules! accumulate_row {
    ($dst:literal, $coords:expr) => {{
      let mut a0 = [0u16; N];
      let mut a1 = [0u16; N];
      let mut a2 = [0u16; N];
      let mut a3 = [0u16; N];
      sample_ntt_four_materialized_into(rho, $coords, [&mut a0, &mut a1, &mut a2, &mut a3]);
      multiply_ntts_accumulate_k4_refs(
        &mut acc[$dst],
        [&a0, &a1, &a2, &a3],
        [&rhs[0], &rhs[1], &rhs[2], &rhs[3]],
      );
    }};
  }

  accumulate_row!(0, [(0, 0), (1, 0), (2, 0), (3, 0)]);
  accumulate_row!(1, [(0, 1), (1, 1), (2, 1), (3, 1)]);
  accumulate_row!(2, [(0, 2), (1, 2), (2, 2), (3, 2)]);
  accumulate_row!(3, [(0, 3), (1, 3), (2, 3), (3, 3)]);
}

#[inline(always)]
fn sample_matrix_ntt_mul_accumulate_materialized_k4_transpose<const K: usize>(
  rho: &[u8; SEED_BYTES],
  rhs: &PolyVec<K>,
  acc: &mut PolyVec<K>,
) {
  debug_assert_eq!(K, 4);

  macro_rules! accumulate_row {
    ($dst:literal, $coords:expr) => {{
      let mut a0 = [0u16; N];
      let mut a1 = [0u16; N];
      let mut a2 = [0u16; N];
      let mut a3 = [0u16; N];
      sample_ntt_four_materialized_into(rho, $coords, [&mut a0, &mut a1, &mut a2, &mut a3]);
      multiply_ntts_accumulate_k4_refs(
        &mut acc[$dst],
        [&a0, &a1, &a2, &a3],
        [&rhs[0], &rhs[1], &rhs[2], &rhs[3]],
      );
    }};
  }

  accumulate_row!(0, [(0, 0), (0, 1), (0, 2), (0, 3)]);
  accumulate_row!(1, [(1, 0), (1, 1), (1, 2), (1, 3)]);
  accumulate_row!(2, [(2, 0), (2, 1), (2, 2), (2, 3)]);
  accumulate_row!(3, [(3, 0), (3, 1), (3, 2), (3, 3)]);
}

#[inline]
fn matrix_accumulate_coord<const K: usize>(entry: usize, transpose: bool) -> ((u8, u8), usize, usize) {
  let dst = entry / K;
  let rhs = entry % K;
  let sample = if transpose {
    (dst as u8, rhs as u8)
  } else {
    (rhs as u8, dst as u8)
  };
  (sample, dst, rhs)
}

#[inline]
fn use_fused_matrix_accumulate<const K: usize>() -> bool {
  if cfg!(any(miri, feature = "portable-only")) {
    return true;
  }

  #[cfg(target_arch = "x86_64")]
  {
    !crate::platform::caps().has(crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41)
  }

  #[cfg(target_arch = "aarch64")]
  {
    K == 4
  }

  #[cfg(target_arch = "s390x")]
  {
    false
  }

  #[cfg(not(any(target_arch = "aarch64", target_arch = "s390x", target_arch = "x86_64")))]
  {
    true
  }
}

#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
fn use_s390x_vector_arithmetic() -> bool {
  crate::platform::caps().has(crate::platform::caps::s390x::VECTOR)
}

fn sample_ntt_into(rho: &[u8; SEED_BYTES], j: u8, i: u8, out: &mut Poly) {
  sample_ntt_from_xof_into(Shake128::xof_seeded_32_2(rho, j, i), out);
}

fn sample_ntt_pair_into(rho: &[u8; SEED_BYTES], j0: u8, i0: u8, j1: u8, i1: u8, out0: &mut Poly, out1: &mut Poly) {
  let (reader0, reader1) = Shake128::xof_seeded_32_2_pair(rho, (j0, i0), (j1, i1));
  sample_ntt_pair_from_xof_into(reader0, reader1, out0, out1);
}

fn sample_ntt_quad_into(rho: &[u8; SEED_BYTES], lanes: [(u8, u8); 4], out: [&mut Poly; 4]) {
  let (reader0, reader1, reader2, reader3) =
    Shake128::xof_seeded_32_2_quad(rho, lanes[0], lanes[1], lanes[2], lanes[3]);
  sample_ntt_quad_from_xof_into([reader0, reader1, reader2, reader3], out);
}

#[inline(always)]
fn sample_ntt_mul_accumulate(rho: &[u8; SEED_BYTES], j: u8, i: u8, rhs: &Poly, acc: &mut Poly) {
  sample_ntt_mul_accumulate_from_xof(Shake128::xof_seeded_32_2(rho, j, i), rhs, acc);
}

#[inline(always)]
fn sample_ntt_pair_mul_accumulate(
  rho: &[u8; SEED_BYTES],
  lane0: (u8, u8, &Poly),
  lane1: (u8, u8, &Poly),
  acc: &mut Poly,
) {
  let (j0, i0, rhs0) = lane0;
  let (j1, i1, rhs1) = lane1;
  let (reader0, reader1) = Shake128::xof_seeded_32_2_pair(rho, (j0, i0), (j1, i1));
  sample_ntt_pair_mul_accumulate_from_xof(reader0, rhs0, reader1, rhs1, acc);
}

#[inline(always)]
fn sample_ntt_quad_mul_accumulate(rho: &[u8; SEED_BYTES], lanes: [(u8, u8, &Poly); 4], acc: &mut Poly) {
  let (reader0, reader1, reader2, reader3) = Shake128::xof_seeded_32_2_quad(
    rho,
    (lanes[0].0, lanes[0].1),
    (lanes[1].0, lanes[1].1),
    (lanes[2].0, lanes[2].1),
    (lanes[3].0, lanes[3].1),
  );
  sample_ntt_quad_mul_accumulate_from_xof(
    [reader0, reader1, reader2, reader3],
    [lanes[0].2, lanes[1].2, lanes[2].2, lanes[3].2],
    acc,
  );
}

#[inline]
#[cfg(test)]
fn sample_ntt_input(rho: &[u8; SEED_BYTES], j: u8, i: u8) -> [u8; SEED_BYTES + 2] {
  let mut input = [0u8; SEED_BYTES + 2];
  input[..SEED_BYTES].copy_from_slice(rho);
  input[SEED_BYTES] = j;
  input[SEED_BYTES.strict_add(1)] = i;
  input
}

fn sample_ntt_from_xof_into(mut reader: impl Xof, out: &mut Poly) {
  let mut filled = 0usize;
  let mut buf = [0u8; SHAKE128_RATE_BYTES];

  while filled < N {
    reader.squeeze(&mut buf);
    sample_ntt_block(&buf, out, &mut filled);
  }
}

fn sample_ntt_pair_from_xof_into(
  mut reader0: Shake128XofReader,
  mut reader1: Shake128XofReader,
  out0: &mut Poly,
  out1: &mut Poly,
) {
  let mut filled0 = 0usize;
  let mut filled1 = 0usize;
  let mut buf0 = [0u8; SHAKE128_RATE_BYTES];
  let mut buf1 = [0u8; SHAKE128_RATE_BYTES];

  while filled0 < N && filled1 < N {
    Shake128XofReader::squeeze_pair(&mut reader0, &mut reader1, &mut buf0, &mut buf1);
    sample_ntt_pair_block(&buf0, out0, &mut filled0, &buf1, out1, &mut filled1);
  }
  while filled0 < N {
    reader0.squeeze(&mut buf0);
    sample_ntt_block(&buf0, out0, &mut filled0);
  }
  while filled1 < N {
    reader1.squeeze(&mut buf1);
    sample_ntt_block(&buf1, out1, &mut filled1);
  }
}

#[inline(always)]
fn sample_ntt_pair_block(
  buf0: &[u8; SHAKE128_RATE_BYTES],
  out0: &mut Poly,
  filled0: &mut usize,
  buf1: &[u8; SHAKE128_RATE_BYTES],
  out1: &mut Poly,
  filled1: &mut usize,
) {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    // SAFETY: aarch64 NEON SampleNTT candidate extraction because:
    // 1. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 2. `buf0` and `buf1` are fixed SHAKE128 rate blocks, and the NEON helper only reads complete
    //    48-byte chunks within those blocks.
    // 3. `out0`/`out1` writes stay in-bounds: the NEON path falls back to the scalar bounded parser
    //    whenever a full block could overflow either output polynomial.
    // 4. Rejection branches and write positions depend only on public matrix-A samples, not secret key,
    //    noise, message, or shared-secret material.
    unsafe {
      sample_ntt_pair_block_neon(buf0, out0, filled0, buf1, out1, filled1);
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  {
    sample_ntt_pair_block_scalar(buf0, out0, filled0, buf1, out1, filled1);
  }
}

#[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
#[inline(always)]
fn sample_ntt_pair_block_scalar(
  buf0: &[u8; SHAKE128_RATE_BYTES],
  out0: &mut Poly,
  filled0: &mut usize,
  buf1: &[u8; SHAKE128_RATE_BYTES],
  out1: &mut Poly,
  filled1: &mut usize,
) {
  const MAX_CANDIDATES: usize = (SHAKE128_RATE_BYTES / 3) * 2;

  if N.strict_sub(*filled0) < MAX_CANDIDATES || N.strict_sub(*filled1) < MAX_CANDIDATES {
    sample_ntt_block(buf0, out0, filled0);
    sample_ntt_block(buf1, out1, filled1);
    return;
  }

  let mut n0 = *filled0;
  let mut n1 = *filled1;
  let mut offset = 0usize;
  while offset.strict_add(2) < SHAKE128_RATE_BYTES {
    let a0 = buf0[offset];
    let a1 = buf0[offset.strict_add(1)];
    let a2 = buf0[offset.strict_add(2)];
    let b0 = buf1[offset];
    let b1 = buf1[offset.strict_add(1)];
    let b2 = buf1[offset.strict_add(2)];

    let d0 = u16::from(a0) | (u16::from(a1 & 0x0f) << 8);
    let d1 = (u16::from(a1) >> 4) | (u16::from(a2) << 4);
    let e0 = u16::from(b0) | (u16::from(b1 & 0x0f) << 8);
    let e1 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);

    if d0 < Q {
      out0[n0] = d0;
      n0 = n0.strict_add(1);
    }
    if d1 < Q {
      out0[n0] = d1;
      n0 = n0.strict_add(1);
    }
    if e0 < Q {
      out1[n1] = e0;
      n1 = n1.strict_add(1);
    }
    if e1 < Q {
      out1[n1] = e1;
      n1 = n1.strict_add(1);
    }

    offset = offset.strict_add(3);
  }

  *filled0 = n0;
  *filled1 = n1;
}

#[cfg(all(
  target_arch = "aarch64",
  not(any(target_os = "macos", target_os = "linux")),
  not(miri),
  not(feature = "portable-only")
))]
macro_rules! sample_ntt_extract_16_candidate_vectors_neon {
  ($input:expr) => {{
    let mask = vdupq_n_u16(0x0f);
    // SAFETY: 48-byte deinterleaved load because:
    // 1. The caller guarantees `$input..$input+48` is readable.
    // 2. `vld3q_u8` reads exactly three 16-byte vectors from that public SampleNTT byte block.
    // 3. The surrounding function is gated by `#[target_feature(enable = "neon")]`.
    let triples = unsafe { vld3q_u8($input) };

    let a0 = vmovl_u8(vget_low_u8(triples.0));
    let a1 = vmovl_u8(vget_low_u8(triples.1));
    let a2 = vmovl_u8(vget_low_u8(triples.2));
    let d0 = vorrq_u16(a0, vshlq_n_u16(vandq_u16(a1, mask), 8));
    let d1 = vorrq_u16(vshrq_n_u16(a1, 4), vshlq_n_u16(a2, 4));

    let a0 = vmovl_u8(vget_high_u8(triples.0));
    let a1 = vmovl_u8(vget_high_u8(triples.1));
    let a2 = vmovl_u8(vget_high_u8(triples.2));
    let d2 = vorrq_u16(a0, vshlq_n_u16(vandq_u16(a1, mask), 8));
    let d3 = vorrq_u16(vshrq_n_u16(a1, 4), vshlq_n_u16(a2, 4));

    (d0, d1, d2, d3)
  }};
}

#[cfg(all(
  target_arch = "aarch64",
  not(any(target_os = "macos", target_os = "linux")),
  not(miri),
  not(feature = "portable-only")
))]
macro_rules! sample_ntt_store_public_candidate_unchecked {
  ($out:expr, $n:ident, $candidate:expr) => {{
    let candidate = $candidate;
    if candidate < Q {
      // SAFETY: unchecked public-sample store because:
      // 1. The caller reserved enough output capacity for every possible accepted candidate in this
      //    block, so the current accepted candidate is in bounds.
      // 2. `$out` comes from a unique mutable polynomial borrow for the duration of parsing.
      // 3. The branch depends only on public matrix-A sample bytes.
      unsafe {
        *$out.add($n) = candidate;
      }
      $n = $n.strict_add(1);
    }
  }};
}

#[cfg(all(
  target_arch = "aarch64",
  not(any(target_os = "macos", target_os = "linux")),
  not(miri),
  not(feature = "portable-only")
))]
macro_rules! sample_ntt_store_candidate_vectors_neon {
  ($out:expr, $n:ident, $candidates:expr) => {{
    let (d0_lo, d1_lo, d0_hi, d1_hi) = $candidates;
    macro_rules! store_pair_lane {
      ($lane:literal, $d0:expr, $d1:expr) => {{
        sample_ntt_store_public_candidate_unchecked!($out, $n, vgetq_lane_u16::<$lane>($d0));
        sample_ntt_store_public_candidate_unchecked!($out, $n, vgetq_lane_u16::<$lane>($d1));
      }};
    }

    store_pair_lane!(0, d0_lo, d1_lo);
    store_pair_lane!(1, d0_lo, d1_lo);
    store_pair_lane!(2, d0_lo, d1_lo);
    store_pair_lane!(3, d0_lo, d1_lo);
    store_pair_lane!(4, d0_lo, d1_lo);
    store_pair_lane!(5, d0_lo, d1_lo);
    store_pair_lane!(6, d0_lo, d1_lo);
    store_pair_lane!(7, d0_lo, d1_lo);
    store_pair_lane!(0, d0_hi, d1_hi);
    store_pair_lane!(1, d0_hi, d1_hi);
    store_pair_lane!(2, d0_hi, d1_hi);
    store_pair_lane!(3, d0_hi, d1_hi);
    store_pair_lane!(4, d0_hi, d1_hi);
    store_pair_lane!(5, d0_hi, d1_hi);
    store_pair_lane!(6, d0_hi, d1_hi);
    store_pair_lane!(7, d0_hi, d1_hi);
  }};
}

#[cfg(all(
  target_arch = "aarch64",
  not(any(target_os = "macos", target_os = "linux")),
  not(miri),
  not(feature = "portable-only")
))]
#[target_feature(enable = "neon")]
/// # Safety
///
/// `input` must point to 48 readable bytes. The caller must guarantee that the active CPU supports
/// NEON. `out` must be a unique 32-candidate destination.
unsafe fn sample_ntt_extract_16_candidates_neon(input: *const u8, out: &mut [u16; 32]) {
  let (d0_lo, d1_lo, d0_hi, d1_hi) = sample_ntt_extract_16_candidate_vectors_neon!(input);

  // SAFETY: store the first 16 interleaved candidates because:
  // 1. `out` is a unique `[u16; 32]` destination.
  // 2. `vst2q_u16` writes exactly 16 u16 values from two eight-lane vectors.
  // 3. The surrounding function is gated by `#[target_feature(enable = "neon")]`.
  unsafe {
    vst2q_u16(out.as_mut_ptr(), uint16x8x2_t(d0_lo, d1_lo));
  }
  // SAFETY: store the last 16 interleaved candidates because:
  // 1. `out.as_mut_ptr().add(16)..add(32)` is inside the 32-candidate destination.
  // 2. `vst2q_u16` writes exactly 16 u16 values from two eight-lane vectors.
  // 3. The surrounding function is gated by `#[target_feature(enable = "neon")]`.
  unsafe {
    vst2q_u16(out.as_mut_ptr().add(16), uint16x8x2_t(d0_hi, d1_hi));
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn sample_ntt_pair_block_neon(
  buf0: &[u8; SHAKE128_RATE_BYTES],
  out0: &mut Poly,
  filled0: &mut usize,
  buf1: &[u8; SHAKE128_RATE_BYTES],
  out1: &mut Poly,
  filled1: &mut usize,
) {
  const MAX_CANDIDATES: usize = (SHAKE128_RATE_BYTES / 3) * 2;

  if N.strict_sub(*filled0) < MAX_CANDIDATES || N.strict_sub(*filled1) < MAX_CANDIDATES {
    sample_ntt_block(buf0, out0, filled0);
    sample_ntt_block(buf1, out1, filled1);
    return;
  }

  #[cfg(any(target_os = "macos", target_os = "linux"))]
  {
    let n0 = *filled0;
    let n1 = *filled1;
    // SAFETY: aarch64 compact assembly SampleNTT block parsing because:
    // 1. `buf0` and `buf1` are full 168-byte SHAKE128 rate blocks.
    // 2. The preflight above reserves capacity for all 112 candidates each block can produce.
    // 3. `out0` and `out1` come from distinct mutable polynomial borrows.
    // 4. Rejection branches and write positions depend only on public matrix-A XOF bytes.
    let (count0, count1) = unsafe {
      (
        aarch64::sample_ntt_rej_uniform_block_asm(out0.as_mut_ptr().add(n0), buf0.as_ptr()),
        aarch64::sample_ntt_rej_uniform_block_asm(out1.as_mut_ptr().add(n1), buf1.as_ptr()),
      )
    };
    *filled0 = n0.strict_add(count0);
    *filled1 = n1.strict_add(count1);
  }

  #[cfg(not(any(target_os = "macos", target_os = "linux")))]
  {
    const NEON_TRIPLES: usize = 16;
    const NEON_BYTES: usize = NEON_TRIPLES * 3;

    let mut n0 = *filled0;
    let mut n1 = *filled1;
    let out0_ptr = out0.as_mut_ptr();
    let out1_ptr = out1.as_mut_ptr();
    let mut offset = 0usize;

    while offset.strict_add(NEON_BYTES) <= SHAKE128_RATE_BYTES {
      // SAFETY: fixed 48-byte NEON extraction plus unchecked public-sample stores because:
      // 1. `offset + NEON_BYTES <= SHAKE128_RATE_BYTES`, so both inputs name complete chunks.
      // 2. The preflight above ensures each output has room for all candidates from the whole rate block.
      // 3. Rejection branches and write counts depend only on public matrix-A samples.
      let candidates0 = sample_ntt_extract_16_candidate_vectors_neon!(buf0.as_ptr().add(offset));
      let candidates1 = sample_ntt_extract_16_candidate_vectors_neon!(buf1.as_ptr().add(offset));
      sample_ntt_store_candidate_vectors_neon!(out0_ptr, n0, candidates0);
      sample_ntt_store_candidate_vectors_neon!(out1_ptr, n1, candidates1);

      offset = offset.strict_add(NEON_BYTES);
    }

    while offset.strict_add(2) < SHAKE128_RATE_BYTES {
      let a0 = buf0[offset];
      let a1 = buf0[offset.strict_add(1)];
      let a2 = buf0[offset.strict_add(2)];
      let b0 = buf1[offset];
      let b1 = buf1[offset.strict_add(1)];
      let b2 = buf1[offset.strict_add(2)];

      let d0 = u16::from(a0) | (u16::from(a1 & 0x0f) << 8);
      let d1 = (u16::from(a1) >> 4) | (u16::from(a2) << 4);
      let e0 = u16::from(b0) | (u16::from(b1 & 0x0f) << 8);
      let e1 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);

      if d0 < Q {
        out0[n0] = d0;
        n0 = n0.strict_add(1);
      }
      if d1 < Q {
        out0[n0] = d1;
        n0 = n0.strict_add(1);
      }
      if e0 < Q {
        out1[n1] = e0;
        n1 = n1.strict_add(1);
      }
      if e1 < Q {
        out1[n1] = e1;
        n1 = n1.strict_add(1);
      }

      offset = offset.strict_add(3);
    }

    *filled0 = n0;
    *filled1 = n1;
  }
}

fn sample_ntt_quad_from_xof_into(mut readers: [Shake128XofReader; 4], out: [&mut Poly; 4]) {
  let mut filled = [0usize; 4];
  let mut bufs = [[0u8; SHAKE128_RATE_BYTES]; 4];
  let [out0, out1, out2, out3] = out;

  while filled[0] < N && filled[1] < N && filled[2] < N && filled[3] < N {
    let [reader0, reader1, reader2, reader3] = &mut readers;

    #[cfg(all(
      target_arch = "aarch64",
      target_endian = "little",
      not(miri),
      not(feature = "portable-only")
    ))]
    {
      Shake128XofReader::with_quad_rate_block(reader0, reader1, reader2, reader3, |state0, state1, state2, state3| {
        sample_ntt_quad_state_block([state0, state1, state2, state3], [out0, out1, out2, out3], &mut filled);
      });
    }

    #[cfg(not(all(
      target_arch = "aarch64",
      target_endian = "little",
      not(miri),
      not(feature = "portable-only")
    )))]
    {
      let [buf0, buf1, buf2, buf3] = &mut bufs;
      Shake128XofReader::squeeze_quad(reader0, reader1, reader2, reader3, buf0, buf1, buf2, buf3);
      sample_ntt_quad_block(&bufs, [out0, out1, out2, out3], &mut filled);
    }
  }

  while filled[0] < N {
    readers[0].squeeze(&mut bufs[0]);
    sample_ntt_block(&bufs[0], out0, &mut filled[0]);
  }
  while filled[1] < N {
    readers[1].squeeze(&mut bufs[1]);
    sample_ntt_block(&bufs[1], out1, &mut filled[1]);
  }
  while filled[2] < N {
    readers[2].squeeze(&mut bufs[2]);
    sample_ntt_block(&bufs[2], out2, &mut filled[2]);
  }
  while filled[3] < N {
    readers[3].squeeze(&mut bufs[3]);
    sample_ntt_block(&bufs[3], out3, &mut filled[3]);
  }
}

struct SampleNttProduct<'a> {
  rhs: &'a Poly,
  chunk: [u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  filled: usize,
  chunk_len: usize,
}

impl<'a> SampleNttProduct<'a> {
  #[inline(always)]
  fn new(rhs: &'a Poly) -> Self {
    Self {
      rhs,
      chunk: [0u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
      filled: 0,
      chunk_len: 0,
    }
  }

  #[inline(always)]
  fn is_done(&self) -> bool {
    self.filled == N
  }

  #[inline(always)]
  #[cfg(not(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  )))]
  fn push_candidate(&mut self, value: u16) -> Option<usize> {
    if value >= Q {
      return None;
    }

    self.push_accepted_candidate(value)
  }

  #[inline(always)]
  #[cfg(not(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  )))]
  fn push_accepted_candidate(&mut self, value: u16) -> Option<usize> {
    debug_assert!(value < Q);
    if self.is_done() {
      return None;
    }

    self.chunk[self.chunk_len] = value;
    self.chunk_len = self.chunk_len.strict_add(1);
    self.filled = self.filled.strict_add(1);

    if self.chunk_len == SAMPLE_NTT_ACC_CHUNK_COEFFS {
      let coeff_offset = self.filled.strict_sub(SAMPLE_NTT_ACC_CHUNK_COEFFS);
      self.chunk_len = 0;
      Some(coeff_offset)
    } else {
      None
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  #[inline(always)]
  fn absorb_candidate(&mut self, value: u16, acc: &mut Poly) {
    if let Some(coeff_offset) = self.push_candidate(value) {
      multiply_ntts_add_assign_chunk(acc, &self.chunk, self.rhs, coeff_offset);
    }
  }

  #[inline(always)]
  fn absorb_block(&mut self, buf: &[u8; SHAKE128_RATE_BYTES], acc: &mut Poly) {
    #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
    {
      // SAFETY: aarch64 NEON fused SampleNTT block dispatch because:
      // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
      // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
      // 3. `buf` is one fixed SHAKE128 rate block and `self` owns its 16-coefficient staging chunk.
      // 4. The memory access schedule depends only on public ML-KEM dimensions and rejection outcomes
      //    from public matrix sampling, not secret coefficient values.
      unsafe {
        sample_ntt_product_absorb_block_neon(self, buf, acc);
      }
    }

    #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
    {
      if crate::platform::caps().has(crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41) {
        // SAFETY: x86_64 AVX2 fused SampleNTT block dispatch because:
        // 1. Runtime capability detection confirmed AVX2 and SSE4.1 before entering the target-feature
        //    function.
        // 2. `buf` is one fixed SHAKE128 rate block and `self` owns its 16-coefficient staging chunk.
        // 3. The borrow checker guarantees `acc` is not aliased by the read-only multiplicand in `self`.
        // 4. The memory access schedule depends only on public ML-KEM dimensions and rejection outcomes
        //    from public matrix sampling, not secret coefficient values.
        unsafe {
          return sample_ntt_product_absorb_block_avx2(self, buf, acc);
        }
      }
    }

    #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
    self.absorb_block_scalar(buf, acc);
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  #[inline(always)]
  fn absorb_block_scalar(&mut self, buf: &[u8; SHAKE128_RATE_BYTES], acc: &mut Poly) {
    for chunk in buf.chunks_exact(3) {
      let d1 = u16::from(chunk[0]) | (u16::from(chunk[1] & 0x0f) << 8);
      let d2 = (u16::from(chunk[1]) >> 4) | (u16::from(chunk[2]) << 4);

      self.absorb_candidate(d1, acc);
      self.absorb_candidate(d2, acc);
      if self.is_done() {
        break;
      }
    }
  }

  #[inline(always)]
  fn finish(&self) {
    debug_assert_eq!(self.filled, N);
    debug_assert_eq!(self.chunk_len, 0);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
macro_rules! multiply_ntts_add_assign_chunk_neon_body {
  ($acc:expr, $a_ptr:expr, $b:expr, $coeff_offset:expr) => {{
    let gamma_offset = $coeff_offset / 2;
    let a_pair = vld2q_u16($a_ptr);
    let b_pair = vld2q_u16($b.as_ptr().add($coeff_offset));
    let gamma = vld1q_s16(GAMMAS_MONT.as_ptr().add(gamma_offset));

    let a0 = vreinterpretq_s16_u16(a_pair.0);
    let a1 = vreinterpretq_s16_u16(a_pair.1);
    let b0 = vreinterpretq_s16_u16(b_pair.0);
    let b1 = vreinterpretq_s16_u16(b_pair.1);

    let a1b1 = mul_i16x8_to_i32x4_neon(a1, b1);
    let a1b1 = montgomery_reduce_i32x8_neon(a1b1.0, a1b1.1);
    let a0b0 = mul_i16x8_to_i32x4_neon(a0, b0);
    let a1b1_gamma = mul_i16x8_to_i32x4_neon(a1b1, gamma);
    let c0 = signed_to_mod_q_s16x8(montgomery_reduce_i32x8_neon(
      vaddq_s32(a0b0.0, a1b1_gamma.0),
      vaddq_s32(a0b0.1, a1b1_gamma.1),
    ));

    let a0b1 = mul_i16x8_to_i32x4_neon(a0, b1);
    let a1b0 = mul_i16x8_to_i32x4_neon(a1, b0);
    let c1 = signed_to_mod_q_s16x8(montgomery_reduce_i32x8_neon(
      vaddq_s32(a0b1.0, a1b0.0),
      vaddq_s32(a0b1.1, a1b0.1),
    ));

    let acc_pair = vld2q_u16($acc.as_ptr().add($coeff_offset));
    let out = uint16x8x2_t(add_mod_u16x8(acc_pair.0, c0), add_mod_u16x8(acc_pair.1, c1));
    vst2q_u16($acc.as_mut_ptr().add($coeff_offset), out);
  }};
}

#[cfg(all(
  target_arch = "aarch64",
  not(any(target_os = "macos", target_os = "linux")),
  not(miri),
  not(feature = "portable-only")
))]
macro_rules! sample_ntt_product_absorb_candidate_neon {
  ($product:expr, $candidate:expr, $acc:expr) => {{
    if let Some(coeff_offset) = $product.push_candidate($candidate) {
      // SAFETY: inlined NEON chunk multiply-accumulate from SampleNTT staging because:
      // 1. `product.chunk` has exactly 16 coefficients filled by `push_candidate` before returning
      //    `Some(coeff_offset)`.
      // 2. `coeff_offset` is emitted only at 16-coefficient boundaries and `product.filled <= N`, so
      //    `coeff_offset..coeff_offset + 16` stays inside `acc` and `product.rhs`.
      // 3. The borrow checker guarantees `acc` is not aliased by the read-only staging chunk or RHS.
      // 4. The surrounding function is gated by `#[target_feature(enable = "neon")]`, and the caller
      //    proves NEON availability.
      unsafe {
        multiply_ntts_add_assign_chunk_neon_body!($acc, $product.chunk.as_ptr(), $product.rhs, coeff_offset);
      }
    }
  }};
}

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[target_feature(enable = "neon")]
#[inline]
/// # Safety
///
/// `a_ptr` must point to 16 readable accepted SampleNTT coefficients. `coeff_offset` must be a
/// 16-coefficient boundary inside `acc` and `b`. The caller must guarantee that the active CPU
/// supports NEON.
unsafe fn multiply_ntts_add_assign_chunk_neon_ptr(acc: &mut Poly, a_ptr: *const u16, b: &Poly, coeff_offset: usize) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= N);

  // SAFETY: pointer-backed NEON chunk multiply-accumulate because:
  // 1. `a_ptr` points to 16 contiguous accepted SampleNTT coefficients by this function's caller
  //    contract.
  // 2. `coeff_offset` is checked to keep the 16-coefficient `acc` and `b` accesses in bounds.
  // 3. The borrow checker guarantees `acc` is not aliased by read-only `b`; `a_ptr` points to local
  //    scratch owned by the caller.
  // 4. The surrounding function is gated by `#[target_feature(enable = "neon")]`.
  unsafe {
    multiply_ntts_add_assign_chunk_neon_body!(acc, a_ptr, b, coeff_offset);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn sample_ntt_product_absorb_block_neon(
  product: &mut SampleNttProduct<'_>,
  buf: &[u8; SHAKE128_RATE_BYTES],
  acc: &mut Poly,
) {
  // SAFETY: Pointer-backed product parsing over one byte buffer because:
  // 1. `buf.as_ptr()` comes from a fixed `[u8; SHAKE128_RATE_BYTES]` rate block.
  // 2. The helper reads only offsets `< SHAKE128_RATE_BYTES`.
  // 3. The surrounding function is gated by `#[target_feature(enable = "neon")]`, and callers prove
  //    NEON availability before selecting the aarch64 backend.
  unsafe {
    sample_ntt_product_absorb_rate_ptr_neon(product, buf.as_ptr(), acc);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
/// # Safety
///
/// `rate_ptr` must point to one readable SHAKE128 rate block containing at least
/// `SHAKE128_RATE_BYTES` bytes. The caller must guarantee that the active CPU supports NEON.
unsafe fn sample_ntt_product_absorb_rate_ptr_neon(
  product: &mut SampleNttProduct<'_>,
  rate_ptr: *const u8,
  acc: &mut Poly,
) {
  #[cfg(any(target_os = "macos", target_os = "linux"))]
  {
    const MAX_CANDIDATES: usize = (SHAKE128_RATE_BYTES / 3) * 2;
    const SCRATCH_CANDIDATES: usize = MAX_CANDIDATES.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS);

    if product.is_done() {
      return;
    }

    let prefix_len = product.chunk_len;
    let base_offset = product.filled.strict_sub(prefix_len);
    let mut accepted = core::mem::MaybeUninit::<[u16; SCRATCH_CANDIDATES]>::uninit();
    let accepted_ptr = accepted.as_mut_ptr().cast::<u16>();

    if prefix_len != 0 {
      // SAFETY: carry partial accepted chunk into scratch because:
      // 1. `prefix_len == product.chunk_len`, and staging chunks are always shorter than the fixed
      //    16-coefficient chunk width.
      // 2. `accepted_ptr` points to `SCRATCH_CANDIDATES >= 16` writable u16 slots.
      // 3. The source and local scratch destination cannot overlap.
      unsafe {
        core::ptr::copy_nonoverlapping(product.chunk.as_ptr(), accepted_ptr, prefix_len);
      }
    }

    // SAFETY: aarch64 compact rejection parser because:
    // 1. `rate_ptr` names a readable SHAKE128 rate block. Callers pass either a `[u8; 168]` buffer or
    //    the first 168 bytes of an initialized Keccak state on little-endian aarch64.
    // 2. `accepted_ptr.add(prefix_len)` has room for every candidate the block can produce plus the
    //    assembly's fixed 8-lane compact stores.
    // 3. Rejection and write positions depend only on public matrix-A samples.
    let accepted_len = unsafe { aarch64::sample_ntt_rej_uniform_block_asm(accepted_ptr.add(prefix_len), rate_ptr) };
    debug_assert!(accepted_len <= MAX_CANDIDATES);

    let take = accepted_len.min(N.strict_sub(product.filled));
    let total_len = prefix_len.strict_add(take);
    let full_len = total_len - (total_len % SAMPLE_NTT_ACC_CHUNK_COEFFS);

    let mut chunk_start = 0usize;
    while chunk_start < full_len {
      let coeff_offset = base_offset.strict_add(chunk_start);
      // SAFETY: compacted accepted-candidate chunk multiply because:
      // 1. `chunk_start + 16 <= full_len <= total_len`, so the pointer names 16 initialized accepted
      //    coefficients in `accepted_ptr`.
      // 2. `coeff_offset` starts at the prior full-chunk boundary and advances in 16-coefficient steps;
      //    `take` is capped at the remaining polynomial length, keeping every chunk in bounds.
      // 3. `accepted_ptr` is local scratch and cannot alias `acc` or `product.rhs`.
      // 4. This function is already gated by `#[target_feature(enable = "neon")]`.
      unsafe {
        multiply_ntts_add_assign_chunk_neon_ptr(acc, accepted_ptr.add(chunk_start), product.rhs, coeff_offset);
      }
      chunk_start = chunk_start.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS);
    }

    let remainder_len = total_len.strict_sub(full_len);
    if remainder_len != 0 {
      // SAFETY: carry tail accepted coefficients back into staging because:
      // 1. `remainder_len < 16`, so the destination prefix of `product.chunk` is in bounds.
      // 2. `full_len..total_len` lies inside the initialized prefix/new-accepted region of
      //    `accepted_ptr`.
      // 3. The local scratch source cannot overlap the product staging array.
      unsafe {
        core::ptr::copy_nonoverlapping(accepted_ptr.add(full_len), product.chunk.as_mut_ptr(), remainder_len);
      }
    }
    product.chunk_len = remainder_len;
    product.filled = product.filled.strict_add(take);
  }

  #[cfg(not(any(target_os = "macos", target_os = "linux")))]
  {
    const NEON_TRIPLES: usize = 16;
    const NEON_BYTES: usize = NEON_TRIPLES * 3;
    const NEON_CANDIDATES: usize = NEON_TRIPLES * 2;

    let mut candidates = [0u16; NEON_CANDIDATES];
    let mut offset = 0usize;

    while offset.strict_add(NEON_BYTES) <= SHAKE128_RATE_BYTES {
      // SAFETY: `offset + NEON_BYTES <= SHAKE128_RATE_BYTES`, so the helper reads exactly one complete
      // 48-byte chunk from the fixed rate block and writes 32 initialized candidates.
      unsafe {
        sample_ntt_extract_16_candidates_neon(rate_ptr.add(offset), &mut candidates);
      }

      for &candidate in &candidates {
        sample_ntt_product_absorb_candidate_neon!(product, candidate, acc);
        if product.is_done() {
          return;
        }
      }

      offset = offset.strict_add(NEON_BYTES);
    }

    while offset.strict_add(2) < SHAKE128_RATE_BYTES {
      // SAFETY: `offset + 2 < SHAKE128_RATE_BYTES`, and the caller guarantees `rate_ptr` names one
      // full readable SHAKE128 rate block.
      let (b0, b1, b2) = unsafe {
        (
          *rate_ptr.add(offset),
          *rate_ptr.add(offset.strict_add(1)),
          *rate_ptr.add(offset.strict_add(2)),
        )
      };
      let d1 = u16::from(b0) | (u16::from(b1 & 0x0f) << 8);
      let d2 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);

      sample_ntt_product_absorb_candidate_neon!(product, d1, acc);
      sample_ntt_product_absorb_candidate_neon!(product, d2, acc);
      if product.is_done() {
        break;
      }

      offset = offset.strict_add(3);
    }
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn sample_ntt_product_absorb_block_avx2(
  product: &mut SampleNttProduct<'_>,
  buf: &[u8; SHAKE128_RATE_BYTES],
  acc: &mut Poly,
) {
  for chunk in buf.chunks_exact(3) {
    let d1 = u16::from(chunk[0]) | (u16::from(chunk[1] & 0x0f) << 8);
    let d2 = (u16::from(chunk[1]) >> 4) | (u16::from(chunk[2]) << 4);

    if let Some(coeff_offset) = product.push_candidate(d1) {
      multiply_ntts_add_assign_chunk_avx2(acc, &product.chunk, product.rhs, coeff_offset);
    }
    if let Some(coeff_offset) = product.push_candidate(d2) {
      multiply_ntts_add_assign_chunk_avx2(acc, &product.chunk, product.rhs, coeff_offset);
    }
    if product.is_done() {
      break;
    }
  }
}

#[inline(always)]
fn sample_ntt_mul_accumulate_from_xof(mut reader: impl Xof, rhs: &Poly, acc: &mut Poly) {
  let mut product = SampleNttProduct::new(rhs);
  let mut buf = [0u8; SHAKE128_RATE_BYTES];

  while !product.is_done() {
    reader.squeeze(&mut buf);
    product.absorb_block(&buf, acc);
  }

  product.finish();
}

#[inline(always)]
fn sample_ntt_pair_mul_accumulate_from_xof(
  mut reader0: Shake128XofReader,
  rhs0: &Poly,
  mut reader1: Shake128XofReader,
  rhs1: &Poly,
  acc: &mut Poly,
) {
  let mut product0 = SampleNttProduct::new(rhs0);
  let mut product1 = SampleNttProduct::new(rhs1);
  let mut buf0 = [0u8; SHAKE128_RATE_BYTES];
  let mut buf1 = [0u8; SHAKE128_RATE_BYTES];

  while !product0.is_done() && !product1.is_done() {
    Shake128XofReader::squeeze_pair(&mut reader0, &mut reader1, &mut buf0, &mut buf1);
    product0.absorb_block(&buf0, acc);
    product1.absorb_block(&buf1, acc);
  }
  while !product0.is_done() {
    reader0.squeeze(&mut buf0);
    product0.absorb_block(&buf0, acc);
  }
  while !product1.is_done() {
    reader1.squeeze(&mut buf1);
    product1.absorb_block(&buf1, acc);
  }

  product0.finish();
  product1.finish();
}

#[inline(always)]
fn sample_ntt_quad_mul_accumulate_from_xof(mut readers: [Shake128XofReader; 4], rhs: [&Poly; 4], acc: &mut Poly) {
  let mut products = [
    SampleNttProduct::new(rhs[0]),
    SampleNttProduct::new(rhs[1]),
    SampleNttProduct::new(rhs[2]),
    SampleNttProduct::new(rhs[3]),
  ];

  #[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    not(miri),
    not(feature = "portable-only")
  ))]
  {
    while !products[0].is_done() && !products[1].is_done() && !products[2].is_done() && !products[3].is_done() {
      let [reader0, reader1, reader2, reader3] = &mut readers;
      Shake128XofReader::with_quad_rate_block(reader0, reader1, reader2, reader3, |state0, state1, state2, state3| {
        let rate_ptrs = [
          state0.as_ptr().cast::<u8>(),
          state1.as_ptr().cast::<u8>(),
          state2.as_ptr().cast::<u8>(),
          state3.as_ptr().cast::<u8>(),
        ];
        // SAFETY: Pointer-backed fused product parsing over Keccak state memory because:
        // 1. This path is little-endian aarch64 only, so each `u64` state has XOF byte order in memory.
        // 2. Each Keccak state has 25 u64 lanes (200 bytes), covering the first 168-byte SHAKE128 rate
        //    block.
        // 3. The helper reads only offsets `< SHAKE128_RATE_BYTES`, and its rejection branches depend only
        //    on public matrix-A samples.
        // 4. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
        unsafe {
          sample_ntt_product_absorb_rate_ptr_neon(&mut products[0], rate_ptrs[0], acc);
          sample_ntt_product_absorb_rate_ptr_neon(&mut products[1], rate_ptrs[1], acc);
          sample_ntt_product_absorb_rate_ptr_neon(&mut products[2], rate_ptrs[2], acc);
          sample_ntt_product_absorb_rate_ptr_neon(&mut products[3], rate_ptrs[3], acc);
        }
      });
    }
  }

  let mut bufs = [[0u8; SHAKE128_RATE_BYTES]; 4];

  #[cfg(not(all(
    target_arch = "aarch64",
    target_endian = "little",
    not(miri),
    not(feature = "portable-only")
  )))]
  {
    while !products[0].is_done() && !products[1].is_done() && !products[2].is_done() && !products[3].is_done() {
      let [reader0, reader1, reader2, reader3] = &mut readers;
      let [buf0, buf1, buf2, buf3] = &mut bufs;
      Shake128XofReader::squeeze_quad(reader0, reader1, reader2, reader3, buf0, buf1, buf2, buf3);
      products[0].absorb_block(&bufs[0], acc);
      products[1].absorb_block(&bufs[1], acc);
      products[2].absorb_block(&bufs[2], acc);
      products[3].absorb_block(&bufs[3], acc);
    }
  }

  for lane in 0..4 {
    while !products[lane].is_done() {
      readers[lane].squeeze(&mut bufs[lane]);
      products[lane].absorb_block(&bufs[lane], acc);
    }
    products[lane].finish();
  }
}

#[inline(always)]
#[cfg(not(all(
  target_arch = "aarch64",
  target_endian = "little",
  not(miri),
  not(feature = "portable-only")
)))]
fn sample_ntt_quad_block(bufs: &[[u8; SHAKE128_RATE_BYTES]; 4], out: [&mut Poly; 4], filled: &mut [usize; 4]) {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    let [out0, out1, out2, out3] = out;
    // SAFETY: aarch64 NEON SampleNTT quad extraction because:
    // 1. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 2. `bufs` contains four fixed SHAKE128 rate blocks, and the helper only reads complete 48-byte
    //    chunks within those blocks.
    // 3. Each output write stays in-bounds: the NEON path falls back to the scalar bounded parser
    //    whenever a full block could overflow any output polynomial.
    // 4. Rejection branches and write positions depend only on public matrix-A samples, not secret key,
    //    noise, message, or shared-secret material.
    unsafe {
      sample_ntt_quad_block_neon(bufs, [out0, out1, out2, out3], filled);
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  {
    let [out0, out1, out2, out3] = out;
    sample_ntt_block(&bufs[0], out0, &mut filled[0]);
    sample_ntt_block(&bufs[1], out1, &mut filled[1]);
    sample_ntt_block(&bufs[2], out2, &mut filled[2]);
    sample_ntt_block(&bufs[3], out3, &mut filled[3]);
  }
}

#[cfg(all(
  target_arch = "aarch64",
  not(target_endian = "little"),
  not(miri),
  not(feature = "portable-only")
))]
#[target_feature(enable = "neon")]
fn sample_ntt_quad_block_neon(bufs: &[[u8; SHAKE128_RATE_BYTES]; 4], out: [&mut Poly; 4], filled: &mut [usize; 4]) {
  let rate_ptrs = [bufs[0].as_ptr(), bufs[1].as_ptr(), bufs[2].as_ptr(), bufs[3].as_ptr()];
  // SAFETY: Pointer-backed quad parsing over byte buffers because:
  // 1. Each pointer comes from a fixed `[u8; SHAKE128_RATE_BYTES]` rate block.
  // 2. The helper reads only offsets `< SHAKE128_RATE_BYTES`.
  // 3. The helper preserves the same bounded output fallback as the scalar parser.
  unsafe {
    sample_ntt_quad_block_neon_ptrs(rate_ptrs, out, filled);
  }
}

#[cfg(all(
  target_arch = "aarch64",
  target_endian = "little",
  not(miri),
  not(feature = "portable-only")
))]
#[inline(always)]
fn sample_ntt_quad_state_block(states: [&[u64; 25]; 4], out: [&mut Poly; 4], filled: &mut [usize; 4]) {
  let [out0, out1, out2, out3] = out;
  let rate_ptrs = [
    states[0].as_ptr().cast::<u8>(),
    states[1].as_ptr().cast::<u8>(),
    states[2].as_ptr().cast::<u8>(),
    states[3].as_ptr().cast::<u8>(),
  ];
  // SAFETY: Pointer-backed quad parsing over Keccak state memory because:
  // 1. This path is little-endian aarch64 only, so the in-memory `u64` state layout is the XOF byte
  //    order.
  // 2. Each Keccak state has 25 u64 lanes (200 bytes), which covers the first 168-byte SHAKE128 rate
  //    block.
  // 3. The helper reads only offsets `< SHAKE128_RATE_BYTES` and keeps bounded output fallback
  //    semantics.
  unsafe {
    sample_ntt_quad_block_neon_ptrs(rate_ptrs, [out0, out1, out2, out3], filled);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
unsafe fn sample_ntt_quad_block_neon_ptrs(rate_ptrs: [*const u8; 4], out: [&mut Poly; 4], filled: &mut [usize; 4]) {
  const MAX_CANDIDATES: usize = (SHAKE128_RATE_BYTES / 3) * 2;

  let [out0, out1, out2, out3] = out;
  if N.strict_sub(filled[0]) < MAX_CANDIDATES
    || N.strict_sub(filled[1]) < MAX_CANDIDATES
    || N.strict_sub(filled[2]) < MAX_CANDIDATES
    || N.strict_sub(filled[3]) < MAX_CANDIDATES
  {
    // SAFETY: Bounded pointer-backed scalar parsing because:
    // 1. The caller guarantees each pointer names one full readable SHAKE128 rate block.
    // 2. `sample_ntt_block_ptr` reads only offsets `< SHAKE128_RATE_BYTES`.
    // 3. Each destination polynomial and `filled` counter is distinct for the duration of its call.
    // 4. Rejection branches depend only on public matrix-A samples, not secret material.
    unsafe {
      sample_ntt_block_ptr(rate_ptrs[0], out0, &mut filled[0]);
      sample_ntt_block_ptr(rate_ptrs[1], out1, &mut filled[1]);
      sample_ntt_block_ptr(rate_ptrs[2], out2, &mut filled[2]);
      sample_ntt_block_ptr(rate_ptrs[3], out3, &mut filled[3]);
    }
    return;
  }

  #[cfg(any(target_os = "macos", target_os = "linux"))]
  {
    let n0 = filled[0];
    let n1 = filled[1];
    let n2 = filled[2];
    let n3 = filled[3];
    // SAFETY: aarch64 compact assembly SampleNTT block parsing because:
    // 1. Each pointer names one full 168-byte SHAKE128 rate block.
    // 2. The preflight above reserves capacity for all 112 candidates each block can produce.
    // 3. Each destination polynomial is a distinct mutable borrow.
    // 4. Rejection branches and write positions depend only on public matrix-A XOF bytes.
    let (count0, count1, count2, count3) = unsafe {
      (
        aarch64::sample_ntt_rej_uniform_block_asm(out0.as_mut_ptr().add(n0), rate_ptrs[0]),
        aarch64::sample_ntt_rej_uniform_block_asm(out1.as_mut_ptr().add(n1), rate_ptrs[1]),
        aarch64::sample_ntt_rej_uniform_block_asm(out2.as_mut_ptr().add(n2), rate_ptrs[2]),
        aarch64::sample_ntt_rej_uniform_block_asm(out3.as_mut_ptr().add(n3), rate_ptrs[3]),
      )
    };
    *filled = [
      n0.strict_add(count0),
      n1.strict_add(count1),
      n2.strict_add(count2),
      n3.strict_add(count3),
    ];
  }

  #[cfg(not(any(target_os = "macos", target_os = "linux")))]
  {
    const NEON_TRIPLES: usize = 16;
    const NEON_BYTES: usize = NEON_TRIPLES * 3;

    let mut n0 = filled[0];
    let mut n1 = filled[1];
    let mut n2 = filled[2];
    let mut n3 = filled[3];
    let out0_ptr = out0.as_mut_ptr();
    let out1_ptr = out1.as_mut_ptr();
    let out2_ptr = out2.as_mut_ptr();
    let out3_ptr = out3.as_mut_ptr();
    let mut offset = 0usize;

    while offset.strict_add(NEON_BYTES) <= SHAKE128_RATE_BYTES {
      // SAFETY: Four fixed-block NEON candidate extractions plus unchecked public-sample stores because:
      // 1. `offset + NEON_BYTES <= SHAKE128_RATE_BYTES`, so each pointer names one complete 48-byte chunk
      //    inside its lane's full SHAKE128 rate block.
      // 2. The preflight above ensures each output polynomial has room for all candidates from the whole
      //    rate block, so each accepted candidate write stays inside its fixed 256-coefficient output.
      // 3. Rejection branches and write counts depend only on public matrix-A samples.
      let candidates0 = sample_ntt_extract_16_candidate_vectors_neon!(rate_ptrs[0].add(offset));
      let candidates1 = sample_ntt_extract_16_candidate_vectors_neon!(rate_ptrs[1].add(offset));
      let candidates2 = sample_ntt_extract_16_candidate_vectors_neon!(rate_ptrs[2].add(offset));
      let candidates3 = sample_ntt_extract_16_candidate_vectors_neon!(rate_ptrs[3].add(offset));
      sample_ntt_store_candidate_vectors_neon!(out0_ptr, n0, candidates0);
      sample_ntt_store_candidate_vectors_neon!(out1_ptr, n1, candidates1);
      sample_ntt_store_candidate_vectors_neon!(out2_ptr, n2, candidates2);
      sample_ntt_store_candidate_vectors_neon!(out3_ptr, n3, candidates3);

      offset = offset.strict_add(NEON_BYTES);
    }

    while offset.strict_add(2) < SHAKE128_RATE_BYTES {
      // SAFETY: `offset + 2 < SHAKE128_RATE_BYTES`, and each caller-provided
      // pointer names one full rate block.
      let (a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2) = unsafe {
        (
          *rate_ptrs[0].add(offset),
          *rate_ptrs[0].add(offset.strict_add(1)),
          *rate_ptrs[0].add(offset.strict_add(2)),
          *rate_ptrs[1].add(offset),
          *rate_ptrs[1].add(offset.strict_add(1)),
          *rate_ptrs[1].add(offset.strict_add(2)),
          *rate_ptrs[2].add(offset),
          *rate_ptrs[2].add(offset.strict_add(1)),
          *rate_ptrs[2].add(offset.strict_add(2)),
          *rate_ptrs[3].add(offset),
          *rate_ptrs[3].add(offset.strict_add(1)),
          *rate_ptrs[3].add(offset.strict_add(2)),
        )
      };

      let x0 = u16::from(a0) | (u16::from(a1 & 0x0f) << 8);
      let x1 = (u16::from(a1) >> 4) | (u16::from(a2) << 4);
      let y0 = u16::from(b0) | (u16::from(b1 & 0x0f) << 8);
      let y1 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);
      let z0 = u16::from(c0) | (u16::from(c1 & 0x0f) << 8);
      let z1 = (u16::from(c1) >> 4) | (u16::from(c2) << 4);
      let w0 = u16::from(d0) | (u16::from(d1 & 0x0f) << 8);
      let w1 = (u16::from(d1) >> 4) | (u16::from(d2) << 4);

      if x0 < Q {
        out0[n0] = x0;
        n0 = n0.strict_add(1);
      }
      if x1 < Q {
        out0[n0] = x1;
        n0 = n0.strict_add(1);
      }
      if y0 < Q {
        out1[n1] = y0;
        n1 = n1.strict_add(1);
      }
      if y1 < Q {
        out1[n1] = y1;
        n1 = n1.strict_add(1);
      }
      if z0 < Q {
        out2[n2] = z0;
        n2 = n2.strict_add(1);
      }
      if z1 < Q {
        out2[n2] = z1;
        n2 = n2.strict_add(1);
      }
      if w0 < Q {
        out3[n3] = w0;
        n3 = n3.strict_add(1);
      }
      if w1 < Q {
        out3[n3] = w1;
        n3 = n3.strict_add(1);
      }

      offset = offset.strict_add(3);
    }

    *filled = [n0, n1, n2, n3];
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
/// # Safety
///
/// `rate_ptr` must point to one readable SHAKE128 rate block containing at least
/// `SHAKE128_RATE_BYTES` bytes.
unsafe fn sample_ntt_block_ptr(rate_ptr: *const u8, out: &mut Poly, filled: &mut usize) {
  let mut n = *filled;
  let mut offset = 0usize;
  while n < N && offset.strict_add(2) < SHAKE128_RATE_BYTES {
    // SAFETY: `offset + 2 < SHAKE128_RATE_BYTES`, and the caller guarantees `rate_ptr` names one
    // full readable SHAKE128 rate block.
    let (b0, b1, b2) = unsafe {
      (
        *rate_ptr.add(offset),
        *rate_ptr.add(offset.strict_add(1)),
        *rate_ptr.add(offset.strict_add(2)),
      )
    };
    let d1 = u16::from(b0) | (u16::from(b1 & 0x0f) << 8);
    let d2 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);

    if d1 < Q {
      out[n] = d1;
      n = n.strict_add(1);
      if n == N {
        break;
      }
    }
    if d2 < Q {
      out[n] = d2;
      n = n.strict_add(1);
    }
    offset = offset.strict_add(3);
  }
  *filled = n;
}

#[inline]
fn sample_ntt_block(buf: &[u8; SHAKE128_RATE_BYTES], out: &mut Poly, filled: &mut usize) {
  let mut n = *filled;
  let mut offset = 0usize;
  while n < N && offset.strict_add(2) < SHAKE128_RATE_BYTES {
    let b0 = buf[offset];
    let b1 = buf[offset.strict_add(1)];
    let b2 = buf[offset.strict_add(2)];
    let d1 = u16::from(b0) | (u16::from(b1 & 0x0f) << 8);
    let d2 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);

    if d1 < Q {
      out[n] = d1;
      n = n.strict_add(1);
      if n == N {
        break;
      }
    }
    if d2 < Q {
      out[n] = d2;
      n = n.strict_add(1);
    }
    offset = offset.strict_add(3);
  }
  *filled = n;
}

fn sample_noise<const RANDOM_BYTES: usize>(seed: &[u8; SEED_BYTES], nonce: u8, out: &mut Poly) {
  let mut buf = [0u8; RANDOM_BYTES];
  prf_eta(seed, nonce, &mut buf);
  match RANDOM_BYTES {
    ETA2_RANDOM_BYTES => sample_poly_cbd_eta2(&buf, out),
    ETA3_RANDOM_BYTES => sample_poly_cbd_eta3(&buf, out),
    _ => unreachable!("unsupported ML-KEM noise width"),
  }
  ct::zeroize(&mut buf);
}

fn sample_noise_pair<const RANDOM_BYTES: usize>(
  seed: &[u8; SEED_BYTES],
  nonce0: u8,
  out0: &mut Poly,
  nonce1: u8,
  out1: &mut Poly,
) {
  let (mut reader0, mut reader1) = Shake256::xof_seeded_32_1_pair(seed, nonce0, nonce1);
  let mut buf0 = [0u8; RANDOM_BYTES];
  let mut buf1 = [0u8; RANDOM_BYTES];
  Shake256XofReader::squeeze_pair(&mut reader0, &mut reader1, &mut buf0, &mut buf1);
  match RANDOM_BYTES {
    ETA2_RANDOM_BYTES => {
      sample_poly_cbd_eta2(&buf0, out0);
      sample_poly_cbd_eta2(&buf1, out1);
    }
    ETA3_RANDOM_BYTES => {
      sample_poly_cbd_eta3(&buf0, out0);
      sample_poly_cbd_eta3(&buf1, out1);
    }
    _ => unreachable!("unsupported ML-KEM noise width"),
  }
  ct::zeroize(&mut buf0);
  ct::zeroize(&mut buf1);
}

#[allow(clippy::too_many_arguments)]
fn sample_noise_quad<const RANDOM_BYTES: usize>(
  seed: &[u8; SEED_BYTES],
  nonce0: u8,
  out0: &mut Poly,
  nonce1: u8,
  out1: &mut Poly,
  nonce2: u8,
  out2: &mut Poly,
  nonce3: u8,
  out3: &mut Poly,
) {
  let (mut reader0, mut reader1, mut reader2, mut reader3) =
    Shake256::xof_seeded_32_1_quad(seed, nonce0, nonce1, nonce2, nonce3);
  let mut buf0 = [0u8; RANDOM_BYTES];
  let mut buf1 = [0u8; RANDOM_BYTES];
  let mut buf2 = [0u8; RANDOM_BYTES];
  let mut buf3 = [0u8; RANDOM_BYTES];
  Shake256XofReader::squeeze_quad(
    &mut reader0,
    &mut reader1,
    &mut reader2,
    &mut reader3,
    &mut buf0,
    &mut buf1,
    &mut buf2,
    &mut buf3,
  );
  match RANDOM_BYTES {
    ETA2_RANDOM_BYTES => {
      sample_poly_cbd_eta2(&buf0, out0);
      sample_poly_cbd_eta2(&buf1, out1);
      sample_poly_cbd_eta2(&buf2, out2);
      sample_poly_cbd_eta2(&buf3, out3);
    }
    ETA3_RANDOM_BYTES => {
      sample_poly_cbd_eta3(&buf0, out0);
      sample_poly_cbd_eta3(&buf1, out1);
      sample_poly_cbd_eta3(&buf2, out2);
      sample_poly_cbd_eta3(&buf3, out3);
    }
    _ => unreachable!("unsupported ML-KEM noise width"),
  }
  ct::zeroize(&mut buf0);
  ct::zeroize(&mut buf1);
  ct::zeroize(&mut buf2);
  ct::zeroize(&mut buf3);
}

fn sample_poly_cbd_eta2(input: &[u8], out: &mut Poly) {
  debug_assert_eq!(input.len(), ETA2_RANDOM_BYTES);

  for (i, byte) in input.iter().copied().enumerate() {
    let x0 = (byte & 1).strict_add((byte >> 1) & 1);
    let y0 = ((byte >> 2) & 1).strict_add((byte >> 3) & 1);
    let x1 = ((byte >> 4) & 1).strict_add((byte >> 5) & 1);
    let y1 = ((byte >> 6) & 1).strict_add((byte >> 7) & 1);
    out[i.strict_mul(2)] = small_signed_to_mod_q(i16::from(x0) - i16::from(y0));
    out[i.strict_mul(2).strict_add(1)] = small_signed_to_mod_q(i16::from(x1) - i16::from(y1));
  }
}

fn sample_poly_cbd_eta3(input: &[u8], out: &mut Poly) {
  debug_assert_eq!(input.len(), ETA3_RANDOM_BYTES);

  for (i, bytes) in input.chunks_exact(3).enumerate() {
    let bits = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
    let counts = (bits & 0x0024_9249) + ((bits >> 1) & 0x0024_9249) + ((bits >> 2) & 0x0024_9249);
    let start = i.strict_mul(4);

    for j in 0usize..4 {
      let shift = j.strict_mul(6);
      let x = (counts >> shift) & 0x7;
      let y = (counts >> shift.strict_add(3)) & 0x7;
      out[start.strict_add(j)] = small_signed_to_mod_q(x as i16 - y as i16);
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
fn ntt(poly: &mut Poly) {
  // SAFETY: aarch64 NEON NTT dispatch because:
  // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
  // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
  // 3. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 4. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
  //    values.
  unsafe {
    ntt_neon(poly);
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
fn ntt(poly: &mut Poly) {
  if crate::platform::caps().has(crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41) {
    // SAFETY: x86_64 AVX2 NTT dispatch because:
    // 1. Runtime capability detection confirmed AVX2 and SSE4.1 before entering the target-feature
    //    function.
    // 2. `poly` is a fixed 256-coefficient polynomial; the kernel's public loop bounds cover only
    //    in-bounds contiguous 8-coefficient chunks.
    // 3. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
    //    values.
    unsafe {
      return ntt_avx2(poly);
    }
  }

  ntt_scalar(poly);
}

#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
fn ntt(poly: &mut Poly) {
  if use_s390x_vector_arithmetic() {
    // SAFETY: s390x z/Vector NTT dispatch because:
    // 1. Runtime capability detection confirmed the z/Vector facility before entering the
    //    target-feature function.
    // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
    // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
    // 4. Secret-fed coefficient products use fixed-work shift/add multiplication rather than native
    //    scalar or vector multiply.
    unsafe {
      return s390x::ntt_vector(poly);
    }
  }

  ntt_scalar(poly);
}

#[cfg(any(
  miri,
  feature = "portable-only",
  not(any(target_arch = "aarch64", target_arch = "s390x", target_arch = "x86_64"))
))]
fn ntt(poly: &mut Poly) {
  ntt_scalar(poly);
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
fn ntt_to_montgomery_product_domain(poly: &mut Poly) {
  // SAFETY: aarch64 NEON NTT-to-product-domain dispatch because:
  // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
  // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
  // 3. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 4. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
  //    values.
  unsafe {
    ntt_to_montgomery_product_domain_neon(poly);
  }
}

#[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
fn ntt_to_montgomery_product_domain(poly: &mut Poly) {
  ntt(poly);
  poly_to_montgomery_product_domain(poly);
}

#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn ntt_scalar(poly: &mut Poly) {
  let mut zeta_index = 1usize;
  let mut len = 128usize;
  while len >= 2 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS_MONT[zeta_index];
      zeta_index = zeta_index.strict_add(1);
      for j in start..start.strict_add(len) {
        let t = mul_mont_const_mod(poly[j.strict_add(len)], zeta);
        let u = poly[j];
        poly[j.strict_add(len)] = sub_mod(u, t);
        poly[j] = add_mod(u, t);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len >>= 1;
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
fn inverse_ntt_scaled(poly: &mut Poly, final_scale_mont: i16) {
  // SAFETY: aarch64 NEON inverse-NTT dispatch because:
  // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
  // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
  // 3. `poly` is a fixed 256-coefficient polynomial; the kernel's public loop bounds cover only
  //    in-bounds contiguous 8-coefficient chunks.
  // 4. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
  //    values.
  unsafe {
    inverse_ntt_neon(poly, final_scale_mont);
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
fn inverse_ntt_scaled(poly: &mut Poly, final_scale_mont: i16) {
  if crate::platform::caps().has(crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41) {
    // SAFETY: x86_64 AVX2 inverse-NTT dispatch because:
    // 1. Runtime capability detection confirmed AVX2 and SSE4.1 before entering the target-feature
    //    function.
    // 2. `poly` is a fixed 256-coefficient polynomial; the kernel's public loop bounds cover only
    //    in-bounds contiguous 8-coefficient chunks.
    // 3. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
    //    values.
    unsafe {
      return inverse_ntt_avx2(poly, final_scale_mont);
    }
  }

  inverse_ntt_scalar_with_scale(poly, final_scale_mont);
}

#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
fn inverse_ntt_scaled(poly: &mut Poly, final_scale_mont: i16) {
  if use_s390x_vector_arithmetic() {
    // SAFETY: s390x z/Vector inverse-NTT dispatch because:
    // 1. Runtime capability detection confirmed the z/Vector facility before entering the
    //    target-feature function.
    // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
    // 3. `final_scale_mont` is one of the public ML-KEM Montgomery scale constants selected by the
    //    caller.
    // 4. The kernel's memory access schedule depends only on public ML-KEM dimensions.
    // 5. Secret-fed coefficient products use fixed-work shift/add multiplication rather than native
    //    scalar or vector multiply.
    unsafe {
      return s390x::inverse_ntt_vector(poly, final_scale_mont);
    }
  }

  inverse_ntt_scalar_with_scale(poly, final_scale_mont);
}

#[cfg(any(
  miri,
  feature = "portable-only",
  not(any(target_arch = "aarch64", target_arch = "s390x", target_arch = "x86_64"))
))]
fn inverse_ntt_scaled(poly: &mut Poly, final_scale_mont: i16) {
  inverse_ntt_scalar_with_scale(poly, final_scale_mont);
}

#[inline]
#[cfg(test)]
fn inverse_ntt(poly: &mut Poly) {
  inverse_ntt_scaled(poly, INV_NTT_SCALE_MONT);
}

#[inline]
fn inverse_ntt_montgomery_product(poly: &mut Poly) {
  inverse_ntt_scaled(poly, INV_NTT_PRODUCT_SCALE_MONT);
}

#[inline]
fn inverse_ntt_montgomery_product_add_assign(poly: &mut Poly, addend: &Poly) {
  inverse_ntt_scaled_add_assign(poly, addend, INV_NTT_PRODUCT_SCALE_MONT);
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
fn inverse_ntt_scaled_add_assign(poly: &mut Poly, addend: &Poly, final_scale_mont: i16) {
  // SAFETY: aarch64 NEON inverse-NTT + add dispatch because:
  // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
  // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
  // 3. `poly` and `addend` are fixed 256-coefficient polynomials; all loads/stores use public fixed
  //    offsets.
  // 4. `addend` is already in canonical modulo-Q representation, so the final add uses the same
  //    constant-time modular addition as `poly_add_assign`.
  // 5. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
  //    values.
  unsafe {
    inverse_ntt_neon_add_assign(poly, addend, final_scale_mont);
  }
}

#[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
fn inverse_ntt_scaled_add_assign(poly: &mut Poly, addend: &Poly, final_scale_mont: i16) {
  inverse_ntt_scaled(poly, final_scale_mont);
  poly_add_assign(poly, addend);
}

#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn inverse_ntt_scalar_with_scale(poly: &mut Poly, final_scale_mont: i16) {
  let mut zeta_index = 127usize;
  let mut len = 2usize;
  while len <= 128 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS_MONT[zeta_index];
      zeta_index = zeta_index.strict_sub(1);
      for j in start..start.strict_add(len) {
        let t = poly[j];
        let u = poly[j.strict_add(len)];
        poly[j] = add_mod(t, u);
        poly[j.strict_add(len)] = mul_mont_const_mod(sub_mod(u, t), zeta);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len <<= 1;
  }

  for coeff in poly {
    *coeff = mul_mont_const_mod(*coeff, final_scale_mont);
  }
}

#[cfg(test)]
fn inverse_ntt_scalar(poly: &mut Poly) {
  inverse_ntt_scalar_with_scale(poly, INV_NTT_SCALE_MONT);
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[inline(always)]
fn multiply_ntts_add_assign_chunk(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= N);

  if crate::platform::caps().has(crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41) {
    // SAFETY: x86_64 AVX2 chunk multiply-accumulate dispatch because:
    // 1. Runtime capability detection confirmed AVX2 and SSE4.1 before entering the target-feature
    //    function.
    // 2. `a` is exactly one 16-coefficient SampleNTT chunk, and `coeff_offset` is checked to keep `acc`
    //    and `b` accesses inside their fixed 256-coefficient arrays.
    // 3. The borrow checker guarantees `acc` is not aliased by read-only `a` or `b`.
    // 4. The memory access schedule depends only on public ML-KEM dimensions, not coefficient values.
    unsafe {
      return multiply_ntts_add_assign_chunk_avx2(acc, a, b, coeff_offset);
    }
  }

  multiply_ntts_add_assign_chunk_scalar(acc, a, b, coeff_offset);
}

#[cfg(any(
  miri,
  feature = "portable-only",
  not(any(target_arch = "aarch64", target_arch = "s390x", target_arch = "x86_64"))
))]
#[inline(always)]
fn multiply_ntts_add_assign_chunk(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  multiply_ntts_add_assign_chunk_scalar(acc, a, b, coeff_offset);
}

#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline(always)]
fn multiply_ntts_add_assign_chunk(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  if use_s390x_vector_arithmetic() {
    // SAFETY: s390x z/Vector chunk multiply-accumulate dispatch because:
    // 1. Runtime capability detection confirmed the z/Vector facility before entering the
    //    target-feature function.
    // 2. `a` is exactly one 16-coefficient SampleNTT chunk, and `coeff_offset` is emitted only at fixed
    //    chunk boundaries inside 256-coefficient polynomials.
    // 3. The borrow checker guarantees `acc` is not aliased by read-only `a` or `b`.
    // 4. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
    //    fixed-work shift/add multiplication rather than native scalar multiply.
    unsafe {
      return s390x::multiply_ntts_add_assign_chunk_vector(acc, a, b, coeff_offset);
    }
  }

  multiply_ntts_add_assign_chunk_scalar(acc, a, b, coeff_offset);
}

#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
#[inline(always)]
fn multiply_ntts_add_assign_chunk_scalar(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= N);
  let gamma_start = coeff_offset / 2;

  for i in 0..(SAMPLE_NTT_ACC_CHUNK_COEFFS / 2) {
    let local = i.strict_mul(2);
    let j = coeff_offset.strict_add(local);
    let (c0, c1) = base_case_multiply(
      a[local],
      a[local.strict_add(1)],
      b[j],
      b[j.strict_add(1)],
      GAMMAS_MONT[gamma_start.strict_add(i)],
    );
    acc[j] = add_mod(acc[j], c0);
    acc[j.strict_add(1)] = add_mod(acc[j.strict_add(1)], c1);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
fn multiply_ntts_add_assign(acc: &mut Poly, a: &Poly, b: &Poly) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: aarch64 Darwin base-multiply assembly dispatch because:
    // 1. This function only compiles on aarch64 macOS with the portable-only escape hatch disabled.
    // 2. Advanced SIMD is baseline for supported aarch64 macOS targets.
    // 3. `acc`, `a`, and `b` are fixed 256-coefficient arrays matching the assembly ABI.
    // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
    // 5. The assembly memory schedule is fixed and independent of secret coefficient values.
    unsafe {
      aarch64::basemul_accumulate_asm(acc, a, b);
    }
  }

  #[cfg(not(target_os = "macos"))]
  {
    // SAFETY: aarch64 NEON multiply-accumulate dispatch because:
    // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
    // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 3. `acc`, `a`, and `b` are fixed 256-coefficient arrays, which is the exact shape required by the
    //    kernel.
    // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; `a` and `b` are read-only.
    // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
    unsafe {
      multiply_ntts_add_assign_neon(acc, a, b);
    }
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
fn multiply_ntts_add_assign(acc: &mut Poly, a: &Poly, b: &Poly) {
  let caps = crate::platform::caps();
  let avx2_required = crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41;
  let avx512_required = crate::platform::caps::x86::AVX512_READY | avx2_required;

  if caps.has(avx512_required) {
    // SAFETY: x86_64 AVX-512 multiply-accumulate dispatch because:
    // 1. Runtime capability detection confirmed AVX2, SSE4.1, AVX512F, AVX512VL, AVX512BW, and AVX512DQ
    //    before entering the target-feature function.
    // 2. `acc`, `a`, and `b` are fixed 256-coefficient arrays, which is the exact shape required by the
    //    kernel.
    // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; `a` and `b` are read-only.
    // 4. The kernel's memory access schedule is fixed and independent of secret coefficient values.
    unsafe {
      return x86_64::multiply_ntts_add_assign_avx512(acc, a, b);
    }
  }

  if caps.has(avx2_required) {
    // SAFETY: x86_64 AVX2 multiply-accumulate dispatch because:
    // 1. Runtime capability detection confirmed AVX2 and SSE4.1 before entering the target-feature
    //    function.
    // 2. `acc`, `a`, and `b` are fixed 256-coefficient arrays, which is the exact shape required by the
    //    kernel.
    // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; `a` and `b` are read-only.
    // 4. The kernel's memory access schedule is fixed and independent of secret coefficient values.
    unsafe {
      return multiply_ntts_add_assign_avx2(acc, a, b);
    }
  }

  multiply_ntts_add_assign_scalar(acc, a, b);
}

#[cfg(any(
  miri,
  feature = "portable-only",
  not(any(target_arch = "aarch64", target_arch = "s390x", target_arch = "x86_64"))
))]
fn multiply_ntts_add_assign(acc: &mut Poly, a: &Poly, b: &Poly) {
  multiply_ntts_add_assign_scalar(acc, a, b);
}

#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
fn multiply_ntts_add_assign(acc: &mut Poly, a: &Poly, b: &Poly) {
  if use_s390x_vector_arithmetic() {
    // SAFETY: s390x z/Vector multiply-accumulate dispatch because:
    // 1. Runtime capability detection confirmed the z/Vector facility before entering the
    //    target-feature function.
    // 2. `acc`, `a`, and `b` are fixed 256-coefficient arrays matching the kernel contract.
    // 3. The borrow checker guarantees `acc` is not aliased by read-only inputs.
    // 4. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
    //    fixed-work shift/add multiplication rather than native scalar multiply.
    unsafe {
      return s390x::multiply_ntts_add_assign_vector(acc, a, b);
    }
  }

  multiply_ntts_add_assign_scalar(acc, a, b);
}

#[inline(always)]
fn multiply_ntts_accumulate_k2_refs(acc: &mut Poly, a: [&Poly; 2], b: [&Poly; 2]) {
  #[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    not(miri),
    not(feature = "portable-only")
  ))]
  {
    // SAFETY: aarch64 NEON K=2 dot-product dispatch because:
    // 1. This function only compiles on aarch64 Linux with the portable-only escape hatch disabled.
    // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 3. `acc` and every input reference is a fixed 256-coefficient polynomial matching the kernel
    //    contract.
    // 4. The borrow checker guarantees `acc` is not aliased by the read-only input polynomials.
    // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
    unsafe {
      multiply_ntts_accumulate_k2_neon(acc, a, b);
    }
  }

  #[cfg(not(all(
    target_arch = "aarch64",
    target_os = "linux",
    not(miri),
    not(feature = "portable-only")
  )))]
  {
    multiply_ntts_add_assign(acc, a[0], b[0]);
    multiply_ntts_add_assign(acc, a[1], b[1]);
  }
}

#[inline(always)]
fn multiply_ntts_accumulate_k3_refs(acc: &mut Poly, a: [&Poly; 3], b: [&Poly; 3]) {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    // SAFETY: aarch64 NEON K=3 dot-product dispatch because:
    // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
    // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 3. `acc` and every input reference is a fixed 256-coefficient polynomial matching the kernel
    //    contract.
    // 4. The borrow checker guarantees `acc` is not aliased by the read-only input polynomials.
    // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
    unsafe {
      multiply_ntts_accumulate_k3_neon(acc, a, b);
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  {
    #[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
    {
      if use_s390x_vector_arithmetic() {
        // SAFETY: s390x z/Vector k=3 dot-product dispatch because:
        // 1. Runtime capability detection confirmed the z/Vector facility before entering the
        //    target-feature function.
        // 2. Each input reference is a fixed 256-coefficient polynomial matching the kernel contract.
        // 3. The borrow checker guarantees `acc` is not aliased by the read-only input polynomials.
        // 4. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
        //    fixed-work shift/add multiplication rather than native scalar multiply.
        unsafe {
          return s390x::multiply_ntts_accumulate_k3_vector(acc, a, b);
        }
      }
    }

    multiply_ntts_add_assign(acc, a[0], b[0]);
    multiply_ntts_add_assign(acc, a[1], b[1]);
    multiply_ntts_add_assign(acc, a[2], b[2]);
  }
}

#[inline(always)]
fn multiply_ntts_accumulate_k4_refs(acc: &mut Poly, a: [&Poly; 4], b: [&Poly; 4]) {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    // SAFETY: aarch64 NEON K=4 dot-product dispatch because:
    // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
    // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 3. `acc` and every input reference is a fixed 256-coefficient polynomial matching the kernel
    //    contract.
    // 4. The borrow checker guarantees `acc` is not aliased by the read-only input polynomials.
    // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
    unsafe {
      multiply_ntts_accumulate_k4_neon(acc, a, b);
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  {
    #[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
    {
      if use_s390x_vector_arithmetic() {
        // SAFETY: s390x z/Vector k=4 dot-product dispatch because:
        // 1. Runtime capability detection confirmed the z/Vector facility before entering the
        //    target-feature function.
        // 2. Each input reference is a fixed 256-coefficient polynomial matching the kernel contract.
        // 3. The borrow checker guarantees `acc` is not aliased by the read-only input polynomials.
        // 4. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
        //    fixed-work shift/add multiplication rather than native scalar multiply.
        unsafe {
          return s390x::multiply_ntts_accumulate_k4_vector(acc, a, b);
        }
      }
    }

    multiply_ntts_add_assign(acc, a[0], b[0]);
    multiply_ntts_add_assign(acc, a[1], b[1]);
    multiply_ntts_add_assign(acc, a[2], b[2]);
    multiply_ntts_add_assign(acc, a[3], b[3]);
  }
}

#[inline]
fn multiply_ntts_accumulate<const K: usize>(acc: &mut Poly, a: &PolyVec<K>, b: &PolyVec<K>) {
  #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
  {
    let caps = crate::platform::caps();
    let avx2_required = crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41;
    let avx512_required = crate::platform::caps::x86::AVX512_READY | avx2_required;

    if caps.has(avx512_required) {
      if K == 3 {
        // SAFETY: x86_64 AVX-512 K=3 dot-product dispatch because:
        // 1. Runtime capability detection confirmed AVX2, SSE4.1, AVX512F, AVX512VL, AVX512BW, and AVX512DQ
        //    before entering the target-feature function.
        // 2. `K == 3` proves the fixed references below are in bounds for both polynomial vectors.
        // 3. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
        //    contract.
        // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
        // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
        unsafe {
          return x86_64::multiply_ntts_accumulate_k3_avx512(acc, [&a[0], &a[1], &a[2]], [&b[0], &b[1], &b[2]]);
        }
      }

      if K == 4 {
        // SAFETY: x86_64 AVX-512 K=4 dot-product dispatch because:
        // 1. Runtime capability detection confirmed AVX2, SSE4.1, AVX512F, AVX512VL, AVX512BW, and AVX512DQ
        //    before entering the target-feature function.
        // 2. `K == 4` proves the fixed references below are in bounds for both polynomial vectors.
        // 3. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
        //    contract.
        // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
        // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
        unsafe {
          return x86_64::multiply_ntts_accumulate_k4_avx512(
            acc,
            [&a[0], &a[1], &a[2], &a[3]],
            [&b[0], &b[1], &b[2], &b[3]],
          );
        }
      }
    }

    if caps.has(avx2_required) {
      if K == 3 {
        // SAFETY: x86_64 AVX2 K=3 dot-product dispatch because:
        // 1. Runtime capability detection confirmed AVX2 and SSE4.1 before entering the target-feature
        //    function.
        // 2. `K == 3` proves the fixed references below are in bounds for both polynomial vectors.
        // 3. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
        //    contract.
        // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
        // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
        unsafe {
          return x86_64::multiply_ntts_accumulate_k3_avx2(acc, [&a[0], &a[1], &a[2]], [&b[0], &b[1], &b[2]]);
        }
      }

      if K == 4 {
        // SAFETY: x86_64 AVX2 K=4 dot-product dispatch because:
        // 1. Runtime capability detection confirmed AVX2 and SSE4.1 before entering the target-feature
        //    function.
        // 2. `K == 4` proves the fixed references below are in bounds for both polynomial vectors.
        // 3. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
        //    contract.
        // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
        // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
        unsafe {
          return x86_64::multiply_ntts_accumulate_k4_avx2(
            acc,
            [&a[0], &a[1], &a[2], &a[3]],
            [&b[0], &b[1], &b[2], &b[3]],
          );
        }
      }
    }
  }

  #[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
  {
    if use_s390x_vector_arithmetic() {
      if K == 3 {
        // SAFETY: s390x z/Vector k=3 dot-product dispatch because:
        // 1. Runtime capability detection confirmed the z/Vector facility before entering the
        //    target-feature function.
        // 2. `K == 3` proves the fixed references below are in bounds for both polynomial vectors.
        // 3. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
        //    contract.
        // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
        // 5. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
        //    fixed-work shift/add multiplication rather than native scalar multiply.
        unsafe {
          return s390x::multiply_ntts_accumulate_k3_vector(acc, [&a[0], &a[1], &a[2]], [&b[0], &b[1], &b[2]]);
        }
      }

      if K == 4 {
        // SAFETY: s390x z/Vector k=4 dot-product dispatch because:
        // 1. Runtime capability detection confirmed the z/Vector facility before entering the
        //    target-feature function.
        // 2. `K == 4` proves the fixed references below are in bounds for both polynomial vectors.
        // 3. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
        //    contract.
        // 4. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
        // 5. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
        //    fixed-work shift/add multiplication rather than native scalar multiply.
        unsafe {
          return s390x::multiply_ntts_accumulate_k4_vector(
            acc,
            [&a[0], &a[1], &a[2], &a[3]],
            [&b[0], &b[1], &b[2], &b[3]],
          );
        }
      }
    }
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    not(miri),
    not(feature = "portable-only")
  ))]
  {
    if K == 2 {
      // SAFETY: aarch64 Linux assembly K=2 dot-product dispatch because:
      // 1. `K == 2` proves `a` and `b` are contiguous `[[u16; 256]; 2]` polynomial vectors.
      // 2. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. Advanced SIMD is baseline for supported aarch64 Linux targets.
      // 5. The assembly memory schedule is fixed and independent of secret coefficient values.
      unsafe {
        return aarch64::basemul_accumulate_k2_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
      }
    }

    if K == 3 {
      // SAFETY: aarch64 Linux assembly K=3 dot-product dispatch because:
      // 1. `K == 3` proves `a` and `b` are contiguous `[[u16; 256]; 3]` polynomial vectors.
      // 2. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. Advanced SIMD is baseline for supported aarch64 Linux targets.
      // 5. The assembly memory schedule is fixed and independent of secret coefficient values.
      unsafe {
        return aarch64::basemul_accumulate_k3_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
      }
    }

    if K == 4 {
      // SAFETY: aarch64 Linux assembly K=4 dot-product dispatch because:
      // 1. `K == 4` proves `a` and `b` are contiguous `[[u16; 256]; 4]` polynomial vectors.
      // 2. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. Advanced SIMD is baseline for supported aarch64 Linux targets.
      // 5. The assembly memory schedule is fixed and independent of secret coefficient values.
      unsafe {
        return aarch64::basemul_accumulate_k4_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
      }
    }
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(miri),
    not(feature = "portable-only")
  ))]
  {
    if K == 2 {
      // SAFETY: aarch64 Darwin assembly K=2 dot-product dispatch because:
      // 1. `K == 2` proves `a` and `b` are contiguous `[[u16; 256]; 2]` polynomial vectors.
      // 2. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. Advanced SIMD is baseline for supported aarch64 macOS targets.
      // 5. The assembly memory schedule is fixed and independent of secret coefficient values.
      unsafe {
        return aarch64::basemul_accumulate_k2_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
      }
    }

    if K == 3 {
      // SAFETY: aarch64 Darwin assembly K=3 dot-product dispatch because:
      // 1. `K == 3` proves `a` and `b` are contiguous `[[u16; 256]; 3]` polynomial vectors.
      // 2. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. Advanced SIMD is baseline for supported aarch64 macOS targets.
      // 5. The assembly memory schedule is fixed and independent of secret coefficient values.
      unsafe {
        return aarch64::basemul_accumulate_k3_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
      }
    }

    if K == 4 {
      // SAFETY: aarch64 Darwin assembly K=4 dot-product dispatch because:
      // 1. `K == 4` proves `a` and `b` are contiguous `[[u16; 256]; 4]` polynomial vectors.
      // 2. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. Advanced SIMD is baseline for supported aarch64 macOS targets.
      // 5. The assembly memory schedule is fixed and independent of secret coefficient values.
      unsafe {
        return aarch64::basemul_accumulate_k4_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
      }
    }
  }

  #[cfg(all(
    target_arch = "aarch64",
    not(any(target_os = "macos", target_os = "linux")),
    not(miri),
    not(feature = "portable-only")
  ))]
  {
    if K == 3 {
      // SAFETY: aarch64 NEON K=3 dot-product dispatch because:
      // 1. `K == 3` proves the fixed references below are in bounds for both polynomial vectors.
      // 2. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
      //    contract.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
      // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
      unsafe {
        return multiply_ntts_accumulate_k3_neon(acc, [&a[0], &a[1], &a[2]], [&b[0], &b[1], &b[2]]);
      }
    }

    if K == 4 {
      // SAFETY: aarch64 NEON K=4 dot-product dispatch because:
      // 1. `K == 4` proves the fixed references below are in bounds for both polynomial vectors.
      // 2. `acc` and every input polynomial are fixed 256-coefficient arrays matching the kernel
      //    contract.
      // 3. The borrow checker guarantees `acc` is not aliased by `a` or `b`; inputs are read-only.
      // 4. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
      // 5. The kernel's memory access schedule is fixed and independent of secret coefficient values.
      unsafe {
        return multiply_ntts_accumulate_k4_neon(acc, [&a[0], &a[1], &a[2], &a[3]], [&b[0], &b[1], &b[2], &b[3]]);
      }
    }
  }

  for i in 0..K {
    multiply_ntts_add_assign(acc, &a[i], &b[i]);
  }
}

#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn multiply_ntts_add_assign_scalar(acc: &mut Poly, a: &Poly, b: &Poly) {
  for (i, &gamma) in GAMMAS_MONT.iter().enumerate() {
    let j = i.strict_mul(2);
    let (c0, c1) = base_case_multiply(a[j], a[j.strict_add(1)], b[j], b[j.strict_add(1)], gamma);
    acc[j] = add_mod(acc[j], c0);
    acc[j.strict_add(1)] = add_mod(acc[j.strict_add(1)], c1);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn ntt_neon(poly: &mut Poly) {
  ntt_neon_butterflies(poly);
  canonicalize_ntt_neon(poly);
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn ntt_to_montgomery_product_domain_neon(poly: &mut Poly) {
  ntt_neon_butterflies(poly);
  canonicalize_ntt_product_domain_neon(poly);
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn ntt_neon_butterflies(poly: &mut Poly) {
  let mut zeta_index = 1usize;
  let mut len = 128usize;
  while len >= 8 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS_MONT[zeta_index];
      zeta_index = zeta_index.strict_add(1);
      let end = start.strict_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: fixed-size NEON NTT butterfly chunk because:
        // 1. `len >= 8`, `j` advances by 8, and `j < start + len`, so `j..j + 8` is in the lower half.
        // 2. `start + (2 * len) <= N`, so `j + len..j + len + 8` is in the upper half.
        // 3. Each load/store touches exactly 8 u16 coefficients inside the fixed 256-coefficient
        //    polynomial.
        // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
        //    availability.
        unsafe {
          let u = vreinterpretq_s16_u16(vld1q_u16(poly.as_ptr().add(j)));
          let t = mul_mont_const_i16x8(
            vreinterpretq_s16_u16(vld1q_u16(poly.as_ptr().add(j.strict_add(len)))),
            zeta,
          );
          vst1q_u16(
            poly.as_mut_ptr().add(j.strict_add(len)),
            vreinterpretq_u16_s16(vsubq_s16(u, t)),
          );
          vst1q_u16(poly.as_mut_ptr().add(j), vreinterpretq_u16_s16(vaddq_s16(u, t)));
        }
        j = j.strict_add(8);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len >>= 1;
  }

  if len == 4 {
    ntt_len4_neon(poly, &mut zeta_index);
    len >>= 1;
  }

  if len == 2 {
    ntt_len2_neon(poly, &mut zeta_index);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn ntt_len2_neon(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta0 = ZETAS_MONT[*zeta_index];
    let zeta1 = ZETAS_MONT[(*zeta_index).strict_add(1)];
    *zeta_index = (*zeta_index).strict_add(2);

    // SAFETY: fixed-size NEON len-2 NTT butterfly pair because:
    // 1. `start` advances by 8 while `start < N == 256`, so `start..start + 8` is in bounds.
    // 2. Each 8-coefficient load contains two public len-2 butterfly groups: `[a0, a1, b0, b1, c0, c1,
    //    d0, d1]`.
    // 3. The zeta vector duplicates the two public twiddle factors as `[z0, z0, z1, z1]`.
    // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let values = vld1q_u16(poly.as_ptr().add(start));
      let pair_lanes = vreinterpretq_u32_u16(values);
      let lower = vget_low_u16(vreinterpretq_u16_u32(vuzp1q_u32(pair_lanes, pair_lanes)));
      let upper = vget_low_u16(vreinterpretq_u16_u32(vuzp2q_u32(pair_lanes, pair_lanes)));
      let twiddles = duplicate_i16_pair_lanes_neon(zeta0, zeta1);
      let t = mul_mont_i16x4(vreinterpret_s16_u16(upper), twiddles);
      let lower = vreinterpret_s16_u16(lower);
      let lower_out = vreinterpret_u16_s16(vadd_s16(lower, t));
      let upper_out = vreinterpret_u16_s16(vsub_s16(lower, t));
      vst1q_u16(
        poly.as_mut_ptr().add(start),
        zip_u16x4_pair_lanes_neon(lower_out, upper_out),
      );
    }

    start = start.strict_add(8);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn ntt_len4_neon(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_add(1);

    // SAFETY: fixed-size NEON len-4 NTT butterfly because:
    // 1. `start` advances by 8 while `start < N == 256`.
    // 2. Each 4-lane load/store touches `start..start + 4` or `start + 4..start + 8`, both in bounds.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let u = vreinterpret_s16_u16(vld1_u16(poly.as_ptr().add(start)));
      let t = mul_mont_const_i16x4(
        vreinterpret_s16_u16(vld1_u16(poly.as_ptr().add(start.strict_add(4)))),
        zeta,
      );
      vst1_u16(
        poly.as_mut_ptr().add(start.strict_add(4)),
        vreinterpret_u16_s16(vsub_s16(u, t)),
      );
      vst1_u16(poly.as_mut_ptr().add(start), vreinterpret_u16_s16(vadd_s16(u, t)));
    }
    start = start.strict_add(8);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn canonicalize_ntt_neon(poly: &mut Poly) {
  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size NEON forward-NTT canonicalization because:
    // 1. `i` advances by 8 while `i < N == 256`, so `i..i + 8` is in bounds.
    // 2. Lazy forward-NTT butterflies keep every signed lane inside `[-8Q, 8Q]`, so the reducer's fixed
    //    add/subtract schedule is sufficient for canonical modulo-Q output.
    // 3. Each load/store touches exactly 8 contiguous coefficients in the fixed polynomial.
    // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let value = vreinterpretq_s16_u16(vld1q_u16(poly.as_ptr().add(i)));
      vst1q_u16(poly.as_mut_ptr().add(i), canonicalize_lazy_i16x8(value));
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn canonicalize_ntt_product_domain_neon(poly: &mut Poly) {
  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size NEON forward-NTT product-domain finalization because:
    // 1. `i` advances by 8 while `i < N == 256`, so `i..i + 8` is in bounds.
    // 2. Lazy forward-NTT butterflies keep every signed lane inside `[-8Q, 8Q]`; Montgomery reduction
    //    of that signed residue is equivalent to canonicalizing and then applying
    //    `poly_to_montgomery_product_domain`.
    // 3. Each load/store touches exactly 8 contiguous coefficients in the fixed polynomial.
    // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let value = vreinterpretq_s16_u16(vld1q_u16(poly.as_ptr().add(i)));
      vst1q_u16(poly.as_mut_ptr().add(i), lazy_i16x8_to_product_domain(value));
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn inverse_ntt_neon(poly: &mut Poly, final_scale_mont: i16) {
  inverse_ntt_neon_butterflies(poly);

  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size NEON final inverse-NTT scale because:
    // 1. `i` advances by 8 while `i < N == 256`.
    // 2. Each load/store touches `i..i + 8`, which is in bounds.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let coeffs = vld1q_u16(poly.as_ptr().add(i));
      vst1q_u16(
        poly.as_mut_ptr().add(i),
        mul_mont_const_mod_u16x8(coeffs, final_scale_mont),
      );
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn inverse_ntt_neon_add_assign(poly: &mut Poly, addend: &Poly, final_scale_mont: i16) {
  inverse_ntt_neon_butterflies(poly);

  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size NEON final inverse-NTT scale plus add because:
    // 1. `i` advances by 8 while `i < N == 256`.
    // 2. Each load/store touches `i..i + 8`, which is in bounds for both polynomials.
    // 3. `addend` contains canonical modulo-Q coefficients, matching the `poly_add_assign` contract.
    // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let coeffs = vld1q_u16(poly.as_ptr().add(i));
      let scaled = mul_mont_const_mod_u16x8(coeffs, final_scale_mont);
      let addend = vld1q_u16(addend.as_ptr().add(i));
      vst1q_u16(poly.as_mut_ptr().add(i), add_mod_u16x8(scaled, addend));
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn inverse_ntt_neon_butterflies(poly: &mut Poly) {
  let mut zeta_index = 127usize;
  let mut len = 2usize;

  if len == 2 {
    inverse_ntt_len2_neon(poly, &mut zeta_index);
    len <<= 1;
  }

  if len == 4 {
    inverse_ntt_len4_neon(poly, &mut zeta_index);
    len <<= 1;
  }

  while len <= 128 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS_MONT[zeta_index];
      zeta_index = zeta_index.strict_sub(1);
      let end = start.strict_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: fixed-size NEON inverse-NTT butterfly chunk because:
        // 1. `len >= 8`, `j` advances by 8, and `j < start + len`, so `j..j + 8` is in the lower half.
        // 2. `start + (2 * len) <= N`, so `j + len..j + len + 8` is in the upper half.
        // 3. Each load/store touches exactly 8 u16 coefficients inside the fixed 256-coefficient
        //    polynomial.
        // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
        //    availability.
        unsafe {
          let t = vld1q_u16(poly.as_ptr().add(j));
          let u = vld1q_u16(poly.as_ptr().add(j.strict_add(len)));
          vst1q_u16(poly.as_mut_ptr().add(j), add_mod_u16x8(t, u));
          vst1q_u16(
            poly.as_mut_ptr().add(j.strict_add(len)),
            mul_mont_const_mod_u16x8(sub_mod_u16x8(u, t), zeta),
          );
        }
        j = j.strict_add(8);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len <<= 1;
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn inverse_ntt_len2_neon(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta0 = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);
    let zeta1 = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);

    // SAFETY: fixed-size NEON len-2 inverse-NTT butterfly pair because:
    // 1. `start` advances by 8 while `start < N == 256`, so `start..start + 8` is in bounds.
    // 2. Each 8-coefficient load contains two public len-2 butterfly groups: `[a0, a1, b0, b1, c0, c1,
    //    d0, d1]`.
    // 3. The zeta vector duplicates the two public twiddle factors as `[z0, z0, z1, z1]`.
    // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let values = vld1q_u16(poly.as_ptr().add(start));
      let pair_lanes = vreinterpretq_u32_u16(values);
      let lower = vget_low_u16(vreinterpretq_u16_u32(vuzp1q_u32(pair_lanes, pair_lanes)));
      let upper = vget_low_u16(vreinterpretq_u16_u32(vuzp2q_u32(pair_lanes, pair_lanes)));
      let twiddles = duplicate_i16_pair_lanes_neon(zeta0, zeta1);
      let lower_out = add_mod_u16x4(lower, upper);
      let upper_out = mul_mont_mod_u16x4(sub_mod_u16x4(upper, lower), twiddles);
      vst1q_u16(
        poly.as_mut_ptr().add(start),
        zip_u16x4_pair_lanes_neon(lower_out, upper_out),
      );
    }

    start = start.strict_add(8);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn inverse_ntt_len4_neon(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);

    // SAFETY: fixed-size NEON len-4 inverse-NTT butterfly because:
    // 1. `start` advances by 8 while `start < N == 256`.
    // 2. Each 4-lane load/store touches `start..start + 4` or `start + 4..start + 8`, both in bounds.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let t = vld1_u16(poly.as_ptr().add(start));
      let u = vld1_u16(poly.as_ptr().add(start.strict_add(4)));
      vst1_u16(poly.as_mut_ptr().add(start), add_mod_u16x4(t, u));
      vst1_u16(
        poly.as_mut_ptr().add(start.strict_add(4)),
        mul_mont_const_mod_u16x4(sub_mod_u16x4(u, t), zeta),
      );
    }
    start = start.strict_add(8);
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn duplicate_i16_pair_lanes_neon(a: i16, b: i16) -> int16x4_t {
  let lanes = vdup_n_s16(a);
  let lanes = vset_lane_s16::<2>(b, lanes);
  vset_lane_s16::<3>(b, lanes)
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn zip_u16x4_pair_lanes_neon(lower: uint16x4_t, upper: uint16x4_t) -> uint16x8_t {
  let lower_pairs: uint32x2_t = vreinterpret_u32_u16(lower);
  let upper_pairs: uint32x2_t = vreinterpret_u32_u16(upper);
  vreinterpretq_u16_u32(vcombine_u32(
    vzip1_u32(lower_pairs, upper_pairs),
    vzip2_u32(lower_pairs, upper_pairs),
  ))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn base_multiply_8_neon(a: &Poly, b: &Poly, gamma: int16x8_t, coeff_offset: usize) -> (uint16x8_t, uint16x8_t) {
  debug_assert_eq!(coeff_offset % 16, 0);
  debug_assert!(coeff_offset.strict_add(16) <= N);

  // SAFETY: fixed-size deinterleaved NEON loads because:
  // 1. `coeff_offset + 16 <= N` is checked above for both fixed 256-coefficient polynomials.
  // 2. Each `vld2q_u16` touches exactly 16 contiguous u16 coefficients from that public chunk.
  // 3. `gamma` was loaded by the caller from the matching public gamma chunk.
  // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
  //    availability.
  unsafe {
    let a_pair = vld2q_u16(a.as_ptr().add(coeff_offset));
    let b_pair = vld2q_u16(b.as_ptr().add(coeff_offset));

    let a0 = vreinterpretq_s16_u16(a_pair.0);
    let a1 = vreinterpretq_s16_u16(a_pair.1);
    let b0 = vreinterpretq_s16_u16(b_pair.0);
    let b1 = vreinterpretq_s16_u16(b_pair.1);

    let a1b1 = mul_i16x8_to_i32x4_neon(a1, b1);
    let a1b1 = montgomery_reduce_i32x8_neon(a1b1.0, a1b1.1);
    let a0b0 = mul_i16x8_to_i32x4_neon(a0, b0);
    let a1b1_gamma = mul_i16x8_to_i32x4_neon(a1b1, gamma);
    let c0 = signed_to_mod_q_s16x8(montgomery_reduce_i32x8_neon(
      vaddq_s32(a0b0.0, a1b1_gamma.0),
      vaddq_s32(a0b0.1, a1b1_gamma.1),
    ));

    let a0b1 = mul_i16x8_to_i32x4_neon(a0, b1);
    let a1b0 = mul_i16x8_to_i32x4_neon(a1, b0);
    let c1 = signed_to_mod_q_s16x8(montgomery_reduce_i32x8_neon(
      vaddq_s32(a0b1.0, a1b0.0),
      vaddq_s32(a0b1.1, a1b0.1),
    ));

    (c0, c1)
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn store_accumulated_8_neon(acc: &mut Poly, coeff_offset: usize, c0: uint16x8_t, c1: uint16x8_t) {
  debug_assert_eq!(coeff_offset % 16, 0);
  debug_assert!(coeff_offset.strict_add(16) <= N);

  // SAFETY: fixed-size deinterleaved NEON accumulator load and store because:
  // 1. `coeff_offset + 16 <= N` is checked above for the fixed 256-coefficient accumulator.
  // 2. Each `vld2q_u16` / `vst2q_u16` touches exactly 16 contiguous u16 coefficients from that public
  //    chunk.
  // 3. `c0` and `c1` are canonical modulo-Q product-domain coefficients.
  // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
  //    availability.
  unsafe {
    let acc_pair = vld2q_u16(acc.as_ptr().add(coeff_offset));
    let out = uint16x8x2_t(add_mod_u16x8(acc_pair.0, c0), add_mod_u16x8(acc_pair.1, c1));
    vst2q_u16(acc.as_mut_ptr().add(coeff_offset), out);
  }
}

#[cfg(all(
  target_arch = "aarch64",
  any(test, not(target_os = "macos")),
  not(miri),
  not(feature = "portable-only")
))]
#[target_feature(enable = "neon")]
fn multiply_ntts_add_assign_neon(acc: &mut Poly, a: &Poly, b: &Poly) {
  for i in (0..GAMMAS_MONT.len()).step_by(8) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size gamma load because:
    // 1. `i` advances by 8 while `i < GAMMAS_MONT.len() == 128`.
    // 2. `i..i + 8` stays inside the public gamma table.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let gamma = vld1q_s16(GAMMAS_MONT.as_ptr().add(i));
      let (c0, c1) = base_multiply_8_neon(a, b, gamma, coeff_offset);
      store_accumulated_8_neon(acc, coeff_offset, c0, c1);
    }
  }
}

#[cfg(all(
  target_arch = "aarch64",
  target_os = "linux",
  not(miri),
  not(feature = "portable-only")
))]
#[target_feature(enable = "neon")]
fn multiply_ntts_accumulate_k2_neon(acc: &mut Poly, a: [&Poly; 2], b: [&Poly; 2]) {
  for i in (0..GAMMAS_MONT.len()).step_by(8) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size gamma load because:
    // 1. `i` advances by 8 while `i < GAMMAS_MONT.len() == 128`.
    // 2. `i..i + 8` stays inside the public gamma table.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let gamma = vld1q_s16(GAMMAS_MONT.as_ptr().add(i));
      let (mut c0, mut c1) = base_multiply_8_neon(a[0], b[0], gamma, coeff_offset);
      let (p0, p1) = base_multiply_8_neon(a[1], b[1], gamma, coeff_offset);
      c0 = add_mod_u16x8(c0, p0);
      c1 = add_mod_u16x8(c1, p1);
      store_accumulated_8_neon(acc, coeff_offset, c0, c1);
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn multiply_ntts_accumulate_k3_neon(acc: &mut Poly, a: [&Poly; 3], b: [&Poly; 3]) {
  for i in (0..GAMMAS_MONT.len()).step_by(8) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size gamma load because:
    // 1. `i` advances by 8 while `i < GAMMAS_MONT.len() == 128`.
    // 2. `i..i + 8` stays inside the public gamma table.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let gamma = vld1q_s16(GAMMAS_MONT.as_ptr().add(i));
      let (mut c0, mut c1) = base_multiply_8_neon(a[0], b[0], gamma, coeff_offset);
      let (p0, p1) = base_multiply_8_neon(a[1], b[1], gamma, coeff_offset);
      c0 = add_mod_u16x8(c0, p0);
      c1 = add_mod_u16x8(c1, p1);
      let (p0, p1) = base_multiply_8_neon(a[2], b[2], gamma, coeff_offset);
      c0 = add_mod_u16x8(c0, p0);
      c1 = add_mod_u16x8(c1, p1);
      store_accumulated_8_neon(acc, coeff_offset, c0, c1);
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn multiply_ntts_accumulate_k4_neon(acc: &mut Poly, a: [&Poly; 4], b: [&Poly; 4]) {
  for i in (0..GAMMAS_MONT.len()).step_by(8) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size gamma load because:
    // 1. `i` advances by 8 while `i < GAMMAS_MONT.len() == 128`.
    // 2. `i..i + 8` stays inside the public gamma table.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    unsafe {
      let gamma = vld1q_s16(GAMMAS_MONT.as_ptr().add(i));
      let (mut c0, mut c1) = base_multiply_8_neon(a[0], b[0], gamma, coeff_offset);
      let (p0, p1) = base_multiply_8_neon(a[1], b[1], gamma, coeff_offset);
      c0 = add_mod_u16x8(c0, p0);
      c1 = add_mod_u16x8(c1, p1);
      let (p0, p1) = base_multiply_8_neon(a[2], b[2], gamma, coeff_offset);
      c0 = add_mod_u16x8(c0, p0);
      c1 = add_mod_u16x8(c1, p1);
      let (p0, p1) = base_multiply_8_neon(a[3], b[3], gamma, coeff_offset);
      c0 = add_mod_u16x8(c0, p0);
      c1 = add_mod_u16x8(c1, p1);
      store_accumulated_8_neon(acc, coeff_offset, c0, c1);
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn mul_i16x8_to_i32x4_neon(a: int16x8_t, b: int16x8_t) -> (int32x4_t, int32x4_t) {
  (
    vmull_s16(vget_low_s16(a), vget_low_s16(b)),
    vmull_s16(vget_high_s16(a), vget_high_s16(b)),
  )
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn mul_mont_const_i16x8(a: int16x8_t, b_mont: i16) -> int16x8_t {
  montgomery_reduce_s16x8(vmulq_n_s16(a, b_mont), vshrq_n_s16::<1>(vqdmulhq_n_s16(a, b_mont)))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn mul_mont_const_i16x4(a: int16x4_t, b_mont: i16) -> int16x4_t {
  montgomery_reduce_i32x4_neon(vmull_n_s16(a, b_mont))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn mul_mont_i16x4(a: int16x4_t, b_mont: int16x4_t) -> int16x4_t {
  montgomery_reduce_i32x4_neon(vmull_s16(a, b_mont))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn montgomery_reduce_i32x8_neon(lo: int32x4_t, hi: int32x4_t) -> int16x8_t {
  vcombine_s16(montgomery_reduce_i32x4_neon(lo), montgomery_reduce_i32x4_neon(hi))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn montgomery_reduce_i32x4_neon(value: int32x4_t) -> int16x4_t {
  let k = vmul_n_s16(vmovn_s32(value), Q_MONT_INV_U16 as i16);
  let c = vshrn_n_s32::<16>(vmull_n_s16(k, Q_I16));
  vsub_s16(vshrn_n_s32::<16>(value), c)
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn mul_mont_const_mod_u16x4(a: uint16x4_t, b_mont: i16) -> uint16x4_t {
  signed_to_mod_q_s16x4(montgomery_reduce_i32x4_neon(vmull_n_s16(
    vreinterpret_s16_u16(a),
    b_mont,
  )))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn mul_mont_mod_u16x4(a: uint16x4_t, b_mont: int16x4_t) -> uint16x4_t {
  signed_to_mod_q_s16x4(montgomery_reduce_i32x4_neon(vmull_s16(vreinterpret_s16_u16(a), b_mont)))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn add_mod_u16x4(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
  let sum = vadd_u16(a, b);
  let q = vdup_n_u16(Q);
  let ge_q = vcge_u16(sum, q);
  vsub_u16(sum, vand_u16(ge_q, q))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn sub_mod_u16x4(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
  let diff = vsub_u16(a, b);
  let q = vdup_n_u16(Q);
  let borrowed = vcgt_u16(b, a);
  vadd_u16(diff, vand_u16(borrowed, q))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn add_mod_u16x8(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
  let sum = vaddq_u16(a, b);
  let q = vdupq_n_u16(Q);
  let ge_q = vcgeq_u16(sum, q);
  vsubq_u16(sum, vandq_u16(ge_q, q))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn sub_mod_u16x8(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
  let diff = vsubq_u16(a, b);
  let q = vdupq_n_u16(Q);
  let borrowed = vcgtq_u16(b, a);
  vaddq_u16(diff, vandq_u16(borrowed, q))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn canonicalize_lazy_i16x8(value: int16x8_t) -> uint16x8_t {
  let negative = vshrq_n_s16::<15>(value);
  let shifted = vaddq_s16(value, vandq_s16(negative, vdupq_n_s16((Q as i16) * 8)));
  let shifted = vreinterpretq_u16_s16(shifted);
  let lo = reduce_lazy_u32x4(vmovl_u16(vget_low_u16(shifted)));
  let hi = reduce_lazy_u32x4(vmovl_u16(vget_high_u16(shifted)));
  vcombine_u16(vmovn_u32(lo), vmovn_u32(hi))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn lazy_i16x8_to_product_domain(value: int16x8_t) -> uint16x8_t {
  signed_to_mod_q_s16x8(montgomery_reduce_s16x8(value, vshrq_n_s16::<15>(value)))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn reduce_lazy_u32x4(value: uint32x4_t) -> uint32x4_t {
  let quotient = vshrq_n_u32::<26>(vaddq_u32(vmulq_n_u32(value, 20_159), vdupq_n_u32(1 << 25)));
  let product = vmulq_n_u32(quotient, Q_U32);
  let reduced = vsubq_u32(value, product);
  let borrowed = vcgtq_u32(product, value);
  vaddq_u32(reduced, vandq_u32(borrowed, vdupq_n_u32(Q_U32)))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn mul_mont_const_mod_u16x8(a: uint16x8_t, b_mont: i16) -> uint16x8_t {
  signed_to_mod_q_s16x8(mul_mont_const_i16x8(vreinterpretq_s16_u16(a), b_mont))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn montgomery_reduce_s16x8(low: int16x8_t, high: int16x8_t) -> int16x8_t {
  let k = vreinterpretq_s16_u16(vmulq_n_u16(vreinterpretq_u16_s16(low), Q_MONT_INV_U16));
  let c = vshrq_n_s16::<1>(vqdmulhq_n_s16(k, Q_I16));
  vsubq_s16(high, c)
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn signed_to_mod_q_s16x8(value: int16x8_t) -> uint16x8_t {
  let negative = vshrq_n_s16::<15>(value);
  vreinterpretq_u16_s16(vaddq_s16(value, vandq_s16(negative, vdupq_n_s16(Q_I16))))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn signed_to_mod_q_s16x4(value: int16x4_t) -> uint16x4_t {
  let negative = vshr_n_s16::<15>(value);
  vreinterpret_u16_s16(vadd_s16(value, vand_s16(negative, vdup_n_s16(Q_I16))))
}

#[cfg(all(
  target_arch = "aarch64",
  any(test, feature = "diag"),
  not(miri),
  not(feature = "portable-only")
))]
#[target_feature(enable = "neon")]
fn to_montgomery_product_domain_neon(poly: &mut Poly) {
  let high = vdupq_n_s16(0);
  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size NEON product-domain conversion because:
    // 1. `i` advances by 8 while `i < N`, so `i..i + 8` is in-bounds for the 256-coefficient
    //    polynomial.
    // 2. Each load and store touches exactly those 8 contiguous `u16` coefficients.
    // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    // 4. The memory access schedule depends only on public ML-KEM dimensions.
    unsafe {
      let coeffs = vld1q_u16(poly.as_ptr().add(i));
      vst1q_u16(
        poly.as_mut_ptr().add(i),
        signed_to_mod_q_s16x8(montgomery_reduce_s16x8(vreinterpretq_s16_u16(coeffs), high)),
      );
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn from_montgomery_product_domain_neon(poly: &mut Poly) {
  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size NEON product-domain exit because:
    // 1. `i` advances by 8 while `i < N`, so `i..i + 8` is in-bounds for the 256-coefficient
    //    polynomial.
    // 2. Each load and store touches exactly those 8 contiguous `u16` coefficients.
    // 3. `MONT_R_SQUARED_MOD_Q` is the public ML-KEM conversion constant.
    // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
    //    availability.
    // 5. The memory access schedule depends only on public ML-KEM dimensions.
    unsafe {
      let coeffs = vld1q_u16(poly.as_ptr().add(i));
      vst1q_u16(
        poly.as_mut_ptr().add(i),
        mul_mont_const_mod_u16x8(coeffs, MONT_R_SQUARED_MOD_Q),
      );
    }
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn ntt_avx2(poly: &mut Poly) {
  let mut zeta_index = 1usize;
  x86_64::ntt_len_ge16_avx2(poly, &mut zeta_index);

  let mut len = 8usize;
  while len >= 8 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS_MONT[zeta_index];
      zeta_index = zeta_index.strict_add(1);
      let end = start.strict_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: fixed-size AVX2 NTT butterfly chunk because:
        // 1. `len >= 8`, `j` advances by 8, and `j < start + len`, so `j..j + 8` is in the lower half.
        // 2. `start + (2 * len) <= N`, so `j + len..j + len + 8` is in the upper half.
        // 3. Each load/store touches exactly 8 u16 coefficients inside the fixed 256-coefficient
        //    polynomial.
        // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
        //    AVX2 and SSE4.1 availability.
        unsafe {
          let u = load_u16x8_avx2(poly.as_ptr().add(j));
          let t = mul_mont_const_mod_u16x8_avx2(load_u16x8_avx2(poly.as_ptr().add(j.strict_add(len))), zeta);
          store_u16x8_avx2(poly.as_mut_ptr().add(j.strict_add(len)), sub_mod_u16x8_avx2(u, t));
          store_u16x8_avx2(poly.as_mut_ptr().add(j), add_mod_u16x8_avx2(u, t));
        }
        j = j.strict_add(8);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len >>= 1;
  }

  if len == 4 {
    ntt_len4_avx2(poly, &mut zeta_index);
    len >>= 1;
  }

  if len == 2 {
    x86_64::ntt_len2_avx2(poly, &mut zeta_index);
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn ntt_len4_avx2(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_add(1);

    // SAFETY: fixed-size AVX2 len-4 NTT butterfly because:
    // 1. `start` advances by 8 while `start < N == 256`.
    // 2. Each 4-lane load/store touches `start..start + 4` or `start + 4..start + 8`, both in bounds.
    // 3. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let u = load_u16x4_avx2(poly.as_ptr().add(start));
      let t = mul_mont_const_mod_u16x8_avx2(load_u16x4_avx2(poly.as_ptr().add(start.strict_add(4))), zeta);
      store_u16x4_avx2(poly.as_mut_ptr().add(start.strict_add(4)), sub_mod_u16x8_avx2(u, t));
      store_u16x4_avx2(poly.as_mut_ptr().add(start), add_mod_u16x8_avx2(u, t));
    }
    start = start.strict_add(8);
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn inverse_ntt_avx2(poly: &mut Poly, final_scale_mont: i16) {
  let mut zeta_index = 127usize;
  let mut len = 2usize;
  if len == 2 {
    x86_64::inverse_ntt_len2_avx2(poly, &mut zeta_index);
    len <<= 1;
  }

  if len == 4 {
    inverse_ntt_len4_avx2(poly, &mut zeta_index);
    len <<= 1;
  }

  while len <= 8 {
    let mut start = 0usize;
    while start < N {
      let zeta = ZETAS_MONT[zeta_index];
      zeta_index = zeta_index.strict_sub(1);
      let end = start.strict_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: fixed-size AVX2 inverse-NTT butterfly chunk because:
        // 1. `len >= 8`, `j` advances by 8, and `j < start + len`, so `j..j + 8` is in the lower half.
        // 2. `start + (2 * len) <= N`, so `j + len..j + len + 8` is in the upper half.
        // 3. Each load/store touches exactly 8 u16 coefficients inside the fixed 256-coefficient
        //    polynomial.
        // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
        //    AVX2 and SSE4.1 availability.
        unsafe {
          let t = load_u16x8_avx2(poly.as_ptr().add(j));
          let u = load_u16x8_avx2(poly.as_ptr().add(j.strict_add(len)));
          store_u16x8_avx2(poly.as_mut_ptr().add(j), add_mod_u16x8_avx2(t, u));
          store_u16x8_avx2(
            poly.as_mut_ptr().add(j.strict_add(len)),
            mul_mont_const_mod_u16x8_avx2(sub_mod_u16x8_avx2(u, t), zeta),
          );
        }
        j = j.strict_add(8);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len <<= 1;
  }

  x86_64::inverse_ntt_len_ge16_avx2(poly, &mut zeta_index);

  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size AVX2 final inverse-NTT scale because:
    // 1. `i` advances by 8 while `i < N == 256`.
    // 2. Each load/store touches `i..i + 8`, which is in bounds.
    // 3. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let coeffs = load_u16x8_avx2(poly.as_ptr().add(i));
      store_u16x8_avx2(
        poly.as_mut_ptr().add(i),
        mul_mont_const_mod_u16x8_avx2(coeffs, final_scale_mont),
      );
    }
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn inverse_ntt_len4_avx2(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);

    // SAFETY: fixed-size AVX2 len-4 inverse-NTT butterfly because:
    // 1. `start` advances by 8 while `start < N == 256`.
    // 2. Each 4-lane load/store touches `start..start + 4` or `start + 4..start + 8`, both in bounds.
    // 3. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let t = load_u16x4_avx2(poly.as_ptr().add(start));
      let u = load_u16x4_avx2(poly.as_ptr().add(start.strict_add(4)));
      store_u16x4_avx2(poly.as_mut_ptr().add(start), add_mod_u16x8_avx2(t, u));
      store_u16x4_avx2(
        poly.as_mut_ptr().add(start.strict_add(4)),
        mul_mont_const_mod_u16x8_avx2(sub_mod_u16x8_avx2(u, t), zeta),
      );
    }
    start = start.strict_add(8);
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn multiply_ntts_add_assign_avx2(acc: &mut Poly, a: &Poly, b: &Poly) {
  let mask = set1_u32x8_avx2(0xffff);
  for i in (0..GAMMAS_MONT.len()).step_by(8) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size AVX2 deinterleaved loads and store because:
    // 1. `i` advances by 8 while `i < GAMMAS_MONT.len() == 128`; `coeff_offset` is at most 240.
    // 2. Each 256-bit load/store touches 16 u16 coefficients from `coeff_offset..coeff_offset + 16`,
    //    which is within each 256-coefficient polynomial.
    // 3. `load_i16x8_as_i32x8_avx2` touches 8 gamma entries from `i..i + 8`, which is within
    //    `GAMMAS_MONT`.
    // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let a_pairs = _mm256_loadu_si256(a.as_ptr().add(coeff_offset).cast::<__m256i>());
      let b_pairs = _mm256_loadu_si256(b.as_ptr().add(coeff_offset).cast::<__m256i>());
      let acc_pairs = _mm256_loadu_si256(acc.as_ptr().add(coeff_offset).cast::<__m256i>());
      let gamma = load_i16x8_as_i32x8_avx2(GAMMAS_MONT.as_ptr().add(i));

      let a0 = _mm256_and_si256(a_pairs, mask);
      let a1 = _mm256_srli_epi32::<16>(a_pairs);
      let b0 = _mm256_and_si256(b_pairs, mask);
      let b1 = _mm256_srli_epi32::<16>(b_pairs);

      let a0b0 = _mm256_mullo_epi32(a0, b0);
      let a1b1 = montgomery_reduce_i32x8_avx2(_mm256_mullo_epi32(a1, b1));
      let a1b1_gamma = _mm256_mullo_epi32(a1b1, gamma);
      let c0 = signed_to_mod_q_i32x8_avx2(montgomery_reduce_i32x8_avx2(_mm256_add_epi32(a0b0, a1b1_gamma)));
      let c1 = signed_to_mod_q_i32x8_avx2(montgomery_reduce_i32x8_avx2(_mm256_add_epi32(
        _mm256_mullo_epi32(a0, b1),
        _mm256_mullo_epi32(a1, b0),
      )));

      let acc0 = _mm256_and_si256(acc_pairs, mask);
      let acc1 = _mm256_srli_epi32::<16>(acc_pairs);
      let out0 = add_mod_u32x8_avx2(acc0, c0);
      let out1 = add_mod_u32x8_avx2(acc1, c1);
      let packed = _mm256_or_si256(out0, _mm256_slli_epi32::<16>(out1));
      _mm256_storeu_si256(acc.as_mut_ptr().add(coeff_offset).cast::<__m256i>(), packed);
    }
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn multiply_ntts_add_assign_chunk_avx2(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= N);
  let gamma_offset = coeff_offset / 2;
  let mask = set1_u32x8_avx2(0xffff);

  // SAFETY: fixed-size AVX2 chunk loads and store because:
  // 1. `a` has exactly 16 coefficients, so the 256-bit unaligned load is fully in bounds.
  // 2. `coeff_offset + 16 <= N` is checked above, so `b` and `acc` 256-bit accesses are in bounds.
  // 3. `gamma_offset + 8 <= GAMMAS_MONT.len()` follows from `coeff_offset + 16 <= N`.
  // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
  //    AVX2 and SSE4.1 availability.
  unsafe {
    let a_pairs = _mm256_loadu_si256(a.as_ptr().cast::<__m256i>());
    let b_pairs = _mm256_loadu_si256(b.as_ptr().add(coeff_offset).cast::<__m256i>());
    let acc_pairs = _mm256_loadu_si256(acc.as_ptr().add(coeff_offset).cast::<__m256i>());
    let gamma = load_i16x8_as_i32x8_avx2(GAMMAS_MONT.as_ptr().add(gamma_offset));

    let a0 = _mm256_and_si256(a_pairs, mask);
    let a1 = _mm256_srli_epi32::<16>(a_pairs);
    let b0 = _mm256_and_si256(b_pairs, mask);
    let b1 = _mm256_srli_epi32::<16>(b_pairs);

    let a0b0 = _mm256_mullo_epi32(a0, b0);
    let a1b1 = montgomery_reduce_i32x8_avx2(_mm256_mullo_epi32(a1, b1));
    let a1b1_gamma = _mm256_mullo_epi32(a1b1, gamma);
    let c0 = signed_to_mod_q_i32x8_avx2(montgomery_reduce_i32x8_avx2(_mm256_add_epi32(a0b0, a1b1_gamma)));
    let c1 = signed_to_mod_q_i32x8_avx2(montgomery_reduce_i32x8_avx2(_mm256_add_epi32(
      _mm256_mullo_epi32(a0, b1),
      _mm256_mullo_epi32(a1, b0),
    )));

    let acc0 = _mm256_and_si256(acc_pairs, mask);
    let acc1 = _mm256_srli_epi32::<16>(acc_pairs);
    let out0 = add_mod_u32x8_avx2(acc0, c0);
    let out1 = add_mod_u32x8_avx2(acc1, c1);
    let packed = _mm256_or_si256(out0, _mm256_slli_epi32::<16>(out1));
    _mm256_storeu_si256(acc.as_mut_ptr().add(coeff_offset).cast::<__m256i>(), packed);
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn set1_u32x8_avx2(value: u32) -> __m256i {
  _mm256_set1_epi32(value as i32)
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn load_u16x8_avx2(ptr: *const u16) -> __m128i {
  // SAFETY: unaligned 8-coefficient input load because:
  // 1. The caller proves `ptr..ptr + 8` is readable.
  // 2. `_mm_loadu_si128` accepts arbitrary alignment.
  // 3. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`.
  unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn load_u16x4_avx2(ptr: *const u16) -> __m128i {
  // SAFETY: unaligned 4-coefficient input load because:
  // 1. The caller proves `ptr..ptr + 4` is readable.
  // 2. `_mm_loadl_epi64` accepts arbitrary alignment.
  // 3. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`.
  unsafe { _mm_loadl_epi64(ptr.cast::<__m128i>()) }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn store_u16x8_avx2(ptr: *mut u16, values: __m128i) {
  // SAFETY: unaligned 8-coefficient output store because:
  // 1. The caller proves `ptr..ptr + 8` is writable.
  // 2. Values are reduced modulo Q and fit in u16 before storing.
  // 3. `_mm_storeu_si128` accepts arbitrary alignment.
  // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`.
  unsafe { _mm_storeu_si128(ptr.cast::<__m128i>(), values) };
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn store_u16x4_avx2(ptr: *mut u16, values: __m128i) {
  // SAFETY: unaligned 4-coefficient output store because:
  // 1. The caller proves `ptr..ptr + 4` is writable.
  // 2. Values in the low four lanes are reduced modulo Q and fit in u16 before storing.
  // 3. `_mm_storel_epi64` accepts arbitrary alignment.
  // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`.
  unsafe { _mm_storel_epi64(ptr.cast::<__m128i>(), values) };
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn mul_mont_const_mod_u16x8_avx2(a: __m128i, b_mont: i16) -> __m128i {
  let b = _mm_set1_epi16(b_mont);
  signed_to_mod_q_s16x8_avx2(montgomery_reduce_s16x8_avx2(
    _mm_mullo_epi16(a, b),
    _mm_mulhi_epi16(a, b),
  ))
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn montgomery_reduce_s16x8_avx2(low: __m128i, high: __m128i) -> __m128i {
  let k = _mm_mullo_epi16(low, _mm_set1_epi16(Q_MONT_INV_U16 as i16));
  let c = _mm_mulhi_epi16(k, _mm_set1_epi16(Q_I16));
  _mm_sub_epi16(high, c)
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn signed_to_mod_q_s16x8_avx2(value: __m128i) -> __m128i {
  let negative = _mm_cmpgt_epi16(_mm_setzero_si128(), value);
  _mm_add_epi16(value, _mm_and_si128(negative, _mm_set1_epi16(Q_I16)))
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn add_mod_u16x8_avx2(a: __m128i, b: __m128i) -> __m128i {
  let sum = _mm_add_epi16(a, b);
  let ge_q = _mm_cmpgt_epi16(sum, _mm_set1_epi16(Q_I16 - 1));
  _mm_sub_epi16(sum, _mm_and_si128(ge_q, _mm_set1_epi16(Q_I16)))
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn sub_mod_u16x8_avx2(a: __m128i, b: __m128i) -> __m128i {
  let diff = _mm_sub_epi16(a, b);
  let borrowed = _mm_cmpgt_epi16(b, a);
  _mm_add_epi16(diff, _mm_and_si128(borrowed, _mm_set1_epi16(Q_I16)))
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn load_i16x8_as_i32x8_avx2(ptr: *const i16) -> __m256i {
  // SAFETY: unaligned 8-coefficient AVX2 input load because:
  // 1. The caller proves `ptr..ptr + 8` is readable.
  // 2. `_mm_loadu_si128` accepts arbitrary alignment.
  // 3. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`.
  let packed = unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) };
  _mm256_cvtepi16_epi32(packed)
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn montgomery_reduce_i32x8_avx2(value: __m256i) -> __m256i {
  let k = _mm256_mullo_epi16(value, _mm256_set1_epi32(i32::from(Q_MONT_INV_U16)));
  let c = _mm256_mulhi_epi16(k, _mm256_set1_epi32(Q_I32));
  let value_high = _mm256_srli_epi32::<16>(value);
  let reduced = _mm256_sub_epi16(value_high, c);
  _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(reduced))
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn signed_to_mod_q_i32x8_avx2(value: __m256i) -> __m256i {
  let negative = _mm256_cmpgt_epi32(_mm256_setzero_si256(), value);
  _mm256_add_epi32(value, _mm256_and_si256(negative, set1_u32x8_avx2(Q_U32)))
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "avx2,sse4.1")]
fn add_mod_u32x8_avx2(a: __m256i, b: __m256i) -> __m256i {
  let sum = _mm256_add_epi32(a, b);
  let q = set1_u32x8_avx2(Q_U32);
  let ge_q = _mm256_cmpgt_epi32(sum, set1_u32x8_avx2(Q_U32 - 1));
  _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, q))
}

#[inline]
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn base_case_multiply(a0: u16, a1: u16, b0: u16, b1: u16, gamma_mont: i16) -> (u16, u16) {
  let a0b0 = mul_i32_secret(i32::from(a0), i32::from(b0));
  let a1b1 = montgomery_reduce_i32(mul_i32_secret(i32::from(a1), i32::from(b1)));
  let c0 = signed_to_mod_q(montgomery_reduce_i32(
    a0b0 + mul_i32_secret(i32::from(a1b1), i32::from(gamma_mont)),
  ));
  let c1 = signed_to_mod_q(montgomery_reduce_i32(
    mul_i32_secret(i32::from(a0), i32::from(b1)) + mul_i32_secret(i32::from(a1), i32::from(b0)),
  ));
  (c0, c1)
}

#[cfg(test)]
fn base_case_multiply_normal_reference(a0: u16, a1: u16, b0: u16, b1: u16, gamma: u16) -> (u16, u16) {
  let a0b0 = mul_mod(a0, b0);
  let a1b1_gamma = mul_mod(mul_mod(a1, b1), gamma);
  let c0 = add_mod(a0b0, a1b1_gamma);
  let c1 = add_mod(mul_mod(a0, b1), mul_mod(a1, b0));
  (c0, c1)
}

#[cfg(any(
  test,
  feature = "diag",
  not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))
))]
fn poly_to_montgomery_product_domain(poly: &mut Poly) {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    // SAFETY: aarch64 NEON product-domain conversion dispatch because:
    // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
    // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 3. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
    // 4. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
    //    values.
    unsafe {
      to_montgomery_product_domain_neon(poly);
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  {
    #[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
    if use_s390x_vector_arithmetic() {
      // SAFETY: runtime dispatch verified the s390x z/Vector facility, and `poly` is a full
      // fixed-size ML-KEM polynomial.
      unsafe {
        s390x::to_montgomery_product_domain_vector(poly);
      }
      return;
    }

    poly_to_montgomery_product_domain_scalar(poly);
  }
}

#[cfg(any(test, not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))))]
fn poly_to_montgomery_product_domain_scalar(poly: &mut Poly) {
  for coeff in poly {
    *coeff = to_montgomery_product_domain(*coeff);
  }
}

fn poly_from_montgomery_product_domain(poly: &mut Poly) {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    // SAFETY: aarch64 NEON product-domain exit dispatch because:
    // 1. This function only compiles on aarch64 with the portable-only escape hatch disabled.
    // 2. NEON/Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 3. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
    // 4. `MONT_R_SQUARED_MOD_Q` is the public ML-KEM conversion constant.
    // 5. The memory access schedule depends only on public ML-KEM dimensions, not on coefficient
    //    values.
    unsafe {
      from_montgomery_product_domain_neon(poly);
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  {
    #[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
    if use_s390x_vector_arithmetic() {
      // SAFETY: runtime dispatch verified the s390x z/Vector facility, and `poly` is a full
      // fixed-size ML-KEM polynomial.
      unsafe {
        s390x::from_montgomery_product_domain_vector(poly);
      }
      return;
    }

    poly_from_montgomery_product_domain_scalar(poly);
  }
}

#[cfg(any(test, not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))))]
fn poly_from_montgomery_product_domain_scalar(poly: &mut Poly) {
  for coeff in poly {
    *coeff = from_montgomery_product_domain(*coeff);
  }
}

#[cfg(any(test, miri, feature = "portable-only", not(target_arch = "aarch64")))]
#[inline]
fn poly_add_assign(lhs: &mut Poly, rhs: &Poly) {
  for i in 0..N {
    lhs[i] = add_mod(lhs[i], rhs[i]);
  }
}

#[cfg(test)]
fn compress_poly<const D: usize>(input: &Poly, out: &mut Poly) {
  for i in (0..N).step_by(4) {
    let values = compress_values_4::<D>([
      input[i],
      input[i.strict_add(1)],
      input[i.strict_add(2)],
      input[i.strict_add(3)],
    ]);
    out[i] = values[0];
    out[i.strict_add(1)] = values[1];
    out[i.strict_add(2)] = values[2];
    out[i.strict_add(3)] = values[3];
  }
}

#[cfg(test)]
fn decompress_poly<const D: usize>(input: &Poly, out: &mut Poly) {
  for i in (0..N).step_by(4) {
    let values = decompress_values_4::<D>([
      input[i],
      input[i.strict_add(1)],
      input[i.strict_add(2)],
      input[i.strict_add(3)],
    ]);
    out[i] = values[0];
    out[i.strict_add(1)] = values[1];
    out[i.strict_add(2)] = values[2];
    out[i.strict_add(3)] = values[3];
  }
}

#[cfg(test)]
fn decompress_poly_add_assign<const D: usize>(input: &Poly, out: &mut Poly) {
  for i in (0..N).step_by(4) {
    let values = decompress_values_4::<D>([
      input[i],
      input[i.strict_add(1)],
      input[i.strict_add(2)],
      input[i.strict_add(3)],
    ]);
    out[i] = add_mod(out[i], values[0]);
    out[i.strict_add(1)] = add_mod(out[i.strict_add(1)], values[1]);
    out[i.strict_add(2)] = add_mod(out[i.strict_add(2)], values[2]);
    out[i.strict_add(3)] = add_mod(out[i.strict_add(3)], values[3]);
  }
}

#[inline]
fn compress_value<const D: usize>(value: u16) -> u16 {
  let numerator = (u32::from(value) << D) + Q_HALF;
  (div_q_compress_u32(numerator) & ((1u32 << D) - 1)) as u16
}

#[inline(always)]
fn compress_values_4<const D: usize>(values: [u16; 4]) -> [u16; 4] {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    match D {
      5 => {
        // SAFETY: aarch64 four-lane compression dispatch because:
        // 1. `values` contains exactly four canonical ML-KEM coefficients.
        // 2. Advanced SIMD is baseline for supported aarch64 rscrypto targets.
        // 3. The helper uses fixed-lane arithmetic and a fixed local store; there are no
        //    coefficient-dependent addresses or branches.
        unsafe {
          return compress_values_4_neon::<5>(values);
        }
      }
      11 => {
        // SAFETY: aarch64 four-lane compression dispatch because:
        // 1. `values` contains exactly four canonical ML-KEM coefficients.
        // 2. Advanced SIMD is baseline for supported aarch64 rscrypto targets.
        // 3. The helper uses fixed-lane arithmetic and a fixed local store; there are no
        //    coefficient-dependent addresses or branches.
        unsafe {
          return compress_values_4_neon::<11>(values);
        }
      }
      _ => {}
    }
  }

  #[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
  {
    if use_s390x_vector_arithmetic() {
      // SAFETY: runtime capability detection confirmed z/Vector before entering the target-feature
      // helper. Inputs are four coefficients and the helper has no secret-dependent control flow.
      unsafe {
        return s390x::compress_values_4::<D>(values);
      }
    }
  }

  [
    compress_value::<D>(values[0]),
    compress_value::<D>(values[1]),
    compress_value::<D>(values[2]),
    compress_value::<D>(values[3]),
  ]
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn compress_values_4_neon<const D: i32>(values: [u16; 4]) -> [u16; 4] {
  debug_assert!(matches!(D, 1 | 4 | 5 | 10 | 11));

  // SAFETY: fixed-size NEON load/store around local arrays because:
  // 1. `values` is a stack-owned `[u16; 4]`, so `vld1_u16` reads exactly four in-bounds lanes.
  // 2. `out` is a stack-owned `[u16; 4]`, so `vst1_u16` writes exactly four in-bounds lanes.
  // 3. The widening, multiply, shift, and narrowing operations operate on fixed public lane counts.
  // 4. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
  //    availability.
  unsafe {
    let lanes = vld1_u16(values.as_ptr());
    let compressed = compress_lanes_4_neon::<D>(lanes);

    let mut out = [0u16; 4];
    vst1_u16(out.as_mut_ptr(), compressed);
    out
  }
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn compress_lanes_4_neon<const D: i32>(lanes: uint16x4_t) -> uint16x4_t {
  debug_assert!(matches!(D, 1 | 4 | 5 | 10 | 11));
  let numerators = vaddq_u32(vshll_n_u16::<D>(lanes), vdupq_n_u32(Q_HALF));
  let recip = vdup_n_u32(Q_COMPRESS_DIV_RECIP as u32);
  let lo = vshrq_n_u64::<Q_COMPRESS_DIV_SHIFT>(vmull_u32(vget_low_u32(numerators), recip));
  let hi = vshrq_n_u64::<Q_COMPRESS_DIV_SHIFT>(vmull_u32(vget_high_u32(numerators), recip));
  let quotients = vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));
  vmovn_u32(vandq_u32(quotients, vdupq_n_u32((1u32 << (D as u32)) - 1)))
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn load_compress_lanes_4_neon<const D: i32>(input: &Poly, offset: usize) -> uint16x4_t {
  debug_assert!(offset.strict_add(4) <= N);

  // SAFETY: fixed-size NEON coefficient load because:
  // 1. `input` is a fixed 256-coefficient polynomial.
  // 2. `offset + 4 <= N` is checked above, so `vld1_u16` reads four in-bounds lanes.
  // 3. The function is gated by `#[target_feature(enable = "neon")]`, and the caller proves NEON
  //    availability.
  // 4. The memory access schedule depends only on public ML-KEM dimensions.
  unsafe { compress_lanes_4_neon::<D>(vld1_u16(input.as_ptr().add(offset))) }
}

#[inline]
fn decompress_value<const D: usize>(value: u16) -> u16 {
  ((mul_u32_secret(Q_U32, u32::from(value)) + (1u32 << (D - 1))) >> D) as u16
}

#[inline(always)]
fn decompress_values_4<const D: usize>(values: [u16; 4]) -> [u16; 4] {
  #[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
  {
    if use_s390x_vector_arithmetic() {
      // SAFETY: runtime capability detection confirmed z/Vector before entering the target-feature
      // helper. Inputs are four decoded coefficients and the helper uses fixed-work multiplication.
      unsafe {
        return s390x::decompress_values_4::<D>(values);
      }
    }
  }

  [
    decompress_value::<D>(values[0]),
    decompress_value::<D>(values[1]),
    decompress_value::<D>(values[2]),
    decompress_value::<D>(values[3]),
  ]
}

fn decompress_message_add_assign(input: &[u8; SEED_BYTES], out: &mut Poly) {
  for (i, byte) in input.iter().copied().enumerate() {
    let start = i.strict_mul(8);
    let lo = decompress_values_4::<1>([
      u16::from(byte & 1),
      u16::from((byte >> 1) & 1),
      u16::from((byte >> 2) & 1),
      u16::from((byte >> 3) & 1),
    ]);
    let hi = decompress_values_4::<1>([
      u16::from((byte >> 4) & 1),
      u16::from((byte >> 5) & 1),
      u16::from((byte >> 6) & 1),
      u16::from(byte >> 7),
    ]);

    out[start] = add_mod(out[start], lo[0]);
    out[start.strict_add(1)] = add_mod(out[start.strict_add(1)], lo[1]);
    out[start.strict_add(2)] = add_mod(out[start.strict_add(2)], lo[2]);
    out[start.strict_add(3)] = add_mod(out[start.strict_add(3)], lo[3]);
    out[start.strict_add(4)] = add_mod(out[start.strict_add(4)], hi[0]);
    out[start.strict_add(5)] = add_mod(out[start.strict_add(5)], hi[1]);
    out[start.strict_add(6)] = add_mod(out[start.strict_add(6)], hi[2]);
    out[start.strict_add(7)] = add_mod(out[start.strict_add(7)], hi[3]);
  }
}

fn compress_encode_poly<const D: usize>(input: &Poly, out: &mut [u8]) {
  debug_assert_eq!(out.len(), 32 * D);

  match D {
    1 => compress_encode_1(input, out),
    4 => compress_encode_4(input, out),
    5 => compress_encode_5(input, out),
    10 => compress_encode_10(input, out),
    11 => compress_encode_11(input, out),
    _ => unreachable!("unsupported ML-KEM fused compress/encode width"),
  }
}

fn compress_encode_compare_poly<const D: usize, const BYTES: usize>(input: &Poly, expected: &[u8]) -> u8 {
  debug_assert_eq!(BYTES, 32 * D);
  debug_assert_eq!(expected.len(), BYTES);

  match D {
    4 => compress_encode_compare_4(input, expected),
    5 => compress_encode_compare_5(input, expected),
    10 => compress_encode_compare_10(input, expected),
    11 => compress_encode_compare_11(input, expected),
    _ => unreachable!("unsupported ML-KEM fused compress/encode compare width"),
  }
}

#[inline]
fn ct_zero_mask_u8(value: u8) -> u8 {
  let nonzero = (value | value.wrapping_neg()) >> 7;
  0u8.wrapping_sub(nonzero ^ 1)
}

#[inline]
fn ct_zero_mask_u64(value: u64) -> u8 {
  let nonzero = ((value | value.wrapping_neg()) >> 63) as u8;
  0u8.wrapping_sub(nonzero ^ 1)
}

fn compress_encode_compare_4(input: &Poly, expected: &[u8]) -> u8 {
  debug_assert_eq!(expected.len(), 128);
  let mut diff = 0u8;
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(2);
    let t = compress_values_4::<4>([
      input[j],
      input[j.strict_add(1)],
      input[j.strict_add(2)],
      input[j.strict_add(3)],
    ]);
    diff |= ((t[0] | (t[1] << 4)) as u8) ^ expected[k];
    diff |= ((t[2] | (t[3] << 4)) as u8) ^ expected[k.strict_add(1)];
  }
  ct_zero_mask_u8(diff)
}

fn compress_encode_compare_5(input: &Poly, expected: &[u8]) -> u8 {
  debug_assert_eq!(expected.len(), 160);
  let mut diff = 0u8;
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(5);
    let lo = compress_values_4::<5>([
      input[j],
      input[j.strict_add(1)],
      input[j.strict_add(2)],
      input[j.strict_add(3)],
    ]);
    let hi = compress_values_4::<5>([
      input[j.strict_add(4)],
      input[j.strict_add(5)],
      input[j.strict_add(6)],
      input[j.strict_add(7)],
    ]);
    let [t0, t1, t2, t3] = lo;
    let [t4, t5, t6, t7] = hi;

    diff |= ((t0 | (t1 << 5)) as u8) ^ expected[k];
    diff |= (((t1 >> 3) | (t2 << 2) | (t3 << 7)) as u8) ^ expected[k.strict_add(1)];
    diff |= (((t3 >> 1) | (t4 << 4)) as u8) ^ expected[k.strict_add(2)];
    diff |= (((t4 >> 4) | (t5 << 1) | (t6 << 6)) as u8) ^ expected[k.strict_add(3)];
    diff |= (((t6 >> 2) | (t7 << 3)) as u8) ^ expected[k.strict_add(4)];
  }
  ct_zero_mask_u8(diff)
}

fn compress_encode_compare_10(input: &Poly, expected: &[u8]) -> u8 {
  debug_assert_eq!(expected.len(), 320);

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  {
    // SAFETY: aarch64 NEON compare dispatch because:
    // 1. `input` is a fixed 256-coefficient polynomial and `expected` length is checked above.
    // 2. Advanced SIMD is baseline for supported aarch64 rscrypto targets.
    // 3. The helper uses fixed public loop bounds and fixed-stride loads.
    // 4. The compare accumulates all byte differences without secret-dependent early exit.
    unsafe {
      return compress_encode_compare_10_neon(input, expected);
    }
  }

  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  {
    let mut diff = 0u8;
    for i in 0usize..64 {
      let j = i.strict_mul(4);
      let k = i.strict_mul(5);
      let [t0, t1, t2, t3] = compress_values_4::<10>([
        input[j],
        input[j.strict_add(1)],
        input[j.strict_add(2)],
        input[j.strict_add(3)],
      ]);

      diff |= (t0 as u8) ^ expected[k];
      diff |= (((t0 >> 8) | (t1 << 2)) as u8) ^ expected[k.strict_add(1)];
      diff |= (((t1 >> 6) | (t2 << 4)) as u8) ^ expected[k.strict_add(2)];
      diff |= (((t2 >> 4) | (t3 << 6)) as u8) ^ expected[k.strict_add(3)];
      diff |= ((t3 >> 2) as u8) ^ expected[k.strict_add(4)];
    }
    ct_zero_mask_u8(diff)
  }
}

fn compress_encode_compare_11(input: &Poly, expected: &[u8]) -> u8 {
  debug_assert_eq!(expected.len(), 352);
  let mut diff = 0u64;
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(11);
    let lo = compress_values_4::<11>([
      input[j],
      input[j.strict_add(1)],
      input[j.strict_add(2)],
      input[j.strict_add(3)],
    ]);
    let hi = compress_values_4::<11>([
      input[j.strict_add(4)],
      input[j.strict_add(5)],
      input[j.strict_add(6)],
      input[j.strict_add(7)],
    ]);
    let [t0, t1, t2, t3] = lo;
    let [t4, t5, t6, t7] = hi;

    let encoded_lo = u64::from(t0 as u8)
      | (u64::from(((t0 >> 8) | (t1 << 3)) as u8) << 8)
      | (u64::from(((t1 >> 5) | (t2 << 6)) as u8) << 16)
      | (u64::from((t2 >> 2) as u8) << 24)
      | (u64::from(((t2 >> 10) | (t3 << 1)) as u8) << 32)
      | (u64::from(((t3 >> 7) | (t4 << 4)) as u8) << 40)
      | (u64::from(((t4 >> 4) | (t5 << 7)) as u8) << 48)
      | (u64::from((t5 >> 1) as u8) << 56);
    let encoded_hi = u32::from(((t5 >> 9) | (t6 << 2)) as u8)
      | (u32::from(((t6 >> 6) | (t7 << 5)) as u8) << 8)
      | (u32::from((t7 >> 3) as u8) << 16);

    let expected_lo = u64::from_le_bytes([
      expected[k],
      expected[k.strict_add(1)],
      expected[k.strict_add(2)],
      expected[k.strict_add(3)],
      expected[k.strict_add(4)],
      expected[k.strict_add(5)],
      expected[k.strict_add(6)],
      expected[k.strict_add(7)],
    ]);
    let expected_hi = u32::from(expected[k.strict_add(8)])
      | (u32::from(expected[k.strict_add(9)]) << 8)
      | (u32::from(expected[k.strict_add(10)]) << 16);

    diff |= encoded_lo ^ expected_lo;
    diff |= u64::from(encoded_hi ^ expected_hi);
  }
  ct_zero_mask_u64(diff)
}

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
#[target_feature(enable = "neon")]
fn compress_encode_compare_10_neon(input: &Poly, expected: &[u8]) -> u8 {
  let mut diff = 0u8;
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(5);
    let t = load_compress_lanes_4_neon::<10>(input, j);
    let t0 = vget_lane_u16::<0>(t);
    let t1 = vget_lane_u16::<1>(t);
    let t2 = vget_lane_u16::<2>(t);
    let t3 = vget_lane_u16::<3>(t);

    diff |= (t0 as u8) ^ expected[k];
    diff |= (((t0 >> 8) | (t1 << 2)) as u8) ^ expected[k.strict_add(1)];
    diff |= (((t1 >> 6) | (t2 << 4)) as u8) ^ expected[k.strict_add(2)];
    diff |= (((t2 >> 4) | (t3 << 6)) as u8) ^ expected[k.strict_add(3)];
    diff |= ((t3 >> 2) as u8) ^ expected[k.strict_add(4)];
  }
  ct_zero_mask_u8(diff)
}

fn decode_decompress_poly<const D: usize>(input: &[u8], out: &mut Poly) {
  debug_assert_eq!(input.len(), 32 * D);

  match D {
    1 => decode_decompress_1(input, out),
    4 => decode_decompress_4(input, out),
    5 => decode_decompress_5(input, out),
    10 => decode_decompress_10(input, out),
    11 => decode_decompress_11(input, out),
    _ => unreachable!("unsupported ML-KEM fused decode/decompress width"),
  }
}

fn byte_encode<const D: usize>(input: &Poly, out: &mut [u8]) {
  debug_assert_eq!(out.len(), 32 * D);

  match D {
    1 => byte_encode_1(input, out),
    4 => byte_encode_4(input, out),
    5 => byte_encode_5(input, out),
    10 => byte_encode_10(input, out),
    11 => byte_encode_11(input, out),
    12 => byte_encode_12(input, out),
    _ => unreachable!("unsupported ML-KEM byte encoding width"),
  }
}

fn byte_decode<const D: usize>(input: &[u8], out: &mut Poly) {
  debug_assert_eq!(input.len(), 32 * D);

  match D {
    1 => byte_decode_1(input, out),
    4 => byte_decode_4(input, out),
    5 => byte_decode_5(input, out),
    10 => byte_decode_10(input, out),
    11 => byte_decode_11(input, out),
    12 => byte_decode_12(input, out),
    _ => unreachable!("unsupported ML-KEM byte decoding width"),
  }
}

fn compress_encode_1(input: &Poly, out: &mut [u8]) {
  for (i, byte) in out.iter_mut().enumerate() {
    let start = i.strict_mul(8);
    let lo = compress_values_4::<1>([
      input[start],
      input[start.strict_add(1)],
      input[start.strict_add(2)],
      input[start.strict_add(3)],
    ]);
    let hi = compress_values_4::<1>([
      input[start.strict_add(4)],
      input[start.strict_add(5)],
      input[start.strict_add(6)],
      input[start.strict_add(7)],
    ]);
    *byte = (lo[0] as u8 & 1)
      | ((lo[1] as u8 & 1) << 1)
      | ((lo[2] as u8 & 1) << 2)
      | ((lo[3] as u8 & 1) << 3)
      | ((hi[0] as u8 & 1) << 4)
      | ((hi[1] as u8 & 1) << 5)
      | ((hi[2] as u8 & 1) << 6)
      | ((hi[3] as u8 & 1) << 7);
  }
}

fn subtract_compress_encode_message(lhs: &Poly, rhs: &Poly, out: &mut [u8; SEED_BYTES]) {
  for (i, byte) in out.iter_mut().enumerate() {
    let start = i.strict_mul(8);
    let lo = compress_values_4::<1>([
      sub_mod(lhs[start], rhs[start]),
      sub_mod(lhs[start.strict_add(1)], rhs[start.strict_add(1)]),
      sub_mod(lhs[start.strict_add(2)], rhs[start.strict_add(2)]),
      sub_mod(lhs[start.strict_add(3)], rhs[start.strict_add(3)]),
    ]);
    let hi = compress_values_4::<1>([
      sub_mod(lhs[start.strict_add(4)], rhs[start.strict_add(4)]),
      sub_mod(lhs[start.strict_add(5)], rhs[start.strict_add(5)]),
      sub_mod(lhs[start.strict_add(6)], rhs[start.strict_add(6)]),
      sub_mod(lhs[start.strict_add(7)], rhs[start.strict_add(7)]),
    ]);
    *byte = (lo[0] as u8 & 1)
      | ((lo[1] as u8 & 1) << 1)
      | ((lo[2] as u8 & 1) << 2)
      | ((lo[3] as u8 & 1) << 3)
      | ((hi[0] as u8 & 1) << 4)
      | ((hi[1] as u8 & 1) << 5)
      | ((hi[2] as u8 & 1) << 6)
      | ((hi[3] as u8 & 1) << 7);
  }
}

fn decode_decompress_1(input: &[u8], out: &mut Poly) {
  for (i, byte) in input.iter().copied().enumerate() {
    let start = i.strict_mul(8);
    let lo = decompress_values_4::<1>([
      u16::from(byte & 1),
      u16::from((byte >> 1) & 1),
      u16::from((byte >> 2) & 1),
      u16::from((byte >> 3) & 1),
    ]);
    let hi = decompress_values_4::<1>([
      u16::from((byte >> 4) & 1),
      u16::from((byte >> 5) & 1),
      u16::from((byte >> 6) & 1),
      u16::from(byte >> 7),
    ]);
    out[start] = lo[0];
    out[start.strict_add(1)] = lo[1];
    out[start.strict_add(2)] = lo[2];
    out[start.strict_add(3)] = lo[3];
    out[start.strict_add(4)] = hi[0];
    out[start.strict_add(5)] = hi[1];
    out[start.strict_add(6)] = hi[2];
    out[start.strict_add(7)] = hi[3];
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

fn compress_encode_4(input: &Poly, out: &mut [u8]) {
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(2);
    let t = compress_values_4::<4>([
      input[j],
      input[j.strict_add(1)],
      input[j.strict_add(2)],
      input[j.strict_add(3)],
    ]);
    out[k] = (t[0] | (t[1] << 4)) as u8;
    out[k.strict_add(1)] = (t[2] | (t[3] << 4)) as u8;
  }
}

fn decode_decompress_4(input: &[u8], out: &mut Poly) {
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(2);
    let b0 = input[k];
    let b1 = input[k.strict_add(1)];
    let t = decompress_values_4::<4>([
      u16::from(b0 & 0x0f),
      u16::from(b0 >> 4),
      u16::from(b1 & 0x0f),
      u16::from(b1 >> 4),
    ]);
    out[j] = t[0];
    out[j.strict_add(1)] = t[1];
    out[j.strict_add(2)] = t[2];
    out[j.strict_add(3)] = t[3];
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

fn compress_encode_5(input: &Poly, out: &mut [u8]) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(5);
    let lo = compress_values_4::<5>([
      input[j],
      input[j.strict_add(1)],
      input[j.strict_add(2)],
      input[j.strict_add(3)],
    ]);
    let hi = compress_values_4::<5>([
      input[j.strict_add(4)],
      input[j.strict_add(5)],
      input[j.strict_add(6)],
      input[j.strict_add(7)],
    ]);
    let [t0, t1, t2, t3] = lo;
    let [t4, t5, t6, t7] = hi;

    out[k] = (t0 | (t1 << 5)) as u8;
    out[k.strict_add(1)] = ((t1 >> 3) | (t2 << 2) | (t3 << 7)) as u8;
    out[k.strict_add(2)] = ((t3 >> 1) | (t4 << 4)) as u8;
    out[k.strict_add(3)] = ((t4 >> 4) | (t5 << 1) | (t6 << 6)) as u8;
    out[k.strict_add(4)] = ((t6 >> 2) | (t7 << 3)) as u8;
  }
}

fn decode_decompress_5(input: &[u8], out: &mut Poly) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(5);
    let b0 = u16::from(input[k]);
    let b1 = u16::from(input[k.strict_add(1)]);
    let b2 = u16::from(input[k.strict_add(2)]);
    let b3 = u16::from(input[k.strict_add(3)]);
    let b4 = u16::from(input[k.strict_add(4)]);

    let lo = decompress_values_4::<5>([
      b0 & 0x001f,
      ((b0 >> 5) | (b1 << 3)) & 0x001f,
      (b1 >> 2) & 0x001f,
      ((b1 >> 7) | (b2 << 1)) & 0x001f,
    ]);
    let hi = decompress_values_4::<5>([
      ((b2 >> 4) | (b3 << 4)) & 0x001f,
      (b3 >> 1) & 0x001f,
      ((b3 >> 6) | (b4 << 2)) & 0x001f,
      b4 >> 3,
    ]);
    out[j] = lo[0];
    out[j.strict_add(1)] = lo[1];
    out[j.strict_add(2)] = lo[2];
    out[j.strict_add(3)] = lo[3];
    out[j.strict_add(4)] = hi[0];
    out[j.strict_add(5)] = hi[1];
    out[j.strict_add(6)] = hi[2];
    out[j.strict_add(7)] = hi[3];
  }
}

fn byte_encode_5(input: &Poly, out: &mut [u8]) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(5);
    let t0 = input[j] & 0x001f;
    let t1 = input[j.strict_add(1)] & 0x001f;
    let t2 = input[j.strict_add(2)] & 0x001f;
    let t3 = input[j.strict_add(3)] & 0x001f;
    let t4 = input[j.strict_add(4)] & 0x001f;
    let t5 = input[j.strict_add(5)] & 0x001f;
    let t6 = input[j.strict_add(6)] & 0x001f;
    let t7 = input[j.strict_add(7)] & 0x001f;

    out[k] = (t0 | (t1 << 5)) as u8;
    out[k.strict_add(1)] = ((t1 >> 3) | (t2 << 2) | (t3 << 7)) as u8;
    out[k.strict_add(2)] = ((t3 >> 1) | (t4 << 4)) as u8;
    out[k.strict_add(3)] = ((t4 >> 4) | (t5 << 1) | (t6 << 6)) as u8;
    out[k.strict_add(4)] = ((t6 >> 2) | (t7 << 3)) as u8;
  }
}

fn byte_decode_5(input: &[u8], out: &mut Poly) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(5);
    let b0 = u16::from(input[k]);
    let b1 = u16::from(input[k.strict_add(1)]);
    let b2 = u16::from(input[k.strict_add(2)]);
    let b3 = u16::from(input[k.strict_add(3)]);
    let b4 = u16::from(input[k.strict_add(4)]);

    out[j] = b0 & 0x001f;
    out[j.strict_add(1)] = ((b0 >> 5) | (b1 << 3)) & 0x001f;
    out[j.strict_add(2)] = (b1 >> 2) & 0x001f;
    out[j.strict_add(3)] = ((b1 >> 7) | (b2 << 1)) & 0x001f;
    out[j.strict_add(4)] = ((b2 >> 4) | (b3 << 4)) & 0x001f;
    out[j.strict_add(5)] = (b3 >> 1) & 0x001f;
    out[j.strict_add(6)] = ((b3 >> 6) | (b4 << 2)) & 0x001f;
    out[j.strict_add(7)] = b4 >> 3;
  }
}

fn compress_encode_10(input: &Poly, out: &mut [u8]) {
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(5);
    let [t0, t1, t2, t3] = compress_values_4::<10>([
      input[j],
      input[j.strict_add(1)],
      input[j.strict_add(2)],
      input[j.strict_add(3)],
    ]);

    out[k] = t0 as u8;
    out[k.strict_add(1)] = ((t0 >> 8) | (t1 << 2)) as u8;
    out[k.strict_add(2)] = ((t1 >> 6) | (t2 << 4)) as u8;
    out[k.strict_add(3)] = ((t2 >> 4) | (t3 << 6)) as u8;
    out[k.strict_add(4)] = (t3 >> 2) as u8;
  }
}

fn decode_decompress_10(input: &[u8], out: &mut Poly) {
  for i in 0usize..64 {
    let j = i.strict_mul(4);
    let k = i.strict_mul(5);
    let b0 = u16::from(input[k]);
    let b1 = u16::from(input[k.strict_add(1)]);
    let b2 = u16::from(input[k.strict_add(2)]);
    let b3 = u16::from(input[k.strict_add(3)]);
    let b4 = u16::from(input[k.strict_add(4)]);

    let t = decompress_values_4::<10>([
      b0 | ((b1 & 0x03) << 8),
      (b1 >> 2) | ((b2 & 0x0f) << 6),
      (b2 >> 4) | ((b3 & 0x3f) << 4),
      (b3 >> 6) | (b4 << 2),
    ]);
    out[j] = t[0];
    out[j.strict_add(1)] = t[1];
    out[j.strict_add(2)] = t[2];
    out[j.strict_add(3)] = t[3];
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

fn compress_encode_11(input: &Poly, out: &mut [u8]) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(11);
    let lo = compress_values_4::<11>([
      input[j],
      input[j.strict_add(1)],
      input[j.strict_add(2)],
      input[j.strict_add(3)],
    ]);
    let hi = compress_values_4::<11>([
      input[j.strict_add(4)],
      input[j.strict_add(5)],
      input[j.strict_add(6)],
      input[j.strict_add(7)],
    ]);
    let [t0, t1, t2, t3] = lo;
    let [t4, t5, t6, t7] = hi;

    out[k] = t0 as u8;
    out[k.strict_add(1)] = ((t0 >> 8) | (t1 << 3)) as u8;
    out[k.strict_add(2)] = ((t1 >> 5) | (t2 << 6)) as u8;
    out[k.strict_add(3)] = (t2 >> 2) as u8;
    out[k.strict_add(4)] = ((t2 >> 10) | (t3 << 1)) as u8;
    out[k.strict_add(5)] = ((t3 >> 7) | (t4 << 4)) as u8;
    out[k.strict_add(6)] = ((t4 >> 4) | (t5 << 7)) as u8;
    out[k.strict_add(7)] = (t5 >> 1) as u8;
    out[k.strict_add(8)] = ((t5 >> 9) | (t6 << 2)) as u8;
    out[k.strict_add(9)] = ((t6 >> 6) | (t7 << 5)) as u8;
    out[k.strict_add(10)] = (t7 >> 3) as u8;
  }
}

fn decode_decompress_11(input: &[u8], out: &mut Poly) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(11);
    let b0 = u16::from(input[k]);
    let b1 = u16::from(input[k.strict_add(1)]);
    let b2 = u16::from(input[k.strict_add(2)]);
    let b3 = u16::from(input[k.strict_add(3)]);
    let b4 = u16::from(input[k.strict_add(4)]);
    let b5 = u16::from(input[k.strict_add(5)]);
    let b6 = u16::from(input[k.strict_add(6)]);
    let b7 = u16::from(input[k.strict_add(7)]);
    let b8 = u16::from(input[k.strict_add(8)]);
    let b9 = u16::from(input[k.strict_add(9)]);
    let b10 = u16::from(input[k.strict_add(10)]);

    let lo = decompress_values_4::<11>([
      b0 | ((b1 & 0x07) << 8),
      (b1 >> 3) | ((b2 & 0x3f) << 5),
      (b2 >> 6) | (b3 << 2) | ((b4 & 0x01) << 10),
      (b4 >> 1) | ((b5 & 0x0f) << 7),
    ]);
    let hi = decompress_values_4::<11>([
      (b5 >> 4) | ((b6 & 0x7f) << 4),
      (b6 >> 7) | (b7 << 1) | ((b8 & 0x03) << 9),
      (b8 >> 2) | ((b9 & 0x1f) << 6),
      (b9 >> 5) | (b10 << 3),
    ]);
    out[j] = lo[0];
    out[j.strict_add(1)] = lo[1];
    out[j.strict_add(2)] = lo[2];
    out[j.strict_add(3)] = lo[3];
    out[j.strict_add(4)] = hi[0];
    out[j.strict_add(5)] = hi[1];
    out[j.strict_add(6)] = hi[2];
    out[j.strict_add(7)] = hi[3];
  }
}

fn byte_encode_11(input: &Poly, out: &mut [u8]) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(11);
    let t0 = input[j] & 0x07ff;
    let t1 = input[j.strict_add(1)] & 0x07ff;
    let t2 = input[j.strict_add(2)] & 0x07ff;
    let t3 = input[j.strict_add(3)] & 0x07ff;
    let t4 = input[j.strict_add(4)] & 0x07ff;
    let t5 = input[j.strict_add(5)] & 0x07ff;
    let t6 = input[j.strict_add(6)] & 0x07ff;
    let t7 = input[j.strict_add(7)] & 0x07ff;

    out[k] = t0 as u8;
    out[k.strict_add(1)] = ((t0 >> 8) | (t1 << 3)) as u8;
    out[k.strict_add(2)] = ((t1 >> 5) | (t2 << 6)) as u8;
    out[k.strict_add(3)] = (t2 >> 2) as u8;
    out[k.strict_add(4)] = ((t2 >> 10) | (t3 << 1)) as u8;
    out[k.strict_add(5)] = ((t3 >> 7) | (t4 << 4)) as u8;
    out[k.strict_add(6)] = ((t4 >> 4) | (t5 << 7)) as u8;
    out[k.strict_add(7)] = (t5 >> 1) as u8;
    out[k.strict_add(8)] = ((t5 >> 9) | (t6 << 2)) as u8;
    out[k.strict_add(9)] = ((t6 >> 6) | (t7 << 5)) as u8;
    out[k.strict_add(10)] = (t7 >> 3) as u8;
  }
}

fn byte_decode_11(input: &[u8], out: &mut Poly) {
  for i in 0usize..32 {
    let j = i.strict_mul(8);
    let k = i.strict_mul(11);
    let b0 = u16::from(input[k]);
    let b1 = u16::from(input[k.strict_add(1)]);
    let b2 = u16::from(input[k.strict_add(2)]);
    let b3 = u16::from(input[k.strict_add(3)]);
    let b4 = u16::from(input[k.strict_add(4)]);
    let b5 = u16::from(input[k.strict_add(5)]);
    let b6 = u16::from(input[k.strict_add(6)]);
    let b7 = u16::from(input[k.strict_add(7)]);
    let b8 = u16::from(input[k.strict_add(8)]);
    let b9 = u16::from(input[k.strict_add(9)]);
    let b10 = u16::from(input[k.strict_add(10)]);

    out[j] = b0 | ((b1 & 0x07) << 8);
    out[j.strict_add(1)] = (b1 >> 3) | ((b2 & 0x3f) << 5);
    out[j.strict_add(2)] = (b2 >> 6) | (b3 << 2) | ((b4 & 0x01) << 10);
    out[j.strict_add(3)] = (b4 >> 1) | ((b5 & 0x0f) << 7);
    out[j.strict_add(4)] = (b5 >> 4) | ((b6 & 0x7f) << 4);
    out[j.strict_add(5)] = (b6 >> 7) | (b7 << 1) | ((b8 & 0x03) << 9);
    out[j.strict_add(6)] = (b8 >> 2) | ((b9 & 0x1f) << 6);
    out[j.strict_add(7)] = (b9 >> 5) | (b10 << 3);
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
  add_q_if_borrowed(reduced) as u16
}

#[inline]
fn sub_mod(a: u16, b: u16) -> u16 {
  let diff = u32::from(a).wrapping_sub(u32::from(b));
  add_q_if_borrowed(diff) as u16
}

#[inline]
fn sub_if_ge_q(value: u16) -> u16 {
  let reduced = u32::from(value).wrapping_sub(Q_U32);
  add_q_if_borrowed(reduced) as u16
}

#[inline]
#[cfg(test)]
fn mul_mod(a: u16, b: u16) -> u16 {
  reduce_u32(u32::from(a) * u32::from(b))
}

#[inline]
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn mul_mont_const_mod(a: u16, b_mont: i16) -> u16 {
  signed_to_mod_q(montgomery_reduce_i32(mul_i32_secret(i32::from(a), i32::from(b_mont))))
}

#[inline]
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn to_montgomery_product_domain(value: u16) -> u16 {
  signed_to_mod_q(montgomery_reduce_i32(i32::from(value)))
}

#[inline]
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn from_montgomery_product_domain(value: u16) -> u16 {
  mul_mont_const_mod(value, MONT_R_SQUARED_MOD_Q)
}

#[inline]
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn montgomery_reduce_i32(value: i32) -> i16 {
  let k = mul_i32_secret(i32::from(value as i16), i32::from(Q_MONT_INV_U16 as i16));
  let c = (mul_i32_secret(i32::from(k as i16), Q_I32) >> 16) as i16;
  ((value >> 16) as i16).wrapping_sub(c)
}

#[inline]
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn signed_to_mod_q(value: i16) -> u16 {
  let value = i32::from(value);
  (value + ((value >> 31) & Q_I32)) as u16
}

#[inline]
#[cfg(test)]
fn reduce_u32(value: u32) -> u16 {
  let quotient = div_q_u32(value);
  value.wrapping_sub(quotient * Q_U32) as u16
}

#[inline]
#[cfg(test)]
fn div_q_u32(value: u32) -> u32 {
  ((u64::from(value) * Q_DIV_RECIP) >> Q_DIV_SHIFT) as u32
}

#[inline]
fn div_q_compress_u32(value: u32) -> u32 {
  #[cfg(target_arch = "s390x")]
  {
    div_q_compress_u32_ct(value)
  }
  #[cfg(not(target_arch = "s390x"))]
  {
    ((u64::from(value) * Q_COMPRESS_DIV_RECIP) >> Q_COMPRESS_DIV_SHIFT) as u32
  }
}

#[inline]
fn mul_u32_secret(a: u32, b: u32) -> u32 {
  #[cfg(target_arch = "s390x")]
  {
    // IBM Z integer multiply latency is operand-dependent; keep secret-fed products on a fixed
    // shift/add path and verify the generated CT artifact.
    mul_u32_16_ct(a, b)
  }
  #[cfg(not(target_arch = "s390x"))]
  {
    a * b
  }
}

#[inline]
#[cfg(any(
  test,
  miri,
  feature = "portable-only",
  target_arch = "x86_64",
  not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
fn mul_i32_secret(a: i32, b: i32) -> i32 {
  #[cfg(target_arch = "s390x")]
  {
    mul_i32_16_ct(a, b)
  }
  #[cfg(not(target_arch = "s390x"))]
  {
    a * b
  }
}

#[cfg_attr(target_arch = "s390x", inline(never))]
#[cfg_attr(not(target_arch = "s390x"), inline)]
#[cfg(any(test, target_arch = "s390x"))]
fn mul_u32_16_ct(a: u32, b: u32) -> u32 {
  debug_assert!(a <= u32::from(u16::MAX));
  debug_assert!(b <= u32::from(u16::MAX));

  let mut acc = 0u32;
  let mut bit = 0u32;
  while bit < 16 {
    let mask = 0u32.wrapping_sub((b >> bit) & 1);
    acc = acc.wrapping_add((a << bit) & mask);
    bit += 1;
  }
  acc
}

#[cfg_attr(target_arch = "s390x", inline(never))]
#[cfg_attr(not(target_arch = "s390x"), inline)]
#[cfg(any(test, target_arch = "s390x"))]
fn mul_i32_16_ct(a: i32, b: i32) -> i32 {
  debug_assert!((i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&a));
  debug_assert!((i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&b));

  let a_sign = (a >> 31) as u32;
  let b_sign = (b >> 31) as u32;
  let abs_a = ((a as u32) ^ a_sign).wrapping_sub(a_sign);
  let abs_b = ((b as u32) ^ b_sign).wrapping_sub(b_sign);
  let magnitude = mul_u32_16_ct(abs_a, abs_b);
  let sign = a_sign ^ b_sign;
  ((magnitude ^ sign).wrapping_sub(sign)) as i32
}

#[cfg_attr(target_arch = "s390x", inline(never))]
#[cfg_attr(not(target_arch = "s390x"), inline)]
#[cfg(any(test, target_arch = "s390x"))]
fn div_q_compress_u32_ct(value: u32) -> u32 {
  debug_assert!(value < (1u32 << 23));

  let mut quotient = 0u32;
  let mut remainder = 0u32;
  let mut bit = 23u32;
  while bit > 0 {
    bit -= 1;
    remainder = (remainder << 1) | ((value >> bit) & 1);
    let reduced = remainder.wrapping_sub(Q_U32);
    let borrow = reduced >> 31;
    let ge = opaque_s390x_bit(borrow ^ 1);
    remainder = add_q_if_borrowed(reduced);
    quotient |= ge << bit;
  }
  quotient
}

#[inline]
fn add_q_if_borrowed(value: u32) -> u32 {
  let borrow = value >> 31;
  value.wrapping_add(0u32.wrapping_sub(opaque_s390x_bit(borrow)) & Q_U32)
}

#[inline]
fn opaque_s390x_bit(value: u32) -> u32 {
  let value = value & 1;
  #[cfg(target_arch = "s390x")]
  {
    // Prevent LLVM from recovering secret-dependent branches from mask arithmetic on IBM Z.
    core::hint::black_box(value)
  }
  #[cfg(not(target_arch = "s390x"))]
  {
    value
  }
}

#[inline]
#[cfg(test)]
fn div_q_compress_u32_recip(value: u32) -> u32 {
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

fn j(z: &[u8], c: &[u8]) -> [u8; SHARED_SECRET_BYTES] {
  let mut xof = Shake256::new();
  xof.update(z);
  xof.update(c);
  let mut reader = xof.finalize_xof();
  let mut out = [0u8; SHARED_SECRET_BYTES];
  reader.squeeze(&mut out);
  out
}

fn prf_eta<const RANDOM_BYTES: usize>(seed: &[u8; SEED_BYTES], nonce: u8, out: &mut [u8; RANDOM_BYTES]) {
  let mut reader = Shake256::xof_seeded_32_1(seed, nonce);
  reader.squeeze(out);
}

fn ct_eq_mask(a: &[u8], b: &[u8]) -> u8 {
  debug_assert_eq!(a.len(), b.len());
  0u8.wrapping_sub(u8::from(ct::constant_time_eq(a, b)))
}

#[inline]
fn zeroize_poly(poly: &mut Poly) {
  zeroize_poly_no_fence(poly);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[inline]
fn zeroize_polyvec<const K: usize>(polyvec: &mut PolyVec<K>) {
  let len = K.strict_mul(N).strict_mul(core::mem::size_of::<u16>());
  // SAFETY: ML-KEM polynomial vector byte view because:
  // 1. `PolyVec<K>` is an array of `K` contiguous `[u16; N]` polynomials.
  // 2. `len` covers exactly the initialized `u16` storage in bytes.
  // 3. `u16` has no padding and a zero byte pattern is the integer value zero.
  let bytes = unsafe { core::slice::from_raw_parts_mut(polyvec.as_mut_ptr().cast::<u8>(), len) };
  ct::zeroize_no_fence(bytes);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[inline]
fn zeroize_poly_no_fence(poly: &mut Poly) {
  let len = N.strict_mul(core::mem::size_of::<u16>());
  // SAFETY: ML-KEM polynomial byte view because:
  // 1. `poly` is a contiguous initialized `[u16; N]`.
  // 2. `len` covers exactly the initialized `u16` storage in bytes.
  // 3. `u16` has no padding and a zero byte pattern is the integer value zero.
  let bytes = unsafe { core::slice::from_raw_parts_mut(poly.as_mut_ptr().cast::<u8>(), len) };
  ct::zeroize_no_fence(bytes);
}

#[cfg(test)]
mod tests {
  use super::*;
  #[cfg(miri)]
  use crate::{Kem, MlKem512, MlKem512Ciphertext};

  #[cfg(miri)]
  #[test]
  fn miri_mlkem512_portable_round_trip_and_rejection() {
    let mut key_random = [0u8; MlKem512::KEY_GENERATION_RANDOM_SIZE];
    for (i, byte) in key_random.iter_mut().enumerate() {
      *byte = (i.strict_mul(29).strict_add(7)) as u8;
    }
    let mut encapsulation_random = [0u8; MlKem512::ENCAPSULATION_RANDOM_SIZE];
    for (i, byte) in encapsulation_random.iter_mut().enumerate() {
      *byte = (i.strict_mul(31).strict_add(11)) as u8;
    }

    let (encapsulation_key, decapsulation_key) = MlKem512::generate_keypair(|out| {
      out.copy_from_slice(&key_random);
      Ok::<(), MlKemError>(())
    })
    .unwrap();
    let (ciphertext, shared_secret) = MlKem512::encapsulate(&encapsulation_key, |out| {
      out.copy_from_slice(&encapsulation_random);
      Ok::<(), MlKemError>(())
    })
    .unwrap();

    let decapsulated = MlKem512::decapsulate(&decapsulation_key, &ciphertext).unwrap();
    assert_eq!(decapsulated, shared_secret);

    let mut modified = ciphertext.to_bytes();
    modified[0] ^= 1;
    let rejected = MlKem512::decapsulate(&decapsulation_key, &MlKem512Ciphertext::from_bytes(modified)).unwrap();
    assert_ne!(rejected, shared_secret);
  }

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

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  #[test]
  fn basemul_accumulate_k2_asm_matches_scalar_reference() {
    assert_basemul_accumulate_k2_asm_matches_scalar_reference([0u16; N], [[0u16; N]; 2], [[0u16; N]; 2], "all zero");
    assert_basemul_accumulate_k2_asm_matches_scalar_reference([Q - 1; N], [[Q - 1; N]; 2], [[Q - 1; N]; 2], "all q-1");

    for seed in 0usize..256 {
      assert_basemul_accumulate_k2_asm_matches_scalar_reference(
        test_poly(seed),
        core::array::from_fn(|i| test_poly(seed.strict_add(0x4500).strict_add(i.strict_mul(0x100)))),
        core::array::from_fn(|i| test_poly(seed.strict_add(0x4a00).strict_add(i.strict_mul(0x100)))),
        "seeded",
      );
    }
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  fn assert_basemul_accumulate_k2_asm_matches_scalar_reference(acc: Poly, a: PolyVec<2>, b: PolyVec<2>, label: &str) {
    let mut scalar = acc;
    for i in 0..2 {
      multiply_ntts_add_assign_scalar(&mut scalar, &a[i], &b[i]);
    }

    let mut asm = acc;
    // SAFETY: direct aarch64 assembly K=2 dot-product test call because:
    // 1. This test only compiles on aarch64 Linux/macOS targets that include the assembly backend.
    // 2. `asm`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
    // 3. The test compares against the scalar/FIPS accumulator before production dispatch.
    // 4. The assembly memory schedule depends only on public ML-KEM dimensions.
    unsafe {
      aarch64::test_basemul_accumulate_k2_asm(&mut asm, &a, &b);
    }

    assert_eq!(asm, scalar, "{label}");
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn ntt_neon_output_is_canonical() {
    for seed in 0usize..1024 {
      let mut poly = test_poly(seed);
      // SAFETY: direct NEON canonical-domain probe because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        ntt_neon(&mut poly);
      }

      for coeff in poly {
        assert!(coeff < Q, "seed {seed} produced non-canonical NEON coefficient {coeff}");
      }
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn ntt_neon_matches_scalar_reference() {
    let zeros = [0u16; N];
    assert_ntt_neon_matches_scalar_reference(zeros, "zero polynomial");

    let all_max = [Q - 1; N];
    assert_ntt_neon_matches_scalar_reference(all_max, "all q-1 polynomial");

    let alternating = core::array::from_fn(|i| if i & 1 == 0 { 0 } else { Q - 1 });
    assert_ntt_neon_matches_scalar_reference(alternating, "alternating 0/q-1 polynomial");

    for seed in 0usize..1024 {
      assert_ntt_neon_matches_scalar_reference(test_poly(seed), "seeded polynomial");
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn ntt_product_domain_neon_matches_two_pass_reference() {
    for seed in 0usize..1024 {
      let mut expected = test_poly(seed);
      ntt_scalar(&mut expected);
      poly_to_montgomery_product_domain_scalar(&mut expected);

      let mut actual = test_poly(seed);
      // SAFETY: direct NEON fused-finalizer test because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `actual` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        ntt_to_montgomery_product_domain_neon(&mut actual);
      }

      assert_eq!(actual, expected, "seed {seed}");
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn lazy_ntt_neon_finalizers_match_scalar_range() {
    let min = -i32::from(Q) * 8;
    let max = i32::from(Q) * 8;
    let mut start = min;

    while start <= max {
      let lanes = [
        start,
        (start + 1).min(max),
        (start + 2).min(max),
        (start + 3).min(max),
        (start + 4).min(max),
        (start + 5).min(max),
        (start + 6).min(max),
        (start + 7).min(max),
      ];
      let lanes_i16 = lanes.map(|value| value as i16);
      let mut canonical = [0u16; 8];
      let mut product_domain = [0u16; 8];

      // SAFETY: direct NEON lazy-finalizer test because:
      // 1. `lanes_i16`, `canonical`, and `product_domain` are fixed eight-lane stack arrays.
      // 2. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 3. The tested values cover the documented signed lazy NTT range `[-8Q, 8Q]`.
      unsafe {
        let values = vld1q_s16(lanes_i16.as_ptr());
        vst1q_u16(canonical.as_mut_ptr(), canonicalize_lazy_i16x8(values));
        vst1q_u16(product_domain.as_mut_ptr(), lazy_i16x8_to_product_domain(values));
      }

      for lane in 0..8 {
        let mut expected = lanes[lane] % i32::from(Q);
        if expected < 0 {
          expected += i32::from(Q);
        }
        let expected = expected as u16;
        assert_eq!(canonical[lane], expected, "canonical start {start} lane {lane}");
        assert_eq!(
          product_domain[lane],
          to_montgomery_product_domain(expected),
          "product-domain start {start} lane {lane}"
        );
      }

      start += 8;
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn ntt_neon_matches_scalar_for_fips_keygen_noise() {
    let d = [
      249, 206, 215, 37, 228, 105, 120, 238, 82, 21, 50, 99, 184, 68, 205, 166, 255, 59, 174, 206, 253, 125, 87, 13,
      254, 16, 123, 248, 146, 130, 47, 191,
    ];
    let mut seed = [0u8; 33];
    seed[..SEED_BYTES].copy_from_slice(&d);
    seed[SEED_BYTES] = 2;

    let expanded = g(&seed);
    let mut sigma = [0u8; SEED_BYTES];
    sigma.copy_from_slice(&expanded[SEED_BYTES..]);

    for nonce in 0u8..4 {
      let mut poly = [0u16; N];
      sample_noise::<ETA3_RANDOM_BYTES>(&sigma, nonce, &mut poly);
      assert_ntt_neon_matches_scalar_reference(poly, "ML-KEM-512 fixed keygen noise");
    }

    ct::zeroize(&mut sigma);
    ct::zeroize(&mut seed);
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  #[test]
  fn basemul_accumulate_asm_matches_scalar_reference() {
    assert_basemul_accumulate_asm_matches_scalar_reference([0u16; N], [0u16; N], [0u16; N], "all zero");
    assert_basemul_accumulate_asm_matches_scalar_reference([Q - 1; N], [Q - 1; N], [Q - 1; N], "all q-1");

    let alternating = core::array::from_fn(|i| if i & 1 == 0 { 0 } else { Q - 1 });
    assert_basemul_accumulate_asm_matches_scalar_reference(
      alternating,
      test_poly(0x31),
      test_poly(0x57),
      "alternating",
    );

    for seed in 0usize..1024 {
      assert_basemul_accumulate_asm_matches_scalar_reference(
        test_poly(seed),
        test_poly(seed.strict_add(0x1000)),
        test_poly(seed.strict_add(0x2000)),
        "seeded",
      );
    }
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  #[test]
  fn basemul_accumulate_chunk_asm_matches_scalar_reference() {
    for seed in 0usize..128 {
      let acc = test_poly(seed);
      let a = test_poly(seed.strict_add(0x3000));
      let b = test_poly(seed.strict_add(0x4000));

      for coeff_offset in (0..N).step_by(SAMPLE_NTT_ACC_CHUNK_COEFFS) {
        let mut chunk = [0u16; SAMPLE_NTT_ACC_CHUNK_COEFFS];
        chunk.copy_from_slice(&a[coeff_offset..coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS)]);

        let mut scalar = acc;
        multiply_ntts_add_assign_chunk_scalar(&mut scalar, &chunk, &b, coeff_offset);

        let mut asm = acc;
        // SAFETY: direct aarch64 assembly chunk test call because:
        // 1. This test only compiles on aarch64 Linux/macOS targets that include the assembly backend.
        // 2. `chunk` contains exactly one 16-coefficient SampleNTT chunk.
        // 3. `coeff_offset` is emitted from fixed 16-coefficient public chunk boundaries.
        // 4. The test compares against the scalar/FIPS chunk accumulator before production dispatch.
        unsafe {
          aarch64::test_basemul_accumulate_chunk_asm(&mut asm, &chunk, &b, coeff_offset);
        }

        assert_eq!(asm, scalar, "seed {seed}, coeff_offset {coeff_offset}");
      }
    }
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  #[test]
  fn basemul_accumulate_k3_asm_matches_scalar_reference() {
    assert_basemul_accumulate_k3_asm_matches_scalar_reference([0u16; N], [[0u16; N]; 3], [[0u16; N]; 3], "all zero");
    assert_basemul_accumulate_k3_asm_matches_scalar_reference([Q - 1; N], [[Q - 1; N]; 3], [[Q - 1; N]; 3], "all q-1");

    for seed in 0usize..256 {
      assert_basemul_accumulate_k3_asm_matches_scalar_reference(
        test_poly(seed),
        core::array::from_fn(|i| test_poly(seed.strict_add(0x5000).strict_add(i.strict_mul(0x100)))),
        core::array::from_fn(|i| test_poly(seed.strict_add(0x6000).strict_add(i.strict_mul(0x100)))),
        "seeded",
      );
    }
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  #[test]
  fn basemul_accumulate_k4_asm_matches_scalar_reference() {
    assert_basemul_accumulate_k4_asm_matches_scalar_reference([0u16; N], [[0u16; N]; 4], [[0u16; N]; 4], "all zero");
    assert_basemul_accumulate_k4_asm_matches_scalar_reference([Q - 1; N], [[Q - 1; N]; 4], [[Q - 1; N]; 4], "all q-1");

    for seed in 0usize..256 {
      assert_basemul_accumulate_k4_asm_matches_scalar_reference(
        test_poly(seed),
        core::array::from_fn(|i| test_poly(seed.strict_add(0x7000).strict_add(i.strict_mul(0x100)))),
        core::array::from_fn(|i| test_poly(seed.strict_add(0x8000).strict_add(i.strict_mul(0x100)))),
        "seeded",
      );
    }
  }

  #[test]
  fn sample_ntt_pair_matches_scalar_samplers() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(37).strict_add(11)) as u8;
    }

    let (left, right) = sample_ntt_pair(&rho, 0, 1, 2, 1);

    assert_eq!(left, sample_ntt(&rho, 0, 1));
    assert_eq!(right, sample_ntt(&rho, 2, 1));
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    not(miri),
    not(feature = "portable-only")
  ))]
  #[test]
  fn sample_ntt_rej_uniform_block_asm_matches_scalar_reference() {
    const MAX_CANDIDATES: usize = (SHAKE128_RATE_BYTES / 3) * 2;

    for seed in 0usize..512 {
      let mut buf = [0u8; SHAKE128_RATE_BYTES];
      for (i, byte) in buf.iter_mut().enumerate() {
        *byte = (seed
          .strict_mul(73)
          .strict_add(i.strict_mul(29))
          .strict_add((seed >> 1).strict_mul(17))
          & 0xff) as u8;
      }

      let mut expected = [0u16; MAX_CANDIDATES];
      let mut expected_len = 0usize;
      let mut offset = 0usize;
      while offset.strict_add(2) < SHAKE128_RATE_BYTES {
        let b0 = buf[offset];
        let b1 = buf[offset.strict_add(1)];
        let b2 = buf[offset.strict_add(2)];
        let d0 = u16::from(b0) | (u16::from(b1 & 0x0f) << 8);
        let d1 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);
        if d0 < Q {
          expected[expected_len] = d0;
          expected_len = expected_len.strict_add(1);
        }
        if d1 < Q {
          expected[expected_len] = d1;
          expected_len = expected_len.strict_add(1);
        }
        offset = offset.strict_add(3);
      }

      let mut actual = [0u16; MAX_CANDIDATES];
      // SAFETY: direct Linux aarch64 block parser test because:
      // 1. `buf` is one full 168-byte SHAKE128 rate block.
      // 2. `actual` has capacity for all 112 candidates in that block.
      // 3. The test compares every accepted public candidate against the scalar parser above.
      let actual_len = unsafe { aarch64::sample_ntt_rej_uniform_block_asm(actual.as_mut_ptr(), buf.as_ptr()) };

      assert_eq!(actual_len, expected_len, "seed {seed}");
      assert_eq!(&actual[..actual_len], &expected[..expected_len], "seed {seed}");
    }
  }

  #[test]
  fn sample_ntt_quad_matches_scalar_samplers() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(41).strict_add(13)) as u8;
    }

    let lanes = [(0, 0), (1, 3), (2, 1), (3, 2)];
    let mut actual = [[0u16; N]; 4];
    {
      let [out0, out1, out2, out3] = &mut actual;
      sample_ntt_quad_into(&rho, lanes, [out0, out1, out2, out3]);
    }

    for (lane, &(j, i)) in lanes.iter().enumerate() {
      assert_eq!(actual[lane], sample_ntt(&rho, j, i), "lane {lane}");
    }
  }

  #[test]
  fn seeded_sample_ntt_matches_generic_xof_input() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(31).strict_add(7)) as u8;
    }

    for j in 0u8..4 {
      for i in 0u8..4 {
        let input = sample_ntt_input(&rho, j, i);
        let mut expected = [0u16; N];
        let mut actual = [0u16; N];
        sample_ntt_from_xof_into(Shake128::xof(&input), &mut expected);
        sample_ntt_into(&rho, j, i, &mut actual);
        assert_eq!(actual, expected, "j={j}, i={i}");
      }
    }
  }

  #[test]
  fn seeded_prf_eta_matches_generic_xof_input() {
    let mut seed = [0u8; SEED_BYTES];
    for (i, byte) in seed.iter_mut().enumerate() {
      *byte = (i.strict_mul(43).strict_add(5)) as u8;
    }

    for nonce in 0u8..8 {
      let mut expected_eta2 = [0u8; ETA2_RANDOM_BYTES];
      let mut actual_eta2 = [0u8; ETA2_RANDOM_BYTES];
      let mut generic = Shake256::new();
      generic.update(&seed);
      generic.update(&[nonce]);
      generic.finalize_xof().squeeze(&mut expected_eta2);
      prf_eta(&seed, nonce, &mut actual_eta2);
      assert_eq!(actual_eta2, expected_eta2, "eta2 nonce={nonce}");

      let mut expected_eta3 = [0u8; ETA3_RANDOM_BYTES];
      let mut actual_eta3 = [0u8; ETA3_RANDOM_BYTES];
      let mut generic = Shake256::new();
      generic.update(&seed);
      generic.update(&[nonce]);
      generic.finalize_xof().squeeze(&mut expected_eta3);
      prf_eta(&seed, nonce, &mut actual_eta3);
      assert_eq!(actual_eta3, expected_eta3, "eta3 nonce={nonce}");
    }
  }

  #[test]
  fn batched_sample_noise_pair_matches_scalar_sampling() {
    let mut seed = [0u8; SEED_BYTES];
    for (i, byte) in seed.iter_mut().enumerate() {
      *byte = (i.strict_mul(47).strict_add(13)) as u8;
    }

    let mut expected0 = [0u16; N];
    let mut expected1 = [0u16; N];
    let mut actual0 = [0u16; N];
    let mut actual1 = [0u16; N];
    sample_noise::<ETA2_RANDOM_BYTES>(&seed, 1, &mut expected0);
    sample_noise::<ETA2_RANDOM_BYTES>(&seed, 6, &mut expected1);
    sample_noise_pair::<ETA2_RANDOM_BYTES>(&seed, 1, &mut actual0, 6, &mut actual1);
    assert_eq!(actual0, expected0, "eta2 lane 0");
    assert_eq!(actual1, expected1, "eta2 lane 1");

    sample_noise::<ETA3_RANDOM_BYTES>(&seed, 2, &mut expected0);
    sample_noise::<ETA3_RANDOM_BYTES>(&seed, 7, &mut expected1);
    sample_noise_pair::<ETA3_RANDOM_BYTES>(&seed, 2, &mut actual0, 7, &mut actual1);
    assert_eq!(actual0, expected0, "eta3 lane 0");
    assert_eq!(actual1, expected1, "eta3 lane 1");
  }

  #[test]
  fn batched_sample_noise_quad_matches_scalar_sampling() {
    let mut seed = [0u8; SEED_BYTES];
    for (i, byte) in seed.iter_mut().enumerate() {
      *byte = (i.strict_mul(59).strict_add(19)) as u8;
    }

    let mut expected = [[0u16; N]; 4];
    let mut actual = [[0u16; N]; 4];
    for lane in 0u8..4 {
      sample_noise::<ETA2_RANDOM_BYTES>(&seed, lane.strict_add(3), &mut expected[usize::from(lane)]);
    }
    let [actual0, actual1, actual2, actual3] = &mut actual;
    sample_noise_quad::<ETA2_RANDOM_BYTES>(&seed, 3, actual0, 4, actual1, 5, actual2, 6, actual3);
    assert_eq!(actual, expected, "eta2 quad");

    for lane in 0u8..4 {
      sample_noise::<ETA3_RANDOM_BYTES>(&seed, lane.strict_add(8), &mut expected[usize::from(lane)]);
    }
    let [actual0, actual1, actual2, actual3] = &mut actual;
    sample_noise_quad::<ETA3_RANDOM_BYTES>(&seed, 8, actual0, 9, actual1, 10, actual2, 11, actual3);
    assert_eq!(actual, expected, "eta3 quad");
  }

  #[test]
  fn fused_sample_ntt_accumulate_matches_sample_then_multiply() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(29).strict_add(17)) as u8;
    }

    for seed in 0usize..8 {
      let rhs = test_poly(seed.strict_add(100));
      let base = test_poly(seed.strict_add(200));
      let mut sampled = [0u16; N];
      let j = (seed % 4) as u8;
      let i = ((seed.strict_mul(3)) % 4) as u8;

      sample_ntt_into(&rho, j, i, &mut sampled);
      let mut expected = base;
      multiply_ntts_add_assign_scalar(&mut expected, &sampled, &rhs);

      let mut actual = base;
      sample_ntt_mul_accumulate(&rho, j, i, &rhs, &mut actual);

      assert_eq!(actual, expected, "seed {seed}");
    }
  }

  #[test]
  fn fused_sample_ntt_pair_accumulate_matches_two_sampled_products() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(41).strict_add(23)) as u8;
    }

    for seed in 0usize..8 {
      let rhs0 = test_poly(seed.strict_add(300));
      let rhs1 = test_poly(seed.strict_add(400));
      let base = test_poly(seed.strict_add(500));
      let j0 = (seed % 4) as u8;
      let i0 = ((seed.strict_mul(5).strict_add(1)) % 4) as u8;
      let j1 = ((seed.strict_add(2)) % 4) as u8;
      let i1 = ((seed.strict_mul(7).strict_add(3)) % 4) as u8;

      let mut sampled0 = [0u16; N];
      let mut sampled1 = [0u16; N];
      sample_ntt_into(&rho, j0, i0, &mut sampled0);
      sample_ntt_into(&rho, j1, i1, &mut sampled1);
      let mut expected = base;
      multiply_ntts_add_assign_scalar(&mut expected, &sampled0, &rhs0);
      multiply_ntts_add_assign_scalar(&mut expected, &sampled1, &rhs1);

      let mut actual = base;
      sample_ntt_pair_mul_accumulate(&rho, (j0, i0, &rhs0), (j1, i1, &rhs1), &mut actual);

      assert_eq!(actual, expected, "seed {seed}");
    }
  }

  #[test]
  fn fused_sample_ntt_quad_accumulate_matches_four_sampled_products() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(53).strict_add(29)) as u8;
    }

    for seed in 0usize..8 {
      let rhs0 = test_poly(seed.strict_add(600));
      let rhs1 = test_poly(seed.strict_add(700));
      let rhs2 = test_poly(seed.strict_add(800));
      let rhs3 = test_poly(seed.strict_add(900));
      let rhs = [&rhs0, &rhs1, &rhs2, &rhs3];
      let coords = [
        ((seed % 4) as u8, ((seed.strict_mul(3).strict_add(1)) % 4) as u8),
        (
          ((seed.strict_add(1)) % 4) as u8,
          ((seed.strict_mul(5).strict_add(2)) % 4) as u8,
        ),
        (
          ((seed.strict_add(2)) % 4) as u8,
          ((seed.strict_mul(7).strict_add(3)) % 4) as u8,
        ),
        (
          ((seed.strict_add(3)) % 4) as u8,
          ((seed.strict_mul(11).strict_add(1)) % 4) as u8,
        ),
      ];
      let base = test_poly(seed.strict_add(1000));

      let mut expected = base;
      for lane in 0..4 {
        let mut sampled = [0u16; N];
        sample_ntt_into(&rho, coords[lane].0, coords[lane].1, &mut sampled);
        multiply_ntts_add_assign_scalar(&mut expected, &sampled, rhs[lane]);
      }

      let mut actual = base;
      sample_ntt_quad_mul_accumulate(
        &rho,
        [
          (coords[0].0, coords[0].1, rhs[0]),
          (coords[1].0, coords[1].1, rhs[1]),
          (coords[2].0, coords[2].1, rhs[2]),
          (coords[3].0, coords[3].1, rhs[3]),
        ],
        &mut actual,
      );

      assert_eq!(actual, expected, "seed {seed}");
    }
  }

  #[test]
  fn materialized_k4_matrix_accumulate_matches_reference_layouts() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(61).strict_add(31)) as u8;
    }

    let rhs = [test_poly(0x10), test_poly(0x20), test_poly(0x30), test_poly(0x40)];

    for transpose in [false, true] {
      let mut expected = [[0u16; N]; 4];
      for entry in 0..16 {
        let ((j, i), dst, rhs_index) = matrix_accumulate_coord::<4>(entry, transpose);
        let mut sampled = [0u16; N];
        sample_ntt_into(&rho, j, i, &mut sampled);
        multiply_ntts_add_assign_scalar(&mut expected[dst], &sampled, &rhs[rhs_index]);
      }

      let mut actual = [[0u16; N]; 4];
      sample_matrix_ntt_mul_accumulate_materialized::<4>(&rho, &rhs, &mut actual, transpose);

      assert_eq!(actual, expected, "transpose={transpose}");
    }
  }

  #[test]
  fn materialized_k3_matrix_accumulate_matches_reference_layouts() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(47).strict_add(23)) as u8;
    }

    let rhs = [test_poly(0x10), test_poly(0x20), test_poly(0x30)];

    for transpose in [false, true] {
      let mut expected = [[0u16; N]; 3];
      for entry in 0..9 {
        let ((j, i), dst, rhs_index) = matrix_accumulate_coord::<3>(entry, transpose);
        let mut sampled = [0u16; N];
        sample_ntt_into(&rho, j, i, &mut sampled);
        multiply_ntts_add_assign_scalar(&mut expected[dst], &sampled, &rhs[rhs_index]);
      }

      let mut actual = [[0u16; N]; 3];
      sample_matrix_ntt_mul_accumulate_materialized::<3>(&rho, &rhs, &mut actual, transpose);

      assert_eq!(actual, expected, "transpose={transpose}");
    }
  }

  #[test]
  fn materialized_k2_matrix_accumulate_matches_reference_layouts() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(43).strict_add(19)) as u8;
    }

    let rhs = [test_poly(0x10), test_poly(0x20)];

    for transpose in [false, true] {
      let mut expected = [[0u16; N]; 2];
      for entry in 0..4 {
        let ((j, i), dst, rhs_index) = matrix_accumulate_coord::<2>(entry, transpose);
        let mut sampled = [0u16; N];
        sample_ntt_into(&rho, j, i, &mut sampled);
        multiply_ntts_add_assign_scalar(&mut expected[dst], &sampled, &rhs[rhs_index]);
      }

      let mut actual = [[0u16; N]; 2];
      sample_matrix_ntt_mul_accumulate_materialized::<2>(&rho, &rhs, &mut actual, transpose);

      assert_eq!(actual, expected, "transpose={transpose}");
    }
  }

  #[test]
  fn prepared_matrix_cache_matches_materialized_transpose() {
    assert_prepared_matrix_cache_matches_materialized_transpose::<2>();
    assert_prepared_matrix_cache_matches_materialized_transpose::<3>();
    assert_prepared_matrix_cache_matches_materialized_transpose::<4>();
  }

  fn assert_prepared_matrix_cache_matches_materialized_transpose<const K: usize>() {
    let mut rho = [0u8; SEED_BYTES];
    for (i, byte) in rho.iter_mut().enumerate() {
      *byte = (i.strict_mul(67).strict_add(29)) as u8;
    }

    let mut rhs = [[0u16; N]; K];
    for (i, poly) in rhs.iter_mut().enumerate() {
      *poly = test_poly(0x50usize.strict_add(i.strict_mul(0x10)));
    }

    let ek = PreparedEncapsulationArithmetic {
      t_hat: [[0u16; N]; K],
      rho,
      a_transpose_hat: prepare_matrix_transpose::<K>(&rho),
    };

    let mut cached = [[0u16; N]; K];
    matrix_ntt_mul_accumulate_prepared_transpose(&ek, &rhs, &mut cached);

    let mut materialized = [[0u16; N]; K];
    sample_matrix_ntt_mul_accumulate_materialized::<K>(&ek.rho, &rhs, &mut materialized, true);

    assert_eq!(cached, materialized);
  }

  fn test_poly(seed: usize) -> Poly {
    let mut poly = [0u16; N];
    for (i, coeff) in poly.iter_mut().enumerate() {
      *coeff = ((seed.strict_mul(37).strict_add(i.strict_mul(19)).strict_add(11)) % usize::from(Q)) as u16;
    }
    poly
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  fn assert_ntt_neon_matches_scalar_reference(poly: Poly, label: &str) {
    let mut scalar = poly;
    ntt_scalar(&mut scalar);

    let mut accelerated = poly;
    // SAFETY: direct aarch64 NEON test call because:
    // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
    // 2. `accelerated` is a fixed 256-coefficient polynomial matching the kernel contract.
    // 3. The NEON kernel returns the scalar/FIPS representation directly.
    // 4. The memory access schedule depends only on public ML-KEM dimensions.
    unsafe {
      ntt_neon(&mut accelerated);
    }

    assert_eq!(accelerated, scalar, "{label}");
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  fn assert_basemul_accumulate_asm_matches_scalar_reference(acc: Poly, a: Poly, b: Poly, label: &str) {
    let mut scalar = acc;
    multiply_ntts_add_assign_scalar(&mut scalar, &a, &b);

    let mut asm = acc;
    // SAFETY: direct aarch64 assembly full-polynomial test call because:
    // 1. This test only compiles on aarch64 Linux/macOS targets that include the assembly backend.
    // 2. `asm`, `a`, and `b` are fixed 256-coefficient polynomials matching the assembly ABI.
    // 3. The test compares against the scalar/FIPS accumulator before production dispatch.
    // 4. The assembly memory schedule depends only on public ML-KEM dimensions.
    unsafe {
      aarch64::test_basemul_accumulate_asm(&mut asm, &a, &b);
    }

    assert_eq!(asm, scalar, "{label}");
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  fn assert_basemul_accumulate_k3_asm_matches_scalar_reference(acc: Poly, a: PolyVec<3>, b: PolyVec<3>, label: &str) {
    let mut scalar = acc;
    for i in 0..3 {
      multiply_ntts_add_assign_scalar(&mut scalar, &a[i], &b[i]);
    }

    let mut asm = acc;
    // SAFETY: direct aarch64 assembly K=3 dot-product test call because:
    // 1. This test only compiles on aarch64 Linux/macOS targets that include the assembly backend.
    // 2. `asm`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
    // 3. The test compares against the scalar/FIPS accumulator before production dispatch.
    // 4. The assembly memory schedule depends only on public ML-KEM dimensions.
    unsafe {
      aarch64::test_basemul_accumulate_k3_asm(&mut asm, &a, &b);
    }

    assert_eq!(asm, scalar, "{label}");
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux"),
    not(miri),
    not(feature = "portable-only")
  ))]
  fn assert_basemul_accumulate_k4_asm_matches_scalar_reference(acc: Poly, a: PolyVec<4>, b: PolyVec<4>, label: &str) {
    let mut scalar = acc;
    for i in 0..4 {
      multiply_ntts_add_assign_scalar(&mut scalar, &a[i], &b[i]);
    }

    let mut asm = acc;
    // SAFETY: direct aarch64 assembly K=4 dot-product test call because:
    // 1. This test only compiles on aarch64 Linux/macOS targets that include the assembly backend.
    // 2. `asm`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
    // 3. The test compares against the scalar/FIPS accumulator before production dispatch.
    // 4. The assembly memory schedule depends only on public ML-KEM dimensions.
    unsafe {
      aarch64::test_basemul_accumulate_k4_asm(&mut asm, &a, &b);
    }

    assert_eq!(asm, scalar, "{label}");
  }

  #[test]
  fn s390x_ct_multiply_helpers_match_scalar_products() {
    const U16_SAMPLES: [u32; 14] = [0, 1, 2, 3, 7, 31, 127, 255, 256, 1024, 3328, 32767, 32768, 65535];
    for &a in &U16_SAMPLES {
      for &b in &U16_SAMPLES {
        assert_eq!(mul_u32_16_ct(a, b), a * b, "u32 {a} * {b}");
      }
    }

    const I16_SAMPLES: [i32; 13] = [
      -32768, -32767, -3329, -3328, -17, -1, 0, 1, 17, 3328, 3329, 32766, 32767,
    ];
    for &a in &I16_SAMPLES {
      for &b in &I16_SAMPLES {
        assert_eq!(mul_i32_16_ct(a, b), a * b, "i32 {a} * {b}");
      }
    }
  }

  #[test]
  fn s390x_ct_compression_division_matches_reciprocal() {
    for quotient in 0u32..=2048 {
      let base = quotient * Q_U32;
      for offset in [0, 1, Q_HALF, Q_U32 - 1, Q_U32] {
        let value = base + offset;
        if value < (1u32 << 23) {
          assert_eq!(
            div_q_compress_u32_ct(value),
            div_q_compress_u32_recip(value),
            "value {value}"
          );
        }
      }
    }
  }

  #[test]
  fn pke_compare_paths_match_full_ciphertext() {
    let mut key_random = [0u8; 64];
    let mut m = [0u8; SEED_BYTES];
    let mut r = [0u8; SEED_BYTES];
    for (i, byte) in key_random.iter_mut().enumerate() {
      *byte = (i.strict_mul(13).strict_add(7)) as u8;
    }
    for (i, byte) in m.iter_mut().enumerate() {
      *byte = (i.strict_mul(17).strict_add(23)) as u8;
    }
    for (i, byte) in r.iter_mut().enumerate() {
      *byte = (i.strict_mul(19).strict_add(29)) as u8;
    }

    let (ek512, _) = keygen::<2, 2, 192, 768, 800, 1632>(&key_random);
    let prepared512 = prepare_encapsulation_key::<2, 800>(&ek512);
    let ciphertext512 = pke_encrypt_prepared::<2, 192, 768, 10, 4, 320, 128>(&prepared512, &m, &r);
    assert_eq!(
      pke_encrypt_prepared_512_compare(&prepared512, &m, &r, &ciphertext512),
      0xff
    );
    let mut modified512 = ciphertext512;
    modified512[701] ^= 0x08;
    assert_eq!(
      pke_encrypt_prepared_512_compare(&prepared512, &m, &r, &modified512),
      0x00
    );

    let (ek768, _) = keygen::<3, 3, 128, 1152, 1184, 2400>(&key_random);
    let prepared768 = prepare_encapsulation_key::<3, 1184>(&ek768);
    let ciphertext768 = pke_encrypt_prepared_768(&prepared768, &m, &r);
    assert_eq!(
      pke_encrypt_prepared_768_compare(&prepared768, &m, &r, &ciphertext768),
      0xff
    );
    let mut modified768 = ciphertext768;
    modified768[137] ^= 0x20;
    assert_eq!(
      pke_encrypt_prepared_768_compare(&prepared768, &m, &r, &modified768),
      0x00
    );

    for (i, byte) in key_random.iter_mut().enumerate() {
      *byte = (i.strict_mul(31).strict_add(11)) as u8;
    }
    let (ek1024, _) = keygen::<4, 4, 128, 1536, 1568, 3168>(&key_random);
    let prepared1024 = prepare_encapsulation_key::<4, 1568>(&ek1024);
    let ciphertext1024 = pke_encrypt_prepared_1024(&prepared1024, &m, &r);
    assert_eq!(
      pke_encrypt_prepared_1024_compare(&prepared1024, &m, &r, &ciphertext1024),
      0xff
    );
    let mut modified1024 = ciphertext1024;
    modified1024[1491] ^= 0x04;
    assert_eq!(
      pke_encrypt_prepared_1024_compare(&prepared1024, &m, &r, &modified1024),
      0x00
    );
  }

  #[test]
  fn compress_encode_compare_matches_encoder_output() {
    fn check<const D: usize, const BYTES: usize>(seed: usize, flip_index: usize, flip_mask: u8) {
      let poly = test_poly(seed);
      let mut encoded = [0u8; BYTES];
      compress_encode_poly::<D>(&poly, &mut encoded);
      assert_eq!(compress_encode_compare_poly::<D, BYTES>(&poly, &encoded), 0xff);

      encoded[flip_index] ^= flip_mask;
      assert_eq!(compress_encode_compare_poly::<D, BYTES>(&poly, &encoded), 0x00);
    }

    for seed in 0usize..16 {
      check::<4, 128>(seed, 17, 0x20);
      check::<5, 160>(seed.strict_add(100), 43, 0x04);
      check::<10, 320>(seed.strict_add(200), 197, 0x80);
      check::<11, 352>(seed.strict_add(300), 281, 0x01);
    }
  }

  #[test]
  fn base_case_multiply_outputs_montgomery_product_domain() {
    const SAMPLES: [u16; 13] = [0, 1, 2, 7, 17, 127, 511, 1024, 1664, 1665, 2048, 3001, 3328];

    for i in 0..GAMMAS.len() {
      for (sample_index, &a0) in SAMPLES.iter().enumerate() {
        let a1 = SAMPLES[(sample_index.strict_mul(3).strict_add(i)) % SAMPLES.len()];
        let b0 = SAMPLES[(sample_index.strict_mul(5).strict_add(i.strict_mul(7))) % SAMPLES.len()];
        let b1 = SAMPLES[(sample_index.strict_mul(11).strict_add(i.strict_mul(13))) % SAMPLES.len()];

        let normal = base_case_multiply_normal_reference(a0, a1, b0, b1, GAMMAS[i]);
        let montgomery = base_case_multiply(a0, a1, b0, b1, GAMMAS_MONT[i]);

        assert_eq!(
          montgomery,
          (
            to_montgomery_product_domain(normal.0),
            to_montgomery_product_domain(normal.1)
          ),
          "gamma lane {i}, sample {sample_index}"
        );
      }
    }
  }

  #[test]
  fn product_domain_inverse_ntt_matches_normal_inverse_ntt() {
    for seed in 0usize..16 {
      let mut normal = test_poly(seed);
      let mut product_domain = normal;

      poly_to_montgomery_product_domain(&mut product_domain);
      inverse_ntt_scalar(&mut normal);
      inverse_ntt_scalar_with_scale(&mut product_domain, INV_NTT_PRODUCT_SCALE_MONT);

      assert_eq!(product_domain, normal, "seed {seed}");
    }
  }

  #[test]
  fn product_domain_round_trip_preserves_normal_coefficients() {
    for seed in 0usize..16 {
      let mut poly = test_poly(seed);
      let normal = poly;

      poly_to_montgomery_product_domain(&mut poly);
      poly_from_montgomery_product_domain(&mut poly);

      assert_eq!(poly, normal, "seed {seed}");
    }
  }

  #[test]
  fn inverse_ntt_product_add_assign_matches_two_pass() {
    for seed in 0usize..32 {
      let input = test_poly(seed);
      let addend = test_poly(seed.strict_add(100));

      let mut expected = input;
      inverse_ntt_montgomery_product(&mut expected);
      poly_add_assign(&mut expected, &addend);

      let mut actual = input;
      inverse_ntt_montgomery_product_add_assign(&mut actual, &addend);

      assert_eq!(actual, expected, "seed {seed}");
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn product_domain_neon_matches_scalar_reference() {
    for seed in 0usize..32 {
      let mut scalar = test_poly(seed);
      let mut accelerated = scalar;

      poly_to_montgomery_product_domain_scalar(&mut scalar);
      // SAFETY: direct NEON product-domain conversion test call because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `accelerated` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        to_montgomery_product_domain_neon(&mut accelerated);
      }
      assert_eq!(accelerated, scalar, "to product-domain seed {seed}");

      let mut scalar = test_poly(seed.strict_add(100));
      let mut accelerated = scalar;

      poly_from_montgomery_product_domain_scalar(&mut scalar);
      // SAFETY: direct NEON product-domain exit test call because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `accelerated` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. `MONT_R_SQUARED_MOD_Q` is the public ML-KEM conversion constant.
      // 4. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        from_montgomery_product_domain_neon(&mut accelerated);
      }
      assert_eq!(accelerated, scalar, "from product-domain seed {seed}");
    }
  }

  #[test]
  fn multiply_ntts_chunk_scalar_matches_full_scalar_accumulator() {
    for seed in 0usize..16 {
      let acc = test_poly(seed);
      let a = test_poly(seed.strict_add(100));
      let b = test_poly(seed.strict_add(200));

      let mut full = acc;
      multiply_ntts_add_assign_scalar(&mut full, &a, &b);

      let mut chunked = acc;
      for coeff_offset in (0..N).step_by(SAMPLE_NTT_ACC_CHUNK_COEFFS) {
        let mut chunk = [0u16; SAMPLE_NTT_ACC_CHUNK_COEFFS];
        chunk.copy_from_slice(&a[coeff_offset..coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS)]);
        multiply_ntts_add_assign_chunk_scalar(&mut chunked, &chunk, &b, coeff_offset);
      }

      assert_eq!(chunked, full, "seed {seed}");
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn ntt_neon_round_trip_matches_scalar_reference() {
    for seed in 0usize..16 {
      let mut scalar = test_poly(seed);
      let mut accelerated = scalar;

      ntt_scalar(&mut scalar);
      // SAFETY: direct NEON kernel test call because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `accelerated` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        ntt_neon(&mut accelerated);
      }

      assert_eq!(accelerated, scalar, "forward seed {seed}");

      inverse_ntt_scalar(&mut scalar);
      // SAFETY: direct NEON inverse kernel test call because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `accelerated` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        inverse_ntt_neon(&mut accelerated, INV_NTT_SCALE_MONT);
      }

      assert_eq!(accelerated, scalar, "inverse seed {seed}");
    }
  }

  #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn ntt_avx2_matches_scalar_reference() {
    if !crate::platform::caps().has(crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41) {
      return;
    }

    for seed in 0usize..16 {
      let mut scalar = test_poly(seed);
      let mut accelerated = scalar;

      ntt_scalar(&mut scalar);
      // SAFETY: direct AVX2/SSE4.1 kernel test call because:
      // 1. Runtime capability detection confirmed AVX2 and SSE4.1 above.
      // 2. `accelerated` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        ntt_avx2(&mut accelerated);
      }

      assert_eq!(accelerated, scalar, "forward seed {seed}");

      inverse_ntt_scalar(&mut scalar);
      // SAFETY: direct AVX2/SSE4.1 inverse kernel test call because:
      // 1. Runtime capability detection confirmed AVX2 and SSE4.1 above.
      // 2. `accelerated` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions.
      unsafe {
        inverse_ntt_avx2(&mut accelerated, INV_NTT_SCALE_MONT);
      }

      assert_eq!(accelerated, scalar, "inverse seed {seed}");
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn multiply_ntts_neon_matches_scalar_accumulator() {
    for seed in 0usize..16 {
      let mut acc = [0u16; N];
      let mut a = [0u16; N];
      let mut b = [0u16; N];

      for i in 0usize..N {
        acc[i] = ((seed.strict_mul(19).strict_add(i.strict_mul(7))) % usize::from(Q)) as u16;
        a[i] = ((seed.strict_mul(31).strict_add(i.strict_mul(11)).strict_add(5)) % usize::from(Q)) as u16;
        b[i] = ((seed.strict_mul(43).strict_add(i.strict_mul(13)).strict_add(17)) % usize::from(Q)) as u16;
      }

      let mut scalar = acc;
      multiply_ntts_add_assign_scalar(&mut scalar, &a, &b);

      let mut neon = acc;
      // SAFETY: direct NEON kernel test call because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `neon`, `a`, and `b` are fixed 256-coefficient arrays matching the kernel contract.
      // 3. The test keeps `neon` distinct from the read-only inputs, so the mutable output cannot alias
      //    them.
      unsafe {
        multiply_ntts_add_assign_neon(&mut neon, &a, &b);
      }

      assert_eq!(neon, scalar, "seed {seed}");
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn multiply_ntts_accumulate_neon_matches_scalar_dot_product() {
    for seed in 0usize..16 {
      let acc = test_poly(seed);
      let a3: PolyVec<3> = core::array::from_fn(|i| test_poly(seed.strict_add(100).strict_add(i.strict_mul(17))));
      let b3: PolyVec<3> = core::array::from_fn(|i| test_poly(seed.strict_add(200).strict_add(i.strict_mul(23))));

      let mut scalar = acc;
      for i in 0..3 {
        multiply_ntts_add_assign_scalar(&mut scalar, &a3[i], &b3[i]);
      }

      let mut neon = acc;
      // SAFETY: direct aarch64 NEON K=3 dot-product test call because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `neon`, `a3`, and `b3` are fixed-size ML-KEM polynomial arrays matching the kernel contract.
      // 3. The test keeps `neon` distinct from the read-only inputs, so the mutable output cannot alias
      //    them.
      // 4. The kernel's memory access schedule is fixed and independent of secret coefficient values.
      unsafe {
        multiply_ntts_accumulate_k3_neon(&mut neon, [&a3[0], &a3[1], &a3[2]], [&b3[0], &b3[1], &b3[2]]);
      }
      assert_eq!(neon, scalar, "K=3 seed {seed}");

      let acc = test_poly(seed.strict_add(300));
      let a4: PolyVec<4> = core::array::from_fn(|i| test_poly(seed.strict_add(400).strict_add(i.strict_mul(29))));
      let b4: PolyVec<4> = core::array::from_fn(|i| test_poly(seed.strict_add(500).strict_add(i.strict_mul(31))));

      let mut scalar = acc;
      for i in 0..4 {
        multiply_ntts_add_assign_scalar(&mut scalar, &a4[i], &b4[i]);
      }

      let mut neon = acc;
      // SAFETY: direct aarch64 NEON K=4 dot-product test call because:
      // 1. This test only compiles on aarch64, where NEON is baseline for supported rscrypto targets.
      // 2. `neon`, `a4`, and `b4` are fixed-size ML-KEM polynomial arrays matching the kernel contract.
      // 3. The test keeps `neon` distinct from the read-only inputs, so the mutable output cannot alias
      //    them.
      // 4. The kernel's memory access schedule is fixed and independent of secret coefficient values.
      unsafe {
        multiply_ntts_accumulate_k4_neon(
          &mut neon,
          [&a4[0], &a4[1], &a4[2], &a4[3]],
          [&b4[0], &b4[1], &b4[2], &b4[3]],
        );
      }
      assert_eq!(neon, scalar, "K=4 seed {seed}");
    }
  }

  #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn multiply_ntts_avx2_matches_scalar_accumulator() {
    if !crate::platform::caps().has(crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41) {
      return;
    }

    for seed in 0usize..16 {
      let acc = test_poly(seed);
      let a = test_poly(seed.strict_add(100));
      let b = test_poly(seed.strict_add(200));

      let mut scalar = acc;
      multiply_ntts_add_assign_scalar(&mut scalar, &a, &b);

      let mut avx2 = acc;
      // SAFETY: direct AVX2/SSE4.1 kernel test call because:
      // 1. Runtime capability detection confirmed AVX2 and SSE4.1 above.
      // 2. `avx2`, `a`, and `b` are fixed 256-coefficient arrays matching the kernel contract.
      // 3. The test keeps `avx2` distinct from the read-only inputs, so the mutable output cannot alias
      //    them.
      unsafe {
        multiply_ntts_add_assign_avx2(&mut avx2, &a, &b);
      }

      assert_eq!(avx2, scalar, "seed {seed}");
    }
  }

  #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn multiply_ntts_avx512_matches_scalar_accumulator() {
    let avx2_required = crate::platform::caps::x86::AVX2 | crate::platform::caps::x86::SSE41;
    let avx512_required = crate::platform::caps::x86::AVX512_READY | avx2_required;
    if !crate::platform::caps().has(avx512_required) {
      return;
    }

    for seed in 0usize..16 {
      let acc = test_poly(seed);
      let a = test_poly(seed.strict_add(100));
      let b = test_poly(seed.strict_add(200));

      let mut scalar = acc;
      multiply_ntts_add_assign_scalar(&mut scalar, &a, &b);

      let mut avx512 = acc;
      // SAFETY: direct AVX-512 kernel test call because:
      // 1. Runtime capability detection confirmed AVX2, SSE4.1, AVX512F, AVX512VL, AVX512BW, and AVX512DQ
      //    above.
      // 2. `avx512`, `a`, and `b` are fixed 256-coefficient arrays matching the kernel contract.
      // 3. The test keeps `avx512` distinct from the read-only inputs, so the mutable output cannot alias
      //    them.
      unsafe {
        x86_64::multiply_ntts_add_assign_avx512(&mut avx512, &a, &b);
      }

      assert_eq!(avx512, scalar, "seed {seed}");
    }
  }

  #[test]
  fn multiply_ntts_accumulate_matches_scalar_dot_product() {
    fn check<const K: usize>(seed: usize) {
      let acc = test_poly(seed);
      let a: PolyVec<K> = core::array::from_fn(|i| test_poly(seed.strict_add(100).strict_add(i.strict_mul(17))));
      let b: PolyVec<K> = core::array::from_fn(|i| test_poly(seed.strict_add(200).strict_add(i.strict_mul(23))));

      let mut scalar = acc;
      for i in 0..K {
        multiply_ntts_add_assign_scalar(&mut scalar, &a[i], &b[i]);
      }

      let mut actual = acc;
      multiply_ntts_accumulate(&mut actual, &a, &b);

      assert_eq!(actual, scalar, "K={K} seed={seed}");
    }

    for seed in 0usize..16 {
      check::<2>(seed);
      check::<3>(seed);
      check::<4>(seed);
    }
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

    let mut encoded_5 = [0u8; 160];
    let mut decoded_5 = [0u16; N];
    byte_encode::<5>(&poly, &mut encoded_5);
    byte_decode::<5>(&encoded_5, &mut decoded_5);
    for i in 0..N {
      assert_eq!(decoded_5[i], poly[i] & 0x001f);
    }

    let mut encoded_10 = [0u8; 320];
    let mut decoded_10 = [0u16; N];
    byte_encode::<10>(&poly, &mut encoded_10);
    byte_decode::<10>(&encoded_10, &mut decoded_10);
    for i in 0..N {
      assert_eq!(decoded_10[i], poly[i] & 0x03ff);
    }

    let mut encoded_11 = [0u8; 352];
    let mut decoded_11 = [0u16; N];
    byte_encode::<11>(&poly, &mut encoded_11);
    byte_decode::<11>(&encoded_11, &mut decoded_11);
    for i in 0..N {
      assert_eq!(decoded_11[i], poly[i] & 0x07ff);
    }

    let mut encoded_12 = [0u8; 384];
    let mut decoded_12 = [0u16; N];
    byte_encode::<12>(&poly, &mut encoded_12);
    byte_decode::<12>(&encoded_12, &mut decoded_12);
    assert_eq!(decoded_12, poly);
  }

  #[test]
  fn fused_compress_encode_matches_two_pass_codec() {
    let poly = test_poly(0x51);

    macro_rules! assert_width {
      ($d:literal, $bytes:literal) => {{
        let mut compressed = [0u16; N];
        let mut expected = [0u8; $bytes];
        let mut actual = [0u8; $bytes];
        compress_poly::<$d>(&poly, &mut compressed);
        byte_encode::<$d>(&compressed, &mut expected);
        compress_encode_poly::<$d>(&poly, &mut actual);
        assert_eq!(actual, expected, "d={}", $d);
      }};
    }

    assert_width!(1, 32);
    assert_width!(4, 128);
    assert_width!(5, 160);
    assert_width!(10, 320);
    assert_width!(11, 352);
  }

  #[test]
  fn fused_decode_decompress_matches_two_pass_codec() {
    let mut input = [0u8; 352];
    for (i, byte) in input.iter_mut().enumerate() {
      *byte = (i.strict_mul(73).strict_add(19)) as u8;
    }

    macro_rules! assert_width {
      ($d:literal, $bytes:literal) => {{
        let input = &input[..$bytes];
        let mut decoded = [0u16; N];
        let mut expected = [0u16; N];
        let mut actual = [0u16; N];
        byte_decode::<$d>(input, &mut decoded);
        decompress_poly::<$d>(&decoded, &mut expected);
        decode_decompress_poly::<$d>(input, &mut actual);
        assert_eq!(actual, expected, "d={}", $d);
      }};
    }

    assert_width!(1, 32);
    assert_width!(4, 128);
    assert_width!(5, 160);
    assert_width!(10, 320);
    assert_width!(11, 352);
  }

  #[test]
  fn fused_message_decompress_add_matches_two_pass_codec() {
    let mut message = [0u8; SEED_BYTES];
    for (i, byte) in message.iter_mut().enumerate() {
      *byte = (i.strict_mul(37).strict_add(11)) as u8;
    }

    let base = test_poly(0x72);
    let mut decoded = [0u16; N];
    let mut expected = base;
    let mut actual = base;
    byte_decode::<1>(&message, &mut decoded);
    decompress_poly_add_assign::<1>(&decoded, &mut expected);
    decompress_message_add_assign(&message, &mut actual);
    assert_eq!(actual, expected);
  }

  #[test]
  fn fused_subtract_compress_encode_message_matches_two_pass_codec() {
    let lhs = test_poly(0x83);
    let rhs = test_poly(0xa7);
    let mut diff = [0u16; N];
    let mut expected = [0u8; SEED_BYTES];
    let mut actual = [0u8; SEED_BYTES];

    for i in 0..N {
      diff[i] = sub_mod(lhs[i], rhs[i]);
    }
    compress_encode_poly::<1>(&diff, &mut expected);
    subtract_compress_encode_message(&lhs, &rhs, &mut actual);

    assert_eq!(actual, expected);
  }

  #[test]
  fn decompressed_values_recompress_to_original() {
    for d in [1usize, 4, 5, 10, 11] {
      let max = 1u16 << d;
      for y in 0..max {
        let x = ((Q_U32 * u32::from(y)) + (1u32 << (d - 1))) >> d;
        let compressed = (div_q_compress_u32((x << d) + Q_HALF) & ((1u32 << d) - 1)) as u16;
        assert_eq!(compressed, y, "d={d} y={y}");
      }
    }
  }
}
