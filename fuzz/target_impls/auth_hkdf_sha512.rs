use rscrypto::HkdfSha512;
use rscrypto_fuzz::{FuzzInput, assert_hkdf_against_oracle, some_or_return, split_at_ratio};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let salt_split: u8 = some_or_return!(input.byte());
  let ikm_split: u8 = some_or_return!(input.byte());
  let out_len_bytes: [u8; 2] = some_or_return!(input.bytes());
  let (salt, remainder) = split_at_ratio(input.rest(), salt_split);
  let (ikm, info) = split_at_ratio(remainder, ikm_split);
  // Valid and over-length expand requests, without unbounded buffers.
  let out_len = usize::from(u16::from_le_bytes(out_len_bytes)).rem_euclid(HkdfSha512::MAX_OUTPUT_SIZE.strict_add(65));

  let hk = HkdfSha512::new(salt, ikm);
  assert_eq!(
    hk.prk(),
    HkdfSha512::extract(salt, ikm).prk(),
    "extract vs new PRK mismatch"
  );
  let mut okm = vec![0u8; out_len];
  let ours_expand = hk.expand(info, &mut okm);
  let mut derived = vec![0u8; out_len];
  let ours_derive = HkdfSha512::derive(salt, ikm, info, &mut derived);
  assert_eq!(
    ours_expand.is_ok(),
    ours_derive.is_ok(),
    "expand vs derive result mismatch"
  );
  if ours_expand.is_ok() {
    assert_eq!(okm, derived, "expand vs derive bytes mismatch");
  }

  // Differential: rscrypto ↔ hkdf crate
  let oracle = hkdf::Hkdf::<sha2::Sha512>::new(Some(salt), ikm);
  let mut oracle_okm = vec![0u8; out_len];
  let oracle_expand = oracle.expand(info, &mut oracle_okm);
  assert_hkdf_against_oracle(ours_expand.is_ok(), &okm, oracle_expand.is_ok(), &oracle_okm);
}
