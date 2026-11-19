use poly1305::universal_hash::KeyInit as _;
use rscrypto::{Poly1305, Poly1305OneTimeKey, Poly1305Tag};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key: [u8; 32] = some_or_return!(input.bytes());
  let split: u8 = some_or_return!(input.byte());
  let message = input.rest();
  let one_time_key = || Poly1305OneTimeKey::from_bytes(key);

  // Differential: rscrypto ↔ RustCrypto poly1305
  let tag = Poly1305::authenticate_once(one_time_key(), message);
  let oracle = poly1305::Poly1305::new(&key.into()).compute_unpadded(message);
  assert_eq!(tag.as_slice(), oracle.as_slice(), "poly1305 oracle mismatch");

  // Property: streaming over an arbitrary split matches the one-shot tag.
  let (a, b) = split_at_ratio(message, split);
  let mut mac = Poly1305::new(one_time_key());
  mac.update(a);
  mac.update(b);
  assert_eq!(mac.finalize().as_bytes(), tag.as_bytes(), "poly1305 streaming mismatch");

  // Property: verification accepts the tag and rejects a one-bit change.
  Poly1305::verify_once(one_time_key(), message, &tag).expect("verify must accept the correct tag");
  let mut forged = tag.to_bytes();
  forged[usize::from(split) % Poly1305::TAG_SIZE] ^= 1;
  assert!(
    Poly1305::verify_once(one_time_key(), message, &Poly1305Tag::from_bytes(forged)).is_err(),
    "poly1305 verification accepted a forged tag"
  );
}
