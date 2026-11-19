use rscrypto::{Aes256GcmKey, Ed25519SecretKey, MlDsa65SecretKey, MlKem768DecapsulationKey, X25519SecretKey};
use serde::de::value::{BytesDeserializer, Error, SeqDeserializer};

/// Deserialize `data` as `T` through both serde visitor paths, bytes and a sequence of `u8`, and require each
/// to agree with `oracle`: the fixed-array constructor for byte owners, and validated import for ML-DSA.
fn check<T: serde::de::DeserializeOwned, B: PartialEq + core::fmt::Debug>(
  data: &[u8],
  oracle: impl Fn(&[u8]) -> Option<T>,
  bytes: impl Fn(&T) -> B,
) {
  let expected = oracle(data).map(|value| bytes(&value));
  let from_bytes = T::deserialize(BytesDeserializer::<Error>::new(data)).ok();
  assert_eq!(
    from_bytes.map(|value| bytes(&value)),
    expected,
    "bytes deserializer mismatch"
  );
  let from_seq = T::deserialize(SeqDeserializer::<_, Error>::new(data.iter().copied())).ok();
  assert_eq!(
    from_seq.map(|value| bytes(&value)),
    expected,
    "sequence deserializer mismatch"
  );
}

pub(super) fn run(data: &[u8]) {
  let Some((&selector, data)) = data.split_first() else {
    return;
  };
  match selector % 5 {
    0 => check(
      data,
      |data| data.try_into().ok().map(Aes256GcmKey::from_bytes),
      |key| *key.as_bytes(),
    ),
    1 => check(
      data,
      |data| data.try_into().ok().map(X25519SecretKey::from_bytes),
      |key| *key.as_bytes(),
    ),
    2 => check(
      data,
      |data| data.try_into().ok().map(Ed25519SecretKey::from_bytes),
      |key| *key.as_bytes(),
    ),
    3 => check(
      data,
      |data| data.try_into().ok().map(MlKem768DecapsulationKey::from_bytes),
      |key| key.as_bytes().to_vec(),
    ),
    _ => check(
      data,
      |data| MlDsa65SecretKey::try_from_slice(data).ok(),
      |key| key.expose_secret().as_bytes().to_vec(),
    ),
  }
}
