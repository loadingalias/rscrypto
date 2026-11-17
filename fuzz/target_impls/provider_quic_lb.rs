use rscrypto::ChaCha20Rng;
use rscrypto_rustls::quic::{FirstOctet, LoadBalancerConfig, LoadBalancerKey, RotationId};
use rscrypto_rustls::replay::{self, ScopedRandom, Stream};

/// Decode arbitrary connection IDs under a QUIC-LB configuration that the first two bytes choose,
/// then issue and decode an ID from the following bytes.
///
/// A decode succeeds exactly when the ID carries the configuration's rotation and has enough
/// octets (draft-ietf-quic-load-balancers-22, section 4.1). An issued ID has the configured
/// length and rotation and decodes to its server ID. Random first-octet bits come from a stream
/// seeded by the input. Outside a scope, `ScopedRandom` refuses: a random-bit configuration then
/// issues nothing, and a length-describing one, which draws nothing, issues the same ID.
pub(super) fn run(data: &[u8]) {
  let Some((&[shape, nonce_shape], rest)) = data.split_first_chunk::<2>() else {
    return;
  };
  let Some((seed, rest)) = rest.split_first_chunk::<{ ChaCha20Rng::SEED_SIZE }>() else {
    return;
  };
  let rotation = RotationId::new(shape % 7).expect("codepoints 0 through 6");
  // The server ID is 1 to 15 octets, so the nonce span is 1 to 15 and nonzero, and the total is
  // at most 19.
  let server_id_len = (usize::from(shape >> 3) % 15).strict_add(1);
  let nonce_span = 16usize.strict_sub(server_id_len);
  let nonce_len = usize::from(nonce_shape & 0x7f)
    .checked_rem(nonce_span)
    .expect("a nonzero nonce span")
    .strict_add(4);
  let first_octet = if nonce_shape & 0x80 == 0 {
    FirstOctet::Length
  } else {
    FirstOctet::Random
  };
  let key = LoadBalancerKey::from_bytes([0x51; 16]);
  let config = LoadBalancerConfig::new(rotation, server_id_len, nonce_len, first_octet, key).expect("allowed lengths");
  let id_len = config.connection_id_len();

  let routable = rest.first().is_some_and(|first| first >> 5 == rotation.get()) && rest.len() >= id_len;
  assert_eq!(config.decode(rest).is_some(), routable);

  let Some((server_id, rest)) = rest.split_at_checked(server_id_len) else {
    return;
  };
  let Some(nonce) = rest.get(..nonce_len) else {
    return;
  };
  let stream = Stream::new(ChaCha20Rng::from_seed(*seed));
  let id = replay::scope(&stream, || config.encode(server_id, nonce, &ScopedRandom)).expect("configured lengths");
  let id = id.as_slice();
  assert_eq!(id.len(), id_len);
  assert_eq!(id[0] >> 5, rotation.get());
  if first_octet == FirstOctet::Length {
    assert_eq!(usize::from(id[0] & 0x1f), id_len.strict_sub(1));
  }
  assert_eq!(config.decode(id).expect("issued ID").as_slice(), server_id);

  let refused = config.encode(server_id, nonce, &ScopedRandom);
  match first_octet {
    FirstOctet::Random => assert!(refused.is_err(), "a refusing source issues nothing"),
    FirstOctet::Length => assert_eq!(refused.expect("no draw").as_slice(), id),
  }
}
