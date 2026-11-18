use rscrypto_rustls::quic::initial_keys;
use rustls::quic::Side;

/// Apply and remove production header protection with arbitrary sample and
/// packet-number lengths. A rejected input is left unchanged, and an accepted
/// one round-trips.
pub(super) fn run(data: &[u8]) {
  if data.len() > 4096 {
    return;
  }
  let Some((&lengths, rest)) = data.split_first() else {
    return;
  };
  let Some((&first, rest)) = rest.split_first() else {
    return;
  };
  // Lengths around the valid values: samples of 0..=33 bytes and packet numbers of 0..=7 bytes.
  let sample_len = usize::from(lengths >> 3).min(rest.len());
  let (sample, rest) = rest.split_at(sample_len);
  let pn_len = usize::from(lengths & 7).min(rest.len());
  let pn = &rest[..pn_len];
  let keys = initial_keys(&[1; 8], Side::Client);
  for key in [&*keys.remote.header, &*keys.local.header] {
    let (mut masked_first, mut masked_pn) = (first, pn.to_vec());
    match key.decrypt_in_place(sample, &mut masked_first, &mut masked_pn) {
      Ok(()) => {
        key
          .encrypt_in_place(sample, &mut masked_first, &mut masked_pn)
          .expect("a removed mask can be applied again");
        assert_eq!((masked_first, masked_pn.as_slice()), (first, pn));
      }
      Err(_) => assert_eq!((masked_first, masked_pn.as_slice()), (first, pn)),
    }
  }
}
