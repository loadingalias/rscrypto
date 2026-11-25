fn encoded_string_len(len: usize) -> usize {
  let bits = len.strict_mul(8);
  let width_bits = usize::BITS.strict_sub(bits.leading_zeros());
  let width = usize::try_from(width_bits)
    .expect("encoded-string width must fit usize")
    .div_ceil(8)
    .max(1);
  1usize.strict_add(width).strict_add(len)
}

fn bytepad_is_aligned(rate: usize, segments: &[usize]) -> bool {
  2usize
    .strict_add(segments.iter().map(|&len| encoded_string_len(len)).sum::<usize>())
    .is_multiple_of(rate)
}

proptest! {
  #[test]
  fn matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..256,
  ) {
    prop_assume!(!bytepad_is_aligned(RATE, &[4, customization.len()]));
    prop_assume!(!bytepad_is_aligned(RATE, &[key.len()]));
    let mut expected = vec![0u8; out_len];
    oracle(&key, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Kmac::mac_into(&key, &customization, &data, &mut actual);

    prop_assert_eq!(actual.as_slice(), expected.as_slice());
    if expected.is_empty() {
      prop_assert!(Kmac::verify_tag(&key, &customization, &data, &expected).is_err());
    } else {
      prop_assert_eq!(
        Kmac::verify_tag(&key, &customization, &data, &expected).is_ok(),
        expected.len() >= Kmac::MIN_AUTH_TAG_SIZE
      );
      prop_assert_eq!(
        Kmac::verify_tag_primitive(&key, &customization, &data, &expected),
        Ok(())
      );
    }
  }

  #[test]
  fn streaming_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..256,
  ) {
    prop_assume!(!bytepad_is_aligned(RATE, &[4, customization.len()]));
    prop_assume!(!bytepad_is_aligned(RATE, &[key.len()]));
    let mut expected = vec![0u8; out_len];
    oracle(&key, &customization, &data, &mut expected);

    let mut kmac = Kmac::new(&key, &customization);
    let mut i = 0usize;
    while i < data.len() {
      let step = (usize::from(data[i]) % 97).strict_add(1);
      let end = core::cmp::min(data.len(), i.strict_add(step));
      kmac.update(&data[i..end]);
      i = end;
    }

    let mut actual = vec![0u8; out_len];
    kmac.finalize_into(&mut actual);
    prop_assert_eq!(actual, expected);
  }
}

#[test]
fn one_shot_block_boundaries_match_tiny_keccak() {
  // Place right_encode(L) across successive rate boundaries and squeeze through the next block.
  let key = [0x5a; 32];
  let customization = b"boundary";
  assert!(!bytepad_is_aligned(RATE, &[4, customization.len()]));
  assert!(!bytepad_is_aligned(RATE, &[key.len()]));
  const LAST: usize = RATE.strict_mul(2).strict_add(1);
  let data = [0xa5; LAST];
  for len in (RATE.strict_sub(4)..=RATE.strict_add(1)).chain(RATE.strict_mul(2).strict_sub(4)..=LAST) {
    for out_len in [1, 32, RATE, RATE.strict_add(1), LAST] {
      let mut expected = vec![0u8; out_len];
      oracle(&key, customization, &data[..len], &mut expected);
      let mut actual = vec![0u8; out_len];
      Kmac::mac_into(&key, customization, &data[..len], &mut actual);
      assert_eq!(actual, expected, "message {len} bytes, output {out_len} bytes");
    }
  }
}
