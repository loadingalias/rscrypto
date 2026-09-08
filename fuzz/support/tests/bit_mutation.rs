use rscrypto_fuzz_support::FuzzInput;

#[test]
fn every_position_and_bit_is_reachable() {
  // ML-KEM ciphertext sizes, the default live-fuzz limit, and beyond u16.
  for len in [1usize, 16, 768, 1088, 1568, 65536, 65537] {
    let mut bytes = vec![0u8; len];
    for position in 0..len {
      for bit in 0..8 {
        let mut control = (position as u64).to_le_bytes().to_vec();
        control.push(bit);
        let mut input = FuzzInput::new(&control);
        input
          .bit_mutation()
          .expect("complete mutation controls must parse")
          .apply(&mut bytes);
        assert_eq!(bytes[position], 1 << bit, "len={len}, position={position}, bit={bit}");
        bytes[position] = 0;
      }
    }
    assert!(bytes.iter().all(|&byte| byte == 0), "mutation changed extra bytes");
  }
}

#[test]
fn wide_controls_wrap_and_leave_payload_untouched() {
  let mut control = u64::MAX.to_le_bytes().to_vec();
  control.extend_from_slice(&[255, 42, 43]);
  let mut input = FuzzInput::new(&control);
  let mutation = input.bit_mutation().expect("complete mutation controls must parse");
  assert_eq!(input.rest(), &[42, 43]);
  let mut bytes = [0u8; 16];
  mutation.apply(&mut bytes);
  assert_eq!(bytes, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128]);
}

#[test]
fn truncated_controls_are_rejected() {
  for len in 0..9 {
    assert!(FuzzInput::new(&[0; 9][..len]).bit_mutation().is_none());
  }
}
