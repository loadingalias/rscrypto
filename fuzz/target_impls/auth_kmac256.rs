use rscrypto::Kmac256;
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

fn encoded_string_len(len: usize) -> usize {
  let bits = len.strict_mul(8);
  let width = ((usize::BITS - bits.leading_zeros()) as usize).div_ceil(8).max(1);
  1usize.strict_add(width).strict_add(len)
}

fn bytepad_is_aligned(rate: usize, segments: &[usize]) -> bool {
  let encoded_len = segments
    .iter()
    .map(|&len| encoded_string_len(len))
    .fold(2usize, usize::strict_add);
  encoded_len.is_multiple_of(rate)
}

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let split: u8 = some_or_return!(input.byte());
  let key_split: u8 = some_or_return!(input.byte());
  let out_len_byte: u8 = some_or_return!(input.byte());
  let rest = input.rest();

  let (key, remainder) = split_at_ratio(rest, key_split);
  let (custom, message) = split_at_ratio(remainder, split);
  let out_len = (out_len_byte as usize % 128).strict_add(1);

  // Property: streaming equivalence
  let mut expected = vec![0u8; out_len];
  Kmac256::mac_into(key, custom, message, &mut expected);

  let (a, b) = split_at_ratio(message, split.wrapping_add(37));
  let mut kmac = Kmac256::new(key, custom);
  kmac.update(a);
  kmac.update(b);
  let mut got = vec![0u8; out_len];
  kmac.finalize_into(&mut got);
  assert_eq!(expected, got, "streaming kmac mismatch");

  // Property: reset restores initial state
  kmac.reset();
  kmac.update(message);
  let mut reset_out = vec![0u8; out_len];
  kmac.finalize_into(&mut reset_out);
  assert_eq!(expected, reset_out, "kmac changed after reset");

  // Property: the authentication API enforces its documented strength floor,
  // while the primitive API accepts every nonempty KMAC output.
  assert_eq!(
    Kmac256::verify_tag(key, custom, message, &expected).is_ok(),
    out_len >= Kmac256::MIN_AUTH_TAG_SIZE,
    "kmac256 authentication policy mismatch"
  );
  Kmac256::verify_tag_primitive(key, custom, message, &expected)
    .expect("primitive verification must accept a correct nonempty output");

  let mut corrupted = expected.clone();
  corrupted[out_len / 2] ^= 1;
  assert!(
    Kmac256::verify_tag_primitive(key, custom, message, &corrupted).is_err(),
    "primitive verification accepted a corrupted output"
  );
  assert!(
    Kmac256::verify_tag(key, custom, message, &corrupted).is_err(),
    "authentication verification accepted a corrupted tag"
  );

  // tiny-keccak 2.0.2 mishandles an exactly full SP 800-185 bytepad block.
  // Independent fixed vectors cover those boundaries in the integration tests.
  if !bytepad_is_aligned(136, &[4, custom.len()]) && !bytepad_is_aligned(136, &[key.len()]) {
    use tiny_keccak::{Hasher, Kmac as OracleKmac};

    let mut oracle = OracleKmac::v256(key, custom);
    oracle.update(message);
    let mut oracle_out = vec![0u8; out_len];
    oracle.finalize(&mut oracle_out);
    assert_eq!(expected, oracle_out, "tiny-keccak kmac oracle mismatch");
  }
}
