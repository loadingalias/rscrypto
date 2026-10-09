use std::io::Read;

use rscrypto::hashes::expert::bao::Decoder;

fn compare(encoded: &[u8], root: &[u8; 32]) {
  let mut ours = Decoder::new(encoded, root);
  let mut oracle = bao::decode::Decoder::new(encoded, &blake3::Hash::from(*root));
  let mut ours_bytes = Vec::new();
  let mut oracle_bytes = Vec::new();
  let ours_result = ours.read_to_end(&mut ours_bytes);
  let oracle_result = oracle.read_to_end(&mut oracle_bytes);
  assert_eq!(ours_result.is_ok(), oracle_result.is_ok(), "Bao acceptance differs");
  if ours_result.is_ok() {
    assert_eq!(ours_bytes, oracle_bytes, "Bao decoded bytes differ");
    assert_eq!(
      blake3::hash(&ours_bytes).as_bytes(),
      root,
      "accepted output has wrong root"
    );
  } else {
    // Reader buffering can return different amounts of the same verified prefix.
    assert!(ours_bytes.starts_with(&oracle_bytes) || oracle_bytes.starts_with(&ours_bytes));
    let mut output = [0xa5; 19];
    assert!(ours.read(&mut output).is_err(), "failed parser resumed");
    assert_eq!(output, [0xa5; 19], "failed parser wrote output");
  }
}

pub(super) fn run(data: &[u8]) {
  let (encoded, root) = bao::encode::encode(data);
  compare(&encoded, root.as_bytes());
  if let Some((&control, rest)) = data.split_first() {
    // Combined Bao encoding always includes its eight-byte length header.
    let position = rest
      .iter()
      .take(8)
      .fold(usize::from(control), |n, &b| {
        n.wrapping_mul(257).wrapping_add(usize::from(b))
      })
      .strict_rem(encoded.len());
    let mut damaged = encoded;
    damaged[position] ^= control | 1;
    compare(&damaged, root.as_bytes());
    compare(&damaged[..position], root.as_bytes());
  }
  if let Some((root, encoded)) = data.split_first_chunk::<32>() {
    compare(encoded, root);
  }
}
