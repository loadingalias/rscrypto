#![cfg(all(feature = "blake3", feature = "std"))]

use std::io::{self, Read};

use rscrypto::hashes::expert::bao::Decoder;

fn input(len: usize) -> Vec<u8> {
  (0..len).map(|i| u8::try_from(i % 251).expect("byte")).collect()
}

fn verify_encoding(data: &[u8], read_len: usize) {
  let (encoding, root) = bao::encode::encode(data);
  assert_eq!(root.as_bytes(), blake3::hash(data).as_bytes());
  let mut decoder = Decoder::new(encoding.as_slice(), root.as_bytes());
  let mut chunk = vec![0u8; read_len];
  let mut output = Vec::new();
  loop {
    let count = decoder.read(&mut chunk).expect("valid independent encoding");
    if count == 0 {
      break;
    }
    output.extend_from_slice(&chunk[..count]);
    assert_eq!(output, data[..output.len()], "verified prefix");
  }
  assert_eq!(output, data);
  assert_eq!(decoder.read(&mut chunk).expect("stable EOF"), 0);
  assert!(decoder.into_inner().is_empty());
}

#[test]
fn independent_encodings_cover_all_small_lengths_and_tree_shapes() {
  for len in 0..=2048 {
    verify_encoding(&input(len), len % 137 + 1);
  }
  for len in [
    2049, 3071, 3072, 3073, 4095, 4096, 4097, 5121, 6145, 7169, 8192, 16385, 65537, 1_048_577,
  ] {
    verify_encoding(&input(len), 1537);
  }
}

#[test]
fn official_combined_vectors_anchor_encoder_and_decoder() {
  let vectors: serde_json::Value =
    serde_json::from_str(include_str!("../testdata/blake3_bao/test_vectors.json")).expect("official Bao vectors");
  for case in vectors["encode"].as_array().expect("combined encodings") {
    let len = usize::try_from(case["input_len"].as_u64().expect("input length")).expect("fixture fits usize");
    let data: Vec<u8> = (1u32..).flat_map(u32::to_le_bytes).take(len).collect();
    let (mut encoding, root) = bao::encode::encode(&data);
    assert_eq!(root.to_hex().as_str(), case["bao_hash"].as_str().expect("root"));
    assert_eq!(
      u64::try_from(encoding.len()).expect("encoding length"),
      case["output_len"].as_u64().expect("output length")
    );
    assert_eq!(
      blake3::hash(&encoding).to_hex().as_str(),
      case["encoded_blake3"].as_str().expect("encoding hash")
    );
    let mut decoded = Vec::new();
    Decoder::new(encoding.as_slice(), root.as_bytes())
      .read_to_end(&mut decoded)
      .expect("official combined encoding");
    assert_eq!(decoded, data);
    for offset in case["corruptions"].as_array().expect("corruption cases") {
      let offset = usize::try_from(offset.as_u64().expect("offset")).expect("fixture offset");
      encoding[offset] ^= 1;
      rejected(&encoding, root.as_bytes(), &data);
      encoding[offset] ^= 1;
    }
  }
}

fn rejected(encoding: &[u8], root: &[u8; 32], original: &[u8]) {
  let mut decoder = Decoder::new(encoding, root);
  let mut output = Vec::new();
  let error = decoder
    .read_to_end(&mut output)
    .expect_err("corrupt encoding must fail");
  assert!(matches!(
    error.kind(),
    io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
  ));
  assert!(original.starts_with(&output), "unverified bytes escaped");
  let mut untouched = [0xa5; 33];
  assert_eq!(
    decoder.read(&mut untouched).expect_err("terminal failure").kind(),
    io::ErrorKind::InvalidData
  );
  assert_eq!(untouched, [0xa5; 33]);
}

#[test]
fn every_encoded_byte_and_every_truncation_are_rejected() {
  // Every parent CV byte, header byte, and leaf byte is independently corrupted.
  // Five chunks exercise both a full left tree and the short right edge.
  let data = input(4097);
  let (mut encoding, root) = bao::encode::encode(&data);
  for position in 0..encoding.len() {
    encoding[position] ^= 1;
    rejected(&encoding, root.as_bytes(), &data);
    encoding[position] ^= 1;
    rejected(&encoding[..position], root.as_bytes(), &data);
  }
  let mut wrong_root = *root.as_bytes();
  for byte in 0..32 {
    wrong_root[byte] ^= 1;
    rejected(&encoding, &wrong_root, &data);
    wrong_root[byte] ^= 1;
  }
}

#[test]
fn empty_eof_and_extreme_headers_cannot_bypass_authentication() {
  let (encoding, root) = bao::encode::encode([]);
  verify_encoding(&[], 1);
  let mut wrong_root = *root.as_bytes();
  wrong_root[0] ^= 1;
  rejected(&encoding, &wrong_root, &[]);
  let data = input(3073);
  let (mut encoding, root) = bao::encode::encode(&data);
  for len in [0, 1, 1024, 2048, 3072, 3074, 4096, 1 << 63, u64::MAX] {
    encoding[..8].copy_from_slice(&u64::to_le_bytes(len));
    rejected(&encoding, root.as_bytes(), &data);
  }
}

#[test]
fn maximum_depth_verifies_a_prefix_without_allocating_for_the_header() {
  use blake3::hazmat::{self, HasherExt as _};
  let leaf = [0x3a; 1024];
  let mut hasher = blake3::Hasher::new();
  hasher.update(&leaf);
  let mut left = hasher.finalize_non_root();
  let right = [0xa7; 32];
  let mut parents = Vec::new();
  // Build only the authenticated left-edge proof. Its missing right subtrees
  // must cause an error after the first verified chunk, never an early EOF.
  for _ in 0..53 {
    parents.push((left, right));
    left = hazmat::merge_subtrees_non_root(&left, &right, hazmat::Mode::Hash);
  }
  let root = hazmat::merge_subtrees_root(&left, &right, hazmat::Mode::Hash);
  let mut encoded = u64::MAX.to_le_bytes().to_vec();
  encoded.extend_from_slice(&left);
  encoded.extend_from_slice(&right);
  for (left, right) in parents.iter().rev() {
    encoded.extend_from_slice(left);
    encoded.extend_from_slice(right);
  }
  encoded.extend_from_slice(&leaf);
  let mut decoder = Decoder::new(encoded.as_slice(), root.as_bytes());
  let mut output = [0u8; 1024];
  assert_eq!(decoder.read(&mut output).expect("maximum-depth verified prefix"), 1024);
  assert_eq!(output, leaf);
  output.fill(0x71);
  assert_eq!(
    decoder.read(&mut output).expect_err("missing right subtree").kind(),
    io::ErrorKind::UnexpectedEof
  );
  assert_eq!(output, [0x71; 1024]);
}

struct ShortReader<'a> {
  input: &'a [u8],
  position: usize,
  interruption: bool,
  fail_at: Option<usize>,
  panic: bool,
}

impl Read for ShortReader<'_> {
  #[expect(
    clippy::panic,
    clippy::panic_in_result_fn,
    reason = "inject a reader panic to test the decoder's unwind state"
  )]
  fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
    if self.interruption {
      self.interruption = false;
      return Err(io::ErrorKind::Interrupted.into());
    }
    if self.fail_at == Some(self.position) {
      if self.panic {
        panic!("injected reader panic");
      }
      return Err(io::Error::from_raw_os_error(5));
    }
    let limit = self.fail_at.unwrap_or(self.input.len()).min(self.input.len());
    let count = out.len().min(7).min(limit.saturating_sub(self.position));
    out[..count].copy_from_slice(&self.input[self.position..self.position.strict_add(count)]);
    self.position = self.position.strict_add(count);
    self.interruption = true;
    Ok(count)
  }
}

#[test]
fn short_reads_interruptions_and_terminal_io_failures() {
  let data = input(4097);
  let (encoding, root) = bao::encode::encode(&data);
  let source = |fail_at, panic| ShortReader {
    input: &encoding,
    position: 0,
    interruption: true,
    fail_at,
    panic,
  };
  let mut decoder = Decoder::new(source(None, false), root.as_bytes());
  let mut output = Vec::new();
  decoder.read_to_end(&mut output).expect("short reads are valid");
  assert_eq!(output, data);
  for position in [0, 3, 8, 27, 71, 72, 201, 1500, encoding.len() - 1] {
    let mut decoder = Decoder::new(source(Some(position), false), root.as_bytes());
    let mut verified = Vec::new();
    let error = decoder.read_to_end(&mut verified).expect_err("injected I/O error");
    assert_eq!(error.raw_os_error(), Some(5), "preserve original I/O error");
    assert!(data.starts_with(&verified));
    let mut untouched = [0xa5; 19];
    assert_eq!(
      decoder
        .read(&mut untouched)
        .expect_err("cannot resume consumed node")
        .kind(),
      io::ErrorKind::InvalidData
    );
    assert_eq!(untouched, [0xa5; 19]);
  }
}

#[test]
fn caught_reader_panic_cannot_skip_verification() {
  let data = input(2049);
  let (encoding, root) = bao::encode::encode(&data);
  let reader = ShortReader {
    input: &encoding,
    position: 0,
    interruption: false,
    fail_at: Some(120),
    panic: true,
  };
  let mut decoder = Decoder::new(reader, root.as_bytes());
  let mut output = [0x59; 1024];
  let panic = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| decoder.read(&mut output)));
  let _panic = panic.expect_err("reader must panic at the injected offset");
  assert_eq!(output, [0x59; 1024]);
  assert_eq!(
    decoder
      .read(&mut output)
      .expect_err("panic leaves terminal failure")
      .kind(),
    io::ErrorKind::InvalidData
  );
}

#[test]
fn trailing_bytes_remain_unread_and_debug_hides_untrusted_state() {
  let data = input(1025);
  let (mut encoding, root) = bao::encode::encode(&data);
  encoding.extend_from_slice(b"next message");
  let mut decoder = Decoder::new(encoding.as_slice(), root.as_bytes());
  assert_eq!(format!("{decoder:?}"), "Decoder { .. }");
  assert_eq!(decoder.read(&mut []).expect("empty reads do no I/O"), 0);
  let mut output = Vec::new();
  decoder.read_to_end(&mut output).expect("valid encoding");
  assert_eq!(output, data);
  assert_eq!(decoder.into_inner(), b"next message");
}
