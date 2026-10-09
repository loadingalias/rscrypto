#![cfg(feature = "blake3")]

mod support;

use rscrypto::{Blake3, Blake3DeriveContext, Blake3KeyedHash};
use support::vector_blob::BlobIterator;

const KEY: [u8; 32] = *b"whats the Elvish word for friend";
const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

/// Anchor the batch APIs to the repository's published BLAKE3 vector corpus.
#[test]
fn batch_official_vectors() {
  let rows: Vec<_> = BlobIterator::<6>::new(include_bytes!("../testdata/blake3/test_vectors.blb"))
    .expect("BLAKE3 vector corpus must parse")
    .map(|row| row.expect("BLAKE3 vector row must decode"))
    .collect();
  let storage: Vec<Vec<u8>> = rows
    .iter()
    .map(|row| {
      assert_eq!(row[0], KEY);
      assert_eq!(row[1], CONTEXT.as_bytes());
      let length = u64::from_le_bytes(row[2].try_into().expect("input length is u64"));
      (0..usize::try_from(length).expect("input length fits in usize"))
        .map(|index| u8::try_from(index % 251).expect("fits in u8"))
        .collect()
    })
    .collect();
  let inputs: Vec<_> = storage.iter().map(Vec::as_slice).collect();
  let mut plain = vec![[0; 32]; rows.len()];
  let mut keyed = vec![Blake3KeyedHash::default(); rows.len()];
  let mut derived = vec![[0; 32]; rows.len()];
  Blake3::digest_batch(&inputs, &mut plain);
  Blake3::keyed_digest_batch(&KEY, &inputs, &mut keyed);
  Blake3DeriveContext::new_const(CONTEXT).derive_key_batch(&inputs, &mut derived);
  for (index, row) in rows.iter().enumerate() {
    assert_eq!(plain[index], row[3][..32], "hash vector {index}");
    assert_eq!(keyed[index].as_bytes(), &row[4][..32], "keyed vector {index}");
    assert_eq!(derived[index], row[5][..32], "derive vector {index}");
  }
}

fn check_batch(inputs: &[&[u8]]) {
  let mut plain = vec![[0xa5; 32]; inputs.len()];
  let mut keyed = vec![Blake3KeyedHash::from_bytes([0xa5; 32]); inputs.len()];
  let mut derived = vec![[0xa5; 32]; inputs.len()];
  Blake3::digest_batch(inputs, &mut plain);
  Blake3::keyed_digest_batch(&KEY, inputs, &mut keyed);
  Blake3DeriveContext::new(CONTEXT).derive_key_batch(inputs, &mut derived);
  for (index, input) in inputs.iter().enumerate() {
    assert_eq!(plain[index], *blake3::hash(input).as_bytes(), "hash index={index}");
    assert_eq!(
      keyed[index].as_bytes(),
      blake3::keyed_hash(&KEY, input).as_bytes(),
      "keyed index={index}"
    );
    assert_eq!(
      derived[index],
      blake3::derive_key(CONTEXT, input),
      "derive index={index}"
    );
  }
}

/// Every final-block size, both sides of the one-chunk/tree boundary, and
/// mixed input lengths are checked against the independent locked upstream.
#[test]
fn batch_all_modes_every_two_chunk_length() {
  let data: Vec<u8> = (0..2 * 1024 + 64)
    .map(|index| u8::try_from(index % 251).expect("fits in u8"))
    .collect();
  let mut inputs: Vec<&[u8]> = (0usize..=2 * 1024)
    .map(|len| {
      let offset = (len % 31).strict_add(1);
      &data[offset..offset.strict_add(len)]
    })
    .collect();
  check_batch(&inputs);
  inputs.reverse();
  check_batch(&inputs);
}

/// Deterministic random mixes cover partial groups and long
/// tree inputs interspersed with empty and partial final blocks.
#[test]
fn batch_random_length_mixes_and_lane_boundaries() {
  const BOUNDARIES: [usize; 15] = [0, 1, 31, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2047, 2048, 8193];
  let mut random = 0x9e37_79b9_7f4a_7c15u64;
  let mut next = || {
    random ^= random << 13;
    random ^= random >> 7;
    random ^= random << 17;
    random
  };
  for count in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129] {
    let storage: Vec<Vec<u8>> = (0..count)
      .map(|index| {
        let len = if index % 3 == 0 {
          BOUNDARIES[usize::try_from(next() % 15).expect("fits in usize")]
        } else {
          usize::try_from(next() % 4097).expect("fits in usize")
        };
        (0..len.strict_add(1)).map(|_| next().to_le_bytes()[0]).collect()
      })
      .collect();
    let inputs: Vec<&[u8]> = storage.iter().map(|bytes| &bytes[1..]).collect();
    check_batch(&inputs);
  }
}

/// Mismatched output counts are rejected before modifying caller storage.
#[test]
fn batch_output_mismatch_is_atomic() {
  let mut keyed = [Blake3KeyedHash::from_bytes([0xa5; 32]); 1];
  let error = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
    Blake3::keyed_digest_batch(&KEY, &[b"first", b"second"], &mut keyed);
  }));
  drop(error.expect_err("keyed batch rejects an output-count mismatch"));
  assert_eq!(keyed[0].as_bytes(), &[0xa5; 32]);

  let mut derived = [[0xa5; 32]; 2];
  let context = Blake3DeriveContext::new(CONTEXT);
  let error = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
    context.derive_key_batch(&[b"first"], &mut derived);
  }));
  drop(error.expect_err("derive batch rejects an output-count mismatch"));
  assert_eq!(derived, [[0xa5; 32]; 2]);
}
