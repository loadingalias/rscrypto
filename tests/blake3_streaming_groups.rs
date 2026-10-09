#![cfg(feature = "blake3")]

use rscrypto::{
  hashes::crypto::Blake3,
  traits::{Digest as _, Xof as _},
};

fn pattern(len: usize) -> Vec<u8> {
  (0..len).map(|i| u8::try_from(i % 251).expect("pattern byte")).collect()
}

/// A saved partial chunk must keep its CV, counter and final block when joined
/// to a group of complete chunks, including an exactly terminal group.
#[test]
fn every_partial_chunk_continues_through_four_chunk_groups() {
  let storage = pattern(8 * 1024 + 32);
  for offset in [0usize, 1, 15] {
    let data = &storage[offset..];
    for prefix in 1usize..1024 {
      for len in [4095usize, 4096, 4097, 5 * 1024 + 1, 6 * 1024 + 64, 7 * 1024 + 70] {
        let mut ours = Blake3::new();
        ours.update(&data[..prefix]);
        ours.update(&data[prefix..len]);
        assert_eq!(
          ours.finalize(),
          *blake3::hash(&data[..len]).as_bytes(),
          "offset={offset} prefix={prefix} len={len}"
        );
        assert_eq!(ours.finalize(), Blake3::digest_const(&data[..len]));

        // Finalization does not consume the state. An empty update and a new
        // tail must preserve the pending terminal CV and canonical tree.
        ours.update(&[]);
        ours.update(&data[len..len + 17]);
        let mut actual = [0; 131];
        ours.finalize_xof().squeeze(&mut actual);
        let mut expected = [0; 131];
        blake3::Hasher::new()
          .update(&data[..len + 17])
          .finalize_xof()
          .fill(&mut expected);
        assert_eq!(actual, expected, "continued offset={offset} prefix={prefix} len={len}");
      }
    }
  }
}

#[test]
fn partial_groups_preserve_nonzero_tree_positions_in_all_modes() {
  const KEY: [u8; 32] = [0x53; 32];
  const CONTEXT: &str = "rscrypto partial streaming groups";
  let data = pattern(80 * 1024 + 1024);
  for chunks in [1usize, 3, 4, 7, 8, 15] {
    for prefix in [1usize, 24, 64, 70, 1023] {
      for bulk in [3104usize, 4096, 65536] {
        let split = chunks * 1024 + prefix;
        let end = split + bulk;
        let ours = [Blake3::new(), Blake3::new_keyed(&KEY), Blake3::new_derive_key(CONTEXT)];
        let references = [
          blake3::Hasher::new(),
          blake3::Hasher::new_keyed(&KEY),
          blake3::Hasher::new_derive_key(CONTEXT),
        ];
        for (mut ours, mut reference) in ours.into_iter().zip(references) {
          ours.update(&data[..chunks * 1024]);
          ours.update(&data[chunks * 1024..split]);
          ours.update(&data[split..end]);
          reference.update(&data[..end]);
          let mut actual = [0; 131];
          ours.finalize_xof().squeeze(&mut actual);
          let mut expected = [0; 131];
          reference.finalize_xof().fill(&mut expected);
          assert_eq!(actual, expected, "chunks={chunks} prefix={prefix} bulk={bulk}");
        }
      }
    }
  }
}

/// Rail's granule digest hashes an envelope, then a field block. Complete
/// chunks before a partial envelope chunk share lanes with that chunk's
/// leading blocks; the suffix must resume it at the same tree position.
#[test]
fn bulk_then_suffix_matches_one_update_in_all_modes() {
  const KEY: [u8; 32] = [0x35; 32];
  const CONTEXT: &str = "rscrypto bulk then suffix";
  let storage = pattern(50 * 1024);
  for offset in [0usize, 1] {
    let data = &storage[offset..];
    for chunks in [3usize, 7, 11, 15, 19, 31, 47] {
      for partial in [1usize, 63, 64, 65, 500, 960, 963, 964, 965, 980, 1023] {
        for suffix in [1usize, 60, 1024 + 17] {
          let split = chunks * 1024 + partial;
          let end = split + suffix;
          let ours = [Blake3::new(), Blake3::new_keyed(&KEY), Blake3::new_derive_key(CONTEXT)];
          let references = [
            blake3::Hasher::new(),
            blake3::Hasher::new_keyed(&KEY),
            blake3::Hasher::new_derive_key(CONTEXT),
          ];
          for (mut ours, mut reference) in ours.into_iter().zip(references) {
            ours.update(&data[..split]);
            ours.update(&data[split..end]);
            reference.update(&data[..end]);
            let mut actual = [0; 131];
            ours.finalize_xof().squeeze(&mut actual);
            let mut expected = [0; 131];
            reference.finalize_xof().fill(&mut expected);
            assert_eq!(
              actual, expected,
              "offset={offset} chunks={chunks} partial={partial} suffix={suffix}"
            );
          }
        }
      }
    }
  }
}
