//! Reach the exported QUIC-LB CT ABI with the google/quiche and hopf-quic connection IDs of
//! `testdata/quic_lb`, which no rscrypto code computed.
use rscrypto_ct_harness::{ct_entry_quic_lb_decrypt, ct_entry_quic_lb_encrypt};

fn decode(hex: &str) -> Vec<u8> {
  let (pairs, rest) = hex.as_bytes().as_chunks::<2>();
  assert!(rest.is_empty(), "whole octets: {hex}");
  pairs
    .iter()
    .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).expect("ASCII"), 16).expect("hexadecimal octet"))
    .collect()
}

/// `prefix`, then 0xa5 up to 19 octets.
fn block(prefix: &[u8]) -> [u8; 19] {
  let mut block = [0xa5; 19];
  block[..prefix.len()].copy_from_slice(prefix);
  block
}

/// Each line is "rotation key server_id nonce connection_id"; the first connection-ID octet is
/// not encrypted. Every length pair passes through both entries, so both cipher forms do.
#[test]
fn exported_entries_match_independent_connection_ids() {
  let files = [
    include_str!("../../../testdata/quic_lb/quiche-vectors.txt"),
    include_str!("../../../testdata/quic_lb/hopf-vectors.txt"),
  ];
  for line in files.iter().flat_map(|file| file.lines()) {
    let fields: Vec<_> = line.split(' ').skip(1).map(decode).collect();
    let [key, server_id, nonce, id] = <[Vec<u8>; 4]>::try_from(fields).expect("five fields");
    let mut buffer = block(&[server_id.as_slice(), nonce.as_slice()].concat());
    // SAFETY: key holds 16 octets and buffer 19 writable octets, both live and distinct.
    let status = unsafe { ct_entry_quic_lb_encrypt(key.as_ptr(), server_id.len(), nonce.len(), buffer.as_mut_ptr()) };
    assert_eq!((status, buffer), (1, block(&id[1..])), "encrypt {line}");

    let mut expected = [0; 19];
    expected[..server_id.len()].copy_from_slice(&server_id);
    // SAFETY: As above.
    let status = unsafe { ct_entry_quic_lb_decrypt(key.as_ptr(), server_id.len(), nonce.len(), buffer.as_mut_ptr()) };
    assert_eq!((status, buffer), (1, expected), "decrypt {line}");
  }
}
