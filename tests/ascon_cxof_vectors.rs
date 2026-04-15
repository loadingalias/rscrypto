#![cfg(feature = "hashes")]

use rscrypto::{AsconCxof128, AsconCxof128Reader, traits::Xof as _};

mod common;
use common::decode_hex_vec as decode_hex;

fn squeeze_all(mut reader: AsconCxof128Reader, len: usize) -> Vec<u8> {
  let mut out = vec![0u8; len];
  reader.squeeze(&mut out);
  out
}

#[test]
fn ascon_cxof128_matches_ascon_c_kat_vector() {
  let msg = decode_hex("");
  let customization = decode_hex("");
  let expected = decode_hex(
    "4F50159EF70BB3DAD8807E034EAEBD44C4FA2CBBC8CF1F05511AB66CDCC529905CA12083FC186AD899B270B1473DC5F7EC88D1052082DCDFE69FB75D269E7B74",
  );

  let actual = squeeze_all(AsconCxof128::xof(&customization, &msg).unwrap(), expected.len());
  assert_eq!(actual, expected);
}

#[test]
fn ascon_cxof128_reset_restores_customized_state() {
  let customization = b"ctx=v1";
  let data = b"abc";

  let mut hasher = AsconCxof128::new(customization).unwrap();
  hasher.update(data);
  let expected = squeeze_all(hasher.finalize_xof(), 64);

  hasher.reset();
  hasher.update(data);
  let actual = squeeze_all(hasher.finalize_xof(), 64);
  assert_eq!(actual, expected);
}

#[test]
fn ascon_cxof128_rejects_long_customization() {
  let err = AsconCxof128::new(&[0u8; 257]).unwrap_err();
  assert_eq!(err.to_string(), "Ascon-CXOF128 customization exceeds 256 bytes");
}
