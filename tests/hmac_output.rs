//! Caller-owned HMAC output retains the non-consuming finalize contract.

#![cfg(feature = "hmac")]

use rscrypto::Mac as _;

macro_rules! output_contract {
  ($name:ident, $mac:ty, $hash:ty, $size:expr) => {
    #[test]
    fn $name() {
      let key = [0xaa; 131];
      let message = [0x17; 129];
      let expected = |parts: &[&[u8]]| {
        let mut oracle = <hmac::Hmac<$hash> as hmac::digest::KeyInit>::new_from_slice(&key).unwrap();
        for part in parts {
          hmac::Mac::update(&mut oracle, part);
        }
        hmac::Mac::finalize(oracle).into_bytes()
      };
      let mut state = <$mac>::new(&key);
      state.update(&message[..63]);
      state.update(&message[63..]);
      let mut output = [0xa7; $size];
      state.finalize_into(&mut output);
      assert_eq!(output.as_slice(), expected(&[&message]).as_slice());
      output.fill(0x93);
      state.finalize_into(&mut output);
      assert_eq!(output.as_slice(), expected(&[&message]).as_slice());

      state.update(b"suffix");
      state.finalize_into(&mut output);
      assert_eq!(
        output.as_slice(),
        expected(&[message.as_slice(), b"suffix".as_slice()]).as_slice()
      );

      state.reset();
      state.update(b"fresh");
      state.finalize_into(&mut output);
      assert_eq!(output.as_slice(), expected(&[b"fresh"]).as_slice());
    }
  };
}

output_contract!(sha256_output_preserves_state, rscrypto::HmacSha256, sha2::Sha256, 32);
output_contract!(sha384_output_preserves_state, rscrypto::HmacSha384, sha2::Sha384, 48);
