//! Reach the exported CT ABI with the independent RFC 8439 section 2.5.2 vector.
use rscrypto_ct_harness::ct_entry_poly1305_authenticate;

const KEY: [u8; 32] = [
  0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33, 0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5, 0x06, 0xa8, 0x01, 0x03, 0x80,
  0x8a, 0xfb, 0x0d, 0xb2, 0xfd, 0x4a, 0xbf, 0xf6, 0xaf, 0x41, 0x49, 0xf5, 0x1b,
];
const MESSAGE: &[u8] = b"Cryptographic Forum Research Group";

#[test]
fn exported_authenticator_matches_rfc8439_and_output_bounds() {
  let mut output = [0xa5; 18];
  // SAFETY: KEY and MESSAGE remain readable; the inner output range has 16 distinct writable bytes.
  let status =
    unsafe { ct_entry_poly1305_authenticate(KEY.as_ptr(), MESSAGE.as_ptr(), MESSAGE.len(), output[1..].as_mut_ptr()) };
  assert_eq!(status, 1);
  assert_eq!(
    &output[1..17],
    &[
      0xa8, 0x06, 0x1d, 0xc1, 0x30, 0x51, 0x36, 0xc6, 0xc2, 0x2b, 0x8b, 0xaf, 0x0c, 0x01, 0x27, 0xa9
    ]
  );
  assert_eq!(output[0], 0xa5);
  assert_eq!(output[17], 0xa5);
}

#[test]
fn exported_authenticator_rejects_null_shapes_and_accepts_empty_message() {
  let mut output = [0xa5; 16];
  // SAFETY: Null key is explicitly rejected; the other pointers expose their required ranges.
  let missing_key =
    unsafe { ct_entry_poly1305_authenticate(core::ptr::null(), MESSAGE.as_ptr(), MESSAGE.len(), output.as_mut_ptr()) };
  assert_eq!(missing_key, 0);
  assert_eq!(output, [0xa5; 16]);
  // SAFETY: Null nonempty data is explicitly rejected; key/output expose their required ranges.
  let missing_data = unsafe { ct_entry_poly1305_authenticate(KEY.as_ptr(), core::ptr::null(), 1, output.as_mut_ptr()) };
  assert_eq!(missing_data, 0);
  assert_eq!(output, [0xa5; 16]);
  // SAFETY: A null output is explicitly rejected; key and message are valid immutable arrays.
  let missing_output =
    unsafe { ct_entry_poly1305_authenticate(KEY.as_ptr(), MESSAGE.as_ptr(), MESSAGE.len(), core::ptr::null_mut()) };
  assert_eq!(missing_output, 0);
  // SAFETY: Null data with zero length denotes an empty message; key/output have valid ranges.
  let empty = unsafe { ct_entry_poly1305_authenticate(KEY.as_ptr(), core::ptr::null(), 0, output.as_mut_ptr()) };
  assert_eq!(empty, 1);
  // Poly1305 of an empty message is the additive pad, the last 16 key bytes.
  assert_eq!(output, KEY[16..]);
}
