//! Reach the exported CT ABI with independent SipHash answers: the SipHash-2-4
//! reference vectors, and SipHash-1-3 table hashes computed by `siphasher`
//! 1.0.4 over the native-endian length prefix and the message.
use rscrypto_ct_harness::ct_entry_siphash_hash;

/// The reference key `00..0f`.
const KEY: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// The reference message `00..0e`.
const MESSAGE: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
/// SipHash-2-4 of the reference message, `vectors_sip64[15]`.
const ONE_SHOT_15: u64 = 0xa129_ca61_49be_45e5;
/// SipHash-2-4 of the empty message, `vectors_sip64[0]`.
const ONE_SHOT_0: u64 = 0x726f_db47_dd0e_0e31;
/// SipHash-1-3 of the 8-byte length prefix 15 and the reference message.
const TABLE_15: u64 = if cfg!(target_endian = "little") {
  0xfeec_6d89_1039_afae
} else {
  0xa885_0441_80fe_1b6e
};
/// SipHash-1-3 of the 8-byte length prefix 0.
const TABLE_0: u64 = 0x5cb9_6f6b_a2a4_fcfc;

fn expected(table: u64, one_shot: u64) -> [u8; 16] {
  let mut out = [0; 16];
  out[..8].copy_from_slice(&table.to_le_bytes());
  out[8..].copy_from_slice(&one_shot.to_le_bytes());
  out
}

#[test]
fn exported_hash_matches_independent_answers_and_output_bounds() {
  let mut output = [0xa5; 18];
  // SAFETY: KEY and MESSAGE remain readable; the inner output range has 16 distinct writable bytes.
  let status =
    unsafe { ct_entry_siphash_hash(KEY.as_ptr(), MESSAGE.as_ptr(), MESSAGE.len(), output[1..].as_mut_ptr()) };
  assert_eq!(status, 1);
  assert_eq!(output[1..17], expected(TABLE_15, ONE_SHOT_15));
  assert_eq!(output[0], 0xa5);
  assert_eq!(output[17], 0xa5);
}

#[test]
fn exported_hash_rejects_null_shapes_and_accepts_empty_message() {
  let mut output = [0xa5; 16];
  // SAFETY: Null key is explicitly rejected; the other pointers expose their required ranges.
  let missing_key =
    unsafe { ct_entry_siphash_hash(core::ptr::null(), MESSAGE.as_ptr(), MESSAGE.len(), output.as_mut_ptr()) };
  assert_eq!(missing_key, 0);
  assert_eq!(output, [0xa5; 16]);
  // SAFETY: Null nonempty data is explicitly rejected; key/output expose their required ranges.
  let missing_data = unsafe { ct_entry_siphash_hash(KEY.as_ptr(), core::ptr::null(), 1, output.as_mut_ptr()) };
  assert_eq!(missing_data, 0);
  assert_eq!(output, [0xa5; 16]);
  // SAFETY: A null output is explicitly rejected; key and message are valid immutable arrays.
  let missing_output =
    unsafe { ct_entry_siphash_hash(KEY.as_ptr(), MESSAGE.as_ptr(), MESSAGE.len(), core::ptr::null_mut()) };
  assert_eq!(missing_output, 0);
  // SAFETY: Null data with zero length denotes an empty message; key/output have valid ranges.
  let empty = unsafe { ct_entry_siphash_hash(KEY.as_ptr(), core::ptr::null(), 0, output.as_mut_ptr()) };
  assert_eq!(empty, 1);
  assert_eq!(output, expected(TABLE_0, ONE_SHOT_0));
}
