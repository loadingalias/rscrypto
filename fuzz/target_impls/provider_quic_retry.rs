use rscrypto_rustls::quic::{MAX_CONNECTION_ID_LEN, RETRY_INTEGRITY_TAG_LEN, is_valid_retry, retry_integrity_tag};

/// Check production Retry tags on arbitrary packets and connection-ID lengths.
///
/// A packet is valid exactly when its last 16 bytes are the tag of the rest,
/// and a connection ID longer than 20 bytes never yields a tag.
pub(super) fn run(data: &[u8]) {
  if data.len() > 4096 {
    return;
  }
  let cid = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
  let expected = data
    .len()
    .checked_sub(RETRY_INTEGRITY_TAG_LEN)
    .is_some_and(|split| retry_integrity_tag(&cid, &data[..split]).is_some_and(|tag| tag.as_slice() == &data[split..]));
  assert_eq!(is_valid_retry(&cid, data), expected);
  let tag = retry_integrity_tag(&cid, data).expect("8-byte connection ID");
  let mut packet = [data, &tag].concat();
  assert!(is_valid_retry(&cid, &packet));
  let last = packet.len().strict_sub(1);
  packet[last] ^= 1;
  assert!(!is_valid_retry(&cid, &packet));

  let id_len = data.first().map_or(0, |byte| usize::from(*byte) % 22).min(data.len());
  let (id, rest) = data.split_at(id_len);
  assert_eq!(
    retry_integrity_tag(id, rest).is_some(),
    id.len() <= MAX_CONNECTION_ID_LEN
  );
}
