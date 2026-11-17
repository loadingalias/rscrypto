use aes::cipher::{Array, BlockCipherDecrypt as _, BlockCipherEncrypt as _, KeyInit as _};
use rscrypto::aead::expert::quic_lb::{QuicLbCipher, QuicLbKey};
use rscrypto_fuzz::{FuzzInput, some_or_return};

/// Run the QUIC-LB cipher on arbitrary keys, lengths, and buffers.
///
/// Construction succeeds exactly for the lengths of draft-ietf-quic-load-balancers-22 section
/// 5.3. Encryption leaves the octets after the plaintext alone, decryption recovers the server ID
/// and clears the rest, and the single-pass form is one block of an independent AES-128.
pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key: [u8; 16] = some_or_return!(input.bytes());
  let [server_id_len, nonce_len]: [u8; 2] = some_or_return!(input.bytes());
  let buffer: [u8; QuicLbCipher::MAX_PLAINTEXT_LEN] = some_or_return!(input.bytes());
  let (server_id_len, nonce_len) = (usize::from(server_id_len % 21), usize::from(nonce_len % 21));
  let len = server_id_len + nonce_len;

  let Some(cipher) = QuicLbCipher::new(&QuicLbKey::from_bytes(key), server_id_len, nonce_len) else {
    assert!(
      server_id_len < 1 || nonce_len < 4 || len > 19,
      "refused {server_id_len}+{nonce_len}"
    );
    return;
  };

  let mut encrypted = buffer;
  cipher.encrypt(&mut encrypted);
  assert_eq!(encrypted[len..], buffer[len..], "encryption leaves the tail");

  let mut decrypted = encrypted;
  cipher.decrypt_server_id(&mut decrypted);
  assert_eq!(decrypted[..server_id_len], buffer[..server_id_len], "round trip");
  assert!(
    decrypted[server_id_len..].iter().all(|&byte| byte == 0),
    "decryption clears the rest"
  );

  let mut other_tail = buffer;
  other_tail[len..].iter_mut().for_each(|byte| *byte = !*byte);
  let mut from_buffer = buffer;
  cipher.decrypt_server_id(&mut from_buffer);
  cipher.decrypt_server_id(&mut other_tail);
  assert_eq!(from_buffer, other_tail, "decryption ignores the tail");

  if len == 16 {
    let oracle = aes::Aes128::new(&Array::from(key));
    let mut block = Array::from(<[u8; 16]>::try_from(&buffer[..16]).expect("one block"));
    oracle.encrypt_block(&mut block);
    assert_eq!(encrypted[..16], block[..], "single-pass encryption");
    let mut block = Array::from(<[u8; 16]>::try_from(&buffer[..16]).expect("one block"));
    oracle.decrypt_block(&mut block);
    assert_eq!(
      from_buffer[..server_id_len],
      block[..server_id_len],
      "single-pass decryption"
    );
  }
}
