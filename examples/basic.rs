//! Basic `rscrypto` usage across checksums, digests, MACs, KDFs, XOFs, fast hashes, AEAD,
//! hex formatting, and I/O adapters.
//!
//! Run with: `cargo run --example basic`

use std::io::{Cursor, Read, Write};

use rscrypto::{
  Blake3, ChaCha20Poly1305, ChaCha20Poly1305Key, Checksum, Crc32C, Digest, Ed25519Keypair, Ed25519SecretKey, FastHash,
  HkdfSha256, HmacSha256, Mac, RapidHash, Sha256, Shake256, Xof, Xxh3, aead::Nonce96,
};

fn main() -> std::io::Result<()> {
  println!("=== rscrypto Basic Examples ===\n");

  checksum_api();
  digest_api();
  auth_api();
  aead_api();
  hex_api();
  serialization_api();
  xof_api();
  fast_hash_api();
  io_api()?;

  Ok(())
}

/// Checksums follow `new` → `update` → `finalize` → `reset`.
fn checksum_api() {
  println!("--- Checksums ---\n");

  let data = b"hello world";

  let oneshot = Crc32C::checksum(data);
  assert_eq!(oneshot, 0xC994_65AA);

  let mut streaming = Crc32C::new();
  streaming.update(b"hello ");
  streaming.update(b"world");
  assert_eq!(streaming.finalize(), oneshot);
  streaming.reset();
  streaming.update(data);
  assert_eq!(streaming.finalize(), oneshot);

  println!("CRC-32C(\"hello world\") = 0x{oneshot:08X}\n");
}

/// Digests follow `new` → `update` → `finalize` → `reset`.
fn digest_api() {
  println!("--- Digests ---\n");

  let data = b"hello world";

  let sha_oneshot = Sha256::digest(data);
  let mut sha_stream = Sha256::new();
  sha_stream.update(b"hello ");
  sha_stream.update(b"world");
  assert_eq!(sha_stream.finalize(), sha_oneshot);
  sha_stream.reset();
  sha_stream.update(data);
  assert_eq!(sha_stream.finalize(), sha_oneshot);

  let blake3_oneshot = Blake3::digest(data);
  let mut blake3_stream = Blake3::new();
  blake3_stream.update(b"hello ");
  blake3_stream.update(b"world");
  assert_eq!(blake3_stream.finalize(), blake3_oneshot);
  blake3_stream.reset();
  blake3_stream.update(data);
  assert_eq!(blake3_stream.finalize(), blake3_oneshot);

  println!("SHA-256 output size = {} bytes", sha_oneshot.len());
  println!("BLAKE3 output size  = {} bytes\n", blake3_oneshot.len());
}

/// MACs and KDFs keep the same one-shot vs stateful split.
fn auth_api() {
  println!("--- Auth ---\n");

  let key = b"shared-secret";
  let data = b"hello world";

  let tag = HmacSha256::mac(key, data);

  let mut mac = HmacSha256::new(key);
  mac.update(b"hello ");
  mac.update(b"world");
  assert_eq!(mac.finalize(), tag);
  assert!(mac.verify(&tag).is_ok());
  mac.reset();
  mac.update(data);
  assert_eq!(mac.finalize(), tag);

  let hkdf = HkdfSha256::new(b"salt", b"input key material");
  let mut okm = [0u8; 42];
  if let Err(err) = hkdf.expand(b"context", &mut okm) {
    panic!("HKDF expand must succeed: {err}");
  }

  let oneshot = match HkdfSha256::derive_array::<42>(b"salt", b"input key material", b"context") {
    Ok(out) => out,
    Err(err) => panic!("HKDF derive_array must succeed: {err}"),
  };
  assert_eq!(okm, oneshot);

  println!("HMAC-SHA256 tag size = {} bytes", tag.len());
  println!("HKDF-SHA256 output   = {} bytes\n", okm.len());
}

/// AEAD: encrypt, authenticate, and decrypt with typed keys and nonces.
fn aead_api() {
  println!("--- AEAD ---\n");

  let key = ChaCha20Poly1305Key::from_bytes([0x11; 32]);
  let nonce = Nonce96::from_bytes([0x22; 12]);
  let aead = ChaCha20Poly1305::new(&key);

  let mut buf = *b"hello";
  let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);
  assert_ne!(&buf, b"hello");
  aead
    .decrypt_in_place(&nonce, b"", &mut buf, &tag)
    .expect("decrypt must succeed");
  assert_eq!(&buf, b"hello");

  println!("ChaCha20-Poly1305 round-trip OK");
  println!("  tag = {tag}");
  println!("  nonce = {nonce}\n");
}

/// Hex Display, FromStr, and secret key masking.
fn hex_api() {
  println!("--- Hex Formatting ---\n");

  let nonce = Nonce96::from_bytes([0xab; 12]);
  println!("Display:  {nonce}");
  println!("LowerHex: {nonce:x}");
  println!("UpperHex: {nonce:X}");
  println!("Debug:    {nonce:?}");

  let parsed: Nonce96 = "abababababababababababab".parse().unwrap();
  assert_eq!(parsed, nonce);
  println!("FromStr:  round-trip OK");

  let key = ChaCha20Poly1305Key::from_bytes([0x42; 32]);
  println!("\nSecret Debug: {key:?}");
  println!("display_secret(): {}", key.display_secret());

  let ed_sk = Ed25519SecretKey::from_bytes([7u8; 32]);
  let kp = Ed25519Keypair::from_secret_key(ed_sk);
  println!("\nEd25519 public key: {}", kp.public_key());
  let sig = kp.sign(b"hello");
  println!("Ed25519 signature:  {sig}");
  println!();
}

/// Serialization via from_bytes / to_bytes / as_bytes.
fn serialization_api() {
  println!("--- Serialization ---\n");

  let key = ChaCha20Poly1305Key::from_bytes([0x42; 32]);
  let raw: [u8; 32] = key.to_bytes();
  let restored = ChaCha20Poly1305Key::from_bytes(raw);
  assert_eq!(key, restored);
  println!("Key round-trip via to_bytes/from_bytes: OK");

  let nonce = Nonce96::from_bytes([0xab; 12]);
  assert_eq!(nonce, Nonce96::from_bytes(nonce.to_bytes()));
  println!("Nonce round-trip: OK\n");
}

/// XOFs support both `new` → `update` → `finalize_xof` and `xof(data)`.
fn xof_api() {
  println!("--- XOFs ---\n");

  let data = b"hello world";

  let mut shake_stream = Shake256::xof(data);
  let mut shake_stream_out = [0u8; 64];
  shake_stream.squeeze(&mut shake_stream_out);

  let mut shake = Shake256::new();
  shake.update(data);
  let mut shake_oneshot_out = [0u8; 64];
  shake.finalize_xof().squeeze(&mut shake_oneshot_out);
  assert_eq!(shake_stream_out, shake_oneshot_out);

  let mut blake3_xof = Blake3::xof(data);
  let mut blake3_out = [0u8; 64];
  blake3_xof.squeeze(&mut blake3_out);
  assert_eq!(&blake3_out[..32], &Blake3::digest(data));

  println!("SHAKE256 produced {} bytes", shake_stream_out.len());
  println!("BLAKE3 XOF produced {} bytes\n", blake3_out.len());
}

fn fast_hash_api() {
  println!("--- Fast Hashes ---\n");

  let data = b"hello world";

  let xxh_default = Xxh3::hash(data);
  let xxh_seeded = Xxh3::hash_with_seed(7, data);
  assert_ne!(xxh_default, xxh_seeded);

  let rapid_default = RapidHash::hash(data);
  let rapid_seeded = RapidHash::hash_with_seed(7, data);
  assert_ne!(rapid_default, rapid_seeded);

  println!("Xxh3(default)       = 0x{xxh_default:016X}");
  println!("RapidHash(default) = 0x{rapid_default:016X}\n");
}

fn io_api() -> std::io::Result<()> {
  println!("--- I/O Adapters ---\n");

  let data = b"stream me through adapters";

  let mut reader = Sha256::reader(Cursor::new(data.to_vec()));
  let mut copied = Vec::new();
  reader.read_to_end(&mut copied)?;
  assert_eq!(copied, data);
  assert_eq!(reader.digest(), Sha256::digest(data));

  let mut checksum_writer = Crc32C::writer(Vec::new());
  checksum_writer.write_all(data)?;
  let (written, crc) = checksum_writer.into_parts();
  assert_eq!(written, data);
  assert_eq!(crc, Crc32C::checksum(data));

  let mut digest_writer = Blake3::writer(Vec::new());
  digest_writer.write_all(data)?;
  let (written, digest) = digest_writer.into_parts();
  assert_eq!(written, data);
  assert_eq!(digest, Blake3::digest(data));

  println!("reader digest matches Sha256::digest()");
  println!("writer checksum matches Crc32C::checksum()");
  println!("writer digest matches Blake3::digest()\n");

  Ok(())
}
