//! Basic `rscrypto` usage across checksums, digests, XOFs, fast hashes, and I/O adapters.
//!
//! Run with: `cargo run --example basic`

use std::io::{Cursor, Read, Write};

use rscrypto::{Blake3, Checksum, Crc32C, Digest, FastHash, RapidHash, Sha256, Shake256, Xof, Xxh3};

fn main() -> std::io::Result<()> {
  println!("=== rscrypto Basic Examples ===\n");

  checksum_api();
  digest_api();
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
