//! Basic checksum usage: one-shot and streaming APIs.
//!
//! Run with: `cargo run --example basic -p checksum`

use checksum::{Checksum, Crc16Ccitt, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};

fn main() {
  println!("=== Checksum Basic Examples ===\n");

  one_shot_examples();
  streaming_examples();
  resume_example();
}

/// One-shot computation: fastest when you have all data in memory.
fn one_shot_examples() {
  println!("--- One-Shot Computation ---\n");

  let data = b"123456789";

  // CRC-32 (IEEE) - Ethernet, gzip, zip, PNG
  let crc32 = Crc32::checksum(data);
  println!("CRC-32 (IEEE):   0x{crc32:08X}");
  assert_eq!(crc32, 0xCBF4_3926);

  // CRC-32C (Castagnoli) - iSCSI, SCTP, ext4, Btrfs
  let crc32c = Crc32C::checksum(data);
  println!("CRC-32C:         0x{crc32c:08X}");
  assert_eq!(crc32c, 0xE306_9283);

  // CRC-64 (XZ/ECMA) - XZ Utils, 7-Zip
  let crc64 = Crc64::checksum(data);
  println!("CRC-64 (XZ):     0x{crc64:016X}");
  assert_eq!(crc64, 0x995D_C9BB_DF19_39FA);

  // CRC-64 (NVMe) - NVMe specification
  let crc64_nvme = Crc64Nvme::checksum(data);
  println!("CRC-64 (NVMe):   0x{crc64_nvme:016X}");
  assert_eq!(crc64_nvme, 0xAE8B_1486_0A79_9888);

  // CRC-24 (OpenPGP) - RFC 4880
  let crc24 = Crc24OpenPgp::checksum(data);
  println!("CRC-24 (OpenPGP): 0x{crc24:06X}");
  assert_eq!(crc24, 0x21_CF02);

  // CRC-16 (CCITT) - X.25, HDLC
  let crc16 = Crc16Ccitt::checksum(data);
  println!("CRC-16 (CCITT):  0x{crc16:04X}");
  assert_eq!(crc16, 0x906E);

  println!();
}

/// Streaming computation: process data in chunks.
fn streaming_examples() {
  println!("--- Streaming Computation ---\n");

  let data = b"123456789";

  // Process in chunks - result matches one-shot
  let mut hasher = Crc32::new();
  hasher.update(b"1234");
  hasher.update(b"56789");
  let crc = hasher.finalize();

  println!("Streaming CRC-32: 0x{crc:08X}");
  assert_eq!(crc, Crc32::checksum(data));

  // finalize() is non-consuming: can continue after
  hasher.update(b"...");
  let extended = hasher.finalize();
  println!("Extended CRC-32:  0x{extended:08X}");

  // reset() clears state for reuse
  hasher.reset();
  hasher.update(b"new data");
  let new_crc = hasher.finalize();
  println!("Reset CRC-32:     0x{new_crc:08X}");

  // Works the same for all CRC types
  let mut h64 = Crc64::new();
  h64.update(b"streaming ");
  h64.update(b"crc64");
  println!("Streaming CRC-64: 0x{:016X}", h64.finalize());

  println!();
}

/// Resume computation from a saved checksum state.
fn resume_example() {
  println!("--- Resume from Saved State ---\n");

  let part1 = b"first part of data";
  let part2 = b" and the second part";

  // Compute partial CRC and save it
  let mut hasher = Crc32::new();
  hasher.update(part1);
  let saved_state = hasher.finalize();
  println!("Saved state after part1: 0x{saved_state:08X}");

  // Later, resume from saved state
  let mut resumed = Crc32::resume(saved_state);
  resumed.update(part2);
  let final_crc = resumed.finalize();
  println!("Final CRC after resume:  0x{final_crc:08X}");

  // Verify: should match processing all at once
  let mut full = Crc32::new();
  full.update(part1);
  full.update(part2);
  assert_eq!(final_crc, full.finalize());
  println!("Verified: matches full computation");

  println!();
}
