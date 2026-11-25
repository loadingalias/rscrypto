#![cfg(any(
  all(feature = "xxh3", feature = "rapidhash"),
  feature = "aes-siv",
  all(feature = "blake3", feature = "std")
))]

#[cfg(all(feature = "xxh3", feature = "rapidhash"))]
use core::hash::{BuildHasher, Hasher};
#[cfg(all(feature = "xxh3", feature = "rapidhash"))]
use std::collections::HashMap;

#[cfg(all(feature = "xxh3", feature = "rapidhash"))]
use rscrypto::{RapidSeededState, RapidStreamHasher, Xxh3_128Hasher, Xxh3BuildHasher};

#[cfg(all(feature = "blake3", feature = "std"))]
#[path = "common/file.rs"]
mod file;

#[path = "common/allocations.rs"]
mod allocations;
use allocations::measure_allocations;

#[cfg(all(feature = "blake3", feature = "std"))]
#[test]
fn small_blake3_file_inputs_keep_buffering_on_the_stack() {
  use rscrypto::{Blake3, Digest};

  let input: Vec<u8> = (0..65_536)
    .map(|i| u8::try_from(i % 251).expect("byte pattern"))
    .collect();
  // Capability detection and fixture allocation are outside the measured operation.
  core::hint::black_box(Blake3::digest(b"warm dispatch"));
  for len in [0, 1, 1024, 4096, 65_535, 65_536] {
    let fixture = file::TemporaryFile::new(&input[..len]);
    for mode in 0..3 {
      let mut file = fixture.open();
      let mut hasher = match mode {
        0 => Blake3::new(),
        1 => Blake3::new_keyed(&[0xa7; 32]),
        _ => Blake3::new_derive_key("rscrypto allocation fixture"),
      };
      let mut reference = match mode {
        0 => blake3::Hasher::new(),
        1 => blake3::Hasher::new_keyed(&[0xa7; 32]),
        _ => blake3::Hasher::new_derive_key("rscrypto allocation fixture"),
      };
      reference.update(&input[..len]);
      let mut count = 0;
      let allocations = measure_allocations(|| {
        count = hasher.update_file(&mut file).expect("read small file");
      });
      assert_eq!(count, len as u64);
      assert_eq!(allocations, 0, "mode={mode} len={len}");
      assert_eq!(hasher.finalize(), *reference.finalize().as_bytes());
    }
  }
}

#[test]
#[cfg(all(feature = "xxh3", feature = "rapidhash"))]
fn fast_hashers_and_preallocated_maps_hash_without_allocating() {
  let xxh3_builder = Xxh3BuildHasher::with_seed(42);
  let rapid_builder = RapidSeededState::new(42);
  let mut xxh3_map = HashMap::with_capacity_and_hasher(8, xxh3_builder);
  let mut rapid_map = HashMap::with_capacity_and_hasher(8, rapid_builder);
  let long_input = [0x5au8; 4096];

  let xxh3_direct = measure_allocations(|| {
    let mut xxh3 = xxh3_builder.build_hasher();
    for chunk in long_input.chunks(127) {
      xxh3.write(chunk);
    }
    let hash = xxh3.finish();
    assert_eq!(xxh3.finish(), hash);
  });

  let xxh3_128_direct = measure_allocations(|| {
    let mut xxh3_128 = Xxh3_128Hasher::with_seed(42);
    for chunk in long_input.chunks(127) {
      xxh3_128.write(chunk);
    }
    let hash = xxh3_128.finish();
    assert_eq!(xxh3_128.finish(), hash);
  });

  let rapid_direct = measure_allocations(|| {
    let mut rapid = rapid_builder.build_hasher();
    rapid.write(&long_input);
    let hash = rapid.finish();
    assert_eq!(rapid.finish(), hash);
  });

  let rapid_stream_direct = measure_allocations(|| {
    let mut rapid_stream = RapidStreamHasher::new();
    for chunk in long_input.chunks(127) {
      rapid_stream.write(chunk);
    }
    let hash = rapid_stream.finish();
    assert_eq!(rapid_stream.finish(), hash);
  });

  let xxh3_map_ops = measure_allocations(|| {
    xxh3_map.insert("allocation-free", 1);
    assert_eq!(xxh3_map.get("allocation-free"), Some(&1));
  });

  let rapid_map_ops = measure_allocations(|| {
    rapid_map.insert("allocation-free", 1);
    assert_eq!(rapid_map.get("allocation-free"), Some(&1));
  });

  assert_eq!(xxh3_direct, 0, "XXH3 Hasher must not allocate");
  assert_eq!(xxh3_128_direct, 0, "XXH3-128 Hasher must not allocate");
  assert_eq!(rapid_direct, 0, "RapidHash Hasher must not allocate");
  assert_eq!(rapid_stream_direct, 0, "RapidHash streaming Hasher must not allocate");
  assert_eq!(
    xxh3_map_ops, 0,
    "preallocated XXH3 HashMap operations must not allocate"
  );
  assert_eq!(
    rapid_map_ops, 0,
    "preallocated RapidHash HashMap operations must not allocate"
  );
}

#[cfg(all(feature = "aes-gcm", feature = "chacha20poly1305"))]
#[test]
fn header_protection_mask_generation_does_not_allocate() {
  use rscrypto::aead::expert::header_protection::{
    Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
    ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey,
  };

  let aes128 = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes([0x11; 16]));
  let aes256 = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes([0x22; 32]));
  let chacha20 = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes([0x33; 32]));
  let sample = [0x44; 16];

  let allocations = measure_allocations(|| {
    core::hint::black_box(aes128.mask(core::hint::black_box(&sample)));
    core::hint::black_box(aes256.mask(core::hint::black_box(&sample)));
    core::hint::black_box(chacha20.mask(core::hint::black_box(&sample)));
  });

  assert_eq!(allocations, 0, "header-protection mask generation must not allocate");
}

#[cfg(all(feature = "aes-gcm", feature = "chacha20poly1305", not(target_arch = "riscv64")))]
#[test]
fn header_protection_context_construction_does_not_allocate_off_riscv64() {
  use rscrypto::aead::expert::header_protection::{
    Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
    ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey,
  };

  let allocations = measure_allocations(|| {
    let key128 = Aes128HeaderProtectionKey::from_bytes([0x11; 16]);
    core::hint::black_box(Aes128HeaderProtection::new(&key128));
    let key256 = Aes256HeaderProtectionKey::from_bytes([0x22; 32]);
    core::hint::black_box(Aes256HeaderProtection::new(&key256));
    let chacha_key = ChaCha20HeaderProtectionKey::from_bytes([0x33; 32]);
    core::hint::black_box(ChaCha20HeaderProtection::new(&chacha_key));
  });

  assert_eq!(
    allocations, 0,
    "header-protection context construction must not allocate off RISC-V"
  );
}

#[test]
#[cfg(feature = "aes-siv")]
fn aes_siv_in_place_operations_do_not_allocate() {
  use rscrypto::{AesSivCmac256, AesSivCmac256Key, AesSivCmac256Nonce};

  let key = AesSivCmac256Key::from_bytes([0x53; 32]);
  let nonce = AesSivCmac256Nonce::try_from(&b"allocation nonce"[..]).expect("nonce is non-empty");

  #[cfg(not(target_arch = "riscv64"))]
  assert_eq!(
    measure_allocations(|| {
      drop(AesSivCmac256::new(&key));
    }),
    0,
    "AES-SIV context construction must not allocate outside the RISC-V boxed schedule backend"
  );

  let cipher = AesSivCmac256::new(&key);
  let mut message = [0x42; 1232];
  let mut tag = None;
  assert_eq!(
    measure_allocations(|| {
      tag = Some(cipher.seal_in_place(nonce, b"associated data", &mut message));
    }),
    0,
    "AES-SIV in-place sealing must not allocate"
  );
  assert_eq!(
    measure_allocations(|| {
      cipher
        .open_in_place(
          nonce,
          b"associated data",
          &mut message,
          tag.as_ref().expect("seal produced a tag"),
        )
        .expect("fresh ciphertext must authenticate");
    }),
    0,
    "AES-SIV in-place opening must not allocate"
  );
}
