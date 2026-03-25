#![allow(unused_imports)]

#[cfg(feature = "auth")]
use rscrypto::auth::HkdfOutputLengthError;
#[cfg(all(feature = "checksums", feature = "alloc"))]
use rscrypto::checksum::buffered::BufferedCrc32C;
#[cfg(feature = "checksums")]
use rscrypto::checksum::config::{
  Crc16Config, Crc16Force, Crc24Config, Crc24Force, Crc32Config, Crc32Force, Crc64Config, Crc64Force,
};
#[cfg(feature = "checksums")]
use rscrypto::checksum::introspect::{DispatchInfo, KernelIntrospect, is_hardware_accelerated, kernel_for};
#[cfg(feature = "checksums")]
use rscrypto::checksum::{Crc32Castagnoli, Crc32Ieee, Crc64Xz};
#[cfg(feature = "hashes")]
use rscrypto::hashes::fast::{RapidHash64, Xxh3_64};
#[cfg(feature = "hashes")]
use rscrypto::hashes::introspect::{HashKernelIntrospect, kernel_for as hash_kernel_for};
#[cfg(all(feature = "hashes", feature = "std"))]
use rscrypto::hashes::{DigestReader, DigestWriter};
#[cfg(feature = "hashes")]
use rscrypto::{
  AsconHash256, AsconXof, AsconXofReader, Blake3, Blake3Xof, Digest, FastHash, RapidHash, RapidHash128, Sha3_224,
  Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256, Sha384, Sha512, Sha512_256, Shake128, Shake128Xof, Shake256,
  Shake256Xof, Xof, Xxh3, Xxh3_128,
};
#[cfg(feature = "checksums")]
use rscrypto::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};
#[cfg(feature = "auth")]
use rscrypto::{
  Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, HkdfSha256, HmacSha256, Mac, verify_ed25519,
};
use rscrypto::{VerificationError, ct};

#[test]
fn root_surface_core_exports_compile() {
  let _ = VerificationError::new();
  assert!(ct::constant_time_eq(b"ok", b"ok"));
}

#[test]
#[cfg(feature = "auth")]
fn root_surface_auth_exports_compile() {
  let key = b"root-surface-key";
  let data = b"root-surface-data";

  let tag = HmacSha256::mac(key, data);

  let mut mac = HmacSha256::new(key);
  mac.update(data);
  assert_eq!(tag, mac.finalize());
  assert!(mac.verify(&tag).is_ok());

  let mut out = [0u8; 32];
  let hkdf = HkdfSha256::new(b"salt", key);
  hkdf.expand(b"info", &mut out).unwrap();
  assert_eq!(out, HkdfSha256::derive_array::<32>(b"salt", key, b"info").unwrap());
  let _ = HkdfOutputLengthError::new();
}

#[test]
#[cfg(feature = "auth")]
fn advanced_auth_modules_compile() {
  let secret = Ed25519SecretKey::from_bytes([7u8; Ed25519SecretKey::LENGTH]);
  let keypair = Ed25519Keypair::from_secret_key(secret.clone());
  let public = keypair.public_key();
  let signature = keypair.sign(b"root-surface-ed25519");

  assert_eq!(secret.as_bytes().len(), 32);
  assert_eq!(public.as_bytes().len(), 32);
  assert_eq!(signature.as_bytes().len(), 64);
  assert!(public.verify(b"root-surface-ed25519", &signature).is_ok());
  assert!(verify_ed25519(b"root-surface-ed25519", &public, &signature).is_ok());
}

#[test]
#[cfg(feature = "checksums")]
fn root_surface_checksum_exports_compile() {
  let data = b"root-surface";
  let (left, right) = data.split_at(4);

  let oneshot = Crc32C::checksum(data);

  let mut streaming = Crc32C::new();
  streaming.update(left);
  streaming.update(right);

  assert_eq!(oneshot, streaming.finalize());
  assert_eq!(
    oneshot,
    Crc32C::combine(Crc32C::checksum(left), Crc32C::checksum(right), right.len())
  );

  assert_eq!(Crc32Ieee::checksum(data), Crc32::checksum(data));
  assert_eq!(Crc32Castagnoli::checksum(data), Crc32C::checksum(data));
  assert_eq!(Crc64Xz::checksum(data), Crc64::checksum(data));
}

#[test]
#[cfg(all(feature = "checksums", feature = "alloc"))]
fn buffered_checksum_constructors_compile() {
  let data = b"root-surface";

  let mut buffered = rscrypto::Crc32C::buffered();
  buffered.update(data);

  assert_eq!(buffered.finalize(), rscrypto::Crc32C::checksum(data));

  let mut explicit = BufferedCrc32C::new();
  explicit.update(data);
  assert_eq!(explicit.finalize(), rscrypto::Crc32C::checksum(data));
}

#[test]
#[cfg(feature = "hashes")]
fn root_surface_hash_exports_compile() {
  let data = b"root-surface";

  let oneshot = Blake3::digest(data);

  let mut streaming = Blake3::new();
  streaming.update(data);
  assert_eq!(oneshot, streaming.finalize());

  let mut xof = Blake3::xof(data);
  let mut out = [0u8; 16];
  xof.squeeze(&mut out);

  let mut shake = Shake256::xof(data);
  shake.squeeze(&mut out);

  let mut ascon = AsconXof::xof(data);
  ascon.squeeze(&mut out);

  assert_eq!(Xxh3::hash(data), Xxh3_64::hash(data));
  assert_eq!(RapidHash::hash(data), RapidHash64::hash(data));
  assert_ne!(Xxh3::hash(data), Xxh3::hash_with_seed(7, data));
}

#[test]
#[cfg(feature = "checksums")]
fn advanced_checksum_modules_compile() {
  fn assert_kernel_introspect<T: KernelIntrospect>() {}

  let _: Crc32Config = Crc32::config();
  let _ = Crc32Force::Auto;
  let _ = DispatchInfo::current();
  let _ = kernel_for::<Crc32>(64);
  let _ = is_hardware_accelerated();
  let _ = rscrypto::platform::describe();
  assert_kernel_introspect::<Crc32>();
}

#[test]
#[cfg(feature = "hashes")]
fn advanced_hash_modules_compile() {
  fn assert_hash_kernel_introspect<T: HashKernelIntrospect>() {}

  let _ = hash_kernel_for::<Sha256>(64);
  let _ = hash_kernel_for::<Shake256>(64);
  let _ = hash_kernel_for::<Blake3>(64);
  let _ = hash_kernel_for::<AsconHash256>(64);
  let _ = hash_kernel_for::<AsconXof>(64);
  let _ = hash_kernel_for::<Xxh3>(64);
  let _ = hash_kernel_for::<RapidHash128>(64);

  assert_hash_kernel_introspect::<Sha224>();
  assert_hash_kernel_introspect::<Sha256>();
  assert_hash_kernel_introspect::<Sha384>();
  assert_hash_kernel_introspect::<Sha512>();
  assert_hash_kernel_introspect::<Sha512_256>();
  assert_hash_kernel_introspect::<Sha3_224>();
  assert_hash_kernel_introspect::<Sha3_256>();
  assert_hash_kernel_introspect::<Sha3_384>();
  assert_hash_kernel_introspect::<Sha3_512>();
  assert_hash_kernel_introspect::<Shake128>();
  assert_hash_kernel_introspect::<Shake256>();
  assert_hash_kernel_introspect::<Blake3>();
  assert_hash_kernel_introspect::<AsconHash256>();
  assert_hash_kernel_introspect::<AsconXof>();
  assert_hash_kernel_introspect::<Xxh3>();
  assert_hash_kernel_introspect::<Xxh3_128>();
  assert_hash_kernel_introspect::<RapidHash>();
  assert_hash_kernel_introspect::<RapidHash128>();
}
