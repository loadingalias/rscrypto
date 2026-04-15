#![allow(unused_imports)]

#[cfg(feature = "aead")]
use rscrypto::Aead;
#[cfg(feature = "kmac")]
use rscrypto::Kmac256;
#[cfg(any(feature = "hmac", feature = "kmac"))]
use rscrypto::Mac;
#[cfg(all(feature = "aead", feature = "diag"))]
use rscrypto::aead::introspect::{DispatchInfo as AeadDispatchInfo, backend_for as aead_backend_for};
#[cfg(feature = "aead")]
use rscrypto::aead::{
  AeadBackend, AeadBufferError, AeadPrimitive, Aegis256, Aegis256Key, Aegis256Tag, BenchLane, ChaCha20Poly1305,
  ChaCha20Poly1305Key, ChaCha20Poly1305Tag, Nonce96, Nonce128, Nonce192, Nonce256, OpenError, XChaCha20Poly1305,
  XChaCha20Poly1305Key, XChaCha20Poly1305Tag, lane_target_backend, select_backend,
};
#[cfg(feature = "hkdf")]
use rscrypto::auth::HkdfOutputLengthError;
#[cfg(all(feature = "checksums", feature = "alloc"))]
use rscrypto::checksum::buffered::BufferedCrc32C;
#[cfg(feature = "checksums")]
use rscrypto::checksum::config::{
  Crc16Config, Crc16Force, Crc24Config, Crc24Force, Crc32Config, Crc32Force, Crc64Config, Crc64Force,
};
#[cfg(all(feature = "checksums", feature = "diag"))]
use rscrypto::checksum::introspect::{DispatchInfo, KernelIntrospect, is_hardware_accelerated, kernel_for};
#[cfg(feature = "checksums")]
use rscrypto::checksum::{Crc32Castagnoli, Crc32Ieee, Crc64Xz};
#[cfg(feature = "hashes")]
use rscrypto::hashes::fast::{RapidHash64, RapidHashFast64, RapidHashFast128, Xxh3_64};
#[cfg(all(feature = "hashes", feature = "diag"))]
use rscrypto::hashes::introspect::{HashKernelIntrospect, kernel_for as hash_kernel_for};
#[cfg(all(feature = "hashes", feature = "std"))]
use rscrypto::hashes::{DigestReader, DigestWriter};
#[cfg(feature = "aead")]
use rscrypto::platform::{Arch, Caps};
#[cfg(feature = "hashes")]
use rscrypto::{
  AsconCxof128, AsconCxof128Reader, AsconHash256, AsconXof, AsconXofReader, Blake3, Blake3XofReader, Cshake256,
  Cshake256XofReader, Digest, FastHash, RapidHash, RapidHash128, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224,
  Sha256, Sha384, Sha512, Sha512_256, Shake128, Shake128XofReader, Shake256, Shake256XofReader, Xof, Xxh3, Xxh3_128,
};
#[cfg(feature = "checksums")]
use rscrypto::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};
#[cfg(feature = "ed25519")]
use rscrypto::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, verify_ed25519};
#[cfg(feature = "hkdf")]
use rscrypto::{HkdfSha256, HkdfSha384};
#[cfg(feature = "hmac")]
use rscrypto::{HmacSha256, HmacSha384, HmacSha512};
use rscrypto::{VerificationError, ct};
#[cfg(feature = "x25519")]
use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};

#[test]
fn root_surface_core_exports_compile() {
  let _ = VerificationError::new();
  assert!(ct::constant_time_eq(b"ok", b"ok"));
}

#[test]
#[cfg(feature = "aead")]
fn root_surface_aead_exports_compile() {
  let nonce96 = Nonce96::from_bytes([0x11; Nonce96::LENGTH]);
  let nonce128 = Nonce128::from_bytes([0x22; Nonce128::LENGTH]);
  let nonce192 = Nonce192::from_bytes([0x33; Nonce192::LENGTH]);

  assert_eq!(nonce96.as_bytes().len(), Nonce96::LENGTH);
  assert_eq!(nonce128.as_bytes().len(), Nonce128::LENGTH);
  assert_eq!(nonce192.as_bytes().len(), Nonce192::LENGTH);

  let nonce256 = Nonce256::from_bytes([0x44; Nonce256::LENGTH]);
  assert_eq!(nonce256.as_bytes().len(), Nonce256::LENGTH);

  let _ = AeadBufferError::new();
  let _ = OpenError::buffer();
  let _ = OpenError::verification();
  let _ = AeadPrimitive::XChaCha20Poly1305.name();
  let _ = AeadBackend::Portable.name();
  let _ = BenchLane::IntelSpr.platform_name();
  let _ = BenchLane::Graviton4.arch();
  let _ = lane_target_backend(AeadPrimitive::Aes256Gcm, BenchLane::IntelSpr);
  let _ = select_backend(AeadPrimitive::AsconAead128, Arch::Wasm32, Caps::NONE);

  #[cfg(feature = "diag")]
  {
    let _ = AeadDispatchInfo::current();
    let _ = aead_backend_for(AeadPrimitive::ChaCha20Poly1305);
  }

  fn assert_aead_trait<T: Aead>() {}

  #[derive(Clone)]
  struct Marker;

  impl Aead for Marker {
    const KEY_SIZE: usize = 32;
    const NONCE_SIZE: usize = Nonce96::LENGTH;
    const TAG_SIZE: usize = 16;

    type Key = [u8; 32];
    type Nonce = Nonce96;
    type Tag = [u8; 16];

    fn new(_key: &Self::Key) -> Self {
      Self
    }

    fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
      if bytes.len() != Self::TAG_SIZE {
        return Err(AeadBufferError::new());
      }

      let mut tag = [0u8; Self::TAG_SIZE];
      tag.copy_from_slice(bytes);
      Ok(tag)
    }

    fn encrypt_in_place(&self, _nonce: &Self::Nonce, _aad: &[u8], _buffer: &mut [u8]) -> Self::Tag {
      [0u8; Self::TAG_SIZE]
    }

    fn decrypt_in_place(
      &self,
      _nonce: &Self::Nonce,
      _aad: &[u8],
      _buffer: &mut [u8],
      _tag: &Self::Tag,
    ) -> Result<(), VerificationError> {
      Ok(())
    }
  }

  assert_aead_trait::<Marker>();

  let key = XChaCha20Poly1305Key::from_bytes([0x44; XChaCha20Poly1305::KEY_SIZE]);
  let cipher = XChaCha20Poly1305::new(&key);
  let mut sealed = [0u8; 20];
  cipher.encrypt(&nonce192, b"aad", b"test", &mut sealed).unwrap();
  let _ = XChaCha20Poly1305Tag::from_bytes([0u8; XChaCha20Poly1305Tag::LENGTH]);

  let key = ChaCha20Poly1305Key::from_bytes([0x55; ChaCha20Poly1305::KEY_SIZE]);
  let cipher = ChaCha20Poly1305::new(&key);
  let mut sealed = [0u8; 20];
  cipher.encrypt(&nonce96, b"aad", b"test", &mut sealed).unwrap();
  let _ = ChaCha20Poly1305Tag::from_bytes([0u8; ChaCha20Poly1305Tag::LENGTH]);

  let key = Aegis256Key::from_bytes([0x66; Aegis256::KEY_SIZE]);
  let cipher = Aegis256::new(&key);
  let mut sealed = [0u8; 20];
  cipher.encrypt(&nonce256, b"aad", b"test", &mut sealed).unwrap();
  let _ = Aegis256Tag::from_bytes([0u8; Aegis256Tag::LENGTH]);
}

#[test]
#[cfg(feature = "hmac")]
fn root_surface_mac_exports_compile() {
  let key = b"root-surface-key";
  let data = b"root-surface-data";

  let tag = HmacSha256::mac(key, data);
  let tag384 = HmacSha384::mac(key, data);
  let tag512 = HmacSha512::mac(key, data);

  let mut mac = HmacSha256::new(key);
  mac.update(data);
  assert_eq!(tag, mac.finalize());
  assert!(mac.verify(&tag).is_ok());

  let mut mac384 = HmacSha384::new(key);
  mac384.update(data);
  assert_eq!(tag384, mac384.finalize());
  assert!(mac384.verify(&tag384).is_ok());

  let mut mac512 = HmacSha512::new(key);
  mac512.update(data);
  assert_eq!(tag512, mac512.finalize());
  assert!(mac512.verify(&tag512).is_ok());
}

#[test]
#[cfg(feature = "hkdf")]
fn root_surface_kdf_exports_compile() {
  let key = b"root-surface-key";

  let mut out = [0u8; 32];
  let hkdf = HkdfSha256::new(b"salt", key);
  hkdf.expand(b"info", &mut out).unwrap();
  assert_eq!(out, HkdfSha256::derive_array::<32>(b"salt", key, b"info").unwrap());

  let mut out384 = [0u8; 48];
  let hkdf384 = HkdfSha384::new(b"salt", key);
  hkdf384.expand(b"info", &mut out384).unwrap();
  assert_eq!(out384, HkdfSha384::derive_array::<48>(b"salt", key, b"info").unwrap());
  let _ = HkdfOutputLengthError::new();
}

#[test]
#[cfg(feature = "kmac")]
fn root_surface_kmac_exports_compile() {
  let key = b"root-surface-key";
  let data = b"root-surface-data";
  let mut out = [0u8; 32];
  let mut kmac = Kmac256::new(key, b"svc=v1");
  kmac.update(data);
  kmac.finalize_into(&mut out);
  assert!(Kmac256::verify(key, b"svc=v1", data, &out).is_ok());
}

#[test]
#[cfg(feature = "ed25519")]
fn root_surface_signature_exports_compile() {
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
#[cfg(feature = "x25519")]
fn root_surface_key_exchange_exports_compile() {
  let alice = X25519SecretKey::from_bytes([11u8; X25519SecretKey::LENGTH]);
  let bob = X25519SecretKey::from_bytes([13u8; X25519SecretKey::LENGTH]);
  let alice_public: X25519PublicKey = (&alice).into();
  let bob_public: X25519PublicKey = (&bob).into();
  let alice_shared = alice.diffie_hellman(&bob_public).unwrap();
  let bob_shared = X25519SharedSecret::diffie_hellman(&bob, &alice_public).unwrap();

  assert_eq!(alice_public.as_bytes().len(), 32);
  assert_eq!(alice_shared.as_bytes().len(), 32);
  assert_eq!(alice_shared, bob_shared);
  let _ = X25519Error::new();
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

  let mut cshake = Cshake256::xof(b"", b"ctx=v1", data);
  cshake.squeeze(&mut out);

  let mut cxof = AsconCxof128::xof(b"ctx=v1", data).unwrap();
  cxof.squeeze(&mut out);

  assert_eq!(Xxh3::hash(data), Xxh3_64::hash(data));
  assert_eq!(RapidHash::hash(data), RapidHash64::hash(data));
  let _ = RapidHashFast64::hash(data);
  let _ = RapidHashFast128::hash(data);
  assert_ne!(Xxh3::hash(data), Xxh3::hash_with_seed(7, data));
}

#[test]
#[cfg(all(feature = "checksums", feature = "diag"))]
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
#[cfg(all(feature = "hashes", feature = "diag"))]
fn advanced_hash_modules_compile() {
  fn assert_hash_kernel_introspect<T: HashKernelIntrospect>() {}

  let _ = hash_kernel_for::<Sha256>(64);
  let _ = hash_kernel_for::<Shake256>(64);
  let _ = hash_kernel_for::<Blake3>(64);
  let _ = hash_kernel_for::<AsconHash256>(64);
  let _ = hash_kernel_for::<AsconXof>(64);
  let _ = hash_kernel_for::<AsconCxof128>(64);
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
  assert_hash_kernel_introspect::<Cshake256>();
  assert_hash_kernel_introspect::<AsconCxof128>();
  assert_hash_kernel_introspect::<Blake3>();
  assert_hash_kernel_introspect::<AsconHash256>();
  assert_hash_kernel_introspect::<AsconXof>();
  assert_hash_kernel_introspect::<Xxh3>();
  assert_hash_kernel_introspect::<Xxh3_128>();
  assert_hash_kernel_introspect::<RapidHash>();
  assert_hash_kernel_introspect::<RapidHash128>();
}
