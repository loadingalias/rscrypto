use rscrypto::{
  Crc32C, Sha256, aead::introspect::chacha20poly1305_backend, checksum::introspect::kernel_for as checksum_kernel,
  hashes::introspect::kernel_for as hash_kernel,
};

fn main() {
  println!("Platform: {}", rscrypto::platform::describe());

  for len in [64, 4096, 1_048_576] {
    println!(
      "{len:>7} bytes: CRC-32C={}, SHA-256={}",
      checksum_kernel::<Crc32C>(len),
      hash_kernel::<Sha256>(len)
    );
  }

  println!("ChaCha20-Poly1305: {}", chacha20poly1305_backend());
}
