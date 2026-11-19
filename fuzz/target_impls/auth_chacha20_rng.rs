use chacha20::cipher::{KeyIvInit as _, StreamCipher as _};
use rscrypto::ChaCha20Rng;
use rscrypto_fuzz::{FuzzInput, some_or_return};

/// Keystream bytes per refill and the output bytes it yields, from `ChaCha20Rng`'s documented stream.
const CHUNK: usize = 512;
const OUTPUT: usize = CHUNK - 32;

/// The documented stream from RustCrypto's ChaCha20: each 512-byte keystream chunk under the current key
/// (nonce zero, counter zero) yields its first 480 bytes; its last 32 bytes are the next key.
fn independent_stream(seed: [u8; 32], len: usize) -> Vec<u8> {
  let mut key = seed;
  let mut stream = Vec::with_capacity(len.strict_add(OUTPUT));
  while stream.len() < len {
    let mut chunk = [0u8; CHUNK];
    chacha20::ChaCha20::new(&key.into(), &[0u8; 12].into()).apply_keystream(&mut chunk);
    stream.extend_from_slice(&chunk[..OUTPUT]);
    key.copy_from_slice(&chunk[OUTPUT..]);
  }
  stream.truncate(len);
  stream
}

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let seed: [u8; 32] = some_or_return!(input.bytes());
  let mut rng = ChaCha20Rng::from_seed(seed);
  let mut produced = Vec::new();
  // Each request pair draws bytes, a u32, or a u64, so draws cross refill boundaries at every offset.
  for &[kind, length] in input.rest().as_chunks::<2>().0 {
    match kind % 3 {
      0 => {
        let mut bytes = vec![0u8; usize::from(length).strict_mul(5)];
        rng.fill_bytes(&mut bytes);
        produced.extend(bytes);
      }
      1 => produced.extend(rng.next_u32().to_le_bytes()),
      _ => produced.extend(rng.next_u64().to_le_bytes()),
    }
  }
  assert_eq!(
    produced,
    independent_stream(seed, produced.len()),
    "ChaCha20Rng stream mismatch"
  );
}
