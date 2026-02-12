//! Benchmark-only kernel accessors.
//!
//! This module exposes stable function-pointer "kernels" for use by the
//! `rscrypto-tune` binary. Production code should not depend on this API.

#![allow(clippy::indexing_slicing)] // Benchmark harness uses deliberate slicing patterns

extern crate alloc;

use alloc::{vec, vec::Vec};

use traits::{Digest, Xof};

use crate::{crypto, fast};

#[derive(Clone, Copy)]
pub struct Kernel {
  pub name: &'static str,
  pub func: fn(&[u8]) -> u64,
}

#[inline]
#[must_use]
fn u64_from_prefix(bytes: &[u8]) -> u64 {
  let (chunks, _) = bytes.as_chunks::<8>();
  let Some(chunk) = chunks.first() else {
    return 0;
  };
  u64::from_le_bytes(*chunk)
}

#[inline]
#[must_use]
fn u64_from_u32_state(state: &[u32; 8]) -> u64 {
  ((state[0] as u64) << 32) | (state[1] as u64)
}

#[inline]
#[must_use]
fn u64_from_u64_state(state: &[u64; 8]) -> u64 {
  state[0]
}

fn sha224_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha224::digest_portable(data))
}
fn sha256_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha256::digest_portable(data))
}
fn sha384_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha384::digest_portable(data))
}
fn sha512_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha512::digest_portable(data))
}
fn sha512_224_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha512_224::digest_portable(data))
}
fn sha512_256_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha512_256::digest_portable(data))
}
fn blake3_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake3::digest_portable(data))
}
fn blake2b_512_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake2b512::digest_portable(data))
}
fn blake2s_256_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake2s256::digest_portable(data))
}

fn sha3_224_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha3_224::digest_portable(data))
}
fn sha3_256_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha3_256::digest_portable(data))
}
fn sha3_384_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha3_384::digest_portable(data))
}
fn sha3_512_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Sha3_512::digest_portable(data))
}

fn shake128_portable(data: &[u8]) -> u64 {
  let mut out = [0u8; 32];
  crypto::Shake128::hash_into_portable(data, &mut out);
  u64_from_prefix(&out)
}

fn shake256_portable(data: &[u8]) -> u64 {
  let mut out = [0u8; 32];
  crypto::Shake256::hash_into_portable(data, &mut out);
  u64_from_prefix(&out)
}

fn xxh3_portable(data: &[u8]) -> u64 {
  let f = fast::xxh3::kernels::hash64_fn(fast::xxh3::kernels::Xxh3KernelId::Portable);
  f(data, 0)
}

fn rapidhash_portable(data: &[u8]) -> u64 {
  let f = fast::rapidhash::kernels::hash64_fn(fast::rapidhash::kernels::RapidHashKernelId::Portable);
  f(data, 0)
}

fn siphash_portable(data: &[u8]) -> u64 {
  let f = fast::siphash::kernels::hash13_fn(fast::siphash::kernels::SipHashKernelId::Portable);
  f([0, 0], data)
}

fn sha224_compress_kernel(id: crypto::sha224::kernels::Sha224KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;
  let compress = crypto::sha224::kernels::compress_blocks_fn(id);
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed as u32; 8];

  if blocks_len != 0 {
    compress(&mut state, &data[..blocks_len]);
  }

  u64_from_u32_state(&state)
}

fn sha224_compress_portable(data: &[u8]) -> u64 {
  sha224_compress_kernel(crypto::sha224::kernels::Sha224KernelId::Portable, data)
}

fn sha224_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed as u32; 8];

  if blocks_len != 0 {
    crypto::sha224::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u32_state(&state)
}

fn sha256_compress_kernel(id: crypto::sha256::kernels::Sha256KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;
  let compress = crypto::sha256::kernels::compress_blocks_fn(id);
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed as u32; 8];

  if blocks_len != 0 {
    compress(&mut state, &data[..blocks_len]);
  }

  u64_from_u32_state(&state)
}

fn sha256_compress_portable(data: &[u8]) -> u64 {
  sha256_compress_kernel(crypto::sha256::kernels::Sha256KernelId::Portable, data)
}

fn sha256_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed as u32; 8];

  if blocks_len != 0 {
    crypto::sha256::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u32_state(&state)
}

fn sha256_compress_unaligned_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;
  if data.len() <= 1 {
    return 0;
  }
  let data = &data[1..];
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed as u32; 8];

  if blocks_len != 0 {
    crypto::sha256::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u32_state(&state)
}

fn sha384_compress_kernel(id: crypto::sha384::kernels::Sha384KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let compress = crypto::sha384::kernels::compress_blocks_fn(id);
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    compress(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha384_compress_portable(data: &[u8]) -> u64 {
  sha384_compress_kernel(crypto::sha384::kernels::Sha384KernelId::Portable, data)
}

fn sha384_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    crypto::sha384::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha512_compress_kernel(id: crypto::sha512::kernels::Sha512KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let compress = crypto::sha512::kernels::compress_blocks_fn(id);
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    compress(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha512_compress_portable(data: &[u8]) -> u64 {
  sha512_compress_kernel(crypto::sha512::kernels::Sha512KernelId::Portable, data)
}

fn sha512_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    crypto::sha512::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha512_compress_unaligned_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  if data.len() <= 1 {
    return 0;
  }
  let data = &data[1..];
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    crypto::sha512::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha512_224_compress_kernel(id: crypto::sha512_224::kernels::Sha512_224KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let compress = crypto::sha512_224::kernels::compress_blocks_fn(id);
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    compress(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha512_224_compress_portable(data: &[u8]) -> u64 {
  sha512_224_compress_kernel(crypto::sha512_224::kernels::Sha512_224KernelId::Portable, data)
}

fn sha512_224_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    crypto::sha512_224::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha512_256_compress_kernel(id: crypto::sha512_256::kernels::Sha512_256KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let compress = crypto::sha512_256::kernels::compress_blocks_fn(id);
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    compress(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha512_256_compress_portable(data: &[u8]) -> u64 {
  sha512_256_compress_kernel(crypto::sha512_256::kernels::Sha512_256KernelId::Portable, data)
}

fn sha512_256_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let blocks_len = data.len() - (data.len() % BLOCK_LEN);

  let seed = u64_from_prefix(data);
  let mut state = [seed; 8];

  if blocks_len != 0 {
    crypto::sha512_256::dispatch::compress_blocks(&mut state, &data[..blocks_len]);
  }

  u64_from_u64_state(&state)
}

fn sha256_stream_chunks_auto(data: &[u8], chunk_size: usize) -> u64 {
  use traits::Digest as _;
  let mut h = crypto::Sha256::new();
  for chunk in data.chunks(chunk_size) {
    h.update(chunk);
  }
  u64_from_prefix(&h.finalize())
}

fn sha256_stream64_auto(data: &[u8]) -> u64 {
  sha256_stream_chunks_auto(data, 64)
}

fn sha256_stream4k_auto(data: &[u8]) -> u64 {
  sha256_stream_chunks_auto(data, 4 * 1024)
}

fn sha512_stream_chunks_auto(data: &[u8], chunk_size: usize) -> u64 {
  use traits::Digest as _;
  let mut h = crypto::Sha512::new();
  for chunk in data.chunks(chunk_size) {
    h.update(chunk);
  }
  u64_from_prefix(&h.finalize())
}

fn sha512_stream64_auto(data: &[u8]) -> u64 {
  sha512_stream_chunks_auto(data, 64)
}

fn sha512_stream4k_auto(data: &[u8]) -> u64 {
  sha512_stream_chunks_auto(data, 4 * 1024)
}

fn blake2b_512_stream_chunks_auto(data: &[u8], chunk_size: usize) -> u64 {
  use traits::Digest as _;
  let mut h = crypto::Blake2b512::new();
  for chunk in data.chunks(chunk_size) {
    h.update(chunk);
  }
  u64_from_prefix(&h.finalize())
}

fn blake2b_512_stream64_auto(data: &[u8]) -> u64 {
  blake2b_512_stream_chunks_auto(data, 64)
}

fn blake2b_512_stream4k_auto(data: &[u8]) -> u64 {
  blake2b_512_stream_chunks_auto(data, 4 * 1024)
}

fn blake2s_256_stream_chunks_auto(data: &[u8], chunk_size: usize) -> u64 {
  use traits::Digest as _;
  let mut h = crypto::Blake2s256::new();
  for chunk in data.chunks(chunk_size) {
    h.update(chunk);
  }
  u64_from_prefix(&h.finalize())
}

fn blake2s_256_stream64_auto(data: &[u8]) -> u64 {
  blake2s_256_stream_chunks_auto(data, 64)
}

fn blake2s_256_stream4k_auto(data: &[u8]) -> u64 {
  blake2s_256_stream_chunks_auto(data, 4 * 1024)
}

#[derive(Clone, Copy)]
enum Blake3StreamMode {
  Plain,
  Keyed,
  Derive,
  Xof,
}

#[derive(Clone, Copy)]
enum Blake3ChunkPattern {
  Fixed(usize),
  MixedBursty,
}

const BLAKE3_STREAM_BENCH_KEY: [u8; 32] = [
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
  0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
];
const BLAKE3_STREAM_BENCH_CONTEXT: &str = "rscrypto-blake3-stream-bench";
const BLAKE3_STREAM_MIXED_BURSTY_PATTERN: &[usize] = &[64, 64, 256, 1024, 4096, 256, 64, 1024, 4096, 64, 256];

#[inline]
fn blake3_update_with_pattern(h: &mut crypto::Blake3, data: &[u8], pattern: Blake3ChunkPattern) {
  match pattern {
    Blake3ChunkPattern::Fixed(chunk_size) => {
      for chunk in data.chunks(chunk_size.max(1)) {
        h.update(chunk);
      }
    }
    Blake3ChunkPattern::MixedBursty => {
      let mut offset = 0usize;
      let mut idx = 0usize;
      while offset < data.len() {
        let step = BLAKE3_STREAM_MIXED_BURSTY_PATTERN[idx % BLAKE3_STREAM_MIXED_BURSTY_PATTERN.len()].max(1);
        let end = offset.saturating_add(step).min(data.len());
        h.update(&data[offset..end]);
        offset = end;
        idx = idx.saturating_add(1);
      }
    }
  }
}

fn blake3_stream_auto_mode(data: &[u8], pattern: Blake3ChunkPattern, mode: Blake3StreamMode) -> u64 {
  use traits::Digest as _;
  let mut h = match mode {
    Blake3StreamMode::Plain => crypto::Blake3::new(),
    Blake3StreamMode::Keyed => crypto::Blake3::new_keyed(&BLAKE3_STREAM_BENCH_KEY),
    Blake3StreamMode::Derive => crypto::Blake3::new_derive_key(BLAKE3_STREAM_BENCH_CONTEXT),
    Blake3StreamMode::Xof => crypto::Blake3::new(),
  };
  blake3_update_with_pattern(&mut h, data, pattern);

  match mode {
    Blake3StreamMode::Xof => {
      let mut xof = h.finalize_xof();
      let mut out = [0u8; 32];
      xof.squeeze(&mut out);
      u64_from_prefix(&out)
    }
    _ => u64_from_prefix(&h.finalize()),
  }
}

fn blake3_stream64_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(64), Blake3StreamMode::Plain)
}

fn blake3_stream256_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(256), Blake3StreamMode::Plain)
}

fn blake3_stream1k_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(1024), Blake3StreamMode::Plain)
}

fn blake3_stream4k_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(4 * 1024), Blake3StreamMode::Plain)
}

fn blake3_stream_mixed_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::MixedBursty, Blake3StreamMode::Plain)
}

fn blake3_stream64_keyed_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(64), Blake3StreamMode::Keyed)
}

fn blake3_stream4k_keyed_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(4 * 1024), Blake3StreamMode::Keyed)
}

fn blake3_stream64_derive_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(64), Blake3StreamMode::Derive)
}

fn blake3_stream4k_derive_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(4 * 1024), Blake3StreamMode::Derive)
}

fn blake3_stream64_xof_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(64), Blake3StreamMode::Xof)
}

fn blake3_stream4k_xof_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::Fixed(4 * 1024), Blake3StreamMode::Xof)
}

fn blake3_stream_mixed_xof_auto(data: &[u8]) -> u64 {
  blake3_stream_auto_mode(data, Blake3ChunkPattern::MixedBursty, Blake3StreamMode::Xof)
}

#[must_use]
fn parse_blake3_stream_algo(algo: &str) -> Option<(Blake3StreamMode, Blake3ChunkPattern)> {
  let suffix = algo.strip_prefix("blake3-stream")?;
  let (pattern_str, mode) = if let Some(rest) = suffix.strip_suffix("-keyed") {
    (rest, Blake3StreamMode::Keyed)
  } else if let Some(rest) = suffix.strip_suffix("-derive") {
    (rest, Blake3StreamMode::Derive)
  } else if let Some(rest) = suffix.strip_suffix("-xof") {
    (rest, Blake3StreamMode::Xof)
  } else {
    (suffix, Blake3StreamMode::Plain)
  };

  let pattern = match pattern_str {
    "64" => Blake3ChunkPattern::Fixed(64),
    "256" => Blake3ChunkPattern::Fixed(256),
    "1k" | "1024" => Blake3ChunkPattern::Fixed(1024),
    "4k" => Blake3ChunkPattern::Fixed(4 * 1024),
    "mixed" | "-mixed" => Blake3ChunkPattern::MixedBursty,
    _ => return None,
  };

  Some((mode, pattern))
}

fn blake3_kernel_id_from_name(name: &str) -> Option<crypto::blake3::kernels::Blake3KernelId> {
  match name {
    "portable" => Some(crypto::blake3::kernels::Blake3KernelId::Portable),
    #[cfg(target_arch = "x86_64")]
    "x86_64/ssse3" => Some(crypto::blake3::kernels::Blake3KernelId::X86Ssse3),
    #[cfg(target_arch = "x86_64")]
    "x86_64/sse4.1" => Some(crypto::blake3::kernels::Blake3KernelId::X86Sse41),
    #[cfg(target_arch = "x86_64")]
    "x86_64/avx2" => Some(crypto::blake3::kernels::Blake3KernelId::X86Avx2),
    #[cfg(target_arch = "x86_64")]
    "x86_64/avx512" => Some(crypto::blake3::kernels::Blake3KernelId::X86Avx512),
    #[cfg(target_arch = "aarch64")]
    "aarch64/neon" => Some(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon),
    _ => None,
  }
}

fn blake3_parse_kernel_pair(
  kernel_name: &str,
) -> Option<(
  crypto::blake3::kernels::Blake3KernelId,
  crypto::blake3::kernels::Blake3KernelId,
)> {
  if let Some((stream, bulk)) = kernel_name.split_once('+') {
    return Some((blake3_kernel_id_from_name(stream)?, blake3_kernel_id_from_name(bulk)?));
  }
  let id = blake3_kernel_id_from_name(kernel_name)?;
  Some((id, id))
}

fn blake3_stream_chunks_kernel_pair(
  stream_id: crypto::blake3::kernels::Blake3KernelId,
  bulk_id: crypto::blake3::kernels::Blake3KernelId,
  mode: Blake3StreamMode,
  data: &[u8],
  pattern: Blake3ChunkPattern,
) -> u64 {
  let out = match (mode, pattern) {
    (Blake3StreamMode::Plain, Blake3ChunkPattern::Fixed(chunk_size)) => {
      crypto::Blake3::stream_chunks_with_kernel_pair_id(stream_id, bulk_id, chunk_size, data)
    }
    (Blake3StreamMode::Keyed, Blake3ChunkPattern::Fixed(chunk_size)) => {
      crypto::Blake3::stream_chunks_keyed_with_kernel_pair_id(stream_id, bulk_id, chunk_size, data)
    }
    (Blake3StreamMode::Derive, Blake3ChunkPattern::Fixed(chunk_size)) => {
      crypto::Blake3::stream_chunks_derive_with_kernel_pair_id(stream_id, bulk_id, chunk_size, data)
    }
    (Blake3StreamMode::Xof, Blake3ChunkPattern::Fixed(chunk_size)) => {
      crypto::Blake3::stream_chunks_xof_with_kernel_pair_id(stream_id, bulk_id, chunk_size, data)
    }
    (Blake3StreamMode::Plain, Blake3ChunkPattern::MixedBursty) => {
      crypto::Blake3::stream_chunks_mixed_with_kernel_pair_id(
        stream_id,
        bulk_id,
        BLAKE3_STREAM_MIXED_BURSTY_PATTERN,
        data,
      )
    }
    (Blake3StreamMode::Keyed, Blake3ChunkPattern::MixedBursty) => {
      crypto::Blake3::stream_chunks_mixed_keyed_with_kernel_pair_id(
        stream_id,
        bulk_id,
        BLAKE3_STREAM_MIXED_BURSTY_PATTERN,
        data,
      )
    }
    (Blake3StreamMode::Derive, Blake3ChunkPattern::MixedBursty) => {
      crypto::Blake3::stream_chunks_mixed_derive_with_kernel_pair_id(
        stream_id,
        bulk_id,
        BLAKE3_STREAM_MIXED_BURSTY_PATTERN,
        data,
      )
    }
    (Blake3StreamMode::Xof, Blake3ChunkPattern::MixedBursty) => {
      crypto::Blake3::stream_chunks_mixed_xof_with_kernel_pair_id(
        stream_id,
        bulk_id,
        BLAKE3_STREAM_MIXED_BURSTY_PATTERN,
        data,
      )
    }
  };
  u64_from_prefix(&out)
}

fn blake3_stream_chunks_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8], chunk_size: usize) -> u64 {
  blake3_stream_chunks_kernel_pair(
    id,
    id,
    Blake3StreamMode::Plain,
    data,
    Blake3ChunkPattern::Fixed(chunk_size),
  )
}

#[must_use]
pub fn run_blake3_stream_forced(algo: &str, kernel_name: &str, data: &[u8]) -> Option<u64> {
  let (mode, pattern) = parse_blake3_stream_algo(algo)?;
  let (stream_id, bulk_id) = blake3_parse_kernel_pair(kernel_name)?;
  Some(blake3_stream_chunks_kernel_pair(
    stream_id, bulk_id, mode, data, pattern,
  ))
}

fn blake3_stream64_portable(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data, 64)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream64_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data, 64)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream64_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data, 64)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream64_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data, 64)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream64_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data, 64)
}

#[cfg(target_arch = "aarch64")]
fn blake3_stream64_aarch64_neon(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data, 64)
}

fn blake3_stream4k_portable(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data, 4 * 1024)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream4k_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data, 4 * 1024)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream4k_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data, 4 * 1024)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream4k_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data, 4 * 1024)
}

#[cfg(target_arch = "x86_64")]
fn blake3_stream4k_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data, 4 * 1024)
}

#[cfg(target_arch = "aarch64")]
fn blake3_stream4k_aarch64_neon(data: &[u8]) -> u64 {
  blake3_stream_chunks_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data, 4 * 1024)
}

fn blake2b_512_compress_kernel(id: crypto::blake2b::kernels::Blake2b512KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;
  let compress = crypto::blake2b::kernels::compress_fn(id);

  let seed = u64_from_prefix(data);
  let mut h = [seed; 8];
  let mut t: u128 = 0;

  let full_blocks = data.len() / BLOCK_LEN;
  let rem = data.len() % BLOCK_LEN;

  if full_blocks == 0 {
    let mut last = [0u8; BLOCK_LEN];
    last[..data.len()].copy_from_slice(data);
    compress(&mut h, &last, &mut t, true, data.len() as u32);
    return h[0] ^ (t as u64);
  }

  if rem == 0 {
    let prefix = (full_blocks - 1) * BLOCK_LEN;
    if prefix != 0 {
      compress(&mut h, &data[..prefix], &mut t, false, 0);
    }
    compress(
      &mut h,
      &data[prefix..prefix + BLOCK_LEN],
      &mut t,
      true,
      BLOCK_LEN as u32,
    );
  } else {
    let prefix = full_blocks * BLOCK_LEN;
    compress(&mut h, &data[..prefix], &mut t, false, 0);
    let mut last = [0u8; BLOCK_LEN];
    last[..rem].copy_from_slice(&data[prefix..]);
    compress(&mut h, &last, &mut t, true, rem as u32);
  }

  h[0] ^ (t as u64)
}

fn blake2b_512_compress_portable(data: &[u8]) -> u64 {
  blake2b_512_compress_kernel(crypto::blake2b::kernels::Blake2b512KernelId::Portable, data)
}

fn blake2b_512_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 128;

  let seed = u64_from_prefix(data);
  let mut h = [seed; 8];
  let mut t: u128 = 0;

  let full_blocks = data.len() / BLOCK_LEN;
  let rem = data.len() % BLOCK_LEN;

  if full_blocks == 0 {
    let mut last = [0u8; BLOCK_LEN];
    last[..data.len()].copy_from_slice(data);
    crypto::blake2b::dispatch::compress(&mut h, &last, &mut t, true, data.len() as u32);
    return h[0] ^ (t as u64);
  }

  if rem == 0 {
    let prefix = (full_blocks - 1) * BLOCK_LEN;
    if prefix != 0 {
      crypto::blake2b::dispatch::compress(&mut h, &data[..prefix], &mut t, false, 0);
    }
    crypto::blake2b::dispatch::compress(
      &mut h,
      &data[prefix..prefix + BLOCK_LEN],
      &mut t,
      true,
      BLOCK_LEN as u32,
    );
  } else {
    let prefix = full_blocks * BLOCK_LEN;
    crypto::blake2b::dispatch::compress(&mut h, &data[..prefix], &mut t, false, 0);
    let mut last = [0u8; BLOCK_LEN];
    last[..rem].copy_from_slice(&data[prefix..]);
    crypto::blake2b::dispatch::compress(&mut h, &last, &mut t, true, rem as u32);
  }

  h[0] ^ (t as u64)
}

fn blake2s_256_compress_kernel(id: crypto::blake2s::kernels::Blake2s256KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;
  let compress = crypto::blake2s::kernels::compress_fn(id);

  let seed = u64_from_prefix(data);
  let mut h = [seed as u32; 8];
  let mut t: u64 = 0;

  let full_blocks = data.len() / BLOCK_LEN;
  let rem = data.len() % BLOCK_LEN;

  if full_blocks == 0 {
    let mut last = [0u8; BLOCK_LEN];
    last[..data.len()].copy_from_slice(data);
    compress(&mut h, &last, &mut t, true, data.len() as u32);
    return u64_from_u32_state(&h) ^ t;
  }

  if rem == 0 {
    let prefix = (full_blocks - 1) * BLOCK_LEN;
    if prefix != 0 {
      compress(&mut h, &data[..prefix], &mut t, false, 0);
    }
    compress(
      &mut h,
      &data[prefix..prefix + BLOCK_LEN],
      &mut t,
      true,
      BLOCK_LEN as u32,
    );
  } else {
    let prefix = full_blocks * BLOCK_LEN;
    compress(&mut h, &data[..prefix], &mut t, false, 0);
    let mut last = [0u8; BLOCK_LEN];
    last[..rem].copy_from_slice(&data[prefix..]);
    compress(&mut h, &last, &mut t, true, rem as u32);
  }

  u64_from_u32_state(&h) ^ t
}

fn blake2s_256_compress_portable(data: &[u8]) -> u64 {
  blake2s_256_compress_kernel(crypto::blake2s::kernels::Blake2s256KernelId::Portable, data)
}

fn blake2s_256_compress_auto(data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;

  let seed = u64_from_prefix(data);
  let mut h = [seed as u32; 8];
  let mut t: u64 = 0;

  let full_blocks = data.len() / BLOCK_LEN;
  let rem = data.len() % BLOCK_LEN;

  if full_blocks == 0 {
    let mut last = [0u8; BLOCK_LEN];
    last[..data.len()].copy_from_slice(data);
    crypto::blake2s::dispatch::compress(&mut h, &last, &mut t, true, data.len() as u32);
    return u64_from_u32_state(&h) ^ t;
  }

  if rem == 0 {
    let prefix = (full_blocks - 1) * BLOCK_LEN;
    if prefix != 0 {
      crypto::blake2s::dispatch::compress(&mut h, &data[..prefix], &mut t, false, 0);
    }
    crypto::blake2s::dispatch::compress(
      &mut h,
      &data[prefix..prefix + BLOCK_LEN],
      &mut t,
      true,
      BLOCK_LEN as u32,
    );
  } else {
    let prefix = full_blocks * BLOCK_LEN;
    crypto::blake2s::dispatch::compress(&mut h, &data[..prefix], &mut t, false, 0);
    let mut last = [0u8; BLOCK_LEN];
    last[..rem].copy_from_slice(&data[prefix..]);
    crypto::blake2s::dispatch::compress(&mut h, &last, &mut t, true, rem as u32);
  }

  u64_from_u32_state(&h) ^ t
}

fn blake3_words16_from_le_bytes_64(bytes: &[u8; 64]) -> [u32; 16] {
  let mut out = [0u32; 16];
  for (i, dst) in out.iter_mut().enumerate() {
    // SAFETY: `bytes` is 64 bytes, and `i < 16` so `i * 4` stays in-bounds.
    // We use `read_unaligned` because the input slice has 1-byte alignment.
    let w = unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(i * 4).cast::<u32>()) };
    *dst = u32::from_le(w);
  }
  out
}

fn blake3_words8_from_le_bytes_32(bytes: &[u8; 32]) -> [u32; 8] {
  let mut out = [0u32; 8];
  for (i, dst) in out.iter_mut().enumerate() {
    // SAFETY: `bytes` is 32 bytes, and `i < 8` so `i * 4` stays in-bounds.
    // We use `read_unaligned` because the input slice has 1-byte alignment.
    let w = unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(i * 4).cast::<u32>()) };
    *dst = u32::from_le(w);
  }
  out
}

fn blake3_oneshot_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake3::digest_with_kernel_id(id, data))
}

fn blake3_oneshot_portable(data: &[u8]) -> u64 {
  blake3_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_oneshot_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_oneshot_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_oneshot_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_oneshot_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data)
}

#[cfg(target_arch = "aarch64")]
fn blake3_oneshot_aarch64_neon(data: &[u8]) -> u64 {
  blake3_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data)
}

fn blake3_oneshot_auto(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake3::digest(data))
}

fn blake3_keyed_oneshot_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake3::keyed_digest_with_kernel_id(
    id,
    &BLAKE3_STREAM_BENCH_KEY,
    data,
  ))
}

fn blake3_keyed_oneshot_portable(data: &[u8]) -> u64 {
  blake3_keyed_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_keyed_oneshot_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_keyed_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_keyed_oneshot_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_keyed_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_keyed_oneshot_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_keyed_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_keyed_oneshot_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_keyed_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data)
}

#[cfg(target_arch = "aarch64")]
fn blake3_keyed_oneshot_aarch64_neon(data: &[u8]) -> u64 {
  blake3_keyed_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data)
}

fn blake3_keyed_oneshot_auto(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake3::keyed_digest(&BLAKE3_STREAM_BENCH_KEY, data))
}

fn blake3_derive_oneshot_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake3::derive_key_with_kernel_id(
    id,
    BLAKE3_STREAM_BENCH_CONTEXT,
    data,
  ))
}

fn blake3_derive_oneshot_portable(data: &[u8]) -> u64 {
  blake3_derive_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_derive_oneshot_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_derive_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_derive_oneshot_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_derive_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_derive_oneshot_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_derive_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_derive_oneshot_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_derive_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data)
}

#[cfg(target_arch = "aarch64")]
fn blake3_derive_oneshot_aarch64_neon(data: &[u8]) -> u64 {
  blake3_derive_oneshot_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data)
}

fn blake3_derive_oneshot_auto(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::Blake3::derive_key(BLAKE3_STREAM_BENCH_CONTEXT, data))
}

fn blake3_parent_cvs_many_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8]) -> u64 {
  const PARENT_BLOCK_LEN: usize = 64;
  const BATCH: usize = 64;

  let key_words = [u64_from_prefix(data) as u32; 8];
  let flags: u32 = 0;
  let mut acc: u32 = 0;

  let mut parents = [[0u32; 16]; BATCH];
  let mut out = [[0u32; 8]; BATCH];
  let mut filled = 0usize;

  for parent_bytes in data.chunks(PARENT_BLOCK_LEN) {
    let mut block = [0u8; PARENT_BLOCK_LEN];
    block[..parent_bytes.len()].copy_from_slice(parent_bytes);
    parents[filled] = blake3_words16_from_le_bytes_64(&block);
    filled += 1;

    if filled == BATCH {
      crypto::blake3::kernels::parent_cvs_many_inline(id, &parents[..filled], key_words, flags, &mut out[..filled]);
      for slot in out.iter().take(filled) {
        acc ^= slot[0];
      }
      filled = 0;
    }
  }

  if filled != 0 {
    crypto::blake3::kernels::parent_cvs_many_inline(id, &parents[..filled], key_words, flags, &mut out[..filled]);
    for slot in out.iter().take(filled) {
      acc ^= slot[0];
    }
  }

  acc as u64
}

fn blake3_parent_cvs_many_portable(data: &[u8]) -> u64 {
  blake3_parent_cvs_many_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cvs_many_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_parent_cvs_many_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cvs_many_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_parent_cvs_many_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cvs_many_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_parent_cvs_many_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cvs_many_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_parent_cvs_many_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data)
}

#[cfg(target_arch = "aarch64")]
fn blake3_parent_cvs_many_aarch64_neon(data: &[u8]) -> u64 {
  blake3_parent_cvs_many_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data)
}

fn blake3_parent_cvs_many_auto(data: &[u8]) -> u64 {
  let dispatch = crypto::blake3::dispatch::kernel_dispatch();
  let kernel = dispatch.select(data.len());
  blake3_parent_cvs_many_kernel(kernel.id, data)
}

fn blake3_parent_fold_root_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8]) -> u64 {
  const CV_BYTES: usize = 32;

  let key_words = [u64_from_prefix(data) as u32; 8];
  let flags: u32 = 0;

  let mut cur: Vec<[u32; 8]> = Vec::with_capacity(data.len().div_ceil(CV_BYTES));
  for cv_bytes in data.chunks(CV_BYTES) {
    let mut tmp = [0u8; CV_BYTES];
    tmp[..cv_bytes.len()].copy_from_slice(cv_bytes);
    cur.push(blake3_words8_from_le_bytes_32(&tmp));
  }

  if cur.is_empty() {
    return 0;
  }

  let mut next = vec![[0u32; 8]; cur.len()];
  let mut cur_len = cur.len();

  while cur_len > 1 {
    let pairs = cur_len / 2;
    crypto::blake3::kernels::parent_cvs_many_from_cvs_inline(
      id,
      &cur[..2 * pairs],
      key_words,
      flags,
      &mut next[..pairs],
    );

    if (cur_len & 1) != 0 {
      next[pairs] = cur[cur_len - 1];
      cur_len = pairs + 1;
    } else {
      cur_len = pairs;
    }

    core::mem::swap(&mut cur, &mut next);
  }

  cur[0][0] as u64
}

fn blake3_parent_fold_root_portable(data: &[u8]) -> u64 {
  blake3_parent_fold_root_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_fold_root_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_parent_fold_root_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_fold_root_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_parent_fold_root_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_fold_root_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_parent_fold_root_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_fold_root_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_parent_fold_root_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data)
}

#[cfg(target_arch = "aarch64")]
fn blake3_parent_fold_root_aarch64_neon(data: &[u8]) -> u64 {
  blake3_parent_fold_root_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data)
}

fn blake3_parent_fold_root_auto(data: &[u8]) -> u64 {
  let dispatch = crypto::blake3::dispatch::kernel_dispatch();
  let kernel = dispatch.select(data.len());
  blake3_parent_fold_root_kernel(kernel.id, data)
}

fn keccakf1600_kernel(id: crypto::keccak::kernels::Keccakf1600KernelId, data: &[u8]) -> u64 {
  let permute = crypto::keccak::kernels::permute_fn(id);

  let mut state = [0u64; 25];
  let (blocks, rest) = data.as_chunks::<200>();

  for block in blocks {
    let (chunks, rem) = block.as_chunks::<8>();
    debug_assert!(rem.is_empty());
    for (lane, chunk) in state.iter_mut().zip(chunks.iter()) {
      *lane ^= u64::from_le_bytes(*chunk);
    }
    permute(&mut state);
  }

  if !rest.is_empty() {
    let mut last = [0u8; 200];
    for (dst, src) in last.iter_mut().zip(rest.iter()) {
      *dst = *src;
    }

    let (chunks, rem) = last.as_chunks::<8>();
    debug_assert!(rem.is_empty());
    for (lane, chunk) in state.iter_mut().zip(chunks.iter()) {
      *lane ^= u64::from_le_bytes(*chunk);
    }
    permute(&mut state);
  }

  *state.first().unwrap_or(&0)
}

fn keccakf1600_portable(data: &[u8]) -> u64 {
  keccakf1600_kernel(crypto::keccak::kernels::Keccakf1600KernelId::Portable, data)
}

fn keccakf1600_auto(data: &[u8]) -> u64 {
  let permute_fn = crypto::keccak::dispatch::permute_dispatch().select(data.len());
  let permute = |state: &mut [u64; 25]| permute_fn(state);

  let mut state = [0u64; 25];
  let (blocks, rest) = data.as_chunks::<200>();

  for block in blocks {
    let (chunks, rem) = block.as_chunks::<8>();
    debug_assert!(rem.is_empty());
    for (lane, chunk) in state.iter_mut().zip(chunks.iter()) {
      *lane ^= u64::from_le_bytes(*chunk);
    }
    permute(&mut state);
  }

  if !rest.is_empty() {
    let mut last = [0u8; 200];
    for (dst, src) in last.iter_mut().zip(rest.iter()) {
      *dst = *src;
    }

    let (chunks, rem) = last.as_chunks::<8>();
    debug_assert!(rem.is_empty());
    for (lane, chunk) in state.iter_mut().zip(chunks.iter()) {
      *lane ^= u64::from_le_bytes(*chunk);
    }
    permute(&mut state);
  }

  *state.first().unwrap_or(&0)
}

fn ascon_hash256_portable(data: &[u8]) -> u64 {
  u64_from_prefix(&crypto::AsconHash256::digest_portable(data))
}

fn ascon_xof128_portable(data: &[u8]) -> u64 {
  let mut out = [0u8; 32];
  crypto::AsconXof128::hash_into_portable(data, &mut out);
  u64_from_prefix(&out)
}

#[must_use]
pub fn get_kernel(algo: &str, name: &str) -> Option<Kernel> {
  match (algo, name) {
    ("sha224-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha224_compress_portable,
    }),
    ("sha256-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha256_compress_portable,
    }),
    ("sha256-compress-unaligned", "portable") => Some(Kernel {
      name: "portable",
      func: sha256_compress_unaligned_auto,
    }),
    ("sha384-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha384_compress_portable,
    }),
    ("sha512-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_compress_portable,
    }),
    ("sha512-compress-unaligned", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_compress_unaligned_auto,
    }),
    ("sha512-224-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_224_compress_portable,
    }),
    ("sha512-256-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_256_compress_portable,
    }),
    ("blake2b-512-compress", "portable") => Some(Kernel {
      name: "portable",
      func: blake2b_512_compress_portable,
    }),
    ("blake2s-256-compress", "portable") => Some(Kernel {
      name: "portable",
      func: blake2s_256_compress_portable,
    }),
    ("blake3-chunk", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_oneshot_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_oneshot_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_oneshot_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_oneshot_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_oneshot_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-chunk", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_oneshot_aarch64_neon,
    }),
    ("blake3-parent", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_parent_cvs_many_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_parent_cvs_many_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_parent_cvs_many_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_parent_cvs_many_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_parent_cvs_many_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-parent", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_parent_cvs_many_aarch64_neon,
    }),
    ("blake3-parent-fold", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_parent_fold_root_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent-fold", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_parent_fold_root_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent-fold", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_parent_fold_root_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent-fold", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_parent_fold_root_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent-fold", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_parent_fold_root_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-parent-fold", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_parent_fold_root_aarch64_neon,
    }),
    ("keccakf1600-permute", "portable") => Some(Kernel {
      name: "portable",
      func: keccakf1600_portable,
    }),
    // Streaming chunking profiles (some algos only have dispatch/auto today).
    ("sha256-stream64", "portable") => Some(Kernel {
      name: "portable",
      func: sha256_stream64_auto,
    }),
    ("sha256-stream4k", "portable") => Some(Kernel {
      name: "portable",
      func: sha256_stream4k_auto,
    }),
    ("sha512-stream64", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_stream64_auto,
    }),
    ("sha512-stream4k", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_stream4k_auto,
    }),
    ("blake2b-512-stream64", "portable") => Some(Kernel {
      name: "portable",
      func: blake2b_512_stream64_auto,
    }),
    ("blake2b-512-stream4k", "portable") => Some(Kernel {
      name: "portable",
      func: blake2b_512_stream4k_auto,
    }),
    ("blake2s-256-stream64", "portable") => Some(Kernel {
      name: "portable",
      func: blake2s_256_stream64_auto,
    }),
    ("blake2s-256-stream4k", "portable") => Some(Kernel {
      name: "portable",
      func: blake2s_256_stream4k_auto,
    }),
    ("blake3-stream64", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_stream64_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream64", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_stream64_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream64", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_stream64_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream64", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_stream64_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream64", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_stream64_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-stream64", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_stream64_aarch64_neon,
    }),
    ("blake3-stream4k", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_stream4k_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream4k", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_stream4k_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream4k", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_stream4k_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream4k", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_stream4k_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-stream4k", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_stream4k_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-stream4k", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_stream4k_aarch64_neon,
    }),
    ("sha224", "portable") => Some(Kernel {
      name: "portable",
      func: sha224_portable,
    }),
    ("sha256", "portable") => Some(Kernel {
      name: "portable",
      func: sha256_portable,
    }),
    ("sha384", "portable") => Some(Kernel {
      name: "portable",
      func: sha384_portable,
    }),
    ("sha512", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_portable,
    }),
    ("sha512-224", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_224_portable,
    }),
    ("sha512-256", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_256_portable,
    }),
    ("blake3", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_portable,
    }),
    ("blake3-keyed", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_keyed_oneshot_portable,
    }),
    ("blake3-derive", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_derive_oneshot_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_oneshot_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-keyed", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_keyed_oneshot_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-derive", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_derive_oneshot_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_oneshot_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-keyed", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_keyed_oneshot_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-derive", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_derive_oneshot_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_oneshot_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-keyed", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_keyed_oneshot_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-derive", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_derive_oneshot_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_oneshot_x86_64_avx512,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-keyed", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_keyed_oneshot_x86_64_avx512,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-derive", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_derive_oneshot_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_oneshot_aarch64_neon,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-keyed", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_keyed_oneshot_aarch64_neon,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-derive", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_derive_oneshot_aarch64_neon,
    }),
    ("blake2b-512", "portable") => Some(Kernel {
      name: "portable",
      func: blake2b_512_portable,
    }),
    ("blake2s-256", "portable") => Some(Kernel {
      name: "portable",
      func: blake2s_256_portable,
    }),
    ("sha3-224", "portable") => Some(Kernel {
      name: "portable",
      func: sha3_224_portable,
    }),
    ("sha3-256", "portable") => Some(Kernel {
      name: "portable",
      func: sha3_256_portable,
    }),
    ("sha3-384", "portable") => Some(Kernel {
      name: "portable",
      func: sha3_384_portable,
    }),
    ("sha3-512", "portable") => Some(Kernel {
      name: "portable",
      func: sha3_512_portable,
    }),
    ("shake128", "portable") => Some(Kernel {
      name: "portable",
      func: shake128_portable,
    }),
    ("shake256", "portable") => Some(Kernel {
      name: "portable",
      func: shake256_portable,
    }),
    ("xxh3", "portable") => Some(Kernel {
      name: "portable",
      func: xxh3_portable,
    }),
    ("rapidhash", "portable") => Some(Kernel {
      name: "portable",
      func: rapidhash_portable,
    }),
    ("siphash", "portable") => Some(Kernel {
      name: "portable",
      func: siphash_portable,
    }),
    ("keccakf1600", "portable") => Some(Kernel {
      name: "portable",
      func: keccakf1600_portable,
    }),
    ("ascon-hash256", "portable") => Some(Kernel {
      name: "portable",
      func: ascon_hash256_portable,
    }),
    ("ascon-xof128", "portable") => Some(Kernel {
      name: "portable",
      func: ascon_xof128_portable,
    }),
    _ => None,
  }
}

#[must_use]
pub fn run_auto(algo: &str, data: &[u8]) -> Option<u64> {
  if let Some((mode, pattern)) = parse_blake3_stream_algo(algo) {
    return Some(blake3_stream_auto_mode(data, pattern, mode));
  }

  match algo {
    "sha224-compress" => Some(sha224_compress_auto(data)),
    "sha256-compress" => Some(sha256_compress_auto(data)),
    "sha256-compress-unaligned" => Some(sha256_compress_unaligned_auto(data)),
    "sha384-compress" => Some(sha384_compress_auto(data)),
    "sha512-compress" => Some(sha512_compress_auto(data)),
    "sha512-compress-unaligned" => Some(sha512_compress_unaligned_auto(data)),
    "sha512-224-compress" => Some(sha512_224_compress_auto(data)),
    "sha512-256-compress" => Some(sha512_256_compress_auto(data)),
    "blake2b-512-compress" => Some(blake2b_512_compress_auto(data)),
    "blake2s-256-compress" => Some(blake2s_256_compress_auto(data)),
    "blake3-chunk" => Some(blake3_oneshot_auto(data)),
    "blake3-parent" => Some(blake3_parent_cvs_many_auto(data)),
    "blake3-parent-fold" => Some(blake3_parent_fold_root_auto(data)),
    "keccakf1600-permute" => Some(keccakf1600_auto(data)),
    "sha256-stream64" => Some(sha256_stream64_auto(data)),
    "sha256-stream4k" => Some(sha256_stream4k_auto(data)),
    "sha512-stream64" => Some(sha512_stream64_auto(data)),
    "sha512-stream4k" => Some(sha512_stream4k_auto(data)),
    "blake2b-512-stream64" => Some(blake2b_512_stream64_auto(data)),
    "blake2b-512-stream4k" => Some(blake2b_512_stream4k_auto(data)),
    "blake2s-256-stream64" => Some(blake2s_256_stream64_auto(data)),
    "blake2s-256-stream4k" => Some(blake2s_256_stream4k_auto(data)),
    "blake3-stream64" => Some(blake3_stream64_auto(data)),
    "blake3-stream256" => Some(blake3_stream256_auto(data)),
    "blake3-stream1k" => Some(blake3_stream1k_auto(data)),
    "blake3-stream4k" => Some(blake3_stream4k_auto(data)),
    "blake3-stream-mixed" => Some(blake3_stream_mixed_auto(data)),
    "blake3-stream64-keyed" => Some(blake3_stream64_keyed_auto(data)),
    "blake3-stream4k-keyed" => Some(blake3_stream4k_keyed_auto(data)),
    "blake3-stream64-derive" => Some(blake3_stream64_derive_auto(data)),
    "blake3-stream4k-derive" => Some(blake3_stream4k_derive_auto(data)),
    "blake3-stream64-xof" => Some(blake3_stream64_xof_auto(data)),
    "blake3-stream4k-xof" => Some(blake3_stream4k_xof_auto(data)),
    "blake3-stream-mixed-xof" => Some(blake3_stream_mixed_xof_auto(data)),
    "sha224" => Some(u64_from_prefix(&<crypto::Sha224 as Digest>::digest(data))),
    "sha256" => Some(u64_from_prefix(&<crypto::Sha256 as Digest>::digest(data))),
    "sha384" => Some(u64_from_prefix(&<crypto::Sha384 as Digest>::digest(data))),
    "sha512" => Some(u64_from_prefix(&<crypto::Sha512 as Digest>::digest(data))),
    "sha512-224" => Some(u64_from_prefix(&<crypto::Sha512_224 as Digest>::digest(data))),
    "sha512-256" => Some(u64_from_prefix(&<crypto::Sha512_256 as Digest>::digest(data))),
    "blake3" => Some(u64_from_prefix(&<crypto::Blake3 as Digest>::digest(data))),
    "blake3-keyed" => Some(blake3_keyed_oneshot_auto(data)),
    "blake3-derive" => Some(blake3_derive_oneshot_auto(data)),
    "blake2b-512" => Some(u64_from_prefix(&<crypto::Blake2b512 as Digest>::digest(data))),
    "blake2s-256" => Some(u64_from_prefix(&<crypto::Blake2s256 as Digest>::digest(data))),
    "sha3-224" => Some(u64_from_prefix(&<crypto::Sha3_224 as Digest>::digest(data))),
    "sha3-256" => Some(u64_from_prefix(&<crypto::Sha3_256 as Digest>::digest(data))),
    "sha3-384" => Some(u64_from_prefix(&<crypto::Sha3_384 as Digest>::digest(data))),
    "sha3-512" => Some(u64_from_prefix(&<crypto::Sha3_512 as Digest>::digest(data))),
    "shake128" => {
      let mut out = [0u8; 32];
      crypto::Shake128::hash_into(data, &mut out);
      Some(u64_from_prefix(&out))
    }
    "shake256" => {
      let mut out = [0u8; 32];
      crypto::Shake256::hash_into(data, &mut out);
      Some(u64_from_prefix(&out))
    }
    "xxh3" => Some(fast::xxh3::dispatch::hash64_with_seed(0, data)),
    "rapidhash" => Some(fast::rapidhash::dispatch::hash64_with_seed(0, data)),
    "siphash" => Some(fast::siphash::dispatch::hash13_with_seed([0, 0], data)),
    "keccakf1600" => Some(keccakf1600_auto(data)),
    "ascon-hash256" => Some(u64_from_prefix(&<crypto::AsconHash256 as Digest>::digest(data))),
    "ascon-xof128" => {
      let mut out = [0u8; 32];
      crypto::AsconXof128::hash_into(data, &mut out);
      Some(u64_from_prefix(&out))
    }
    _ => None,
  }
}

#[must_use]
pub fn kernel_name_for_len(algo: &str, len: usize) -> Option<&'static str> {
  if parse_blake3_stream_algo(algo).is_some() {
    return Some(crypto::blake3::dispatch::kernel_name_for_len(len));
  }

  match algo {
    "sha224-compress" => Some(crypto::sha224::dispatch::kernel_name_for_len(len)),
    "sha256-compress" | "sha256-compress-unaligned" => Some(crypto::sha256::dispatch::kernel_name_for_len(len)),
    "sha384-compress" => Some(crypto::sha384::dispatch::kernel_name_for_len(len)),
    "sha512-compress" | "sha512-compress-unaligned" => Some(crypto::sha512::dispatch::kernel_name_for_len(len)),
    "sha512-224-compress" => Some(crypto::sha512_224::dispatch::kernel_name_for_len(len)),
    "sha512-256-compress" => Some(crypto::sha512_256::dispatch::kernel_name_for_len(len)),
    "blake2b-512-compress" => Some(crypto::blake2b::dispatch::kernel_name_for_len(len)),
    "blake2s-256-compress" => Some(crypto::blake2s::dispatch::kernel_name_for_len(len)),
    "blake3-chunk" | "blake3-parent" | "blake3-parent-fold" => Some(crypto::blake3::dispatch::kernel_name_for_len(len)),
    "keccakf1600-permute" => Some(crypto::keccak::dispatch::kernel_name_for_len(len)),
    "sha256-stream64" | "sha256-stream4k" => Some(crypto::sha256::dispatch::kernel_name_for_len(len)),
    "sha512-stream64" | "sha512-stream4k" => Some(crypto::sha512::dispatch::kernel_name_for_len(len)),
    "blake2b-512-stream64" | "blake2b-512-stream4k" => Some(crypto::blake2b::dispatch::kernel_name_for_len(len)),
    "blake2s-256-stream64" | "blake2s-256-stream4k" => Some(crypto::blake2s::dispatch::kernel_name_for_len(len)),
    "blake3-stream64"
    | "blake3-stream256"
    | "blake3-stream1k"
    | "blake3-stream4k"
    | "blake3-stream-mixed"
    | "blake3-stream64-keyed"
    | "blake3-stream4k-keyed"
    | "blake3-stream64-derive"
    | "blake3-stream4k-derive"
    | "blake3-stream64-xof"
    | "blake3-stream4k-xof"
    | "blake3-stream-mixed-xof" => Some(crypto::blake3::dispatch::kernel_name_for_len(len)),
    "sha224" => Some(crypto::sha224::dispatch::kernel_name_for_len(len)),
    "sha256" => Some(crypto::sha256::dispatch::kernel_name_for_len(len)),
    "sha384" => Some(crypto::sha384::dispatch::kernel_name_for_len(len)),
    "sha512" => Some(crypto::sha512::dispatch::kernel_name_for_len(len)),
    "sha512-224" => Some(crypto::sha512_224::dispatch::kernel_name_for_len(len)),
    "sha512-256" => Some(crypto::sha512_256::dispatch::kernel_name_for_len(len)),
    "blake3" | "blake3-keyed" | "blake3-derive" => Some(crypto::blake3::dispatch::kernel_name_for_len(len)),
    "blake2b-512" => Some(crypto::blake2b::dispatch::kernel_name_for_len(len)),
    "blake2s-256" => Some(crypto::blake2s::dispatch::kernel_name_for_len(len)),
    "sha3-224" | "sha3-256" | "sha3-384" | "sha3-512" | "shake128" | "shake256" => {
      Some(crypto::keccak::dispatch::kernel_name_for_len(len))
    }
    "xxh3" => Some(fast::xxh3::dispatch::kernel_name_for_len(len)),
    "rapidhash" => Some(fast::rapidhash::dispatch::kernel_name_for_len(len)),
    "siphash" => Some(fast::siphash::dispatch::kernel_name_for_len(len)),
    "keccakf1600" => Some(crypto::keccak::dispatch::kernel_name_for_len(len)),
    "ascon-hash256" => Some(crypto::ascon::dispatch::kernel_name_for_len(len)),
    "ascon-xof128" => Some(crypto::ascon::dispatch::kernel_name_for_len(len)),
    _ => None,
  }
}
