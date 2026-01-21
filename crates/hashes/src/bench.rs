//! Benchmark-only kernel accessors.
//!
//! This module exposes stable function-pointer "kernels" for use by the
//! `rscrypto-tune` binary. Production code should not depend on this API.

#![allow(clippy::indexing_slicing)] // Benchmark harness uses deliberate slicing patterns

use traits::Digest;

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

fn blake3_stream_chunks_auto(data: &[u8], chunk_size: usize) -> u64 {
  use traits::Digest as _;
  let mut h = crypto::Blake3::new();
  for chunk in data.chunks(chunk_size) {
    h.update(chunk);
  }
  u64_from_prefix(&h.finalize())
}

fn blake3_stream64_auto(data: &[u8]) -> u64 {
  blake3_stream_chunks_auto(data, 64)
}

fn blake3_stream4k_auto(data: &[u8]) -> u64 {
  blake3_stream_chunks_auto(data, 4 * 1024)
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

#[inline(always)]
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

fn blake3_first_8_words(words: [u32; 16]) -> [u32; 8] {
  [
    words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
  ]
}

fn blake3_chunk_compress_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8]) -> u64 {
  const BLOCK_LEN: usize = 64;
  const CHUNK_LEN: usize = 1024;
  const CHUNK_START: u32 = 1 << 0;
  const CHUNK_END: u32 = 1 << 1;

  let kernel = crypto::blake3::kernels::kernel(id);
  let mut acc: u32 = 0;
  let key_words = [u64_from_prefix(data) as u32; 8];
  let mut chunk_counter: u64 = 0;
  let flags: u32 = 0;

  for chunk in data.chunks(CHUNK_LEN) {
    let blocks = core::cmp::max(1usize, chunk.len().div_ceil(BLOCK_LEN));
    let (full_blocks, last_len) = if chunk.is_empty() {
      (0usize, 0usize)
    } else if chunk.len().is_multiple_of(BLOCK_LEN) {
      (blocks - 1, BLOCK_LEN)
    } else {
      (blocks - 1, chunk.len() % BLOCK_LEN)
    };

    let mut cv = key_words;
    let mut blocks_compressed: u8 = 0;
    let full_bytes = full_blocks * BLOCK_LEN;
    (kernel.chunk_compress_blocks)(
      &mut cv,
      chunk_counter,
      flags,
      &mut blocks_compressed,
      &chunk[..full_bytes],
    );

    let mut last_block = [0u8; BLOCK_LEN];
    if !chunk.is_empty() {
      let offset = full_blocks * BLOCK_LEN;
      last_block[..last_len].copy_from_slice(&chunk[offset..offset + last_len]);
    }

    let block_words = blake3_words16_from_le_bytes_64(&last_block);
    let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };
    let words = (kernel.compress)(
      &cv,
      &block_words,
      chunk_counter,
      last_len as u32,
      flags | start | CHUNK_END,
    );
    let out_cv = blake3_first_8_words(words);
    acc ^= out_cv[0];

    chunk_counter = chunk_counter.wrapping_add(1);
  }

  acc as u64
}

fn blake3_chunk_compress_portable(data: &[u8]) -> u64 {
  blake3_chunk_compress_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_chunk_compress_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_chunk_compress_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_chunk_compress_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_chunk_compress_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_chunk_compress_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_chunk_compress_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_chunk_compress_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_chunk_compress_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data)
}

#[cfg(target_arch = "aarch64")]
fn blake3_chunk_compress_aarch64_neon(data: &[u8]) -> u64 {
  blake3_chunk_compress_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data)
}

fn blake3_chunk_compress_auto(data: &[u8]) -> u64 {
  let dispatch = crypto::blake3::dispatch::kernel_dispatch();
  let kernel = dispatch.select(data.len());
  let mut acc: u32 = 0;

  const BLOCK_LEN: usize = 64;
  const CHUNK_LEN: usize = 1024;
  const CHUNK_START: u32 = 1 << 0;
  const CHUNK_END: u32 = 1 << 1;

  let key_words = [u64_from_prefix(data) as u32; 8];
  let mut chunk_counter: u64 = 0;
  let flags: u32 = 0;

  for chunk in data.chunks(CHUNK_LEN) {
    let blocks = core::cmp::max(1usize, chunk.len().div_ceil(BLOCK_LEN));
    let (full_blocks, last_len) = if chunk.is_empty() {
      (0usize, 0usize)
    } else if chunk.len().is_multiple_of(BLOCK_LEN) {
      (blocks - 1, BLOCK_LEN)
    } else {
      (blocks - 1, chunk.len() % BLOCK_LEN)
    };

    let mut cv = key_words;
    let mut blocks_compressed: u8 = 0;
    let full_bytes = full_blocks * BLOCK_LEN;
    (kernel.chunk_compress_blocks)(
      &mut cv,
      chunk_counter,
      flags,
      &mut blocks_compressed,
      &chunk[..full_bytes],
    );

    let mut last_block = [0u8; BLOCK_LEN];
    if !chunk.is_empty() {
      let offset = full_blocks * BLOCK_LEN;
      last_block[..last_len].copy_from_slice(&chunk[offset..offset + last_len]);
    }

    let block_words = blake3_words16_from_le_bytes_64(&last_block);
    let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };
    let words = (kernel.compress)(
      &cv,
      &block_words,
      chunk_counter,
      last_len as u32,
      flags | start | CHUNK_END,
    );
    let out_cv = blake3_first_8_words(words);
    acc ^= out_cv[0];

    chunk_counter = chunk_counter.wrapping_add(1);
  }

  acc as u64
}

fn blake3_parent_cv_kernel(id: crypto::blake3::kernels::Blake3KernelId, data: &[u8]) -> u64 {
  let kernel = crypto::blake3::kernels::kernel(id);
  let key_words = [u64_from_prefix(data) as u32; 8];
  let mut acc: u32 = 0;

  let pairs = data.len() / 64;
  for i in 0..pairs {
    let base = i * 64;
    // SAFETY: `base + 64` is in-bounds by construction (`pairs = len/64`).
    let left: &[u8; 32] = unsafe { &*(data.as_ptr().add(base).cast()) };
    // SAFETY: `base + 64` is in-bounds by construction (`pairs = len/64`).
    let right: &[u8; 32] = unsafe { &*(data.as_ptr().add(base + 32).cast()) };

    let l = blake3_words8_from_le_bytes_32(left);
    let r = blake3_words8_from_le_bytes_32(right);

    let out = (kernel.parent_cv)(l, r, key_words, 0);
    acc ^= out[0];
  }

  acc as u64
}

fn blake3_parent_cv_portable(data: &[u8]) -> u64 {
  blake3_parent_cv_kernel(crypto::blake3::kernels::Blake3KernelId::Portable, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cv_x86_64_ssse3(data: &[u8]) -> u64 {
  blake3_parent_cv_kernel(crypto::blake3::kernels::Blake3KernelId::X86Ssse3, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cv_x86_64_sse41(data: &[u8]) -> u64 {
  blake3_parent_cv_kernel(crypto::blake3::kernels::Blake3KernelId::X86Sse41, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cv_x86_64_avx2(data: &[u8]) -> u64 {
  blake3_parent_cv_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx2, data)
}

#[cfg(target_arch = "x86_64")]
fn blake3_parent_cv_x86_64_avx512(data: &[u8]) -> u64 {
  blake3_parent_cv_kernel(crypto::blake3::kernels::Blake3KernelId::X86Avx512, data)
}

#[cfg(target_arch = "aarch64")]
fn blake3_parent_cv_aarch64_neon(data: &[u8]) -> u64 {
  blake3_parent_cv_kernel(crypto::blake3::kernels::Blake3KernelId::Aarch64Neon, data)
}

fn blake3_parent_cv_auto(data: &[u8]) -> u64 {
  let dispatch = crypto::blake3::dispatch::kernel_dispatch();
  let kernel = dispatch.select(data.len());
  let key_words = [u64_from_prefix(data) as u32; 8];
  let mut acc: u32 = 0;

  let pairs = data.len() / 64;
  for i in 0..pairs {
    let base = i * 64;
    // SAFETY: `base + 64` is in-bounds by construction (`pairs = len/64`).
    let left: &[u8; 32] = unsafe { &*(data.as_ptr().add(base).cast()) };
    // SAFETY: `base + 64` is in-bounds by construction (`pairs = len/64`).
    let right: &[u8; 32] = unsafe { &*(data.as_ptr().add(base + 32).cast()) };

    let l = blake3_words8_from_le_bytes_32(left);
    let r = blake3_words8_from_le_bytes_32(right);

    let out = (kernel.parent_cv)(l, r, key_words, 0);
    acc ^= out[0];
  }

  acc as u64
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
    ("sha384-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha384_compress_portable,
    }),
    ("sha512-compress", "portable") => Some(Kernel {
      name: "portable",
      func: sha512_compress_portable,
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
      func: blake3_chunk_compress_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_chunk_compress_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_chunk_compress_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_chunk_compress_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-chunk", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_chunk_compress_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-chunk", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_chunk_compress_aarch64_neon,
    }),
    ("blake3-parent", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_parent_cv_portable,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/ssse3") => Some(Kernel {
      name: "x86_64/ssse3",
      func: blake3_parent_cv_x86_64_ssse3,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/sse4.1") => Some(Kernel {
      name: "x86_64/sse4.1",
      func: blake3_parent_cv_x86_64_sse41,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/avx2") => Some(Kernel {
      name: "x86_64/avx2",
      func: blake3_parent_cv_x86_64_avx2,
    }),
    #[cfg(target_arch = "x86_64")]
    ("blake3-parent", "x86_64/avx512") => Some(Kernel {
      name: "x86_64/avx512",
      func: blake3_parent_cv_x86_64_avx512,
    }),
    #[cfg(target_arch = "aarch64")]
    ("blake3-parent", "aarch64/neon") => Some(Kernel {
      name: "aarch64/neon",
      func: blake3_parent_cv_aarch64_neon,
    }),
    ("keccakf1600-permute", "portable") => Some(Kernel {
      name: "portable",
      func: keccakf1600_portable,
    }),
    // Streaming chunking profiles (currently always dispatch/auto; forcing is a no-op until multiple kernels exist).
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
      func: blake3_stream64_auto,
    }),
    ("blake3-stream4k", "portable") => Some(Kernel {
      name: "portable",
      func: blake3_stream4k_auto,
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
    "blake3-chunk" => Some(blake3_chunk_compress_auto(data)),
    "blake3-parent" => Some(blake3_parent_cv_auto(data)),
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
    "blake3-stream4k" => Some(blake3_stream4k_auto(data)),
    "sha224" => Some(u64_from_prefix(&<crypto::Sha224 as Digest>::digest(data))),
    "sha256" => Some(u64_from_prefix(&<crypto::Sha256 as Digest>::digest(data))),
    "sha384" => Some(u64_from_prefix(&<crypto::Sha384 as Digest>::digest(data))),
    "sha512" => Some(u64_from_prefix(&<crypto::Sha512 as Digest>::digest(data))),
    "sha512-224" => Some(u64_from_prefix(&<crypto::Sha512_224 as Digest>::digest(data))),
    "sha512-256" => Some(u64_from_prefix(&<crypto::Sha512_256 as Digest>::digest(data))),
    "blake3" => Some(u64_from_prefix(&<crypto::Blake3 as Digest>::digest(data))),
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
  match algo {
    "sha224-compress" => Some(crypto::sha224::dispatch::kernel_name_for_len(len)),
    "sha256-compress" | "sha256-compress-unaligned" => Some(crypto::sha256::dispatch::kernel_name_for_len(len)),
    "sha384-compress" => Some(crypto::sha384::dispatch::kernel_name_for_len(len)),
    "sha512-compress" | "sha512-compress-unaligned" => Some(crypto::sha512::dispatch::kernel_name_for_len(len)),
    "sha512-224-compress" => Some(crypto::sha512_224::dispatch::kernel_name_for_len(len)),
    "sha512-256-compress" => Some(crypto::sha512_256::dispatch::kernel_name_for_len(len)),
    "blake2b-512-compress" => Some(crypto::blake2b::dispatch::kernel_name_for_len(len)),
    "blake2s-256-compress" => Some(crypto::blake2s::dispatch::kernel_name_for_len(len)),
    "blake3-chunk" | "blake3-parent" => Some(crypto::blake3::dispatch::kernel_name_for_len(len)),
    "keccakf1600-permute" => Some(crypto::keccak::dispatch::kernel_name_for_len(len)),
    "sha256-stream64" | "sha256-stream4k" => Some(crypto::sha256::dispatch::kernel_name_for_len(len)),
    "sha512-stream64" | "sha512-stream4k" => Some(crypto::sha512::dispatch::kernel_name_for_len(len)),
    "blake2b-512-stream64" | "blake2b-512-stream4k" => Some(crypto::blake2b::dispatch::kernel_name_for_len(len)),
    "blake2s-256-stream64" | "blake2s-256-stream4k" => Some(crypto::blake2s::dispatch::kernel_name_for_len(len)),
    "blake3-stream64" | "blake3-stream4k" => Some(crypto::blake3::dispatch::kernel_name_for_len(len)),
    "sha224" => Some(crypto::sha224::dispatch::kernel_name_for_len(len)),
    "sha256" => Some(crypto::sha256::dispatch::kernel_name_for_len(len)),
    "sha384" => Some(crypto::sha384::dispatch::kernel_name_for_len(len)),
    "sha512" => Some(crypto::sha512::dispatch::kernel_name_for_len(len)),
    "sha512-224" => Some(crypto::sha512_224::dispatch::kernel_name_for_len(len)),
    "sha512-256" => Some(crypto::sha512_256::dispatch::kernel_name_for_len(len)),
    "blake3" => Some(crypto::blake3::dispatch::kernel_name_for_len(len)),
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
