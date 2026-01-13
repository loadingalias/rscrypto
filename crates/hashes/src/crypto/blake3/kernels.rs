use platform::Caps;

use super::{BLOCK_LEN, CHUNK_START, PARENT, first_8_words, words16_from_le_bytes_64};

pub(crate) type CompressFn = fn(&[u32; 8], &[u32; 16], u64, u32, u32) -> [u32; 16];
pub(crate) type ChunkCompressBlocksFn = fn(&mut [u32; 8], u64, u32, &mut u8, &[u8]);
pub(crate) type ParentCvFn = fn([u32; 8], [u32; 8], [u32; 8], u32) -> [u32; 8];

#[derive(Clone, Copy)]
pub(crate) struct Kernel {
  pub(crate) compress: CompressFn,
  pub(crate) chunk_compress_blocks: ChunkCompressBlocksFn,
  pub(crate) parent_cv: ParentCvFn,
  pub(crate) name: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Blake3KernelId {
  Portable = 0,
}

pub const ALL: &[Blake3KernelId] = &[Blake3KernelId::Portable];

impl Blake3KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Blake3KernelId> {
  match name {
    "portable" => Some(Blake3KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn kernel(id: Blake3KernelId) -> Kernel {
  match id {
    Blake3KernelId::Portable => Kernel {
      compress: super::compress,
      chunk_compress_blocks: chunk_compress_blocks_portable,
      parent_cv: parent_cv_portable,
      name: id.as_str(),
    },
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Blake3KernelId) -> Caps {
  match id {
    Blake3KernelId::Portable => Caps::NONE,
  }
}

#[inline]
fn chunk_compress_blocks_portable(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    *chaining_value = first_8_words((super::compress)(
      chaining_value,
      &block_words,
      chunk_counter,
      BLOCK_LEN as u32,
      flags | start,
    ));
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

#[inline]
fn parent_cv_portable(left_child_cv: [u32; 8], right_child_cv: [u32; 8], key_words: [u32; 8], flags: u32) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words((super::compress)(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}
