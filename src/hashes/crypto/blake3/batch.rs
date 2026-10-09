//! Batched hashing of independent inputs.
//!
//! Runs of equal-length unkeyed inputs retain their whole-input SIMD kernel.
//! Keyed and derive-key inputs use the existing one-shot path. All modes
//! preserve input order and use public input lengths for dispatch.

use super::{
  Blake3, Blake3DeriveContext, Blake3KeyedHash, DERIVE_KEY_MATERIAL, IV, KEY_LEN, KEYED_HASH, OUT_LEN,
  digest_public_oneshot,
};
use crate::traits::ct;

impl Blake3 {
  /// Hash independent inputs under one key, preserving their order.
  ///
  /// `outputs[i]` becomes [`Self::keyed_digest`] of `inputs[i]`, using the
  /// existing one-shot path. This method does not allocate.
  ///
  /// Input lengths and the number of inputs are public. The batch-owned decoded
  /// key and per-input key/digest scratch are cleared when their owners drop,
  /// including during unwinding. Caller-owned inputs and output tags remain live.
  ///
  /// # Panics
  ///
  /// Panics before writing any output unless both slices have the same length.
  ///
  /// # Examples
  ///
  /// ```
  /// use rscrypto::{Blake3, Blake3KeyedHash};
  ///
  /// let key = [42; 32];
  /// let inputs: [&[u8]; 2] = [b"one", b"another message"];
  /// let mut outputs = [Blake3KeyedHash::default(); 2];
  /// Blake3::keyed_digest_batch(&key, &inputs, &mut outputs);
  /// assert!(outputs[1].ct_eq(&Blake3::keyed_digest(&key, inputs[1])).declassify());
  /// ```
  pub fn keyed_digest_batch(key: &[u8; KEY_LEN], inputs: &[&[u8]], outputs: &mut [Blake3KeyedHash]) {
    assert_eq!(
      inputs.len(),
      outputs.len(),
      "Blake3::keyed_digest_batch needs one output per input"
    );
    let mut key_words = BatchKey([0; 8]);
    for (word, bytes) in key_words.0.iter_mut().zip(key.as_chunks::<4>().0) {
      *word = u32::from_le_bytes(*bytes);
    }
    digest_with_key(&key_words.0, KEYED_HASH, inputs, &mut Outputs::Keyed(outputs));
  }
}

impl Blake3DeriveContext {
  /// Derive one key from each input under this prehashed context.
  ///
  /// `outputs[i]` equals [`Blake3::derive_key_with`] for `inputs[i]`, using the
  /// existing one-shot path. This method does not allocate.
  ///
  /// Input lengths and the number of inputs are public. Per-input key/digest
  /// scratch is cleared when its owner drops, including during unwinding.
  /// Caller-owned key material and derived outputs remain live and are the
  /// caller's cleanup responsibility.
  ///
  /// # Panics
  ///
  /// Panics before writing any output unless both slices have the same length.
  ///
  /// # Examples
  ///
  /// ```
  /// use rscrypto::{Blake3, Blake3DeriveContext};
  ///
  /// let context = Blake3DeriveContext::new("example.com 2026 session keys");
  /// let inputs: [&[u8]; 2] = [b"first key material", b"second key material"];
  /// let mut outputs = [[0; 32]; 2];
  /// context.derive_key_batch(&inputs, &mut outputs);
  /// assert_eq!(outputs[1], Blake3::derive_key_with(&context, inputs[1]));
  /// ```
  pub fn derive_key_batch(&self, inputs: &[&[u8]], outputs: &mut [[u8; OUT_LEN]]) {
    assert_eq!(
      inputs.len(),
      outputs.len(),
      "Blake3DeriveContext::derive_key_batch needs one output per input"
    );
    digest_with_key(
      &self.key_words,
      DERIVE_KEY_MATERIAL,
      inputs,
      &mut Outputs::Bytes(outputs),
    );
  }
}

struct BatchKey([u32; 8]);

impl Drop for BatchKey {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.0);
  }
}

enum Outputs<'a> {
  Bytes(&'a mut [[u8; OUT_LEN]]),
  Keyed(&'a mut [Blake3KeyedHash]),
}

impl Outputs<'_> {
  fn get_mut(&mut self, index: usize) -> &mut [u8; OUT_LEN] {
    match self {
      Self::Bytes(outputs) => &mut outputs[index],
      Self::Keyed(outputs) => &mut outputs[index].0,
    }
  }
}

fn digest_with_key(key: &[u32; 8], flags: u32, inputs: &[&[u8]], outputs: &mut Outputs<'_>) {
  for (index, input) in inputs.iter().enumerate() {
    digest_one(key, flags, input, outputs.get_mut(index));
  }
}

/// Borrow the batch key; the existing one-shot route consumes a cleared copy.
fn digest_one(key: &[u32; 8], flags: u32, input: &[u8], output: &mut [u8; OUT_LEN]) {
  struct Scratch {
    key: [u32; 8],
    digest: [u8; OUT_LEN],
    secret: bool,
  }
  impl Drop for Scratch {
    fn drop(&mut self) {
      if self.secret {
        ct::zeroize_words_no_fence(&mut self.key);
        ct::zeroize_no_fence(&mut self.digest);
        ct::zeroize_fence();
      }
    }
  }
  let mut scratch = Scratch {
    key: [0; 8],
    digest: [0; OUT_LEN],
    secret: flags != 0,
  };
  scratch.key.copy_from_slice(key);
  scratch.digest = digest_public_oneshot(&mut scratch.key, flags, input);
  *output = scratch.digest;
}

/// Hash `inputs[i]` into `outputs[i]` with the unkeyed hash.
pub(super) fn digest_batch(inputs: &[&[u8]], outputs: &mut [[u8; OUT_LEN]]) {
  debug_assert_eq!(inputs.len(), outputs.len());
  #[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only"))
  ))]
  if let Some(lanes) = lanes::Lanes::select() {
    lanes::digest_batch(lanes, inputs, outputs);
    return;
  }
  digest_serial(inputs, outputs);
}

fn digest_serial(inputs: &[&[u8]], outputs: &mut [[u8; OUT_LEN]]) {
  for (input, output) in inputs.iter().zip(outputs) {
    let mut iv = IV;
    *output = digest_public_oneshot(&mut iv, 0, input);
  }
}

#[cfg(any(
  all(target_arch = "x86_64", target_feature = "sse2"),
  target_arch = "aarch64",
  all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only"))
))]
mod lanes {
  #[cfg(any(test, not(target_arch = "wasm32")))]
  use super::super::IV;
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  use super::super::{BLOCK_LEN, CHUNK_END, CHUNK_START, ROOT, x86_64};
  use super::super::{CHUNK_LEN, OUT_LEN, dispatch, kernels::Blake3KernelId};
  use super::digest_serial;

  /// Widest lane count of any kernel below.
  const MAX_DEGREE: usize = 16;

  /// A lane-parallel kernel whose CPU features are available.
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub(super) enum Lanes {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Sse41,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Avx2,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
    Wasm,
  }

  impl Lanes {
    /// The lane kernel for the bulk backend that dispatch selected, if it has lanes.
    pub(super) fn select() -> Option<Self> {
      let bulk = dispatch::hasher_dispatch().bulk_kernel_for_update(usize::MAX).id;
      #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
      {
        [Self::Avx512, Self::Avx2, Self::Sse41].into_iter().find(|lanes| {
          lanes.available()
            && match lanes {
              Self::Avx512 => bulk == Blake3KernelId::X86Avx512,
              Self::Avx2 => matches!(bulk, Blake3KernelId::X86Avx512 | Blake3KernelId::X86Avx2),
              Self::Sse41 => bulk != Blake3KernelId::Portable,
            }
        })
      }
      #[cfg(target_arch = "aarch64")]
      {
        (bulk == Blake3KernelId::Aarch64Neon && Self::Neon.available()).then_some(Self::Neon)
      }
      #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
      {
        (bulk == Blake3KernelId::WasmSimd128 && Self::Wasm.available()).then_some(Self::Wasm)
      }
    }

    /// Whether the current CPU has this kernel's target features.
    fn available(self) -> bool {
      match self {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Self::Sse41 | Self::Avx2 | Self::Avx512 => {
          use crate::platform::caps::x86;
          crate::platform::caps().has(match self {
            Self::Sse41 => x86::SSE41.union(x86::SSSE3),
            Self::Avx2 => x86::AVX2,
            Self::Avx512 => x86::AVX512F.union(x86::AVX512VL).union(x86::AVX512DQ).union(x86::AVX2),
          })
        }
        #[cfg(target_arch = "aarch64")]
        Self::Neon => crate::platform::caps().has(crate::platform::caps::aarch64::NEON),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
        Self::Wasm => crate::platform::caps().has(crate::platform::caps::wasm::SIMD128),
      }
    }

    const fn degree(self) -> usize {
      match self {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Self::Sse41 => x86_64::sse41::DEGREE,
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Self::Avx2 => x86_64::avx2::DEGREE,
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Self::Avx512 => x86_64::avx512::DEGREE,
        #[cfg(target_arch = "aarch64")]
        Self::Neon => 4,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
        Self::Wasm => 4,
      }
    }
  }

  pub(super) fn digest_batch(lanes: Lanes, inputs: &[&[u8]], outputs: &mut [[u8; OUT_LEN]]) {
    let degree = lanes.degree();
    let mut rest = inputs;
    let mut rest_out = outputs;
    while let Some((first, _)) = rest.split_first() {
      let len = first.len();
      // The run of inputs that share this length, capped at one lane group.
      let group = rest.iter().take(degree).take_while(|input| input.len() == len).count();
      let (group_inputs, next) = rest.split_at(group);
      let (group_outputs, next_out) = rest_out.split_at_mut(group);
      if group >= 2 && (1..=CHUNK_LEN).contains(&len) {
        hash_group(lanes, len, group_inputs, group_outputs);
      } else {
        digest_serial(group_inputs, group_outputs);
      }
      rest = next;
      rest_out = next_out;
    }
  }

  /// Hash 2..=degree inputs of `len` bytes each, `len` in `1..=CHUNK_LEN`, in one lane-parallel call.
  ///
  /// Unused lanes repeat the first input; their outputs are discarded.
  fn hash_group(lanes: Lanes, len: usize, inputs: &[&[u8]], outputs: &mut [[u8; OUT_LEN]]) {
    debug_assert!((2..=lanes.degree()).contains(&inputs.len()) && inputs.len() == outputs.len());
    debug_assert!((1..=CHUNK_LEN).contains(&len) && inputs.iter().all(|input| input.len() == len));

    let mut ptrs = [inputs[0].as_ptr(); MAX_DEGREE];
    for (ptr, input) in ptrs.iter_mut().zip(inputs) {
      *ptr = input.as_ptr();
    }
    let mut out = [[0u8; OUT_LEN]; MAX_DEGREE];

    match lanes {
      #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
      Lanes::Sse41 | Lanes::Avx2 | Lanes::Avx512 => {
        let blocks = len.div_ceil(BLOCK_LEN);
        let padded_len = blocks.strict_mul(BLOCK_LEN);
        let last_block_len =
          u32::try_from(len.strict_sub(padded_len.strict_sub(BLOCK_LEN))).expect("a block length fits in u32");
        // The x86 kernels read whole blocks, so a partial final block needs zeroed
        // padding; scratch sized to the padded length keeps the zeroing short.
        // SAFETY: each `ptrs` passed to `hash` is readable for `padded_len` bytes: the inputs
        // themselves when `padded_len == len` (so `last_block_len == BLOCK_LEN`), otherwise
        // `with_padded_lanes::<N>` lanes with `N >= padded_len` and zeroed padding.
        // `len` in `1..=CHUNK_LEN` bounds `last_block_len` to `1..=BLOCK_LEN`.
        let hash = |ptrs: &[*const u8; MAX_DEGREE], out: &mut [[u8; OUT_LEN]; MAX_DEGREE]| unsafe {
          hash_many_x86(lanes, ptrs, blocks, last_block_len, out)
        };
        if padded_len == len {
          hash(&ptrs, &mut out);
        } else if padded_len <= BLOCK_LEN {
          with_padded_lanes::<BLOCK_LEN>(inputs, |padded| hash(padded, &mut out));
        } else if padded_len <= 4 * BLOCK_LEN {
          with_padded_lanes::<{ 4 * BLOCK_LEN }>(inputs, |padded| hash(padded, &mut out));
        } else {
          with_padded_lanes::<CHUNK_LEN>(inputs, |padded| hash(padded, &mut out));
        }
      }
      #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
      Lanes::Wasm => {
        let (lane_ptrs, _) = ptrs.split_first_chunk::<4>().expect("MAX_DEGREE covers SIMD128");
        let (lane_out, _) = out.split_first_chunk_mut::<4>().expect("MAX_DEGREE covers SIMD128");
        // SAFETY: selection requires the artifact's SIMD128 capability. Every
        // pointer addresses len bytes, including repeated unused lanes, and
        // separate output scratch cannot overlap any input. The kernel pads tails.
        unsafe { super::super::wasm32::hash4_roots(*lane_ptrs, len, lane_out) };
      }
      #[cfg(target_arch = "aarch64")]
      Lanes::Neon => {
        let (lane_ptrs, _) = ptrs
          .split_first_chunk::<4>()
          .expect("MAX_DEGREE covers the NEON degree");
        let (lane_out, _) = out
          .split_first_chunk_mut::<4>()
          .expect("MAX_DEGREE covers the NEON degree");
        // SAFETY: `Lanes::select` chose NEON only after detecting it. Every lane
        // pointer addresses an input of `len` readable bytes (unused lanes repeat
        // input 0), `hash_group`'s contract bounds `len` to `1..=CHUNK_LEN`, and
        // the kernel pads the partial final block itself.
        unsafe { super::super::aarch64::hash4_roots_neon(*lane_ptrs, len, &IV, 0, lane_out) };
      }
    }

    let (used, _) = out.split_at(outputs.len());
    outputs.copy_from_slice(used);
  }

  /// Copy each input into zeroed `N`-byte lane scratch and pass the lane pointers to `f`.
  ///
  /// Unused lanes repeat the first input.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  fn with_padded_lanes<const N: usize>(inputs: &[&[u8]], f: impl FnOnce(&[*const u8; MAX_DEGREE])) {
    let mut scratch = [[0u8; N]; MAX_DEGREE];
    for (lane, input) in scratch.iter_mut().zip(inputs) {
      let (prefix, _) = lane.split_at_mut(input.len());
      prefix.copy_from_slice(input);
    }
    let mut ptrs = [scratch[0].as_ptr(); MAX_DEGREE];
    for (ptr, lane) in ptrs.iter_mut().zip(&scratch[..inputs.len()]) {
      *ptr = lane.as_ptr();
    }
    f(&ptrs);
  }

  /// Hash one lane group with the owned x86 kernel for `lanes`.
  ///
  /// # Safety
  ///
  /// Every pointer in `ptrs` must be readable for `blocks * BLOCK_LEN` bytes,
  /// with the final block's bytes past `last_block_len` zero, and
  /// `last_block_len` must be in `1..=BLOCK_LEN`.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  unsafe fn hash_many_x86(
    lanes: Lanes,
    ptrs: &[*const u8; MAX_DEGREE],
    blocks: usize,
    last_block_len: u32,
    out: &mut [[u8; OUT_LEN]; MAX_DEGREE],
  ) {
    macro_rules! hash_many {
      ($module:ident :: $kernel:ident) => {{
        const DEGREE: usize = x86_64::$module::DEGREE;
        let (lane_ptrs, _) = ptrs
          .split_first_chunk::<DEGREE>()
          .expect("MAX_DEGREE covers every x86 degree");
        // SAFETY: `Lanes::select` chose this kernel only after `Lanes::available`
        // detected its target features. The caller guarantees every lane pointer
        // is readable for `blocks * BLOCK_LEN` bytes, `hash_group` bounds
        // `last_block_len` to `1..=BLOCK_LEN`, and `out` holds
        // `MAX_DEGREE >= DEGREE` outputs of `OUT_LEN` bytes.
        unsafe {
          x86_64::$module::$kernel(
            x86_64::HashManyRequest {
              inputs: lane_ptrs,
              blocks,
              key: &IV,
              counter: 0,
              increment_counter: false,
              flags: 0,
              flags_start: CHUNK_START,
              flags_end: CHUNK_END | ROOT,
              out: out.as_mut_ptr().cast::<u8>(),
            },
            last_block_len,
          )
        }
      }};
    }
    match lanes {
      Lanes::Sse41 => hash_many!(sse41::hash4_with_last_block_len),
      Lanes::Avx2 => hash_many!(avx2::hash8_owned_with_last_block_len),
      Lanes::Avx512 => hash_many!(avx512::hash16_owned_with_last_block_len),
    }
  }

  #[cfg(test)]
  mod tests {
    use super::*;

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    const ALL: &[Lanes] = &[Lanes::Sse41, Lanes::Avx2, Lanes::Avx512];
    #[cfg(target_arch = "aarch64")]
    const ALL: &[Lanes] = &[Lanes::Neon];
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable-only")))]
    const ALL: &[Lanes] = &[Lanes::Wasm];

    /// Every lane kernel the CPU supports matches the one-shot path for every
    /// one-chunk length, full and partial lane groups, and unaligned inputs.
    #[test]
    fn every_available_lane_kernel_matches_one_shot() {
      let data: alloc::vec::Vec<u8> = (0..MAX_DEGREE * (CHUNK_LEN + 1) + 1)
        .map(|i| u8::try_from(i % 251).expect("fits in u8"))
        .collect();
      for &lanes in ALL.iter().filter(|lanes| lanes.available()) {
        for len in 0..=CHUNK_LEN + 1 {
          // Offset by one byte so no input is aligned.
          let inputs: alloc::vec::Vec<&[u8]> = data[1..]
            .chunks_exact(len.max(1))
            .take(MAX_DEGREE)
            .map(|chunk| &chunk[..len])
            .collect();
          for count in [2, lanes.degree(), inputs.len().min(MAX_DEGREE)] {
            let mut outputs = alloc::vec![[0u8; OUT_LEN]; count];
            digest_batch(lanes, &inputs[..count], &mut outputs);
            for (input, output) in inputs.iter().zip(&outputs) {
              let mut iv = IV;
              assert_eq!(
                *output,
                super::super::super::digest_public_oneshot(&mut iv, 0, input),
                "{lanes:?} len={len} count={count}"
              );
            }
          }
        }
      }
    }
  }
}
