use super::EXPANDED_KEY_WORDS;

/// AES-256 key for the KM (Cipher Message) instruction.
///
/// Caches the raw 32-byte key (extracted once at key-expansion time)
/// alongside the full expanded schedule. This avoids 8 BE serializations
/// per `encrypt_block` call — critical for GCM-SIV which does 7 AES
/// calls for even a 0-byte message.
#[derive(Clone)]
#[repr(C, align(8))]
pub(in crate::aead) struct KmKey {
  /// Raw 32-byte AES-256 key, ready for the KM parameter block.
  raw: [u8; 32],
  /// Full expanded schedule (kept for potential future use / uniform sizing).
  rk: [u32; EXPANDED_KEY_WORDS],
}

impl KmKey {
  /// Wrap an already-expanded portable round key schedule for KM.
  pub(super) fn from_portable(rk: [u32; EXPANDED_KEY_WORDS]) -> Self {
    // Extract the raw 32-byte key from the first 8 big-endian u32 words.
    let mut raw = [0u8; 32];
    let mut i = 0usize;
    while i < 8 {
      let off = i.strict_mul(4);
      let bytes = rk[i].to_be_bytes();
      raw[off] = bytes[0];
      raw[off.strict_add(1)] = bytes[1];
      raw[off.strict_add(2)] = bytes[2];
      raw[off.strict_add(3)] = bytes[3];
      i = i.strict_add(1);
    }
    Self { raw, rk }
  }

  /// Zeroize both the raw key and the full key schedule.
  pub(super) fn zeroize(&mut self) {
    crate::traits::ct::zeroize(&mut self.raw);
    // SAFETY: [u32; 60] is layout-compatible with [u8; 240].
    let bytes =
      unsafe { core::slice::from_raw_parts_mut(self.rk.as_mut_ptr().cast::<u8>(), EXPANDED_KEY_WORDS.strict_mul(4)) };
    crate::traits::ct::zeroize(bytes);
  }
}

/// Encrypt a single 16-byte block using a raw AES-256 key and the KM instruction.
///
/// This avoids rebuilding a portable key schedule when the caller already has
/// the derived 32-byte key material and only needs KM's raw parameter block.
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
pub(super) unsafe fn encrypt_block_raw(raw_key: &[u8; 32], block: &mut [u8; 16]) {
  // KM requires non-overlapping source and destination. Copy the
  // plaintext to a stack buffer, encrypt from there into `block`.
  let mut src: [u8; 16] = *block;

  let parm = raw_key.as_ptr();
  let src_ptr = src.as_ptr();
  let dest_ptr = block.as_mut_ptr();

  // KM function code 20 = AES-256 encrypt.
  //
  // Instruction: .insn rre, 0xB92E0000, R1, R2
  //   R0  = function code (20)
  //   R1  = parameter block pointer (32-byte key)
  //   R2  = destination address (updated)
  //   R4  = source address (updated)
  //   R5  = source length in bytes (decremented)
  //
  // CC=0: complete. CC=2: partial (kernel preemption) - retry.
  //
  // SAFETY: MSA verified by caller. Parameter block is the raw
  // 32-byte AES-256 key. Source and destination are valid,
  // non-overlapping 16-byte buffers.
  unsafe {
    core::arch::asm!(
      "0:",
      ".insn rre, 0xB92E0000, 2, 4",
      "jo 0b",
      inout("r0") 20u64 => _,
      in("r1") parm,
      inout("r2") dest_ptr => _,
      inout("r3") 16u64 => _,
      inout("r4") src_ptr => _,
      inout("r5") 16u64 => _,
      options(nostack),
    );
  }

  crate::traits::ct::zeroize(&mut src);
}

/// Encrypt a single 16-byte block using the KM instruction (AES-256 ECB).
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
pub(super) unsafe fn encrypt_block(key: &KmKey, block: &mut [u8; 16]) {
  // SAFETY: caller guarantees MSA; KmKey stores a raw 32-byte AES-256 key.
  unsafe { encrypt_block_raw(&key.raw, block) }
}

/// Encrypt multiple independent 16-byte blocks using a single KM call.
///
/// KM in ECB mode (function code 20) processes `count` contiguous blocks
/// in one instruction invocation. This is critical for GCM-SIV key
/// derivation which requires 6 independent AES-ECB encryptions.
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
/// `blocks` must contain exactly `count * 16` bytes.
pub(super) unsafe fn encrypt_blocks(key: &KmKey, blocks: &mut [u8], count: usize) {
  debug_assert_eq!(blocks.len(), count.strict_mul(16));

  // KM requires non-overlapping source and destination.
  // Allocate a stack buffer for the source copy.
  let len = count.strict_mul(16);
  let mut src = [0u8; 16 * 8]; // max 8 blocks (128 bytes), enough for GCM-SIV's 6
  src[..len].copy_from_slice(&blocks[..len]);

  let parm = key.raw.as_ptr();
  let src_ptr = src.as_ptr();
  let dest_ptr = blocks.as_mut_ptr();

  // SAFETY: Same as encrypt_block. len = count*16 bytes, all within bounds.
  unsafe {
    core::arch::asm!(
      "0:",
      ".insn rre, 0xB92E0000, 2, 4",
      "jo 0b",
      inout("r0") 20u64 => _,
      in("r1") parm,
      inout("r2") dest_ptr => _,
      inout("r3") len as u64 => _,
      inout("r4") src_ptr => _,
      inout("r5") len as u64 => _,
      options(nostack),
    );
  }

  crate::traits::ct::zeroize(&mut src[..len]);
}
