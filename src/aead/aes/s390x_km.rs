/// AES-256 key for the KM (Cipher Message) instruction.
///
/// Stores exactly the raw 32-byte parameter block consumed by KM.
#[repr(C, align(8))]
pub(in crate::aead) struct KmKey {
  /// Raw 32-byte AES-256 key, ready for the KM parameter block.
  raw: [u8; 32],
}

impl KmKey {
  /// Copy a raw AES-256 key into the KM parameter-block owner.
  pub(super) fn new(raw: &[u8; 32]) -> Self {
    Self { raw: *raw }
  }

  /// Zeroize the raw KM parameter block.
  pub(super) fn zeroize(&mut self) {
    crate::traits::ct::zeroize(&mut self.raw);
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
  // SAFETY: The caller guarantees the MSA facility before entering this raw wrapper. The parameter
  // block is the raw 32-byte AES-256 key. Source and destination are valid,
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

/// Bytes that one KM batch copies through the stack source buffer: eight AES blocks.
const KM_BATCH_BYTES: usize = 16 * 8;

/// Encrypt complete 16-byte blocks in place with KM in ECB mode.
///
/// KM requires non-overlapping source and destination, so each batch of up to
/// eight blocks is copied to a fixed stack source and encrypted back into
/// `blocks`. The batches bound that copy for every block count.
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available, that `function`
/// is the KM encrypt function code for the key, and that `parm` points to the
/// key's initialized KM parameter block.
unsafe fn encrypt_blocks_km(function: u64, parm: *const u8, blocks: &mut [u8]) {
  debug_assert!(blocks.len().is_multiple_of(16));

  let mut src = [0u8; KM_BATCH_BYTES];
  for batch in blocks.chunks_mut(KM_BATCH_BYTES) {
    let len = batch.len();
    src[..len].copy_from_slice(batch);

    let src_ptr = src.as_ptr();
    let dest_ptr = batch.as_mut_ptr();

    // SAFETY: The caller guarantees the MSA facility, the function code and its
    // parameter block. `src[..len]` and `batch` are valid, initialized,
    // non-overlapping `len`-byte spans, and KM writes exactly the bytes it reads.
    unsafe {
      core::arch::asm!(
        "0:",
        ".insn rre, 0xB92E0000, 2, 4",
        "jo 0b",
        inout("r0") function => _,
        in("r1") parm,
        inout("r2") dest_ptr => _,
        inout("r3") len as u64 => _,
        inout("r4") src_ptr => _,
        inout("r5") len as u64 => _,
        options(nostack),
      );
    }
  }

  crate::traits::ct::zeroize(&mut src);
}

/// Encrypt multiple independent 16-byte blocks using a raw AES-256 key.
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
/// `blocks` must contain exactly `count * 16` bytes.
#[cfg(feature = "aes-gcm-siv")]
pub(super) unsafe fn encrypt_blocks_raw(raw_key: &[u8; 32], blocks: &mut [u8], count: usize) {
  debug_assert_eq!(blocks.len(), count.strict_mul(16));
  // SAFETY: The caller guarantees MSA; function code 20 consumes the raw 32-byte AES-256 key.
  unsafe { encrypt_blocks_km(20, raw_key.as_ptr(), &mut blocks[..count.strict_mul(16)]) }
}

/// Encrypt a single 16-byte block using the KM instruction (AES-256 ECB).
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
pub(super) unsafe fn encrypt_block(key: &KmKey, block: &mut [u8; 16]) {
  // SAFETY: caller guarantees MSA; KmKey stores a raw 32-byte AES-256 key.
  unsafe { encrypt_block_raw(&key.raw, block) }
}

/// Encrypt multiple independent 16-byte blocks with KM (AES-256 ECB).
///
/// KM function code 20 processes each batch of up to eight contiguous blocks
/// in one instruction invocation, which covers GCM-SIV key derivation's six
/// blocks in one call.
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
/// `blocks` must contain exactly `count * 16` bytes.
pub(super) unsafe fn encrypt_blocks(key: &KmKey, blocks: &mut [u8], count: usize) {
  debug_assert_eq!(blocks.len(), count.strict_mul(16));
  // SAFETY: The caller guarantees MSA; KmKey stores the raw 32-byte AES-256 parameter block.
  unsafe { encrypt_blocks_km(20, key.raw.as_ptr(), &mut blocks[..count.strict_mul(16)]) }
}

// AES-128 (KM function code 18)

/// AES-128 key for the KM (Cipher Message) instruction.
#[repr(C, align(8))]
pub(in crate::aead) struct Km128Key {
  /// Raw 16-byte AES-128 key, ready for the KM parameter block.
  raw: [u8; 16],
}

impl Km128Key {
  /// Copy a raw AES-128 key into the KM parameter-block owner.
  pub(super) fn new(raw: &[u8; 16]) -> Self {
    Self { raw: *raw }
  }

  /// Zeroize the raw KM parameter block.
  pub(super) fn zeroize(&mut self) {
    crate::traits::ct::zeroize(&mut self.raw);
  }
}

/// Encrypt a single 16-byte block using a raw AES-128 key and the KM instruction.
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
pub(super) unsafe fn encrypt_block_raw_128(raw_key: &[u8; 16], block: &mut [u8; 16]) {
  let mut src: [u8; 16] = *block;

  let parm = raw_key.as_ptr();
  let src_ptr = src.as_ptr();
  let dest_ptr = block.as_mut_ptr();

  // KM function code 18 = AES-128 encrypt. Same encoding as AES-256
  // (.insn rre, 0xB92E0000); only the function code in r0 differs.
  //
  // SAFETY: The caller guarantees the MSA facility before entering this raw wrapper. The parameter
  // block is the raw 16-byte AES-128 key. Source and destination are valid,
  // non-overlapping 16-byte buffers.
  unsafe {
    core::arch::asm!(
      "0:",
      ".insn rre, 0xB92E0000, 2, 4",
      "jo 0b",
      inout("r0") 18u64 => _,
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

/// Encrypt multiple independent 16-byte blocks using a raw AES-128 key.
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
/// `blocks` must contain exactly `count * 16` bytes.
#[cfg(feature = "aes-gcm-siv")]
pub(super) unsafe fn encrypt_blocks_raw_128(raw_key: &[u8; 16], blocks: &mut [u8], count: usize) {
  debug_assert_eq!(blocks.len(), count.strict_mul(16));
  // SAFETY: The caller guarantees MSA; function code 18 consumes the raw 16-byte AES-128 key.
  unsafe { encrypt_blocks_km(18, raw_key.as_ptr(), &mut blocks[..count.strict_mul(16)]) }
}

/// Encrypt a single 16-byte block using the KM instruction (AES-128 ECB).
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
pub(super) unsafe fn encrypt_block_128(key: &Km128Key, block: &mut [u8; 16]) {
  // SAFETY: caller guarantees MSA; Km128Key stores a raw 16-byte AES-128 key.
  unsafe { encrypt_block_raw_128(&key.raw, block) }
}

/// Encrypt multiple independent 16-byte blocks with KM (AES-128 ECB).
///
/// # Safety
/// Caller must ensure the MSA (CPACF) facility is available.
/// `blocks` must contain exactly `count * 16` bytes.
#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "aes-siv",
  feature = "aegis256"
))]
pub(super) unsafe fn encrypt_blocks_128(key: &Km128Key, blocks: &mut [u8], count: usize) {
  debug_assert_eq!(blocks.len(), count.strict_mul(16));
  // SAFETY: The caller guarantees MSA; Km128Key stores the raw 16-byte AES-128 parameter block.
  unsafe { encrypt_blocks_km(18, key.raw.as_ptr(), &mut blocks[..count.strict_mul(16)]) }
}
