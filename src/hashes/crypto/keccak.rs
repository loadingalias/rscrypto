//! Keccak-f[1600] sponge core (internal).
//!
//! This module intentionally exposes only the minimum surface needed by SHA-3,
//! SHAKE, and SP800-185 derived constructions.

#![allow(clippy::indexing_slicing)] // Keccak state is fixed-size; indexing is audited

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[doc(hidden)]
pub(crate) mod dispatch;
#[doc(hidden)]
pub(crate) mod dispatch_tables;
#[cfg(test)]
pub(crate) mod kernel_test;
pub(crate) mod kernels;
#[cfg(target_arch = "s390x")]
pub(crate) mod s390x;

const KECCAKF_ROUNDS: usize = 24;

// Round constants.
const RC: [u64; KECCAKF_ROUNDS] = [
  0x0000_0000_0000_0001,
  0x0000_0000_0000_8082,
  0x8000_0000_0000_808a,
  0x8000_0000_8000_8000,
  0x0000_0000_0000_808b,
  0x0000_0000_8000_0001,
  0x8000_0000_8000_8081,
  0x8000_0000_0000_8009,
  0x0000_0000_0000_008a,
  0x0000_0000_0000_0088,
  0x0000_0000_8000_8009,
  0x0000_0000_8000_000a,
  0x0000_0000_8000_808b,
  0x8000_0000_0000_008b,
  0x8000_0000_0000_8089,
  0x8000_0000_0000_8003,
  0x8000_0000_0000_8002,
  0x8000_0000_0000_0080,
  0x0000_0000_0000_800a,
  0x8000_0000_8000_000a,
  0x8000_0000_8000_8081,
  0x8000_0000_0000_8080,
  0x0000_0000_8000_0001,
  0x8000_0000_8000_8008,
];

// Two platform-tuned implementations of the same Keccak-f[1600] permutation.
//
// **x86-64 / register-poor targets (≤16 GPRs):** array-based state access.
// State stays as `&mut [u64; 25]` so LLVM generates uniform `[rsp + const]`
// spill patterns. A 5-element buffer is reused for θ and χ. Serial ρ+π chain
// with hardcoded PI/RHO indices. Max ~8 simultaneous live locals.
// Measured: +28% Zen5, +30% SPR, +9% Zen4/ICL vs the named-variable version.
//
// **aarch64 / register-rich targets (≥30 GPRs):** named-variable state.
// All 25 lanes live in registers with good ILP. The 25 `b` temporaries for
// ρ+π also fit without spilling. Avoids load/store traffic that the array
// version introduces.
// Measured: 8% faster than array-based on Graviton4.

/// x86-64 / s390x / generic: array-based Keccak-f[1600].
///
/// Uses the XKCP "Bebigokimisa" lane-complementing technique: lanes
/// {1, 2, 8, 12, 17, 20} are stored in complemented form during the
/// 24 rounds, replacing 20 of 25 NOT operations per round with AND/OR.
/// Pre/post complementation costs 12 ops total, saving ~480 NOTs across
/// all rounds.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
#[allow(unused_assignments)] // final ρ+π iteration assigns `last` which is intentionally unused
pub(crate) fn keccakf_portable(state: &mut [u64; 25]) {
  // ρ+π chain: swap state[PI] with rotated last, hardcoded for constant folding.
  macro_rules! rho_pi {
    ($state:ident, $last:ident, $(($pi:expr, $rho:expr)),+ $(,)?) => {
      $(
        let tmp = $state[$pi];
        $state[$pi] = $last.rotate_left($rho);
        $last = tmp;
      )+
    };
  }

  // Lane-complementing: complement selected lanes before rounds.
  // The modified χ formulas account for this, reducing NOT count from 25 to 5
  // per round. Complement set: {1, 2, 8, 12, 17, 20}.
  state[1] = !state[1];
  state[2] = !state[2];
  state[8] = !state[8];
  state[12] = !state[12];
  state[17] = !state[17];
  state[20] = !state[20];

  // Not unrolled: ~110 ops per round × 24 = too large for L1i if unrolled.
  for &rc in &RC {
    // θ: column parity → all 5 d-values upfront (independent, max OOO parallelism).
    let c0 = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
    let c1 = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
    let c2 = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
    let c3 = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
    let c4 = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

    let d0 = c4 ^ c1.rotate_left(1);
    let d1 = c0 ^ c2.rotate_left(1);
    let d2 = c1 ^ c3.rotate_left(1);
    let d3 = c2 ^ c4.rotate_left(1);
    let d4 = c3 ^ c0.rotate_left(1);

    state[0] ^= d0;
    state[5] ^= d0;
    state[10] ^= d0;
    state[15] ^= d0;
    state[20] ^= d0;
    state[1] ^= d1;
    state[6] ^= d1;
    state[11] ^= d1;
    state[16] ^= d1;
    state[21] ^= d1;
    state[2] ^= d2;
    state[7] ^= d2;
    state[12] ^= d2;
    state[17] ^= d2;
    state[22] ^= d2;
    state[3] ^= d3;
    state[8] ^= d3;
    state[13] ^= d3;
    state[18] ^= d3;
    state[23] ^= d3;
    state[4] ^= d4;
    state[9] ^= d4;
    state[14] ^= d4;
    state[19] ^= d4;
    state[24] ^= d4;

    // ρ + π: serial chain (hardcoded PI/RHO for guaranteed constant folding)
    let mut last = state[1];
    rho_pi!(
      state,
      last,
      (10, 1),
      (7, 3),
      (11, 6),
      (17, 10),
      (18, 15),
      (3, 21),
      (5, 28),
      (16, 36),
      (8, 45),
      (21, 55),
      (24, 2),
      (4, 14),
      (15, 27),
      (23, 41),
      (19, 56),
      (13, 8),
      (12, 25),
      (2, 43),
      (20, 62),
      (14, 18),
      (22, 39),
      (9, 61),
      (6, 20),
      (1, 44),
    );

    // χ: lane-complementing formulas (XKCP "Bebigokimisa" pattern).
    // Each row uses a mix of AND/OR/NOT instead of uniform NOT-AND,
    // reducing total NOT ops from 25 to 5 per round.
    {
      let (b0, b1, b2, b3, b4) = (state[0], state[1], state[2], state[3], state[4]);
      state[0] = b0 ^ (b1 | b2);
      state[1] = b1 ^ ((!b2) | b3);
      state[2] = b2 ^ (b3 & b4);
      state[3] = b3 ^ (b4 | b0);
      state[4] = b4 ^ (b0 & b1);
    }
    {
      let (b0, b1, b2, b3, b4) = (state[5], state[6], state[7], state[8], state[9]);
      state[5] = b0 ^ (b1 | b2);
      state[6] = b1 ^ (b2 & b3);
      state[7] = b2 ^ (b3 | (!b4));
      state[8] = b3 ^ (b4 | b0);
      state[9] = b4 ^ (b0 & b1);
    }
    {
      let (b0, b1, b2, b3, b4) = (state[10], state[11], state[12], state[13], state[14]);
      let nb3 = !b3;
      state[10] = b0 ^ (b1 | b2);
      state[11] = b1 ^ (b2 & b3);
      state[12] = b2 ^ (nb3 & b4);
      state[13] = nb3 ^ (b4 | b0);
      state[14] = b4 ^ (b0 & b1);
    }
    {
      let (b0, b1, b2, b3, b4) = (state[15], state[16], state[17], state[18], state[19]);
      let nb3 = !b3;
      state[15] = b0 ^ (b1 & b2);
      state[16] = b1 ^ (b2 | b3);
      state[17] = b2 ^ (nb3 | b4);
      state[18] = nb3 ^ (b4 & b0);
      state[19] = b4 ^ (b0 | b1);
    }
    {
      let (b0, b1, b2, b3, b4) = (state[20], state[21], state[22], state[23], state[24]);
      let nb1 = !b1;
      state[20] = b0 ^ (nb1 & b2);
      state[21] = nb1 ^ (b2 | b3);
      state[22] = b2 ^ (b3 & b4);
      state[23] = b3 ^ (b4 | b0);
      state[24] = b4 ^ (b0 & b1);
    }

    // ι
    state[0] ^= rc;
  }

  // Un-complement to restore standard form.
  state[1] = !state[1];
  state[2] = !state[2];
  state[8] = !state[8];
  state[12] = !state[12];
  state[17] = !state[17];
  state[20] = !state[20];
}

// ---------------------------------------------------------------------------
// aarch64 named-variable round loop (shared by keccakf_portable and
// keccakf_absorb_portable). All 25 state variables are passed explicitly
// to avoid macro hygiene issues with module-level macros.
// ---------------------------------------------------------------------------

/// 24-round Keccak-f[1600] loop on named variables (aarch64).
///
/// All 25 mutable state variables are passed as parameters so the macro
/// works regardless of where the caller defined them (no hygiene issues).
#[cfg(target_arch = "aarch64")]
macro_rules! keccak_round_loop_aarch64 {
  ($a0:ident, $a1:ident, $a2:ident, $a3:ident, $a4:ident,
   $a5:ident, $a6:ident, $a7:ident, $a8:ident, $a9:ident,
   $a10:ident, $a11:ident, $a12:ident, $a13:ident, $a14:ident,
   $a15:ident, $a16:ident, $a17:ident, $a18:ident, $a19:ident,
   $a20:ident, $a21:ident, $a22:ident, $a23:ident, $a24:ident) => {{
    // Not unrolled: ~110 ops per round × 24 = too large for L1i if unrolled.
    for &rc in &RC {
      // θ
      let c0 = $a0 ^ $a5 ^ $a10 ^ $a15 ^ $a20;
      let c1 = $a1 ^ $a6 ^ $a11 ^ $a16 ^ $a21;
      let c2 = $a2 ^ $a7 ^ $a12 ^ $a17 ^ $a22;
      let c3 = $a3 ^ $a8 ^ $a13 ^ $a18 ^ $a23;
      let c4 = $a4 ^ $a9 ^ $a14 ^ $a19 ^ $a24;

      let d0 = c4 ^ c1.rotate_left(1);
      let d1 = c0 ^ c2.rotate_left(1);
      let d2 = c1 ^ c3.rotate_left(1);
      let d3 = c2 ^ c4.rotate_left(1);
      let d4 = c3 ^ c0.rotate_left(1);

      $a0 ^= d0;
      $a5 ^= d0;
      $a10 ^= d0;
      $a15 ^= d0;
      $a20 ^= d0;
      $a1 ^= d1;
      $a6 ^= d1;
      $a11 ^= d1;
      $a16 ^= d1;
      $a21 ^= d1;
      $a2 ^= d2;
      $a7 ^= d2;
      $a12 ^= d2;
      $a17 ^= d2;
      $a22 ^= d2;
      $a3 ^= d3;
      $a8 ^= d3;
      $a13 ^= d3;
      $a18 ^= d3;
      $a23 ^= d3;
      $a4 ^= d4;
      $a9 ^= d4;
      $a14 ^= d4;
      $a19 ^= d4;
      $a24 ^= d4;

      // ρ + π
      let b0 = $a0;
      let b10 = $a1.rotate_left(1);
      let b20 = $a2.rotate_left(62);
      let b5 = $a3.rotate_left(28);
      let b15 = $a4.rotate_left(27);
      let b16 = $a5.rotate_left(36);
      let b1 = $a6.rotate_left(44);
      let b11 = $a7.rotate_left(6);
      let b21 = $a8.rotate_left(55);
      let b6 = $a9.rotate_left(20);
      let b7 = $a10.rotate_left(3);
      let b17 = $a11.rotate_left(10);
      let b2 = $a12.rotate_left(43);
      let b12 = $a13.rotate_left(25);
      let b22 = $a14.rotate_left(39);
      let b23 = $a15.rotate_left(41);
      let b8 = $a16.rotate_left(45);
      let b18 = $a17.rotate_left(15);
      let b3 = $a18.rotate_left(21);
      let b13 = $a19.rotate_left(8);
      let b14 = $a20.rotate_left(18);
      let b24 = $a21.rotate_left(2);
      let b9 = $a22.rotate_left(61);
      let b19 = $a23.rotate_left(56);
      let b4 = $a24.rotate_left(14);

      // χ: standard a ^ (!b & c) — on aarch64, BIC (bit-clear) already
      // fuses NOT+AND into one instruction, so lane-complementing adds
      // overhead instead of saving it.
      $a0 = b0 ^ ((!b1) & b2);
      $a1 = b1 ^ ((!b2) & b3);
      $a2 = b2 ^ ((!b3) & b4);
      $a3 = b3 ^ ((!b4) & b0);
      $a4 = b4 ^ ((!b0) & b1);

      $a5 = b5 ^ ((!b6) & b7);
      $a6 = b6 ^ ((!b7) & b8);
      $a7 = b7 ^ ((!b8) & b9);
      $a8 = b8 ^ ((!b9) & b5);
      $a9 = b9 ^ ((!b5) & b6);

      $a10 = b10 ^ ((!b11) & b12);
      $a11 = b11 ^ ((!b12) & b13);
      $a12 = b12 ^ ((!b13) & b14);
      $a13 = b13 ^ ((!b14) & b10);
      $a14 = b14 ^ ((!b10) & b11);

      $a15 = b15 ^ ((!b16) & b17);
      $a16 = b16 ^ ((!b17) & b18);
      $a17 = b17 ^ ((!b18) & b19);
      $a18 = b18 ^ ((!b19) & b15);
      $a19 = b19 ^ ((!b15) & b16);

      $a20 = b20 ^ ((!b21) & b22);
      $a21 = b21 ^ ((!b22) & b23);
      $a22 = b22 ^ ((!b23) & b24);
      $a23 = b23 ^ ((!b24) & b20);
      $a24 = b24 ^ ((!b20) & b21);

      // ι
      $a0 ^= rc;
    }
  }};
}

/// aarch64: named-variable Keccak-f[1600] — all 25 lanes in registers.
///
/// ARM's 30 GPRs hold the full state + temporaries without chaotic spills.
/// The explicit `b0..b24` ρ+π temporaries enable maximum ILP from the
/// out-of-order engine. Array-based access adds unnecessary load/store
/// traffic on this register-rich architecture.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn keccakf_portable(state: &mut [u64; 25]) {
  let mut a0 = state[0];
  let mut a1 = state[1];
  let mut a2 = state[2];
  let mut a3 = state[3];
  let mut a4 = state[4];
  let mut a5 = state[5];
  let mut a6 = state[6];
  let mut a7 = state[7];
  let mut a8 = state[8];
  let mut a9 = state[9];
  let mut a10 = state[10];
  let mut a11 = state[11];
  let mut a12 = state[12];
  let mut a13 = state[13];
  let mut a14 = state[14];
  let mut a15 = state[15];
  let mut a16 = state[16];
  let mut a17 = state[17];
  let mut a18 = state[18];
  let mut a19 = state[19];
  let mut a20 = state[20];
  let mut a21 = state[21];
  let mut a22 = state[22];
  let mut a23 = state[23];
  let mut a24 = state[24];

  keccak_round_loop_aarch64!(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24
  );

  state[0] = a0;
  state[1] = a1;
  state[2] = a2;
  state[3] = a3;
  state[4] = a4;
  state[5] = a5;
  state[6] = a6;
  state[7] = a7;
  state[8] = a8;
  state[9] = a9;
  state[10] = a10;
  state[11] = a11;
  state[12] = a12;
  state[13] = a13;
  state[14] = a14;
  state[15] = a15;
  state[16] = a16;
  state[17] = a17;
  state[18] = a18;
  state[19] = a19;
  state[20] = a20;
  state[21] = a21;
  state[22] = a22;
  state[23] = a23;
  state[24] = a24;
}

/// aarch64: fused absorb + Keccak-f[1600] — XOR block data during register load.
///
/// Loads `state[i] ^ block_lane_i` directly into named register variables for
/// rate lanes, and `state[i]` for capacity lanes. This eliminates the
/// write-then-reload round-trip that separate `xor_block_into` + `keccakf_portable`
/// incurs (~34 memory ops saved per SHA3-256 block).
///
/// Since `RATE` is a const generic, `RATE / 8` is compile-time known and LLVM
/// eliminates all `if lane < lanes` branches — the result is straight-line code.
#[cfg(target_arch = "aarch64")]
#[inline]
fn keccakf_absorb_portable<const RATE: usize>(state: &mut [u64; 25], block: &[u8; RATE]) {
  debug_assert_eq!(RATE % 8, 0);
  let lanes = RATE / 8;

  #[inline(always)]
  fn read_block_lane(block: *const u8, i: usize) -> u64 {
    // SAFETY: caller guarantees `i < block.len() / 8`, so the read is in bounds.
    // `read_unaligned` supports the 1-byte alignment of `[u8; RATE]`.
    u64::from_le(unsafe { core::ptr::read_unaligned(block.cast::<u64>().add(i)) })
  }

  let ptr = block.as_ptr();

  // Fused load: `state[i] ^ block_lane_i` for absorbed (rate) lanes,
  // plain `state[i]` for capacity lanes. Each `$i` is a literal, so
  // `$i < lanes` (where `lanes = RATE / 8`, a compile-time constant)
  // is evaluated by LLVM at compile time — no runtime branches.
  macro_rules! fused_load {
    ($i:expr) => {
      if $i < lanes {
        state[$i] ^ read_block_lane(ptr, $i)
      } else {
        state[$i]
      }
    };
  }

  let mut a0 = fused_load!(0);
  let mut a1 = fused_load!(1);
  let mut a2 = fused_load!(2);
  let mut a3 = fused_load!(3);
  let mut a4 = fused_load!(4);
  let mut a5 = fused_load!(5);
  let mut a6 = fused_load!(6);
  let mut a7 = fused_load!(7);
  let mut a8 = fused_load!(8);
  let mut a9 = fused_load!(9);
  let mut a10 = fused_load!(10);
  let mut a11 = fused_load!(11);
  let mut a12 = fused_load!(12);
  let mut a13 = fused_load!(13);
  let mut a14 = fused_load!(14);
  let mut a15 = fused_load!(15);
  let mut a16 = fused_load!(16);
  let mut a17 = fused_load!(17);
  let mut a18 = fused_load!(18);
  let mut a19 = fused_load!(19);
  let mut a20 = fused_load!(20);
  let mut a21 = fused_load!(21);
  let mut a22 = fused_load!(22);
  let mut a23 = fused_load!(23);
  let mut a24 = fused_load!(24);

  keccak_round_loop_aarch64!(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24
  );

  state[0] = a0;
  state[1] = a1;
  state[2] = a2;
  state[3] = a3;
  state[4] = a4;
  state[5] = a5;
  state[6] = a6;
  state[7] = a7;
  state[8] = a8;
  state[9] = a9;
  state[10] = a10;
  state[11] = a11;
  state[12] = a12;
  state[13] = a13;
  state[14] = a14;
  state[15] = a15;
  state[16] = a16;
  state[17] = a17;
  state[18] = a18;
  state[19] = a19;
  state[20] = a20;
  state[21] = a21;
  state[22] = a22;
  state[23] = a23;
  state[24] = a24;
}

/// Fused absorb + permute: XOR a rate-sized block into state and run
/// Keccak-f[1600]. On aarch64 this loads `state[i] ^ block[i]` directly
/// into registers, eliminating the write-then-reload round-trip. On other
/// architectures the array-based permutation accesses state via memory
/// throughout, so fusion has no benefit — we fall back to separate steps.
#[inline(always)]
fn absorb_and_permute<const RATE: usize>(state: &mut [u64; 25], block: &[u8; RATE]) {
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64 as aarch64_caps;
    if crate::platform::caps().has(aarch64_caps::SHA3) {
      xor_block_into::<RATE>(state, block);
      aarch64::keccakf_aarch64_sha3_single(state);
    } else {
      keccakf_absorb_portable::<RATE>(state, block);
    }
  }
  #[cfg(not(target_arch = "aarch64"))]
  {
    xor_block_into::<RATE>(state, block);
    keccakf_portable(state);
  }
}

// ---------------------------------------------------------------------------
// Permuter trait + platform-specific implementations
// ---------------------------------------------------------------------------
//
// Direct-call permuters replace the old function-pointer dispatch. Each
// platform gets a concrete `Permuter` that calls the best kernel directly,
// allowing LLVM to inline the permutation into the absorb loop. This is the
// single most impactful change for SHA-3 throughput.
//
// - x86_64 / generic: `InlinePermuter` → `keccakf_portable`. Verified optimal — SIMD evaluation
//   (KECCAK-4) concluded no viable optimization path exists:
//   * AVX-512 χ-only: 9-38% SLOWER on Zen4/5, ICL, SPR (GPR↔SIMD crossing > VPTERNLOG savings)
//   * AVX2: worse than AVX-512 (no VPTERNLOG, 3 ops for χ vs 1)
//   * BMI2: LLVM already emits RORX; ANDN saves <5 ops/round after lane-complementing chi
//   * Full SIMD: 25 u64 lanes need 13+ YMM registers; θ/ρ/π have no efficient SIMD mapping
//   See docs/tasks/acceleration.md KECCAK-4 for full rationale.
// - aarch64: `Aarch64Permuter` → portable for single-state (the 1-state SHA3 CE kernel is ~1.9×
//   slower on Neoverse V1/V2). SHA3 CE is used only for the 2-state interleaved path
//   (`digest_pair`).
// - s390x: `S390xPermuter` → portable permutation + KIMD batch-absorb.

pub(crate) trait Permuter: Copy {
  fn permute(self, state: &mut [u64; 25], len_hint: usize);

  /// Permute two independent states in parallel.
  /// Default: two sequential single-state permutations.
  #[inline(always)]
  fn permute_x2(self, state_a: &mut [u64; 25], state_b: &mut [u64; 25], len_hint: usize) {
    self.permute(state_a, len_hint);
    self.permute(state_b, len_hint);
  }

  /// Try to batch-absorb complete rate-sized blocks via hardware instruction
  /// (e.g., s390x KIMD). Returns `true` if the hardware path was taken, in
  /// which case the caller must NOT perform the per-block XOR + permute loop.
  ///
  /// Default: returns `false` (fall back to per-block XOR + permute).
  #[inline(always)]
  fn try_absorb_blocks(self, _state: &mut [u64; 25], _blocks: &[u8], _rate: usize) -> bool {
    false
  }
}

/// Direct-call permuter using the portable scalar kernel. No function pointer
/// indirection — LLVM can inline `keccakf_portable` into the absorb loop.
#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
pub(crate) struct InlinePermuter;

impl Permuter for InlinePermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 25], _len_hint: usize) {
    keccakf_portable(state);
  }
}

/// aarch64 permuter: SHA3 CE for single-state and 2-state interleaved when
/// available, portable scalar fallback otherwise.
///
/// The single-state NEON kernel loads each lane via `vdupq_n_u64` and runs
/// 24 rounds through the shared SHA3 CE macro (EOR3/RAX1/XAR/BCAX). Both
/// elements of each `uint64x2_t` carry the same data, so lane 0 produces
/// the correct result. The 2-state kernel packs two independent states
/// lane-wise for ~2× aggregate throughput.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) struct Aarch64Permuter {
  has_sha3: bool,
}

#[cfg(target_arch = "aarch64")]
impl Default for Aarch64Permuter {
  #[inline]
  fn default() -> Self {
    Self {
      has_sha3: {
        use crate::platform::caps::aarch64 as aarch64_caps;
        crate::platform::caps().has(aarch64_caps::SHA3)
      },
    }
  }
}

#[cfg(target_arch = "aarch64")]
impl Permuter for Aarch64Permuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 25], _len_hint: usize) {
    if self.has_sha3 {
      aarch64::keccakf_aarch64_sha3_single(state);
    } else {
      keccakf_portable(state);
    }
  }

  #[inline(always)]
  fn permute_x2(self, state_a: &mut [u64; 25], state_b: &mut [u64; 25], len_hint: usize) {
    if self.has_sha3 {
      // The 2-state kernel uses both NEON lanes meaningfully (state_a in
      // lane 0, state_b in lane 1), achieving ~2× aggregate throughput.
      aarch64::keccakf_aarch64_sha3_x2(state_a, state_b);
    } else {
      self.permute(state_a, len_hint);
      self.permute(state_b, len_hint);
    }
  }
}

/// s390x permuter: portable permutation + CPACF KIMD batch-absorb.
#[cfg(target_arch = "s390x")]
#[derive(Clone, Copy)]
pub(crate) struct S390xPermuter {
  has_kimd: bool,
}

#[cfg(target_arch = "s390x")]
impl Default for S390xPermuter {
  #[inline]
  fn default() -> Self {
    Self {
      has_kimd: {
        use crate::platform::caps::s390x as s390x_caps;
        crate::platform::caps().has(s390x_caps::MSA8)
      },
    }
  }
}

#[cfg(target_arch = "s390x")]
impl Permuter for S390xPermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 25], _len_hint: usize) {
    keccakf_portable(state);
  }

  #[inline(always)]
  fn try_absorb_blocks(self, state: &mut [u64; 25], blocks: &[u8], rate: usize) -> bool {
    if self.has_kimd
      && let Some(fc) = s390x::kimd_fc_for_rate(rate)
    {
      // SAFETY: MSA8 verified at construction time (has_kimd flag).
      unsafe {
        s390x::absorb_blocks_kimd(state, blocks, fc);
      }
      return true;
    }
    false
  }
}

// ---------------------------------------------------------------------------
// Platform-specific permuter selection
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub(crate) type PlatformPermuter = Aarch64Permuter;

#[cfg(target_arch = "s390x")]
pub(crate) type PlatformPermuter = S390xPermuter;

#[cfg(not(any(target_arch = "aarch64", target_arch = "s390x")))]
pub(crate) type PlatformPermuter = InlinePermuter;

pub(crate) type KeccakCore<const RATE: usize> = KeccakCoreImpl<RATE, PlatformPermuter>;

pub(crate) type KeccakXof<const RATE: usize> = KeccakXofImpl<RATE, PlatformPermuter>;

// ---------------------------------------------------------------------------
// Sponge core
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct KeccakCoreImpl<const RATE: usize, P: Permuter> {
  state: [u64; 25],
  buf: [u8; RATE],
  buf_len: usize,
  permuter: P,
}

impl<const RATE: usize> Default for KeccakCoreImpl<RATE, PlatformPermuter> {
  #[inline]
  fn default() -> Self {
    Self {
      state: [0u64; 25],
      buf: [0u8; RATE],
      buf_len: 0,
      permuter: PlatformPermuter::default(),
    }
  }
}

// On aarch64/s390x, `PlatformPermuter` is NOT `InlinePermuter`, so the
// portable-reference type alias needs its own `Default`. On other targets
// `PlatformPermuter = InlinePermuter` and the impl above already covers it.
#[cfg(all(any(test, feature = "std"), any(target_arch = "aarch64", target_arch = "s390x")))]
impl<const RATE: usize> Default for KeccakCoreImpl<RATE, InlinePermuter> {
  #[inline]
  fn default() -> Self {
    Self {
      state: [0u64; 25],
      buf: [0u8; RATE],
      buf_len: 0,
      permuter: InlinePermuter,
    }
  }
}

impl<const RATE: usize, P: Permuter> Drop for KeccakCoreImpl<RATE, P> {
  fn drop(&mut self) {
    for word in self.state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.buf);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl<const RATE: usize, P: Permuter> KeccakCoreImpl<RATE, P> {
  #[inline(always)]
  fn absorb_block(_permuter: P, state: &mut [u64; 25], block: &[u8; RATE]) {
    absorb_and_permute::<RATE>(state, block);
  }

  pub(crate) fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

    let permuter = self.permuter;
    if self.buf_len != 0 {
      let take = core::cmp::min(RATE - self.buf_len, data.len());
      self.buf[self.buf_len..self.buf_len.strict_add(take)].copy_from_slice(&data[..take]);
      self.buf_len = self.buf_len.strict_add(take);
      data = &data[take..];

      if self.buf_len == RATE {
        let state = &mut self.state;
        let block = &self.buf;
        Self::absorb_block(permuter, state, block);
        self.buf_len = 0;
      }
    }

    let state = &mut self.state;
    let (blocks, rest) = data.as_chunks::<RATE>();
    if !blocks.is_empty() {
      let block_bytes = &data[..blocks.len().strict_mul(RATE)];
      if !permuter.try_absorb_blocks(state, block_bytes, RATE) {
        for block in blocks {
          Self::absorb_block(permuter, state, block);
        }
      }
    }
    data = rest;

    if !data.is_empty() {
      self.buf[..data.len()].copy_from_slice(data);
      self.buf_len = data.len();
    }
  }

  #[inline(always)]
  fn finalize_state(&self, ds: u8) -> [u64; 25] {
    let permuter = self.permuter;
    let mut state = self.state;
    let mut buf = self.buf;
    let buf_len = self.buf_len;

    debug_assert!(buf_len < RATE, "buf_len={} should be < RATE={}", buf_len, RATE);

    // Ensure padding happens over a zero-padded block.
    buf[buf_len..].fill(0);

    // Domain separator, then pad10*1 with final 0x80.
    // SAFETY: buf_len < RATE is guaranteed by the assertion above.
    buf[buf_len] ^= ds;
    buf[RATE - 1] ^= 0x80;

    Self::absorb_block(permuter, &mut state, &buf);
    state
  }

  pub(crate) fn finalize_into_fixed<const OUT: usize>(&self, ds: u8, out: &mut [u8; OUT]) {
    debug_assert!(OUT <= RATE);
    let state = self.finalize_state(ds);

    let (chunks, rem) = out.as_chunks_mut::<8>();
    for (chunk, &word) in chunks.iter_mut().zip(state.iter()) {
      *chunk = word.to_le_bytes();
    }
    if !rem.is_empty() {
      let bytes = state[chunks.len()].to_le_bytes();
      rem.copy_from_slice(&bytes[..rem.len()]);
    }
  }

  pub(crate) fn finalize_xof(&self, ds: u8) -> KeccakXofImpl<RATE, P> {
    let permuter = self.permuter;
    let state = self.finalize_state(ds);
    let mut buf = [0u8; RATE];
    KeccakXofImpl::<RATE, P>::fill_buf(&state, &mut buf);
    KeccakXofImpl {
      state,
      buf,
      pos: 0,
      permuter,
    }
  }
}

// ---------------------------------------------------------------------------
// Single-state oneshot (Digest::digest fast-path)
// ---------------------------------------------------------------------------

/// Hash a message in one shot, bypassing the `KeccakCoreImpl` sponge wrapper.
///
/// This avoids allocating the rate-sized buffer, eliminates the `Drop`
/// zeroization of that buffer, and removes all buffer-management conditionals.
/// The state is stack-allocated, absorbed into directly, and zeroized at the
/// end.
///
/// Uses `PlatformPermuter` for dispatch: on s390x this enables KIMD hardware
/// batch-absorb (XOR + permute in a single instruction), replacing the
/// software absorb loop for complete blocks.
#[inline]
pub(crate) fn oneshot_fixed<const RATE: usize, const OUT: usize>(ds: u8, data: &[u8]) -> [u8; OUT] {
  debug_assert!(OUT <= RATE);
  debug_assert_eq!(RATE % 8, 0);

  let permuter = PlatformPermuter::default();
  let mut state = [0u64; 25];

  // Absorb complete blocks.
  let (blocks, rest) = data.as_chunks::<RATE>();
  if !blocks.is_empty() {
    // Try hardware batch-absorb (s390x KIMD replaces both XOR + permute).
    let block_bytes = &data[..blocks.len().strict_mul(RATE)];
    if !permuter.try_absorb_blocks(&mut state, block_bytes, RATE) {
      for block in blocks {
        absorb_and_permute::<RATE>(&mut state, block);
      }
    }
  }

  // Pad the final block and absorb (direct XOR into state lanes).
  finalize_pad_absorb::<RATE>(&mut state, rest, ds, permuter);

  // Extract output.
  let mut out = [0u8; OUT];
  extract_output::<OUT>(&state, &mut out);

  // Zeroize state.
  for word in state.iter_mut() {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

  out
}

// ---------------------------------------------------------------------------
// 2-state parallel oneshot (digest_pair)
// ---------------------------------------------------------------------------

/// XOR a RATE-sized block into the Keccak state.
#[inline(always)]
fn xor_block_into<const RATE: usize>(state: &mut [u64; 25], block: &[u8; RATE]) {
  debug_assert_eq!(RATE % 8, 0);
  let lanes = RATE / 8;
  let ptr = block.as_ptr() as *const u64;
  let mut i = 0usize;
  while i < lanes {
    // SAFETY: `RATE % 8 == 0` and `i < lanes == RATE / 8`, so this reads within `block`;
    // `read_unaligned` supports the 1-byte alignment of `[u8; RATE]`.
    let v = unsafe { core::ptr::read_unaligned(ptr.add(i)) };
    state[i] ^= u64::from_le(v);
    i += 1;
  }
}

/// XOR remainder + padding into a Keccak state **without** permuting.
///
/// Used by `oneshot_pair` to pad both states independently before a single
/// `permute_x2` call (saving one permutation vs two sequential `finalize_pad_absorb`).
#[inline(always)]
fn pad_into_state<const RATE: usize>(state: &mut [u64; 25], remainder: &[u8], ds: u8) {
  debug_assert_eq!(RATE % 8, 0);
  debug_assert!(remainder.len() < RATE);

  // XOR remainder into state lane-by-lane.
  let full_lanes = remainder.len() / 8;
  let ptr = remainder.as_ptr() as *const u64;
  let mut i = 0usize;
  while i < full_lanes {
    // SAFETY: `i < full_lanes == remainder.len() / 8`, so `ptr.add(i)` reads within `remainder`.
    let v = unsafe { core::ptr::read_unaligned(ptr.add(i)) };
    state[i] ^= u64::from_le(v);
    i += 1;
  }

  // Build the partial lane: trailing remainder bytes + domain separator byte.
  let partial_start = full_lanes.strict_mul(8);
  let partial_len = remainder.len().strict_sub(partial_start);
  let mut partial = [0u8; 8];
  partial[..partial_len].copy_from_slice(&remainder[partial_start..]);
  partial[partial_len] = ds;
  state[full_lanes] ^= u64::from_le_bytes(partial);

  // pad10*1: XOR 0x80 into the last byte of the rate.
  let last_lane = (RATE - 1) / 8;
  let last_byte_pos = (RATE - 1) % 8;
  state[last_lane] ^= 0x80_u64 << last_byte_pos.strict_mul(8);
}

/// Pad and absorb the final block for a Keccak sponge.
///
/// Equivalent to [`pad_into_state`] followed by a single permutation.
#[inline(always)]
fn finalize_pad_absorb<const RATE: usize>(state: &mut [u64; 25], remainder: &[u8], ds: u8, permuter: PlatformPermuter) {
  pad_into_state::<RATE>(state, remainder, ds);
  permuter.permute(state, 0);
}

/// Extract fixed-size output from a Keccak state.
#[inline(always)]
fn extract_output<const OUT: usize>(state: &[u64; 25], out: &mut [u8; OUT]) {
  let (chunks, rem) = out.as_chunks_mut::<8>();
  for (chunk, &word) in chunks.iter_mut().zip(state.iter()) {
    *chunk = word.to_le_bytes();
  }
  if !rem.is_empty() {
    let bytes = state[chunks.len()].to_le_bytes();
    rem.copy_from_slice(&bytes[..rem.len()]);
  }
}

/// Hash two independent messages in parallel using 2-state interleaved
/// permutation (aarch64 SHA3 CE) or sequential fallback.
///
/// On aarch64 with SHA3 CE, this achieves ~2× the aggregate throughput of
/// two sequential hash computations.
pub(crate) fn oneshot_pair<const RATE: usize, const OUT: usize>(
  ds: u8,
  data_a: &[u8],
  data_b: &[u8],
) -> ([u8; OUT], [u8; OUT]) {
  debug_assert!(OUT <= RATE);
  let permuter = PlatformPermuter::default();
  let mut state_a = [0u64; 25];
  let mut state_b = [0u64; 25];

  // Split into complete blocks + remainder.
  let (blocks_a, rest_a) = data_a.as_chunks::<RATE>();
  let (blocks_b, rest_b) = data_b.as_chunks::<RATE>();
  let min_blocks = core::cmp::min(blocks_a.len(), blocks_b.len());

  // Process paired blocks using 2-state interleaved permutation.
  for i in 0..min_blocks {
    xor_block_into::<RATE>(&mut state_a, &blocks_a[i]);
    xor_block_into::<RATE>(&mut state_b, &blocks_b[i]);
    permuter.permute_x2(&mut state_a, &mut state_b, 0);
  }

  // Process remaining blocks of the longer input with single-state.
  for block in &blocks_a[min_blocks..] {
    absorb_and_permute::<RATE>(&mut state_a, block);
  }
  for block in &blocks_b[min_blocks..] {
    absorb_and_permute::<RATE>(&mut state_b, block);
  }

  // Finalize both states: pad into each, then permute both in parallel.
  pad_into_state::<RATE>(&mut state_a, rest_a, ds);
  pad_into_state::<RATE>(&mut state_b, rest_b, ds);
  permuter.permute_x2(&mut state_a, &mut state_b, 0);

  let mut out_a = [0u8; OUT];
  let mut out_b = [0u8; OUT];
  extract_output::<OUT>(&state_a, &mut out_a);
  extract_output::<OUT>(&state_b, &mut out_b);

  // Zeroize state.
  for word in state_a.iter_mut().chain(state_b.iter_mut()) {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

  (out_a, out_b)
}

#[derive(Clone)]
pub(crate) struct KeccakXofImpl<const RATE: usize, P: Permuter> {
  state: [u64; 25],
  buf: [u8; RATE],
  pos: usize,
  permuter: P,
}

impl<const RATE: usize, P: Permuter> Drop for KeccakXofImpl<RATE, P> {
  fn drop(&mut self) {
    for word in self.state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.buf);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl<const RATE: usize, P: Permuter> KeccakXofImpl<RATE, P> {
  #[inline(always)]
  fn fill_buf(state: &[u64; 25], out: &mut [u8; RATE]) {
    debug_assert_eq!(RATE % 8, 0);
    let lanes = RATE / 8;
    let mut i = 0usize;
    while i < lanes {
      let bytes = state[i].to_le_bytes();
      out[i * 8..i * 8 + 8].copy_from_slice(&bytes);
      i += 1;
    }
  }

  pub(crate) fn squeeze_into(&mut self, mut out: &mut [u8]) {
    while !out.is_empty() {
      if self.pos == RATE {
        let permuter = self.permuter;
        permuter.permute(&mut self.state, 0);
        Self::fill_buf(&self.state, &mut self.buf);
        self.pos = 0;
      }

      let take = core::cmp::min(RATE - self.pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.pos..self.pos.strict_add(take)]);
      self.pos = self.pos.strict_add(take);
      out = &mut out[take..];
    }
  }
}
