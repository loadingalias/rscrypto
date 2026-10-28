//! Owned x86-64 Montgomery arithmetic for RSA widths that are multiples of eight limbs.
//!
//! Products, squares, and Montgomery reduction run in blocks of eight rows. A row multiplies an
//! eight-limb operand chunk by the multiplier limb in RDX and adds the products into an
//! eight-limb window of the running value held in R8–R15, with two carry chains: ADCX adds each
//! product's low half and ADOX the previous product's high half. MULX and MOV leave the flags
//! untouched, so both chains run unbroken across the row. The window's lowest limb is then final
//! for the block: the row stores it and reuses its register for the row's carry-out limb, so the
//! window rotates by one register per row and returns to R8–R15 after eight rows. Between chunks
//! the window adds the next eight limbs of the running value from memory, so the running value
//! moves through memory once per eight rows instead of once per product.
//!
//! Both carry chains end every row clear. Each row still starts with an XOR zero idiom, which
//! clears CF and OF without reading them, so a row's chains wait only for the window limbs they
//! add to, not for the previous row's last carry; the rows then overlap. Without it, both chains
//! run serially through all eight rows, which measured 12–14% slower on Granite Rapids. Each row
//! stores its final limb just before its last product and loads the next multiplier right after
//! its last MULX; RBX holds the zero that the last product's two carry additions add.
//!
//! Reduction is separated from the product (SOS) and runs eight rows at a time: the block over
//! the modulus's first chunk computes each multiplier `m = w·n0 mod 2^64` from the window's
//! lowest limb and keeps the eight multipliers on the kernel's stack frame for the remaining
//! chunks. Squaring adds each cross product once, as a triangle within each eight-limb block and
//! rows against the limbs above it, then doubles the sum and adds the diagonal squares.
//!
//! Every loop bound and address depends only on the public width; secret limbs reach only
//! multiplications, carry-chained additions, and the branch-free final subtraction. Each kernel
//! clears the stack frame words that can hold secret limbs before it returns.
//!
//! Each kernel saves RBX, then keeps a 176-byte frame below it. In bytes from RSP inside the
//! kernel: 0..64 multipliers, 72 negated chunk carry, 80 top carry, 88 `n0`, then public counts
//! and pointers from 96.

use core::arch::asm;

use crate::platform::{caps, caps::x86};

/// Whether the kernels support `words` limbs on this CPU.
#[inline]
pub(super) fn supports(words: usize) -> bool {
  words >= 8 && words.is_multiple_of(8) && caps().has(x86::BMI2) && caps().has(x86::ADX)
}

/// Scratch limbs the kernels need for `words`-limb operands.
#[inline]
pub(super) fn scratch_words(words: usize) -> usize {
  words.strict_mul(2)
}

/// `$lo += low(x·rdx)` on the ADCX chain and `$hi += high(x·rdx)` on the ADOX chain for the
/// operand limb `x` at `[rsi + $off]`.
#[rustfmt::skip]
macro_rules! product {
  ($off:literal, $lo:literal, $hi:literal) => {
    concat!(
      "mulx rax, rcx, qword ptr [rsi + ", $off, "]\n",
      "adcx ", $lo, ", rcx\n",
      "adox ", $hi, ", rax\n",
    )
  };
}

/// The row's last product: its low half joins `$w7` and its high half, with both chains' carries,
/// becomes the new top limb in `$w0`, whose limb the row has already stored. `$next` runs after
/// the last MULX, which is the row's last use of RDX. Leaves CF and OF clear: the row sum fits
/// nine limbs, so the top limb cannot wrap.
#[rustfmt::skip]
macro_rules! top {
  ($w0:literal, $w7:literal) => {
    top!($w0, $w7, "")
  };
  ($w0:literal, $w7:literal, $next:literal) => {
    concat!(
      "mulx ", $w0, ", rcx, qword ptr [rsi + 56]\n",
      $next,
      "adcx ", $w7, ", rcx\n",
      "adox ", $w0, ", rbx\n",
      "adcx ", $w0, ", rbx\n",
    )
  };
}

/// One row: the window `$w0..$w7` adds the operand chunk at RSI times the multiplier in RDX and
/// stores its lowest limb at `[rdi + $m]`; `$next` loads the next row's multiplier.
#[rustfmt::skip]
macro_rules! row {
  (
    $m:literal,
    $next:literal,
    $w0:literal,
    $w1:literal,
    $w2:literal,
    $w3:literal,
    $w4:literal,
    $w5:literal,
    $w6:literal,
    $w7:literal
  ) => {
    concat!(
      "xor eax, eax\n",
      "mulx rax, rcx, qword ptr [rsi]\n",
      "adcx ", $w0, ", rcx\n",
      "adox ", $w1, ", rax\n",
      product!("8", $w1, $w2),
      product!("16", $w2, $w3),
      product!("24", $w3, $w4),
      product!("32", $w4, $w5),
      product!("40", $w5, $w6),
      product!("48", $w6, $w7),
      "mov qword ptr [rdi + ", $m, "], ", $w0, "\n",
      top!($w0, $w7, $next),
    )
  };
}

/// Eight rows with the eight multipliers at `[rsp + 0..64]`, storing eight final limbs at RDI.
/// RBX must be zero.
#[rustfmt::skip]
macro_rules! rows {
  () => {
    concat!(
      "mov rdx, qword ptr [rsp]\n",
      row!("0", "mov rdx, qword ptr [rsp + 8]\n", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"),
      row!("8", "mov rdx, qword ptr [rsp + 16]\n", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "r8"),
      row!("16", "mov rdx, qword ptr [rsp + 24]\n", "r10", "r11", "r12", "r13", "r14", "r15", "r8", "r9"),
      row!("24", "mov rdx, qword ptr [rsp + 32]\n", "r11", "r12", "r13", "r14", "r15", "r8", "r9", "r10"),
      row!("32", "mov rdx, qword ptr [rsp + 40]\n", "r12", "r13", "r14", "r15", "r8", "r9", "r10", "r11"),
      row!("40", "mov rdx, qword ptr [rsp + 48]\n", "r13", "r14", "r15", "r8", "r9", "r10", "r11", "r12"),
      row!("48", "mov rdx, qword ptr [rsp + 56]\n", "r14", "r15", "r8", "r9", "r10", "r11", "r12", "r13"),
      row!("56", "", "r15", "r8", "r9", "r10", "r11", "r12", "r13", "r14"),
    )
  };
}

/// One reduction row over the modulus's first chunk with `m` in RDX: stores `m` at `[rsp + $m]`,
/// clears the window's lowest limb, and leaves the next row's multiplier in RDX. That multiplier
/// comes from the new lowest limb as soon as the second product completes it, by a MULX that
/// leaves the flags intact; the old lowest limb's register holds it until the row ends.
#[rustfmt::skip]
macro_rules! head_row {
  (
    $m:literal,
    $w0:literal,
    $w1:literal,
    $w2:literal,
    $w3:literal,
    $w4:literal,
    $w5:literal,
    $w6:literal,
    $w7:literal
  ) => {
    concat!(
      "mov qword ptr [rsp + ", $m, "], rdx\n",
      "xor eax, eax\n",
      "mulx rax, rcx, qword ptr [rsi]\n",
      "adcx ", $w0, ", rcx\n",
      "adox ", $w1, ", rax\n",
      product!("8", $w1, $w2),
      "mov rcx, rdx\n",
      "mov rdx, qword ptr [rsp + 88]\n",
      "mulx rax, ", $w0, ", ", $w1, "\n",
      "mov rdx, rcx\n",
      product!("16", $w2, $w3),
      product!("24", $w3, $w4),
      product!("32", $w4, $w5),
      product!("40", $w5, $w6),
      product!("48", $w6, $w7),
      "mulx rax, rcx, qword ptr [rsi + 56]\n",
      "adcx ", $w7, ", rcx\n",
      "adox rax, rbx\n",
      "adcx rax, rbx\n",
      "mov rdx, ", $w0, "\n",
      "mov ", $w0, ", rax\n",
    )
  };
}

/// The eight reduction rows over the modulus's first chunk, starting with `m` in RDX. The last row
/// computes no further multiplier: the next eight limbs are not final yet.
#[rustfmt::skip]
macro_rules! head {
  () => {
    concat!(
      head_row!("0", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"),
      head_row!("8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "r8"),
      head_row!("16", "r10", "r11", "r12", "r13", "r14", "r15", "r8", "r9"),
      head_row!("24", "r11", "r12", "r13", "r14", "r15", "r8", "r9", "r10"),
      head_row!("32", "r12", "r13", "r14", "r15", "r8", "r9", "r10", "r11"),
      head_row!("40", "r13", "r14", "r15", "r8", "r9", "r10", "r11", "r12"),
      head_row!("48", "r14", "r15", "r8", "r9", "r10", "r11", "r12", "r13"),
      // Row 7: window r15, r8..r14.
      "mov qword ptr [rsp + 56], rdx\n",
      "xor eax, eax\n",
      "mulx rax, rcx, qword ptr [rsi]\n",
      "adcx r15, rcx\n",
      "adox r8, rax\n",
      product!("8", "r8", "r9"),
      product!("16", "r9", "r10"),
      product!("24", "r10", "r11"),
      product!("32", "r11", "r12"),
      product!("40", "r12", "r13"),
      product!("48", "r13", "r14"),
      top!("r15", "r14"),
    )
  };
}

/// The cross products `x_i·x_j` for `i < j` within the eight-limb block at RSI, added into the
/// window. Row `i` multiplies by `x_i` the limbs above it, which land from window limb `i + 1`
/// up, and stores the window's lowest limb at `[rdi + 8i]`. RBX must be zero.
#[rustfmt::skip]
macro_rules! triangle {
  () => {
    concat!(
      // Row 0: window r8..r15.
      "mov rdx, qword ptr [rsi]\n",
      product!("8", "r9", "r10"),
      product!("16", "r10", "r11"),
      product!("24", "r11", "r12"),
      product!("32", "r12", "r13"),
      product!("40", "r13", "r14"),
      product!("48", "r14", "r15"),
      "mov qword ptr [rdi], r8\n",
      top!("r8", "r15"),
      // Row 1: window r9..r15, r8.
      "xor eax, eax\n",
      "mov rdx, qword ptr [rsi + 8]\n",
      product!("16", "r11", "r12"),
      product!("24", "r12", "r13"),
      product!("32", "r13", "r14"),
      product!("40", "r14", "r15"),
      product!("48", "r15", "r8"),
      "mov qword ptr [rdi + 8], r9\n",
      top!("r9", "r8"),
      // Row 2: window r10..r15, r8, r9.
      "xor eax, eax\n",
      "mov rdx, qword ptr [rsi + 16]\n",
      product!("24", "r13", "r14"),
      product!("32", "r14", "r15"),
      product!("40", "r15", "r8"),
      product!("48", "r8", "r9"),
      "mov qword ptr [rdi + 16], r10\n",
      top!("r10", "r9"),
      // Row 3: window r11..r15, r8..r10.
      "xor eax, eax\n",
      "mov rdx, qword ptr [rsi + 24]\n",
      product!("32", "r15", "r8"),
      product!("40", "r8", "r9"),
      product!("48", "r9", "r10"),
      "mov qword ptr [rdi + 24], r11\n",
      top!("r11", "r10"),
      // Row 4: window r12..r15, r8..r11.
      "xor eax, eax\n",
      "mov rdx, qword ptr [rsi + 32]\n",
      product!("40", "r9", "r10"),
      product!("48", "r10", "r11"),
      "mov qword ptr [rdi + 32], r12\n",
      top!("r12", "r11"),
      // Row 5: window r13..r15, r8..r12.
      "xor eax, eax\n",
      "mov rdx, qword ptr [rsi + 40]\n",
      product!("48", "r11", "r12"),
      "mov qword ptr [rdi + 40], r13\n",
      top!("r13", "r12"),
      // Row 6: window r14, r15, r8..r13.
      "xor eax, eax\n",
      "mov rdx, qword ptr [rsi + 48]\n",
      "mov qword ptr [rdi + 48], r14\n",
      top!("r14", "r13"),
      // Row 7: nothing lands; the new top limb is zero.
      "mov qword ptr [rdi + 56], r15\n",
      "xor r15d, r15d\n",
    )
  };
}

/// Clears the window.
#[rustfmt::skip]
macro_rules! zero_window {
  () => {
    concat!(
      "xor r8d, r8d\n",
      "xor r9d, r9d\n",
      "xor r10d, r10d\n",
      "xor r11d, r11d\n",
      "xor r12d, r12d\n",
      "xor r13d, r13d\n",
      "xor r14d, r14d\n",
      "xor r15d, r15d\n",
    )
  };
}

/// Copies the eight multipliers at RAX to `[rsp + 0..64]`.
#[rustfmt::skip]
macro_rules! copy_multipliers {
  () => {
    concat!(
      "mov rcx, qword ptr [rax]\n",
      "mov qword ptr [rsp], rcx\n",
      "mov rcx, qword ptr [rax + 8]\n",
      "mov qword ptr [rsp + 8], rcx\n",
      "mov rcx, qword ptr [rax + 16]\n",
      "mov qword ptr [rsp + 16], rcx\n",
      "mov rcx, qword ptr [rax + 24]\n",
      "mov qword ptr [rsp + 24], rcx\n",
      "mov rcx, qword ptr [rax + 32]\n",
      "mov qword ptr [rsp + 32], rcx\n",
      "mov rcx, qword ptr [rax + 40]\n",
      "mov qword ptr [rsp + 40], rcx\n",
      "mov rcx, qword ptr [rax + 48]\n",
      "mov qword ptr [rsp + 48], rcx\n",
      "mov rcx, qword ptr [rax + 56]\n",
      "mov qword ptr [rsp + 56], rcx\n",
    )
  };
}

/// Adds the eight running limbs at RDI and the chunk carry to the window, saving the carry out.
#[rustfmt::skip]
macro_rules! add_running {
  () => {
    concat!(
      "mov rax, qword ptr [rsp + 72]\n",
      "neg rax\n",
      "adc r8, qword ptr [rdi]\n",
      "adc r9, qword ptr [rdi + 8]\n",
      "adc r10, qword ptr [rdi + 16]\n",
      "adc r11, qword ptr [rdi + 24]\n",
      "adc r12, qword ptr [rdi + 32]\n",
      "adc r13, qword ptr [rdi + 40]\n",
      "adc r14, qword ptr [rdi + 48]\n",
      "adc r15, qword ptr [rdi + 56]\n",
      "sbb rax, rax\n",
      "mov qword ptr [rsp + 72], rax\n",
    )
  };
}

/// Adds the chunk carry to the window and stores it at RDI as the pass's top eight limbs. The
/// pass's partial sum fits below them, so nothing carries out.
#[rustfmt::skip]
macro_rules! store_top {
  () => {
    concat!(
      "mov rax, qword ptr [rsp + 72]\n",
      "neg rax\n",
      "adc r8, 0\n",
      "adc r9, 0\n",
      "adc r10, 0\n",
      "adc r11, 0\n",
      "adc r12, 0\n",
      "adc r13, 0\n",
      "adc r14, 0\n",
      "adc r15, 0\n",
      store_window!(),
    )
  };
}

/// Stores the window at RDI.
#[rustfmt::skip]
macro_rules! store_window {
  () => {
    concat!(
      "mov qword ptr [rdi], r8\n",
      "mov qword ptr [rdi + 8], r9\n",
      "mov qword ptr [rdi + 16], r10\n",
      "mov qword ptr [rdi + 24], r11\n",
      "mov qword ptr [rdi + 32], r12\n",
      "mov qword ptr [rdi + 40], r13\n",
      "mov qword ptr [rdi + 48], r14\n",
      "mov qword ptr [rdi + 56], r15\n",
    )
  };
}

/// Clears the frame words that can hold secret limbs, then releases the frame.
#[rustfmt::skip]
macro_rules! release_frame {
  () => {
    concat!(
      "xor eax, eax\n",
      "mov qword ptr [rsp], rax\n",
      "mov qword ptr [rsp + 8], rax\n",
      "mov qword ptr [rsp + 16], rax\n",
      "mov qword ptr [rsp + 24], rax\n",
      "mov qword ptr [rsp + 32], rax\n",
      "mov qword ptr [rsp + 40], rax\n",
      "mov qword ptr [rsp + 48], rax\n",
      "mov qword ptr [rsp + 56], rax\n",
      "mov qword ptr [rsp + 72], rax\n",
      "mov qword ptr [rsp + 80], rax\n",
      "mov qword ptr [rsp + 88], rax\n",
      "add rsp, 176\n",
      "pop rbx\n",
    )
  };
}

/// One diagonal step: `t[2i..2i + 2] = 2·t[2i..2i + 2] + x_i²`, doubling on the ADOX chain and
/// adding the square on the ADCX chain.
#[rustfmt::skip]
macro_rules! square_step {
  ($x:literal, $t0:literal, $t1:literal, $a:literal, $b:literal, $lo:literal, $hi:literal) => {
    concat!(
      "mov rdx, qword ptr [rsi + ", $x, "]\n",
      "mov ", $a, ", qword ptr [rdi + ", $t0, "]\n",
      "mov ", $b, ", qword ptr [rdi + ", $t1, "]\n",
      "mulx ", $hi, ", ", $lo, ", rdx\n",
      "adox ", $a, ", ", $a, "\n",
      "adox ", $b, ", ", $b, "\n",
      "adcx ", $a, ", ", $lo, "\n",
      "adcx ", $b, ", ", $hi, "\n",
      "mov qword ptr [rdi + ", $t0, "], ", $a, "\n",
      "mov qword ptr [rdi + ", $t1, "], ", $b, "\n",
    )
  };
}

/// `t[0..2n] = a·b`.
///
/// Pass `g` adds `a·b[8g..8g + 8]` from limb `8g`. Every chunk of `a` first adds the next eight
/// running limbs to the window, which starts at zero, except in the first pass, which has no
/// running limbs yet. The pass's partial product is below `2^(64(8g + 8 + n))`, so its top eight
/// limbs at `8g + n` are fresh and nothing carries out of them.
///
/// # Safety
///
/// BMI2 and ADX must be available. `n` must be a nonzero multiple of eight, `a` and `b` valid
/// for reading `n` limbs, and `t` valid for writing `2n` limbs.
#[inline(never)]
unsafe fn product(t: *mut u64, a: *const u64, b: *const u64, n: usize) {
  // SAFETY: The caller guarantees BMI2 and ADX and the memory ranges. Pass `g` reads `b[8g..8g +
  // 8]` and `a[0..n]`, reads `t[8g..8g + n]` after earlier passes wrote it, and writes `t[8g..8g +
  // n + 8]`, inside `t[0..2n]` because `8g + 8 <= n`. The block pushes RBX and keeps its frame
  // 176 bytes below that, under the ABI-aligned stack pointer; it restores both before it ends.
  // Every branch closes a loop over a count derived from `n`.
  unsafe {
    asm!(
      "push rbx",
      "sub rsp, 176",
      "xor ebx, ebx",
      "mov qword ptr [rsp + 112], rsi",
      "mov qword ptr [rsp + 128], rdx",
      "lea rax, [8*rcx - 64]",
      "mov qword ptr [rsp + 144], rax",
      "shr rcx, 3",
      "mov qword ptr [rsp + 120], rcx",
      "dec rcx",
      "mov qword ptr [rsp + 96], rcx",
      // First pass: t[0..n + 8] = a·b[0..8].
      "mov rax, rdx",
      copy_multipliers!(),
      zero_window!(),
      "mov rcx, qword ptr [rsp + 120]",
      "mov qword ptr [rsp + 104], rcx",
      "20:",
      "xor eax, eax",
      rows!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "dec qword ptr [rsp + 104]",
      "jnz 20b",
      store_window!(),
      "jmp 22f",
      // Passes 1..n/8.
      "21:",
      "sub rdi, qword ptr [rsp + 144]",
      "mov rax, qword ptr [rsp + 128]",
      "add rax, 64",
      "mov qword ptr [rsp + 128], rax",
      copy_multipliers!(),
      zero_window!(),
      "mov rsi, qword ptr [rsp + 112]",
      "mov qword ptr [rsp + 72], 0",
      "mov rax, qword ptr [rsp + 120]",
      "mov qword ptr [rsp + 104], rax",
      "23:",
      add_running!(),
      "xor eax, eax",
      rows!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "dec qword ptr [rsp + 104]",
      "jnz 23b",
      store_top!(),
      "22:",
      "sub qword ptr [rsp + 96], 1",
      "jae 21b",
      release_frame!(),
      inout("rdi") t => _,
      inout("rsi") a => _,
      inout("rdx") b => _,
      inout("rcx") n => _,
      out("rax") _,
      out("r8") _, out("r9") _, out("r10") _, out("r11") _,
      out("r12") _, out("r13") _, out("r14") _, out("r15") _,
    );
  }
}

/// `t[0..2n] = value²`.
///
/// Pass `b` takes the block `x = value[8b..8b + 8]` as multipliers: the triangle adds the cross
/// products within `x` from limb `16b`, then each chunk above the block adds its products with `x`.
/// As in `product`, the window starts at zero and adds the running limbs before the triangle and
/// before each chunk, except in the first pass. Afterwards `t` holds `Σ value_i·value_j·2^(64(i +
/// j))` for `i < j`, below `2^(64(2n - 1))`; doubling it and adding the squares fits `2n` limbs, so
/// both final carries are zero.
///
/// # Safety
///
/// BMI2 and ADX must be available. `n` must be a nonzero multiple of eight, `value` valid for
/// reading `n` limbs, and `t` valid for reading and writing `2n` limbs.
#[inline(never)]
unsafe fn square_product(t: *mut u64, value: *const u64, n: usize) {
  // SAFETY: The caller guarantees BMI2 and ADX and the memory ranges. Pass `b` reads
  // `value[8b..n]`, reads `t[16b..8b + n]` after earlier passes wrote it, and writes `t[16b..8b +
  // n + 8]`, inside `t[0..2n]` because `8b + 8 <= n`; the passes write every limb of `t[0..2n]`
  // before the diagonal pass reads it. The block pushes RBX and keeps its frame 176 bytes below
  // that, under the ABI-aligned stack pointer; it restores both before it ends. Every branch closes a loop over a count derived from `n`; the
  // diagonal loop counts with LEA and JRCXZ, which leave the carry flags intact.
  unsafe {
    asm!(
      "push rbx",
      "sub rsp, 176",
      "xor ebx, ebx",
      "mov qword ptr [rsp + 112], rsi",
      "mov qword ptr [rsp + 128], rsi",
      "mov qword ptr [rsp + 136], rdi",
      "mov qword ptr [rsp + 160], rdi",
      "mov qword ptr [rsp + 144], rcx",
      "shr rcx, 3",
      "dec rcx",
      "mov qword ptr [rsp + 96], rcx",
      "mov qword ptr [rsp + 104], rcx",
      "dec rcx",
      "mov qword ptr [rsp + 120], rcx",
      // First pass: the triangle of value[0..8] and its rows against value[8..n].
      "mov rax, rsi",
      copy_multipliers!(),
      zero_window!(),
      "xor eax, eax",
      triangle!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "jmp 31f",
      "30:",
      "xor eax, eax",
      rows!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "31:",
      "sub qword ptr [rsp + 104], 1",
      "jae 30b",
      store_window!(),
      "jmp 33f",
      // Passes 1..n/8; [rsp + 120] holds the current pass's chunk count.
      "32:",
      "mov rax, qword ptr [rsp + 128]",
      "add rax, 64",
      "mov qword ptr [rsp + 128], rax",
      copy_multipliers!(),
      "mov rsi, rax",
      "mov rdi, qword ptr [rsp + 160]",
      "add rdi, 128",
      "mov qword ptr [rsp + 160], rdi",
      zero_window!(),
      "mov qword ptr [rsp + 72], 0",
      add_running!(),
      "xor eax, eax",
      triangle!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "mov rax, qword ptr [rsp + 120]",
      "mov qword ptr [rsp + 104], rax",
      "jmp 35f",
      "34:",
      add_running!(),
      "xor eax, eax",
      rows!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "35:",
      "sub qword ptr [rsp + 104], 1",
      "jae 34b",
      store_top!(),
      "sub qword ptr [rsp + 120], 1",
      "33:",
      "sub qword ptr [rsp + 96], 1",
      "jae 32b",
      // Diagonal pass, four limbs of `value` per iteration.
      "mov rdi, qword ptr [rsp + 136]",
      "mov rsi, qword ptr [rsp + 112]",
      "mov rcx, qword ptr [rsp + 144]",
      "shr rcx, 2",
      "xor eax, eax",
      "36:",
      square_step!("0", "0", "8", "r8", "r9", "r10", "r11"),
      square_step!("8", "16", "24", "r12", "r13", "r14", "r15"),
      square_step!("16", "32", "40", "r8", "r9", "r10", "r11"),
      square_step!("24", "48", "56", "r12", "r13", "r14", "r15"),
      "lea rsi, [rsi + 32]",
      "lea rdi, [rdi + 64]",
      "lea rcx, [rcx - 1]",
      "jrcxz 37f",
      "jmp 36b",
      "37:",
      release_frame!(),
      inout("rdi") t => _,
      inout("rsi") value => _,
      inout("rcx") n => _,
      out("rax") _,
      out("rdx") _,
      out("r8") _, out("r9") _, out("r10") _, out("r11") _,
      out("r12") _, out("r13") _, out("r14") _, out("r15") _,
    );
  }
}

/// `out = t·2^(-64·n) mod modulus` for `t < modulus·2^(64·n)` held in `t[0..2n]`; `t` is
/// overwritten.
///
/// Pass `g` reduces the eight limbs at `8g`: its first chunk derives the multipliers, and the
/// remaining chunks of the modulus apply them, each adding the next eight running limbs first.
/// The window's top eight limbs then add the running limbs at `8g + n` and the previous pass's
/// top carry; the carry out of that sum is this pass's top carry, one limb higher. The result is
/// `t[n..2n]` plus the last top carry, below `2·modulus`, and the final subtraction removes one
/// modulus when that sum reaches it.
///
/// # Safety
///
/// BMI2 and ADX must be available. `n` must be a nonzero multiple of eight, `modulus` valid for
/// reading `n` limbs, `t` valid for reading and writing `2n` limbs, and `out` valid for writing
/// `n` limbs that do not overlap `t`.
#[inline(never)]
unsafe fn reduce(out: *mut u64, t: *mut u64, modulus: *const u64, n0: u64, n: usize) {
  // SAFETY: The caller guarantees BMI2 and ADX and the memory ranges. Pass `g` reads
  // `modulus[0..n]` and reads and writes `t[8g..8g + n + 8]`, inside `t[0..2n]` because `8g + 8
  // <= n`. The final subtraction reads `t[n..2n]` and `modulus[0..n]` and writes `out[0..n]`
  // twice; `out` does not overlap `t`. The block pushes RBX and keeps its frame 176 bytes below
  // that, under the ABI-aligned stack pointer; it restores both before it ends. Every branch tests a public count; the subtraction loop
  // counts with DEC, which leaves CF intact, and selects its result with a mask.
  unsafe {
    asm!(
      "push rbx",
      "sub rsp, 176",
      "xor ebx, ebx",
      "mov qword ptr [rsp + 80], 0",
      "mov qword ptr [rsp + 88], rdx",
      "mov qword ptr [rsp + 112], rsi",
      "mov qword ptr [rsp + 136], rax",
      "mov qword ptr [rsp + 128], rcx",
      "lea rax, [8*rcx - 64]",
      "mov qword ptr [rsp + 144], rax",
      "shr rcx, 3",
      "mov qword ptr [rsp + 96], rcx",
      "dec rcx",
      "mov qword ptr [rsp + 120], rcx",
      "40:",
      "mov r8, qword ptr [rdi]",
      "mov r9, qword ptr [rdi + 8]",
      "mov r10, qword ptr [rdi + 16]",
      "mov r11, qword ptr [rdi + 24]",
      "mov r12, qword ptr [rdi + 32]",
      "mov r13, qword ptr [rdi + 40]",
      "mov r14, qword ptr [rdi + 48]",
      "mov r15, qword ptr [rdi + 56]",
      "mov rsi, qword ptr [rsp + 112]",
      "mov qword ptr [rsp + 72], 0",
      "mov rdx, qword ptr [rsp + 88]",
      "imul rdx, r8",
      "xor eax, eax",
      head!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "mov rax, qword ptr [rsp + 120]",
      "mov qword ptr [rsp + 104], rax",
      "jmp 42f",
      "41:",
      add_running!(),
      "xor eax, eax",
      rows!(),
      "lea rsi, [rsi + 64]",
      "lea rdi, [rdi + 64]",
      "42:",
      "sub qword ptr [rsp + 104], 1",
      "jae 41b",
      // Top eight limbs: window + running limbs + chunk carry, then + the previous top carry.
      "mov rax, qword ptr [rsp + 72]",
      "neg rax",
      "adc r8, qword ptr [rdi]",
      "adc r9, qword ptr [rdi + 8]",
      "adc r10, qword ptr [rdi + 16]",
      "adc r11, qword ptr [rdi + 24]",
      "adc r12, qword ptr [rdi + 32]",
      "adc r13, qword ptr [rdi + 40]",
      "adc r14, qword ptr [rdi + 48]",
      "adc r15, qword ptr [rdi + 56]",
      "mov eax, 0",
      "adc eax, 0",
      "add r8, qword ptr [rsp + 80]",
      "adc r9, 0",
      "adc r10, 0",
      "adc r11, 0",
      "adc r12, 0",
      "adc r13, 0",
      "adc r14, 0",
      "adc r15, 0",
      "adc rax, 0",
      "mov qword ptr [rsp + 80], rax",
      store_window!(),
      "sub rdi, qword ptr [rsp + 144]",
      "dec qword ptr [rsp + 96]",
      "jnz 40b",
      // RDI is at t[n]. out = t[n..2n] - modulus, eight limbs per iteration.
      "mov rsi, qword ptr [rsp + 112]",
      "mov rdx, qword ptr [rsp + 136]",
      "mov rcx, qword ptr [rsp + 128]",
      "shr rcx, 3",
      "xor eax, eax",
      "45:",
      "mov r8, qword ptr [rdi]",
      "mov r9, qword ptr [rdi + 8]",
      "mov r10, qword ptr [rdi + 16]",
      "mov r11, qword ptr [rdi + 24]",
      "mov r12, qword ptr [rdi + 32]",
      "mov r13, qword ptr [rdi + 40]",
      "mov r14, qword ptr [rdi + 48]",
      "mov r15, qword ptr [rdi + 56]",
      "sbb r8, qword ptr [rsi]",
      "sbb r9, qword ptr [rsi + 8]",
      "sbb r10, qword ptr [rsi + 16]",
      "sbb r11, qword ptr [rsi + 24]",
      "sbb r12, qword ptr [rsi + 32]",
      "sbb r13, qword ptr [rsi + 40]",
      "sbb r14, qword ptr [rsi + 48]",
      "sbb r15, qword ptr [rsi + 56]",
      "mov qword ptr [rdx], r8",
      "mov qword ptr [rdx + 8], r9",
      "mov qword ptr [rdx + 16], r10",
      "mov qword ptr [rdx + 24], r11",
      "mov qword ptr [rdx + 32], r12",
      "mov qword ptr [rdx + 40], r13",
      "mov qword ptr [rdx + 48], r14",
      "mov qword ptr [rdx + 56], r15",
      "lea rdi, [rdi + 64]",
      "lea rsi, [rsi + 64]",
      "lea rdx, [rdx + 64]",
      "dec rcx",
      "jnz 45b",
      // Keep t[n..2n] instead when it is below the modulus and the top carry is zero.
      "sbb rax, rax",
      "mov rcx, qword ptr [rsp + 80]",
      "neg rcx",
      "sbb rcx, rcx",
      "not rcx",
      "and rax, rcx",
      "mov rcx, qword ptr [rsp + 128]",
      "shl rcx, 3",
      "sub rdi, rcx",
      "sub rdx, rcx",
      "shr rcx, 5",
      "46:",
      "mov r8, qword ptr [rdx]",
      "mov r9, qword ptr [rdx + 8]",
      "mov r10, qword ptr [rdx + 16]",
      "mov r11, qword ptr [rdx + 24]",
      "mov r12, qword ptr [rdi]",
      "mov r13, qword ptr [rdi + 8]",
      "mov r14, qword ptr [rdi + 16]",
      "mov r15, qword ptr [rdi + 24]",
      "xor r12, r8",
      "xor r13, r9",
      "xor r14, r10",
      "xor r15, r11",
      "and r12, rax",
      "and r13, rax",
      "and r14, rax",
      "and r15, rax",
      "xor r8, r12",
      "xor r9, r13",
      "xor r10, r14",
      "xor r11, r15",
      "mov qword ptr [rdx], r8",
      "mov qword ptr [rdx + 8], r9",
      "mov qword ptr [rdx + 16], r10",
      "mov qword ptr [rdx + 24], r11",
      "lea rdi, [rdi + 32]",
      "lea rdx, [rdx + 32]",
      "dec rcx",
      "jnz 46b",
      release_frame!(),
      inout("rdi") t => _,
      inout("rsi") modulus => _,
      inout("rdx") n0 => _,
      inout("rcx") n => _,
      inout("rax") out => _,
      out("r8") _, out("r9") _, out("r10") _, out("r11") _,
      out("r12") _, out("r13") _, out("r14") _, out("r15") _,
    );
  }
}

/// `out = a·b·2^(-64·n) mod modulus` for `a, b < modulus`.
#[inline]
pub(super) fn mul(out: &mut [u64], a: &[u64], b: &[u64], modulus: &[u64], n0: u64, t: &mut [u64]) {
  let n = modulus.len();
  assert!(supports(n) && a.len() == n && b.len() == n && out.len() == n && t.len() >= scratch_words(n));
  // SAFETY: `supports` established BMI2 and ADX and the width; the slices cover every range the
  // kernels touch, and `out` cannot overlap the separately borrowed `t`.
  unsafe {
    product(t.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    reduce(out.as_mut_ptr(), t.as_mut_ptr(), modulus.as_ptr(), n0, n);
  }
}

/// `left = left·right·2^(-64·n) mod modulus` for `left, right < modulus`.
#[inline]
pub(super) fn mul_in_place_left(left: &mut [u64], right: &[u64], modulus: &[u64], n0: u64, t: &mut [u64]) {
  let n = modulus.len();
  assert!(supports(n) && left.len() == n && right.len() == n && t.len() >= scratch_words(n));
  // SAFETY: As for `mul`. The product no longer reads `left` when the reduction writes it.
  unsafe {
    product(t.as_mut_ptr(), left.as_ptr(), right.as_ptr(), n);
    reduce(left.as_mut_ptr(), t.as_mut_ptr(), modulus.as_ptr(), n0, n);
  }
}

/// `value = value²·2^(-64·n) mod modulus` for `value < modulus`.
#[inline]
pub(super) fn square_in_place(value: &mut [u64], modulus: &[u64], n0: u64, t: &mut [u64]) {
  let n = modulus.len();
  assert!(supports(n) && value.len() == n && t.len() >= scratch_words(n));
  // SAFETY: As for `mul`. The square no longer reads `value` when the reduction writes it.
  unsafe {
    square_product(t.as_mut_ptr(), value.as_ptr(), n);
    reduce(value.as_mut_ptr(), t.as_mut_ptr(), modulus.as_ptr(), n0, n);
  }
}
