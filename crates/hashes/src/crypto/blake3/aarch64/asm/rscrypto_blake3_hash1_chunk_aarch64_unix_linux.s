

    .text
    .macro G wa, wb, wc, wd, wmx, wmy
        add \wa, \wa, \wb
        add \wa, \wa, \wmx
        eor \wd, \wd, \wa
        ror \wd, \wd, #16
        add \wc, \wc, \wd
        eor \wb, \wb, \wc
        ror \wb, \wb, #12
        add \wa, \wa, \wb
        add \wa, \wa, \wmy
        eor \wd, \wd, \wa
        ror \wd, \wd, #8
        add \wc, \wc, \wd
        eor \wb, \wb, \wc
        ror \wb, \wb, #7
    .endm

    .macro LOAD_IV
        // IV[0..4)
        movz w8,  #0xE667
        movk w8,  #0x6A09, lsl #16

        movz w9,  #0xAE85
        movk w9,  #0xBB67, lsl #16

        movz w10, #0xF372
        movk w10, #0x3C6E, lsl #16

        movz w11, #0xF53A
        movk w11, #0xA54F, lsl #16
    .endm

    .macro LOAD_MSG
        // Load 16 little-endian u32 words as 8 x-register pairs.
        // x19 = m0|m1, x20 = m2|m3, ..., x26 = m14|m15.
        ldr x19, [x16, #0]
        ldr x20, [x16, #8]
        ldr x21, [x16, #16]
        ldr x22, [x16, #24]
        ldr x23, [x16, #32]
        ldr x24, [x16, #40]
        ldr x25, [x16, #48]
        ldr x26, [x16, #56]
        add x16, x16, #64
    .endm

    // Message word mapping:
    // - Even indices (low halves):
    //   m0=w19 m2=w20 m4=w21 m6=w22 m8=w23 m10=w24 m12=w25 m14=w26
    // - Odd indices (high halves):
    //   m1=lsr x27,x19,#32 -> w27, m3=lsr x27,x20,#32 -> w27, etc.

    .macro COMPRESS_7ROUNDS
        // Round 0
        lsr x27, x19, #32
        G w0, w4, w8,  w12, w19, w27

        lsr x27, x20, #32
        G w1, w5, w9,  w13, w20, w27

        lsr x27, x21, #32
        G w2, w6, w10, w14, w21, w27

        lsr x27, x22, #32
        G w3, w7, w11, w15, w22, w27

        lsr x27, x23, #32
        G w0, w5, w10, w15, w23, w27

        lsr x27, x24, #32
        G w1, w6, w11, w12, w24, w27

        lsr x27, x25, #32
        G w2, w7, w8,  w13, w25, w27

        lsr x27, x26, #32
        G w3, w4, w9,  w14, w26, w27

        // Round 1
        G w0, w4, w8,  w12, w20, w22

        lsr x27, x20, #32
        G w1, w5, w9,  w13, w27, w24

        lsr x27, x22, #32
        G w2, w6, w10, w14, w27, w19

        lsr x27, x25, #32
        G w3, w7, w11, w15, w21, w27

        lsr x27, x19, #32
        lsr x28, x24, #32
        G w0, w5, w10, w15, w27, w28

        lsr x27, x21, #32
        G w1, w6, w11, w12, w25, w27

        lsr x27, x23, #32
        G w2, w7, w8,  w13, w27, w26

        lsr x27, x26, #32
        G w3, w4, w9,  w14, w27, w23

        // Round 2
        lsr x27, x20, #32
        G w0, w4, w8,  w12, w27, w21

        G w1, w5, w9,  w13, w24, w25

        lsr x27, x25, #32
        G w2, w6, w10, w14, w27, w20

        lsr x27, x22, #32
        G w3, w7, w11, w15, w27, w26

        lsr x27, x21, #32
        G w0, w5, w10, w15, w22, w27

        lsr x27, x23, #32
        G w1, w6, w11, w12, w27, w19

        lsr x27, x24, #32
        lsr x28, x26, #32
        G w2, w7, w8,  w13, w27, w28

        lsr x27, x19, #32
        G w3, w4, w9,  w14, w23, w27

        // Round 3
        lsr x27, x22, #32
        G w0, w4, w8,  w12, w24, w27

        lsr x27, x23, #32
        G w1, w5, w9,  w13, w25, w27

        lsr x27, x20, #32
        G w2, w6, w10, w14, w26, w27

        lsr x27, x25, #32
        lsr x28, x26, #32
        G w3, w7, w11, w15, w27, w28

        G w0, w5, w10, w15, w21, w19

        lsr x27, x24, #32
        G w1, w6, w11, w12, w27, w20

        lsr x27, x21, #32
        G w2, w7, w8,  w13, w27, w23

        lsr x27, x19, #32
        G w3, w4, w9,  w14, w27, w22

        // Round 4
        lsr x27, x25, #32
        G w0, w4, w8,  w12, w25, w27

        lsr x27, x23, #32
        lsr x28, x24, #32
        G w1, w5, w9,  w13, w27, w28

        lsr x27, x26, #32
        G w2, w6, w10, w14, w27, w24

        G w3, w7, w11, w15, w26, w23

        lsr x27, x22, #32
        G w0, w5, w10, w15, w27, w20

        lsr x27, x21, #32
        lsr x28, x20, #32
        G w1, w6, w11, w12, w27, w28

        lsr x27, x19, #32
        G w2, w7, w8,  w13, w19, w27

        G w3, w4, w9,  w14, w22, w21

        // Round 5
        lsr x27, x23, #32
        G w0, w4, w8,  w12, w27, w26

        lsr x27, x24, #32
        lsr x28, x21, #32
        G w1, w5, w9,  w13, w27, w28

        G w2, w6, w10, w14, w23, w25

        lsr x27, x26, #32
        lsr x28, x19, #32
        G w3, w7, w11, w15, w27, w28

        lsr x27, x25, #32
        lsr x28, x20, #32
        G w0, w5, w10, w15, w27, w28

        G w1, w6, w11, w12, w19, w24

        G w2, w7, w8,  w13, w20, w22

        lsr x27, x22, #32
        G w3, w4, w9,  w14, w21, w27

        // Round 6
        lsr x27, x24, #32
        lsr x28, x26, #32
        G w0, w4, w8,  w12, w27, w28

        lsr x27, x21, #32
        G w1, w5, w9,  w13, w27, w19

        lsr x27, x19, #32
        lsr x28, x23, #32
        G w2, w6, w10, w14, w27, w28

        G w3, w7, w11, w15, w23, w22

        G w0, w5, w10, w15, w26, w24

        G w1, w6, w11, w12, w20, w25

        lsr x27, x20, #32
        G w2, w7, w8,  w13, w27, w21

        lsr x27, x22, #32
        lsr x28, x25, #32
        G w3, w4, w9,  w14, w27, w28

        // XOR finalize for chaining value (first 8 words only)
        eor w0, w0, w8
        eor w1, w1, w9
        eor w2, w2, w10
        eor w3, w3, w11
        eor w4, w4, w12
        eor w5, w5, w13
        eor w6, w6, w14
        eor w7, w7, w15
    .endm

    .p2align 4
    .globl rscrypto_blake3_hash1_chunk_root_aarch64_unix_linux
rscrypto_blake3_hash1_chunk_root_aarch64_unix_linux:
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    sub sp, sp, #16
    str w2, [sp, #0]

    mov x16, x0
    mov x17, x3

    mov x20, x1

    // Load initial chaining value (key) into w0..w7.
    // Use 32-bit loads to avoid any alignment requirements.
    ldr w0, [x20, #0]
    ldr w1, [x20, #4]
    ldr w2, [x20, #8]
    ldr w3, [x20, #12]
    ldr w4, [x20, #16]
    ldr w5, [x20, #20]
    ldr w6, [x20, #24]
    ldr w7, [x20, #28]

    // Block 0 (CHUNK_START)
    LOAD_MSG
    LOAD_IV
    mov w12, wzr
    mov w13, wzr
    mov w14, #64
    ldr w15, [sp, #0]
    orr w15, w15, #1
    COMPRESS_7ROUNDS

    // Blocks 1..14
    .rept 14
        LOAD_MSG
        LOAD_IV
        mov w12, wzr
        mov w13, wzr
        mov w14, #64
        ldr w15, [sp, #0]
        COMPRESS_7ROUNDS
    .endr

    // Block 15 (CHUNK_END | ROOT)
    LOAD_MSG
    LOAD_IV
    mov w12, wzr
    mov w13, wzr
    mov w14, #64
    ldr w15, [sp, #0]
    orr w15, w15, #2
    orr w15, w15, #8
    COMPRESS_7ROUNDS

    // Store 32 output bytes (8 little-endian u32 words).
    str w0, [x17, #0]
    str w1, [x17, #4]
    str w2, [x17, #8]
    str w3, [x17, #12]
    str w4, [x17, #16]
    str w5, [x17, #20]
    str w6, [x17, #24]
    str w7, [x17, #28]

    add sp, sp, #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ret

    .p2align 4
    .globl rscrypto_blake3_hash1_chunk_cv_aarch64_unix_linux
rscrypto_blake3_hash1_chunk_cv_aarch64_unix_linux:
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    sub sp, sp, #16
    str w3, [sp, #0]
    str x2, [sp, #8]

    mov x16, x0
    mov x17, x4

    mov x20, x1

    // Load initial chaining value (key) into w0..w7.
    // Use 32-bit loads to avoid any alignment requirements.
    ldr w0, [x20, #0]
    ldr w1, [x20, #4]
    ldr w2, [x20, #8]
    ldr w3, [x20, #12]
    ldr w4, [x20, #16]
    ldr w5, [x20, #20]
    ldr w6, [x20, #24]
    ldr w7, [x20, #28]

    // Block 0 (CHUNK_START)
    LOAD_MSG
    LOAD_IV
    ldr w12, [sp, #8]
    ldr w13, [sp, #12]
    mov w14, #64
    ldr w15, [sp, #0]
    orr w15, w15, #1
    COMPRESS_7ROUNDS

    // Blocks 1..14
    .rept 14
        LOAD_MSG
        LOAD_IV
        ldr w12, [sp, #8]
        ldr w13, [sp, #12]
        mov w14, #64
        ldr w15, [sp, #0]
        COMPRESS_7ROUNDS
    .endr

    // Block 15 (CHUNK_END)
    LOAD_MSG
    LOAD_IV
    ldr w12, [sp, #8]
    ldr w13, [sp, #12]
    mov w14, #64
    ldr w15, [sp, #0]
    orr w15, w15, #2
    COMPRESS_7ROUNDS

    // Store 32 output bytes (8 little-endian u32 words).
    str w0, [x17, #0]
    str w1, [x17, #4]
    str w2, [x17, #8]
    str w3, [x17, #12]
    str w4, [x17, #16]
    str w5, [x17, #20]
    str w6, [x17, #24]
    str w7, [x17, #28]

    add sp, sp, #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ret


    .p2align 4
    .globl rscrypto_blake3_hash1_chunk_state_aarch64_unix_linux
rscrypto_blake3_hash1_chunk_state_aarch64_unix_linux:
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    sub sp, sp, #32
    str w3, [sp, #0]
    str x5, [sp, #8]
    str x2, [sp, #16]

    mov x16, x0
    mov x17, x4

    mov x20, x1

    // Load initial chaining value (key) into w0..w7.
    // Note: do not load from x1 directly, because loading w1 would clobber x1.
    ldr w0, [x20, #0]
    ldr w1, [x20, #4]
    ldr w2, [x20, #8]
    ldr w3, [x20, #12]
    ldr w4, [x20, #16]
    ldr w5, [x20, #20]
    ldr w6, [x20, #24]
    ldr w7, [x20, #28]

    // Blocks 0..14 (no CHUNK_END)
    // Block 0 (CHUNK_START)
    LOAD_MSG
    LOAD_IV
    ldr w12, [sp, #16]
    ldr w13, [sp, #20]
    mov w14, #64
    ldr w15, [sp, #0]
    orr w15, w15, #1
    COMPRESS_7ROUNDS

    // Blocks 1..14
    .rept 14
        LOAD_MSG
        LOAD_IV
        ldr w12, [sp, #16]
        ldr w13, [sp, #20]
        mov w14, #64
        ldr w15, [sp, #0]
        COMPRESS_7ROUNDS
    .endr

    // Store the chaining value *before* the final chunk block.
    str w0, [x17, #0]
    str w1, [x17, #4]
    str w2, [x17, #8]
    str w3, [x17, #12]
    str w4, [x17, #16]
    str w5, [x17, #20]
    str w6, [x17, #24]
    str w7, [x17, #28]

    // Store the final chunk block bytes (64B) without compressing it.
    // At this point, x16 points to input + 15*64.
    ldr x19, [x16, #0]
    ldr x20, [x16, #8]
    ldr x21, [x16, #16]
    ldr x22, [x16, #24]
    ldr x23, [x16, #32]
    ldr x24, [x16, #40]
    ldr x25, [x16, #48]
    ldr x26, [x16, #56]

    ldr x10, [sp, #8]

    str x19, [x10, #0]
    str x20, [x10, #8]
    str x21, [x10, #16]
    str x22, [x10, #24]
    str x23, [x10, #32]
    str x24, [x10, #40]
    str x25, [x10, #48]
    str x26, [x10, #56]

    add sp, sp, #32

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ret


    // Compress `num_blocks` full 64-byte blocks into an existing chunk chaining
    // value, updating `*blocks_compressed`.
    //
    // Signature:
    //   void rscrypto_blake3_chunk_compress_blocks_aarch64_unix_linux(
    //     const uint8_t* blocks,        // x0
    //     uint32_t*      chaining_value,// x1 (8 u32 words, in/out)
    //     uint64_t       chunk_counter, // x2
    //     uint32_t       flags,         // w3
    //     uint8_t*       blocks_comp,   // x4 (in/out)
    //     size_t         num_blocks     // x5
    //   );
    //
    // Notes:
    // - Applies CHUNK_START (bit 0) iff *blocks_comp == 0 for the first block.
    // - Does NOT apply CHUNK_END; callers keep the final chunk block buffered.
    .p2align 4
    .globl rscrypto_blake3_chunk_compress_blocks_aarch64_unix_linux
rscrypto_blake3_chunk_compress_blocks_aarch64_unix_linux:
    // NOTE: This function can't keep pointers/counters in x0..x7, because
    // COMPRESS_7ROUNDS uses w0..w15 heavily, and writing to wN clobbers xN.

    stp x29, x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    // Stack locals:
    // - [sp, #0]: flags (u32)
    // - [sp, #8]: chunk_counter (u64)
    sub sp, sp, #16
    str w3, [sp, #0]
    str x2, [sp, #8]

    // Move pointers/counters to safe registers (not overlapped by w0..w15).
    mov x16, x0        // message pointer for LOAD_MSG
    mov x20, x1         // chaining value pointer
    mov x29, x4        // blocks_compressed pointer (in/out)
    mov x30, x5        // num_blocks loop counter

    // Early out for num_blocks == 0.
    cbz x30, 3f

    // Load chaining value into w0..w7.
    ldr w0, [x20, #0]
    ldr w1, [x20, #4]
    ldr w2, [x20, #8]
    ldr w3, [x20, #12]
    ldr w4, [x20, #16]
    ldr w5, [x20, #20]
    ldr w6, [x20, #24]
    ldr w7, [x20, #28]

    // Load blocks_compressed (u8) into w17.
    ldrb w17, [x29]

1:
    LOAD_MSG
    LOAD_IV

    // Reload row3 for each compression (compression mutates w12..w15).
    ldr w12, [sp, #8]
    ldr w13, [sp, #12]
    mov w14, #64
    ldr w15, [sp, #0]

    // Apply CHUNK_START on the first block of the chunk.
    cbnz w17, 2f
    orr w15, w15, #1
2:
    COMPRESS_7ROUNDS

    // blocks_compressed++
    add w17, w17, #1

    // Loop.
    subs x30, x30, #1
    bne 1b

    // Store updated chaining value and blocks_compressed.
    str w0, [x20, #0]
    str w1, [x20, #4]
    str w2, [x20, #8]
    str w3, [x20, #12]
    str w4, [x20, #16]
    str w5, [x20, #20]
    str w6, [x20, #24]
    str w7, [x20, #28]
    strb w17, [x29]

3:
    add sp, sp, #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret
