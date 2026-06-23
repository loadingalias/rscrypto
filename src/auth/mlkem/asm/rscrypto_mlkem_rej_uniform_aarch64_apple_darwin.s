// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// rscrypto-owned aarch64 ML-KEM SampleNTT rejection parser for Darwin.
//
// ABI:
//   x0: uint16_t out[112] write-only accepted candidates
//   x1: const uint8_t input[168] read-only SHAKE128 rate block
// Returns:
//   x0: accepted candidate count
//
// The input is the public matrix-A XOF stream. Rejection branches and write
// positions therefore depend only on public bytes, never secret key, noise,
// message, or shared-secret material.

.text
.balign 4

.macro SAMPLE_NTT_COMPACT_STORE candidates
        cmhi    v4.8h, v30.8h, \candidates\().8h
        and     v5.16b, v4.16b, v31.16b
        uaddlv  s20, v5.8h
        fmov    w12, s20
        ldr     q24, [x3, x12, lsl #4]
        cnt     v5.16b, v5.16b
        uaddlv  s20, v5.8h
        fmov    w12, s20
        tbl     \candidates\().16b, {{ \candidates\().16b }}, v24.16b
        st1     {{ \candidates\().8h }}, [x0]
        add     x0, x0, x12, lsl #1
        add     x9, x9, x12
.endm

.macro SAMPLE_NTT_DECODE_COMPACT_32 in_ptr
        ld3     {{ v0.16b, v1.16b, v2.16b }}, [\in_ptr], #48
        zip1    v4.16b, v0.16b, v1.16b
        zip2    v5.16b, v0.16b, v1.16b
        zip1    v6.16b, v1.16b, v2.16b
        zip2    v7.16b, v1.16b, v2.16b
        bic     v4.8h, #0xf0, lsl #8
        bic     v5.8h, #0xf0, lsl #8
        ushr    v6.8h, v6.8h, #4
        ushr    v7.8h, v7.8h, #4
        zip1    v16.8h, v4.8h, v6.8h
        zip2    v17.8h, v4.8h, v6.8h
        zip1    v18.8h, v5.8h, v7.8h
        zip2    v19.8h, v5.8h, v7.8h
        SAMPLE_NTT_COMPACT_STORE v16
        SAMPLE_NTT_COMPACT_STORE v17
        SAMPLE_NTT_COMPACT_STORE v18
        SAMPLE_NTT_COMPACT_STORE v19
.endm

.macro SAMPLE_NTT_DECODE_COMPACT_16 in_ptr
        ld3     {{ v0.8b, v1.8b, v2.8b }}, [\in_ptr], #24
        zip1    v4.16b, v0.16b, v1.16b
        zip1    v5.16b, v1.16b, v2.16b
        bic     v4.8h, #0xf0, lsl #8
        ushr    v5.8h, v5.8h, #4
        zip1    v16.8h, v4.8h, v5.8h
        zip2    v17.8h, v4.8h, v5.8h
        SAMPLE_NTT_COMPACT_STORE v16
        SAMPLE_NTT_COMPACT_STORE v17
.endm

.globl _rscrypto_mlkem_rej_uniform_block_aarch64_apple_darwin
.private_extern _rscrypto_mlkem_rej_uniform_block_aarch64_apple_darwin
_rscrypto_mlkem_rej_uniform_block_aarch64_apple_darwin:
        mov     x13, x1
        mov     x9, #0
        adr     x3, Lrscrypto_mlkem_rej_uniform_compact_table
        mov     x8, #1
        movk    x8, #2, lsl #16
        movk    x8, #4, lsl #32
        movk    x8, #8, lsl #48
        mov     v31.d[0], x8
        mov     x8, #16
        movk    x8, #32, lsl #16
        movk    x8, #64, lsl #32
        movk    x8, #128, lsl #48
        mov     v31.d[1], x8
        mov     w8, #3329
        dup     v30.8h, w8

        mov     w14, #3
Lsample_ntt_block_loop:
        SAMPLE_NTT_DECODE_COMPACT_32 x13
        subs    w14, w14, #1
        b.ne    Lsample_ntt_block_loop

        SAMPLE_NTT_DECODE_COMPACT_16 x13

        mov     x0, x9
        ret

.balign 16
Lrscrypto_mlkem_rej_uniform_compact_table:
        .set Lmask, 0
        .rept 256
        .set Lpos, 0
        .set Llane, 0
        .rept 8
        .if ((Lmask >> Llane) & 1)
        .byte 2 * Llane
        .byte 2 * Llane + 1
        .set Lpos, Lpos + 2
        .endif
        .set Llane, Llane + 1
        .endr
        .rept 16 - Lpos
        .byte 0
        .endr
        .set Lmask, Lmask + 1
        .endr
