// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// rscrypto-owned aarch64 ML-KEM SampleNTT rejection parser.
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

.section .rodata
.balign 16
.Lrscrypto_mlkem_rej_uniform_compact_table:
        .set .Lmask, 0
        .rept 256
        .set .Lpos, 0
        .set .Llane, 0
        .rept 8
        .if ((.Lmask >> .Llane) & 1)
        .byte 2 * .Llane
        .byte 2 * .Llane + 1
        .set .Lpos, .Lpos + 2
        .endif
        .set .Llane, .Llane + 1
        .endr
        .rept 16 - .Lpos
        .byte 0
        .endr
        .set .Lmask, .Lmask + 1
        .endr

.text
.balign 4

.macro SAMPLE_NTT_COMPACT_STORE candidates, out_ptr, accepted_count, table_ptr
        cmhi    v4.8h, v30.8h, \candidates\().8h
        and     v5.16b, v4.16b, v31.16b
        uaddlv  s20, v5.8h
        fmov    w12, s20
        ldr     q24, [\table_ptr, x12, lsl #4]
        cnt     v5.16b, v5.16b
        uaddlv  s20, v5.8h
        fmov    w12, s20
        tbl     \candidates\().16b, {{ \candidates\().16b }}, v24.16b
        st1     {{ \candidates\().8h }}, [\out_ptr]
        add     \out_ptr, \out_ptr, x12, lsl #1
        add     \accepted_count, \accepted_count, x12
.endm

.macro SAMPLE_NTT_DECODE_COMPACT_32 in_ptr, out_ptr, accepted_count, table_ptr
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
        SAMPLE_NTT_COMPACT_STORE v16, \out_ptr, \accepted_count, \table_ptr
        SAMPLE_NTT_COMPACT_STORE v17, \out_ptr, \accepted_count, \table_ptr
        SAMPLE_NTT_COMPACT_STORE v18, \out_ptr, \accepted_count, \table_ptr
        SAMPLE_NTT_COMPACT_STORE v19, \out_ptr, \accepted_count, \table_ptr
.endm

.macro SAMPLE_NTT_DECODE_COMPACT_16 in_ptr, out_ptr, accepted_count, table_ptr
        ld3     {{ v0.8b, v1.8b, v2.8b }}, [\in_ptr], #24
        zip1    v4.16b, v0.16b, v1.16b
        zip1    v5.16b, v1.16b, v2.16b
        bic     v4.8h, #0xf0, lsl #8
        ushr    v5.8h, v5.8h, #4
        zip1    v16.8h, v4.8h, v5.8h
        zip2    v17.8h, v4.8h, v5.8h
        SAMPLE_NTT_COMPACT_STORE v16, \out_ptr, \accepted_count, \table_ptr
        SAMPLE_NTT_COMPACT_STORE v17, \out_ptr, \accepted_count, \table_ptr
.endm

.globl rscrypto_mlkem_rej_uniform_block_aarch64_linux
.type rscrypto_mlkem_rej_uniform_block_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_block_aarch64_linux
rscrypto_mlkem_rej_uniform_block_aarch64_linux:
        mov     x9, #0
        adrp    x15, .Lrscrypto_mlkem_rej_uniform_compact_table
        add     x15, x15, :lo12:.Lrscrypto_mlkem_rej_uniform_compact_table
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
.Lsample_ntt_block_loop:
        SAMPLE_NTT_DECODE_COMPACT_32 x1, x0, x9, x15
        subs    w14, w14, #1
        b.ne    .Lsample_ntt_block_loop

        SAMPLE_NTT_DECODE_COMPACT_16 x1, x0, x9, x15

        mov     x0, x9
        ret
.size rscrypto_mlkem_rej_uniform_block_aarch64_linux, .-rscrypto_mlkem_rej_uniform_block_aarch64_linux

.globl rscrypto_mlkem_rej_uniform_block4_aarch64_linux
.type rscrypto_mlkem_rej_uniform_block4_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_block4_aarch64_linux
rscrypto_mlkem_rej_uniform_block4_aarch64_linux:
        mov     x8, #0
        mov     x9, #0
        mov     x10, #0
        mov     x11, #0
        adrp    x15, .Lrscrypto_mlkem_rej_uniform_compact_table
        add     x15, x15, :lo12:.Lrscrypto_mlkem_rej_uniform_compact_table
        mov     x12, #1
        movk    x12, #2, lsl #16
        movk    x12, #4, lsl #32
        movk    x12, #8, lsl #48
        mov     v31.d[0], x12
        mov     x12, #16
        movk    x12, #32, lsl #16
        movk    x12, #64, lsl #32
        movk    x12, #128, lsl #48
        mov     v31.d[1], x12
        mov     w12, #3329
        dup     v30.8h, w12

        mov     w14, #3
.Lsample_ntt_block4_loop:
        SAMPLE_NTT_DECODE_COMPACT_32 x1, x0, x8, x15
        SAMPLE_NTT_DECODE_COMPACT_32 x3, x2, x9, x15
        SAMPLE_NTT_DECODE_COMPACT_32 x5, x4, x10, x15
        SAMPLE_NTT_DECODE_COMPACT_32 x7, x6, x11, x15
        subs    w14, w14, #1
        b.ne    .Lsample_ntt_block4_loop

        SAMPLE_NTT_DECODE_COMPACT_16 x1, x0, x8, x15
        SAMPLE_NTT_DECODE_COMPACT_16 x3, x2, x9, x15
        SAMPLE_NTT_DECODE_COMPACT_16 x5, x4, x10, x15
        SAMPLE_NTT_DECODE_COMPACT_16 x7, x6, x11, x15

        orr     x0, x8, x9, lsl #16
        orr     x0, x0, x10, lsl #32
        orr     x0, x0, x11, lsl #48
        ret
.size rscrypto_mlkem_rej_uniform_block4_aarch64_linux, .-rscrypto_mlkem_rej_uniform_block4_aarch64_linux
