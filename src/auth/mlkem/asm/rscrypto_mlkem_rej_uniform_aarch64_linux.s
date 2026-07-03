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
// Bounded ABI:
//   x0: uint16_t out[cap] write-only accepted candidates
//   x1: const uint8_t input[168] read-only SHAKE128 rate block
//   x2: output capacity in accepted candidates
// Returns:
//   x0: accepted candidate count, capped at x2
//
// Three-block ABI:
//   x0: uint16_t out[256] write-only accepted candidates
//   x1: const uint8_t input[3 * 168] read-only contiguous SHAKE128 rate blocks
// Returns:
//   x0: accepted candidate count, capped at 256
//
// Triple-block ABI:
//   x0: uint16_t out0[112] write-only accepted candidates
//   x1: const uint8_t input0[168] read-only SHAKE128 rate block
//   x2: uint16_t out1[112] write-only accepted candidates
//   x3: const uint8_t input1[168] read-only SHAKE128 rate block
//   x4: uint16_t out2[112] write-only accepted candidates
//   x5: const uint8_t input2[168] read-only SHAKE128 rate block
// Returns:
//   x0: count0 | (count1 << 16) | (count2 << 32)
//
// Bounded triple-block ABI:
//   x0: uint16_t out0[cap0] write-only accepted candidates
//   x1: const uint8_t input0[168] read-only SHAKE128 rate block
//   x2: uint16_t out1[cap1] write-only accepted candidates
//   x3: const uint8_t input1[168] read-only SHAKE128 rate block
//   x4: uint16_t out2[cap2] write-only accepted candidates
//   x5: const uint8_t input2[168] read-only SHAKE128 rate block
//   x6: const size_t caps[3] = (cap0, cap1, cap2)
// Returns:
//   x0: count0 | (count1 << 16) | (count2 << 32)
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

.macro SAMPLE_NTT_SETUP_CONSTANTS
        adrp    x3, .Lrscrypto_mlkem_rej_uniform_compact_table
        add     x3, x3, :lo12:.Lrscrypto_mlkem_rej_uniform_compact_table
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
.endm

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

.macro SAMPLE_NTT_SCALAR_STORE_LANE candidates, lane, done
        umov    w15, \candidates\().h[\lane]
        cmp     w15, w8
        b.hs    1f
        strh    w15, [x0], #2
        add     x9, x9, #1
        cmp     x9, x2
        b.hs    \done
1:
.endm

.macro SAMPLE_NTT_SCALAR_STORE candidates, done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 0, \done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 1, \done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 2, \done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 3, \done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 4, \done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 5, \done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 6, \done
        SAMPLE_NTT_SCALAR_STORE_LANE \candidates, 7, \done
.endm

.macro SAMPLE_NTT_COMPACT_STORE_BOUNDED candidates, done
        // Bounded tail parsing must not use the compact eight-lane store: it
        // writes slack lanes beyond the accepted count, which is safe only for
        // full-capacity scratch buffers. Store accepted lanes exactly here.
        SAMPLE_NTT_SCALAR_STORE \candidates, \done
.endm

.macro SAMPLE_NTT_DECODE_COMPACT_32_BOUNDED in_ptr, done
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
        SAMPLE_NTT_COMPACT_STORE_BOUNDED v16, \done
        SAMPLE_NTT_COMPACT_STORE_BOUNDED v17, \done
        SAMPLE_NTT_COMPACT_STORE_BOUNDED v18, \done
        SAMPLE_NTT_COMPACT_STORE_BOUNDED v19, \done
.endm

.macro SAMPLE_NTT_DECODE_COMPACT_16_BOUNDED in_ptr, done
        ld3     {{ v0.8b, v1.8b, v2.8b }}, [\in_ptr], #24
        zip1    v4.16b, v0.16b, v1.16b
        zip1    v5.16b, v1.16b, v2.16b
        bic     v4.8h, #0xf0, lsl #8
        ushr    v5.8h, v5.8h, #4
        zip1    v16.8h, v4.8h, v5.8h
        zip2    v17.8h, v4.8h, v5.8h
        SAMPLE_NTT_COMPACT_STORE_BOUNDED v16, \done
        SAMPLE_NTT_COMPACT_STORE_BOUNDED v17, \done
.endm

.macro SAMPLE_NTT_PARSE_BLOCK_FULL out_reg, in_reg, count_reg, loop_label
        mov     x0, \out_reg
        mov     x13, \in_reg
        mov     x9, #0

        mov     w14, #3
\loop_label:
        SAMPLE_NTT_DECODE_COMPACT_32 x13
        subs    w14, w14, #1
        b.ne    \loop_label

        SAMPLE_NTT_DECODE_COMPACT_16 x13
        mov     \count_reg, x9
.endm

.macro SAMPLE_NTT_PARSE_BLOCK_BOUNDED out_reg, in_reg, cap_reg, count_reg, loop_label, done_label
        mov     x0, \out_reg
        mov     x13, \in_reg
        mov     x2, \cap_reg
        mov     x9, #0
        cbz     x2, \done_label

        mov     w14, #3
\loop_label:
        SAMPLE_NTT_DECODE_COMPACT_32_BOUNDED x13, \done_label
        subs    w14, w14, #1
        b.ne    \loop_label

        SAMPLE_NTT_DECODE_COMPACT_16_BOUNDED x13, \done_label
\done_label:
        mov     \count_reg, x9
.endm

.globl rscrypto_mlkem_rej_uniform_block_aarch64_linux
.type rscrypto_mlkem_rej_uniform_block_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_block_aarch64_linux
rscrypto_mlkem_rej_uniform_block_aarch64_linux:
        mov     x13, x1
        mov     x9, #0
        SAMPLE_NTT_SETUP_CONSTANTS

        mov     w14, #3
.Lsample_ntt_block_loop:
        SAMPLE_NTT_DECODE_COMPACT_32 x13
        subs    w14, w14, #1
        b.ne    .Lsample_ntt_block_loop

        SAMPLE_NTT_DECODE_COMPACT_16 x13

        mov     x0, x9
        ret
.size rscrypto_mlkem_rej_uniform_block_aarch64_linux, .-rscrypto_mlkem_rej_uniform_block_aarch64_linux

.globl rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux
.type rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux
rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux:
        mov     x9, #0
        cbz     x2, .Lsample_ntt_bounded_done
        mov     x13, x1
        SAMPLE_NTT_SETUP_CONSTANTS

        mov     w14, #3
.Lsample_ntt_bounded_block_loop:
        SAMPLE_NTT_DECODE_COMPACT_32_BOUNDED x13, .Lsample_ntt_bounded_done
        subs    w14, w14, #1
        b.ne    .Lsample_ntt_bounded_block_loop

        SAMPLE_NTT_DECODE_COMPACT_16_BOUNDED x13, .Lsample_ntt_bounded_done

.Lsample_ntt_bounded_done:
        mov     x0, x9
        ret
.size rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux, .-rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux

.globl rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux
.type rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux
rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux:
        mov     x9, #0
        mov     x2, #256
        mov     x13, x1
        SAMPLE_NTT_SETUP_CONSTANTS

        mov     w14, #6
.Lsample_ntt_3blocks_full_loop:
        SAMPLE_NTT_DECODE_COMPACT_32 x13
        subs    w14, w14, #1
        b.ne    .Lsample_ntt_3blocks_full_loop

        SAMPLE_NTT_DECODE_COMPACT_16 x13
        SAMPLE_NTT_DECODE_COMPACT_16 x13

        mov     w14, #3
.Lsample_ntt_3blocks_tail_loop:
        SAMPLE_NTT_DECODE_COMPACT_32_BOUNDED x13, .Lsample_ntt_3blocks_done
        subs    w14, w14, #1
        b.ne    .Lsample_ntt_3blocks_tail_loop

        SAMPLE_NTT_DECODE_COMPACT_16_BOUNDED x13, .Lsample_ntt_3blocks_done

.Lsample_ntt_3blocks_done:
        mov     x0, x9
        ret
.size rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux, .-rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux

.globl rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux
.type rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux
rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux:
        mov     x16, x0
        mov     x17, x1
        mov     x10, x2
        mov     x11, x3
        mov     x6, x4
        mov     x7, x5
        SAMPLE_NTT_SETUP_CONSTANTS

        SAMPLE_NTT_PARSE_BLOCK_FULL x16, x17, x1, .Lsample_ntt_triple_block_lane0_loop
        SAMPLE_NTT_PARSE_BLOCK_FULL x10, x11, x2, .Lsample_ntt_triple_block_lane1_loop
        SAMPLE_NTT_PARSE_BLOCK_FULL x6, x7, x4, .Lsample_ntt_triple_block_lane2_loop

        and     x1, x1, #0xffff
        and     x2, x2, #0xffff
        and     x4, x4, #0xffff
        orr     x0, x1, x2, lsl #16
        orr     x0, x0, x4, lsl #32
        ret
.size rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux, .-rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux

.globl rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux
.type rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux
rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux:
        mov     x16, x0
        mov     x17, x1
        mov     x11, x2
        mov     x1, x3
        SAMPLE_NTT_SETUP_CONSTANTS

        ldr     x2, [x6]
        SAMPLE_NTT_PARSE_BLOCK_BOUNDED x16, x17, x2, x7, .Lsample_ntt_triple_bounded_lane0_loop, .Lsample_ntt_triple_bounded_lane0_done
        ldr     x2, [x6, #8]
        SAMPLE_NTT_PARSE_BLOCK_BOUNDED x11, x1, x2, x16, .Lsample_ntt_triple_bounded_lane1_loop, .Lsample_ntt_triple_bounded_lane1_done
        ldr     x2, [x6, #16]
        SAMPLE_NTT_PARSE_BLOCK_BOUNDED x4, x5, x2, x17, .Lsample_ntt_triple_bounded_lane2_loop, .Lsample_ntt_triple_bounded_lane2_done

        and     x7, x7, #0xffff
        and     x16, x16, #0xffff
        and     x17, x17, #0xffff
        orr     x0, x7, x16, lsl #16
        orr     x0, x0, x17, lsl #32
        ret
.size rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux, .-rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux
