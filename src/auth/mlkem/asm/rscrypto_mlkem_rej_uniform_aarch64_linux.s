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

.text
.balign 4

.macro SAMPLE_NTT_DECODE_32 in_ptr, scratch
        ld3     {{ v0.16b, v1.16b, v2.16b }}, [\in_ptr], #48

        ushll   v4.8h, v0.8b, #0
        ushll   v5.8h, v1.8b, #0
        ushll   v6.8h, v2.8b, #0
        and     v7.16b, v5.16b, v29.16b
        shl     v7.8h, v7.8h, #8
        orr     v16.16b, v4.16b, v7.16b
        ushr    v17.8h, v5.8h, #4
        shl     v6.8h, v6.8h, #4
        orr     v17.16b, v17.16b, v6.16b

        ushll2  v4.8h, v0.16b, #0
        ushll2  v5.8h, v1.16b, #0
        ushll2  v6.8h, v2.16b, #0
        and     v7.16b, v5.16b, v29.16b
        shl     v7.8h, v7.8h, #8
        orr     v18.16b, v4.16b, v7.16b
        ushr    v19.8h, v5.8h, #4
        shl     v6.8h, v6.8h, #4
        orr     v19.16b, v19.16b, v6.16b

        st2     {{ v16.8h, v17.8h }}, [\scratch], #32
        st2     {{ v18.8h, v19.8h }}, [\scratch]
.endm

.macro SAMPLE_NTT_DECODE_16 in_ptr, scratch
        ld3     {{ v0.8b, v1.8b, v2.8b }}, [\in_ptr], #24

        ushll   v4.8h, v0.8b, #0
        ushll   v5.8h, v1.8b, #0
        ushll   v6.8h, v2.8b, #0
        and     v7.16b, v5.16b, v29.16b
        shl     v7.8h, v7.8h, #8
        orr     v16.16b, v4.16b, v7.16b
        ushr    v17.8h, v5.8h, #4
        shl     v6.8h, v6.8h, #4
        orr     v17.16b, v17.16b, v6.16b

        st2     {{ v16.8h, v17.8h }}, [\scratch]
.endm

.macro SAMPLE_NTT_FILTER count
        mov     w15, #\count
.Lsample_ntt_filter_loop\@:
        ldrh    w12, [x10], #2
        cmp     w12, w11
        b.hs    .Lsample_ntt_filter_skip\@
        strh    w12, [x0], #2
        add     x9, x9, #1
.Lsample_ntt_filter_skip\@:
        subs    w15, w15, #1
        b.ne    .Lsample_ntt_filter_loop\@
.endm

.globl rscrypto_mlkem_rej_uniform_block_aarch64_linux
.type rscrypto_mlkem_rej_uniform_block_aarch64_linux, %function
.hidden rscrypto_mlkem_rej_uniform_block_aarch64_linux
rscrypto_mlkem_rej_uniform_block_aarch64_linux:
        sub     sp, sp, #64
        mov     x13, x1
        mov     x9, #0
        mov     w11, #3329
        movi    v29.16b, #15

        mov     w14, #3
.Lsample_ntt_block_loop:
        mov     x10, sp
        SAMPLE_NTT_DECODE_32 x13, x10
        mov     x10, sp
        SAMPLE_NTT_FILTER 32
        subs    w14, w14, #1
        b.ne    .Lsample_ntt_block_loop

        mov     x10, sp
        SAMPLE_NTT_DECODE_16 x13, x10
        mov     x10, sp
        SAMPLE_NTT_FILTER 16

        mov     x0, x9
        add     sp, sp, #64
        ret
.size rscrypto_mlkem_rej_uniform_block_aarch64_linux, .-rscrypto_mlkem_rej_uniform_block_aarch64_linux
