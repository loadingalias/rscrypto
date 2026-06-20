// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// rscrypto-owned aarch64 ML-KEM base-multiply accumulate kernels.
//
// ABI:
//   x0: uint16_t acc[256]            read/write accumulator
//   x1: const uint16_t a[256|16]     read-only multiplicand
//   x2: const uint16_t b[256|16]     read-only multiplicand
//   x3: const int16_t gamma[128|8]   read-only Montgomery gammas
//
// The full-polynomial entry processes 16 fixed chunks. The K=3/K=4 entries take
// x1/x2 as contiguous PolyVec bases with 512-byte polynomial strides and fuse
// the whole dot product into one accumulator pass. The chunk entry processes
// exactly one 16-coefficient chunk and is used by the fused SampleNTT path tests.
// All branches and memory addresses depend only on public ML-KEM dimensions.

.text
.balign 4

.macro MONT_REDUCE out, lo, hi
        xtn     v20.4h, \lo\().4s
        xtn2    v20.8h, \hi\().4s
        mul     v20.8h, v20.8h, v31.8h
        smull   v21.4s, v20.4h, v30.4h
        smull2  v22.4s, v20.8h, v30.8h
        shrn    v21.4h, v21.4s, #16
        shrn2   v21.8h, v22.4s, #16
        shrn    v22.4h, \lo\().4s, #16
        shrn2   v22.8h, \hi\().4s, #16
        sub     \out\().8h, v22.8h, v21.8h
.endm

.macro SIGNED_TO_MOD_Q value
        sshr    v20.8h, \value\().8h, #15
        and     v20.16b, v20.16b, v30.16b
        add     \value\().8h, \value\().8h, v20.8h
.endm

.macro ADD_MOD_Q acc, addend
        add     \acc\().8h, \acc\().8h, \addend\().8h
        cmhs    v20.8h, \acc\().8h, v30.8h
        bic     v20.8h, #13, lsl #8
        add     \acc\().8h, \acc\().8h, v20.8h
.endm

.macro BASEMUL_PRODUCT_16 a_ptr, b_ptr
        ld2     {{ v2.8h, v3.8h }}, [\a_ptr], #32
        ld2     {{ v4.8h, v5.8h }}, [\b_ptr], #32
        mov     v6.16b, v21.16b

        smull   v7.4s, v3.4h, v5.4h
        smull2  v16.4s, v3.8h, v5.8h
        xtn     v17.4h, v7.4s
        mul     v17.4h, v17.4h, v31.4h
        smull   v17.4s, v17.4h, v30.4h
        shrn    v17.4h, v17.4s, #16
        shrn    v7.4h, v7.4s, #16
        sub     v7.4h, v7.4h, v17.4h
        xtn     v17.4h, v16.4s
        mul     v17.4h, v17.4h, v31.4h
        smull   v17.4s, v17.4h, v30.4h
        shrn    v17.4h, v17.4s, #16
        shrn    v16.4h, v16.4s, #16
        sub     v16.4h, v16.4h, v17.4h

        smull   v17.4s, v2.4h, v4.4h
        smull2  v18.4s, v2.8h, v4.8h
        smull   v7.4s, v7.4h, v6.4h
        ext     v6.16b, v6.16b, v6.16b, #8
        smull   v6.4s, v16.4h, v6.4h
        addhn   v16.4h, v7.4s, v17.4s
        smlal   v7.4s, v2.4h, v4.4h
        addhn2  v16.8h, v6.4s, v18.4s
        smlal2  v6.4s, v2.8h, v4.8h
        xtn     v17.4h, v7.4s
        mul     v17.4h, v17.4h, v31.4h
        smull   v17.4s, v17.4h, v30.4h
        xtn     v18.4h, v6.4s
        mul     v18.4h, v18.4h, v31.4h
        smull   v18.4s, v18.4h, v30.4h
        uzp2    v17.8h, v17.8h, v18.8h
        sub     v6.8h, v16.8h, v17.8h
        SIGNED_TO_MOD_Q v6

        smull   v16.4s, v2.4h, v5.4h
        smull2  v17.4s, v2.8h, v5.8h
        smull   v18.4s, v3.4h, v4.4h
        smull2  v19.4s, v3.8h, v4.8h
        addhn   v20.4h, v16.4s, v18.4s
        smlal   v16.4s, v3.4h, v4.4h
        addhn2  v20.8h, v17.4s, v19.4s
        smlal2  v17.4s, v3.8h, v4.8h
        xtn     v18.4h, v16.4s
        mul     v18.4h, v18.4h, v31.4h
        smull   v18.4s, v18.4h, v30.4h
        xtn     v19.4h, v17.4s
        mul     v19.4h, v19.4h, v31.4h
        smull   v19.4s, v19.4h, v30.4h
        uzp2    v18.8h, v18.8h, v19.8h
        sub     v7.8h, v20.8h, v18.8h
        SIGNED_TO_MOD_Q v7
.endm

.macro BASEMUL_ACCUMULATE_16
        ldr     q21, [x3], #16
        BASEMUL_PRODUCT_16 x1, x2
        ld2     {{ v18.8h, v19.8h }}, [x0]
        ADD_MOD_Q v18, v6
        ADD_MOD_Q v19, v7
        st2     {{ v18.8h, v19.8h }}, [x0], #32
.endm

.macro BASEMUL_ACCUMULATE_K3_16
        ldr     q21, [x3], #16
        ld2     {{ v23.8h, v24.8h }}, [x0]

        BASEMUL_PRODUCT_16 x1, x2
        ADD_MOD_Q v23, v6
        ADD_MOD_Q v24, v7

        BASEMUL_PRODUCT_16 x4, x5
        ADD_MOD_Q v23, v6
        ADD_MOD_Q v24, v7

        BASEMUL_PRODUCT_16 x6, x7
        ADD_MOD_Q v23, v6
        ADD_MOD_Q v24, v7

        st2     {{ v23.8h, v24.8h }}, [x0], #32
.endm

.macro BASEMUL_ACCUMULATE_K4_16
        ldr     q21, [x3], #16
        ld2     {{ v23.8h, v24.8h }}, [x0]

        BASEMUL_PRODUCT_16 x1, x2
        ADD_MOD_Q v23, v6
        ADD_MOD_Q v24, v7

        BASEMUL_PRODUCT_16 x4, x7
        ADD_MOD_Q v23, v6
        ADD_MOD_Q v24, v7

        BASEMUL_PRODUCT_16 x5, x8
        ADD_MOD_Q v23, v6
        ADD_MOD_Q v24, v7

        BASEMUL_PRODUCT_16 x6, x9
        ADD_MOD_Q v23, v6
        ADD_MOD_Q v24, v7

        st2     {{ v23.8h, v24.8h }}, [x0], #32
.endm

.globl _rscrypto_mlkem_basemul_accumulate_aarch64_apple_darwin
.private_extern _rscrypto_mlkem_basemul_accumulate_aarch64_apple_darwin
_rscrypto_mlkem_basemul_accumulate_aarch64_apple_darwin:
        mov     w8, #3329
        dup     v30.8h, w8
        mov     w8, #62209
        dup     v31.8h, w8
        mov     w9, #8
1:
        BASEMUL_ACCUMULATE_16
        BASEMUL_ACCUMULATE_16
        subs    w9, w9, #1
        b.ne    1b
        ret

.globl _rscrypto_mlkem_basemul_accumulate_chunk_aarch64_apple_darwin
.private_extern _rscrypto_mlkem_basemul_accumulate_chunk_aarch64_apple_darwin
_rscrypto_mlkem_basemul_accumulate_chunk_aarch64_apple_darwin:
        mov     w8, #3329
        dup     v30.8h, w8
        mov     w8, #62209
        dup     v31.8h, w8
        BASEMUL_ACCUMULATE_16
        ret

.globl _rscrypto_mlkem_basemul_accumulate_k3_aarch64_apple_darwin
.private_extern _rscrypto_mlkem_basemul_accumulate_k3_aarch64_apple_darwin
_rscrypto_mlkem_basemul_accumulate_k3_aarch64_apple_darwin:
        mov     w8, #3329
        dup     v30.8h, w8
        mov     w8, #62209
        dup     v31.8h, w8
        add     x4, x1, #512
        add     x6, x1, #1024
        add     x5, x2, #512
        add     x7, x2, #1024
        mov     w9, #8
1:
        BASEMUL_ACCUMULATE_K3_16
        BASEMUL_ACCUMULATE_K3_16
        subs    w9, w9, #1
        b.ne    1b
        ret

.globl _rscrypto_mlkem_basemul_accumulate_k4_aarch64_apple_darwin
.private_extern _rscrypto_mlkem_basemul_accumulate_k4_aarch64_apple_darwin
_rscrypto_mlkem_basemul_accumulate_k4_aarch64_apple_darwin:
        mov     w12, #3329
        dup     v30.8h, w12
        mov     w12, #62209
        dup     v31.8h, w12
        add     x4, x1, #512
        add     x5, x1, #1024
        add     x6, x1, #1536
        add     x7, x2, #512
        add     x8, x2, #1024
        add     x9, x2, #1536
        mov     w10, #8
1:
        BASEMUL_ACCUMULATE_K4_16
        BASEMUL_ACCUMULATE_K4_16
        subs    w10, w10, #1
        b.ne    1b
        ret
