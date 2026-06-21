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

.globl rscrypto_mlkem_basemul_accumulate_aarch64_linux
.type rscrypto_mlkem_basemul_accumulate_aarch64_linux, %function
.hidden rscrypto_mlkem_basemul_accumulate_aarch64_linux
rscrypto_mlkem_basemul_accumulate_aarch64_linux:
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
.size rscrypto_mlkem_basemul_accumulate_aarch64_linux, .-rscrypto_mlkem_basemul_accumulate_aarch64_linux

.globl rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux
.type rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux, %function
.hidden rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux
rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux:
        mov     w8, #3329
        dup     v30.8h, w8
        mov     w8, #62209
        dup     v31.8h, w8
        BASEMUL_ACCUMULATE_16
        ret
.size rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux, .-rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux

.globl rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux
.type rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux, %function
.hidden rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux
rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux:
        mov     x8, #0
        mov     x9, x1
        add     x10, x1, #512
        mov     x11, x2
        add     x12, x2, #512
        add     x13, x1, #1024
        add     x14, x2, #1024
        mov     x15, x3
        mov     w16, #62209
        dup     v0.4h, w16
        mov     w16, #3329
        dup     v1.8h, w16
.Lbasemul_k3_loop:
        ldr     q2, [x15], #16
        add     x16, x9, x8
        add     x17, x11, x8
        ld2     {{ v6.8h, v7.8h }}, [x16]
        ld2     {{ v16.8h, v17.8h }}, [x17]
        smull   v3.4s, v17.4h, v7.4h
        smull2  v4.4s, v17.8h, v7.8h
        xtn     v5.4h, v3.4s
        mul     v5.4h, v5.4h, v0.4h
        smull   v5.4s, v5.4h, v1.4h
        shrn    v5.4h, v5.4s, #16
        shrn    v3.4h, v3.4s, #16
        sub     v3.4h, v3.4h, v5.4h
        xtn     v5.4h, v4.4s
        mul     v5.4h, v5.4h, v0.4h
        smull   v5.4s, v5.4h, v1.4h
        shrn    v5.4h, v5.4s, #16
        shrn    v4.4h, v4.4s, #16
        sub     v4.4h, v4.4h, v5.4h
        smull   v5.4s, v16.4h, v6.4h
        smull2  v20.4s, v16.8h, v6.8h
        smull   v22.4s, v3.4h, v2.4h
        ext     v3.16b, v2.16b, v2.16b, #8
        smull   v18.4s, v17.4h, v6.4h
        smull   v23.4s, v4.4h, v3.4h
        smull2  v19.4s, v17.8h, v6.8h
        smull   v4.4s, v16.4h, v7.4h
        smull2  v21.4s, v16.8h, v7.8h
        addhn   v24.4h, v18.4s, v4.4s
        smlal   v18.4s, v16.4h, v7.4h
        addhn   v4.4h, v22.4s, v5.4s
        addhn2  v24.8h, v19.4s, v21.4s
        smlal2  v19.4s, v16.8h, v7.8h
        xtn     v5.4h, v18.4s
        mul     v5.4h, v5.4h, v0.4h
        smlal   v22.4s, v16.4h, v6.4h
        smull   v5.4s, v5.4h, v1.4h
        xtn     v18.4h, v19.4s
        mul     v18.4h, v18.4h, v0.4h
        smull   v18.4s, v18.4h, v1.4h
        uzp2    v5.8h, v5.8h, v18.8h
        sub     v5.8h, v24.8h, v5.8h
        add     x16, x10, x8
        ld2     {{ v18.8h, v19.8h }}, [x16]
        addhn2  v4.8h, v23.4s, v20.4s
        add     x16, x12, x8
        ld2     {{ v20.8h, v21.8h }}, [x16]
        smull   v24.4s, v21.4h, v19.4h
        smull2  v25.4s, v21.8h, v19.8h
        xtn     v26.4h, v24.4s
        mul     v26.4h, v26.4h, v0.4h
        smull   v26.4s, v26.4h, v1.4h
        xtn     v22.4h, v22.4s
        shrn    v26.4h, v26.4s, #16
        shrn    v24.4h, v24.4s, #16
        sub     v24.4h, v24.4h, v26.4h
        xtn     v26.4h, v25.4s
        mul     v26.4h, v26.4h, v0.4h
        smlal2  v23.4s, v16.8h, v6.8h
        smull   v6.4s, v26.4h, v1.4h
        shrn    v6.4h, v6.4s, #16
        shrn    v7.4h, v25.4s, #16
        sub     v6.4h, v7.4h, v6.4h
        smull   v7.4s, v20.4h, v18.4h
        mul     v16.4h, v22.4h, v0.4h
        smull2  v17.4s, v20.8h, v18.8h
        smull   v22.4s, v24.4h, v2.4h
        smull   v24.4s, v6.4h, v3.4h
        addhn   v6.4h, v22.4s, v7.4s
        addhn2  v6.8h, v24.4s, v17.4s
        xtn     v7.4h, v23.4s
        smull   v17.4s, v21.4h, v18.4h
        smull2  v23.4s, v21.8h, v18.8h
        smull   v25.4s, v20.4h, v19.4h
        smull2  v26.4s, v20.8h, v19.8h
        addhn   v25.4h, v17.4s, v25.4s
        smull   v16.4s, v16.4h, v1.4h
        addhn2  v25.8h, v23.4s, v26.4s
        cmlt    v26.8h, v5.8h, #0
        and     v26.16b, v26.16b, v1.16b
        smlal   v22.4s, v20.4h, v18.4h
        mul     v7.4h, v7.4h, v0.4h
        smlal2  v24.4s, v20.8h, v18.8h
        xtn     v22.4h, v22.4s
        mul     v22.4h, v22.4h, v0.4h
        smull   v22.4s, v22.4h, v1.4h
        xtn     v24.4h, v24.4s
        mul     v24.4h, v24.4h, v0.4h
        smlal   v17.4s, v20.4h, v19.4h
        smlal2  v23.4s, v20.8h, v19.8h
        xtn     v17.4h, v17.4s
        mul     v17.4h, v17.4h, v0.4h
        smull   v7.4s, v7.4h, v1.4h
        smull   v17.4s, v17.4h, v1.4h
        xtn     v18.4h, v23.4s
        mul     v18.4h, v18.4h, v0.4h
        smull   v18.4s, v18.4h, v1.4h
        uzp2    v17.8h, v17.8h, v18.8h
        smull   v18.4s, v24.4h, v1.4h
        sub     v17.8h, v25.8h, v17.8h
        cmlt    v19.8h, v17.8h, #0
        and     v20.16b, v19.16b, v1.16b
        add     v5.8h, v5.8h, v17.8h
        add     v5.8h, v26.8h, v5.8h
        uzp2    v7.8h, v16.8h, v7.8h
        add     x16, x13, x8
        ld2     {{ v16.8h, v17.8h }}, [x16]
        uzp2    v21.8h, v22.8h, v18.8h
        add     x16, x14, x8
        ld2     {{ v18.8h, v19.8h }}, [x16]
        sub     v7.8h, v4.8h, v7.8h
        add     v5.8h, v5.8h, v20.8h
        cmhs    v4.8h, v5.8h, v1.8h
        bic     v4.8h, #13, lsl #8
        smull   v20.4s, v19.4h, v17.4h
        smull2  v22.4s, v19.8h, v17.8h
        sub     v6.8h, v6.8h, v21.8h
        xtn     v21.4h, v20.4s
        mul     v21.4h, v21.4h, v0.4h
        smull   v21.4s, v21.4h, v1.4h
        shrn    v21.4h, v21.4s, #16
        shrn    v20.4h, v20.4s, #16
        cmlt    v23.8h, v7.8h, #0
        sub     v20.4h, v20.4h, v21.4h
        xtn     v21.4h, v22.4s
        mul     v21.4h, v21.4h, v0.4h
        smull   v21.4s, v21.4h, v1.4h
        shrn    v21.4h, v21.4s, #16
        cmlt    v24.8h, v6.8h, #0
        shrn    v22.4h, v22.4s, #16
        sub     v21.4h, v22.4h, v21.4h
        smull   v22.4s, v18.4h, v16.4h
        smull2  v25.4s, v18.8h, v16.8h
        smull   v2.4s, v20.4h, v2.4h
        add     v6.8h, v7.8h, v6.8h
        smull   v3.4s, v21.4h, v3.4h
        addhn   v7.4h, v2.4s, v22.4s
        smlal   v2.4s, v18.4h, v16.4h
        addhn2  v7.8h, v3.4s, v25.4s
        and     v20.16b, v23.16b, v1.16b
        smlal2  v3.4s, v18.8h, v16.8h
        xtn     v2.4h, v2.4s
        mul     v2.4h, v2.4h, v0.4h
        smull   v2.4s, v2.4h, v1.4h
        xtn     v3.4h, v3.4s
        and     v21.16b, v24.16b, v1.16b
        mul     v3.4h, v3.4h, v0.4h
        smull   v3.4s, v3.4h, v1.4h
        uzp2    v2.8h, v2.8h, v3.8h
        sub     v2.8h, v7.8h, v2.8h
        cmlt    v3.8h, v2.8h, #0
        and     v3.16b, v3.16b, v1.16b
        smull   v7.4s, v19.4h, v16.4h
        smull2  v22.4s, v19.8h, v16.8h
        smull   v23.4s, v18.4h, v17.4h
        smull2  v24.4s, v18.8h, v17.8h
        addhn   v23.4h, v7.4s, v23.4s
        add     v6.8h, v20.8h, v6.8h
        smlal   v7.4s, v18.4h, v17.4h
        addhn2  v23.8h, v22.4s, v24.4s
        smlal2  v22.4s, v18.8h, v17.8h
        xtn     v7.4h, v7.4s
        add     v6.8h, v6.8h, v21.8h
        mul     v7.4h, v7.4h, v0.4h
        smull   v7.4s, v7.4h, v1.4h
        xtn     v16.4h, v22.4s
        mul     v16.4h, v16.4h, v0.4h
        smull   v16.4s, v16.4h, v1.4h
        cmhs    v17.8h, v6.8h, v1.8h
        uzp2    v7.8h, v7.8h, v16.8h
        sub     v7.8h, v23.8h, v7.8h
        cmlt    v16.8h, v7.8h, #0
        and     v16.16b, v16.16b, v1.16b
        add     v2.8h, v6.8h, v2.8h
        bic     v17.8h, #13, lsl #8
        add     v2.8h, v2.8h, v3.8h
        add     v2.8h, v17.8h, v2.8h
        add     v3.8h, v5.8h, v7.8h
        add     v3.8h, v3.8h, v16.8h
        add     v3.8h, v4.8h, v3.8h
        cmhs    v4.8h, v2.8h, v1.8h
        add     x16, x0, x8
        ld2     {{ v5.8h, v6.8h }}, [x16]
        bic     v4.8h, #13, lsl #8
        cmhs    v7.8h, v3.8h, v1.8h
        bic     v7.8h, #13, lsl #8
        add     v2.8h, v2.8h, v5.8h
        add     v2.8h, v2.8h, v4.8h
        cmhs    v4.8h, v2.8h, v1.8h
        bic     v4.8h, #13, lsl #8
        add     v16.8h, v4.8h, v2.8h
        add     v2.8h, v3.8h, v6.8h
        add     v2.8h, v2.8h, v7.8h
        cmhs    v3.8h, v2.8h, v1.8h
        bic     v3.8h, #13, lsl #8
        add     v17.8h, v3.8h, v2.8h
        st2     {{ v16.8h, v17.8h }}, [x16]
        add     x8, x8, #32
        cmp     x8, #512
        b.ne    .Lbasemul_k3_loop
        ret
.size rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux, .-rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux

.globl rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux
.type rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux, %function
.hidden rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux
rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux:
        mov     x8, #0
        mov     x9, x1
        add     x10, x1, #512
        mov     x11, x2
        add     x12, x2, #512
        mov     x13, x3
        mov     w14, #62209
        dup     v0.4h, w14
        add     x14, x1, #1024
        mov     w16, #3329
        dup     v1.8h, w16
        add     x15, x1, #1536
        add     x16, x2, #1024
        add     x17, x2, #1536
.Lbasemul_k4_loop:
        ldr     q2, [x13], #16
        add     x18, x9, x8
        add     x1, x11, x8
        ld2     {{ v5.8h, v6.8h }}, [x18]
        ld2     {{ v16.8h, v17.8h }}, [x1]
        smull   v3.4s, v17.4h, v6.4h
        smull2  v4.4s, v17.8h, v6.8h
        xtn     v7.4h, v3.4s
        mul     v7.4h, v7.4h, v0.4h
        smull   v7.4s, v7.4h, v1.4h
        shrn    v7.4h, v7.4s, #16
        shrn    v3.4h, v3.4s, #16
        sub     v3.4h, v3.4h, v7.4h
        xtn     v7.4h, v4.4s
        mul     v7.4h, v7.4h, v0.4h
        smull   v7.4s, v7.4h, v1.4h
        shrn    v7.4h, v7.4s, #16
        shrn    v4.4h, v4.4s, #16
        sub     v4.4h, v4.4h, v7.4h
        smull   v7.4s, v16.4h, v5.4h
        smull2  v20.4s, v16.8h, v5.8h
        smull   v22.4s, v3.4h, v2.4h
        ext     v3.16b, v2.16b, v2.16b, #8
        smull   v18.4s, v17.4h, v5.4h
        smull   v23.4s, v4.4h, v3.4h
        smull2  v19.4s, v17.8h, v5.8h
        smull   v4.4s, v16.4h, v6.4h
        smull2  v21.4s, v16.8h, v6.8h
        addhn   v24.4h, v18.4s, v4.4s
        smlal   v18.4s, v16.4h, v6.4h
        addhn   v4.4h, v22.4s, v7.4s
        addhn2  v24.8h, v19.4s, v21.4s
        smlal2  v19.4s, v16.8h, v6.8h
        xtn     v7.4h, v18.4s
        mul     v7.4h, v7.4h, v0.4h
        smlal   v22.4s, v16.4h, v5.4h
        smull   v7.4s, v7.4h, v1.4h
        xtn     v18.4h, v19.4s
        mul     v18.4h, v18.4h, v0.4h
        smull   v18.4s, v18.4h, v1.4h
        uzp2    v7.8h, v7.8h, v18.8h
        sub     v7.8h, v24.8h, v7.8h
        add     x18, x10, x8
        ld2     {{ v18.8h, v19.8h }}, [x18]
        addhn2  v4.8h, v23.4s, v20.4s
        add     x18, x12, x8
        ld2     {{ v20.8h, v21.8h }}, [x18]
        cmlt    v24.8h, v7.8h, #0
        smull   v25.4s, v21.4h, v19.4h
        smull2  v26.4s, v21.8h, v19.8h
        xtn     v27.4h, v25.4s
        mul     v27.4h, v27.4h, v0.4h
        xtn     v22.4h, v22.4s
        smull   v27.4s, v27.4h, v1.4h
        shrn    v27.4h, v27.4s, #16
        shrn    v25.4h, v25.4s, #16
        sub     v25.4h, v25.4h, v27.4h
        xtn     v27.4h, v26.4s
        smlal2  v23.4s, v16.8h, v5.8h
        mul     v5.4h, v27.4h, v0.4h
        smull   v5.4s, v5.4h, v1.4h
        shrn    v5.4h, v5.4s, #16
        shrn    v6.4h, v26.4s, #16
        sub     v5.4h, v6.4h, v5.4h
        mul     v16.4h, v22.4h, v0.4h
        smull   v6.4s, v20.4h, v18.4h
        smull2  v17.4s, v20.8h, v18.8h
        smull   v22.4s, v25.4h, v2.4h
        smull   v25.4s, v5.4h, v3.4h
        addhn   v5.4h, v22.4s, v6.4s
        and     v6.16b, v24.16b, v1.16b
        smlal   v22.4s, v20.4h, v18.4h
        addhn2  v5.8h, v25.4s, v17.4s
        smlal2  v25.4s, v20.8h, v18.8h
        xtn     v17.4h, v22.4s
        mul     v17.4h, v17.4h, v0.4h
        smull   v22.4s, v21.4h, v18.4h
        smull2  v24.4s, v21.8h, v18.8h
        smull   v26.4s, v20.4h, v19.4h
        smull2  v27.4s, v20.8h, v19.8h
        addhn   v26.4h, v22.4s, v26.4s
        xtn     v23.4h, v23.4s
        smlal   v22.4s, v20.4h, v19.4h
        addhn2  v26.8h, v24.4s, v27.4s
        smlal2  v24.4s, v20.8h, v19.8h
        xtn     v18.4h, v22.4s
        xtn     v21.4h, v25.4s
        mul     v18.4h, v18.4h, v0.4h
        smull   v18.4s, v18.4h, v1.4h
        xtn     v19.4h, v24.4s
        mul     v19.4h, v19.4h, v0.4h
        smull   v19.4s, v19.4h, v1.4h
        smull   v16.4s, v16.4h, v1.4h
        uzp2    v18.8h, v18.8h, v19.8h
        sub     v18.8h, v26.8h, v18.8h
        cmlt    v19.8h, v18.8h, #0
        and     v22.16b, v19.16b, v1.16b
        add     v7.8h, v7.8h, v18.8h
        smull   v24.4s, v17.4h, v1.4h
        add     x18, x14, x8
        ld2     {{ v17.8h, v18.8h }}, [x18]
        mul     v23.4h, v23.4h, v0.4h
        add     x18, x16, x8
        ld2     {{ v19.8h, v20.8h }}, [x18]
        mul     v21.4h, v21.4h, v0.4h
        add     v6.8h, v6.8h, v7.8h
        add     v7.8h, v6.8h, v22.8h
        cmhs    v6.8h, v7.8h, v1.8h
        smull   v22.4s, v20.4h, v18.4h
        smull2  v25.4s, v20.8h, v18.8h
        smull   v23.4s, v23.4h, v1.4h
        xtn     v26.4h, v22.4s
        mul     v26.4h, v26.4h, v0.4h
        smull   v26.4s, v26.4h, v1.4h
        shrn    v26.4h, v26.4s, #16
        shrn    v22.4h, v22.4s, #16
        smull   v21.4s, v21.4h, v1.4h
        sub     v22.4h, v22.4h, v26.4h
        xtn     v26.4h, v25.4s
        mul     v26.4h, v26.4h, v0.4h
        smull   v26.4s, v26.4h, v1.4h
        shrn    v26.4h, v26.4s, #16
        bic     v6.8h, #13, lsl #8
        shrn    v25.4h, v25.4s, #16
        sub     v25.4h, v25.4h, v26.4h
        smull   v26.4s, v19.4h, v17.4h
        smull2  v27.4s, v19.8h, v17.8h
        smull   v22.4s, v22.4h, v2.4h
        uzp2    v16.8h, v16.8h, v23.8h
        smull   v23.4s, v25.4h, v3.4h
        addhn   v25.4h, v22.4s, v26.4s
        smlal   v22.4s, v19.4h, v17.4h
        addhn2  v25.8h, v23.4s, v27.4s
        uzp2    v21.8h, v24.8h, v21.8h
        smlal2  v23.4s, v19.8h, v17.8h
        xtn     v22.4h, v22.4s
        sub     v24.8h, v4.8h, v16.8h
        mul     v22.4h, v22.4h, v0.4h
        smull   v22.4s, v22.4h, v1.4h
        xtn     v23.4h, v23.4s
        mul     v4.4h, v23.4h, v0.4h
        smull   v4.4s, v4.4h, v1.4h
        uzp2    v4.8h, v22.8h, v4.8h
        smull   v22.4s, v20.4h, v17.4h
        smull2  v23.4s, v20.8h, v17.8h
        sub     v5.8h, v5.8h, v21.8h
        smull   v16.4s, v19.4h, v18.4h
        smull2  v21.4s, v19.8h, v18.8h
        addhn   v26.4h, v22.4s, v16.4s
        smlal   v22.4s, v19.4h, v18.4h
        addhn2  v26.8h, v23.4s, v21.4s
        sub     v16.8h, v25.8h, v4.8h
        smlal2  v23.4s, v19.8h, v18.8h
        xtn     v4.4h, v22.4s
        mul     v4.4h, v4.4h, v0.4h
        smull   v4.4s, v4.4h, v1.4h
        cmlt    v19.8h, v24.8h, #0
        xtn     v17.4h, v23.4s
        mul     v17.4h, v17.4h, v0.4h
        smull   v17.4s, v17.4h, v1.4h
        uzp2    v4.8h, v4.8h, v17.8h
        sub     v4.8h, v26.8h, v4.8h
        cmlt    v20.8h, v5.8h, #0
        cmlt    v17.8h, v4.8h, #0
        and     v17.16b, v17.16b, v1.16b
        add     v4.8h, v7.8h, v4.8h
        add     v4.8h, v4.8h, v17.8h
        add     v4.8h, v6.8h, v4.8h
        add     v21.8h, v24.8h, v5.8h
        add     x18, x15, x8
        ld2     {{ v6.8h, v7.8h }}, [x18]
        cmlt    v22.8h, v16.8h, #0
        add     x18, x17, x8
        ld2     {{ v17.8h, v18.8h }}, [x18]
        and     v19.16b, v19.16b, v1.16b
        cmhs    v5.8h, v4.8h, v1.8h
        bic     v5.8h, #13, lsl #8
        smull   v23.4s, v18.4h, v7.4h
        smull2  v24.4s, v18.8h, v7.8h
        xtn     v25.4h, v23.4s
        and     v20.16b, v20.16b, v1.16b
        mul     v25.4h, v25.4h, v0.4h
        smull   v25.4s, v25.4h, v1.4h
        shrn    v25.4h, v25.4s, #16
        shrn    v23.4h, v23.4s, #16
        sub     v23.4h, v23.4h, v25.4h
        and     v22.16b, v22.16b, v1.16b
        xtn     v25.4h, v24.4s
        mul     v25.4h, v25.4h, v0.4h
        smull   v25.4s, v25.4h, v1.4h
        shrn    v25.4h, v25.4s, #16
        shrn    v24.4h, v24.4s, #16
        add     v19.8h, v19.8h, v21.8h
        sub     v21.4h, v24.4h, v25.4h
        smull   v24.4s, v17.4h, v6.4h
        smull2  v25.4s, v17.8h, v6.8h
        smull   v2.4s, v23.4h, v2.4h
        smull   v3.4s, v21.4h, v3.4h
        add     v19.8h, v19.8h, v20.8h
        addhn   v20.4h, v2.4s, v24.4s
        smlal   v2.4s, v17.4h, v6.4h
        addhn2  v20.8h, v3.4s, v25.4s
        smlal2  v3.4s, v17.8h, v6.8h
        cmhs    v21.8h, v19.8h, v1.8h
        xtn     v2.4h, v2.4s
        mul     v2.4h, v2.4h, v0.4h
        smull   v2.4s, v2.4h, v1.4h
        xtn     v3.4h, v3.4s
        mul     v3.4h, v3.4h, v0.4h
        add     v16.8h, v19.8h, v16.8h
        smull   v3.4s, v3.4h, v1.4h
        uzp2    v2.8h, v2.8h, v3.8h
        sub     v2.8h, v20.8h, v2.8h
        cmlt    v3.8h, v2.8h, #0
        and     v3.16b, v3.16b, v1.16b
        add     v16.8h, v16.8h, v22.8h
        smull   v19.4s, v18.4h, v6.4h
        smull2  v20.4s, v18.8h, v6.8h
        smull   v22.4s, v17.4h, v7.4h
        smull2  v23.4s, v17.8h, v7.8h
        addhn   v22.4h, v19.4s, v22.4s
        bic     v21.8h, #13, lsl #8
        smlal   v19.4s, v17.4h, v7.4h
        addhn2  v22.8h, v20.4s, v23.4s
        smlal2  v20.4s, v17.8h, v7.8h
        xtn     v6.4h, v19.4s
        add     v7.8h, v21.8h, v16.8h
        mul     v6.4h, v6.4h, v0.4h
        smull   v6.4s, v6.4h, v1.4h
        xtn     v16.4h, v20.4s
        mul     v16.4h, v16.4h, v0.4h
        smull   v16.4s, v16.4h, v1.4h
        cmhs    v17.8h, v7.8h, v1.8h
        uzp2    v6.8h, v6.8h, v16.8h
        sub     v6.8h, v22.8h, v6.8h
        cmlt    v16.8h, v6.8h, #0
        and     v16.16b, v16.16b, v1.16b
        add     v2.8h, v7.8h, v2.8h
        bic     v17.8h, #13, lsl #8
        add     v2.8h, v2.8h, v3.8h
        add     v2.8h, v17.8h, v2.8h
        add     v3.8h, v4.8h, v6.8h
        add     v3.8h, v3.8h, v16.8h
        add     v3.8h, v5.8h, v3.8h
        cmhs    v4.8h, v2.8h, v1.8h
        add     x18, x0, x8
        ld2     {{ v5.8h, v6.8h }}, [x18]
        bic     v4.8h, #13, lsl #8
        cmhs    v7.8h, v3.8h, v1.8h
        bic     v7.8h, #13, lsl #8
        add     v2.8h, v2.8h, v5.8h
        add     v2.8h, v2.8h, v4.8h
        cmhs    v4.8h, v2.8h, v1.8h
        bic     v4.8h, #13, lsl #8
        add     v16.8h, v4.8h, v2.8h
        add     v2.8h, v3.8h, v6.8h
        add     v2.8h, v2.8h, v7.8h
        cmhs    v3.8h, v2.8h, v1.8h
        bic     v3.8h, #13, lsl #8
        add     v17.8h, v3.8h, v2.8h
        st2     {{ v16.8h, v17.8h }}, [x18]
        add     x8, x8, #32
        cmp     x8, #512
        b.ne    .Lbasemul_k4_loop
        ret
.size rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux, .-rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux
