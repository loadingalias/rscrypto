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
// The full-polynomial entry processes 16 fixed chunks. The chunk entry processes
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

.macro BASEMUL_ACCUMULATE_16
        ld2     {{ v0.8h, v1.8h }}, [x1], #32
        ld2     {{ v2.8h, v3.8h }}, [x2], #32
        ldr     q4, [x3], #16

        smull   v16.4s, v1.4h, v3.4h
        smull2  v17.4s, v1.8h, v3.8h
        MONT_REDUCE v5, v16, v17

        smull   v16.4s, v0.4h, v2.4h
        smull2  v17.4s, v0.8h, v2.8h
        smull   v18.4s, v5.4h, v4.4h
        smull2  v19.4s, v5.8h, v4.8h
        add     v16.4s, v16.4s, v18.4s
        add     v17.4s, v17.4s, v19.4s
        MONT_REDUCE v6, v16, v17
        SIGNED_TO_MOD_Q v6

        smull   v16.4s, v0.4h, v3.4h
        smull2  v17.4s, v0.8h, v3.8h
        smlal   v16.4s, v1.4h, v2.4h
        smlal2  v17.4s, v1.8h, v2.8h
        MONT_REDUCE v7, v16, v17
        SIGNED_TO_MOD_Q v7

        ld2     {{ v18.8h, v19.8h }}, [x0]
        ADD_MOD_Q v18, v6
        ADD_MOD_Q v19, v7
        st2     {{ v18.8h, v19.8h }}, [x0], #32
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
