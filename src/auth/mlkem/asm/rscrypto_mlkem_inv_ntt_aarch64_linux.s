// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// rscrypto-owned aarch64 ML-KEM inverse NTT kernels.
//
// ABI:
//   rscrypto_mlkem_inv_ntt_aarch64_linux:
//     x0: uint16_t poly[256]       read/write polynomial
//     x1: const int16_t zetas[128] read-only Montgomery zetas
//     w2: int16_t final_scale      public Montgomery scale
//
//   rscrypto_mlkem_inv_ntt_add_aarch64_linux:
//     x0: uint16_t poly[256]       read/write polynomial
//     x1: const uint16_t add[256]  read-only canonical addend
//     x2: const int16_t zetas[128] read-only Montgomery zetas
//     w3: int16_t final_scale      public Montgomery scale
//
// Both entry points follow the scalar/FIPS inverse-NTT schedule exactly:
// zetas[127..=1] in reverse order, canonical modular butterflies, and final
// Montgomery scaling. Branches and memory addresses depend only on public
// ML-KEM dimensions.

.text
.balign 4

.macro ADD_MOD_Q_8 out, a, b
        add     \out\().8h, \a\().8h, \b\().8h
        cmhs    v28.8h, \out\().8h, v30.8h
        bic     v28.8h, #13, lsl #8
        add     \out\().8h, \out\().8h, v28.8h
.endm

.macro SUB_MOD_Q_8 out, a, b
        sub     \out\().8h, \a\().8h, \b\().8h
        cmhi    v28.8h, \b\().8h, \a\().8h
        and     v28.16b, v28.16b, v30.16b
        add     \out\().8h, \out\().8h, v28.8h
.endm

.macro SIGNED_TO_MOD_Q_8 value
        sshr    v28.8h, \value\().8h, #15
        and     v28.16b, v28.16b, v30.16b
        add     \value\().8h, \value\().8h, v28.8h
.endm

.macro MUL_MONT_Q_8_PRE out, a, zeta, zeta_qinv
        sqdmulh v23.8h, \a\().8h, \zeta\().8h
        sshr    v23.8h, v23.8h, #1
        mul     v20.8h, \a\().8h, \zeta_qinv\().8h
        sqdmulh v21.8h, v20.8h, v30.8h
        sshr    v21.8h, v21.8h, #1
        sub     \out\().8h, v23.8h, v21.8h
        SIGNED_TO_MOD_Q_8 \out
.endm

.macro ADD_MOD_Q_4 out, a, b
        add     \out\().4h, \a\().4h, \b\().4h
        cmhs    v28.4h, \out\().4h, v30.4h
        bic     v28.4h, #13, lsl #8
        add     \out\().4h, \out\().4h, v28.4h
.endm

.macro SUB_MOD_Q_4 out, a, b
        sub     \out\().4h, \a\().4h, \b\().4h
        cmhi    v28.4h, \b\().4h, \a\().4h
        and     v28.8b, v28.8b, v30.8b
        add     \out\().4h, \out\().4h, v28.4h
.endm

.macro SIGNED_TO_MOD_Q_4 value
        sshr    v28.4h, \value\().4h, #15
        and     v28.8b, v28.8b, v30.8b
        add     \value\().4h, \value\().4h, v28.4h
.endm

.macro MUL_MONT_Q_4_PRE out, a, zeta, zeta_qinv
        smull   v22.4s, \a\().4h, \zeta\().4h
        mul     v20.4h, \a\().4h, \zeta_qinv\().4h
        smull   v21.4s, v20.4h, v30.4h
        shrn    v21.4h, v21.4s, #16
        shrn    v22.4h, v22.4s, #16
        sub     \out\().4h, v22.4h, v21.4h
        SIGNED_TO_MOD_Q_4 \out
.endm

.macro INIT_CONSTANTS scale_reg
        mov     w8, #3329
        dup     v30.8h, w8
        mov     w8, #62209
        dup     v31.8h, w8
        dup     v29.8h, \scale_reg
        mul     v27.8h, v29.8h, v31.8h
.endm

.macro INV_NTT_BUTTERFLIES zetas_ptr
        add     x10, \zetas_ptr, #254
        mov     x9, x0
        mov     w8, #32
.Linv_len2_loop\@:
        ldr     q0, [x9]
        ldrh    w12, [x10]
        ldrh    w13, [x10, #-2]
        sub     x10, x10, #4
        uzp1    v1.4s, v0.4s, v0.4s
        uzp2    v2.4s, v0.4s, v0.4s
        dup     v3.4h, w12
        ins     v3.h[2], w13
        ins     v3.h[3], w13
        mul     v24.4h, v3.4h, v31.4h
        ADD_MOD_Q_4 v4, v1, v2
        SUB_MOD_Q_4 v5, v2, v1
        MUL_MONT_Q_4_PRE v5, v5, v3, v24
        zip1    v6.2s, v4.2s, v5.2s
        zip2    v7.2s, v4.2s, v5.2s
        mov     v6.d[1], v7.d[0]
        str     q6, [x9], #16
        subs    w8, w8, #1
        b.ne    .Linv_len2_loop\@

        mov     x9, x0
        mov     w8, #32
.Linv_len4_loop\@:
        ldr     d0, [x9]
        ldr     d1, [x9, #8]
        ldrh    w12, [x10]
        sub     x10, x10, #2
        dup     v3.4h, w12
        mul     v24.4h, v3.4h, v31.4h
        ADD_MOD_Q_4 v2, v0, v1
        SUB_MOD_Q_4 v4, v1, v0
        MUL_MONT_Q_4_PRE v5, v4, v3, v24
        str     d2, [x9]
        str     d5, [x9, #8]
        add     x9, x9, #16
        subs    w8, w8, #1
        b.ne    .Linv_len4_loop\@

        mov     x11, #16
.Linv_len_ge8_loop\@:
        mov     x12, #0
.Linv_start_loop\@:
        ldrh    w13, [x10]
        sub     x10, x10, #2
        dup     v3.8h, w13
        mul     v24.8h, v3.8h, v31.8h
        add     x14, x0, x12
        add     x15, x14, x11
        add     x16, x14, x11
.Linv_j_loop\@:
        ldr     q0, [x14]
        ldr     q1, [x15]
        ADD_MOD_Q_8 v2, v0, v1
        SUB_MOD_Q_8 v4, v1, v0
        MUL_MONT_Q_8_PRE v5, v4, v3, v24
        str     q2, [x14], #16
        str     q5, [x15], #16
        cmp     x14, x16
        b.ne    .Linv_j_loop\@
        add     x12, x12, x11, lsl #1
        cmp     x12, #512
        b.lo    .Linv_start_loop\@
        lsl     x11, x11, #1
        cmp     x11, #512
        b.lo    .Linv_len_ge8_loop\@
.endm

.macro FINAL_SCALE
        mov     x9, x0
        mov     w8, #32
.Linv_final_scale_loop\@:
        ldr     q0, [x9]
        MUL_MONT_Q_8_PRE v0, v0, v29, v27
        str     q0, [x9], #16
        subs    w8, w8, #1
        b.ne    .Linv_final_scale_loop\@
.endm

.macro FINAL_SCALE_ADD add_ptr
        mov     x9, x0
        mov     x10, \add_ptr
        mov     w8, #32
.Linv_final_scale_add_loop\@:
        ldr     q0, [x9]
        ldr     q1, [x10], #16
        MUL_MONT_Q_8_PRE v0, v0, v29, v27
        ADD_MOD_Q_8 v0, v0, v1
        str     q0, [x9], #16
        subs    w8, w8, #1
        b.ne    .Linv_final_scale_add_loop\@
.endm

.globl rscrypto_mlkem_inv_ntt_aarch64_linux
.type rscrypto_mlkem_inv_ntt_aarch64_linux, %function
.hidden rscrypto_mlkem_inv_ntt_aarch64_linux
rscrypto_mlkem_inv_ntt_aarch64_linux:
        INIT_CONSTANTS w2
        INV_NTT_BUTTERFLIES x1
        FINAL_SCALE
        ret
.size rscrypto_mlkem_inv_ntt_aarch64_linux, .-rscrypto_mlkem_inv_ntt_aarch64_linux

.globl rscrypto_mlkem_inv_ntt_add_aarch64_linux
.type rscrypto_mlkem_inv_ntt_add_aarch64_linux, %function
.hidden rscrypto_mlkem_inv_ntt_add_aarch64_linux
rscrypto_mlkem_inv_ntt_add_aarch64_linux:
        INIT_CONSTANTS w3
        INV_NTT_BUTTERFLIES x2
        FINAL_SCALE_ADD x1
        ret
.size rscrypto_mlkem_inv_ntt_add_aarch64_linux, .-rscrypto_mlkem_inv_ntt_add_aarch64_linux
