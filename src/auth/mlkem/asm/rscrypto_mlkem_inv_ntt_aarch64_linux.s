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

.macro INIT_CONSTANTS scale_reg
        mov     w8, #3329
        dup     v30.8h, w8
        mov     w8, #62209
        dup     v31.8h, w8
        dup     v29.8h, \scale_reg
        mul     v27.8h, v29.8h, v31.8h
.endm

// Unrolled inverse-NTT butterfly schedule matching the existing Rust NEON
// scalar-oracle path. Loop bounds, zeta loads, and polynomial addresses are
// fixed by public ML-KEM dimensions.
.p2align 2
.type rscrypto_mlkem_inv_ntt_butterflies_aarch64_linux, %function
rscrypto_mlkem_inv_ntt_butterflies_aarch64_linux:
	mov	x8, #-8
	mov	w10, #3329
	add	x9, x1, #254
	dup	v0.4h, w10
	mov	w10, #62209
	dup	v1.4h, w10
	mov	x10, x0
.Linv_bfly_1:
	ldrh	w11, [x9]
	ldurh	w12, [x9, #-2]
	ldr	q2, [x10]
	xtn	v3.2s, v2.2d
	uzp2	v2.4s, v2.4s, v0.4s
	fmov	s4, w11
	mov	v4.h[1], w11
	mov	v4.h[2], w12
	mov	v4.h[3], w12
	add	v5.4h, v2.4h, v3.4h
	cmhs	v6.4h, v5.4h, v0.4h
	bic	v6.4h, #13, lsl #8
	add	v5.4h, v6.4h, v5.4h
	sub	v6.4h, v2.4h, v3.4h
	cmhi	v2.4h, v3.4h, v2.4h
	and	v2.8b, v2.8b, v0.8b
	add	v2.4h, v6.4h, v2.4h
	smull	v2.4s, v2.4h, v4.4h
	xtn	v3.4h, v2.4s
	mul	v3.4h, v3.4h, v1.4h
	smull	v3.4s, v3.4h, v0.4h
	shrn	v3.4h, v3.4s, #16
	shrn	v2.4h, v2.4s, #16
	sub	v2.4h, v2.4h, v3.4h
	cmlt	v3.4h, v2.4h, #0
	and	v3.8b, v3.8b, v0.8b
	add	v2.4h, v3.4h, v2.4h
	zip1	v2.4s, v5.4s, v2.4s
	str	q2, [x10], #16
	add	x8, x8, #8
	sub	x9, x9, #4
	cmp	x8, #248
	b.lo	.Linv_bfly_1
	add	x8, x0, #8
	mov	x9, #-8
	add	x10, x1, #126
	mov	w11, #3329
	dup	v0.4h, w11
	mov	w11, #62209
	dup	v1.4h, w11
.Linv_bfly_3:
	ldrh	w11, [x10], #-2
	ldp	d2, d3, [x8, #-8]
	add	v4.4h, v3.4h, v2.4h
	cmhs	v5.4h, v4.4h, v0.4h
	bic	v5.4h, #13, lsl #8
	add	v4.4h, v5.4h, v4.4h
	stur	d4, [x8, #-8]
	sub	v4.4h, v3.4h, v2.4h
	cmhi	v2.4h, v2.4h, v3.4h
	and	v2.8b, v2.8b, v0.8b
	add	v2.4h, v4.4h, v2.4h
	dup	v3.4h, w11
	smull	v2.4s, v2.4h, v3.4h
	xtn	v3.4h, v2.4s
	mul	v3.4h, v3.4h, v1.4h
	smull	v3.4s, v3.4h, v0.4h
	shrn	v3.4h, v3.4s, #16
	shrn	v2.4h, v2.4s, #16
	sub	v2.4h, v2.4h, v3.4h
	cmlt	v3.4h, v2.4h, #0
	and	v3.8b, v3.8b, v0.8b
	add	v2.4h, v3.4h, v2.4h
	str	d2, [x8], #16
	add	x9, x9, #8
	cmp	x9, #248
	b.lo	.Linv_bfly_3
	add	x8, x0, #16
	mov	x9, #-16
	add	x10, x1, #62
	mov	w11, #3329
	dup	v0.8h, w11
	mov	w11, #62209
	dup	v1.8h, w11
.Linv_bfly_5:
	ldrh	w11, [x10], #-2
	dup	v2.8h, w11
	ldp	q3, q4, [x8, #-16]
	sub	v5.8h, v4.8h, v3.8h
	cmhi	v6.8h, v3.8h, v4.8h
	and	v6.16b, v6.16b, v0.16b
	add	v5.8h, v6.8h, v5.8h
	add	v3.8h, v4.8h, v3.8h
	cmhs	v4.8h, v3.8h, v0.8h
	bic	v4.8h, #13, lsl #8
	add	v3.8h, v4.8h, v3.8h
	sqdmulh	v4.8h, v5.8h, v2.8h
	sshr	v4.8h, v4.8h, #1
	mul	v2.8h, v2.8h, v1.8h
	mul	v2.8h, v2.8h, v5.8h
	sqdmulh	v2.8h, v2.8h, v0.8h
	sshr	v2.8h, v2.8h, #1
	sub	v2.8h, v4.8h, v2.8h
	cmlt	v4.8h, v2.8h, #0
	and	v4.16b, v4.16b, v0.16b
	add	v2.8h, v4.8h, v2.8h
	stp	q3, q2, [x8, #-16]
	add	x9, x9, #16
	add	x8, x8, #32
	cmp	x9, #240
	b.lo	.Linv_bfly_5
	add	x9, x0, #32
	add	x8, x1, #30
	mov	x10, #-32
	mov	w11, #62209
	dup	v0.8h, w11
	mov	w11, #3329
	dup	v1.8h, w11
.Linv_bfly_7:
	ldrh	w11, [x8], #-2
	dup	v2.8h, w11
	mul	v3.8h, v2.8h, v0.8h
	ldp	q4, q5, [x9, #-32]
	ldp	q6, q7, [x9]
	add	v16.8h, v6.8h, v4.8h
	cmhs	v17.8h, v16.8h, v1.8h
	bic	v17.8h, #13, lsl #8
	add	v16.8h, v17.8h, v16.8h
	sub	v17.8h, v6.8h, v4.8h
	cmhi	v4.8h, v4.8h, v6.8h
	and	v4.16b, v4.16b, v1.16b
	add	v4.8h, v4.8h, v17.8h
	sqdmulh	v6.8h, v4.8h, v2.8h
	sshr	v6.8h, v6.8h, #1
	mul	v4.8h, v3.8h, v4.8h
	sqdmulh	v4.8h, v4.8h, v1.8h
	sshr	v4.8h, v4.8h, #1
	sub	v4.8h, v6.8h, v4.8h
	cmlt	v6.8h, v4.8h, #0
	and	v6.16b, v6.16b, v1.16b
	add	v4.8h, v6.8h, v4.8h
	add	v6.8h, v7.8h, v5.8h
	cmhs	v17.8h, v6.8h, v1.8h
	bic	v17.8h, #13, lsl #8
	add	v6.8h, v17.8h, v6.8h
	stp	q16, q6, [x9, #-32]
	sub	v6.8h, v7.8h, v5.8h
	cmhi	v5.8h, v5.8h, v7.8h
	and	v5.16b, v5.16b, v1.16b
	add	v5.8h, v5.8h, v6.8h
	sqdmulh	v2.8h, v5.8h, v2.8h
	sshr	v2.8h, v2.8h, #1
	mul	v3.8h, v3.8h, v5.8h
	sqdmulh	v3.8h, v3.8h, v1.8h
	sshr	v3.8h, v3.8h, #1
	sub	v2.8h, v2.8h, v3.8h
	cmlt	v3.8h, v2.8h, #0
	and	v3.16b, v3.16b, v1.16b
	add	v2.8h, v3.8h, v2.8h
	stp	q4, q2, [x9], #64
	add	x10, x10, #32
	cmp	x10, #224
	b.lo	.Linv_bfly_7
	add	x9, x0, #64
	mov	x10, #-64
	mov	w11, #62209
	dup	v0.8h, w11
	mov	w11, #3329
	dup	v1.8h, w11
.Linv_bfly_9:
	ldrh	w11, [x8], #-2
	dup	v3.8h, w11
	mul	v2.8h, v3.8h, v0.8h
	ldp	q4, q5, [x9, #-64]
	ldp	q6, q7, [x9]
	add	v16.8h, v6.8h, v4.8h
	cmhs	v17.8h, v16.8h, v1.8h
	bic	v17.8h, #13, lsl #8
	add	v16.8h, v17.8h, v16.8h
	sub	v17.8h, v6.8h, v4.8h
	cmhi	v4.8h, v4.8h, v6.8h
	and	v4.16b, v4.16b, v1.16b
	add	v4.8h, v4.8h, v17.8h
	sqdmulh	v6.8h, v4.8h, v3.8h
	sshr	v6.8h, v6.8h, #1
	mul	v4.8h, v2.8h, v4.8h
	sqdmulh	v4.8h, v4.8h, v1.8h
	sshr	v4.8h, v4.8h, #1
	sub	v4.8h, v6.8h, v4.8h
	cmlt	v6.8h, v4.8h, #0
	and	v6.16b, v6.16b, v1.16b
	add	v4.8h, v6.8h, v4.8h
	add	v6.8h, v7.8h, v5.8h
	cmhs	v17.8h, v6.8h, v1.8h
	bic	v17.8h, #13, lsl #8
	add	v6.8h, v17.8h, v6.8h
	stp	q16, q6, [x9, #-64]
	sub	v6.8h, v7.8h, v5.8h
	cmhi	v5.8h, v5.8h, v7.8h
	and	v5.16b, v5.16b, v1.16b
	add	v5.8h, v5.8h, v6.8h
	sqdmulh	v6.8h, v5.8h, v3.8h
	sshr	v6.8h, v6.8h, #1
	mul	v5.8h, v2.8h, v5.8h
	sqdmulh	v5.8h, v5.8h, v1.8h
	sshr	v5.8h, v5.8h, #1
	sub	v5.8h, v6.8h, v5.8h
	cmlt	v6.8h, v5.8h, #0
	and	v6.16b, v6.16b, v1.16b
	add	v5.8h, v6.8h, v5.8h
	stp	q4, q5, [x9]
	ldp	q4, q5, [x9, #-32]
	ldp	q6, q7, [x9, #32]
	add	v16.8h, v6.8h, v4.8h
	cmhs	v17.8h, v16.8h, v1.8h
	bic	v17.8h, #13, lsl #8
	add	v16.8h, v17.8h, v16.8h
	sub	v17.8h, v6.8h, v4.8h
	cmhi	v4.8h, v4.8h, v6.8h
	and	v4.16b, v4.16b, v1.16b
	add	v4.8h, v4.8h, v17.8h
	sqdmulh	v6.8h, v4.8h, v3.8h
	sshr	v6.8h, v6.8h, #1
	mul	v4.8h, v2.8h, v4.8h
	sqdmulh	v4.8h, v4.8h, v1.8h
	sshr	v4.8h, v4.8h, #1
	sub	v4.8h, v6.8h, v4.8h
	cmlt	v6.8h, v4.8h, #0
	and	v6.16b, v6.16b, v1.16b
	add	v4.8h, v6.8h, v4.8h
	add	v6.8h, v7.8h, v5.8h
	cmhs	v17.8h, v6.8h, v1.8h
	bic	v17.8h, #13, lsl #8
	add	v6.8h, v17.8h, v6.8h
	stp	q16, q6, [x9, #-32]
	sub	v6.8h, v7.8h, v5.8h
	cmhi	v5.8h, v5.8h, v7.8h
	and	v5.16b, v5.16b, v1.16b
	add	v5.8h, v5.8h, v6.8h
	sqdmulh	v3.8h, v5.8h, v3.8h
	sshr	v3.8h, v3.8h, #1
	mul	v2.8h, v2.8h, v5.8h
	sqdmulh	v2.8h, v2.8h, v1.8h
	sshr	v2.8h, v2.8h, #1
	sub	v2.8h, v3.8h, v2.8h
	cmlt	v3.8h, v2.8h, #0
	and	v3.16b, v3.16b, v1.16b
	add	v2.8h, v3.8h, v2.8h
	stp	q4, q2, [x9, #32]
	add	x10, x10, #64
	add	x9, x9, #128
	cmp	x10, #192
	b.lo	.Linv_bfly_9
	mov	x10, #0
	mov	w9, #1
	mov	w11, #62209
	dup	v0.8h, w11
	mov	w11, #3329
	dup	v1.8h, w11
.Linv_bfly_11:
	ldrh	w11, [x8], #-2
	dup	v2.8h, w11
	add	x10, x0, x10, lsl #1
	ldp	q4, q7, [x10, #128]
	ldp	q5, q6, [x10]
	mul	v3.8h, v2.8h, v0.8h
	add	v16.8h, v4.8h, v5.8h
	cmhs	v17.8h, v16.8h, v1.8h
	bic	v17.8h, #13, lsl #8
	sub	v18.8h, v4.8h, v5.8h
	add	v16.8h, v17.8h, v16.8h
	cmhi	v4.8h, v5.8h, v4.8h
	and	v4.16b, v4.16b, v1.16b
	add	v4.8h, v4.8h, v18.8h
	sqdmulh	v5.8h, v4.8h, v2.8h
	add	v17.8h, v7.8h, v6.8h
	mul	v4.8h, v3.8h, v4.8h
	cmhs	v18.8h, v17.8h, v1.8h
	bic	v18.8h, #13, lsl #8
	add	v17.8h, v18.8h, v17.8h
	stp	q16, q17, [x10]
	sshr	v5.8h, v5.8h, #1
	sub	v16.8h, v7.8h, v6.8h
	cmhi	v6.8h, v6.8h, v7.8h
	and	v6.16b, v6.16b, v1.16b
	add	v6.8h, v6.8h, v16.8h
	sqdmulh	v7.8h, v6.8h, v2.8h
	sqdmulh	v4.8h, v4.8h, v1.8h
	sshr	v16.8h, v7.8h, #1
	mul	v6.8h, v3.8h, v6.8h
	sqdmulh	v6.8h, v6.8h, v1.8h
	ldp	q17, q18, [x10, #32]
	sshr	v4.8h, v4.8h, #1
	ldp	q19, q20, [x10, #160]
	add	v7.8h, v19.8h, v17.8h
	cmhs	v21.8h, v7.8h, v1.8h
	bic	v21.8h, #13, lsl #8
	sshr	v6.8h, v6.8h, #1
	add	v21.8h, v21.8h, v7.8h
	add	v7.8h, v20.8h, v18.8h
	cmhs	v22.8h, v7.8h, v1.8h
	bic	v22.8h, #13, lsl #8
	add	v22.8h, v22.8h, v7.8h
	sub	v7.8h, v5.8h, v4.8h
	stp	q21, q22, [x10, #32]
	sub	v4.8h, v19.8h, v17.8h
	cmhi	v5.8h, v17.8h, v19.8h
	and	v5.16b, v5.16b, v1.16b
	add	v5.8h, v5.8h, v4.8h
	sub	v4.8h, v16.8h, v6.8h
	sqdmulh	v6.8h, v5.8h, v2.8h
	sshr	v6.8h, v6.8h, #1
	mul	v5.8h, v3.8h, v5.8h
	sqdmulh	v5.8h, v5.8h, v1.8h
	sshr	v5.8h, v5.8h, #1
	sub	v5.8h, v6.8h, v5.8h
	sub	v6.8h, v20.8h, v18.8h
	cmhi	v16.8h, v18.8h, v20.8h
	and	v16.16b, v16.16b, v1.16b
	add	v6.8h, v16.8h, v6.8h
	cmlt	v16.8h, v7.8h, #0
	sqdmulh	v17.8h, v6.8h, v2.8h
	sshr	v17.8h, v17.8h, #1
	mul	v6.8h, v3.8h, v6.8h
	sqdmulh	v6.8h, v6.8h, v1.8h
	sshr	v6.8h, v6.8h, #1
	cmlt	v18.8h, v4.8h, #0
	sub	v6.8h, v17.8h, v6.8h
	ldp	q17, q19, [x10, #64]
	ldp	q20, q21, [x10, #192]
	cmlt	v22.8h, v5.8h, #0
	add	v23.8h, v20.8h, v17.8h
	cmhs	v24.8h, v23.8h, v1.8h
	bic	v24.8h, #13, lsl #8
	add	v23.8h, v24.8h, v23.8h
	sub	v24.8h, v20.8h, v17.8h
	cmlt	v25.8h, v6.8h, #0
	cmhi	v17.8h, v17.8h, v20.8h
	and	v17.16b, v17.16b, v1.16b
	add	v17.8h, v17.8h, v24.8h
	sqdmulh	v20.8h, v17.8h, v2.8h
	sshr	v20.8h, v20.8h, #1
	and	v24.16b, v16.16b, v1.16b
	mul	v16.8h, v3.8h, v17.8h
	sqdmulh	v16.8h, v16.8h, v1.8h
	sshr	v16.8h, v16.8h, #1
	sub	v16.8h, v20.8h, v16.8h
	add	v17.8h, v21.8h, v19.8h
	and	v18.16b, v18.16b, v1.16b
	cmhs	v20.8h, v17.8h, v1.8h
	bic	v20.8h, #13, lsl #8
	add	v17.8h, v20.8h, v17.8h
	stp	q23, q17, [x10, #64]
	and	v17.16b, v22.16b, v1.16b
	sub	v20.8h, v21.8h, v19.8h
	cmhi	v19.8h, v19.8h, v21.8h
	and	v19.16b, v19.16b, v1.16b
	add	v19.8h, v19.8h, v20.8h
	sqdmulh	v20.8h, v19.8h, v2.8h
	and	v21.16b, v25.16b, v1.16b
	sshr	v20.8h, v20.8h, #1
	mul	v19.8h, v3.8h, v19.8h
	sqdmulh	v19.8h, v19.8h, v1.8h
	sshr	v19.8h, v19.8h, #1
	sub	v19.8h, v20.8h, v19.8h
	add	v7.8h, v24.8h, v7.8h
	ldp	q20, q22, [x10, #96]
	ldp	q23, q24, [x10, #224]
	add	v25.8h, v23.8h, v20.8h
	add	v4.8h, v18.8h, v4.8h
	cmhs	v18.8h, v25.8h, v1.8h
	bic	v18.8h, #13, lsl #8
	add	v18.8h, v18.8h, v25.8h
	cmlt	v25.8h, v16.8h, #0
	and	v25.16b, v25.16b, v1.16b
	add	v5.8h, v17.8h, v5.8h
	sub	v17.8h, v23.8h, v20.8h
	cmhi	v20.8h, v20.8h, v23.8h
	cmlt	v23.8h, v19.8h, #0
	and	v23.16b, v23.16b, v1.16b
	and	v20.16b, v20.16b, v1.16b
	add	v6.8h, v21.8h, v6.8h
	add	v17.8h, v20.8h, v17.8h
	sqdmulh	v20.8h, v17.8h, v2.8h
	sshr	v20.8h, v20.8h, #1
	mul	v17.8h, v3.8h, v17.8h
	sqdmulh	v17.8h, v17.8h, v1.8h
	add	v16.8h, v25.8h, v16.8h
	sshr	v17.8h, v17.8h, #1
	sub	v17.8h, v20.8h, v17.8h
	add	v20.8h, v24.8h, v22.8h
	cmhs	v21.8h, v20.8h, v1.8h
	bic	v21.8h, #13, lsl #8
	add	v19.8h, v23.8h, v19.8h
	add	v20.8h, v21.8h, v20.8h
	stp	q18, q20, [x10, #96]
	cmlt	v18.8h, v17.8h, #0
	and	v18.16b, v18.16b, v1.16b
	add	v17.8h, v18.8h, v17.8h
	sub	v18.8h, v24.8h, v22.8h
	cmhi	v20.8h, v22.8h, v24.8h
	and	v20.16b, v20.16b, v1.16b
	add	v18.8h, v20.8h, v18.8h
	stp	q7, q4, [x10, #128]
	sqdmulh	v2.8h, v18.8h, v2.8h
	mul	v3.8h, v3.8h, v18.8h
	mov	x11, x9
	sshr	v2.8h, v2.8h, #1
	stp	q5, q6, [x10, #160]
	sqdmulh	v3.8h, v3.8h, v1.8h
	sshr	v3.8h, v3.8h, #1
	sub	v2.8h, v2.8h, v3.8h
	cmlt	v3.8h, v2.8h, #0
	stp	q16, q19, [x10, #192]
	and	v3.16b, v3.16b, v1.16b
	add	v2.8h, v3.8h, v2.8h
	stp	q17, q2, [x10, #224]
	mov	w10, #128
	mov	w9, #0
	tbnz	w11, #0, .Linv_bfly_11
	mov	x9, #0
	ld1r	{{ v0.8h }}, [x8]
	mov	w8, #62209
	dup	v1.8h, w8
	mul	v1.8h, v0.8h, v1.8h
	add	x8, x0, #256
	mov	w10, #3329
	cmp	x9, #128
	b.hs	.Linv_bfly_14
.Linv_bfly_13:
	ldur	q2, [x8, #-256]
	ldr	q3, [x8]
	add	v4.8h, v3.8h, v2.8h
	dup	v5.8h, w10
	cmhs	v6.8h, v4.8h, v5.8h
	bic	v6.8h, #13, lsl #8
	add	v4.8h, v6.8h, v4.8h
	stur	q4, [x8, #-256]
	sub	v4.8h, v3.8h, v2.8h
	cmhi	v2.8h, v2.8h, v3.8h
	and	v2.16b, v2.16b, v5.16b
	add	v2.8h, v2.8h, v4.8h
	sqdmulh	v3.8h, v2.8h, v0.h[0]
	sshr	v3.8h, v3.8h, #1
	mul	v2.8h, v1.8h, v2.8h
	sqdmulh	v2.8h, v2.8h, v5.8h
	sshr	v2.8h, v2.8h, #1
	sub	v2.8h, v3.8h, v2.8h
	cmlt	v3.8h, v2.8h, #0
	and	v3.16b, v3.16b, v5.16b
	add	v2.8h, v3.8h, v2.8h
	str	q2, [x8], #16
	add	x9, x9, #8
	cmp	x9, #128
	b.lo	.Linv_bfly_13
.Linv_bfly_14:
	ret
.size rscrypto_mlkem_inv_ntt_butterflies_aarch64_linux, .-rscrypto_mlkem_inv_ntt_butterflies_aarch64_linux

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
        stp     x19, x30, [sp, #-16]!
        mov     w19, w2
        bl      rscrypto_mlkem_inv_ntt_butterflies_aarch64_linux
        INIT_CONSTANTS w19
        FINAL_SCALE
        ldp     x19, x30, [sp], #16
        ret
.size rscrypto_mlkem_inv_ntt_aarch64_linux, .-rscrypto_mlkem_inv_ntt_aarch64_linux

.globl rscrypto_mlkem_inv_ntt_add_aarch64_linux
.type rscrypto_mlkem_inv_ntt_add_aarch64_linux, %function
.hidden rscrypto_mlkem_inv_ntt_add_aarch64_linux
rscrypto_mlkem_inv_ntt_add_aarch64_linux:
        stp     x29, x30, [sp, #-32]!
        stp     x19, x20, [sp, #16]
        mov     x29, sp
        mov     x19, x1
        mov     x1, x2
        mov     w20, w3
        bl      rscrypto_mlkem_inv_ntt_butterflies_aarch64_linux
        INIT_CONSTANTS w20
        FINAL_SCALE_ADD x19
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #32
        ret
.size rscrypto_mlkem_inv_ntt_add_aarch64_linux, .-rscrypto_mlkem_inv_ntt_add_aarch64_linux
