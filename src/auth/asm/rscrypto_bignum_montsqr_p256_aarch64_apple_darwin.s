// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p256/bignum_montsqr_p256.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

        .globl _rscrypto_bignum_montsqr_p256

        .private_extern _rscrypto_bignum_montsqr_p256
        .text
        .balign 4

_rscrypto_bignum_montsqr_p256:
        .cfi_startproc

        ldr q19, [x1]
        ldp x9, x13, [x1]
        ldr q23, [x1, #16]
        ldr q0, [x1]
        ldp x1, x10, [x1, #16]
        uzp2 v29.4S, v19.4S, v19.4S
        xtn v4.2S, v19.2D
        umulh x8, x9, x13
        rev64 v20.4S, v23.4S
        umull v16.2D, v19.2S, v19.2S
        umull v1.2D, v29.2S, v4.2S
        mul v20.4S, v20.4S, v0.4S
        subs x14, x9, x13
        umulh x15, x9, x1
        mov x16, v16.d[1]
        umull2 v4.2D, v19.4S, v19.4S
        mov x4, v16.d[0]
        uzp1 v17.4S, v23.4S, v0.4S
        uaddlp v19.2D, v20.4S
        lsr x7, x8, #63
        mul x11, x9, x13
        mov x12, v1.d[0]
        csetm x5, cc
        cneg x6, x14, cc
        mov x3, v4.d[1]
        mov x14, v4.d[0]
        subs x2, x10, x1
        mov x9, v1.d[1]
        cneg x17, x2, cc
        cinv x2, x5, cc
        adds x5, x4, x12, lsl #33
        extr x4, x8, x11, #63
        lsr x8, x12, #31
        uzp1 v20.4S, v0.4S, v0.4S
        shl v19.2D, v19.2D, #32
        adc x16, x16, x8
        adds x8, x14, x9, lsl #33
        lsr x14, x9, #31
        lsl x9, x5, #32
        umlal v19.2D, v20.2S, v17.2S
        adc x14, x3, x14
        adds x16, x16, x11, lsl #1
        lsr x3, x5, #32
        umulh x12, x6, x17
        adcs x4, x8, x4
        adc x11, x14, x7
        subs x8, x5, x9
        sbc x5, x5, x3
        adds x16, x16, x9
        mov x14, v19.d[0]
        mul x17, x6, x17
        adcs x3, x4, x3
        lsl x7, x16, #32
        umulh x13, x13, x10
        adcs x11, x11, x8
        lsr x8, x16, #32
        adc x5, x5, xzr
        subs x9, x16, x7
        sbc x16, x16, x8
        adds x7, x3, x7
        mov x3, v19.d[1]
        adcs x6, x11, x8
        umulh x11, x1, x10
        adcs x5, x5, x9
        eor x8, x12, x2
        adc x9, x16, xzr
        adds x16, x14, x15
        adc x15, x15, xzr
        adds x12, x16, x3
        eor x16, x17, x2
        mul x4, x1, x10
        adcs x15, x15, x13
        adc x17, x13, xzr
        adds x15, x15, x3
        adc x3, x17, xzr
        cmn x2, #0x1
        mul x17, x10, x10
        adcs x12, x12, x16
        adcs x16, x15, x8
        umulh x10, x10, x10
        adc x2, x3, x2
        adds x14, x14, x14
        adcs x12, x12, x12
        adcs x16, x16, x16
        adcs x2, x2, x2
        adc x15, xzr, xzr
        adds x14, x14, x7
        mul x3, x1, x1
        adcs x12, x12, x6
        lsr x7, x14, #32
        adcs x16, x16, x5
        lsl x5, x14, #32
        umulh x13, x1, x1
        adcs x2, x2, x9
        mov x6, #0xffffffff
        adc x15, x15, xzr
        adds x8, x4, x4
        adcs x1, x11, x11
        mov x11, #0xffffffff00000001
        adc x4, xzr, xzr
        subs x9, x14, x5
        sbc x14, x14, x7
        adds x12, x12, x5
        adcs x16, x16, x7
        lsl x5, x12, #32
        lsr x7, x12, #32
        adcs x2, x2, x9
        adcs x14, x15, x14
        adc x15, xzr, xzr
        subs x9, x12, x5
        sbc x12, x12, x7
        adds x16, x16, x5
        adcs x2, x2, x7
        adcs x14, x14, x9
        adcs x12, x15, x12
        adc x15, xzr, xzr
        adds x16, x16, x3
        adcs x2, x2, x13
        adcs x14, x14, x17
        adcs x12, x12, x10
        adc x15, x15, xzr
        adds x2, x2, x8
        adcs x14, x14, x1
        adcs x12, x12, x4
        adcs x15, x15, xzr
        adds x3, x16, #0x1
        sbcs x5, x2, x6
        sbcs x8, x14, xzr
        sbcs x11, x12, x11
        sbcs xzr, x15, xzr
        csel x16, x3, x16, cs
        csel x14, x8, x14, cs
        csel x12, x11, x12, cs
        csel x2, x5, x2, cs
        stp x14, x12, [x0, #16]
        stp x16, x2, [x0]
        ret %% .cfi_endproc
