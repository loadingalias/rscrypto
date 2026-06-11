// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p384/bignum_montsqr_p384.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

        .globl _rscrypto_bignum_montsqr_p384

        .private_extern _rscrypto_bignum_montsqr_p384
        .text
        .balign 4

_rscrypto_bignum_montsqr_p384:
        .cfi_startproc

        ldr q1, [x1]
        ldp x9, x2, [x1]
        ldr q0, [x1]
        ldp x4, x6, [x1, #16]
        rev64 v21.4S, v1.4S
        uzp2 v28.4S, v1.4S, v1.4S
        umulh x7, x9, x2
        xtn v17.2S, v1.2D
        mul v27.4S, v21.4S, v0.4S
        ldr q20, [x1, #32]
        xtn v30.2S, v0.2D
        ldr q1, [x1, #32]
        uzp2 v31.4S, v0.4S, v0.4S
        ldp x5, x10, [x1, #32]
        umulh x8, x9, x4
        uaddlp v3.2D, v27.4S
        umull v16.2D, v30.2S, v17.2S
        mul x16, x9, x4
        umull v27.2D, v30.2S, v28.2S
        shrn v0.2S, v20.2D, #32
        xtn v7.2S, v20.2D
        shl v20.2D, v3.2D, #32
        umull v3.2D, v31.2S, v28.2S
        mul x3, x2, x4
        umlal v20.2D, v30.2S, v17.2S
        umull v22.2D, v7.2S, v0.2S
        usra v27.2D, v16.2D, #32
        umulh x11, x2, x4
        movi v21.2D, #0x00000000ffffffff
        uzp2 v28.4S, v1.4S, v1.4S
        adds x15, x16, x7
        and v5.16B, v27.16B, v21.16B
        adcs x3, x3, x8
        usra v3.2D, v27.2D, #32
        dup v29.2D, x6
        adcs x16, x11, xzr
        mov x14, v20.d[0]
        umlal v5.2D, v31.2S, v17.2S
        mul x8, x9, x2
        mov x7, v20.d[1]
        shl v19.2D, v22.2D, #33
        xtn v25.2S, v29.2D
        rev64 v31.4S, v1.4S
        lsl x13, x14, #32
        uzp2 v6.4S, v29.4S, v29.4S
        umlal v19.2D, v7.2S, v7.2S
        usra v3.2D, v5.2D, #32
        adds x1, x8, x8
        umulh x8, x4, x4
        add x12, x13, x14
        mul v17.4S, v31.4S, v29.4S
        xtn v4.2S, v1.2D
        adcs x14, x15, x15
        lsr x13, x12, #32
        adcs x15, x3, x3
        umull v31.2D, v25.2S, v28.2S
        adcs x11, x16, x16
        umull v21.2D, v25.2S, v4.2S
        mov x17, v3.d[0]
        umull v18.2D, v6.2S, v28.2S
        adc x16, x8, xzr
        uaddlp v16.2D, v17.4S
        movi v1.2D, #0x00000000ffffffff
        subs x13, x13, x12
        usra v31.2D, v21.2D, #32
        sbc x8, x12, xzr
        adds x17, x17, x1
        mul x1, x4, x4
        shl v28.2D, v16.2D, #32
        mov x3, v3.d[1]
        adcs x14, x7, x14
        extr x7, x8, x13, #32
        adcs x13, x3, x15
        and v3.16B, v31.16B, v1.16B
        adcs x11, x1, x11
        lsr x1, x8, #32
        umlal v3.2D, v6.2S, v4.2S
        usra v18.2D, v31.2D, #32
        adc x3, x16, xzr
        adds x1, x1, x12
        umlal v28.2D, v25.2S, v4.2S
        adc x16, xzr, xzr
        subs x15, x17, x7
        sbcs x7, x14, x1
        lsl x1, x15, #32
        sbcs x16, x13, x16
        add x8, x1, x15
        usra v18.2D, v3.2D, #32
        sbcs x14, x11, xzr
        lsr x1, x8, #32
        sbcs x17, x3, xzr
        sbc x11, x12, xzr
        subs x13, x1, x8
        umulh x12, x4, x10
        sbc x1, x8, xzr
        extr x13, x1, x13, #32
        lsr x1, x1, #32
        adds x15, x1, x8
        adc x1, xzr, xzr
        subs x7, x7, x13
        sbcs x13, x16, x15
        lsl x3, x7, #32
        umulh x16, x2, x5
        sbcs x15, x14, x1
        add x7, x3, x7
        sbcs x3, x17, xzr
        lsr x1, x7, #32
        sbcs x14, x11, xzr
        sbc x11, x8, xzr
        subs x8, x1, x7
        sbc x1, x7, xzr
        extr x8, x1, x8, #32
        lsr x1, x1, #32
        adds x1, x1, x7
        adc x17, xzr, xzr
        subs x13, x13, x8
        umulh x8, x9, x6
        sbcs x1, x15, x1
        sbcs x15, x3, x17
        sbcs x3, x14, xzr
        mul x17, x2, x5
        sbcs x11, x11, xzr
        stp x13, x1, [x0]
        sbc x14, x7, xzr
        mul x7, x4, x10
        subs x1, x9, x2
        stp x15, x3, [x0, #16]
        csetm x15, cc
        cneg x1, x1, cc
        stp x11, x14, [x0, #32]
        mul x14, x9, x6
        adds x17, x8, x17
        adcs x7, x16, x7
        adc x13, x12, xzr
        subs x12, x5, x6
        cneg x3, x12, cc
        cinv x16, x15, cc
        mul x8, x1, x3
        umulh x1, x1, x3
        eor x12, x8, x16
        adds x11, x17, x14
        adcs x3, x7, x17
        adcs x15, x13, x7
        adc x8, x13, xzr
        adds x3, x3, x14
        adcs x15, x15, x17
        adcs x17, x8, x7
        eor x1, x1, x16
        adc x13, x13, xzr
        subs x9, x9, x4
        csetm x8, cc
        cneg x9, x9, cc
        subs x4, x2, x4
        cneg x4, x4, cc
        csetm x7, cc
        subs x2, x10, x6
        cinv x8, x8, cc
        cneg x2, x2, cc
        cmn x16, #0x1
        adcs x11, x11, x12
        mul x12, x9, x2
        adcs x3, x3, x1
        adcs x15, x15, x16
        umulh x9, x9, x2
        adcs x17, x17, x16
        adc x13, x13, x16
        subs x1, x10, x5
        cinv x2, x7, cc
        cneg x1, x1, cc
        eor x9, x9, x8
        cmn x8, #0x1
        eor x7, x12, x8
        mul x12, x4, x1
        adcs x3, x3, x7
        adcs x7, x15, x9
        adcs x15, x17, x8
        ldp x9, x17, [x0, #16]
        umulh x4, x4, x1
        adc x8, x13, x8
        cmn x2, #0x1
        eor x1, x12, x2
        adcs x1, x7, x1
        ldp x7, x16, [x0]
        eor x12, x4, x2
        adcs x4, x15, x12
        ldp x15, x12, [x0, #32]
        adc x8, x8, x2
        adds x13, x14, x14
        umulh x14, x5, x10
        adcs x2, x11, x11
        adcs x3, x3, x3
        adcs x1, x1, x1
        adcs x4, x4, x4
        adcs x11, x8, x8
        adc x8, xzr, xzr
        adds x13, x13, x7
        adcs x2, x2, x16
        mul x16, x5, x10
        adcs x3, x3, x9
        adcs x1, x1, x17
        umulh x5, x5, x5
        lsl x9, x13, #32
        add x9, x9, x13
        adcs x4, x4, x15
        mov x13, v28.d[1]
        adcs x15, x11, x12
        lsr x7, x9, #32
        adc x11, x8, xzr
        subs x7, x7, x9
        umulh x10, x10, x10
        sbc x17, x9, xzr
        extr x7, x17, x7, #32
        lsr x17, x17, #32
        adds x17, x17, x9
        adc x12, xzr, xzr
        subs x8, x2, x7
        sbcs x17, x3, x17
        lsl x7, x8, #32
        sbcs x2, x1, x12
        add x3, x7, x8
        sbcs x12, x4, xzr
        lsr x1, x3, #32
        sbcs x7, x15, xzr
        sbc x15, x9, xzr
        subs x1, x1, x3
        sbc x4, x3, xzr
        lsr x9, x4, #32
        extr x8, x4, x1, #32
        adds x9, x9, x3
        adc x4, xzr, xzr
        subs x1, x17, x8
        lsl x17, x1, #32
        sbcs x8, x2, x9
        sbcs x9, x12, x4
        add x17, x17, x1
        mov x1, v18.d[1]
        lsr x2, x17, #32
        sbcs x7, x7, xzr
        mov x12, v18.d[0]
        sbcs x15, x15, xzr
        sbc x3, x3, xzr
        subs x4, x2, x17
        sbc x2, x17, xzr
        adds x12, x13, x12
        adcs x16, x16, x1
        lsr x13, x2, #32
        extr x1, x2, x4, #32
        adc x2, x14, xzr
        adds x4, x13, x17
        mul x13, x6, x6
        adc x14, xzr, xzr
        subs x1, x8, x1
        sbcs x4, x9, x4
        mov x9, v28.d[0]
        sbcs x7, x7, x14
        sbcs x8, x15, xzr
        sbcs x3, x3, xzr
        sbc x14, x17, xzr
        adds x17, x9, x9
        adcs x12, x12, x12
        mov x15, v19.d[0]
        adcs x9, x16, x16
        umulh x6, x6, x6
        adcs x16, x2, x2
        adc x2, xzr, xzr
        adds x11, x11, x8
        adcs x3, x3, xzr
        adcs x14, x14, xzr
        adcs x8, xzr, xzr
        adds x13, x1, x13
        mov x1, v19.d[1]
        adcs x6, x4, x6
        mov x4, #0xffffffff
        adcs x15, x7, x15
        adcs x7, x11, x5
        adcs x1, x3, x1
        adcs x14, x14, x10
        adc x11, x8, xzr
        adds x6, x6, x17
        adcs x8, x15, x12
        adcs x3, x7, x9
        adcs x15, x1, x16
        mov x16, #0xffffffff00000001
        adcs x14, x14, x2
        mov x2, #0x1
        adc x17, x11, xzr
        cmn x13, x16
        adcs xzr, x6, x4
        adcs xzr, x8, x2
        adcs xzr, x3, xzr
        adcs xzr, x15, xzr
        adcs xzr, x14, xzr
        adc x1, x17, xzr
        neg x9, x1
        and x1, x16, x9
        adds x11, x13, x1
        and x13, x4, x9
        adcs x5, x6, x13
        and x1, x2, x9
        adcs x7, x8, x1
        stp x11, x5, [x0]
        adcs x11, x3, xzr
        adcs x2, x15, xzr
        stp x7, x11, [x0, #16]
        adc x17, x14, xzr
        stp x2, x17, [x0, #32]

        ret %% .cfi_endproc
