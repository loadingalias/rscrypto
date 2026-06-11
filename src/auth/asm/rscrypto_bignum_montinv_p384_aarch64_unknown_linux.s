// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p384/bignum_montinv_p384.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

        .globl rscrypto_bignum_montinv_p384

        .hidden rscrypto_bignum_montinv_p384

        .text
        .balign 4
rscrypto_bignum_montinv_p384:
        .cfi_startproc



        stp x19, x20, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x19, 0
        .cfi_rel_offset x20, 8
        stp x21, x22, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x21, 0
        .cfi_rel_offset x22, 8
        stp x23, x24, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x23, 0
        .cfi_rel_offset x24, 8
        sub sp, sp, #(32*8 +0)
        .cfi_adjust_cfa_offset 32*8



        mov x20, x0




        mov x10, #0x00000000ffffffff
        mov x11, #0xffffffff00000000
        mov x12, #0xfffffffffffffffe
        mov x15, #0xffffffffffffffff
        stp x10, x11, [sp, #0]
        stp x12, x15, [sp, #0 +2*8]
        stp x15, x15, [sp, #0 +4*8]
        str xzr, [sp, #0 +6*8]

        ldp x2, x3, [x1]
        subs x10, x2, x10
        sbcs x11, x3, x11
        ldp x4, x5, [x1, #(2*8)]
        sbcs x12, x4, x12
        sbcs x13, x5, x15
        ldp x6, x7, [x1, #(4*8)]
        sbcs x14, x6, x15
        sbcs x15, x7, x15

        csel x2, x2, x10, cc
        csel x3, x3, x11, cc
        csel x4, x4, x12, cc
        csel x5, x5, x13, cc
        csel x6, x6, x14, cc
        csel x7, x7, x15, cc

        stp x2, x3, [sp, #(8*8)]
        stp x4, x5, [sp, #(8*8)+2*8]
        stp x6, x7, [sp, #(8*8)+4*8]
        str xzr, [sp, #(8*8)+6*8]
        stp xzr, xzr, [sp, #(16*8)]
        stp xzr, xzr, [sp, #(16*8)+2*8]
        stp xzr, xzr, [sp, #(16*8)+4*8]







        mov x12, #0xfffff00000000000
        orr x10, x12, #0x0000000000000800
        stp xzr, x10, [sp, #(24*8)]
        mov x11, #0x00000000000007ff
        orr x11, x11, #0x0000100000000000
        stp x11, x12, [sp, #(24*8)+2*8]
        mov x12, #0x0000000000000800
        stp x11, x12, [sp, #(24*8)+4*8]





        mov x21, #15
        mov x22, #1
        b Lbignum_montinv_p384_midloop

Lbignum_montinv_p384_loop:



        cmp x10, xzr
        csetm x14, mi
        cneg x10, x10, mi

        cmp x11, xzr
        csetm x15, mi
        cneg x11, x11, mi

        cmp x12, xzr
        csetm x16, mi
        cneg x12, x12, mi

        cmp x13, xzr
        csetm x17, mi
        cneg x13, x13, mi





        and x0, x10, x14
        and x1, x11, x15
        add x9, x0, x1

        and x0, x12, x16
        and x1, x13, x17
        add x19, x0, x1
        ldr x7, [sp, #0]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #(8*8)]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        adc x2, x2, x1

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x5, x19, x0
        adc x3, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x5, x5, x0
        adc x3, x3, x1



        ldr x7, [sp, #0 +8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #(8*8)+8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        adc x6, x6, x1
        extr x4, x2, x4, #59
        str x4, [sp, #0]

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x3, x3, x0
        adc x4, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x3, x3, x0
        adc x4, x4, x1
        extr x5, x3, x5, #59
        str x5, [sp, #(8*8)]



        ldr x7, [sp, #0 +2*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #(8*8)+2*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        adc x5, x5, x1
        extr x2, x6, x2, #59
        str x2, [sp, #0 +8]

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x4, x4, x0
        adc x2, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x4, x4, x0
        adc x2, x2, x1
        extr x3, x4, x3, #59
        str x3, [sp, #(8*8)+8]



        ldr x7, [sp, #0 +3*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, xzr, x1
        ldr x8, [sp, #(8*8)+3*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        adc x3, x3, x1
        extr x6, x5, x6, #59
        str x6, [sp, #0 +2*8]

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x2, x2, x0
        adc x6, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x2, x2, x0
        adc x6, x6, x1
        extr x4, x2, x4, #59
        str x4, [sp, #(8*8)+2*8]



        ldr x7, [sp, #0 +4*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x3, x3, x0
        adc x4, xzr, x1
        ldr x8, [sp, #(8*8)+4*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x3, x3, x0
        adc x4, x4, x1
        extr x5, x3, x5, #59
        str x5, [sp, #0 +3*8]

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x6, x6, x0
        adc x5, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x6, x6, x0
        adc x5, x5, x1
        extr x2, x6, x2, #59
        str x2, [sp, #(8*8)+3*8]



        ldr x7, [sp, #0 +5*8]
        eor x1, x7, x14
        ldr x23, [sp, #0 +6*8]
        eor x2, x23, x14
        and x2, x2, x10
        neg x2, x2
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x4, x0
        adc x2, x2, x1
        ldr x8, [sp, #(8*8)+5*8]
        eor x1, x8, x15
        ldr x24, [sp, #(8*8)+6*8]
        eor x0, x24, x15
        and x0, x0, x11
        sub x2, x2, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        adc x2, x2, x1
        extr x3, x4, x3, #59
        str x3, [sp, #0 +4*8]
        extr x4, x2, x4, #59
        str x4, [sp, #0 +5*8]
        asr x2, x2, #59
        str x2, [sp, #0 +6*8]

        eor x1, x7, x16
        eor x4, x23, x16
        and x4, x4, x12
        neg x4, x4
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x5, x5, x0
        adc x4, x4, x1
        eor x1, x8, x17
        eor x0, x24, x17
        and x0, x0, x13
        sub x4, x4, x0
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x5, x5, x0
        adc x4, x4, x1
        extr x6, x5, x6, #59
        str x6, [sp, #(8*8)+4*8]
        extr x5, x4, x5, #59
        str x5, [sp, #(8*8)+5*8]
        asr x4, x4, #59
        str x4, [sp, #(8*8)+6*8]







        ldr x7, [sp, #(16*8)]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #(24*8)]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #(16*8)]
        adc x2, x2, x1

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x5, x19, x0
        adc x3, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x5, x5, x0
        str x5, [sp, #(24*8)]
        adc x3, x3, x1



        ldr x7, [sp, #(16*8)+8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #(24*8)+8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        str x2, [sp, #(16*8)+8]
        adc x6, x6, x1

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x3, x3, x0
        adc x4, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x3, x3, x0
        str x3, [sp, #(24*8)+8]
        adc x4, x4, x1



        ldr x7, [sp, #(16*8)+2*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #(24*8)+2*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        str x6, [sp, #(16*8)+2*8]
        adc x5, x5, x1

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x4, x4, x0
        adc x2, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x4, x4, x0
        str x4, [sp, #(24*8)+2*8]
        adc x2, x2, x1



        ldr x7, [sp, #(16*8)+3*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, xzr, x1
        ldr x8, [sp, #(24*8)+3*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        str x5, [sp, #(16*8)+3*8]
        adc x3, x3, x1

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x2, x2, x0
        adc x6, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x2, x2, x0
        str x2, [sp, #(24*8)+3*8]
        adc x6, x6, x1



        ldr x7, [sp, #(16*8)+4*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x3, x3, x0
        adc x4, xzr, x1
        ldr x8, [sp, #(24*8)+4*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x3, x3, x0
        str x3, [sp, #(16*8)+4*8]
        adc x4, x4, x1

        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x6, x6, x0
        adc x5, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x6, x6, x0
        str x6, [sp, #(24*8)+4*8]
        adc x5, x5, x1



        ldr x7, [sp, #(16*8)+5*8]
        eor x1, x7, x14
        and x2, x14, x10
        neg x2, x2
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x4, x0
        adc x2, x2, x1
        ldr x8, [sp, #(24*8)+5*8]
        eor x1, x8, x15
        and x0, x15, x11
        sub x2, x2, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #(16*8)+5*8]
        adc x2, x2, x1
        str x2, [sp, #(16*8)+6*8]

        eor x1, x7, x16
        and x4, x16, x12
        neg x4, x4
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x5, x5, x0
        adc x4, x4, x1
        eor x1, x8, x17
        and x0, x17, x13
        sub x4, x4, x0
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x5, x5, x0
        str x5, [sp, #(24*8)+5*8]
        adc x4, x4, x1
        str x4, [sp, #(24*8)+6*8]



        ldp x0, x1, [sp, #(16*8)]
        ldp x2, x3, [sp, #(16*8)+16]
        ldp x4, x5, [sp, #(16*8)+32]
        ldr x6, [sp, #(16*8)+48]
        mov x7, #0xe000000000000000
        adds x0, x0, x7
        mov x8, #0x000000001fffffff
        adcs x1, x1, x8
        mov x9, #0xffffffffe0000000
        bic x9, x9, #0x2000000000000000
        adcs x2, x2, x9
        sbcs x3, x3, xzr
        sbcs x4, x4, xzr
        sbcs x5, x5, xzr
        mov x7, #0x1fffffffffffffff
        adc x6, x6, x7
        add x0, x0, x0, lsl #32
        mov x7, #0xffffffff00000001
        umulh x7, x7, x0
        mov x8, #0x00000000ffffffff
        mul x9, x8, x0
        umulh x8, x8, x0
        adds x7, x7, x9
        adcs x8, x8, x0
        cset x9, cs
        adds x6, x6, x0
        cset x0, cs
        subs x1, x1, x7
        sbcs x2, x2, x8
        sbcs x3, x3, x9
        sbcs x4, x4, xzr
        sbcs x5, x5, xzr
        sbcs x6, x6, xzr
        sbcs x0, x0, xzr
        neg x0, x0
        and x7, x0, #0x00000000ffffffff
        and x8, x0, #0xffffffff00000000
        and x9, x0, #0xfffffffffffffffe
        subs x1, x1, x7
        sbcs x2, x2, x8
        sbcs x3, x3, x9
        sbcs x4, x4, x0
        sbcs x5, x5, x0
        sbc x6, x6, x0
        stp x1, x2, [sp, #(16*8)]
        stp x3, x4, [sp, #(16*8)+16]
        stp x5, x6, [sp, #(16*8)+32]



        ldp x0, x1, [sp, #(24*8)]
        ldp x2, x3, [sp, #(24*8)+16]
        ldp x4, x5, [sp, #(24*8)+32]
        ldr x6, [sp, #(24*8)+48]
        mov x7, #0xe000000000000000
        adds x0, x0, x7
        mov x8, #0x000000001fffffff
        adcs x1, x1, x8
        mov x9, #0xffffffffe0000000
        bic x9, x9, #0x2000000000000000
        adcs x2, x2, x9
        sbcs x3, x3, xzr
        sbcs x4, x4, xzr
        sbcs x5, x5, xzr
        mov x7, #0x1fffffffffffffff
        adc x6, x6, x7
        add x0, x0, x0, lsl #32
        mov x7, #0xffffffff00000001
        umulh x7, x7, x0
        mov x8, #0x00000000ffffffff
        mul x9, x8, x0
        umulh x8, x8, x0
        adds x7, x7, x9
        adcs x8, x8, x0
        cset x9, cs
        adds x6, x6, x0
        cset x0, cs
        subs x1, x1, x7
        sbcs x2, x2, x8
        sbcs x3, x3, x9
        sbcs x4, x4, xzr
        sbcs x5, x5, xzr
        sbcs x6, x6, xzr
        sbcs x0, x0, xzr
        neg x0, x0
        and x7, x0, #0x00000000ffffffff
        and x8, x0, #0xffffffff00000000
        and x9, x0, #0xfffffffffffffffe
        subs x1, x1, x7
        sbcs x2, x2, x8
        sbcs x3, x3, x9
        sbcs x4, x4, x0
        sbcs x5, x5, x0
        sbc x6, x6, x0
        stp x1, x2, [sp, #(24*8)]
        stp x3, x4, [sp, #(24*8)+16]
        stp x5, x6, [sp, #(24*8)+32]

Lbignum_montinv_p384_midloop:

        mov x1, x22
        ldr x2, [sp, #0]
        ldr x3, [sp, #(8*8)]
        and x4, x2, #0xfffff
        orr x4, x4, #0xfffffe0000000000
        and x5, x3, #0xfffff
        orr x5, x5, #0xc000000000000000
        tst x5, #0x1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        asr x5, x5, #1
        add x8, x4, #0x100, lsl #12
        sbfx x8, x8, #21, #21
        mov x11, #0x100000
        add x11, x11, x11, lsl #21
        add x9, x4, x11
        asr x9, x9, #42
        add x10, x5, #0x100, lsl #12
        sbfx x10, x10, #21, #21
        add x11, x5, x11
        asr x11, x11, #42
        mul x6, x8, x2
        mul x7, x9, x3
        mul x2, x10, x2
        mul x3, x11, x3
        add x4, x6, x7
        add x5, x2, x3
        asr x2, x4, #20
        asr x3, x5, #20
        and x4, x2, #0xfffff
        orr x4, x4, #0xfffffe0000000000
        and x5, x3, #0xfffff
        orr x5, x5, #0xc000000000000000
        tst x5, #0x1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        asr x5, x5, #1
        add x12, x4, #0x100, lsl #12
        sbfx x12, x12, #21, #21
        mov x15, #0x100000
        add x15, x15, x15, lsl #21
        add x13, x4, x15
        asr x13, x13, #42
        add x14, x5, #0x100, lsl #12
        sbfx x14, x14, #21, #21
        add x15, x5, x15
        asr x15, x15, #42
        mul x6, x12, x2
        mul x7, x13, x3
        mul x2, x14, x2
        mul x3, x15, x3
        add x4, x6, x7
        add x5, x2, x3
        asr x2, x4, #20
        asr x3, x5, #20
        and x4, x2, #0xfffff
        orr x4, x4, #0xfffffe0000000000
        and x5, x3, #0xfffff
        orr x5, x5, #0xc000000000000000
        tst x5, #0x1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        mul x2, x12, x8
        mul x3, x12, x9
        mul x6, x14, x8
        mul x7, x14, x9
        madd x8, x13, x10, x2
        madd x9, x13, x11, x3
        madd x16, x15, x10, x6
        madd x17, x15, x11, x7
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        asr x5, x5, #1
        add x12, x4, #0x100, lsl #12
        sbfx x12, x12, #22, #21
        mov x15, #0x100000
        add x15, x15, x15, lsl #21
        add x13, x4, x15
        asr x13, x13, #43
        add x14, x5, #0x100, lsl #12
        sbfx x14, x14, #22, #21
        add x15, x5, x15
        asr x15, x15, #43
        mneg x2, x12, x8
        mneg x3, x12, x9
        mneg x4, x14, x8
        mneg x5, x14, x9
        msub x10, x13, x16, x2
        msub x11, x13, x17, x3
        msub x12, x15, x16, x4
        msub x13, x15, x17, x5
        mov x22, x1



        subs x21, x21, #1
        bne Lbignum_montinv_p384_loop
        ldr x0, [sp, #0]
        ldr x1, [sp, #(8*8)]
        mul x0, x0, x10
        madd x1, x1, x11, x0
        asr x0, x1, #63
        cmp x10, xzr
        csetm x14, mi
        cneg x10, x10, mi
        eor x14, x14, x0

        cmp x11, xzr
        csetm x15, mi
        cneg x11, x11, mi
        eor x15, x15, x0

        cmp x12, xzr
        csetm x16, mi
        cneg x12, x12, mi
        eor x16, x16, x0

        cmp x13, xzr
        csetm x17, mi
        cneg x13, x13, mi
        eor x17, x17, x0



        and x0, x10, x14
        and x1, x11, x15
        add x9, x0, x1



        ldr x7, [sp, #(16*8)]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #(24*8)]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #(16*8)]
        adc x2, x2, x1



        ldr x7, [sp, #(16*8)+8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #(24*8)+8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        str x2, [sp, #(16*8)+8]
        adc x6, x6, x1



        ldr x7, [sp, #(16*8)+2*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #(24*8)+2*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        str x6, [sp, #(16*8)+2*8]
        adc x5, x5, x1



        ldr x7, [sp, #(16*8)+3*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, xzr, x1
        ldr x8, [sp, #(24*8)+3*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        str x5, [sp, #(16*8)+3*8]
        adc x3, x3, x1



        ldr x7, [sp, #(16*8)+4*8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x3, x3, x0
        adc x4, xzr, x1
        ldr x8, [sp, #(24*8)+4*8]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x3, x3, x0
        str x3, [sp, #(16*8)+4*8]
        adc x4, x4, x1



        ldr x7, [sp, #(16*8)+5*8]
        eor x1, x7, x14
        and x2, x14, x10
        neg x2, x2
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x4, x0
        adc x2, x2, x1
        ldr x8, [sp, #(24*8)+5*8]
        eor x1, x8, x15
        and x0, x15, x11
        sub x2, x2, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #(16*8)+5*8]
        adc x2, x2, x1
        str x2, [sp, #(16*8)+6*8]




        ldp x10, x0, [sp, #(16*8)]
        ldp x1, x2, [sp, #(16*8)+16]
        ldp x3, x4, [sp, #(16*8)+32]
        ldr x5, [sp, #(16*8)+48]
        mov x7, #0xe000000000000000
        adds x10, x10, x7
        mov x8, #0x000000001fffffff
        adcs x0, x0, x8
        mov x9, #0xffffffffe0000000
        bic x9, x9, #0x2000000000000000
        adcs x1, x1, x9
        sbcs x2, x2, xzr
        sbcs x3, x3, xzr
        sbcs x4, x4, xzr
        mov x7, #0x1fffffffffffffff
        adc x5, x5, x7
        add x10, x10, x10, lsl #32
        mov x7, #0xffffffff00000001
        umulh x7, x7, x10
        mov x8, #0x00000000ffffffff
        mul x9, x8, x10
        umulh x8, x8, x10
        adds x7, x7, x9
        adcs x8, x8, x10
        cset x9, cs
        adds x5, x5, x10
        cset x10, cs
        subs x0, x0, x7
        sbcs x1, x1, x8
        sbcs x2, x2, x9
        sbcs x3, x3, xzr
        sbcs x4, x4, xzr
        sbcs x5, x5, xzr
        sbcs x10, x10, xzr
        neg x10, x10
        and x7, x10, #0x00000000ffffffff
        and x8, x10, #0xffffffff00000000
        and x9, x10, #0xfffffffffffffffe
        subs x0, x0, x7
        sbcs x1, x1, x8
        sbcs x2, x2, x9
        sbcs x3, x3, x10
        sbcs x4, x4, x10
        sbc x5, x5, x10

        mov x10, #0x00000000ffffffff
        subs x10, x0, x10
        mov x11, #0xffffffff00000000
        sbcs x11, x1, x11
        mov x12, #0xfffffffffffffffe
        sbcs x12, x2, x12
        mov x15, #0xffffffffffffffff
        sbcs x13, x3, x15
        sbcs x14, x4, x15
        sbcs x15, x5, x15

        csel x0, x0, x10, cc
        csel x1, x1, x11, cc
        csel x2, x2, x12, cc
        csel x3, x3, x13, cc
        csel x4, x4, x14, cc
        csel x5, x5, x15, cc



        stp x0, x1, [x20]
        stp x2, x3, [x20, #16]
        stp x4, x5, [x20, #32]



        add sp, sp, #(32*8 +0)
        .cfi_adjust_cfa_offset -32*8
        ldp x23, x24, [sp], #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x23
        .cfi_restore x24
        ldp x21, x22, [sp], #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x21
        .cfi_restore x22
        ldp x19, x20, [sp], #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x19
        .cfi_restore x20
        ret
        .cfi_endproc
