// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p384/p384_montjdouble_alt.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

        .globl rscrypto_p384_montjdouble_alt

        .hidden rscrypto_p384_montjdouble_alt
        .text
        .balign 4
rscrypto_p384_montjdouble_alt:
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
        sub sp, sp, #(48*7 +0)
        .cfi_adjust_cfa_offset 48*7



        mov x23, x0
        mov x24, x1






        ldp x2, x3, [x24, #(2*48)]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x24, #(2*48)+16]
        mul x8, x2, x4
        adds x10, x10, x8
        mul x11, x2, x5
        mul x8, x3, x4
        adcs x11, x11, x8
        umulh x12, x2, x5
        mul x8, x3, x5
        adcs x12, x12, x8
        ldp x6, x7, [x24, #(2*48)+32]
        mul x13, x2, x7
        mul x8, x3, x6
        adcs x13, x13, x8
        umulh x14, x2, x7
        mul x8, x3, x7
        adcs x14, x14, x8
        mul x15, x5, x6
        adcs x15, x15, xzr
        umulh x16, x5, x6
        adc x16, x16, xzr
        umulh x8, x2, x4
        adds x11, x11, x8
        umulh x8, x3, x4
        adcs x12, x12, x8
        umulh x8, x3, x5
        adcs x13, x13, x8
        umulh x8, x3, x6
        adcs x14, x14, x8
        umulh x8, x3, x7
        adcs x15, x15, x8
        adc x16, x16, xzr
        mul x8, x2, x6
        adds x12, x12, x8
        mul x8, x4, x5
        adcs x13, x13, x8
        mul x8, x4, x6
        adcs x14, x14, x8
        mul x8, x4, x7
        adcs x15, x15, x8
        mul x8, x5, x7
        adcs x16, x16, x8
        mul x17, x6, x7
        adcs x17, x17, xzr
        umulh x19, x6, x7
        adc x19, x19, xzr
        umulh x8, x2, x6
        adds x13, x13, x8
        umulh x8, x4, x5
        adcs x14, x14, x8
        umulh x8, x4, x6
        adcs x15, x15, x8
        umulh x8, x4, x7
        adcs x16, x16, x8
        umulh x8, x5, x7
        adcs x17, x17, x8
        adc x19, x19, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        adcs x15, x15, x15
        adcs x16, x16, x16
        adcs x17, x17, x17
        adcs x19, x19, x19
        cset x20, hs
        umulh x8, x2, x2
        mul x2, x2, x2
        adds x9, x9, x8
        mul x8, x3, x3
        adcs x10, x10, x8
        umulh x8, x3, x3
        adcs x11, x11, x8
        mul x8, x4, x4
        adcs x12, x12, x8
        umulh x8, x4, x4
        adcs x13, x13, x8
        mul x8, x5, x5
        adcs x14, x14, x8
        umulh x8, x5, x5
        adcs x15, x15, x8
        mul x8, x6, x6
        adcs x16, x16, x8
        umulh x8, x6, x6
        adcs x17, x17, x8
        mul x8, x7, x7
        adcs x19, x19, x8
        umulh x8, x7, x7
        adc x20, x20, x8
        lsl x5, x2, #32
        add x2, x5, x2
        mov x5, #-4294967295
        umulh x5, x5, x2
        mov x4, #4294967295
        mul x3, x4, x2
        umulh x4, x4, x2
        adds x5, x5, x3
        adcs x4, x4, x2
        adc x3, xzr, xzr
        subs x9, x9, x5
        sbcs x10, x10, x4
        sbcs x11, x11, x3
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x2, x2, xzr
        lsl x5, x9, #32
        add x9, x5, x9
        mov x5, #-4294967295
        umulh x5, x5, x9
        mov x4, #4294967295
        mul x3, x4, x9
        umulh x4, x4, x9
        adds x5, x5, x3
        adcs x4, x4, x9
        adc x3, xzr, xzr
        subs x10, x10, x5
        sbcs x11, x11, x4
        sbcs x12, x12, x3
        sbcs x13, x13, xzr
        sbcs x2, x2, xzr
        sbc x9, x9, xzr
        lsl x5, x10, #32
        add x10, x5, x10
        mov x5, #-4294967295
        umulh x5, x5, x10
        mov x4, #4294967295
        mul x3, x4, x10
        umulh x4, x4, x10
        adds x5, x5, x3
        adcs x4, x4, x10
        adc x3, xzr, xzr
        subs x11, x11, x5
        sbcs x12, x12, x4
        sbcs x13, x13, x3
        sbcs x2, x2, xzr
        sbcs x9, x9, xzr
        sbc x10, x10, xzr
        lsl x5, x11, #32
        add x11, x5, x11
        mov x5, #-4294967295
        umulh x5, x5, x11
        mov x4, #4294967295
        mul x3, x4, x11
        umulh x4, x4, x11
        adds x5, x5, x3
        adcs x4, x4, x11
        adc x3, xzr, xzr
        subs x12, x12, x5
        sbcs x13, x13, x4
        sbcs x2, x2, x3
        sbcs x9, x9, xzr
        sbcs x10, x10, xzr
        sbc x11, x11, xzr
        lsl x5, x12, #32
        add x12, x5, x12
        mov x5, #-4294967295
        umulh x5, x5, x12
        mov x4, #4294967295
        mul x3, x4, x12
        umulh x4, x4, x12
        adds x5, x5, x3
        adcs x4, x4, x12
        adc x3, xzr, xzr
        subs x13, x13, x5
        sbcs x2, x2, x4
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbcs x11, x11, xzr
        sbc x12, x12, xzr
        lsl x5, x13, #32
        add x13, x5, x13
        mov x5, #-4294967295
        umulh x5, x5, x13
        mov x4, #4294967295
        mul x3, x4, x13
        umulh x4, x4, x13
        adds x5, x5, x3
        adcs x4, x4, x13
        adc x3, xzr, xzr
        subs x2, x2, x5
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        sbcs x11, x11, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        adds x2, x2, x14
        adcs x9, x9, x15
        adcs x10, x10, x16
        adcs x11, x11, x17
        adcs x12, x12, x19
        adcs x13, x13, x20
        adc x6, xzr, xzr
        mov x8, #-4294967295
        adds x14, x2, x8
        mov x8, #4294967295
        adcs x15, x9, x8
        mov x8, #1
        adcs x16, x10, x8
        adcs x17, x11, xzr
        adcs x19, x12, xzr
        adcs x20, x13, xzr
        adcs x6, x6, xzr
        csel x2, x2, x14, eq
        csel x9, x9, x15, eq
        csel x10, x10, x16, eq
        csel x11, x11, x17, eq
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        stp x2, x9, [sp, #(48*0)]
        stp x10, x11, [sp, #(48*0)+16]
        stp x12, x13, [sp, #(48*0)+32]
        ldp x2, x3, [x24, #48]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x24, #48 +16]
        mul x8, x2, x4
        adds x10, x10, x8
        mul x11, x2, x5
        mul x8, x3, x4
        adcs x11, x11, x8
        umulh x12, x2, x5
        mul x8, x3, x5
        adcs x12, x12, x8
        ldp x6, x7, [x24, #48 +32]
        mul x13, x2, x7
        mul x8, x3, x6
        adcs x13, x13, x8
        umulh x14, x2, x7
        mul x8, x3, x7
        adcs x14, x14, x8
        mul x15, x5, x6
        adcs x15, x15, xzr
        umulh x16, x5, x6
        adc x16, x16, xzr
        umulh x8, x2, x4
        adds x11, x11, x8
        umulh x8, x3, x4
        adcs x12, x12, x8
        umulh x8, x3, x5
        adcs x13, x13, x8
        umulh x8, x3, x6
        adcs x14, x14, x8
        umulh x8, x3, x7
        adcs x15, x15, x8
        adc x16, x16, xzr
        mul x8, x2, x6
        adds x12, x12, x8
        mul x8, x4, x5
        adcs x13, x13, x8
        mul x8, x4, x6
        adcs x14, x14, x8
        mul x8, x4, x7
        adcs x15, x15, x8
        mul x8, x5, x7
        adcs x16, x16, x8
        mul x17, x6, x7
        adcs x17, x17, xzr
        umulh x19, x6, x7
        adc x19, x19, xzr
        umulh x8, x2, x6
        adds x13, x13, x8
        umulh x8, x4, x5
        adcs x14, x14, x8
        umulh x8, x4, x6
        adcs x15, x15, x8
        umulh x8, x4, x7
        adcs x16, x16, x8
        umulh x8, x5, x7
        adcs x17, x17, x8
        adc x19, x19, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        adcs x15, x15, x15
        adcs x16, x16, x16
        adcs x17, x17, x17
        adcs x19, x19, x19
        cset x20, hs
        umulh x8, x2, x2
        mul x2, x2, x2
        adds x9, x9, x8
        mul x8, x3, x3
        adcs x10, x10, x8
        umulh x8, x3, x3
        adcs x11, x11, x8
        mul x8, x4, x4
        adcs x12, x12, x8
        umulh x8, x4, x4
        adcs x13, x13, x8
        mul x8, x5, x5
        adcs x14, x14, x8
        umulh x8, x5, x5
        adcs x15, x15, x8
        mul x8, x6, x6
        adcs x16, x16, x8
        umulh x8, x6, x6
        adcs x17, x17, x8
        mul x8, x7, x7
        adcs x19, x19, x8
        umulh x8, x7, x7
        adc x20, x20, x8
        lsl x5, x2, #32
        add x2, x5, x2
        mov x5, #-4294967295
        umulh x5, x5, x2
        mov x4, #4294967295
        mul x3, x4, x2
        umulh x4, x4, x2
        adds x5, x5, x3
        adcs x4, x4, x2
        adc x3, xzr, xzr
        subs x9, x9, x5
        sbcs x10, x10, x4
        sbcs x11, x11, x3
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x2, x2, xzr
        lsl x5, x9, #32
        add x9, x5, x9
        mov x5, #-4294967295
        umulh x5, x5, x9
        mov x4, #4294967295
        mul x3, x4, x9
        umulh x4, x4, x9
        adds x5, x5, x3
        adcs x4, x4, x9
        adc x3, xzr, xzr
        subs x10, x10, x5
        sbcs x11, x11, x4
        sbcs x12, x12, x3
        sbcs x13, x13, xzr
        sbcs x2, x2, xzr
        sbc x9, x9, xzr
        lsl x5, x10, #32
        add x10, x5, x10
        mov x5, #-4294967295
        umulh x5, x5, x10
        mov x4, #4294967295
        mul x3, x4, x10
        umulh x4, x4, x10
        adds x5, x5, x3
        adcs x4, x4, x10
        adc x3, xzr, xzr
        subs x11, x11, x5
        sbcs x12, x12, x4
        sbcs x13, x13, x3
        sbcs x2, x2, xzr
        sbcs x9, x9, xzr
        sbc x10, x10, xzr
        lsl x5, x11, #32
        add x11, x5, x11
        mov x5, #-4294967295
        umulh x5, x5, x11
        mov x4, #4294967295
        mul x3, x4, x11
        umulh x4, x4, x11
        adds x5, x5, x3
        adcs x4, x4, x11
        adc x3, xzr, xzr
        subs x12, x12, x5
        sbcs x13, x13, x4
        sbcs x2, x2, x3
        sbcs x9, x9, xzr
        sbcs x10, x10, xzr
        sbc x11, x11, xzr
        lsl x5, x12, #32
        add x12, x5, x12
        mov x5, #-4294967295
        umulh x5, x5, x12
        mov x4, #4294967295
        mul x3, x4, x12
        umulh x4, x4, x12
        adds x5, x5, x3
        adcs x4, x4, x12
        adc x3, xzr, xzr
        subs x13, x13, x5
        sbcs x2, x2, x4
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbcs x11, x11, xzr
        sbc x12, x12, xzr
        lsl x5, x13, #32
        add x13, x5, x13
        mov x5, #-4294967295
        umulh x5, x5, x13
        mov x4, #4294967295
        mul x3, x4, x13
        umulh x4, x4, x13
        adds x5, x5, x3
        adcs x4, x4, x13
        adc x3, xzr, xzr
        subs x2, x2, x5
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        sbcs x11, x11, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        adds x2, x2, x14
        adcs x9, x9, x15
        adcs x10, x10, x16
        adcs x11, x11, x17
        adcs x12, x12, x19
        adcs x13, x13, x20
        adc x6, xzr, xzr
        mov x8, #-4294967295
        adds x14, x2, x8
        mov x8, #4294967295
        adcs x15, x9, x8
        mov x8, #1
        adcs x16, x10, x8
        adcs x17, x11, xzr
        adcs x19, x12, xzr
        adcs x20, x13, xzr
        adcs x6, x6, xzr
        csel x2, x2, x14, eq
        csel x9, x9, x15, eq
        csel x10, x10, x16, eq
        csel x11, x11, x17, eq
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        stp x2, x9, [sp, #(48*1)]
        stp x10, x11, [sp, #(48*1)+16]
        stp x12, x13, [sp, #(48*1)+32]



        ldp x5, x6, [x24, #0]
        ldp x4, x3, [sp, #(48*0)]
        adds x5, x5, x4
        adcs x6, x6, x3
        ldp x7, x8, [x24, #0 +16]
        ldp x4, x3, [sp, #(48*0)+16]
        adcs x7, x7, x4
        adcs x8, x8, x3
        ldp x9, x10, [x24, #0 +32]
        ldp x4, x3, [sp, #(48*0)+32]
        adcs x9, x9, x4
        adcs x10, x10, x3
        csetm x3, cs
        mov x4, #0xffffffff
        and x4, x4, x3
        subs x5, x5, x4
        eor x4, x4, x3
        sbcs x6, x6, x4
        mov x4, #0xfffffffffffffffe
        and x4, x4, x3
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        sbcs x9, x9, x3
        sbc x10, x10, x3
        stp x5, x6, [sp, #(48*5)]
        stp x7, x8, [sp, #(48*5)+16]
        stp x9, x10, [sp, #(48*5)+32]
        ldp x5, x6, [x24, #0]
        ldp x4, x3, [sp, #(48*0)]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [x24, #0 +16]
        ldp x4, x3, [sp, #(48*0)+16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        ldp x9, x10, [x24, #0 +32]
        ldp x4, x3, [sp, #(48*0)+32]
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        csetm x3, lo
        mov x4, #4294967295
        and x4, x4, x3
        adds x5, x5, x4
        eor x4, x4, x3
        adcs x6, x6, x4
        mov x4, #-2
        and x4, x4, x3
        adcs x7, x7, x4
        adcs x8, x8, x3
        adcs x9, x9, x3
        adc x10, x10, x3
        stp x5, x6, [sp, #(48*4)]
        stp x7, x8, [sp, #(48*4)+16]
        stp x9, x10, [sp, #(48*4)+32]
        ldp x3, x4, [sp, #(48*5)]
        ldp x5, x6, [sp, #(48*4)]
        mul x12, x3, x5
        umulh x13, x3, x5
        mul x11, x3, x6
        umulh x14, x3, x6
        adds x13, x13, x11
        ldp x7, x8, [sp, #(48*4)+16]
        mul x11, x3, x7
        umulh x15, x3, x7
        adcs x14, x14, x11
        mul x11, x3, x8
        umulh x16, x3, x8
        adcs x15, x15, x11
        ldp x9, x10, [sp, #(48*4)+32]
        mul x11, x3, x9
        umulh x17, x3, x9
        adcs x16, x16, x11
        mul x11, x3, x10
        umulh x19, x3, x10
        adcs x17, x17, x11
        adc x19, x19, xzr
        mul x11, x4, x5
        adds x13, x13, x11
        mul x11, x4, x6
        adcs x14, x14, x11
        mul x11, x4, x7
        adcs x15, x15, x11
        mul x11, x4, x8
        adcs x16, x16, x11
        mul x11, x4, x9
        adcs x17, x17, x11
        mul x11, x4, x10
        adcs x19, x19, x11
        cset x20, cs
        umulh x11, x4, x5
        adds x14, x14, x11
        umulh x11, x4, x6
        adcs x15, x15, x11
        umulh x11, x4, x7
        adcs x16, x16, x11
        umulh x11, x4, x8
        adcs x17, x17, x11
        umulh x11, x4, x9
        adcs x19, x19, x11
        umulh x11, x4, x10
        adc x20, x20, x11
        ldp x3, x4, [sp, #(48*5)+16]
        mul x11, x3, x5
        adds x14, x14, x11
        mul x11, x3, x6
        adcs x15, x15, x11
        mul x11, x3, x7
        adcs x16, x16, x11
        mul x11, x3, x8
        adcs x17, x17, x11
        mul x11, x3, x9
        adcs x19, x19, x11
        mul x11, x3, x10
        adcs x20, x20, x11
        cset x21, cs
        umulh x11, x3, x5
        adds x15, x15, x11
        umulh x11, x3, x6
        adcs x16, x16, x11
        umulh x11, x3, x7
        adcs x17, x17, x11
        umulh x11, x3, x8
        adcs x19, x19, x11
        umulh x11, x3, x9
        adcs x20, x20, x11
        umulh x11, x3, x10
        adc x21, x21, x11
        mul x11, x4, x5
        adds x15, x15, x11
        mul x11, x4, x6
        adcs x16, x16, x11
        mul x11, x4, x7
        adcs x17, x17, x11
        mul x11, x4, x8
        adcs x19, x19, x11
        mul x11, x4, x9
        adcs x20, x20, x11
        mul x11, x4, x10
        adcs x21, x21, x11
        cset x22, cs
        umulh x11, x4, x5
        adds x16, x16, x11
        umulh x11, x4, x6
        adcs x17, x17, x11
        umulh x11, x4, x7
        adcs x19, x19, x11
        umulh x11, x4, x8
        adcs x20, x20, x11
        umulh x11, x4, x9
        adcs x21, x21, x11
        umulh x11, x4, x10
        adc x22, x22, x11
        ldp x3, x4, [sp, #(48*5)+32]
        mul x11, x3, x5
        adds x16, x16, x11
        mul x11, x3, x6
        adcs x17, x17, x11
        mul x11, x3, x7
        adcs x19, x19, x11
        mul x11, x3, x8
        adcs x20, x20, x11
        mul x11, x3, x9
        adcs x21, x21, x11
        mul x11, x3, x10
        adcs x22, x22, x11
        cset x2, cs
        umulh x11, x3, x5
        adds x17, x17, x11
        umulh x11, x3, x6
        adcs x19, x19, x11
        umulh x11, x3, x7
        adcs x20, x20, x11
        umulh x11, x3, x8
        adcs x21, x21, x11
        umulh x11, x3, x9
        adcs x22, x22, x11
        umulh x11, x3, x10
        adc x2, x2, x11
        mul x11, x4, x5
        adds x17, x17, x11
        mul x11, x4, x6
        adcs x19, x19, x11
        mul x11, x4, x7
        adcs x20, x20, x11
        mul x11, x4, x8
        adcs x21, x21, x11
        mul x11, x4, x9
        adcs x22, x22, x11
        mul x11, x4, x10
        adcs x2, x2, x11
        cset x1, cs
        umulh x11, x4, x5
        adds x19, x19, x11
        umulh x11, x4, x6
        adcs x20, x20, x11
        umulh x11, x4, x7
        adcs x21, x21, x11
        umulh x11, x4, x8
        adcs x22, x22, x11
        umulh x11, x4, x9
        adcs x2, x2, x11
        umulh x11, x4, x10
        adc x1, x1, x11
        lsl x7, x12, #32
        add x12, x7, x12
        mov x7, #0xffffffff00000001
        umulh x7, x7, x12
        mov x6, #0xffffffff
        mul x5, x6, x12
        umulh x6, x6, x12
        adds x7, x7, x5
        adcs x6, x6, x12
        adc x5, xzr, xzr
        subs x13, x13, x7
        sbcs x14, x14, x6
        sbcs x15, x15, x5
        sbcs x16, x16, xzr
        sbcs x17, x17, xzr
        sbc x12, x12, xzr
        lsl x7, x13, #32
        add x13, x7, x13
        mov x7, #0xffffffff00000001
        umulh x7, x7, x13
        mov x6, #0xffffffff
        mul x5, x6, x13
        umulh x6, x6, x13
        adds x7, x7, x5
        adcs x6, x6, x13
        adc x5, xzr, xzr
        subs x14, x14, x7
        sbcs x15, x15, x6
        sbcs x16, x16, x5
        sbcs x17, x17, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        lsl x7, x14, #32
        add x14, x7, x14
        mov x7, #0xffffffff00000001
        umulh x7, x7, x14
        mov x6, #0xffffffff
        mul x5, x6, x14
        umulh x6, x6, x14
        adds x7, x7, x5
        adcs x6, x6, x14
        adc x5, xzr, xzr
        subs x15, x15, x7
        sbcs x16, x16, x6
        sbcs x17, x17, x5
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x14, x14, xzr
        lsl x7, x15, #32
        add x15, x7, x15
        mov x7, #0xffffffff00000001
        umulh x7, x7, x15
        mov x6, #0xffffffff
        mul x5, x6, x15
        umulh x6, x6, x15
        adds x7, x7, x5
        adcs x6, x6, x15
        adc x5, xzr, xzr
        subs x16, x16, x7
        sbcs x17, x17, x6
        sbcs x12, x12, x5
        sbcs x13, x13, xzr
        sbcs x14, x14, xzr
        sbc x15, x15, xzr
        lsl x7, x16, #32
        add x16, x7, x16
        mov x7, #0xffffffff00000001
        umulh x7, x7, x16
        mov x6, #0xffffffff
        mul x5, x6, x16
        umulh x6, x6, x16
        adds x7, x7, x5
        adcs x6, x6, x16
        adc x5, xzr, xzr
        subs x17, x17, x7
        sbcs x12, x12, x6
        sbcs x13, x13, x5
        sbcs x14, x14, xzr
        sbcs x15, x15, xzr
        sbc x16, x16, xzr
        lsl x7, x17, #32
        add x17, x7, x17
        mov x7, #0xffffffff00000001
        umulh x7, x7, x17
        mov x6, #0xffffffff
        mul x5, x6, x17
        umulh x6, x6, x17
        adds x7, x7, x5
        adcs x6, x6, x17
        adc x5, xzr, xzr
        subs x12, x12, x7
        sbcs x13, x13, x6
        sbcs x14, x14, x5
        sbcs x15, x15, xzr
        sbcs x16, x16, xzr
        sbc x17, x17, xzr
        adds x12, x12, x19
        adcs x13, x13, x20
        adcs x14, x14, x21
        adcs x15, x15, x22
        adcs x16, x16, x2
        adcs x17, x17, x1
        adc x10, xzr, xzr
        mov x11, #0xffffffff00000001
        adds x19, x12, x11
        mov x11, #0xffffffff
        adcs x20, x13, x11
        mov x11, #0x1
        adcs x21, x14, x11
        adcs x22, x15, xzr
        adcs x2, x16, xzr
        adcs x1, x17, xzr
        adcs x10, x10, xzr
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        csel x14, x14, x21, eq
        csel x15, x15, x22, eq
        csel x16, x16, x2, eq
        csel x17, x17, x1, eq
        stp x12, x13, [sp, #(48*2)]
        stp x14, x15, [sp, #(48*2)+16]
        stp x16, x17, [sp, #(48*2)+32]





        ldp x5, x6, [x24, #48]
        ldp x4, x3, [x24, #(2*48)]
        adds x5, x5, x4
        adcs x6, x6, x3
        ldp x7, x8, [x24, #48 +16]
        ldp x4, x3, [x24, #(2*48)+16]
        adcs x7, x7, x4
        adcs x8, x8, x3
        ldp x9, x10, [x24, #48 +32]
        ldp x4, x3, [x24, #(2*48)+32]
        adcs x9, x9, x4
        adcs x10, x10, x3
        adc x3, xzr, xzr
        mov x4, #0xffffffff
        cmp x5, x4
        mov x4, #0xffffffff00000000
        sbcs xzr, x6, x4
        mov x4, #0xfffffffffffffffe
        sbcs xzr, x7, x4
        adcs xzr, x8, xzr
        adcs xzr, x9, xzr
        adcs xzr, x10, xzr
        adcs x3, x3, xzr
        csetm x3, ne
        mov x4, #0xffffffff
        and x4, x4, x3
        subs x5, x5, x4
        eor x4, x4, x3
        sbcs x6, x6, x4
        mov x4, #0xfffffffffffffffe
        and x4, x4, x3
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        sbcs x9, x9, x3
        sbc x10, x10, x3
        stp x5, x6, [sp, #(48*5)]
        stp x7, x8, [sp, #(48*5)+16]
        stp x9, x10, [sp, #(48*5)+32]
        ldp x2, x3, [sp, #(48*2)]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #(48*2)+16]
        mul x8, x2, x4
        adds x10, x10, x8
        mul x11, x2, x5
        mul x8, x3, x4
        adcs x11, x11, x8
        umulh x12, x2, x5
        mul x8, x3, x5
        adcs x12, x12, x8
        ldp x6, x7, [sp, #(48*2)+32]
        mul x13, x2, x7
        mul x8, x3, x6
        adcs x13, x13, x8
        umulh x14, x2, x7
        mul x8, x3, x7
        adcs x14, x14, x8
        mul x15, x5, x6
        adcs x15, x15, xzr
        umulh x16, x5, x6
        adc x16, x16, xzr
        umulh x8, x2, x4
        adds x11, x11, x8
        umulh x8, x3, x4
        adcs x12, x12, x8
        umulh x8, x3, x5
        adcs x13, x13, x8
        umulh x8, x3, x6
        adcs x14, x14, x8
        umulh x8, x3, x7
        adcs x15, x15, x8
        adc x16, x16, xzr
        mul x8, x2, x6
        adds x12, x12, x8
        mul x8, x4, x5
        adcs x13, x13, x8
        mul x8, x4, x6
        adcs x14, x14, x8
        mul x8, x4, x7
        adcs x15, x15, x8
        mul x8, x5, x7
        adcs x16, x16, x8
        mul x17, x6, x7
        adcs x17, x17, xzr
        umulh x19, x6, x7
        adc x19, x19, xzr
        umulh x8, x2, x6
        adds x13, x13, x8
        umulh x8, x4, x5
        adcs x14, x14, x8
        umulh x8, x4, x6
        adcs x15, x15, x8
        umulh x8, x4, x7
        adcs x16, x16, x8
        umulh x8, x5, x7
        adcs x17, x17, x8
        adc x19, x19, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        adcs x15, x15, x15
        adcs x16, x16, x16
        adcs x17, x17, x17
        adcs x19, x19, x19
        cset x20, hs
        umulh x8, x2, x2
        mul x2, x2, x2
        adds x9, x9, x8
        mul x8, x3, x3
        adcs x10, x10, x8
        umulh x8, x3, x3
        adcs x11, x11, x8
        mul x8, x4, x4
        adcs x12, x12, x8
        umulh x8, x4, x4
        adcs x13, x13, x8
        mul x8, x5, x5
        adcs x14, x14, x8
        umulh x8, x5, x5
        adcs x15, x15, x8
        mul x8, x6, x6
        adcs x16, x16, x8
        umulh x8, x6, x6
        adcs x17, x17, x8
        mul x8, x7, x7
        adcs x19, x19, x8
        umulh x8, x7, x7
        adc x20, x20, x8
        lsl x5, x2, #32
        add x2, x5, x2
        mov x5, #-4294967295
        umulh x5, x5, x2
        mov x4, #4294967295
        mul x3, x4, x2
        umulh x4, x4, x2
        adds x5, x5, x3
        adcs x4, x4, x2
        adc x3, xzr, xzr
        subs x9, x9, x5
        sbcs x10, x10, x4
        sbcs x11, x11, x3
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x2, x2, xzr
        lsl x5, x9, #32
        add x9, x5, x9
        mov x5, #-4294967295
        umulh x5, x5, x9
        mov x4, #4294967295
        mul x3, x4, x9
        umulh x4, x4, x9
        adds x5, x5, x3
        adcs x4, x4, x9
        adc x3, xzr, xzr
        subs x10, x10, x5
        sbcs x11, x11, x4
        sbcs x12, x12, x3
        sbcs x13, x13, xzr
        sbcs x2, x2, xzr
        sbc x9, x9, xzr
        lsl x5, x10, #32
        add x10, x5, x10
        mov x5, #-4294967295
        umulh x5, x5, x10
        mov x4, #4294967295
        mul x3, x4, x10
        umulh x4, x4, x10
        adds x5, x5, x3
        adcs x4, x4, x10
        adc x3, xzr, xzr
        subs x11, x11, x5
        sbcs x12, x12, x4
        sbcs x13, x13, x3
        sbcs x2, x2, xzr
        sbcs x9, x9, xzr
        sbc x10, x10, xzr
        lsl x5, x11, #32
        add x11, x5, x11
        mov x5, #-4294967295
        umulh x5, x5, x11
        mov x4, #4294967295
        mul x3, x4, x11
        umulh x4, x4, x11
        adds x5, x5, x3
        adcs x4, x4, x11
        adc x3, xzr, xzr
        subs x12, x12, x5
        sbcs x13, x13, x4
        sbcs x2, x2, x3
        sbcs x9, x9, xzr
        sbcs x10, x10, xzr
        sbc x11, x11, xzr
        lsl x5, x12, #32
        add x12, x5, x12
        mov x5, #-4294967295
        umulh x5, x5, x12
        mov x4, #4294967295
        mul x3, x4, x12
        umulh x4, x4, x12
        adds x5, x5, x3
        adcs x4, x4, x12
        adc x3, xzr, xzr
        subs x13, x13, x5
        sbcs x2, x2, x4
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbcs x11, x11, xzr
        sbc x12, x12, xzr
        lsl x5, x13, #32
        add x13, x5, x13
        mov x5, #-4294967295
        umulh x5, x5, x13
        mov x4, #4294967295
        mul x3, x4, x13
        umulh x4, x4, x13
        adds x5, x5, x3
        adcs x4, x4, x13
        adc x3, xzr, xzr
        subs x2, x2, x5
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        sbcs x11, x11, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        adds x2, x2, x14
        adcs x9, x9, x15
        adcs x10, x10, x16
        adcs x11, x11, x17
        adcs x12, x12, x19
        adcs x13, x13, x20
        adc x6, xzr, xzr
        mov x8, #-4294967295
        adds x14, x2, x8
        mov x8, #4294967295
        adcs x15, x9, x8
        mov x8, #1
        adcs x16, x10, x8
        adcs x17, x11, xzr
        adcs x19, x12, xzr
        adcs x20, x13, xzr
        adcs x6, x6, xzr
        csel x2, x2, x14, eq
        csel x9, x9, x15, eq
        csel x10, x10, x16, eq
        csel x11, x11, x17, eq
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        stp x2, x9, [sp, #(48*6)]
        stp x10, x11, [sp, #(48*6)+16]
        stp x12, x13, [sp, #(48*6)+32]
        ldp x3, x4, [x24, #0]
        ldp x5, x6, [sp, #(48*1)]
        mul x12, x3, x5
        umulh x13, x3, x5
        mul x11, x3, x6
        umulh x14, x3, x6
        adds x13, x13, x11
        ldp x7, x8, [sp, #(48*1)+16]
        mul x11, x3, x7
        umulh x15, x3, x7
        adcs x14, x14, x11
        mul x11, x3, x8
        umulh x16, x3, x8
        adcs x15, x15, x11
        ldp x9, x10, [sp, #(48*1)+32]
        mul x11, x3, x9
        umulh x17, x3, x9
        adcs x16, x16, x11
        mul x11, x3, x10
        umulh x19, x3, x10
        adcs x17, x17, x11
        adc x19, x19, xzr
        mul x11, x4, x5
        adds x13, x13, x11
        mul x11, x4, x6
        adcs x14, x14, x11
        mul x11, x4, x7
        adcs x15, x15, x11
        mul x11, x4, x8
        adcs x16, x16, x11
        mul x11, x4, x9
        adcs x17, x17, x11
        mul x11, x4, x10
        adcs x19, x19, x11
        cset x20, cs
        umulh x11, x4, x5
        adds x14, x14, x11
        umulh x11, x4, x6
        adcs x15, x15, x11
        umulh x11, x4, x7
        adcs x16, x16, x11
        umulh x11, x4, x8
        adcs x17, x17, x11
        umulh x11, x4, x9
        adcs x19, x19, x11
        umulh x11, x4, x10
        adc x20, x20, x11
        ldp x3, x4, [x24, #0 +16]
        mul x11, x3, x5
        adds x14, x14, x11
        mul x11, x3, x6
        adcs x15, x15, x11
        mul x11, x3, x7
        adcs x16, x16, x11
        mul x11, x3, x8
        adcs x17, x17, x11
        mul x11, x3, x9
        adcs x19, x19, x11
        mul x11, x3, x10
        adcs x20, x20, x11
        cset x21, cs
        umulh x11, x3, x5
        adds x15, x15, x11
        umulh x11, x3, x6
        adcs x16, x16, x11
        umulh x11, x3, x7
        adcs x17, x17, x11
        umulh x11, x3, x8
        adcs x19, x19, x11
        umulh x11, x3, x9
        adcs x20, x20, x11
        umulh x11, x3, x10
        adc x21, x21, x11
        mul x11, x4, x5
        adds x15, x15, x11
        mul x11, x4, x6
        adcs x16, x16, x11
        mul x11, x4, x7
        adcs x17, x17, x11
        mul x11, x4, x8
        adcs x19, x19, x11
        mul x11, x4, x9
        adcs x20, x20, x11
        mul x11, x4, x10
        adcs x21, x21, x11
        cset x22, cs
        umulh x11, x4, x5
        adds x16, x16, x11
        umulh x11, x4, x6
        adcs x17, x17, x11
        umulh x11, x4, x7
        adcs x19, x19, x11
        umulh x11, x4, x8
        adcs x20, x20, x11
        umulh x11, x4, x9
        adcs x21, x21, x11
        umulh x11, x4, x10
        adc x22, x22, x11
        ldp x3, x4, [x24, #0 +32]
        mul x11, x3, x5
        adds x16, x16, x11
        mul x11, x3, x6
        adcs x17, x17, x11
        mul x11, x3, x7
        adcs x19, x19, x11
        mul x11, x3, x8
        adcs x20, x20, x11
        mul x11, x3, x9
        adcs x21, x21, x11
        mul x11, x3, x10
        adcs x22, x22, x11
        cset x2, cs
        umulh x11, x3, x5
        adds x17, x17, x11
        umulh x11, x3, x6
        adcs x19, x19, x11
        umulh x11, x3, x7
        adcs x20, x20, x11
        umulh x11, x3, x8
        adcs x21, x21, x11
        umulh x11, x3, x9
        adcs x22, x22, x11
        umulh x11, x3, x10
        adc x2, x2, x11
        mul x11, x4, x5
        adds x17, x17, x11
        mul x11, x4, x6
        adcs x19, x19, x11
        mul x11, x4, x7
        adcs x20, x20, x11
        mul x11, x4, x8
        adcs x21, x21, x11
        mul x11, x4, x9
        adcs x22, x22, x11
        mul x11, x4, x10
        adcs x2, x2, x11
        cset x1, cs
        umulh x11, x4, x5
        adds x19, x19, x11
        umulh x11, x4, x6
        adcs x20, x20, x11
        umulh x11, x4, x7
        adcs x21, x21, x11
        umulh x11, x4, x8
        adcs x22, x22, x11
        umulh x11, x4, x9
        adcs x2, x2, x11
        umulh x11, x4, x10
        adc x1, x1, x11
        lsl x7, x12, #32
        add x12, x7, x12
        mov x7, #0xffffffff00000001
        umulh x7, x7, x12
        mov x6, #0xffffffff
        mul x5, x6, x12
        umulh x6, x6, x12
        adds x7, x7, x5
        adcs x6, x6, x12
        adc x5, xzr, xzr
        subs x13, x13, x7
        sbcs x14, x14, x6
        sbcs x15, x15, x5
        sbcs x16, x16, xzr
        sbcs x17, x17, xzr
        sbc x12, x12, xzr
        lsl x7, x13, #32
        add x13, x7, x13
        mov x7, #0xffffffff00000001
        umulh x7, x7, x13
        mov x6, #0xffffffff
        mul x5, x6, x13
        umulh x6, x6, x13
        adds x7, x7, x5
        adcs x6, x6, x13
        adc x5, xzr, xzr
        subs x14, x14, x7
        sbcs x15, x15, x6
        sbcs x16, x16, x5
        sbcs x17, x17, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        lsl x7, x14, #32
        add x14, x7, x14
        mov x7, #0xffffffff00000001
        umulh x7, x7, x14
        mov x6, #0xffffffff
        mul x5, x6, x14
        umulh x6, x6, x14
        adds x7, x7, x5
        adcs x6, x6, x14
        adc x5, xzr, xzr
        subs x15, x15, x7
        sbcs x16, x16, x6
        sbcs x17, x17, x5
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x14, x14, xzr
        lsl x7, x15, #32
        add x15, x7, x15
        mov x7, #0xffffffff00000001
        umulh x7, x7, x15
        mov x6, #0xffffffff
        mul x5, x6, x15
        umulh x6, x6, x15
        adds x7, x7, x5
        adcs x6, x6, x15
        adc x5, xzr, xzr
        subs x16, x16, x7
        sbcs x17, x17, x6
        sbcs x12, x12, x5
        sbcs x13, x13, xzr
        sbcs x14, x14, xzr
        sbc x15, x15, xzr
        lsl x7, x16, #32
        add x16, x7, x16
        mov x7, #0xffffffff00000001
        umulh x7, x7, x16
        mov x6, #0xffffffff
        mul x5, x6, x16
        umulh x6, x6, x16
        adds x7, x7, x5
        adcs x6, x6, x16
        adc x5, xzr, xzr
        subs x17, x17, x7
        sbcs x12, x12, x6
        sbcs x13, x13, x5
        sbcs x14, x14, xzr
        sbcs x15, x15, xzr
        sbc x16, x16, xzr
        lsl x7, x17, #32
        add x17, x7, x17
        mov x7, #0xffffffff00000001
        umulh x7, x7, x17
        mov x6, #0xffffffff
        mul x5, x6, x17
        umulh x6, x6, x17
        adds x7, x7, x5
        adcs x6, x6, x17
        adc x5, xzr, xzr
        subs x12, x12, x7
        sbcs x13, x13, x6
        sbcs x14, x14, x5
        sbcs x15, x15, xzr
        sbcs x16, x16, xzr
        sbc x17, x17, xzr
        adds x12, x12, x19
        adcs x13, x13, x20
        adcs x14, x14, x21
        adcs x15, x15, x22
        adcs x16, x16, x2
        adcs x17, x17, x1
        adc x10, xzr, xzr
        mov x11, #0xffffffff00000001
        adds x19, x12, x11
        mov x11, #0xffffffff
        adcs x20, x13, x11
        mov x11, #0x1
        adcs x21, x14, x11
        adcs x22, x15, xzr
        adcs x2, x16, xzr
        adcs x1, x17, xzr
        adcs x10, x10, xzr
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        csel x14, x14, x21, eq
        csel x15, x15, x22, eq
        csel x16, x16, x2, eq
        csel x17, x17, x1, eq
        stp x12, x13, [sp, #(48*3)]
        stp x14, x15, [sp, #(48*3)+16]
        stp x16, x17, [sp, #(48*3)+32]



        ldp x2, x3, [sp, #(48*5)]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #(48*5)+16]
        mul x8, x2, x4
        adds x10, x10, x8
        mul x11, x2, x5
        mul x8, x3, x4
        adcs x11, x11, x8
        umulh x12, x2, x5
        mul x8, x3, x5
        adcs x12, x12, x8
        ldp x6, x7, [sp, #(48*5)+32]
        mul x13, x2, x7
        mul x8, x3, x6
        adcs x13, x13, x8
        umulh x14, x2, x7
        mul x8, x3, x7
        adcs x14, x14, x8
        mul x15, x5, x6
        adcs x15, x15, xzr
        umulh x16, x5, x6
        adc x16, x16, xzr
        umulh x8, x2, x4
        adds x11, x11, x8
        umulh x8, x3, x4
        adcs x12, x12, x8
        umulh x8, x3, x5
        adcs x13, x13, x8
        umulh x8, x3, x6
        adcs x14, x14, x8
        umulh x8, x3, x7
        adcs x15, x15, x8
        adc x16, x16, xzr
        mul x8, x2, x6
        adds x12, x12, x8
        mul x8, x4, x5
        adcs x13, x13, x8
        mul x8, x4, x6
        adcs x14, x14, x8
        mul x8, x4, x7
        adcs x15, x15, x8
        mul x8, x5, x7
        adcs x16, x16, x8
        mul x17, x6, x7
        adcs x17, x17, xzr
        umulh x19, x6, x7
        adc x19, x19, xzr
        umulh x8, x2, x6
        adds x13, x13, x8
        umulh x8, x4, x5
        adcs x14, x14, x8
        umulh x8, x4, x6
        adcs x15, x15, x8
        umulh x8, x4, x7
        adcs x16, x16, x8
        umulh x8, x5, x7
        adcs x17, x17, x8
        adc x19, x19, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        adcs x15, x15, x15
        adcs x16, x16, x16
        adcs x17, x17, x17
        adcs x19, x19, x19
        cset x20, hs
        umulh x8, x2, x2
        mul x2, x2, x2
        adds x9, x9, x8
        mul x8, x3, x3
        adcs x10, x10, x8
        umulh x8, x3, x3
        adcs x11, x11, x8
        mul x8, x4, x4
        adcs x12, x12, x8
        umulh x8, x4, x4
        adcs x13, x13, x8
        mul x8, x5, x5
        adcs x14, x14, x8
        umulh x8, x5, x5
        adcs x15, x15, x8
        mul x8, x6, x6
        adcs x16, x16, x8
        umulh x8, x6, x6
        adcs x17, x17, x8
        mul x8, x7, x7
        adcs x19, x19, x8
        umulh x8, x7, x7
        adc x20, x20, x8
        lsl x5, x2, #32
        add x2, x5, x2
        mov x5, #-4294967295
        umulh x5, x5, x2
        mov x4, #4294967295
        mul x3, x4, x2
        umulh x4, x4, x2
        adds x5, x5, x3
        adcs x4, x4, x2
        adc x3, xzr, xzr
        subs x9, x9, x5
        sbcs x10, x10, x4
        sbcs x11, x11, x3
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x2, x2, xzr
        lsl x5, x9, #32
        add x9, x5, x9
        mov x5, #-4294967295
        umulh x5, x5, x9
        mov x4, #4294967295
        mul x3, x4, x9
        umulh x4, x4, x9
        adds x5, x5, x3
        adcs x4, x4, x9
        adc x3, xzr, xzr
        subs x10, x10, x5
        sbcs x11, x11, x4
        sbcs x12, x12, x3
        sbcs x13, x13, xzr
        sbcs x2, x2, xzr
        sbc x9, x9, xzr
        lsl x5, x10, #32
        add x10, x5, x10
        mov x5, #-4294967295
        umulh x5, x5, x10
        mov x4, #4294967295
        mul x3, x4, x10
        umulh x4, x4, x10
        adds x5, x5, x3
        adcs x4, x4, x10
        adc x3, xzr, xzr
        subs x11, x11, x5
        sbcs x12, x12, x4
        sbcs x13, x13, x3
        sbcs x2, x2, xzr
        sbcs x9, x9, xzr
        sbc x10, x10, xzr
        lsl x5, x11, #32
        add x11, x5, x11
        mov x5, #-4294967295
        umulh x5, x5, x11
        mov x4, #4294967295
        mul x3, x4, x11
        umulh x4, x4, x11
        adds x5, x5, x3
        adcs x4, x4, x11
        adc x3, xzr, xzr
        subs x12, x12, x5
        sbcs x13, x13, x4
        sbcs x2, x2, x3
        sbcs x9, x9, xzr
        sbcs x10, x10, xzr
        sbc x11, x11, xzr
        lsl x5, x12, #32
        add x12, x5, x12
        mov x5, #-4294967295
        umulh x5, x5, x12
        mov x4, #4294967295
        mul x3, x4, x12
        umulh x4, x4, x12
        adds x5, x5, x3
        adcs x4, x4, x12
        adc x3, xzr, xzr
        subs x13, x13, x5
        sbcs x2, x2, x4
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbcs x11, x11, xzr
        sbc x12, x12, xzr
        lsl x5, x13, #32
        add x13, x5, x13
        mov x5, #-4294967295
        umulh x5, x5, x13
        mov x4, #4294967295
        mul x3, x4, x13
        umulh x4, x4, x13
        adds x5, x5, x3
        adcs x4, x4, x13
        adc x3, xzr, xzr
        subs x2, x2, x5
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        sbcs x11, x11, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        adds x2, x2, x14
        adcs x9, x9, x15
        adcs x10, x10, x16
        adcs x11, x11, x17
        adcs x12, x12, x19
        adcs x13, x13, x20
        adc x6, xzr, xzr
        mov x8, #-4294967295
        adds x14, x2, x8
        mov x8, #4294967295
        adcs x15, x9, x8
        mov x8, #1
        adcs x16, x10, x8
        adcs x17, x11, xzr
        adcs x19, x12, xzr
        adcs x20, x13, xzr
        adcs x6, x6, xzr
        csel x2, x2, x14, eq
        csel x9, x9, x15, eq
        csel x10, x10, x16, eq
        csel x11, x11, x17, eq
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        stp x2, x9, [sp, #(48*4)]
        stp x10, x11, [sp, #(48*4)+16]
        stp x12, x13, [sp, #(48*4)+32]




        ldp x0, x1, [sp, #(48*6)]
        mov x6, #0x00000000ffffffff
        subs x6, x6, x0
        mov x7, #0xffffffff00000000
        sbcs x7, x7, x1
        ldp x0, x1, [sp, #(48*6)+16]
        mov x8, #0xfffffffffffffffe
        sbcs x8, x8, x0
        mov x13, #0xffffffffffffffff
        sbcs x9, x13, x1
        ldp x0, x1, [sp, #(48*6)+32]
        sbcs x10, x13, x0
        sbc x11, x13, x1
        mov x12, 9
        mul x0, x12, x6
        mul x1, x12, x7
        mul x2, x12, x8
        mul x3, x12, x9
        mul x4, x12, x10
        mul x5, x12, x11
        umulh x6, x12, x6
        umulh x7, x12, x7
        umulh x8, x12, x8
        umulh x9, x12, x9
        umulh x10, x12, x10
        umulh x12, x12, x11
        adds x1, x1, x6
        adcs x2, x2, x7
        adcs x3, x3, x8
        adcs x4, x4, x9
        adcs x5, x5, x10
        mov x6, #1
        adc x6, x12, x6
        ldp x8, x9, [sp, #(48*3)]
        ldp x10, x11, [sp, #(48*3)+16]
        ldp x12, x13, [sp, #(48*3)+32]
        mov x14, 12
        mul x15, x14, x8
        umulh x8, x14, x8
        adds x0, x0, x15
        mul x15, x14, x9
        umulh x9, x14, x9
        adcs x1, x1, x15
        mul x15, x14, x10
        umulh x10, x14, x10
        adcs x2, x2, x15
        mul x15, x14, x11
        umulh x11, x14, x11
        adcs x3, x3, x15
        mul x15, x14, x12
        umulh x12, x14, x12
        adcs x4, x4, x15
        mul x15, x14, x13
        umulh x13, x14, x13
        adcs x5, x5, x15
        adc x6, x6, xzr
        adds x1, x1, x8
        adcs x2, x2, x9
        adcs x3, x3, x10
        adcs x4, x4, x11
        adcs x5, x5, x12
        adcs x6, x6, x13
        lsl x7, x6, #32
        subs x8, x6, x7
        sbc x7, x7, xzr
        adds x0, x0, x8
        adcs x1, x1, x7
        adcs x2, x2, x6
        adcs x3, x3, xzr
        adcs x4, x4, xzr
        adcs x5, x5, xzr
        csetm x6, cc
        mov x7, #0xffffffff
        and x7, x7, x6
        adds x0, x0, x7
        eor x7, x7, x6
        adcs x1, x1, x7
        mov x7, #0xfffffffffffffffe
        and x7, x7, x6
        adcs x2, x2, x7
        adcs x3, x3, x6
        adcs x4, x4, x6
        adc x5, x5, x6
        stp x0, x1, [sp, #(48*6)]
        stp x2, x3, [sp, #(48*6)+16]
        stp x4, x5, [sp, #(48*6)+32]
        ldp x5, x6, [sp, #(48*4)]
        ldp x4, x3, [sp, #(48*0)]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #(48*4)+16]
        ldp x4, x3, [sp, #(48*0)+16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        ldp x9, x10, [sp, #(48*4)+32]
        ldp x4, x3, [sp, #(48*0)+32]
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        csetm x3, lo
        mov x4, #4294967295
        and x4, x4, x3
        adds x5, x5, x4
        eor x4, x4, x3
        adcs x6, x6, x4
        mov x4, #-2
        and x4, x4, x3
        adcs x7, x7, x4
        adcs x8, x8, x3
        adcs x9, x9, x3
        adc x10, x10, x3
        stp x5, x6, [sp, #(48*5)]
        stp x7, x8, [sp, #(48*5)+16]
        stp x9, x10, [sp, #(48*5)+32]



        ldp x2, x3, [sp, #(48*1)]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #(48*1)+16]
        mul x8, x2, x4
        adds x10, x10, x8
        mul x11, x2, x5
        mul x8, x3, x4
        adcs x11, x11, x8
        umulh x12, x2, x5
        mul x8, x3, x5
        adcs x12, x12, x8
        ldp x6, x7, [sp, #(48*1)+32]
        mul x13, x2, x7
        mul x8, x3, x6
        adcs x13, x13, x8
        umulh x14, x2, x7
        mul x8, x3, x7
        adcs x14, x14, x8
        mul x15, x5, x6
        adcs x15, x15, xzr
        umulh x16, x5, x6
        adc x16, x16, xzr
        umulh x8, x2, x4
        adds x11, x11, x8
        umulh x8, x3, x4
        adcs x12, x12, x8
        umulh x8, x3, x5
        adcs x13, x13, x8
        umulh x8, x3, x6
        adcs x14, x14, x8
        umulh x8, x3, x7
        adcs x15, x15, x8
        adc x16, x16, xzr
        mul x8, x2, x6
        adds x12, x12, x8
        mul x8, x4, x5
        adcs x13, x13, x8
        mul x8, x4, x6
        adcs x14, x14, x8
        mul x8, x4, x7
        adcs x15, x15, x8
        mul x8, x5, x7
        adcs x16, x16, x8
        mul x17, x6, x7
        adcs x17, x17, xzr
        umulh x19, x6, x7
        adc x19, x19, xzr
        umulh x8, x2, x6
        adds x13, x13, x8
        umulh x8, x4, x5
        adcs x14, x14, x8
        umulh x8, x4, x6
        adcs x15, x15, x8
        umulh x8, x4, x7
        adcs x16, x16, x8
        umulh x8, x5, x7
        adcs x17, x17, x8
        adc x19, x19, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        adcs x15, x15, x15
        adcs x16, x16, x16
        adcs x17, x17, x17
        adcs x19, x19, x19
        cset x20, hs
        umulh x8, x2, x2
        mul x2, x2, x2
        adds x9, x9, x8
        mul x8, x3, x3
        adcs x10, x10, x8
        umulh x8, x3, x3
        adcs x11, x11, x8
        mul x8, x4, x4
        adcs x12, x12, x8
        umulh x8, x4, x4
        adcs x13, x13, x8
        mul x8, x5, x5
        adcs x14, x14, x8
        umulh x8, x5, x5
        adcs x15, x15, x8
        mul x8, x6, x6
        adcs x16, x16, x8
        umulh x8, x6, x6
        adcs x17, x17, x8
        mul x8, x7, x7
        adcs x19, x19, x8
        umulh x8, x7, x7
        adc x20, x20, x8
        lsl x5, x2, #32
        add x2, x5, x2
        mov x5, #-4294967295
        umulh x5, x5, x2
        mov x4, #4294967295
        mul x3, x4, x2
        umulh x4, x4, x2
        adds x5, x5, x3
        adcs x4, x4, x2
        adc x3, xzr, xzr
        subs x9, x9, x5
        sbcs x10, x10, x4
        sbcs x11, x11, x3
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x2, x2, xzr
        lsl x5, x9, #32
        add x9, x5, x9
        mov x5, #-4294967295
        umulh x5, x5, x9
        mov x4, #4294967295
        mul x3, x4, x9
        umulh x4, x4, x9
        adds x5, x5, x3
        adcs x4, x4, x9
        adc x3, xzr, xzr
        subs x10, x10, x5
        sbcs x11, x11, x4
        sbcs x12, x12, x3
        sbcs x13, x13, xzr
        sbcs x2, x2, xzr
        sbc x9, x9, xzr
        lsl x5, x10, #32
        add x10, x5, x10
        mov x5, #-4294967295
        umulh x5, x5, x10
        mov x4, #4294967295
        mul x3, x4, x10
        umulh x4, x4, x10
        adds x5, x5, x3
        adcs x4, x4, x10
        adc x3, xzr, xzr
        subs x11, x11, x5
        sbcs x12, x12, x4
        sbcs x13, x13, x3
        sbcs x2, x2, xzr
        sbcs x9, x9, xzr
        sbc x10, x10, xzr
        lsl x5, x11, #32
        add x11, x5, x11
        mov x5, #-4294967295
        umulh x5, x5, x11
        mov x4, #4294967295
        mul x3, x4, x11
        umulh x4, x4, x11
        adds x5, x5, x3
        adcs x4, x4, x11
        adc x3, xzr, xzr
        subs x12, x12, x5
        sbcs x13, x13, x4
        sbcs x2, x2, x3
        sbcs x9, x9, xzr
        sbcs x10, x10, xzr
        sbc x11, x11, xzr
        lsl x5, x12, #32
        add x12, x5, x12
        mov x5, #-4294967295
        umulh x5, x5, x12
        mov x4, #4294967295
        mul x3, x4, x12
        umulh x4, x4, x12
        adds x5, x5, x3
        adcs x4, x4, x12
        adc x3, xzr, xzr
        subs x13, x13, x5
        sbcs x2, x2, x4
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbcs x11, x11, xzr
        sbc x12, x12, xzr
        lsl x5, x13, #32
        add x13, x5, x13
        mov x5, #-4294967295
        umulh x5, x5, x13
        mov x4, #4294967295
        mul x3, x4, x13
        umulh x4, x4, x13
        adds x5, x5, x3
        adcs x4, x4, x13
        adc x3, xzr, xzr
        subs x2, x2, x5
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        sbcs x11, x11, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        adds x2, x2, x14
        adcs x9, x9, x15
        adcs x10, x10, x16
        adcs x11, x11, x17
        adcs x12, x12, x19
        adcs x13, x13, x20
        adc x6, xzr, xzr
        mov x8, #-4294967295
        adds x14, x2, x8
        mov x8, #4294967295
        adcs x15, x9, x8
        mov x8, #1
        adcs x16, x10, x8
        adcs x17, x11, xzr
        adcs x19, x12, xzr
        adcs x20, x13, xzr
        adcs x6, x6, xzr
        csel x2, x2, x14, eq
        csel x9, x9, x15, eq
        csel x10, x10, x16, eq
        csel x11, x11, x17, eq
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        stp x2, x9, [sp, #(48*4)]
        stp x10, x11, [sp, #(48*4)+16]
        stp x12, x13, [sp, #(48*4)+32]




        ldp x5, x6, [sp, #(48*5)]
        ldp x4, x3, [sp, #(48*1)]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #(48*5)+16]
        ldp x4, x3, [sp, #(48*1)+16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        ldp x9, x10, [sp, #(48*5)+32]
        ldp x4, x3, [sp, #(48*1)+32]
        sbcs x9, x9, x4
        sbcs x10, x10, x3
        csetm x3, lo
        mov x4, #4294967295
        and x4, x4, x3
        adds x5, x5, x4
        eor x4, x4, x3
        adcs x6, x6, x4
        mov x4, #-2
        and x4, x4, x3
        adcs x7, x7, x4
        adcs x8, x8, x3
        adcs x9, x9, x3
        adc x10, x10, x3
        stp x5, x6, [x23, #(2*48)]
        stp x7, x8, [x23, #(2*48)+16]
        stp x9, x10, [x23, #(2*48)+32]
        ldp x3, x4, [sp, #(48*6)]
        ldp x5, x6, [sp, #(48*2)]
        mul x12, x3, x5
        umulh x13, x3, x5
        mul x11, x3, x6
        umulh x14, x3, x6
        adds x13, x13, x11
        ldp x7, x8, [sp, #(48*2)+16]
        mul x11, x3, x7
        umulh x15, x3, x7
        adcs x14, x14, x11
        mul x11, x3, x8
        umulh x16, x3, x8
        adcs x15, x15, x11
        ldp x9, x10, [sp, #(48*2)+32]
        mul x11, x3, x9
        umulh x17, x3, x9
        adcs x16, x16, x11
        mul x11, x3, x10
        umulh x19, x3, x10
        adcs x17, x17, x11
        adc x19, x19, xzr
        mul x11, x4, x5
        adds x13, x13, x11
        mul x11, x4, x6
        adcs x14, x14, x11
        mul x11, x4, x7
        adcs x15, x15, x11
        mul x11, x4, x8
        adcs x16, x16, x11
        mul x11, x4, x9
        adcs x17, x17, x11
        mul x11, x4, x10
        adcs x19, x19, x11
        cset x20, cs
        umulh x11, x4, x5
        adds x14, x14, x11
        umulh x11, x4, x6
        adcs x15, x15, x11
        umulh x11, x4, x7
        adcs x16, x16, x11
        umulh x11, x4, x8
        adcs x17, x17, x11
        umulh x11, x4, x9
        adcs x19, x19, x11
        umulh x11, x4, x10
        adc x20, x20, x11
        ldp x3, x4, [sp, #(48*6)+16]
        mul x11, x3, x5
        adds x14, x14, x11
        mul x11, x3, x6
        adcs x15, x15, x11
        mul x11, x3, x7
        adcs x16, x16, x11
        mul x11, x3, x8
        adcs x17, x17, x11
        mul x11, x3, x9
        adcs x19, x19, x11
        mul x11, x3, x10
        adcs x20, x20, x11
        cset x21, cs
        umulh x11, x3, x5
        adds x15, x15, x11
        umulh x11, x3, x6
        adcs x16, x16, x11
        umulh x11, x3, x7
        adcs x17, x17, x11
        umulh x11, x3, x8
        adcs x19, x19, x11
        umulh x11, x3, x9
        adcs x20, x20, x11
        umulh x11, x3, x10
        adc x21, x21, x11
        mul x11, x4, x5
        adds x15, x15, x11
        mul x11, x4, x6
        adcs x16, x16, x11
        mul x11, x4, x7
        adcs x17, x17, x11
        mul x11, x4, x8
        adcs x19, x19, x11
        mul x11, x4, x9
        adcs x20, x20, x11
        mul x11, x4, x10
        adcs x21, x21, x11
        cset x22, cs
        umulh x11, x4, x5
        adds x16, x16, x11
        umulh x11, x4, x6
        adcs x17, x17, x11
        umulh x11, x4, x7
        adcs x19, x19, x11
        umulh x11, x4, x8
        adcs x20, x20, x11
        umulh x11, x4, x9
        adcs x21, x21, x11
        umulh x11, x4, x10
        adc x22, x22, x11
        ldp x3, x4, [sp, #(48*6)+32]
        mul x11, x3, x5
        adds x16, x16, x11
        mul x11, x3, x6
        adcs x17, x17, x11
        mul x11, x3, x7
        adcs x19, x19, x11
        mul x11, x3, x8
        adcs x20, x20, x11
        mul x11, x3, x9
        adcs x21, x21, x11
        mul x11, x3, x10
        adcs x22, x22, x11
        cset x2, cs
        umulh x11, x3, x5
        adds x17, x17, x11
        umulh x11, x3, x6
        adcs x19, x19, x11
        umulh x11, x3, x7
        adcs x20, x20, x11
        umulh x11, x3, x8
        adcs x21, x21, x11
        umulh x11, x3, x9
        adcs x22, x22, x11
        umulh x11, x3, x10
        adc x2, x2, x11
        mul x11, x4, x5
        adds x17, x17, x11
        mul x11, x4, x6
        adcs x19, x19, x11
        mul x11, x4, x7
        adcs x20, x20, x11
        mul x11, x4, x8
        adcs x21, x21, x11
        mul x11, x4, x9
        adcs x22, x22, x11
        mul x11, x4, x10
        adcs x2, x2, x11
        cset x1, cs
        umulh x11, x4, x5
        adds x19, x19, x11
        umulh x11, x4, x6
        adcs x20, x20, x11
        umulh x11, x4, x7
        adcs x21, x21, x11
        umulh x11, x4, x8
        adcs x22, x22, x11
        umulh x11, x4, x9
        adcs x2, x2, x11
        umulh x11, x4, x10
        adc x1, x1, x11
        lsl x7, x12, #32
        add x12, x7, x12
        mov x7, #0xffffffff00000001
        umulh x7, x7, x12
        mov x6, #0xffffffff
        mul x5, x6, x12
        umulh x6, x6, x12
        adds x7, x7, x5
        adcs x6, x6, x12
        adc x5, xzr, xzr
        subs x13, x13, x7
        sbcs x14, x14, x6
        sbcs x15, x15, x5
        sbcs x16, x16, xzr
        sbcs x17, x17, xzr
        sbc x12, x12, xzr
        lsl x7, x13, #32
        add x13, x7, x13
        mov x7, #0xffffffff00000001
        umulh x7, x7, x13
        mov x6, #0xffffffff
        mul x5, x6, x13
        umulh x6, x6, x13
        adds x7, x7, x5
        adcs x6, x6, x13
        adc x5, xzr, xzr
        subs x14, x14, x7
        sbcs x15, x15, x6
        sbcs x16, x16, x5
        sbcs x17, x17, xzr
        sbcs x12, x12, xzr
        sbc x13, x13, xzr
        lsl x7, x14, #32
        add x14, x7, x14
        mov x7, #0xffffffff00000001
        umulh x7, x7, x14
        mov x6, #0xffffffff
        mul x5, x6, x14
        umulh x6, x6, x14
        adds x7, x7, x5
        adcs x6, x6, x14
        adc x5, xzr, xzr
        subs x15, x15, x7
        sbcs x16, x16, x6
        sbcs x17, x17, x5
        sbcs x12, x12, xzr
        sbcs x13, x13, xzr
        sbc x14, x14, xzr
        lsl x7, x15, #32
        add x15, x7, x15
        mov x7, #0xffffffff00000001
        umulh x7, x7, x15
        mov x6, #0xffffffff
        mul x5, x6, x15
        umulh x6, x6, x15
        adds x7, x7, x5
        adcs x6, x6, x15
        adc x5, xzr, xzr
        subs x16, x16, x7
        sbcs x17, x17, x6
        sbcs x12, x12, x5
        sbcs x13, x13, xzr
        sbcs x14, x14, xzr
        sbc x15, x15, xzr
        lsl x7, x16, #32
        add x16, x7, x16
        mov x7, #0xffffffff00000001
        umulh x7, x7, x16
        mov x6, #0xffffffff
        mul x5, x6, x16
        umulh x6, x6, x16
        adds x7, x7, x5
        adcs x6, x6, x16
        adc x5, xzr, xzr
        subs x17, x17, x7
        sbcs x12, x12, x6
        sbcs x13, x13, x5
        sbcs x14, x14, xzr
        sbcs x15, x15, xzr
        sbc x16, x16, xzr
        lsl x7, x17, #32
        add x17, x7, x17
        mov x7, #0xffffffff00000001
        umulh x7, x7, x17
        mov x6, #0xffffffff
        mul x5, x6, x17
        umulh x6, x6, x17
        adds x7, x7, x5
        adcs x6, x6, x17
        adc x5, xzr, xzr
        subs x12, x12, x7
        sbcs x13, x13, x6
        sbcs x14, x14, x5
        sbcs x15, x15, xzr
        sbcs x16, x16, xzr
        sbc x17, x17, xzr
        adds x12, x12, x19
        adcs x13, x13, x20
        adcs x14, x14, x21
        adcs x15, x15, x22
        adcs x16, x16, x2
        adcs x17, x17, x1
        adc x10, xzr, xzr
        mov x11, #0xffffffff00000001
        adds x19, x12, x11
        mov x11, #0xffffffff
        adcs x20, x13, x11
        mov x11, #0x1
        adcs x21, x14, x11
        adcs x22, x15, xzr
        adcs x2, x16, xzr
        adcs x1, x17, xzr
        adcs x10, x10, xzr
        csel x12, x12, x19, eq
        csel x13, x13, x20, eq
        csel x14, x14, x21, eq
        csel x15, x15, x22, eq
        csel x16, x16, x2, eq
        csel x17, x17, x1, eq
        stp x12, x13, [sp, #(48*5)]
        stp x14, x15, [sp, #(48*5)+16]
        stp x16, x17, [sp, #(48*5)+32]



        ldp x1, x2, [sp, #(48*3)]
        ldp x3, x4, [sp, #(48*3)+16]
        ldp x5, x6, [sp, #(48*3)+32]
        lsl x0, x1, #2
        ldp x7, x8, [sp, #(48*6)]
        subs x0, x0, x7
        extr x1, x2, x1, #62
        sbcs x1, x1, x8
        ldp x7, x8, [sp, #(48*6)+16]
        extr x2, x3, x2, #62
        sbcs x2, x2, x7
        extr x3, x4, x3, #62
        sbcs x3, x3, x8
        extr x4, x5, x4, #62
        ldp x7, x8, [sp, #(48*6)+32]
        sbcs x4, x4, x7
        extr x5, x6, x5, #62
        sbcs x5, x5, x8
        lsr x6, x6, #62
        adc x6, x6, xzr
        lsl x7, x6, #32
        subs x8, x6, x7
        sbc x7, x7, xzr
        adds x0, x0, x8
        adcs x1, x1, x7
        adcs x2, x2, x6
        adcs x3, x3, xzr
        adcs x4, x4, xzr
        adcs x5, x5, xzr
        csetm x8, cc
        mov x9, #0xffffffff
        and x9, x9, x8
        adds x0, x0, x9
        eor x9, x9, x8
        adcs x1, x1, x9
        mov x9, #0xfffffffffffffffe
        and x9, x9, x8
        adcs x2, x2, x9
        adcs x3, x3, x8
        adcs x4, x4, x8
        adc x5, x5, x8
        stp x0, x1, [x23, #0]
        stp x2, x3, [x23, #0 +16]
        stp x4, x5, [x23, #0 +32]



        ldp x0, x1, [sp, #(48*4)]
        mov x6, #0x00000000ffffffff
        subs x6, x6, x0
        mov x7, #0xffffffff00000000
        sbcs x7, x7, x1
        ldp x0, x1, [sp, #(48*4)+16]
        mov x8, #0xfffffffffffffffe
        sbcs x8, x8, x0
        mov x13, #0xffffffffffffffff
        sbcs x9, x13, x1
        ldp x0, x1, [sp, #(48*4)+32]
        sbcs x10, x13, x0
        sbc x11, x13, x1
        lsl x0, x6, #3
        extr x1, x7, x6, #61
        extr x2, x8, x7, #61
        extr x3, x9, x8, #61
        extr x4, x10, x9, #61
        extr x5, x11, x10, #61
        lsr x6, x11, #61
        add x6, x6, #1
        ldp x8, x9, [sp, #(48*5)]
        ldp x10, x11, [sp, #(48*5)+16]
        ldp x12, x13, [sp, #(48*5)+32]
        mov x14, 3
        mul x15, x14, x8
        umulh x8, x14, x8
        adds x0, x0, x15
        mul x15, x14, x9
        umulh x9, x14, x9
        adcs x1, x1, x15
        mul x15, x14, x10
        umulh x10, x14, x10
        adcs x2, x2, x15
        mul x15, x14, x11
        umulh x11, x14, x11
        adcs x3, x3, x15
        mul x15, x14, x12
        umulh x12, x14, x12
        adcs x4, x4, x15
        mul x15, x14, x13
        umulh x13, x14, x13
        adcs x5, x5, x15
        adc x6, x6, xzr
        adds x1, x1, x8
        adcs x2, x2, x9
        adcs x3, x3, x10
        adcs x4, x4, x11
        adcs x5, x5, x12
        adcs x6, x6, x13
        lsl x7, x6, #32
        subs x8, x6, x7
        sbc x7, x7, xzr
        adds x0, x0, x8
        adcs x1, x1, x7
        adcs x2, x2, x6
        adcs x3, x3, xzr
        adcs x4, x4, xzr
        adcs x5, x5, xzr
        csetm x6, cc
        mov x7, #0xffffffff
        and x7, x7, x6
        adds x0, x0, x7
        eor x7, x7, x6
        adcs x1, x1, x7
        mov x7, #0xfffffffffffffffe
        and x7, x7, x6
        adcs x2, x2, x7
        adcs x3, x3, x6
        adcs x4, x4, x6
        adc x5, x5, x6
        stp x0, x1, [x23, #48]
        stp x2, x3, [x23, #48 +16]
        stp x4, x5, [x23, #48 +32]



        add sp, sp, #(48*7 +0)
        .cfi_adjust_cfa_offset -48*7

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
