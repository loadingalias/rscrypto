// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p384/bignum_mod_n384.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.


        .globl rscrypto_bignum_mod_n384

        .hidden rscrypto_bignum_mod_n384
        .globl rscrypto_bignum_mod_n384_alt

        .hidden rscrypto_bignum_mod_n384_alt
        .text
        .balign 4
rscrypto_bignum_mod_n384:

rscrypto_bignum_mod_n384_alt:
        .cfi_startproc



        cmp x1, #6
        bcc Lbignum_mod_n384_short



        sub x1, x1, #6
        lsl x9, x1, #3
        add x9, x9, x2
        ldp x7, x8, [x9, #32]
        ldp x5, x6, [x9, #16]
        ldp x3, x4, [x9]



        movz x15, #0xd68d
        movk x15, #0x333a, lsl #16
        movk x15, #0xe695, lsl #32
        movk x15, #0x1313, lsl #48
        movz x16, #0x5885
        movk x16, #0xb74f, lsl #16
        movk x16, #0xf24d, lsl #32
        movk x16, #0xa7e5, lsl #48
        movz x17, #0xd220
        movk x17, #0x0bc8, lsl #16
        movk x17, #0xb27e, lsl #32
        movk x17, #0x389c, lsl #48



        adds x9, x3, x15
        adcs x10, x4, x16
        adcs x11, x5, x17
        adcs x12, x6, xzr
        adcs x13, x7, xzr
        adcs x14, x8, xzr
        csel x3, x3, x9, cc
        csel x4, x4, x10, cc
        csel x5, x5, x11, cc
        csel x6, x6, x12, cc
        csel x7, x7, x13, cc
        csel x8, x8, x14, cc



        cbz x1, Lbignum_mod_n384_writeback
Lbignum_mod_n384_loop:



        adds x13, x8, #1
        csetm x9, cs
        orr x13, x13, x9



        mul x9, x15, x13
        mul x10, x16, x13
        mul x11, x17, x13

        umulh x12, x15, x13
        adds x10, x10, x12
        umulh x12, x16, x13
        adcs x11, x11, x12
        umulh x12, x17, x13
        adc x12, xzr, x12



        sub x1, x1, #1
        ldr x14, [x2, x1, lsl #3]



        sub x8, x8, x13



        adds x9, x14, x9
        adcs x10, x3, x10
        adcs x11, x4, x11
        adcs x12, x5, x12
        adcs x13, x6, xzr
        adcs x7, x7, xzr
        adc x8, x8, xzr





        and x14, x8, x15
        subs x3, x9, x14
        and x14, x8, x16
        sbcs x4, x10, x14
        and x14, x8, x17
        sbcs x5, x11, x14
        sbcs x6, x12, xzr
        sbcs x14, x13, xzr
        sbc x8, x7, xzr
        mov x7, x14

        cbnz x1, Lbignum_mod_n384_loop



Lbignum_mod_n384_writeback:
        stp x3, x4, [x0]
        stp x5, x6, [x0, #16]
        stp x7, x8, [x0, #32]

        ret
        .cfi_endproc



Lbignum_mod_n384_short:
        mov x3, xzr
        mov x4, xzr
        mov x5, xzr
        mov x6, xzr
        mov x7, xzr
        mov x8, xzr

        cbz x1, Lbignum_mod_n384_writeback
        ldr x3, [x2]
        subs x1, x1, #1
        beq Lbignum_mod_n384_writeback
        ldr x4, [x2, #8]
        subs x1, x1, #1
        beq Lbignum_mod_n384_writeback
        ldr x5, [x2, #16]
        subs x1, x1, #1
        beq Lbignum_mod_n384_writeback
        ldr x6, [x2, #24]
        subs x1, x1, #1
        beq Lbignum_mod_n384_writeback
        ldr x7, [x2, #32]
        b Lbignum_mod_n384_writeback
