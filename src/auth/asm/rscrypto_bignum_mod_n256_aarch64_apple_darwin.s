// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p256/bignum_mod_n256.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.


        .globl _rscrypto_bignum_mod_n256

        .private_extern _rscrypto_bignum_mod_n256
        .globl _rscrypto_bignum_mod_n256_alt

        .private_extern _rscrypto_bignum_mod_n256_alt
        .text
        .balign 4
_rscrypto_bignum_mod_n256:

_rscrypto_bignum_mod_n256_alt:
        .cfi_startproc



        cmp x1, #4
        bcc Lbignum_mod_n256_short



        sub x1, x1, #4
        lsl x7, x1, #3
        add x7, x7, x2
        ldp x5, x6, [x7, #16]
        ldp x3, x4, [x7]



        movz x12, #0xdaaf %% movk x12, #0x039c, lsl #16 %% movk x12, #0x353d, lsl #32 %% movk x12, #0x0c46, lsl #48
        movz x13, #0x617b %% movk x13, #0x58e8, lsl #16 %% movk x13, #0x0552, lsl #32 %% movk x13, #0x4319, lsl #48
        mov x14, #0x00000000ffffffff



        adds x7, x3, x12
        adcs x8, x4, x13
        adcs x9, x5, xzr
        adcs x10, x6, x14
        csel x3, x3, x7, cc
        csel x4, x4, x8, cc
        csel x5, x5, x9, cc
        csel x6, x6, x10, cc



        cbz x1, Lbignum_mod_n256_writeback
Lbignum_mod_n256_loop:




        subs xzr, xzr, xzr
        extr x15, x6, x5, #32
        adcs xzr, x5, x15
        lsr x15, x6, #32
        adcs x15, x6, x15
        csetm x7, cs
        orr x15, x15, x7



        mul x7, x12, x15
        mul x8, x13, x15
        mul x10, x14, x15
        umulh x9, x12, x15
        adds x8, x8, x9
        umulh x9, x13, x15
        adc x9, x9, xzr
        umulh x11, x14, x15



        sub x6, x6, x15



        sub x1, x1, #1
        ldr x15, [x2, x1, lsl #3]



        adds x7, x15, x7
        adcs x8, x3, x8
        adcs x9, x4, x9
        adcs x10, x5, x10
        adc x11, x6, x11





        and x15, x11, x12
        subs x3, x7, x15
        and x15, x11, x13
        sbcs x4, x8, x15
        sbcs x5, x9, xzr
        and x15, x11, x14
        sbc x6, x10, x15

        cbnz x1, Lbignum_mod_n256_loop



Lbignum_mod_n256_writeback:
        stp x3, x4, [x0]
        stp x5, x6, [x0, #16]
        ret %% .cfi_endproc



Lbignum_mod_n256_short:
        mov x3, xzr
        mov x4, xzr
        mov x5, xzr
        mov x6, xzr

        cbz x1, Lbignum_mod_n256_writeback
        ldr x3, [x2]
        subs x1, x1, #1
        beq Lbignum_mod_n256_writeback
        ldr x4, [x2, #8]
        subs x1, x1, #1
        beq Lbignum_mod_n256_writeback
        ldr x5, [x2, #16]
        b Lbignum_mod_n256_writeback
