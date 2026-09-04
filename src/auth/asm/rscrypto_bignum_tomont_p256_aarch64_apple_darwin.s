// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p256/bignum_tomont_p256.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

        .globl _rscrypto_bignum_tomont_p256

        .private_extern _rscrypto_bignum_tomont_p256
        .globl _rscrypto_bignum_tomont_p256_alt

        .private_extern _rscrypto_bignum_tomont_p256_alt
        .text
        .balign 4
_rscrypto_bignum_tomont_p256:

_rscrypto_bignum_tomont_p256_alt:
        .cfi_startproc



        ldp x2, x3, [x1]
        ldp x4, x5, [x1, #16]





        mov x1, #0xffffffffffffffff
        mov x7, #0x00000000ffffffff
        mov x9, #0xffffffff00000001
        subs x1, x2, x1
        sbcs x7, x3, x7
        sbcs x8, x4, xzr
        sbcs x9, x5, x9
        csel x2, x2, x1, cc
        csel x3, x3, x7, cc
        csel x4, x4, x8, cc
        csel x5, x5, x9, cc



        subs xzr, xzr, xzr %% extr x9, x5, x4, #32 %% adcs xzr, x4, x9 %% lsr x9, x5, #32 %% adcs x9, x5, x9 %% csetm x6, cs %% orr x9, x9, x6 %% lsl x7, x9, #32 %% lsr x8, x9, #32 %% adds x4, x4, x7 %% adc x5, x5, x8 %% subs x6, xzr, x9 %% sbcs x7, x7, xzr %% sbc x8, x8, xzr %% subs x6, xzr, x6 %% sbcs x2, x2, x7 %% sbcs x3, x3, x8 %% sbcs x4, x4, x9 %% sbcs x5, x5, x9 %% adds x6, x6, x5 %% mov x7, #0x00000000ffffffff %% and x7, x7, x5 %% adcs x2, x2, x7 %% adcs x3, x3, xzr %% mov x7, #0xffffffff00000001 %% and x7, x7, x5 %% adc x4, x4, x7
        subs xzr, xzr, xzr %% extr x9, x4, x3, #32 %% adcs xzr, x3, x9 %% lsr x9, x4, #32 %% adcs x9, x4, x9 %% csetm x5, cs %% orr x9, x9, x5 %% lsl x7, x9, #32 %% lsr x8, x9, #32 %% adds x3, x3, x7 %% adc x4, x4, x8 %% subs x5, xzr, x9 %% sbcs x7, x7, xzr %% sbc x8, x8, xzr %% subs x5, xzr, x5 %% sbcs x6, x6, x7 %% sbcs x2, x2, x8 %% sbcs x3, x3, x9 %% sbcs x4, x4, x9 %% adds x5, x5, x4 %% mov x7, #0x00000000ffffffff %% and x7, x7, x4 %% adcs x6, x6, x7 %% adcs x2, x2, xzr %% mov x7, #0xffffffff00000001 %% and x7, x7, x4 %% adc x3, x3, x7
        subs xzr, xzr, xzr %% extr x9, x3, x2, #32 %% adcs xzr, x2, x9 %% lsr x9, x3, #32 %% adcs x9, x3, x9 %% csetm x4, cs %% orr x9, x9, x4 %% lsl x7, x9, #32 %% lsr x8, x9, #32 %% adds x2, x2, x7 %% adc x3, x3, x8 %% subs x4, xzr, x9 %% sbcs x7, x7, xzr %% sbc x8, x8, xzr %% subs x4, xzr, x4 %% sbcs x5, x5, x7 %% sbcs x6, x6, x8 %% sbcs x2, x2, x9 %% sbcs x3, x3, x9 %% adds x4, x4, x3 %% mov x7, #0x00000000ffffffff %% and x7, x7, x3 %% adcs x5, x5, x7 %% adcs x6, x6, xzr %% mov x7, #0xffffffff00000001 %% and x7, x7, x3 %% adc x2, x2, x7
        subs xzr, xzr, xzr %% extr x9, x2, x6, #32 %% adcs xzr, x6, x9 %% lsr x9, x2, #32 %% adcs x9, x2, x9 %% csetm x3, cs %% orr x9, x9, x3 %% lsl x7, x9, #32 %% lsr x8, x9, #32 %% adds x6, x6, x7 %% adc x2, x2, x8 %% subs x3, xzr, x9 %% sbcs x7, x7, xzr %% sbc x8, x8, xzr %% subs x3, xzr, x3 %% sbcs x4, x4, x7 %% sbcs x5, x5, x8 %% sbcs x6, x6, x9 %% sbcs x2, x2, x9 %% adds x3, x3, x2 %% mov x7, #0x00000000ffffffff %% and x7, x7, x2 %% adcs x4, x4, x7 %% adcs x5, x5, xzr %% mov x7, #0xffffffff00000001 %% and x7, x7, x2 %% adc x6, x6, x7



        stp x3, x4, [x0]
        stp x5, x6, [x0, #16]

        ret %% .cfi_endproc
