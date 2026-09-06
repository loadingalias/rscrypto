// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_tomont_p256_alt.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_tomont_p256_alt
        .type rscrypto_bignum_tomont_p256_alt, %function
        .hidden rscrypto_bignum_tomont_p256_alt
        .text
rscrypto_bignum_tomont_p256_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movl $0x0000000000000003, %ecx
        movq (%rsi), %rax
        mulq %rcx
        movq %rax, %r8
        movq %rdx, %r9
        movq 8(%rsi), %rax
        mulq %rcx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 16(%rsi), %rax
        mulq %rcx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 24(%rsi), %rax
        mulq %rcx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq $0xfffffffbffffffff, %rcx
        xorl %r13d, %r13d
        movq (%rsi), %rax ; mulq %rcx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r14, %r14
        movq 8(%rsi), %rax ; mulq %rcx; subq %r14, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r14, %r14
        movq 16(%rsi), %rax ; mulq %rcx; subq %r14, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r14, %r14
        movq 24(%rsi), %rax ; mulq %rcx; subq %r14, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        xorl %r14d, %r14d
        movq $0x0000000100000000, %rcx
        movq %r8, %rax ; mulq %rcx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r15, %r15
        movq %r9, %rax ; mulq %rcx; subq %r15, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r15, %r15
        notq %rcx
        leaq 2(%rcx), %rcx
        movq %r8, %rax ; mulq %rcx; subq %r15, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r15, %r15
        movq %r9, %rax ; mulq %rcx; subq %r15, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        adcq %r14, %r14
        movq $0xfffffffffffffffe, %rcx
        xorl %r15d, %r15d
        movq (%rsi), %rax ; mulq %rcx; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r8, %r8
        movq 8(%rsi), %rax ; mulq %rcx; subq %r8, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r8, %r8
        movq 16(%rsi), %rax ; mulq %rcx; subq %r8, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r8, %r8
        movq 24(%rsi), %rax ; mulq %rcx; subq %r8, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14
        adcq %r15, %r15
        movq $0x00000004fffffffd, %rcx
        xorl %r8d, %r8d
        movq (%rsi), %rax ; mulq %rcx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r9, %r9
        movq 8(%rsi), %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r9, %r9
        movq 16(%rsi), %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r9, %r9
        movq 24(%rsi), %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15
        adcq %r8, %r8
        movq $0x0000000100000000, %rcx
        movq %r10, %rax ; mulq %rcx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r9, %r9
        movq %r11, %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r9, %r9
        notq %rcx
        leaq 2(%rcx), %rcx
        movq %r10, %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r9, %r9
        movq %r11, %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %r9, %r9
        subq %r9, %r8
        xorl %edx, %edx
        leaq -1(%rdx), %r9
        incq %rdx
        addq %r12, %rdx
        decq %rcx
        adcq %r13, %rcx
        movq %r9, %rax
        adcq %r14, %r9
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rdx, %r12
        cmovcq %rcx, %r13
        cmovcq %r9, %r14
        cmovcq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 8(%rdi)
        movq %r14, 16(%rdi)
        movq %r15, 24(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        retq ; .cfi_endproc
.size rscrypto_bignum_tomont_p256_alt, .-rscrypto_bignum_tomont_p256_alt
