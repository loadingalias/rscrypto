// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_montsqr_p256_alt.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montsqr_p256_alt
        .type rscrypto_bignum_montsqr_p256_alt, %function
        .hidden rscrypto_bignum_montsqr_p256_alt
        .text
rscrypto_bignum_montsqr_p256_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq (%rsi), %rax
        movq %rax, %rbx
        mulq %rax
        movq %rax, %r8
        movq %rdx, %r15
        movq 8(%rsi), %rax
        mulq %rbx
        movq %rax, %r9
        movq %rdx, %r10
        movq 24(%rsi), %rax
        movq %rax, %r13
        mulq %rbx
        movq %rax, %r11
        movq %rdx, %r12
        movq 16(%rsi), %rax
        movq %rax, %rbx
        mulq %r13
        movq %rax, %r13
        movq %rdx, %r14
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx
        movq 8(%rsi), %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rcx, %rcx
        movq 24(%rsi), %rbx
        movq 8(%rsi), %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        adcq $0, %r14
        xorl %ecx, %ecx
        addq %r9, %r9
        adcq %r10, %r10
        adcq %r11, %r11
        adcq %r12, %r12
        adcq %r13, %r13
        adcq %r14, %r14
        adcq %rcx, %rcx
        movq 8(%rsi), %rax
        mulq %rax
        addq %r15, %r9
        adcq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        movq 16(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r12
        adcq %rdx, %r13
        sbbq %r15, %r15
        movq 24(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r14
        adcq %rcx, %rdx
        movq %rdx, %r15
        movq $0x0000000100000000, %rbx
        movq %r8, %rax ; mulq %rbx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx
        movq %r9, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx
        notq %rbx
        leaq 2(%rbx), %rbx
        movq %r8, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rcx, %rcx
        xorl %r8d, %r8d
        movq %r9, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        adcq %r8, %r14
        adcq %r8, %r15
        adcq %r8, %r8
        movq $0x0000000100000000, %rbx
        movq %r10, %rax ; mulq %rbx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rcx, %rcx
        movq %r11, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %rcx, %rcx
        notq %rbx
        leaq 2(%rbx), %rbx
        movq %r10, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %rcx, %rcx
        xorl %r9d, %r9d
        movq %r11, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15
        adcq %r9, %r8
        movl $1, %ecx
        addq %r12, %rcx
        leaq -1(%rbx), %rbx
        adcq %r13, %rbx
        leaq -1(%r9), %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rcx, %r12
        cmovcq %rbx, %r13
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
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size rscrypto_bignum_montsqr_p256_alt, .-rscrypto_bignum_montsqr_p256_alt
