// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_montsqr_p256.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montsqr_p256
        .type rscrypto_bignum_montsqr_p256, %function
        .hidden rscrypto_bignum_montsqr_p256
        .text
rscrypto_bignum_montsqr_p256:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq (%rsi), %rdx
        mulxq %rdx, %r8, %r15
        mulxq 8(%rsi), %r9, %r10
        mulxq 24(%rsi), %r11, %r12
        movq 16(%rsi), %rdx
        mulxq 24(%rsi), %r13, %r14
        xorl %ebp, %ebp
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        movq 24(%rsi), %rdx
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcxq %rbp, %r13
        adoxq %rbp, %r14
        adcq %rbp, %r14
        xorl %ebp, %ebp
        adcxq %r9, %r9
        adoxq %r15, %r9
        movq 8(%rsi), %rdx
        mulxq %rdx, %rax, %rdx
        adcxq %r10, %r10
        adoxq %rax, %r10
        adcxq %r11, %r11
        adoxq %rdx, %r11
        movq 16(%rsi), %rdx
        mulxq %rdx, %rax, %rdx
        adcxq %r12, %r12
        adoxq %rax, %r12
        adcxq %r13, %r13
        adoxq %rdx, %r13
        movq 24(%rsi), %rdx
        mulxq %rdx, %rax, %r15
        adcxq %r14, %r14
        adoxq %rax, %r14
        adcxq %rbp, %r15
        adoxq %rbp, %r15
        xorl %ebp, %ebp
        movq $0x0000000100000000, %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        movq $0xffffffff00000001, %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcxq %rbp, %r13
        movl %ebp, %r9d
        adoxq %rbp, %r9
        adcxq %rbp, %r9
        addq %r9, %r14
        adcq %rbp, %r15
        movl %ebp, %r8d
        adcq %rbp, %r8
        xorl %ebp, %ebp
        movq $0x0000000100000000, %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        movq $0xffffffff00000001, %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        adcxq %rbp, %r15
        adoxq %rbp, %r8
        adcq %rbp, %r8
        movl $1, %ecx
        addq %r12, %rcx
        leaq -1(%rdx), %rdx
        adcq %r13, %rdx
        leaq -1(%rbp), %rbp
        movq %rbp, %rax
        adcq %r14, %rbp
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rcx, %r12
        cmovcq %rdx, %r13
        cmovcq %rbp, %r14
        cmovcq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 8(%rdi)
        movq %r14, 16(%rdi)
        movq %r15, 24(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size rscrypto_bignum_montsqr_p256, .-rscrypto_bignum_montsqr_p256
