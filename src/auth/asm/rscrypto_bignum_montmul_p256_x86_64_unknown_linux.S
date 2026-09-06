// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_montmul_p256.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montmul_p256
        .type rscrypto_bignum_montmul_p256, %function
        .hidden rscrypto_bignum_montmul_p256
        .text
rscrypto_bignum_montmul_p256:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq %rdx, %rcx
        xorl %r13d, %r13d
        movq (%rcx), %rdx
        mulxq (%rsi), %r8, %r9
        mulxq 8(%rsi), %rbx, %r10
        adcq %rbx, %r9
        mulxq 16(%rsi), %rbx, %r11
        adcq %rbx, %r10
        mulxq 24(%rsi), %rbx, %r12
        adcq %rbx, %r11
        adcq %r13, %r12
        movq 8(%rcx), %rdx
        xorl %r14d, %r14d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcq %r14, %r13
        xorl %r15d, %r15d
        movq $0x0000000100000000, %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        notq %rdx
        leaq 2(%rdx), %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcxq %r15, %r13
        adoxq %r15, %r14
        adcq %r15, %r14
        movq 16(%rcx), %rdx
        xorl %r8d, %r8d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adoxq %r8, %r14
        mulxq 24(%rsi), %rax, %rbx
        adcq %rax, %r13
        adcq %rbx, %r14
        adcq %r8, %r15
        movq 24(%rcx), %rdx
        xorl %r9d, %r9d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        adoxq %r9, %r15
        mulxq 24(%rsi), %rax, %rbx
        adcq %rax, %r14
        adcq %rbx, %r15
        adcq %r9, %r8
        xorl %r9d, %r9d
        movq $0x0000000100000000, %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        notq %rdx
        leaq 2(%rdx), %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        adcxq %r9, %r15
        adoxq %r9, %r8
        adcq %r9, %r8
        movl $1, %ecx
        addq %r12, %rcx
        decq %rdx
        adcq %r13, %rdx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rcx, %r12
        cmovcq %rdx, %r13
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
.size rscrypto_bignum_montmul_p256, .-rscrypto_bignum_montmul_p256
