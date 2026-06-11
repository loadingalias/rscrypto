// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p384/bignum_montmul_p384.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montmul_p384
        .type rscrypto_bignum_montmul_p384, %function
        .hidden rscrypto_bignum_montmul_p384
        .text
rscrypto_bignum_montmul_p384:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq %rdx, %rcx
        movq (%rcx), %rdx
        xorl %r15d, %r15d
        mulxq (%rsi), %r8, %r9
        mulxq 8(%rsi), %rbx, %r10
        addq %rbx, %r9
        mulxq 16(%rsi), %rbx, %r11
        adcq %rbx, %r10
        mulxq 24(%rsi), %rbx, %r12
        adcq %rbx, %r11
        mulxq 32(%rsi), %rbx, %r13
        adcq %rbx, %r12
        mulxq 40(%rsi), %rbx, %r14
        adcq %rbx, %r13
        adcq %r15, %r14
        movq %r8, %rdx ; shlq $32, %rdx ; addq %r8, %rdx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulxq %rax, %rbx, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %r8, %rbx ; adcq %r8, %rax ; adcq %rdx, %rbx ; adcl %ebp, %ebp ; subq %rax, %r9 ; sbbq %rbx, %r10 ; sbbq %rbp, %r11 ; sbbq $0, %r12 ; sbbq $0, %r13 ; sbbq $0, %rdx ; addq %rdx, %r14 ; adcq $0, %r15
        movq 8(%rcx), %rdx
        xorl %r8d, %r8d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        mulxq 32(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        adoxq %r8, %r15
        mulxq 40(%rsi), %rax, %rbx
        adcq %rax, %r14
        adcq %rbx, %r15
        adcq %r8, %r8
        movq %r9, %rdx ; shlq $32, %rdx ; addq %r9, %rdx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulxq %rax, %rbx, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %r9, %rbx ; adcq %r9, %rax ; adcq %rdx, %rbx ; adcl %ebp, %ebp ; subq %rax, %r10 ; sbbq %rbx, %r11 ; sbbq %rbp, %r12 ; sbbq $0, %r13 ; sbbq $0, %r14 ; sbbq $0, %rdx ; addq %rdx, %r15 ; adcq $0, %r8
        movq 16(%rcx), %rdx
        xorl %r9d, %r9d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq 32(%rsi), %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        adoxq %r9, %r8
        mulxq 40(%rsi), %rax, %rbx
        adcq %rax, %r15
        adcq %rbx, %r8
        adcq %r9, %r9
        movq %r10, %rdx ; shlq $32, %rdx ; addq %r10, %rdx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulxq %rax, %rbx, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %r10, %rbx ; adcq %r10, %rax ; adcq %rdx, %rbx ; adcl %ebp, %ebp ; subq %rax, %r11 ; sbbq %rbx, %r12 ; sbbq %rbp, %r13 ; sbbq $0, %r14 ; sbbq $0, %r15 ; sbbq $0, %rdx ; addq %rdx, %r8 ; adcq $0, %r9
        movq 24(%rcx), %rdx
        xorl %r10d, %r10d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        mulxq 32(%rsi), %rax, %rbx ; adcxq %rax, %r15 ; adoxq %rbx, %r8
        adoxq %r10, %r9
        mulxq 40(%rsi), %rax, %rbx
        adcq %rax, %r8
        adcq %rbx, %r9
        adcq %r10, %r10
        movq %r11, %rdx ; shlq $32, %rdx ; addq %r11, %rdx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulxq %rax, %rbx, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %r11, %rbx ; adcq %r11, %rax ; adcq %rdx, %rbx ; adcl %ebp, %ebp ; subq %rax, %r12 ; sbbq %rbx, %r13 ; sbbq %rbp, %r14 ; sbbq $0, %r15 ; sbbq $0, %r8 ; sbbq $0, %rdx ; addq %rdx, %r9 ; adcq $0, %r10
        movq 32(%rcx), %rdx
        xorl %r11d, %r11d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r15 ; adoxq %rbx, %r8
        mulxq 32(%rsi), %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9
        adoxq %r11, %r10
        mulxq 40(%rsi), %rax, %rbx
        adcq %rax, %r9
        adcq %rbx, %r10
        adcq %r11, %r11
        movq %r12, %rdx ; shlq $32, %rdx ; addq %r12, %rdx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulxq %rax, %rbx, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %r12, %rbx ; adcq %r12, %rax ; adcq %rdx, %rbx ; adcl %ebp, %ebp ; subq %rax, %r13 ; sbbq %rbx, %r14 ; sbbq %rbp, %r15 ; sbbq $0, %r8 ; sbbq $0, %r9 ; sbbq $0, %rdx ; addq %rdx, %r10 ; adcq $0, %r11
        movq 40(%rcx), %rdx
        xorl %r12d, %r12d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r15 ; adoxq %rbx, %r8
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9
        mulxq 32(%rsi), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        adoxq %r12, %r11
        mulxq 40(%rsi), %rax, %rbx
        adcq %rax, %r10
        adcq %rbx, %r11
        adcq %r12, %r12
        movq %r13, %rdx ; shlq $32, %rdx ; addq %r13, %rdx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulxq %rax, %rbx, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %r13, %rbx ; adcq %r13, %rax ; adcq %rdx, %rbx ; adcl %ebp, %ebp ; subq %rax, %r14 ; sbbq %rbx, %r15 ; sbbq %rbp, %r8 ; sbbq $0, %r9 ; sbbq $0, %r10 ; sbbq $0, %rdx ; addq %rdx, %r11 ; adcq $0, %r12
        xorl %edx, %edx
        xorl %ebp, %ebp
        xorl %r13d, %r13d
        movq $0xffffffff00000001, %rax
        addq %r14, %rax
        movl $0x00000000ffffffff, %ebx
        adcq %r15, %rbx
        movl $0x0000000000000001, %ecx
        adcq %r8, %rcx
        adcq %r9, %rdx
        adcq %r10, %rbp
        adcq %r11, %r13
        adcq $0, %r12
        cmovnzq %rax, %r14
        cmovnzq %rbx, %r15
        cmovnzq %rcx, %r8
        cmovnzq %rdx, %r9
        cmovnzq %rbp, %r10
        cmovnzq %r13, %r11
        movq %r14, (%rdi)
        movq %r15, 8(%rdi)
        movq %r8, 16(%rdi)
        movq %r9, 24(%rdi)
        movq %r10, 32(%rdi)
        movq %r11, 40(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size rscrypto_bignum_montmul_p384, .-rscrypto_bignum_montmul_p384
