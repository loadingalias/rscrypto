// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p384/bignum_montsqr_p384.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montsqr_p384
        .type rscrypto_bignum_montsqr_p384, %function
        .hidden rscrypto_bignum_montsqr_p384
        .text
rscrypto_bignum_montsqr_p384:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq (%rsi), %rdx
        mulxq 8(%rsi), %r9, %r10
        mulxq 24(%rsi), %r11, %r12
        mulxq 40(%rsi), %r13, %r14
        movq 24(%rsi), %rdx
        mulxq 32(%rsi), %r15, %rcx
        xorl %ebp, %ebp
        movq 16(%rsi), %rdx
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        movq 8(%rsi), %rdx
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        mulxq 32(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq 40(%rsi), %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        adcxq %rbp, %r15
        adoxq %rbp, %rcx
        adcq %rbp, %rcx
        xorl %ebp, %ebp
        movq 32(%rsi), %rdx
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        movq 16(%rsi), %rdx
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq 32(%rsi), %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        mulxq 40(%rsi), %rax, %rdx
        adcxq %rax, %r15
        adoxq %rdx, %rcx
        movq 40(%rsi), %rdx
        mulxq 32(%rsi), %rbx, %rbp
        mulxq 24(%rsi), %rax, %rdx
        adcxq %rax, %rcx
        adoxq %rdx, %rbx
        movl $0, %eax
        adcxq %rax, %rbx
        adoxq %rax, %rbp
        adcq %rax, %rbp
        xorq %rax, %rax
        movq (%rsi), %rdx
        mulxq (%rsi), %r8, %rax
        adcxq %r9, %r9
        adoxq %rax, %r9
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
        mulxq %rdx, %rax, %rdx
        adcxq %r14, %r14
        adoxq %rax, %r14
        adcxq %r15, %r15
        adoxq %rdx, %r15
        movq 32(%rsi), %rdx
        mulxq %rdx, %rax, %rdx
        adcxq %rcx, %rcx
        adoxq %rax, %rcx
        adcxq %rbx, %rbx
        adoxq %rdx, %rbx
        movq 40(%rsi), %rdx
        mulxq %rdx, %rax, %rsi
        adcxq %rbp, %rbp
        adoxq %rax, %rbp
        movl $0, %eax
        adcxq %rax, %rsi
        adoxq %rax, %rsi
        movq %rbx, (%rdi)
        movq %r8, %rdx ; shlq $32, %rdx ; addq %r8, %rdx ; movq $0xffffffff00000001, %rax ; mulxq %rax, %r8, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %rbx, %r8 ; addq %rbx, %rax ; adcq %rdx, %r8 ; movl $0, %ebx ; adcq %rbx, %rbx ; subq %rax, %r9 ; sbbq %r8, %r10 ; sbbq %rbx, %r11 ; sbbq $0, %r12 ; sbbq $0, %r13 ; movq %rdx, %r8 ; sbbq $0, %r8
        movq %r9, %rdx ; shlq $32, %rdx ; addq %r9, %rdx ; movq $0xffffffff00000001, %rax ; mulxq %rax, %r9, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %rbx, %r9 ; addq %rbx, %rax ; adcq %rdx, %r9 ; movl $0, %ebx ; adcq %rbx, %rbx ; subq %rax, %r10 ; sbbq %r9, %r11 ; sbbq %rbx, %r12 ; sbbq $0, %r13 ; sbbq $0, %r8 ; movq %rdx, %r9 ; sbbq $0, %r9
        movq %r10, %rdx ; shlq $32, %rdx ; addq %r10, %rdx ; movq $0xffffffff00000001, %rax ; mulxq %rax, %r10, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %rbx, %r10 ; addq %rbx, %rax ; adcq %rdx, %r10 ; movl $0, %ebx ; adcq %rbx, %rbx ; subq %rax, %r11 ; sbbq %r10, %r12 ; sbbq %rbx, %r13 ; sbbq $0, %r8 ; sbbq $0, %r9 ; movq %rdx, %r10 ; sbbq $0, %r10
        movq %r11, %rdx ; shlq $32, %rdx ; addq %r11, %rdx ; movq $0xffffffff00000001, %rax ; mulxq %rax, %r11, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %rbx, %r11 ; addq %rbx, %rax ; adcq %rdx, %r11 ; movl $0, %ebx ; adcq %rbx, %rbx ; subq %rax, %r12 ; sbbq %r11, %r13 ; sbbq %rbx, %r8 ; sbbq $0, %r9 ; sbbq $0, %r10 ; movq %rdx, %r11 ; sbbq $0, %r11
        movq %r12, %rdx ; shlq $32, %rdx ; addq %r12, %rdx ; movq $0xffffffff00000001, %rax ; mulxq %rax, %r12, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %rbx, %r12 ; addq %rbx, %rax ; adcq %rdx, %r12 ; movl $0, %ebx ; adcq %rbx, %rbx ; subq %rax, %r13 ; sbbq %r12, %r8 ; sbbq %rbx, %r9 ; sbbq $0, %r10 ; sbbq $0, %r11 ; movq %rdx, %r12 ; sbbq $0, %r12
        movq %r13, %rdx ; shlq $32, %rdx ; addq %r13, %rdx ; movq $0xffffffff00000001, %rax ; mulxq %rax, %r13, %rax ; movl $0x00000000ffffffff, %ebx ; mulxq %rbx, %rbx, %r13 ; addq %rbx, %rax ; adcq %rdx, %r13 ; movl $0, %ebx ; adcq %rbx, %rbx ; subq %rax, %r8 ; sbbq %r13, %r9 ; sbbq %rbx, %r10 ; sbbq $0, %r11 ; sbbq $0, %r12 ; movq %rdx, %r13 ; sbbq $0, %r13
        movq (%rdi), %rbx
        addq %r8, %r14
        adcq %r9, %r15
        adcq %r10, %rcx
        adcq %r11, %rbx
        adcq %r12, %rbp
        adcq %r13, %rsi
        movl $0, %r8d
        adcq %r8, %r8
        xorq %r11, %r11
        xorq %r12, %r12
        xorq %r13, %r13
        movq $0xffffffff00000001, %rax
        addq %r14, %rax
        movl $0x00000000ffffffff, %r9d
        adcq %r15, %r9
        movl $0x0000000000000001, %r10d
        adcq %rcx, %r10
        adcq %rbx, %r11
        adcq %rbp, %r12
        adcq %rsi, %r13
        adcq $0, %r8
        cmovnzq %rax, %r14
        cmovnzq %r9, %r15
        cmovnzq %r10, %rcx
        cmovnzq %r11, %rbx
        cmovnzq %r12, %rbp
        cmovnzq %r13, %rsi
        movq %r14, (%rdi)
        movq %r15, 8(%rdi)
        movq %rcx, 16(%rdi)
        movq %rbx, 24(%rdi)
        movq %rbp, 32(%rdi)
        movq %rsi, 40(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size rscrypto_bignum_montsqr_p384, .-rscrypto_bignum_montsqr_p384
