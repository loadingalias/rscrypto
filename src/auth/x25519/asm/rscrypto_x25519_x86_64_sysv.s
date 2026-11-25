// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from the s2n-bignum x86_64 Curve25519 X25519 byte backends.
// The symbols are renamed to the rscrypto namespace and embedded with Rust global_asm!.

 .globl rscrypto_curve25519_x25519
        .if {elf}
        .type rscrypto_curve25519_x25519, %function
        .hidden rscrypto_curve25519_x25519
        .endif
        .globl rscrypto_curve25519_x25519_byte
        .if {elf}
        .type rscrypto_curve25519_x25519_byte, %function
        .hidden rscrypto_curve25519_x25519_byte
        .endif
        .text
rscrypto_curve25519_x25519:
rscrypto_curve25519_x25519_byte:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $13*32, %rsp ; .cfi_adjust_cfa_offset 13*32
        movq %rdi, 12*32(%rsp)
        movq (%rsi), %rax
        movq %rax, (%rsp)
        movq 8(%rsi), %rax
        movq %rax, 8(%rsp)
        movq 16(%rsi), %rax
        movq %rax, 16(%rsp)
        movq 24(%rsi), %rax
        movq %rax, 24(%rsp)
        movq (%rdx), %r8
        movq 8(%rdx), %r9
        movq 16(%rdx), %r10
        movq 24(%rdx), %r11
        btr $63, %r11
        movq %r8, 32(%rsp)
        movq %r9, 40(%rsp)
        movq %r10, 48(%rsp)
        movq %r11, 56(%rsp)
        movl $1, %eax
        movq %rax, 12*32 +16(%rsp)
        movq %r8, 256(%rsp)
        movq %rax, 96(%rsp)
        xorl %eax, %eax
        movq %r9, 264(%rsp)
        movq %rax, 104(%rsp)
        movq %r10, 272(%rsp)
        movq %rax, 112(%rsp)
        movq %r11, 280(%rsp)
        movq %rax, 120(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (3*32)(%rsp), %r8 ; movq 8+(8*32)(%rsp), %r9 ; sbbq 8+(3*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(8*32)(%rsp), %r10 ; sbbq 16+(3*32)(%rsp), %r10 ; movq 24+(8*32)(%rsp), %rax ; sbbq 24+(3*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (3*32)(%rsp), %r8 ; movq 0x8+(8*32)(%rsp), %r9 ; adcq 0x8+(3*32)(%rsp), %r9 ; movq 0x10+(8*32)(%rsp), %r10 ; adcq 0x10+(3*32)(%rsp), %r10 ; movq 0x18+(8*32)(%rsp), %r11 ; adcq 0x18+(3*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(11*32)(%rsp), %r9, %r10 ; mulxq 0x18+(11*32)(%rsp), %r11, %r12 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq 0x18+(11*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(10*32)(%rsp), %r9, %r10 ; mulxq 0x18+(10*32)(%rsp), %r11, %r12 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq 0x18+(10*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq (11*32)(%rsp), %r8 ; movq 8+(11*32)(%rsp), %r9 ; movq 16+(11*32)(%rsp), %r10 ; movq 24+(11*32)(%rsp), %r11 ; xorl %edi, %edi ; movq $0x1db42, %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq 8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 16+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 24+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rdi, %rbx ; adcxq %rdi, %rbx ; shldq $0x1, %r11, %rbx ; btr $63, %r11 ; movl $0x13, %edx ; imulq %rdx, %rbx ; addq %rbx, %r8 ; adcq %rdi, %r9 ; adcq %rdi, %r10 ; adcq %rdi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        xorl %ecx, %ecx ; movq (11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %r8, %r9 ; mulxq 0x8+(10*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(10*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(10*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(10*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(10*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(10*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        xorl %ecx, %ecx ; movq (5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %r8, %r9 ; mulxq 0x8+(6*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(6*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(6*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(6*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(6*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(6*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        movl $253, %eax
        movq %rax, 12*32 +8(%rsp)
Lcurve25519_x25519_scalarloop:
        movq (8*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (3*32)(%rsp), %r8 ; movq 8+(8*32)(%rsp), %r9 ; sbbq 8+(3*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(8*32)(%rsp), %r10 ; sbbq 16+(3*32)(%rsp), %r10 ; movq 24+(8*32)(%rsp), %rax ; sbbq 24+(3*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (2*32)(%rsp) ; movq %r9, 8+(2*32)(%rsp) ; movq %r10, 16+(2*32)(%rsp) ; movq %rax, 24+(2*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (4*32)(%rsp) ; movq %r9, 0x8+(4*32)(%rsp) ; movq %r10, 0x10+(4*32)(%rsp) ; movq %r11, 0x18+(4*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (5*32)(%rsp) ; movq %r9, 8+(5*32)(%rsp) ; movq %r10, 16+(5*32)(%rsp) ; movq %rax, 24+(5*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (3*32)(%rsp), %r8 ; movq 0x8+(8*32)(%rsp), %r9 ; adcq 0x8+(3*32)(%rsp), %r9 ; movq 0x10+(8*32)(%rsp), %r10 ; adcq 0x10+(3*32)(%rsp), %r10 ; movq 0x18+(8*32)(%rsp), %r11 ; adcq 0x18+(3*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq 12*32 +8(%rsp), %rdx
        movq %rdx, %rcx
        shrq $6, %rdx
        movq (%rsp,%rdx,8), %rdx
        shrq %cl, %rdx
        andq $1, %rdx
        cmpq 12*32 +16(%rsp), %rdx
        movq %rdx, 12*32 +16(%rsp)
        movq (2*32)(%rsp), %rax ; movq (5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (11*32)(%rsp) ; movq 8+(2*32)(%rsp), %rax ; movq 8+(5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(11*32)(%rsp) ; movq 16+(2*32)(%rsp), %rax ; movq 16+(5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(11*32)(%rsp) ; movq 24+(2*32)(%rsp), %rax ; movq 24+(5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(11*32)(%rsp)
        movq (3*32)(%rsp), %rax ; movq (4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (10*32)(%rsp) ; movq 8+(3*32)(%rsp), %rax ; movq 8+(4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(10*32)(%rsp) ; movq 16+(3*32)(%rsp), %rax ; movq 16+(4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(10*32)(%rsp) ; movq 24+(3*32)(%rsp), %rax ; movq 24+(4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(10*32)(%rsp)
        xorl %edi, %edi ; movq (5*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %r8, %r9 ; mulxq 0x8+(3*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(3*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(3*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rdi, %r12 ; xorl %edi, %edi ; movq 0x8+(5*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(3*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rdi, %r13 ; adcxq %rdi, %r13 ; xorl %edi, %edi ; movq 0x10+(5*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(3*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rdi, %r14 ; adcxq %rdi, %r14 ; xorl %edi, %edi ; movq 0x18+(5*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(3*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rdi, %r15 ; adcxq %rdi, %r15 ; movl $0x26, %edx ; xorl %edi, %edi ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rdi, %r12 ; adcxq %rdi, %r12 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp) ; movq %r12, 0x20+(8*32)(%rsp)
        xorl %edi, %edi ; movq (2*32)(%rsp), %rdx ; mulxq (4*32)(%rsp), %r8, %r9 ; mulxq 0x8+(4*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(4*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(4*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rdi, %r12 ; xorl %edi, %edi ; movq 0x8+(2*32)(%rsp), %rdx ; mulxq (4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(4*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rdi, %r13 ; adcxq %rdi, %r13 ; xorl %edi, %edi ; movq 0x10+(2*32)(%rsp), %rdx ; mulxq (4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(4*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rdi, %r14 ; adcxq %rdi, %r14 ; xorl %edi, %edi ; movq 0x18+(2*32)(%rsp), %rdx ; mulxq (4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(4*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(4*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rdi, %r15 ; adcxq %rdi, %r15 ; movl $0x26, %edx ; xorl %edi, %edi ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rdi, %r12 ; adcxq %rdi, %r12 ; movq %r8, (6*32)(%rsp) ; movq %r9, 0x8+(6*32)(%rsp) ; movq %r10, 0x10+(6*32)(%rsp) ; movq %r11, 0x18+(6*32)(%rsp) ; movq %r12, 0x20+(6*32)(%rsp)
        movq (11*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(11*32)(%rsp), %r9, %r10 ; mulxq 0x18+(11*32)(%rsp), %r11, %r12 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq 0x18+(11*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (6*32)(%rsp), %r8 ; subq (8*32)(%rsp), %r8 ; movq 8+(6*32)(%rsp), %r9 ; sbbq 8+(8*32)(%rsp), %r9 ; movq 16+(6*32)(%rsp), %r10 ; sbbq 16+(8*32)(%rsp), %r10 ; movq 24+(6*32)(%rsp), %r11 ; sbbq 24+(8*32)(%rsp), %r11 ; movq 32+(6*32)(%rsp), %r12 ; sbbq 32+(8*32)(%rsp), %r12 ; xorl %ebx, %ebx ; subq $19000, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %r11 ; sbbq %rbx, %r12 ; addq $500, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq (6*32)(%rsp), %r8 ; addq (8*32)(%rsp), %r8 ; movq 8+(6*32)(%rsp), %r9 ; adcq 8+(8*32)(%rsp), %r9 ; movq 16+(6*32)(%rsp), %r10 ; adcq 16+(8*32)(%rsp), %r10 ; movq 24+(6*32)(%rsp), %r11 ; adcq 24+(8*32)(%rsp), %r11 ; movq 32+(6*32)(%rsp), %r12 ; adcq 32+(8*32)(%rsp), %r12 ; xorl %ebx, %ebx ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq (10*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(10*32)(%rsp), %r9, %r10 ; mulxq 0x18+(10*32)(%rsp), %r11, %r12 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq 0x18+(10*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (3*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(3*32)(%rsp), %r9, %r10 ; mulxq 0x18+(3*32)(%rsp), %r11, %r12 ; movq 0x10+(3*32)(%rsp), %rdx ; mulxq 0x18+(3*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (3*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(3*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(3*32)(%rsp), %rdx ; mulxq 0x8+(3*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(3*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(3*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(3*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq (8*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(8*32)(%rsp), %r9, %r10 ; mulxq 0x18+(8*32)(%rsp), %r11, %r12 ; movq 0x10+(8*32)(%rsp), %rdx ; mulxq 0x18+(8*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (8*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(8*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(8*32)(%rsp), %rdx ; mulxq 0x8+(8*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(8*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(8*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(8*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq (11*32)(%rsp), %r8 ; movq 8+(11*32)(%rsp), %r9 ; movq 16+(11*32)(%rsp), %r10 ; movq 24+(11*32)(%rsp), %r11 ; xorl %edi, %edi ; movq $0x1db42, %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq 8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 16+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 24+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rdi, %rbx ; adcxq %rdi, %rbx ; shldq $0x1, %r11, %rbx ; btr $63, %r11 ; movl $0x13, %edx ; imulq %rdx, %rbx ; addq %rbx, %r8 ; adcq %rdi, %r9 ; adcq %rdi, %r10 ; adcq %rdi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        xorl %ecx, %ecx ; movq (11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %r8, %r9 ; mulxq 0x8+(10*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(10*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(10*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(10*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(10*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(10*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        xorl %ecx, %ecx ; movq (5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %r8, %r9 ; mulxq 0x8+(6*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(6*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(6*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(6*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(6*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(6*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        xorl %ecx, %ecx ; movq (1*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %r8, %r9 ; mulxq 0x8+(3*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(3*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(3*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(1*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(3*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(1*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(3*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(1*32)(%rsp), %rdx ; mulxq (3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(3*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(3*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq 12*32 +8(%rsp), %rax
        subq $1, %rax
        movq %rax, 12*32 +8(%rsp)
        cmpq $3, %rax
        jnc Lcurve25519_x25519_scalarloop
        movq 12*32 +16(%rsp), %rdx
        testq %rdx, %rdx
        movq (8*32)(%rsp), %rax ; movq (10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (10*32)(%rsp) ; movq 8+(8*32)(%rsp), %rax ; movq 8+(10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(10*32)(%rsp) ; movq 16+(8*32)(%rsp), %rax ; movq 16+(10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(10*32)(%rsp) ; movq 24+(8*32)(%rsp), %rax ; movq 24+(10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(10*32)(%rsp)
        movq (3*32)(%rsp), %rax ; movq (7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (7*32)(%rsp) ; movq 8+(3*32)(%rsp), %rax ; movq 8+(7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(7*32)(%rsp) ; movq 16+(3*32)(%rsp), %rax ; movq 16+(7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(7*32)(%rsp) ; movq 24+(3*32)(%rsp), %rax ; movq 24+(7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(7*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(11*32)(%rsp), %r9, %r10 ; mulxq 0x18+(11*32)(%rsp), %r11, %r12 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq 0x18+(11*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(10*32)(%rsp), %r9, %r10 ; mulxq 0x18+(10*32)(%rsp), %r11, %r12 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq 0x18+(10*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq (11*32)(%rsp), %r8 ; movq 8+(11*32)(%rsp), %r9 ; movq 16+(11*32)(%rsp), %r10 ; movq 24+(11*32)(%rsp), %r11 ; xorl %edi, %edi ; movq $0x1db42, %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq 8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 16+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 24+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rdi, %rbx ; adcxq %rdi, %rbx ; shldq $0x1, %r11, %rbx ; btr $63, %r11 ; movl $0x13, %edx ; imulq %rdx, %rbx ; addq %rbx, %r8 ; adcq %rdi, %r9 ; adcq %rdi, %r10 ; adcq %rdi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        xorl %ecx, %ecx ; movq (11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %r8, %r9 ; mulxq 0x8+(10*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(10*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(10*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(10*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(10*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(10*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        xorl %ecx, %ecx ; movq (5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %r8, %r9 ; mulxq 0x8+(6*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(6*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(6*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(6*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(6*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(6*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(11*32)(%rsp), %r9, %r10 ; mulxq 0x18+(11*32)(%rsp), %r11, %r12 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq 0x18+(11*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(10*32)(%rsp), %r9, %r10 ; mulxq 0x18+(10*32)(%rsp), %r11, %r12 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq 0x18+(10*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq (11*32)(%rsp), %r8 ; movq 8+(11*32)(%rsp), %r9 ; movq 16+(11*32)(%rsp), %r10 ; movq 24+(11*32)(%rsp), %r11 ; xorl %edi, %edi ; movq $0x1db42, %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq 8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 16+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 24+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rdi, %rbx ; adcxq %rdi, %rbx ; shldq $0x1, %r11, %rbx ; btr $63, %r11 ; movl $0x13, %edx ; imulq %rdx, %rbx ; addq %rbx, %r8 ; adcq %rdi, %r9 ; adcq %rdi, %r10 ; adcq %rdi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        xorl %ecx, %ecx ; movq (11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %r8, %r9 ; mulxq 0x8+(10*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(10*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(10*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(10*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(10*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(10*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        xorl %ecx, %ecx ; movq (5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %r8, %r9 ; mulxq 0x8+(6*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(6*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(6*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(6*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(6*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(6*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(11*32)(%rsp), %r9, %r10 ; mulxq 0x18+(11*32)(%rsp), %r11, %r12 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq 0x18+(11*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq 0x8+(11*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rdx ; mulxq %rdx, %r8, %r15 ; mulxq 0x8+(10*32)(%rsp), %r9, %r10 ; mulxq 0x18+(10*32)(%rsp), %r11, %r12 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq 0x18+(10*32)(%rsp), %r13, %r14 ; xorl %ebx, %ebx ; mulxq (10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq 0x8+(10*32)(%rsp), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rbx, %r13 ; adoxq %rbx, %r14 ; adcq %rbx, %r14 ; xorl %ebx, %ebx ; adcxq %r9, %r9 ; adoxq %r15, %r9 ; movq 0x8+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r10, %r10 ; adoxq %rax, %r10 ; adcxq %r11, %r11 ; adoxq %rdx, %r11 ; movq 0x10+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %rdx ; adcxq %r12, %r12 ; adoxq %rax, %r12 ; adcxq %r13, %r13 ; adoxq %rdx, %r13 ; movq 0x18+(10*32)(%rsp), %rdx ; mulxq %rdx, %rax, %r15 ; adcxq %r14, %r14 ; adoxq %rax, %r14 ; adcxq %rbx, %r15 ; adoxq %rbx, %r15 ; movl $0x26, %edx ; xorl %ebx, %ebx ; mulxq %r12, %rax, %rcx ; adcxq %rax, %r8 ; adoxq %rcx, %r9 ; mulxq %r13, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10 ; mulxq %r14, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; adcxq %rbx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq (11*32)(%rsp), %r8 ; movq 8+(11*32)(%rsp), %r9 ; movq 16+(11*32)(%rsp), %r10 ; movq 24+(11*32)(%rsp), %r11 ; xorl %edi, %edi ; movq $0x1db42, %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq 8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 16+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 24+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rdi, %rbx ; adcxq %rdi, %rbx ; shldq $0x1, %r11, %rbx ; btr $63, %r11 ; movl $0x13, %edx ; imulq %rdx, %rbx ; addq %rbx, %r8 ; adcq %rdi, %r9 ; adcq %rdi, %r10 ; adcq %rdi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        xorl %ecx, %ecx ; movq (11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %r8, %r9 ; mulxq 0x8+(10*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(10*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(10*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(10*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(10*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(11*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(10*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        xorl %ecx, %ecx ; movq (5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %r8, %r9 ; mulxq 0x8+(6*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(6*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(6*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(6*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(6*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(5*32)(%rsp), %rdx ; mulxq (6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(6*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(6*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        leaq 224(%rsp), %rdi
        leaq 224(%rsp), %rsi
        movq %rdi, 0xc0(%rsp)
        xorl %eax, %eax
        leaq -0x13(%rax), %rcx
        notq %rax
        movq %rcx, (%rsp)
        movq %rax, 0x8(%rsp)
        movq %rax, 0x10(%rsp)
        btr $0x3f, %rax
        movq %rax, 0x18(%rsp)
        movq (%rsi), %rdx
        movq 0x8(%rsi), %rcx
        movq 0x10(%rsi), %r8
        movq 0x18(%rsi), %r9
        movl $0x1, %eax
        xorl %r10d, %r10d
        bts $0x3f, %r9
        adcq %r10, %rax
        imulq $0x13, %rax, %rax
        addq %rax, %rdx
        adcq %r10, %rcx
        adcq %r10, %r8
        adcq %r10, %r9
        movl $0x13, %eax
        cmovbq %r10, %rax
        subq %rax, %rdx
        sbbq %r10, %rcx
        sbbq %r10, %r8
        sbbq %r10, %r9
        btr $0x3f, %r9
        movq %rdx, 0x20(%rsp)
        movq %rcx, 0x28(%rsp)
        movq %r8, 0x30(%rsp)
        movq %r9, 0x38(%rsp)
        xorl %eax, %eax
        movq %rax, 0x40(%rsp)
        movq %rax, 0x48(%rsp)
        movq %rax, 0x50(%rsp)
        movq %rax, 0x58(%rsp)
        movabsq $0xa0f99e2375022099, %rax
        movq %rax, 0x60(%rsp)
        movabsq $0xa8c68f3f1d132595, %rax
        movq %rax, 0x68(%rsp)
        movabsq $0x6c6c893805ac5242, %rax
        movq %rax, 0x70(%rsp)
        movabsq $0x276508b241770615, %rax
        movq %rax, 0x78(%rsp)
        movq $0xa, 0x90(%rsp)
        movq $0x1, 0x98(%rsp)
        jmp Lcurve25519_x25519_midloop
Lcurve25519_x25519_inverseloop:
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %rdi
        andq %r11, %rdi
        addq %rax, %rdi
        movq %rdi, 0x80(%rsp)
        movq %r12, %rax
        andq %r13, %rax
        movq %r14, %rsi
        andq %r15, %rsi
        addq %rax, %rsi
        movq %rsi, 0x88(%rsp)
        xorl %ebx, %ebx
        movq (%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x20(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rdi
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq (%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x20(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        xorl %ecx, %ecx
        movq 0x8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x28(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, (%rsp)
        xorl %edi, %edi
        movq 0x8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rbp
        adcq %rdx, %rdi
        movq 0x28(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rdi
        shrdq $0x3b, %rbp, %rsi
        movq %rsi, 0x20(%rsp)
        xorl %esi, %esi
        movq 0x10(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rsi
        movq 0x30(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rsi
        shrdq $0x3b, %rcx, %rbx
        movq %rbx, 0x8(%rsp)
        xorl %ebx, %ebx
        movq 0x10(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x30(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rdi
        adcq %rdx, %rbx
        shrdq $0x3b, %rdi, %rbp
        movq %rbp, 0x28(%rsp)
        movq 0x18(%rsp), %rax
        xorq %r9, %rax
        movq %rax, %rbp
        sarq $0x3f, %rbp
        andq %r8, %rbp
        negq %rbp
        mulq %r8
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x38(%rsp), %rax
        xorq %r11, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r10, %rdx
        subq %rdx, %rbp
        mulq %r10
        addq %rax, %rsi
        adcq %rdx, %rbp
        shrdq $0x3b, %rsi, %rcx
        movq %rcx, 0x10(%rsp)
        shrdq $0x3b, %rbp, %rsi
        movq 0x18(%rsp), %rax
        movq %rsi, 0x18(%rsp)
        xorq %r13, %rax
        movq %rax, %rsi
        sarq $0x3f, %rsi
        andq %r12, %rsi
        negq %rsi
        mulq %r12
        addq %rax, %rbx
        adcq %rdx, %rsi
        movq 0x38(%rsp), %rax
        xorq %r15, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r14, %rdx
        subq %rdx, %rsi
        mulq %r14
        addq %rax, %rbx
        adcq %rdx, %rsi
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, 0x30(%rsp)
        shrdq $0x3b, %rsi, %rbx
        movq %rbx, 0x38(%rsp)
        movq 0x80(%rsp), %rbx
        movq 0x88(%rsp), %rbp
        xorl %ecx, %ecx
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x40(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x40(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x60(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x60(%rsp)
        xorl %ebx, %ebx
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq 0x48(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rcx, 0x48(%rsp)
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x68(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq %rsi, 0x68(%rsp)
        xorl %ecx, %ecx
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x50(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x50(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x70(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x70(%rsp)
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        movq %r9, %rbx
        andq %r8, %rbx
        negq %rbx
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbx
        mulq %r10
        addq %rax, %rcx
        adcq %rbx, %rdx
        movq %rdx, %rbx
        shldq $0x1, %rcx, %rdx
        sarq $0x3f, %rbx
        addq %rbx, %rdx
        movl $0x13, %eax
        imulq %rdx
        movq 0x40(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x40(%rsp)
        movq 0x48(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x48(%rsp)
        movq 0x50(%rsp), %r8
        adcq %rbx, %r8
        movq %r8, 0x50(%rsp)
        adcq %rbx, %rcx
        shlq $0x3f, %rax
        addq %rax, %rcx
        movq 0x58(%rsp), %rax
        movq %rcx, 0x58(%rsp)
        xorq %r13, %rax
        movq %r13, %rcx
        andq %r12, %rcx
        negq %rcx
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rcx
        movq 0x78(%rsp), %rax
        xorq %r15, %rax
        movq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rcx
        mulq %r14
        addq %rax, %rsi
        adcq %rcx, %rdx
        movq %rdx, %rcx
        shldq $0x1, %rsi, %rdx
        sarq $0x3f, %rcx
        movl $0x13, %eax
        addq %rcx, %rdx
        imulq %rdx
        movq 0x60(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x60(%rsp)
        movq 0x68(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x68(%rsp)
        movq 0x70(%rsp), %r8
        adcq %rcx, %r8
        movq %r8, 0x70(%rsp)
        adcq %rcx, %rsi
        shlq $0x3f, %rax
        addq %rax, %rsi
        movq %rsi, 0x78(%rsp)
Lcurve25519_x25519_midloop:
        movq 0x98(%rsp), %rsi
        movq (%rsp), %rdx
        movq 0x20(%rsp), %rcx
        movq %rdx, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        xorl %ebp, %ebp
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %rdx
        leaq (%rcx,%rax), %rdi
        shlq $0x16, %rdx
        shlq $0x16, %rdi
        sarq $0x2b, %rdx
        sarq $0x2b, %rdi
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %rbx
        leaq (%rcx,%rax), %rcx
        sarq $0x2a, %rbx
        sarq $0x2a, %rcx
        movq %rdx, 0xa0(%rsp)
        movq %rbx, 0xa8(%rsp)
        movq %rdi, 0xb0(%rsp)
        movq %rcx, 0xb8(%rsp)
        movq (%rsp), %r12
        imulq %r12, %rdi
        imulq %rdx, %r12
        movq 0x20(%rsp), %r13
        imulq %r13, %rbx
        imulq %rcx, %r13
        addq %rbx, %r12
        addq %rdi, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r10
        shlq $0x16, %r8
        shlq $0x16, %r10
        sarq $0x2b, %r8
        sarq $0x2b, %r10
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r15
        leaq (%rcx,%rax), %r11
        sarq $0x2a, %r15
        sarq $0x2a, %r11
        movq %r13, %rbx
        movq %r12, %rcx
        imulq %r8, %r12
        imulq %r15, %rbx
        addq %rbx, %r12
        imulq %r11, %r13
        imulq %r10, %rcx
        addq %rcx, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq 0xa0(%rsp), %rax
        imulq %r8, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r15, %rdx
        imulq 0xa8(%rsp), %r8
        imulq 0xb8(%rsp), %r15
        addq %r8, %r15
        leaq (%rax,%rdx), %r9
        movq 0xa0(%rsp), %rax
        imulq %r10, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r11, %rdx
        imulq 0xa8(%rsp), %r10
        imulq 0xb8(%rsp), %r11
        addq %r10, %r11
        leaq (%rax,%rdx), %r13
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r12
        shlq $0x15, %r8
        shlq $0x15, %r12
        sarq $0x2b, %r8
        sarq $0x2b, %r12
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r10
        leaq (%rcx,%rax), %r14
        sarq $0x2b, %r10
        sarq $0x2b, %r14
        movq %r9, %rax
        imulq %r8, %rax
        movq %r13, %rdx
        imulq %r10, %rdx
        imulq %r15, %r8
        imulq %r11, %r10
        addq %r8, %r10
        leaq (%rax,%rdx), %r8
        movq %r9, %rax
        imulq %r12, %rax
        movq %r13, %rdx
        imulq %r14, %rdx
        imulq %r15, %r12
        imulq %r11, %r14
        addq %r12, %r14
        leaq (%rax,%rdx), %r12
        movq %rsi, 0x98(%rsp)
        decq 0x90(%rsp)
        jne Lcurve25519_x25519_inverseloop
        movq (%rsp), %rax
        movq 0x20(%rsp), %rcx
        imulq %r8, %rax
        imulq %r10, %rcx
        addq %rcx, %rax
        sarq $0x3f, %rax
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        xorq %rax, %r9
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        xorq %rax, %r11
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        xorq %rax, %r13
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        xorq %rax, %r15
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %r12
        andq %r11, %r12
        addq %rax, %r12
        xorl %r13d, %r13d
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r12
        adcq %rdx, %r13
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r13
        adcq %rdx, %r14
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r13
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        andq %r8, %r9
        negq %r9
        mulq %r8
        addq %rax, %r15
        adcq %rdx, %r9
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %r9
        mulq %r10
        addq %rax, %r15
        adcq %rdx, %r9
        movq %r9, %rax
        shldq $0x1, %r15, %rax
        sarq $0x3f, %r9
        movl $0x13, %ebx
        leaq 0x1(%rax,%r9,1), %rax
        imulq %rbx
        xorl %ebp, %ebp
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r9, %r14
        adcq %r9, %r15
        shlq $0x3f, %rax
        addq %rax, %r15
        cmovns %rbp, %rbx
        subq %rbx, %r12
        sbbq %rbp, %r13
        sbbq %rbp, %r14
        sbbq %rbp, %r15
        btr $0x3f, %r15
        movq 0xc0(%rsp), %rdi
        movq %r12, (%rdi)
        movq %r13, 0x8(%rdi)
        movq %r14, 0x10(%rdi)
        movq %r15, 0x18(%rdi)
        movq 12*32(%rsp), %rbp
        xorl %edi, %edi ; movq (7*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %r8, %r9 ; mulxq 0x8+(10*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(10*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(10*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rdi, %r12 ; xorl %edi, %edi ; movq 0x8+(7*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(10*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rdi, %r13 ; adcxq %rdi, %r13 ; xorl %edi, %edi ; movq 0x10+(7*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(10*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rdi, %r14 ; adcxq %rdi, %r14 ; xorl %edi, %edi ; movq 0x18+(7*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(10*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rdi, %r15 ; adcxq %rdi, %r15 ; movl $0x26, %edx ; xorl %edi, %edi ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rdi, %r12 ; adcxq %rdi, %r12 ; shldq $0x1, %r11, %r12 ; movl $0x13, %edx ; incq %r12; bts $63, %r11 ; mulxq %r12, %rax, %rbx ; addq %rax, %r8 ; adcq %rbx, %r9 ; adcq %rdi, %r10 ; adcq %rdi, %r11 ; sbbq %rax, %rax ; notq %rax; andq %rdx, %rax ; subq %rax, %r8 ; sbbq %rdi, %r9 ; sbbq %rdi, %r10 ; sbbq %rdi, %r11 ; btr $63, %r11 ; movq %r8, 0(%rbp) ; movq %r9, 0x8+0(%rbp) ; movq %r10, 0x10+0(%rbp) ; movq %r11, 0x18+0(%rbp)
        addq $13*32, %rsp ; .cfi_adjust_cfa_offset -13*32
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.if {elf}
.size rscrypto_curve25519_x25519, .-rscrypto_curve25519_x25519
.size rscrypto_curve25519_x25519_byte, .-rscrypto_curve25519_x25519_byte
.endif

 .globl rscrypto_curve25519_x25519base
        .if {elf}
        .type rscrypto_curve25519_x25519base, %function
        .hidden rscrypto_curve25519_x25519base
        .endif
        .globl rscrypto_curve25519_x25519base_byte
        .if {elf}
        .type rscrypto_curve25519_x25519base_byte, %function
        .hidden rscrypto_curve25519_x25519base_byte
        .endif
        .text
rscrypto_curve25519_x25519base:
rscrypto_curve25519_x25519base_byte:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $488, %rsp ; .cfi_adjust_cfa_offset 488
        movq %rdi, 14*32(%rsp)
        movq (%rsi), %rax
        movq %rax, (%rsp)
        movq 8(%rsi), %rax
        movq %rax, 8(%rsp)
        movq 16(%rsi), %rax
        movq %rax, 16(%rsp)
        movq $0x3fffffffffffffff, %rax
        andq 24(%rsi), %rax
        movq %rax, 24(%rsp)
        movq (%rsp), %rax
        andq $8, %rax
        leaq Lcurve25519_x25519base_edwards25519_0g(%rip), %r10
        leaq Lcurve25519_x25519base_edwards25519_8g(%rip), %r11
        movq (%r10), %rax
        movq (%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*16(%rsp)
        movq 8*1(%r10), %rax
        movq 8*1(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*17(%rsp)
        movq 8*2(%r10), %rax
        movq 8*2(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*18(%rsp)
        movq 8*3(%r10), %rax
        movq 8*3(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*19(%rsp)
        movq 8*4(%r10), %rax
        movq 8*4(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*20(%rsp)
        movq 8*5(%r10), %rax
        movq 8*5(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*21(%rsp)
        movq 8*6(%r10), %rax
        movq 8*6(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*22(%rsp)
        movq 8*7(%r10), %rax
        movq 8*7(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*23(%rsp)
        movl $1, %eax
        movq %rax, 8*24(%rsp)
        movl $0, %eax
        movq %rax, 8*25(%rsp)
        movq %rax, 8*26(%rsp)
        movq %rax, 8*27(%rsp)
        movq 8*8(%r10), %rax
        movq 8*8(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*28(%rsp)
        movq 8*9(%r10), %rax
        movq 8*9(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*29(%rsp)
        movq 8*10(%r10), %rax
        movq 8*10(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*30(%rsp)
        movq 8*11(%r10), %rax
        movq 8*11(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*31(%rsp)
        movq $4, 14*32 +8(%rsp)
        leaq Lcurve25519_x25519base_edwards25519_gtable(%rip), %rax
        movq %rax, 15*32(%rsp)
        movq $0, 14*32 +16(%rsp)
Lcurve25519_x25519base_scalarloop:
        movq 14*32 +8(%rsp), %rax
        movq %rax, %rcx
        shrq $6, %rax
        movq (%rsp,%rax,8), %rax
        shrq %cl, %rax
        andq $15, %rax
        addq 14*32 +16(%rsp), %rax
        movq %rax, 14*32 +24(%rsp)
        cmpq $9, 14*32 +24(%rsp)
        sbbq %rax, %rax
        incq %rax
        movq %rax, 14*32 +16(%rsp)
        movq $16, %rdi
        subq 14*32 +24(%rsp), %rdi
        cmpq $0, 14*32 +16(%rsp)
        cmovzq 14*32 +24(%rsp), %rdi
        movq %rdi, 14*32 +24(%rsp)
        movl $1, %eax
        xorl %ebx, %ebx
        xorl %ecx, %ecx
        xorl %edx, %edx
        movl $1, %r8d
        xorl %r9d, %r9d
        xorl %r10d, %r10d
        xorl %r11d, %r11d
        xorl %r12d, %r12d
        xorl %r13d, %r13d
        xorl %r14d, %r14d
        xorl %r15d, %r15d
        movq 15*32(%rsp), %rbp
        cmpq $1, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $2, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $3, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $4, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $5, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $6, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $7, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $8, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        movq %rbp, 15*32(%rsp)
        cmpq $0, 14*32 +16(%rsp)
        movq %rax, %rsi
        cmovnzq %r8, %rsi
        cmovnzq %rax, %r8
        movq %rsi, 32(%rsp)
        movq %r8, 64(%rsp)
        movq %rbx, %rsi
        cmovnzq %r9, %rsi
        cmovnzq %rbx, %r9
        movq %rsi, 40(%rsp)
        movq %r9, 72(%rsp)
        movq %rcx, %rsi
        cmovnzq %r10, %rsi
        cmovnzq %rcx, %r10
        movq %rsi, 48(%rsp)
        movq %r10, 80(%rsp)
        movq %rdx, %rsi
        cmovnzq %r11, %rsi
        cmovnzq %rdx, %r11
        movq %rsi, 56(%rsp)
        movq %r11, 88(%rsp)
        movq $-19, %rax
        movq $-1, %rbx
        movq $-1, %rcx
        movq $0x7fffffffffffffff, %rdx
        subq %r12, %rax
        sbbq %r13, %rbx
        sbbq %r14, %rcx
        sbbq %r15, %rdx
        movq 14*32 +24(%rsp), %r8
        movq 14*32 +16(%rsp), %r9
        testq %r8, %r8
        cmovzq %r8, %r9
        testq %r9, %r9
        cmovzq %r12, %rax
        cmovzq %r13, %rbx
        cmovzq %r14, %rcx
        cmovzq %r15, %rdx
        movq %rax, 96(%rsp)
        movq %rbx, 104(%rsp)
        movq %rcx, 112(%rsp)
        movq %rdx, 120(%rsp)
        movq (6*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq %r8, %r8 ; movq 0x8+(6*32)(%rsp), %r9 ; adcq %r9, %r9 ; movq 0x10+(6*32)(%rsp), %r10 ; adcq %r10, %r10 ; movq 0x18+(6*32)(%rsp), %r11 ; adcq %r11, %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq (5*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (4*32)(%rsp), %r8 ; movq 8+(5*32)(%rsp), %r9 ; sbbq 8+(4*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(5*32)(%rsp), %r10 ; sbbq 16+(4*32)(%rsp), %r10 ; movq 24+(5*32)(%rsp), %rax ; sbbq 24+(4*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (9*32)(%rsp) ; movq %r9, 8+(9*32)(%rsp) ; movq %r10, 16+(9*32)(%rsp) ; movq %rax, 24+(9*32)(%rsp)
        movq (5*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (4*32)(%rsp), %r8 ; movq 0x8+(5*32)(%rsp), %r9 ; adcq 0x8+(4*32)(%rsp), %r9 ; movq 0x10+(5*32)(%rsp), %r10 ; adcq 0x10+(4*32)(%rsp), %r10 ; movq 0x18+(5*32)(%rsp), %r11 ; adcq 0x18+(4*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        xorl %ecx, %ecx ; movq (3*32)(%rsp), %rdx ; mulxq (7*32)(%rsp), %r8, %r9 ; mulxq 0x8+(7*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(7*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(7*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(3*32)(%rsp), %rdx ; mulxq (7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(7*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(3*32)(%rsp), %rdx ; mulxq (7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(7*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(3*32)(%rsp), %rdx ; mulxq (7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(7*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(7*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        xorl %ecx, %ecx ; movq (1*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %r8, %r9 ; mulxq 0x8+(9*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(9*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(9*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(1*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(9*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(1*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(9*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(1*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(9*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (9*32)(%rsp) ; movq %r9, 0x8+(9*32)(%rsp) ; movq %r10, 0x10+(9*32)(%rsp) ; movq %r11, 0x18+(9*32)(%rsp)
        xorl %ecx, %ecx ; movq (2*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %r8, %r9 ; mulxq 0x8+(10*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(10*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(10*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(2*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(10*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(2*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(10*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(2*32)(%rsp), %rdx ; mulxq (10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(10*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(10*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(8*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(8*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(8*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (12*32)(%rsp) ; movq %r9, 8+(12*32)(%rsp) ; movq %r10, 16+(12*32)(%rsp) ; movq %rax, 24+(12*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (11*32)(%rsp), %r8 ; movq 0x8+(8*32)(%rsp), %r9 ; adcq 0x8+(11*32)(%rsp), %r9 ; movq 0x10+(8*32)(%rsp), %r10 ; adcq 0x10+(11*32)(%rsp), %r10 ; movq 0x18+(8*32)(%rsp), %r11 ; adcq 0x18+(11*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (9*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(9*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(9*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(9*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (13*32)(%rsp) ; movq %r9, 8+(13*32)(%rsp) ; movq %r10, 16+(13*32)(%rsp) ; movq %rax, 24+(13*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (9*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(9*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(9*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(9*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (9*32)(%rsp) ; movq %r9, 0x8+(9*32)(%rsp) ; movq %r10, 0x10+(9*32)(%rsp) ; movq %r11, 0x18+(9*32)(%rsp)
        xorl %ecx, %ecx ; movq (8*32)(%rsp), %rdx ; mulxq (12*32)(%rsp), %r8, %r9 ; mulxq 0x8+(12*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(12*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(12*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(8*32)(%rsp), %rdx ; mulxq (12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(12*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(8*32)(%rsp), %rdx ; mulxq (12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(12*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(8*32)(%rsp), %rdx ; mulxq (12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(12*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(12*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (6*32)(%rsp) ; movq %r9, 0x8+(6*32)(%rsp) ; movq %r10, 0x10+(6*32)(%rsp) ; movq %r11, 0x18+(6*32)(%rsp)
        xorl %ecx, %ecx ; movq (12*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %r8, %r9 ; mulxq 0x8+(13*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(13*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(13*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(12*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(13*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(12*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(13*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(12*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(13*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (4*32)(%rsp) ; movq %r9, 0x8+(4*32)(%rsp) ; movq %r10, 0x10+(4*32)(%rsp) ; movq %r11, 0x18+(4*32)(%rsp)
        xorl %ecx, %ecx ; movq (9*32)(%rsp), %rdx ; mulxq (8*32)(%rsp), %r8, %r9 ; mulxq 0x8+(8*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(8*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(8*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(9*32)(%rsp), %rdx ; mulxq (8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(8*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(9*32)(%rsp), %rdx ; mulxq (8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(8*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(9*32)(%rsp), %rdx ; mulxq (8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(8*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(8*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        xorl %ecx, %ecx ; movq (9*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %r8, %r9 ; mulxq 0x8+(13*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(13*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(13*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rcx, %r12 ; xorl %ecx, %ecx ; movq 0x8+(9*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(13*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rcx, %r13 ; adcxq %rcx, %r13 ; xorl %ecx, %ecx ; movq 0x10+(9*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(13*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rcx, %r14 ; adcxq %rcx, %r14 ; xorl %ecx, %ecx ; movq 0x18+(9*32)(%rsp), %rdx ; mulxq (13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(13*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(13*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rcx, %r15 ; adcxq %rcx, %r15 ; movl $0x26, %edx ; xorl %ecx, %ecx ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rcx, %r12 ; adcxq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        addq $4, 14*32 +8(%rsp)
        cmpq $256, 14*32 +8(%rsp)
        jc Lcurve25519_x25519base_scalarloop
        movq (4*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(4*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(4*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(4*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (9*32)(%rsp) ; movq %r9, 0x8+(9*32)(%rsp) ; movq %r10, 0x10+(9*32)(%rsp) ; movq %r11, 0x18+(9*32)(%rsp)
        movq (4*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(4*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(4*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(4*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (10*32)(%rsp) ; movq %r9, 8+(10*32)(%rsp) ; movq %r10, 16+(10*32)(%rsp) ; movq %rax, 24+(10*32)(%rsp)
        leaq 256(%rsp), %rdi
        leaq 320(%rsp), %rsi
        movq %rdi, 0xc0(%rsp)
        xorl %eax, %eax
        leaq -0x13(%rax), %rcx
        notq %rax
        movq %rcx, (%rsp)
        movq %rax, 0x8(%rsp)
        movq %rax, 0x10(%rsp)
        btr $0x3f, %rax
        movq %rax, 0x18(%rsp)
        movq (%rsi), %rdx
        movq 0x8(%rsi), %rcx
        movq 0x10(%rsi), %r8
        movq 0x18(%rsi), %r9
        movl $0x1, %eax
        xorl %r10d, %r10d
        bts $0x3f, %r9
        adcq %r10, %rax
        imulq $0x13, %rax, %rax
        addq %rax, %rdx
        adcq %r10, %rcx
        adcq %r10, %r8
        adcq %r10, %r9
        movl $0x13, %eax
        cmovbq %r10, %rax
        subq %rax, %rdx
        sbbq %r10, %rcx
        sbbq %r10, %r8
        sbbq %r10, %r9
        btr $0x3f, %r9
        movq %rdx, 0x20(%rsp)
        movq %rcx, 0x28(%rsp)
        movq %r8, 0x30(%rsp)
        movq %r9, 0x38(%rsp)
        xorl %eax, %eax
        movq %rax, 0x40(%rsp)
        movq %rax, 0x48(%rsp)
        movq %rax, 0x50(%rsp)
        movq %rax, 0x58(%rsp)
        movabsq $0xa0f99e2375022099, %rax
        movq %rax, 0x60(%rsp)
        movabsq $0xa8c68f3f1d132595, %rax
        movq %rax, 0x68(%rsp)
        movabsq $0x6c6c893805ac5242, %rax
        movq %rax, 0x70(%rsp)
        movabsq $0x276508b241770615, %rax
        movq %rax, 0x78(%rsp)
        movq $0xa, 0x90(%rsp)
        movq $0x1, 0x98(%rsp)
        jmp Lcurve25519_x25519base_midloop
Lcurve25519_x25519base_inverseloop:
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %rdi
        andq %r11, %rdi
        addq %rax, %rdi
        movq %rdi, 0x80(%rsp)
        movq %r12, %rax
        andq %r13, %rax
        movq %r14, %rsi
        andq %r15, %rsi
        addq %rax, %rsi
        movq %rsi, 0x88(%rsp)
        xorl %ebx, %ebx
        movq (%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x20(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rdi
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq (%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x20(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        xorl %ecx, %ecx
        movq 0x8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x28(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, (%rsp)
        xorl %edi, %edi
        movq 0x8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rbp
        adcq %rdx, %rdi
        movq 0x28(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rdi
        shrdq $0x3b, %rbp, %rsi
        movq %rsi, 0x20(%rsp)
        xorl %esi, %esi
        movq 0x10(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rsi
        movq 0x30(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rsi
        shrdq $0x3b, %rcx, %rbx
        movq %rbx, 0x8(%rsp)
        xorl %ebx, %ebx
        movq 0x10(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x30(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rdi
        adcq %rdx, %rbx
        shrdq $0x3b, %rdi, %rbp
        movq %rbp, 0x28(%rsp)
        movq 0x18(%rsp), %rax
        xorq %r9, %rax
        movq %rax, %rbp
        sarq $0x3f, %rbp
        andq %r8, %rbp
        negq %rbp
        mulq %r8
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x38(%rsp), %rax
        xorq %r11, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r10, %rdx
        subq %rdx, %rbp
        mulq %r10
        addq %rax, %rsi
        adcq %rdx, %rbp
        shrdq $0x3b, %rsi, %rcx
        movq %rcx, 0x10(%rsp)
        shrdq $0x3b, %rbp, %rsi
        movq 0x18(%rsp), %rax
        movq %rsi, 0x18(%rsp)
        xorq %r13, %rax
        movq %rax, %rsi
        sarq $0x3f, %rsi
        andq %r12, %rsi
        negq %rsi
        mulq %r12
        addq %rax, %rbx
        adcq %rdx, %rsi
        movq 0x38(%rsp), %rax
        xorq %r15, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r14, %rdx
        subq %rdx, %rsi
        mulq %r14
        addq %rax, %rbx
        adcq %rdx, %rsi
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, 0x30(%rsp)
        shrdq $0x3b, %rsi, %rbx
        movq %rbx, 0x38(%rsp)
        movq 0x80(%rsp), %rbx
        movq 0x88(%rsp), %rbp
        xorl %ecx, %ecx
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x40(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x40(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x60(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x60(%rsp)
        xorl %ebx, %ebx
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq 0x48(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rcx, 0x48(%rsp)
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x68(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq %rsi, 0x68(%rsp)
        xorl %ecx, %ecx
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x50(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x50(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x70(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x70(%rsp)
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        movq %r9, %rbx
        andq %r8, %rbx
        negq %rbx
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbx
        mulq %r10
        addq %rax, %rcx
        adcq %rbx, %rdx
        movq %rdx, %rbx
        shldq $0x1, %rcx, %rdx
        sarq $0x3f, %rbx
        addq %rbx, %rdx
        movl $0x13, %eax
        imulq %rdx
        movq 0x40(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x40(%rsp)
        movq 0x48(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x48(%rsp)
        movq 0x50(%rsp), %r8
        adcq %rbx, %r8
        movq %r8, 0x50(%rsp)
        adcq %rbx, %rcx
        shlq $0x3f, %rax
        addq %rax, %rcx
        movq 0x58(%rsp), %rax
        movq %rcx, 0x58(%rsp)
        xorq %r13, %rax
        movq %r13, %rcx
        andq %r12, %rcx
        negq %rcx
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rcx
        movq 0x78(%rsp), %rax
        xorq %r15, %rax
        movq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rcx
        mulq %r14
        addq %rax, %rsi
        adcq %rcx, %rdx
        movq %rdx, %rcx
        shldq $0x1, %rsi, %rdx
        sarq $0x3f, %rcx
        movl $0x13, %eax
        addq %rcx, %rdx
        imulq %rdx
        movq 0x60(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x60(%rsp)
        movq 0x68(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x68(%rsp)
        movq 0x70(%rsp), %r8
        adcq %rcx, %r8
        movq %r8, 0x70(%rsp)
        adcq %rcx, %rsi
        shlq $0x3f, %rax
        addq %rax, %rsi
        movq %rsi, 0x78(%rsp)
Lcurve25519_x25519base_midloop:
        movq 0x98(%rsp), %rsi
        movq (%rsp), %rdx
        movq 0x20(%rsp), %rcx
        movq %rdx, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        xorl %ebp, %ebp
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %rdx
        leaq (%rcx,%rax), %rdi
        shlq $0x16, %rdx
        shlq $0x16, %rdi
        sarq $0x2b, %rdx
        sarq $0x2b, %rdi
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %rbx
        leaq (%rcx,%rax), %rcx
        sarq $0x2a, %rbx
        sarq $0x2a, %rcx
        movq %rdx, 0xa0(%rsp)
        movq %rbx, 0xa8(%rsp)
        movq %rdi, 0xb0(%rsp)
        movq %rcx, 0xb8(%rsp)
        movq (%rsp), %r12
        imulq %r12, %rdi
        imulq %rdx, %r12
        movq 0x20(%rsp), %r13
        imulq %r13, %rbx
        imulq %rcx, %r13
        addq %rbx, %r12
        addq %rdi, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r10
        shlq $0x16, %r8
        shlq $0x16, %r10
        sarq $0x2b, %r8
        sarq $0x2b, %r10
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r15
        leaq (%rcx,%rax), %r11
        sarq $0x2a, %r15
        sarq $0x2a, %r11
        movq %r13, %rbx
        movq %r12, %rcx
        imulq %r8, %r12
        imulq %r15, %rbx
        addq %rbx, %r12
        imulq %r11, %r13
        imulq %r10, %rcx
        addq %rcx, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq 0xa0(%rsp), %rax
        imulq %r8, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r15, %rdx
        imulq 0xa8(%rsp), %r8
        imulq 0xb8(%rsp), %r15
        addq %r8, %r15
        leaq (%rax,%rdx), %r9
        movq 0xa0(%rsp), %rax
        imulq %r10, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r11, %rdx
        imulq 0xa8(%rsp), %r10
        imulq 0xb8(%rsp), %r11
        addq %r10, %r11
        leaq (%rax,%rdx), %r13
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r12
        shlq $0x15, %r8
        shlq $0x15, %r12
        sarq $0x2b, %r8
        sarq $0x2b, %r12
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r10
        leaq (%rcx,%rax), %r14
        sarq $0x2b, %r10
        sarq $0x2b, %r14
        movq %r9, %rax
        imulq %r8, %rax
        movq %r13, %rdx
        imulq %r10, %rdx
        imulq %r15, %r8
        imulq %r11, %r10
        addq %r8, %r10
        leaq (%rax,%rdx), %r8
        movq %r9, %rax
        imulq %r12, %rax
        movq %r13, %rdx
        imulq %r14, %rdx
        imulq %r15, %r12
        imulq %r11, %r14
        addq %r12, %r14
        leaq (%rax,%rdx), %r12
        movq %rsi, 0x98(%rsp)
        decq 0x90(%rsp)
        jne Lcurve25519_x25519base_inverseloop
        movq (%rsp), %rax
        movq 0x20(%rsp), %rcx
        imulq %r8, %rax
        imulq %r10, %rcx
        addq %rcx, %rax
        sarq $0x3f, %rax
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        xorq %rax, %r9
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        xorq %rax, %r11
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        xorq %rax, %r13
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        xorq %rax, %r15
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %r12
        andq %r11, %r12
        addq %rax, %r12
        xorl %r13d, %r13d
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r12
        adcq %rdx, %r13
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r13
        adcq %rdx, %r14
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r13
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        andq %r8, %r9
        negq %r9
        mulq %r8
        addq %rax, %r15
        adcq %rdx, %r9
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %r9
        mulq %r10
        addq %rax, %r15
        adcq %rdx, %r9
        movq %r9, %rax
        shldq $0x1, %r15, %rax
        sarq $0x3f, %r9
        movl $0x13, %ebx
        leaq 0x1(%rax,%r9,1), %rax
        imulq %rbx
        xorl %ebp, %ebp
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r9, %r14
        adcq %r9, %r15
        shlq $0x3f, %rax
        addq %rax, %r15
        cmovns %rbp, %rbx
        subq %rbx, %r12
        sbbq %rbp, %r13
        sbbq %rbp, %r14
        sbbq %rbp, %r15
        btr $0x3f, %r15
        movq 0xc0(%rsp), %rdi
        movq %r12, (%rdi)
        movq %r13, 0x8(%rdi)
        movq %r14, 0x10(%rdi)
        movq %r15, 0x18(%rdi)
        movq 14*32(%rsp), %rbp
        xorl %esi, %esi ; movq (8*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %r8, %r9 ; mulxq 0x8+(9*32)(%rsp), %rax, %r10 ; addq %rax, %r9 ; mulxq 0x10+(9*32)(%rsp), %rax, %r11 ; adcq %rax, %r10 ; mulxq 0x18+(9*32)(%rsp), %rax, %r12 ; adcq %rax, %r11 ; adcq %rsi, %r12 ; xorl %esi, %esi ; movq 0x8+(8*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq 0x8+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x10+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x18+(9*32)(%rsp), %rax, %r13 ; adcxq %rax, %r12 ; adoxq %rsi, %r13 ; adcxq %rsi, %r13 ; xorl %esi, %esi ; movq 0x10+(8*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq 0x8+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x10+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x18+(9*32)(%rsp), %rax, %r14 ; adcxq %rax, %r13 ; adoxq %rsi, %r14 ; adcxq %rsi, %r14 ; xorl %esi, %esi ; movq 0x18+(8*32)(%rsp), %rdx ; mulxq (9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12 ; mulxq 0x8+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13 ; mulxq 0x10+(9*32)(%rsp), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14 ; mulxq 0x18+(9*32)(%rsp), %rax, %r15 ; adcxq %rax, %r14 ; adoxq %rsi, %r15 ; adcxq %rsi, %r15 ; movl $0x26, %edx ; xorl %esi, %esi ; mulxq %r12, %rax, %rbx ; adcxq %rax, %r8 ; adoxq %rbx, %r9 ; mulxq %r13, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10 ; mulxq %r14, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11 ; mulxq %r15, %rax, %r12 ; adcxq %rax, %r11 ; adoxq %rsi, %r12 ; adcxq %rsi, %r12 ; shldq $0x1, %r11, %r12 ; movl $0x13, %edx ; incq %r12; bts $63, %r11 ; mulxq %r12, %rax, %rbx ; addq %rax, %r8 ; adcq %rbx, %r9 ; adcq %rsi, %r10 ; adcq %rsi, %r11 ; sbbq %rax, %rax ; notq %rax; andq %rdx, %rax ; subq %rax, %r8 ; sbbq %rsi, %r9 ; sbbq %rsi, %r10 ; sbbq %rsi, %r11 ; btr $63, %r11 ; movq %r8, (0*32)(%rbp) ; movq %r9, 0x8+(0*32)(%rbp) ; movq %r10, 0x10+(0*32)(%rbp) ; movq %r11, 0x18+(0*32)(%rbp)
        addq $488, %rsp ; .cfi_adjust_cfa_offset -488
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.if {elf}
.size rscrypto_curve25519_x25519base, .-rscrypto_curve25519_x25519base
        .section .rodata.rscrypto_curve25519_x25519base, "a", %progbits
.else
        .section .rdata, "dr"
.endif
// The tables are read-only data: in .text, a disassembler would read them as instructions.
        .p2align 3
Lcurve25519_x25519base_edwards25519_0g:
        .quad 0x251037f7cf4e861d
        .quad 0x10ede0fb19fb128f
        .quad 0x96c033b175f5e2c8
        .quad 0x055f070d6c15fb0d
        .quad 0x7c52af2c97473e69
        .quad 0x022f82391bad8378
        .quad 0x9991e1b02adb476f
        .quad 0x511144a03a99b855
        .quad 0x5fafc3b88ff2e4ae
        .quad 0x855e4ff0de1230ff
        .quad 0x72e302a348492870
        .quad 0x1253c19e53dbe1bc
Lcurve25519_x25519base_edwards25519_8g:
        .quad 0x331d086e0d9abcaa
        .quad 0x1e23c96d311a10c9
        .quad 0x96d0f95e58c13478
        .quad 0x2f72f7384fcfcc59
        .quad 0x39a6cd1cfd7d87c9
        .quad 0x9867a0abd8ae153a
        .quad 0xa49d2a5f35986745
        .quad 0x57012940cdfe82e1
        .quad 0x5046a6532ec5544a
        .quad 0x6d674004739ff6c9
        .quad 0x9bbaa44b234a70e3
        .quad 0x5e6d8901138cf386
Lcurve25519_x25519base_edwards25519_gtable:
        rscrypto_x25519_base_table

 .globl rscrypto_curve25519_x25519_alt
        .if {elf}
        .type rscrypto_curve25519_x25519_alt, %function
        .hidden rscrypto_curve25519_x25519_alt
        .endif
        .globl rscrypto_curve25519_x25519_byte_alt
        .if {elf}
        .type rscrypto_curve25519_x25519_byte_alt, %function
        .hidden rscrypto_curve25519_x25519_byte_alt
        .endif
        .text
rscrypto_curve25519_x25519_alt:
rscrypto_curve25519_x25519_byte_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $13*32, %rsp ; .cfi_adjust_cfa_offset 13*32
        movq %rdi, 12*32(%rsp)
        movq (%rsi), %rax
        movq %rax, (%rsp)
        movq 8(%rsi), %rax
        movq %rax, 8(%rsp)
        movq 16(%rsi), %rax
        movq %rax, 16(%rsp)
        movq 24(%rsi), %rax
        movq %rax, 24(%rsp)
        movq (%rdx), %r8
        movq 8(%rdx), %r9
        movq 16(%rdx), %r10
        movq 24(%rdx), %r11
        btr $63, %r11
        movq %r8, 32(%rsp)
        movq %r9, 40(%rsp)
        movq %r10, 48(%rsp)
        movq %r11, 56(%rsp)
        movl $1, %eax
        movq %rax, 12*32 +16(%rsp)
        movq %r8, 256(%rsp)
        movq %rax, 96(%rsp)
        xorl %eax, %eax
        movq %r9, 264(%rsp)
        movq %rax, 104(%rsp)
        movq %r10, 272(%rsp)
        movq %rax, 112(%rsp)
        movq %r11, 280(%rsp)
        movq %rax, 120(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (3*32)(%rsp), %r8 ; movq 8+(8*32)(%rsp), %r9 ; sbbq 8+(3*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(8*32)(%rsp), %r10 ; sbbq 16+(3*32)(%rsp), %r10 ; movq 24+(8*32)(%rsp), %rax ; sbbq 24+(3*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (3*32)(%rsp), %r8 ; movq 0x8+(8*32)(%rsp), %r9 ; adcq 0x8+(3*32)(%rsp), %r9 ; movq 0x10+(8*32)(%rsp), %r10 ; adcq 0x10+(3*32)(%rsp), %r10 ; movq 0x18+(8*32)(%rsp), %r11 ; adcq 0x18+(3*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (11*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq $0x1db42, %rsi ; movq (6*32)(%rsp), %rax ; mulq %rsi; movq %rax, %r8 ; movq %rdx, %r9 ; movq 0x8+(6*32)(%rsp), %rax ; xorq %r10, %r10 ; mulq %rsi; addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x10+(6*32)(%rsp), %rax ; mulq %rsi; addq %rax, %r10 ; adcq $0x0, %rdx ; movq 0x18+(6*32)(%rsp), %rax ; movq %rdx, %r11 ; mulq %rsi; xorl %esi, %esi ; addq %rax, %r11 ; adcq %rsi, %rdx ; addq (11*32)(%rsp), %r8 ; adcq 0x8+(11*32)(%rsp), %r9 ; adcq 0x10+(11*32)(%rsp), %r10 ; adcq 0x18+(11*32)(%rsp), %r11 ; adcq %rsi, %rdx ; shldq $0x1, %r11, %rdx ; btr $63, %r11 ; movl $0x13, %ebx ; imulq %rbx, %rdx ; addq %rdx, %r8 ; adcq %rsi, %r9 ; adcq %rsi, %r10 ; adcq %rsi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq (11*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (6*32)(%rsp), %rax ; mulq (5*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        movl $253, %eax
        movq %rax, 12*32 +8(%rsp)
Lcurve25519_x25519_alt_scalarloop:
        movq (8*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (3*32)(%rsp), %r8 ; movq 8+(8*32)(%rsp), %r9 ; sbbq 8+(3*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(8*32)(%rsp), %r10 ; sbbq 16+(3*32)(%rsp), %r10 ; movq 24+(8*32)(%rsp), %rax ; sbbq 24+(3*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (2*32)(%rsp) ; movq %r9, 8+(2*32)(%rsp) ; movq %r10, 16+(2*32)(%rsp) ; movq %rax, 24+(2*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (4*32)(%rsp) ; movq %r9, 0x8+(4*32)(%rsp) ; movq %r10, 0x10+(4*32)(%rsp) ; movq %r11, 0x18+(4*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (5*32)(%rsp) ; movq %r9, 8+(5*32)(%rsp) ; movq %r10, 16+(5*32)(%rsp) ; movq %rax, 24+(5*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (3*32)(%rsp), %r8 ; movq 0x8+(8*32)(%rsp), %r9 ; adcq 0x8+(3*32)(%rsp), %r9 ; movq 0x10+(8*32)(%rsp), %r10 ; adcq 0x10+(3*32)(%rsp), %r10 ; movq 0x18+(8*32)(%rsp), %r11 ; adcq 0x18+(3*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq 12*32 +8(%rsp), %rdx
        movq %rdx, %rcx
        shrq $6, %rdx
        movq (%rsp,%rdx,8), %rdx
        shrq %cl, %rdx
        andq $1, %rdx
        cmpq 12*32 +16(%rsp), %rdx
        movq %rdx, 12*32 +16(%rsp)
        movq (2*32)(%rsp), %rax ; movq (5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (11*32)(%rsp) ; movq 8+(2*32)(%rsp), %rax ; movq 8+(5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(11*32)(%rsp) ; movq 16+(2*32)(%rsp), %rax ; movq 16+(5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(11*32)(%rsp) ; movq 24+(2*32)(%rsp), %rax ; movq 24+(5*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(11*32)(%rsp)
        movq (3*32)(%rsp), %rax ; movq (4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (10*32)(%rsp) ; movq 8+(3*32)(%rsp), %rax ; movq 8+(4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(10*32)(%rsp) ; movq 16+(3*32)(%rsp), %rax ; movq 16+(4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(10*32)(%rsp) ; movq 24+(3*32)(%rsp), %rax ; movq 24+(4*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(10*32)(%rsp)
        movq (3*32)(%rsp), %rax ; mulq (5*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (3*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(3*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (3*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(3*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (3*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(3*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(3*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(3*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(3*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(3*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(3*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(3*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp) ; movq %r12, 0x20+(8*32)(%rsp)
        movq (4*32)(%rsp), %rax ; mulq (2*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (4*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(4*32)(%rsp), %rax ; mulq (2*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (4*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(4*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(4*32)(%rsp), %rax ; mulq (2*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (4*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(4*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(4*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(4*32)(%rsp), %rax ; mulq (2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(4*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(4*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(4*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(4*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(4*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(4*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; movq %r8, (6*32)(%rsp) ; movq %r9, 0x8+(6*32)(%rsp) ; movq %r10, 0x10+(6*32)(%rsp) ; movq %r11, 0x18+(6*32)(%rsp) ; movq %r12, 0x20+(6*32)(%rsp)
        movq (11*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (11*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (6*32)(%rsp), %r8 ; subq (8*32)(%rsp), %r8 ; movq 8+(6*32)(%rsp), %r9 ; sbbq 8+(8*32)(%rsp), %r9 ; movq 16+(6*32)(%rsp), %r10 ; sbbq 16+(8*32)(%rsp), %r10 ; movq 24+(6*32)(%rsp), %r11 ; sbbq 24+(8*32)(%rsp), %r11 ; movq 32+(6*32)(%rsp), %r12 ; sbbq 32+(8*32)(%rsp), %r12 ; xorl %ebx, %ebx ; subq $19000, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %r11 ; sbbq %rbx, %r12 ; addq $500, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq (6*32)(%rsp), %r8 ; addq (8*32)(%rsp), %r8 ; movq 8+(6*32)(%rsp), %r9 ; adcq 8+(8*32)(%rsp), %r9 ; movq 16+(6*32)(%rsp), %r10 ; adcq 16+(8*32)(%rsp), %r10 ; movq 24+(6*32)(%rsp), %r11 ; adcq 24+(8*32)(%rsp), %r11 ; movq 32+(6*32)(%rsp), %r12 ; adcq 32+(8*32)(%rsp), %r12 ; xorl %ebx, %ebx ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rbx, %r9 ; adcq %rbx, %r10 ; adcq %rbx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (3*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (3*32)(%rsp), %rax ; mulq 0x8+(3*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(3*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (3*32)(%rsp), %rax ; mulq 0x10+(3*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (3*32)(%rsp), %rax ; mulq 0x18+(3*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x10+(3*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x18+(3*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(3*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(3*32)(%rsp), %rax ; mulq 0x18+(3*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(3*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq (8*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (8*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(8*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (8*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (8*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(8*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(8*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(8*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(8*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(8*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq $0x1db42, %rsi ; movq (6*32)(%rsp), %rax ; mulq %rsi; movq %rax, %r8 ; movq %rdx, %r9 ; movq 0x8+(6*32)(%rsp), %rax ; xorq %r10, %r10 ; mulq %rsi; addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x10+(6*32)(%rsp), %rax ; mulq %rsi; addq %rax, %r10 ; adcq $0x0, %rdx ; movq 0x18+(6*32)(%rsp), %rax ; movq %rdx, %r11 ; mulq %rsi; xorl %esi, %esi ; addq %rax, %r11 ; adcq %rsi, %rdx ; addq (11*32)(%rsp), %r8 ; adcq 0x8+(11*32)(%rsp), %r9 ; adcq 0x10+(11*32)(%rsp), %r10 ; adcq 0x18+(11*32)(%rsp), %r11 ; adcq %rsi, %rdx ; shldq $0x1, %r11, %rdx ; btr $63, %r11 ; movl $0x13, %ebx ; imulq %rbx, %rdx ; addq %rdx, %r8 ; adcq %rsi, %r9 ; adcq %rsi, %r10 ; adcq %rsi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq (11*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (6*32)(%rsp), %rax ; mulq (5*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        movq (3*32)(%rsp), %rax ; mulq (1*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (3*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(3*32)(%rsp), %rax ; mulq (1*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (3*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(3*32)(%rsp), %rax ; mulq (1*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (3*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(3*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(3*32)(%rsp), %rax ; mulq (1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(3*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(3*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(3*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(3*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(3*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(3*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (3*32)(%rsp) ; movq %r9, 0x8+(3*32)(%rsp) ; movq %r10, 0x10+(3*32)(%rsp) ; movq %r11, 0x18+(3*32)(%rsp)
        movq 12*32 +8(%rsp), %rax
        subq $1, %rax
        movq %rax, 12*32 +8(%rsp)
        cmpq $3, %rax
        jnc Lcurve25519_x25519_alt_scalarloop
        movq 12*32 +16(%rsp), %rdx
        testq %rdx, %rdx
        movq (8*32)(%rsp), %rax ; movq (10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (10*32)(%rsp) ; movq 8+(8*32)(%rsp), %rax ; movq 8+(10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(10*32)(%rsp) ; movq 16+(8*32)(%rsp), %rax ; movq 16+(10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(10*32)(%rsp) ; movq 24+(8*32)(%rsp), %rax ; movq 24+(10*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(10*32)(%rsp)
        movq (3*32)(%rsp), %rax ; movq (7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, (7*32)(%rsp) ; movq 8+(3*32)(%rsp), %rax ; movq 8+(7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 8+(7*32)(%rsp) ; movq 16+(3*32)(%rsp), %rax ; movq 16+(7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 16+(7*32)(%rsp) ; movq 24+(3*32)(%rsp), %rax ; movq 24+(7*32)(%rsp), %rcx ; cmovzq %rcx, %rax ; movq %rax, 24+(7*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (11*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq $0x1db42, %rsi ; movq (6*32)(%rsp), %rax ; mulq %rsi; movq %rax, %r8 ; movq %rdx, %r9 ; movq 0x8+(6*32)(%rsp), %rax ; xorq %r10, %r10 ; mulq %rsi; addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x10+(6*32)(%rsp), %rax ; mulq %rsi; addq %rax, %r10 ; adcq $0x0, %rdx ; movq 0x18+(6*32)(%rsp), %rax ; movq %rdx, %r11 ; mulq %rsi; xorl %esi, %esi ; addq %rax, %r11 ; adcq %rsi, %rdx ; addq (11*32)(%rsp), %r8 ; adcq 0x8+(11*32)(%rsp), %r9 ; adcq 0x10+(11*32)(%rsp), %r10 ; adcq 0x18+(11*32)(%rsp), %r11 ; adcq %rsi, %rdx ; shldq $0x1, %r11, %rdx ; btr $63, %r11 ; movl $0x13, %ebx ; imulq %rbx, %rdx ; addq %rdx, %r8 ; adcq %rsi, %r9 ; adcq %rsi, %r10 ; adcq %rsi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq (11*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (6*32)(%rsp), %rax ; mulq (5*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (11*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq $0x1db42, %rsi ; movq (6*32)(%rsp), %rax ; mulq %rsi; movq %rax, %r8 ; movq %rdx, %r9 ; movq 0x8+(6*32)(%rsp), %rax ; xorq %r10, %r10 ; mulq %rsi; addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x10+(6*32)(%rsp), %rax ; mulq %rsi; addq %rax, %r10 ; adcq $0x0, %rdx ; movq 0x18+(6*32)(%rsp), %rax ; movq %rdx, %r11 ; mulq %rsi; xorl %esi, %esi ; addq %rax, %r11 ; adcq %rsi, %rdx ; addq (11*32)(%rsp), %r8 ; adcq 0x8+(11*32)(%rsp), %r9 ; adcq 0x10+(11*32)(%rsp), %r10 ; adcq 0x18+(11*32)(%rsp), %r11 ; adcq %rsi, %rdx ; shldq $0x1, %r11, %rdx ; btr $63, %r11 ; movl $0x13, %ebx ; imulq %rbx, %rdx ; addq %rdx, %r8 ; adcq %rsi, %r9 ; adcq %rsi, %r10 ; adcq %rsi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq (11*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (6*32)(%rsp), %rax ; mulq (5*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (11*32)(%rsp) ; movq %r9, 8+(11*32)(%rsp) ; movq %r10, 16+(11*32)(%rsp) ; movq %rax, 24+(11*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (11*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (11*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(11*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(11*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq %rax; movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r11 ; addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r12 ; addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r13 ; addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r14 ; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(10*32)(%rsp); addq %rax, %rax ; adcq %rdx, %rdx ; adcq $0x0, %r15 ; addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq %rax; addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (6*32)(%rsp) ; movq %r9, 8+(6*32)(%rsp) ; movq %r10, 16+(6*32)(%rsp) ; movq %rax, 24+(6*32)(%rsp)
        movq $0x1db42, %rsi ; movq (6*32)(%rsp), %rax ; mulq %rsi; movq %rax, %r8 ; movq %rdx, %r9 ; movq 0x8+(6*32)(%rsp), %rax ; xorq %r10, %r10 ; mulq %rsi; addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x10+(6*32)(%rsp), %rax ; mulq %rsi; addq %rax, %r10 ; adcq $0x0, %rdx ; movq 0x18+(6*32)(%rsp), %rax ; movq %rdx, %r11 ; mulq %rsi; xorl %esi, %esi ; addq %rax, %r11 ; adcq %rsi, %rdx ; addq (11*32)(%rsp), %r8 ; adcq 0x8+(11*32)(%rsp), %r9 ; adcq 0x10+(11*32)(%rsp), %r10 ; adcq 0x18+(11*32)(%rsp), %r11 ; adcq %rsi, %rdx ; shldq $0x1, %r11, %rdx ; btr $63, %r11 ; movl $0x13, %ebx ; imulq %rbx, %rdx ; addq %rdx, %r8 ; adcq %rsi, %r9 ; adcq %rsi, %r10 ; adcq %rsi, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq (11*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(10*32)(%rsp), %rax ; mulq (11*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x8+(11*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x10+(11*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x18+(11*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (6*32)(%rsp), %rax ; mulq (5*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(6*32)(%rsp), %rax ; mulq (5*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x8+(5*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x10+(5*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(6*32)(%rsp), %rax ; mulq 0x18+(5*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        leaq 224(%rsp), %rdi
        leaq 224(%rsp), %rsi
        movq %rdi, 0xc0(%rsp)
        xorl %eax, %eax
        leaq -0x13(%rax), %rcx
        notq %rax
        movq %rcx, (%rsp)
        movq %rax, 0x8(%rsp)
        movq %rax, 0x10(%rsp)
        btr $0x3f, %rax
        movq %rax, 0x18(%rsp)
        movq (%rsi), %rdx
        movq 0x8(%rsi), %rcx
        movq 0x10(%rsi), %r8
        movq 0x18(%rsi), %r9
        movl $0x1, %eax
        xorl %r10d, %r10d
        bts $0x3f, %r9
        adcq %r10, %rax
        imulq $0x13, %rax, %rax
        addq %rax, %rdx
        adcq %r10, %rcx
        adcq %r10, %r8
        adcq %r10, %r9
        movl $0x13, %eax
        cmovbq %r10, %rax
        subq %rax, %rdx
        sbbq %r10, %rcx
        sbbq %r10, %r8
        sbbq %r10, %r9
        btr $0x3f, %r9
        movq %rdx, 0x20(%rsp)
        movq %rcx, 0x28(%rsp)
        movq %r8, 0x30(%rsp)
        movq %r9, 0x38(%rsp)
        xorl %eax, %eax
        movq %rax, 0x40(%rsp)
        movq %rax, 0x48(%rsp)
        movq %rax, 0x50(%rsp)
        movq %rax, 0x58(%rsp)
        movabsq $0xa0f99e2375022099, %rax
        movq %rax, 0x60(%rsp)
        movabsq $0xa8c68f3f1d132595, %rax
        movq %rax, 0x68(%rsp)
        movabsq $0x6c6c893805ac5242, %rax
        movq %rax, 0x70(%rsp)
        movabsq $0x276508b241770615, %rax
        movq %rax, 0x78(%rsp)
        movq $0xa, 0x90(%rsp)
        movq $0x1, 0x98(%rsp)
        jmp Lcurve25519_x25519_alt_midloop
Lcurve25519_x25519_alt_inverseloop:
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %rdi
        andq %r11, %rdi
        addq %rax, %rdi
        movq %rdi, 0x80(%rsp)
        movq %r12, %rax
        andq %r13, %rax
        movq %r14, %rsi
        andq %r15, %rsi
        addq %rax, %rsi
        movq %rsi, 0x88(%rsp)
        xorl %ebx, %ebx
        movq (%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x20(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rdi
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq (%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x20(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        xorl %ecx, %ecx
        movq 0x8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x28(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, (%rsp)
        xorl %edi, %edi
        movq 0x8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rbp
        adcq %rdx, %rdi
        movq 0x28(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rdi
        shrdq $0x3b, %rbp, %rsi
        movq %rsi, 0x20(%rsp)
        xorl %esi, %esi
        movq 0x10(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rsi
        movq 0x30(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rsi
        shrdq $0x3b, %rcx, %rbx
        movq %rbx, 0x8(%rsp)
        xorl %ebx, %ebx
        movq 0x10(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x30(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rdi
        adcq %rdx, %rbx
        shrdq $0x3b, %rdi, %rbp
        movq %rbp, 0x28(%rsp)
        movq 0x18(%rsp), %rax
        xorq %r9, %rax
        movq %rax, %rbp
        sarq $0x3f, %rbp
        andq %r8, %rbp
        negq %rbp
        mulq %r8
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x38(%rsp), %rax
        xorq %r11, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r10, %rdx
        subq %rdx, %rbp
        mulq %r10
        addq %rax, %rsi
        adcq %rdx, %rbp
        shrdq $0x3b, %rsi, %rcx
        movq %rcx, 0x10(%rsp)
        shrdq $0x3b, %rbp, %rsi
        movq 0x18(%rsp), %rax
        movq %rsi, 0x18(%rsp)
        xorq %r13, %rax
        movq %rax, %rsi
        sarq $0x3f, %rsi
        andq %r12, %rsi
        negq %rsi
        mulq %r12
        addq %rax, %rbx
        adcq %rdx, %rsi
        movq 0x38(%rsp), %rax
        xorq %r15, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r14, %rdx
        subq %rdx, %rsi
        mulq %r14
        addq %rax, %rbx
        adcq %rdx, %rsi
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, 0x30(%rsp)
        shrdq $0x3b, %rsi, %rbx
        movq %rbx, 0x38(%rsp)
        movq 0x80(%rsp), %rbx
        movq 0x88(%rsp), %rbp
        xorl %ecx, %ecx
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x40(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x40(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x60(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x60(%rsp)
        xorl %ebx, %ebx
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq 0x48(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rcx, 0x48(%rsp)
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x68(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq %rsi, 0x68(%rsp)
        xorl %ecx, %ecx
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x50(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x50(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x70(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x70(%rsp)
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        movq %r9, %rbx
        andq %r8, %rbx
        negq %rbx
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbx
        mulq %r10
        addq %rax, %rcx
        adcq %rbx, %rdx
        movq %rdx, %rbx
        shldq $0x1, %rcx, %rdx
        sarq $0x3f, %rbx
        addq %rbx, %rdx
        movl $0x13, %eax
        imulq %rdx
        movq 0x40(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x40(%rsp)
        movq 0x48(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x48(%rsp)
        movq 0x50(%rsp), %r8
        adcq %rbx, %r8
        movq %r8, 0x50(%rsp)
        adcq %rbx, %rcx
        shlq $0x3f, %rax
        addq %rax, %rcx
        movq 0x58(%rsp), %rax
        movq %rcx, 0x58(%rsp)
        xorq %r13, %rax
        movq %r13, %rcx
        andq %r12, %rcx
        negq %rcx
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rcx
        movq 0x78(%rsp), %rax
        xorq %r15, %rax
        movq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rcx
        mulq %r14
        addq %rax, %rsi
        adcq %rcx, %rdx
        movq %rdx, %rcx
        shldq $0x1, %rsi, %rdx
        sarq $0x3f, %rcx
        movl $0x13, %eax
        addq %rcx, %rdx
        imulq %rdx
        movq 0x60(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x60(%rsp)
        movq 0x68(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x68(%rsp)
        movq 0x70(%rsp), %r8
        adcq %rcx, %r8
        movq %r8, 0x70(%rsp)
        adcq %rcx, %rsi
        shlq $0x3f, %rax
        addq %rax, %rsi
        movq %rsi, 0x78(%rsp)
Lcurve25519_x25519_alt_midloop:
        movq 0x98(%rsp), %rsi
        movq (%rsp), %rdx
        movq 0x20(%rsp), %rcx
        movq %rdx, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        xorl %ebp, %ebp
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %rdx
        leaq (%rcx,%rax), %rdi
        shlq $0x16, %rdx
        shlq $0x16, %rdi
        sarq $0x2b, %rdx
        sarq $0x2b, %rdi
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %rbx
        leaq (%rcx,%rax), %rcx
        sarq $0x2a, %rbx
        sarq $0x2a, %rcx
        movq %rdx, 0xa0(%rsp)
        movq %rbx, 0xa8(%rsp)
        movq %rdi, 0xb0(%rsp)
        movq %rcx, 0xb8(%rsp)
        movq (%rsp), %r12
        imulq %r12, %rdi
        imulq %rdx, %r12
        movq 0x20(%rsp), %r13
        imulq %r13, %rbx
        imulq %rcx, %r13
        addq %rbx, %r12
        addq %rdi, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r10
        shlq $0x16, %r8
        shlq $0x16, %r10
        sarq $0x2b, %r8
        sarq $0x2b, %r10
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r15
        leaq (%rcx,%rax), %r11
        sarq $0x2a, %r15
        sarq $0x2a, %r11
        movq %r13, %rbx
        movq %r12, %rcx
        imulq %r8, %r12
        imulq %r15, %rbx
        addq %rbx, %r12
        imulq %r11, %r13
        imulq %r10, %rcx
        addq %rcx, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq 0xa0(%rsp), %rax
        imulq %r8, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r15, %rdx
        imulq 0xa8(%rsp), %r8
        imulq 0xb8(%rsp), %r15
        addq %r8, %r15
        leaq (%rax,%rdx), %r9
        movq 0xa0(%rsp), %rax
        imulq %r10, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r11, %rdx
        imulq 0xa8(%rsp), %r10
        imulq 0xb8(%rsp), %r11
        addq %r10, %r11
        leaq (%rax,%rdx), %r13
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r12
        shlq $0x15, %r8
        shlq $0x15, %r12
        sarq $0x2b, %r8
        sarq $0x2b, %r12
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r10
        leaq (%rcx,%rax), %r14
        sarq $0x2b, %r10
        sarq $0x2b, %r14
        movq %r9, %rax
        imulq %r8, %rax
        movq %r13, %rdx
        imulq %r10, %rdx
        imulq %r15, %r8
        imulq %r11, %r10
        addq %r8, %r10
        leaq (%rax,%rdx), %r8
        movq %r9, %rax
        imulq %r12, %rax
        movq %r13, %rdx
        imulq %r14, %rdx
        imulq %r15, %r12
        imulq %r11, %r14
        addq %r12, %r14
        leaq (%rax,%rdx), %r12
        movq %rsi, 0x98(%rsp)
        decq 0x90(%rsp)
        jne Lcurve25519_x25519_alt_inverseloop
        movq (%rsp), %rax
        movq 0x20(%rsp), %rcx
        imulq %r8, %rax
        imulq %r10, %rcx
        addq %rcx, %rax
        sarq $0x3f, %rax
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        xorq %rax, %r9
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        xorq %rax, %r11
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        xorq %rax, %r13
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        xorq %rax, %r15
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %r12
        andq %r11, %r12
        addq %rax, %r12
        xorl %r13d, %r13d
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r12
        adcq %rdx, %r13
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r13
        adcq %rdx, %r14
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r13
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        andq %r8, %r9
        negq %r9
        mulq %r8
        addq %rax, %r15
        adcq %rdx, %r9
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %r9
        mulq %r10
        addq %rax, %r15
        adcq %rdx, %r9
        movq %r9, %rax
        shldq $0x1, %r15, %rax
        sarq $0x3f, %r9
        movl $0x13, %ebx
        leaq 0x1(%rax,%r9,1), %rax
        imulq %rbx
        xorl %ebp, %ebp
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r9, %r14
        adcq %r9, %r15
        shlq $0x3f, %rax
        addq %rax, %r15
        cmovns %rbp, %rbx
        subq %rbx, %r12
        sbbq %rbp, %r13
        sbbq %rbp, %r14
        sbbq %rbp, %r15
        btr $0x3f, %r15
        movq 0xc0(%rsp), %rdi
        movq %r12, (%rdi)
        movq %r13, 0x8(%rdi)
        movq %r14, 0x10(%rdi)
        movq %r15, 0x18(%rdi)
        movq 12*32(%rsp), %rbp
        movq (10*32)(%rsp), %rax ; mulq (7*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(7*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(10*32)(%rsp), %rax ; mulq (7*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(7*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x8+(7*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(10*32)(%rsp), %rax ; mulq (7*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(7*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(7*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x8+(7*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(10*32)(%rsp), %rax ; mulq (7*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(7*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x10+(7*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x8+(7*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(7*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x10+(7*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x18+(7*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; leaq 0x1(%r12), %rax ; movl $0x13, %esi ; bts $63, %r11 ; imulq %rsi, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; sbbq %rax, %rax ; notq %rax; andq %rsi, %rax ; subq %rax, %r8 ; sbbq %rcx, %r9 ; sbbq %rcx, %r10 ; sbbq %rcx, %r11 ; btr $63, %r11 ; movq %r8, 0(%rbp) ; movq %r9, 0x8+0(%rbp) ; movq %r10, 0x10+0(%rbp) ; movq %r11, 0x18+0(%rbp)
        addq $13*32, %rsp ; .cfi_adjust_cfa_offset -13*32
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.if {elf}
.size rscrypto_curve25519_x25519_alt, .-rscrypto_curve25519_x25519_alt
.size rscrypto_curve25519_x25519_byte_alt, .-rscrypto_curve25519_x25519_byte_alt
.endif

 .globl rscrypto_curve25519_x25519base_alt
        .if {elf}
        .type rscrypto_curve25519_x25519base_alt, %function
        .hidden rscrypto_curve25519_x25519base_alt
        .endif
        .globl rscrypto_curve25519_x25519base_byte_alt
        .if {elf}
        .type rscrypto_curve25519_x25519base_byte_alt, %function
        .hidden rscrypto_curve25519_x25519base_byte_alt
        .endif
        .text
rscrypto_curve25519_x25519base_alt:
rscrypto_curve25519_x25519base_byte_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $488, %rsp ; .cfi_adjust_cfa_offset 488
        movq %rdi, 14*32(%rsp)
        movq (%rsi), %rax
        movq %rax, (%rsp)
        movq 8(%rsi), %rax
        movq %rax, 8(%rsp)
        movq 16(%rsi), %rax
        movq %rax, 16(%rsp)
        movq $0x3fffffffffffffff, %rax
        andq 24(%rsi), %rax
        movq %rax, 24(%rsp)
        movq (%rsp), %rax
        andq $8, %rax
        leaq Lcurve25519_x25519base_alt_edwards25519_0g(%rip), %r10
        leaq Lcurve25519_x25519base_alt_edwards25519_8g(%rip), %r11
        movq (%r10), %rax
        movq (%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*16(%rsp)
        movq 8*1(%r10), %rax
        movq 8*1(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*17(%rsp)
        movq 8*2(%r10), %rax
        movq 8*2(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*18(%rsp)
        movq 8*3(%r10), %rax
        movq 8*3(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*19(%rsp)
        movq 8*4(%r10), %rax
        movq 8*4(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*20(%rsp)
        movq 8*5(%r10), %rax
        movq 8*5(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*21(%rsp)
        movq 8*6(%r10), %rax
        movq 8*6(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*22(%rsp)
        movq 8*7(%r10), %rax
        movq 8*7(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*23(%rsp)
        movl $1, %eax
        movq %rax, 8*24(%rsp)
        movl $0, %eax
        movq %rax, 8*25(%rsp)
        movq %rax, 8*26(%rsp)
        movq %rax, 8*27(%rsp)
        movq 8*8(%r10), %rax
        movq 8*8(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*28(%rsp)
        movq 8*9(%r10), %rax
        movq 8*9(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*29(%rsp)
        movq 8*10(%r10), %rax
        movq 8*10(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*30(%rsp)
        movq 8*11(%r10), %rax
        movq 8*11(%r11), %rcx
        cmovnzq %rcx, %rax
        movq %rax, 8*31(%rsp)
        movq $4, 14*32 +8(%rsp)
        leaq Lcurve25519_x25519base_alt_edwards25519_gtable(%rip), %rax
        movq %rax, 15*32(%rsp)
        movq $0, 14*32 +16(%rsp)
Lcurve25519_x25519base_alt_scalarloop:
        movq 14*32 +8(%rsp), %rax
        movq %rax, %rcx
        shrq $6, %rax
        movq (%rsp,%rax,8), %rax
        shrq %cl, %rax
        andq $15, %rax
        addq 14*32 +16(%rsp), %rax
        movq %rax, 14*32 +24(%rsp)
        cmpq $9, 14*32 +24(%rsp)
        sbbq %rax, %rax
        incq %rax
        movq %rax, 14*32 +16(%rsp)
        movq $16, %rdi
        subq 14*32 +24(%rsp), %rdi
        cmpq $0, 14*32 +16(%rsp)
        cmovzq 14*32 +24(%rsp), %rdi
        movq %rdi, 14*32 +24(%rsp)
        movl $1, %eax
        xorl %ebx, %ebx
        xorl %ecx, %ecx
        xorl %edx, %edx
        movl $1, %r8d
        xorl %r9d, %r9d
        xorl %r10d, %r10d
        xorl %r11d, %r11d
        xorl %r12d, %r12d
        xorl %r13d, %r13d
        xorl %r14d, %r14d
        xorl %r15d, %r15d
        movq 15*32(%rsp), %rbp
        cmpq $1, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $2, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $3, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $4, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $5, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $6, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $7, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        cmpq $8, 14*32 +24(%rsp)
        movq (%rbp), %rsi
        cmovzq %rsi, %rax
        movq 8(%rbp), %rsi
        cmovzq %rsi, %rbx
        movq 16(%rbp), %rsi
        cmovzq %rsi, %rcx
        movq 24(%rbp), %rsi
        cmovzq %rsi, %rdx
        movq 32(%rbp), %rsi
        cmovzq %rsi, %r8
        movq 40(%rbp), %rsi
        cmovzq %rsi, %r9
        movq 48(%rbp), %rsi
        cmovzq %rsi, %r10
        movq 56(%rbp), %rsi
        cmovzq %rsi, %r11
        movq 64(%rbp), %rsi
        cmovzq %rsi, %r12
        movq 72(%rbp), %rsi
        cmovzq %rsi, %r13
        movq 80(%rbp), %rsi
        cmovzq %rsi, %r14
        movq 88(%rbp), %rsi
        cmovzq %rsi, %r15
        addq $96, %rbp
        movq %rbp, 15*32(%rsp)
        cmpq $0, 14*32 +16(%rsp)
        movq %rax, %rsi
        cmovnzq %r8, %rsi
        cmovnzq %rax, %r8
        movq %rsi, 32(%rsp)
        movq %r8, 64(%rsp)
        movq %rbx, %rsi
        cmovnzq %r9, %rsi
        cmovnzq %rbx, %r9
        movq %rsi, 40(%rsp)
        movq %r9, 72(%rsp)
        movq %rcx, %rsi
        cmovnzq %r10, %rsi
        cmovnzq %rcx, %r10
        movq %rsi, 48(%rsp)
        movq %r10, 80(%rsp)
        movq %rdx, %rsi
        cmovnzq %r11, %rsi
        cmovnzq %rdx, %r11
        movq %rsi, 56(%rsp)
        movq %r11, 88(%rsp)
        movq $-19, %rax
        movq $-1, %rbx
        movq $-1, %rcx
        movq $0x7fffffffffffffff, %rdx
        subq %r12, %rax
        sbbq %r13, %rbx
        sbbq %r14, %rcx
        sbbq %r15, %rdx
        movq 14*32 +24(%rsp), %r8
        movq 14*32 +16(%rsp), %r9
        testq %r8, %r8
        cmovzq %r8, %r9
        testq %r9, %r9
        cmovzq %r12, %rax
        cmovzq %r13, %rbx
        cmovzq %r14, %rcx
        cmovzq %r15, %rdx
        movq %rax, 96(%rsp)
        movq %rbx, 104(%rsp)
        movq %rcx, 112(%rsp)
        movq %rdx, 120(%rsp)
        movq (6*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq %r8, %r8 ; movq 0x8+(6*32)(%rsp), %r9 ; adcq %r9, %r9 ; movq 0x10+(6*32)(%rsp), %r10 ; adcq %r10, %r10 ; movq 0x18+(6*32)(%rsp), %r11 ; adcq %r11, %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq (5*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (4*32)(%rsp), %r8 ; movq 8+(5*32)(%rsp), %r9 ; sbbq 8+(4*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(5*32)(%rsp), %r10 ; sbbq 16+(4*32)(%rsp), %r10 ; movq 24+(5*32)(%rsp), %rax ; sbbq 24+(4*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (9*32)(%rsp) ; movq %r9, 8+(9*32)(%rsp) ; movq %r10, 16+(9*32)(%rsp) ; movq %rax, 24+(9*32)(%rsp)
        movq (5*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (4*32)(%rsp), %r8 ; movq 0x8+(5*32)(%rsp), %r9 ; adcq 0x8+(4*32)(%rsp), %r9 ; movq 0x10+(5*32)(%rsp), %r10 ; adcq 0x10+(4*32)(%rsp), %r10 ; movq 0x18+(5*32)(%rsp), %r11 ; adcq 0x18+(4*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (7*32)(%rsp), %rax ; mulq (3*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (7*32)(%rsp), %rax ; mulq 0x8+(3*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(7*32)(%rsp), %rax ; mulq (3*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (7*32)(%rsp), %rax ; mulq 0x10+(3*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(7*32)(%rsp), %rax ; mulq 0x8+(3*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(7*32)(%rsp), %rax ; mulq (3*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (7*32)(%rsp), %rax ; mulq 0x18+(3*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(7*32)(%rsp), %rax ; mulq 0x10+(3*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(7*32)(%rsp), %rax ; mulq 0x8+(3*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(7*32)(%rsp), %rax ; mulq (3*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(7*32)(%rsp), %rax ; mulq 0x18+(3*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(7*32)(%rsp), %rax ; mulq 0x10+(3*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(7*32)(%rsp), %rax ; mulq 0x8+(3*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(7*32)(%rsp), %rax ; mulq 0x18+(3*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(7*32)(%rsp), %rax ; mulq 0x10+(3*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(7*32)(%rsp), %rax ; mulq 0x18+(3*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %ebx ; movq %r12, %rax ; mulq %rbx; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rbx; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (11*32)(%rsp) ; movq %r9, 0x8+(11*32)(%rsp) ; movq %r10, 0x10+(11*32)(%rsp) ; movq %r11, 0x18+(11*32)(%rsp)
        movq (9*32)(%rsp), %rax ; mulq (1*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (9*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(9*32)(%rsp), %rax ; mulq (1*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (9*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(9*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(9*32)(%rsp), %rax ; mulq (1*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (9*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(9*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(9*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(9*32)(%rsp), %rax ; mulq (1*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(9*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(9*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(9*32)(%rsp), %rax ; mulq 0x8+(1*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(9*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(9*32)(%rsp), %rax ; mulq 0x10+(1*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(9*32)(%rsp), %rax ; mulq 0x18+(1*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %ebx ; movq %r12, %rax ; mulq %rbx; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rbx; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (9*32)(%rsp) ; movq %r9, 0x8+(9*32)(%rsp) ; movq %r10, 0x10+(9*32)(%rsp) ; movq %r11, 0x18+(9*32)(%rsp)
        movq (10*32)(%rsp), %rax ; mulq (2*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (10*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(10*32)(%rsp), %rax ; mulq (2*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (10*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(10*32)(%rsp), %rax ; mulq (2*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (10*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(10*32)(%rsp), %rax ; mulq (2*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(10*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x8+(2*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(10*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x10+(2*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(10*32)(%rsp), %rax ; mulq 0x18+(2*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %ebx ; movq %r12, %rax ; mulq %rbx; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rbx; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (10*32)(%rsp) ; movq %r9, 0x8+(10*32)(%rsp) ; movq %r10, 0x10+(10*32)(%rsp) ; movq %r11, 0x18+(10*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (11*32)(%rsp), %r8 ; movq 8+(8*32)(%rsp), %r9 ; sbbq 8+(11*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(8*32)(%rsp), %r10 ; sbbq 16+(11*32)(%rsp), %r10 ; movq 24+(8*32)(%rsp), %rax ; sbbq 24+(11*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (12*32)(%rsp) ; movq %r9, 8+(12*32)(%rsp) ; movq %r10, 16+(12*32)(%rsp) ; movq %rax, 24+(12*32)(%rsp)
        movq (8*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (11*32)(%rsp), %r8 ; movq 0x8+(8*32)(%rsp), %r9 ; adcq 0x8+(11*32)(%rsp), %r9 ; movq 0x10+(8*32)(%rsp), %r10 ; adcq 0x10+(11*32)(%rsp), %r10 ; movq 0x18+(8*32)(%rsp), %r11 ; adcq 0x18+(11*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (8*32)(%rsp) ; movq %r9, 0x8+(8*32)(%rsp) ; movq %r10, 0x10+(8*32)(%rsp) ; movq %r11, 0x18+(8*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (9*32)(%rsp), %r8 ; movq 8+(10*32)(%rsp), %r9 ; sbbq 8+(9*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(10*32)(%rsp), %r10 ; sbbq 16+(9*32)(%rsp), %r10 ; movq 24+(10*32)(%rsp), %rax ; sbbq 24+(9*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (13*32)(%rsp) ; movq %r9, 8+(13*32)(%rsp) ; movq %r10, 16+(13*32)(%rsp) ; movq %rax, 24+(13*32)(%rsp)
        movq (10*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (9*32)(%rsp), %r8 ; movq 0x8+(10*32)(%rsp), %r9 ; adcq 0x8+(9*32)(%rsp), %r9 ; movq 0x10+(10*32)(%rsp), %r10 ; adcq 0x10+(9*32)(%rsp), %r10 ; movq 0x18+(10*32)(%rsp), %r11 ; adcq 0x18+(9*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (9*32)(%rsp) ; movq %r9, 0x8+(9*32)(%rsp) ; movq %r10, 0x10+(9*32)(%rsp) ; movq %r11, 0x18+(9*32)(%rsp)
        movq (12*32)(%rsp), %rax ; mulq (8*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (12*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(12*32)(%rsp), %rax ; mulq (8*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (12*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(12*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(12*32)(%rsp), %rax ; mulq (8*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (12*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(12*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(12*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(12*32)(%rsp), %rax ; mulq (8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(12*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(12*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(12*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(12*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(12*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(12*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %ebx ; movq %r12, %rax ; mulq %rbx; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rbx; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (6*32)(%rsp) ; movq %r9, 0x8+(6*32)(%rsp) ; movq %r10, 0x10+(6*32)(%rsp) ; movq %r11, 0x18+(6*32)(%rsp)
        movq (13*32)(%rsp), %rax ; mulq (12*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (13*32)(%rsp), %rax ; mulq 0x8+(12*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(13*32)(%rsp), %rax ; mulq (12*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (13*32)(%rsp), %rax ; mulq 0x10+(12*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(13*32)(%rsp), %rax ; mulq 0x8+(12*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(13*32)(%rsp), %rax ; mulq (12*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (13*32)(%rsp), %rax ; mulq 0x18+(12*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(13*32)(%rsp), %rax ; mulq 0x10+(12*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(13*32)(%rsp), %rax ; mulq 0x8+(12*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(13*32)(%rsp), %rax ; mulq (12*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(13*32)(%rsp), %rax ; mulq 0x18+(12*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(13*32)(%rsp), %rax ; mulq 0x10+(12*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(13*32)(%rsp), %rax ; mulq 0x8+(12*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(13*32)(%rsp), %rax ; mulq 0x18+(12*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(13*32)(%rsp), %rax ; mulq 0x10+(12*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(13*32)(%rsp), %rax ; mulq 0x18+(12*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %ebx ; movq %r12, %rax ; mulq %rbx; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rbx; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (4*32)(%rsp) ; movq %r9, 0x8+(4*32)(%rsp) ; movq %r10, 0x10+(4*32)(%rsp) ; movq %r11, 0x18+(4*32)(%rsp)
        movq (8*32)(%rsp), %rax ; mulq (9*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (8*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(8*32)(%rsp), %rax ; mulq (9*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (8*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(8*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(8*32)(%rsp), %rax ; mulq (9*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (8*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(8*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(8*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(8*32)(%rsp), %rax ; mulq (9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(8*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(8*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(8*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(8*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(8*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(8*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %ebx ; movq %r12, %rax ; mulq %rbx; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rbx; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (5*32)(%rsp) ; movq %r9, 0x8+(5*32)(%rsp) ; movq %r10, 0x10+(5*32)(%rsp) ; movq %r11, 0x18+(5*32)(%rsp)
        movq (13*32)(%rsp), %rax ; mulq (9*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (13*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(13*32)(%rsp), %rax ; mulq (9*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (13*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(13*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(13*32)(%rsp), %rax ; mulq (9*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (13*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(13*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(13*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(13*32)(%rsp), %rax ; mulq (9*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(13*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(13*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(13*32)(%rsp), %rax ; mulq 0x8+(9*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(13*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(13*32)(%rsp), %rax ; mulq 0x10+(9*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(13*32)(%rsp), %rax ; mulq 0x18+(9*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %ebx ; movq %r12, %rax ; mulq %rbx; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rbx; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; btr $0x3f, %r11 ; movl $0x13, %edx ; imulq %r12, %rdx ; addq %rdx, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (7*32)(%rsp) ; movq %r9, 0x8+(7*32)(%rsp) ; movq %r10, 0x10+(7*32)(%rsp) ; movq %r11, 0x18+(7*32)(%rsp)
        addq $4, 14*32 +8(%rsp)
        cmpq $256, 14*32 +8(%rsp)
        jc Lcurve25519_x25519base_alt_scalarloop
        movq (4*32)(%rsp), %r8 ; xorl %ecx, %ecx ; addq (7*32)(%rsp), %r8 ; movq 0x8+(4*32)(%rsp), %r9 ; adcq 0x8+(7*32)(%rsp), %r9 ; movq 0x10+(4*32)(%rsp), %r10 ; adcq 0x10+(7*32)(%rsp), %r10 ; movq 0x18+(4*32)(%rsp), %r11 ; adcq 0x18+(7*32)(%rsp), %r11 ; movl $38, %eax ; cmovncq %rcx, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; movq %r8, (9*32)(%rsp) ; movq %r9, 0x8+(9*32)(%rsp) ; movq %r10, 0x10+(9*32)(%rsp) ; movq %r11, 0x18+(9*32)(%rsp)
        movq (4*32)(%rsp), %r8 ; xorl %ebx, %ebx ; subq (7*32)(%rsp), %r8 ; movq 8+(4*32)(%rsp), %r9 ; sbbq 8+(7*32)(%rsp), %r9 ; movl $38, %ecx ; movq 16+(4*32)(%rsp), %r10 ; sbbq 16+(7*32)(%rsp), %r10 ; movq 24+(4*32)(%rsp), %rax ; sbbq 24+(7*32)(%rsp), %rax ; cmovncq %rbx, %rcx ; subq %rcx, %r8 ; sbbq %rbx, %r9 ; sbbq %rbx, %r10 ; sbbq %rbx, %rax ; movq %r8, (10*32)(%rsp) ; movq %r9, 8+(10*32)(%rsp) ; movq %r10, 16+(10*32)(%rsp) ; movq %rax, 24+(10*32)(%rsp)
        leaq 256(%rsp), %rdi
        leaq 320(%rsp), %rsi
        movq %rdi, 0xc0(%rsp)
        xorl %eax, %eax
        leaq -0x13(%rax), %rcx
        notq %rax
        movq %rcx, (%rsp)
        movq %rax, 0x8(%rsp)
        movq %rax, 0x10(%rsp)
        btr $0x3f, %rax
        movq %rax, 0x18(%rsp)
        movq (%rsi), %rdx
        movq 0x8(%rsi), %rcx
        movq 0x10(%rsi), %r8
        movq 0x18(%rsi), %r9
        movl $0x1, %eax
        xorl %r10d, %r10d
        bts $0x3f, %r9
        adcq %r10, %rax
        imulq $0x13, %rax, %rax
        addq %rax, %rdx
        adcq %r10, %rcx
        adcq %r10, %r8
        adcq %r10, %r9
        movl $0x13, %eax
        cmovbq %r10, %rax
        subq %rax, %rdx
        sbbq %r10, %rcx
        sbbq %r10, %r8
        sbbq %r10, %r9
        btr $0x3f, %r9
        movq %rdx, 0x20(%rsp)
        movq %rcx, 0x28(%rsp)
        movq %r8, 0x30(%rsp)
        movq %r9, 0x38(%rsp)
        xorl %eax, %eax
        movq %rax, 0x40(%rsp)
        movq %rax, 0x48(%rsp)
        movq %rax, 0x50(%rsp)
        movq %rax, 0x58(%rsp)
        movabsq $0xa0f99e2375022099, %rax
        movq %rax, 0x60(%rsp)
        movabsq $0xa8c68f3f1d132595, %rax
        movq %rax, 0x68(%rsp)
        movabsq $0x6c6c893805ac5242, %rax
        movq %rax, 0x70(%rsp)
        movabsq $0x276508b241770615, %rax
        movq %rax, 0x78(%rsp)
        movq $0xa, 0x90(%rsp)
        movq $0x1, 0x98(%rsp)
        jmp Lcurve25519_x25519base_alt_midloop
Lcurve25519_x25519base_alt_inverseloop:
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %rdi
        andq %r11, %rdi
        addq %rax, %rdi
        movq %rdi, 0x80(%rsp)
        movq %r12, %rax
        andq %r13, %rax
        movq %r14, %rsi
        andq %r15, %rsi
        addq %rax, %rsi
        movq %rsi, 0x88(%rsp)
        xorl %ebx, %ebx
        movq (%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x20(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rdi
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq (%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x20(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        xorl %ecx, %ecx
        movq 0x8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x28(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, (%rsp)
        xorl %edi, %edi
        movq 0x8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rbp
        adcq %rdx, %rdi
        movq 0x28(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rdi
        shrdq $0x3b, %rbp, %rsi
        movq %rsi, 0x20(%rsp)
        xorl %esi, %esi
        movq 0x10(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rsi
        movq 0x30(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rsi
        shrdq $0x3b, %rcx, %rbx
        movq %rbx, 0x8(%rsp)
        xorl %ebx, %ebx
        movq 0x10(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x30(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rdi
        adcq %rdx, %rbx
        shrdq $0x3b, %rdi, %rbp
        movq %rbp, 0x28(%rsp)
        movq 0x18(%rsp), %rax
        xorq %r9, %rax
        movq %rax, %rbp
        sarq $0x3f, %rbp
        andq %r8, %rbp
        negq %rbp
        mulq %r8
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x38(%rsp), %rax
        xorq %r11, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r10, %rdx
        subq %rdx, %rbp
        mulq %r10
        addq %rax, %rsi
        adcq %rdx, %rbp
        shrdq $0x3b, %rsi, %rcx
        movq %rcx, 0x10(%rsp)
        shrdq $0x3b, %rbp, %rsi
        movq 0x18(%rsp), %rax
        movq %rsi, 0x18(%rsp)
        xorq %r13, %rax
        movq %rax, %rsi
        sarq $0x3f, %rsi
        andq %r12, %rsi
        negq %rsi
        mulq %r12
        addq %rax, %rbx
        adcq %rdx, %rsi
        movq 0x38(%rsp), %rax
        xorq %r15, %rax
        movq %rax, %rdx
        sarq $0x3f, %rdx
        andq %r14, %rdx
        subq %rdx, %rsi
        mulq %r14
        addq %rax, %rbx
        adcq %rdx, %rsi
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, 0x30(%rsp)
        shrdq $0x3b, %rsi, %rbx
        movq %rbx, 0x38(%rsp)
        movq 0x80(%rsp), %rbx
        movq 0x88(%rsp), %rbp
        xorl %ecx, %ecx
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x40(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x40(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x60(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x60(%rsp)
        xorl %ebx, %ebx
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq 0x48(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rcx, 0x48(%rsp)
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x68(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq %rsi, 0x68(%rsp)
        xorl %ecx, %ecx
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x50(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x50(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x70(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x70(%rsp)
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        movq %r9, %rbx
        andq %r8, %rbx
        negq %rbx
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbx
        mulq %r10
        addq %rax, %rcx
        adcq %rbx, %rdx
        movq %rdx, %rbx
        shldq $0x1, %rcx, %rdx
        sarq $0x3f, %rbx
        addq %rbx, %rdx
        movl $0x13, %eax
        imulq %rdx
        movq 0x40(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x40(%rsp)
        movq 0x48(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x48(%rsp)
        movq 0x50(%rsp), %r8
        adcq %rbx, %r8
        movq %r8, 0x50(%rsp)
        adcq %rbx, %rcx
        shlq $0x3f, %rax
        addq %rax, %rcx
        movq 0x58(%rsp), %rax
        movq %rcx, 0x58(%rsp)
        xorq %r13, %rax
        movq %r13, %rcx
        andq %r12, %rcx
        negq %rcx
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rcx
        movq 0x78(%rsp), %rax
        xorq %r15, %rax
        movq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rcx
        mulq %r14
        addq %rax, %rsi
        adcq %rcx, %rdx
        movq %rdx, %rcx
        shldq $0x1, %rsi, %rdx
        sarq $0x3f, %rcx
        movl $0x13, %eax
        addq %rcx, %rdx
        imulq %rdx
        movq 0x60(%rsp), %r8
        addq %rax, %r8
        movq %r8, 0x60(%rsp)
        movq 0x68(%rsp), %r8
        adcq %rdx, %r8
        movq %r8, 0x68(%rsp)
        movq 0x70(%rsp), %r8
        adcq %rcx, %r8
        movq %r8, 0x70(%rsp)
        adcq %rcx, %rsi
        shlq $0x3f, %rax
        addq %rax, %rsi
        movq %rsi, 0x78(%rsp)
Lcurve25519_x25519base_alt_midloop:
        movq 0x98(%rsp), %rsi
        movq (%rsp), %rdx
        movq 0x20(%rsp), %rcx
        movq %rdx, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        xorl %ebp, %ebp
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %rdx
        leaq (%rcx,%rax), %rdi
        shlq $0x16, %rdx
        shlq $0x16, %rdi
        sarq $0x2b, %rdx
        sarq $0x2b, %rdi
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %rbx
        leaq (%rcx,%rax), %rcx
        sarq $0x2a, %rbx
        sarq $0x2a, %rcx
        movq %rdx, 0xa0(%rsp)
        movq %rbx, 0xa8(%rsp)
        movq %rdi, 0xb0(%rsp)
        movq %rcx, 0xb8(%rsp)
        movq (%rsp), %r12
        imulq %r12, %rdi
        imulq %rdx, %r12
        movq 0x20(%rsp), %r13
        imulq %r13, %rbx
        imulq %rcx, %r13
        addq %rbx, %r12
        addq %rdi, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r10
        shlq $0x16, %r8
        shlq $0x16, %r10
        sarq $0x2b, %r8
        sarq $0x2b, %r10
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r15
        leaq (%rcx,%rax), %r11
        sarq $0x2a, %r15
        sarq $0x2a, %r11
        movq %r13, %rbx
        movq %r12, %rcx
        imulq %r8, %r12
        imulq %r15, %rbx
        addq %rbx, %r12
        imulq %r11, %r13
        imulq %r10, %rcx
        addq %rcx, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq 0xa0(%rsp), %rax
        imulq %r8, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r15, %rdx
        imulq 0xa8(%rsp), %r8
        imulq 0xb8(%rsp), %r15
        addq %r8, %r15
        leaq (%rax,%rdx), %r9
        movq 0xa0(%rsp), %rax
        imulq %r10, %rax
        movq 0xb0(%rsp), %rdx
        imulq %r11, %rdx
        imulq 0xa8(%rsp), %r10
        imulq 0xb8(%rsp), %r11
        addq %r10, %r11
        leaq (%rax,%rdx), %r13
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r12
        shlq $0x15, %r8
        shlq $0x15, %r12
        sarq $0x2b, %r8
        sarq $0x2b, %r12
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r10
        leaq (%rcx,%rax), %r14
        sarq $0x2b, %r10
        sarq $0x2b, %r14
        movq %r9, %rax
        imulq %r8, %rax
        movq %r13, %rdx
        imulq %r10, %rdx
        imulq %r15, %r8
        imulq %r11, %r10
        addq %r8, %r10
        leaq (%rax,%rdx), %r8
        movq %r9, %rax
        imulq %r12, %rax
        movq %r13, %rdx
        imulq %r14, %rdx
        imulq %r15, %r12
        imulq %r11, %r14
        addq %r12, %r14
        leaq (%rax,%rdx), %r12
        movq %rsi, 0x98(%rsp)
        decq 0x90(%rsp)
        jne Lcurve25519_x25519base_alt_inverseloop
        movq (%rsp), %rax
        movq 0x20(%rsp), %rcx
        imulq %r8, %rax
        imulq %r10, %rcx
        addq %rcx, %rax
        sarq $0x3f, %rax
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        xorq %rax, %r9
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        xorq %rax, %r11
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        xorq %rax, %r13
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        xorq %rax, %r15
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %r12
        andq %r11, %r12
        addq %rax, %r12
        xorl %r13d, %r13d
        movq 0x40(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r12
        adcq %rdx, %r13
        movq 0x60(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movq 0x48(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r13
        adcq %rdx, %r14
        movq 0x68(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r13
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x70(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        andq %r8, %r9
        negq %r9
        mulq %r8
        addq %rax, %r15
        adcq %rdx, %r9
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %r9
        mulq %r10
        addq %rax, %r15
        adcq %rdx, %r9
        movq %r9, %rax
        shldq $0x1, %r15, %rax
        sarq $0x3f, %r9
        movl $0x13, %ebx
        leaq 0x1(%rax,%r9,1), %rax
        imulq %rbx
        xorl %ebp, %ebp
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r9, %r14
        adcq %r9, %r15
        shlq $0x3f, %rax
        addq %rax, %r15
        cmovns %rbp, %rbx
        subq %rbx, %r12
        sbbq %rbp, %r13
        sbbq %rbp, %r14
        sbbq %rbp, %r15
        btr $0x3f, %r15
        movq 0xc0(%rsp), %rdi
        movq %r12, (%rdi)
        movq %r13, 0x8(%rdi)
        movq %r14, 0x10(%rdi)
        movq %r15, 0x18(%rdi)
        movq 14*32(%rsp), %rbp
        movq (9*32)(%rsp), %rax ; mulq (8*32)(%rsp); movq %rax, %r8 ; movq %rdx, %r9 ; xorq %r10, %r10 ; xorq %r11, %r11 ; movq (9*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; movq 0x8+(9*32)(%rsp), %rax ; mulq (8*32)(%rsp); addq %rax, %r9 ; adcq %rdx, %r10 ; adcq $0x0, %r11 ; xorq %r12, %r12 ; movq (9*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq %r12, %r12 ; movq 0x8+(9*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; movq 0x10+(9*32)(%rsp), %rax ; mulq (8*32)(%rsp); addq %rax, %r10 ; adcq %rdx, %r11 ; adcq $0x0, %r12 ; xorq %r13, %r13 ; movq (9*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq %r13, %r13 ; movq 0x8+(9*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x10+(9*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; movq 0x18+(9*32)(%rsp), %rax ; mulq (8*32)(%rsp); addq %rax, %r11 ; adcq %rdx, %r12 ; adcq $0x0, %r13 ; xorq %r14, %r14 ; movq 0x8+(9*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq %r14, %r14 ; movq 0x10+(9*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; movq 0x18+(9*32)(%rsp), %rax ; mulq 0x8+(8*32)(%rsp); addq %rax, %r12 ; adcq %rdx, %r13 ; adcq $0x0, %r14 ; xorq %r15, %r15 ; movq 0x10+(9*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq %r15, %r15 ; movq 0x18+(9*32)(%rsp), %rax ; mulq 0x10+(8*32)(%rsp); addq %rax, %r13 ; adcq %rdx, %r14 ; adcq $0x0, %r15 ; movq 0x18+(9*32)(%rsp), %rax ; mulq 0x18+(8*32)(%rsp); addq %rax, %r14 ; adcq %rdx, %r15 ; movl $0x26, %esi ; movq %r12, %rax ; mulq %rsi; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %rcx, %rcx ; movq %r13, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx ; movq %r14, %rax ; mulq %rsi; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx ; movq %r15, %rax ; mulq %rsi; subq %rcx, %rdx ; xorq %rcx, %rcx ; addq %rax, %r11 ; movq %rdx, %r12 ; adcq %rcx, %r12 ; shldq $0x1, %r11, %r12 ; leaq 0x1(%r12), %rax ; movl $0x13, %esi ; bts $63, %r11 ; imulq %rsi, %rax ; addq %rax, %r8 ; adcq %rcx, %r9 ; adcq %rcx, %r10 ; adcq %rcx, %r11 ; sbbq %rax, %rax ; notq %rax; andq %rsi, %rax ; subq %rax, %r8 ; sbbq %rcx, %r9 ; sbbq %rcx, %r10 ; sbbq %rcx, %r11 ; btr $63, %r11 ; movq %r8, (0*32)(%rbp) ; movq %r9, 0x8+(0*32)(%rbp) ; movq %r10, 0x10+(0*32)(%rbp) ; movq %r11, 0x18+(0*32)(%rbp)
        addq $488, %rsp ; .cfi_adjust_cfa_offset -488
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.if {elf}
.size rscrypto_curve25519_x25519base_alt, .-rscrypto_curve25519_x25519base_alt
        .section .rodata.rscrypto_curve25519_x25519base_alt, "a", %progbits
.else
        .section .rdata, "dr"
.endif
// The tables are read-only data: in .text, a disassembler would read them as instructions.
        .p2align 3
Lcurve25519_x25519base_alt_edwards25519_0g:
        .quad 0x251037f7cf4e861d
        .quad 0x10ede0fb19fb128f
        .quad 0x96c033b175f5e2c8
        .quad 0x055f070d6c15fb0d
        .quad 0x7c52af2c97473e69
        .quad 0x022f82391bad8378
        .quad 0x9991e1b02adb476f
        .quad 0x511144a03a99b855
        .quad 0x5fafc3b88ff2e4ae
        .quad 0x855e4ff0de1230ff
        .quad 0x72e302a348492870
        .quad 0x1253c19e53dbe1bc
Lcurve25519_x25519base_alt_edwards25519_8g:
        .quad 0x331d086e0d9abcaa
        .quad 0x1e23c96d311a10c9
        .quad 0x96d0f95e58c13478
        .quad 0x2f72f7384fcfcc59
        .quad 0x39a6cd1cfd7d87c9
        .quad 0x9867a0abd8ae153a
        .quad 0xa49d2a5f35986745
        .quad 0x57012940cdfe82e1
        .quad 0x5046a6532ec5544a
        .quad 0x6d674004739ff6c9
        .quad 0x9bbaa44b234a70e3
        .quad 0x5e6d8901138cf386
Lcurve25519_x25519base_alt_edwards25519_gtable:
        rscrypto_x25519_base_table

.if {elf}
.section .note.GNU-stack, "", %progbits
.endif
