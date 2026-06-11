// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p384/bignum_mod_n384.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_mod_n384
        .type rscrypto_bignum_mod_n384, %function
        .hidden rscrypto_bignum_mod_n384
        .text
rscrypto_bignum_mod_n384:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        cmpq $6, %rsi
        jc Lrscrypto_bignum_mod_n384_shortinput
        subq $6, %rsi
        movq 40(%rdx,%rsi,8), %r13
        movq 32(%rdx,%rsi,8), %r12
        movq 24(%rdx,%rsi,8), %r11
        movq 16(%rdx,%rsi,8), %r10
        movq 8(%rdx,%rsi,8), %r9
        movq (%rdx,%rsi,8), %r8
        movq %rdx, %rcx
        movq $0x1313e695333ad68d, %rax
        movq $0xa7e5f24db74f5885, %rbx
        movq $0x389cb27e0bc8d220, %rdx
        addq %rax, %r8
        adcq %rbx, %r9
        adcq %rdx, %r10
        adcq $0, %r11
        adcq $0, %r12
        adcq $0, %r13
        sbbq %r14, %r14
        notq %r14
        andq %r14, %rax
        andq %r14, %rbx
        andq %r14, %rdx
        subq %rax, %r8
        sbbq %rbx, %r9
        sbbq %rdx, %r10
        sbbq $0, %r11
        sbbq $0, %r12
        sbbq $0, %r13
        testq %rsi, %rsi
        jz Lrscrypto_bignum_mod_n384_writeback
Lrscrypto_bignum_mod_n384_loop:
        movl $1, %edx
        addq %r13, %rdx
        sbbq %r14, %r14
        orq %r14, %rdx
        movq -8(%rcx,%rsi,8), %r14
        subq %rdx, %r13
        xorq %rax, %rax
        movq $0x1313e695333ad68d, %rax
        mulxq %rax, %rax, %rbx
        adcxq %rax, %r14
        adoxq %rbx, %r8
        movq $0xa7e5f24db74f5885, %rax
        mulxq %rax, %rax, %rbx
        adcxq %rax, %r8
        adoxq %rbx, %r9
        movq $0x389cb27e0bc8d220, %rax
        mulxq %rax, %rax, %rbx
        adcxq %rax, %r9
        movl $0, %eax
        adoxq %rax, %rbx
        adcxq %rbx, %r10
        adcq $0, %r11
        adcq $0, %r12
        adcq $0, %r13
        movq $0x1313e695333ad68d, %rax
        andq %r13, %rax
        movq $0xa7e5f24db74f5885, %rbx
        andq %r13, %rbx
        movq $0x389cb27e0bc8d220, %rdx
        andq %r13, %rdx
        subq %rax, %r14
        sbbq %rbx, %r8
        sbbq %rdx, %r9
        sbbq $0, %r10
        sbbq $0, %r11
        sbbq $0, %r12
        movq %r12, %r13
        movq %r11, %r12
        movq %r10, %r11
        movq %r9, %r10
        movq %r8, %r9
        movq %r14, %r8
        decq %rsi
        jnz Lrscrypto_bignum_mod_n384_loop
Lrscrypto_bignum_mod_n384_writeback:
        movq %r8, (%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 16(%rdi)
        movq %r11, 24(%rdi)
        movq %r12, 32(%rdi)
        movq %r13, 40(%rdi)
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
Lrscrypto_bignum_mod_n384_shortinput:
        xorq %r8, %r8
        xorq %r9, %r9
        xorq %r10, %r10
        xorq %r11, %r11
        xorq %r12, %r12
        xorq %r13, %r13
        testq %rsi, %rsi
        jz Lrscrypto_bignum_mod_n384_writeback
        movq (%rdx), %r8
        decq %rsi
        jz Lrscrypto_bignum_mod_n384_writeback
        movq 8(%rdx), %r9
        decq %rsi
        jz Lrscrypto_bignum_mod_n384_writeback
        movq 16(%rdx), %r10
        decq %rsi
        jz Lrscrypto_bignum_mod_n384_writeback
        movq 24(%rdx), %r11
        decq %rsi
        jz Lrscrypto_bignum_mod_n384_writeback
        movq 32(%rdx), %r12
        jmp Lrscrypto_bignum_mod_n384_writeback
.size rscrypto_bignum_mod_n384, .-rscrypto_bignum_mod_n384
