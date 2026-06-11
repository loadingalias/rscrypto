// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_mod_n256.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_mod_n256
        .type rscrypto_bignum_mod_n256, %function
        .hidden rscrypto_bignum_mod_n256
        .text
rscrypto_bignum_mod_n256:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        cmpq $4, %rsi
        jc Lrscrypto_bignum_mod_n256_shortinput
        subq $4, %rsi
        movq 24(%rdx,%rsi,8), %r11
        movq 16(%rdx,%rsi,8), %r10
        movq 8(%rdx,%rsi,8), %r9
        movq (%rdx,%rsi,8), %r8
        movq %rdx, %rcx
        movq $0x0c46353d039cdaaf, %rax
        movq $0x4319055258e8617b, %rbx
        movl $0x00000000ffffffff, %edx
        addq %rax, %r8
        adcq %rbx, %r9
        adcq $0, %r10
        adcq %rdx, %r11
        sbbq %r12, %r12
        notq %r12
        andq %r12, %rax
        andq %r12, %rbx
        andq %r12, %rdx
        subq %rax, %r8
        sbbq %rbx, %r9
        sbbq $0, %r10
        sbbq %rdx, %r11
        testq %rsi, %rsi
        jz Lrscrypto_bignum_mod_n256_writeback
Lrscrypto_bignum_mod_n256_loop:
        movq %r11, %rax
        shldq $32, %r10, %rax
        movq %r11, %rdx
        shrq $32, %rdx
        xorq %rbx, %rbx
        subq $1, %rbx
        adcq %r10, %rax
        adcq %r11, %rdx
        sbbq %rax, %rax
        orq %rax, %rdx
        movq -8(%rcx,%rsi,8), %r12
        subq %rdx, %r11
        movq $0x0c46353d039cdaaf, %rax
        mulxq %rax, %rax, %rbx
        addq %rax, %r12
        adcq %rbx, %r8
        movq $0x4319055258e8617b, %rax
        mulxq %rax, %rax, %rbx
        adcq $0, %rbx
        addq %rax, %r8
        adcq %rbx, %r9
        movl $0x00000000ffffffff, %eax
        mulxq %rax, %rax, %rbx
        adcq %rax, %r10
        adcq %rbx, %r11
        movq $0x0c46353d039cdaaf, %rax
        andq %r11, %rax
        movq $0x4319055258e8617b, %rbx
        andq %r11, %rbx
        movl $0x00000000ffffffff, %edx
        andq %r11, %rdx
        subq %rax, %r12
        sbbq %rbx, %r8
        sbbq $0, %r9
        sbbq %rdx, %r10
        movq %r10, %r11
        movq %r9, %r10
        movq %r8, %r9
        movq %r12, %r8
        decq %rsi
        jnz Lrscrypto_bignum_mod_n256_loop
Lrscrypto_bignum_mod_n256_writeback:
        movq %r8, (%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 16(%rdi)
        movq %r11, 24(%rdi)
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
Lrscrypto_bignum_mod_n256_shortinput:
        xorq %r8, %r8
        xorq %r9, %r9
        xorq %r10, %r10
        xorq %r11, %r11
        testq %rsi, %rsi
        jz Lrscrypto_bignum_mod_n256_writeback
        movq (%rdx), %r8
        decq %rsi
        jz Lrscrypto_bignum_mod_n256_writeback
        movq 8(%rdx), %r9
        decq %rsi
        jz Lrscrypto_bignum_mod_n256_writeback
        movq 16(%rdx), %r10
        jmp Lrscrypto_bignum_mod_n256_writeback
.size rscrypto_bignum_mod_n256, .-rscrypto_bignum_mod_n256
