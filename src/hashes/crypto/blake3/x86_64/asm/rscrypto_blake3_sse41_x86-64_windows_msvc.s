// BLAKE3 SSE4.1 compress_in_place — Windows (SysV64 calling convention)
// We call this with sysv64 ABI from Rust to avoid Windows x64 callee-saved XMM regs.
// Same register mapping as Linux/macOS (rdi, rsi, rdx, rcx, r8).

.globl rscrypto_blake3_compress_in_place_sse41
.text
.p2align 6
rscrypto_blake3_compress_in_place_sse41:
        // Load chaining value
        movdqu  xmm0, xmmword ptr [rdi]
        movdqu  xmm1, xmmword ptr [rdi+0x10]
        // Build row3 = [counter_lo, counter_hi, block_len, flags]
        movzx   eax, r8b
        movzx   ecx, cl
        shl     rax, 32
        add     rcx, rax
        movq    xmm3, rdx
        movq    xmm4, rcx
        punpcklqdq xmm3, xmm4
        // Load IV
        movaps  xmm2, xmmword ptr [RSCRYPTO_SSE41_BLAKE3_IV+rip]
        // Load and pre-shuffle message block
        movups  xmm8, xmmword ptr [rsi]
        movups  xmm9, xmmword ptr [rsi+0x10]
        movaps  xmm4, xmm8
        shufps  xmm4, xmm9, 136
        movaps  xmm5, xmm8
        shufps  xmm5, xmm9, 221
        movups  xmm8, xmmword ptr [rsi+0x20]
        movups  xmm9, xmmword ptr [rsi+0x30]
        movaps  xmm6, xmm8
        shufps  xmm6, xmm9, 136
        movaps  xmm7, xmm8
        shufps  xmm7, xmm9, 221
        pshufd  xmm6, xmm6, 0x93
        pshufd  xmm7, xmm7, 0x93
        // Load rotation masks
        movdqa  xmm14, xmmword ptr [ROT16_SSE41+rip]
        movdqa  xmm15, xmmword ptr [ROT8_SSE41+rip]
        mov     al, 7
.p2align 5
9:
        // Column step
        paddd   xmm0, xmm4
        paddd   xmm0, xmm1
        pxor    xmm3, xmm0
        pshufb  xmm3, xmm14
        paddd   xmm2, xmm3
        pxor    xmm1, xmm2
        movdqa  xmm10, xmm1
        psrld   xmm1, 12
        pslld   xmm10, 20
        por     xmm1, xmm10
        paddd   xmm0, xmm5
        paddd   xmm0, xmm1
        pxor    xmm3, xmm0
        pshufb  xmm3, xmm15
        paddd   xmm2, xmm3
        pxor    xmm1, xmm2
        movdqa  xmm10, xmm1
        psrld   xmm1, 7
        pslld   xmm10, 25
        por     xmm1, xmm10
        // Diagonalize
        pshufd  xmm0, xmm0, 0x93
        pshufd  xmm3, xmm3, 0x4E
        pshufd  xmm2, xmm2, 0x39
        // Diagonal step
        paddd   xmm0, xmm6
        paddd   xmm0, xmm1
        pxor    xmm3, xmm0
        pshufb  xmm3, xmm14
        paddd   xmm2, xmm3
        pxor    xmm1, xmm2
        movdqa  xmm10, xmm1
        psrld   xmm1, 12
        pslld   xmm10, 20
        por     xmm1, xmm10
        paddd   xmm0, xmm7
        paddd   xmm0, xmm1
        pxor    xmm3, xmm0
        pshufb  xmm3, xmm15
        paddd   xmm2, xmm3
        pxor    xmm1, xmm2
        movdqa  xmm10, xmm1
        psrld   xmm1, 7
        pslld   xmm10, 25
        por     xmm1, xmm10
        // Un-diagonalize
        pshufd  xmm0, xmm0, 0x39
        pshufd  xmm3, xmm3, 0x4E
        pshufd  xmm2, xmm2, 0x93
        dec     al
        jz      9f
        // Message schedule permutation
        movaps  xmm8, xmm4
        shufps  xmm8, xmm5, 214
        pshufd  xmm9, xmm4, 0x0F
        pshufd  xmm4, xmm8, 0x39
        movaps  xmm8, xmm6
        shufps  xmm8, xmm7, 250
        pblendw xmm9, xmm8, 0xCC
        movdqa  xmm8, xmm7
        punpcklqdq xmm8, xmm5
        pblendw xmm8, xmm6, 0xC0
        pshufd  xmm8, xmm8, 0x78
        movdqa  xmm5, xmm5
        punpckhdq xmm5, xmm7
        punpckldq xmm6, xmm5
        pshufd  xmm7, xmm6, 0x1E
        movdqa  xmm5, xmm9
        movdqa  xmm6, xmm8
        jmp     9b
9:
        pxor    xmm0, xmm2
        pxor    xmm1, xmm3
        movdqu  xmmword ptr [rdi], xmm0
        movdqu  xmmword ptr [rdi+0x10], xmm1
        ret

.section .rdata,"dr"
.p2align 4
ROT16_SSE41:
        .byte  2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13
ROT8_SSE41:
        .byte  1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12
RSCRYPTO_SSE41_BLAKE3_IV:
        .long  0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A
