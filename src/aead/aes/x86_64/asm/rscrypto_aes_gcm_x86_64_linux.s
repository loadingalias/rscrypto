.text

.macro GCM_COUNTER_BLOCK off, inc
  mov QWORD PTR [rsp + \off], r10
  mov DWORD PTR [rsp + \off + 8], r11d
  mov eax, r12d
  .if \inc
    add eax, \inc
  .endif
  bswap eax
  mov DWORD PTR [rsp + \off + 12], eax
.endm

.macro GCM_COUNTERS_16
  GCM_COUNTER_BLOCK 0, 0
  GCM_COUNTER_BLOCK 16, 1
  GCM_COUNTER_BLOCK 32, 2
  GCM_COUNTER_BLOCK 48, 3
  GCM_COUNTER_BLOCK 64, 4
  GCM_COUNTER_BLOCK 80, 5
  GCM_COUNTER_BLOCK 96, 6
  GCM_COUNTER_BLOCK 112, 7
  GCM_COUNTER_BLOCK 128, 8
  GCM_COUNTER_BLOCK 144, 9
  GCM_COUNTER_BLOCK 160, 10
  GCM_COUNTER_BLOCK 176, 11
  GCM_COUNTER_BLOCK 192, 12
  GCM_COUNTER_BLOCK 208, 13
  GCM_COUNTER_BLOCK 224, 14
  GCM_COUNTER_BLOCK 240, 15
.endm

.macro AES_ROUND off
  vbroadcasti32x4 zmm27, XMMWORD PTR [rdi + \off]
  vaesenc zmm0, zmm0, zmm27
  vaesenc zmm1, zmm1, zmm27
  vaesenc zmm2, zmm2, zmm27
  vaesenc zmm3, zmm3, zmm27
.endm

.macro AES_LAST off
  vbroadcasti32x4 zmm27, XMMWORD PTR [rdi + \off]
  vaesenclast zmm0, zmm0, zmm27
  vaesenclast zmm1, zmm1, zmm27
  vaesenclast zmm2, zmm2, zmm27
  vaesenclast zmm3, zmm3, zmm27
.endm

.macro AES_ENCRYPT_16 aes256
  vbroadcasti32x4 zmm27, XMMWORD PTR [rdi]
  vpxord zmm0, zmm0, zmm27
  vpxord zmm1, zmm1, zmm27
  vpxord zmm2, zmm2, zmm27
  vpxord zmm3, zmm3, zmm27

  AES_ROUND 16
  AES_ROUND 32
  AES_ROUND 48
  AES_ROUND 64
  AES_ROUND 80
  AES_ROUND 96
  AES_ROUND 112
  AES_ROUND 128
  AES_ROUND 144
  .if \aes256
    AES_ROUND 160
    AES_ROUND 176
    AES_ROUND 192
    AES_ROUND 208
    AES_LAST 224
  .else
    AES_LAST 160
  .endif
.endm

.macro GHASH_FOLD data, off
  vmovdqu64 zmm20, ZMMWORD PTR [r8 + \off]
  vpclmulqdq zmm21, \data, zmm20, 0x00
  vpclmulqdq zmm22, \data, zmm20, 0x11

  vpshufd zmm23, \data, 0x4e
  vpxord zmm23, zmm23, \data
  vpshufd zmm24, zmm20, 0x4e
  vpxord zmm24, zmm24, zmm20
  vpclmulqdq zmm23, zmm23, zmm24, 0x00
  vpxord zmm23, zmm23, zmm21
  vpxord zmm23, zmm23, zmm22

  vpslldq zmm24, zmm23, 8
  vpxord zmm21, zmm21, zmm24
  vpsrldq zmm24, zmm23, 8
  vpxord zmm22, zmm22, zmm24

  vpxord zmm28, zmm28, zmm21
  vpxord zmm29, zmm29, zmm22
.endm

.macro GHASH16
  vmovdqu64 zmm31, ZMMWORD PTR [rip + .Lrscrypto_x86_gcm_bswap]
  vpshufb zmm0, zmm0, zmm31
  vpshufb zmm1, zmm1, zmm31
  vpshufb zmm2, zmm2, zmm31
  vpshufb zmm3, zmm3, zmm31

  vpxord zmm30, zmm30, zmm30
  vmovdqu64 xmm30, XMMWORD PTR [r9]
  vpxord zmm0, zmm0, zmm30

  vpxord zmm28, zmm28, zmm28
  vpxord zmm29, zmm29, zmm29
  GHASH_FOLD zmm0, 0
  GHASH_FOLD zmm1, 64
  GHASH_FOLD zmm2, 128
  GHASH_FOLD zmm3, 192

  vextracti64x2 xmm16, zmm28, 0
  vextracti64x2 xmm17, zmm28, 1
  vpxorq xmm16, xmm16, xmm17
  vextracti64x2 xmm17, zmm28, 2
  vpxorq xmm16, xmm16, xmm17
  vextracti64x2 xmm17, zmm28, 3
  vpxorq xmm16, xmm16, xmm17

  vextracti64x2 xmm19, zmm29, 0
  vextracti64x2 xmm17, zmm29, 1
  vpxorq xmm19, xmm19, xmm17
  vextracti64x2 xmm17, zmm29, 2
  vpxorq xmm19, xmm19, xmm17
  vextracti64x2 xmm17, zmm29, 3
  vpxorq xmm19, xmm19, xmm17

  vpsllq xmm17, xmm16, 63
  vpsllq xmm18, xmm16, 62
  vpxorq xmm17, xmm17, xmm18
  vpsllq xmm18, xmm16, 57
  vpxorq xmm17, xmm17, xmm18
  vpslldq xmm17, xmm17, 8
  vpxorq xmm16, xmm16, xmm17

  vpsrlq xmm17, xmm16, 1
  vpxorq xmm17, xmm17, xmm16
  vpsrlq xmm18, xmm16, 2
  vpxorq xmm17, xmm17, xmm18
  vpsrlq xmm18, xmm16, 7
  vpxorq xmm17, xmm17, xmm18

  vpsllq xmm18, xmm16, 63
  vpsllq xmm20, xmm16, 62
  vpxorq xmm18, xmm18, xmm20
  vpsllq xmm20, xmm16, 57
  vpxorq xmm18, xmm18, xmm20
  vpsrldq xmm18, xmm18, 8

  vpxorq xmm17, xmm17, xmm18
  vpxorq xmm17, xmm17, xmm19
  vmovdqu64 XMMWORD PTR [r9], xmm17
.endm

.macro AES_GCM_16X_FUNC name, aes256, open
  .p2align 4
  .globl \name
  .type \name, @function
\name:
  push r12
  sub rsp, 256

  mov r10, QWORD PTR [rsi]
  mov r11d, DWORD PTR [rsi + 8]
  mov r12d, DWORD PTR [r9 + 16]

  cmp rcx, 256
  jb .L\name\()_done

.L\name\()_loop:
  GCM_COUNTERS_16
  vmovdqu64 zmm0, ZMMWORD PTR [rsp]
  vmovdqu64 zmm1, ZMMWORD PTR [rsp + 64]
  vmovdqu64 zmm2, ZMMWORD PTR [rsp + 128]
  vmovdqu64 zmm3, ZMMWORD PTR [rsp + 192]

  AES_ENCRYPT_16 \aes256

  .if \open
    vmovdqu64 zmm4, ZMMWORD PTR [rdx]
    vmovdqu64 zmm5, ZMMWORD PTR [rdx + 64]
    vmovdqu64 zmm6, ZMMWORD PTR [rdx + 128]
    vmovdqu64 zmm7, ZMMWORD PTR [rdx + 192]
    vpxord zmm8, zmm0, zmm4
    vpxord zmm9, zmm1, zmm5
    vpxord zmm10, zmm2, zmm6
    vpxord zmm11, zmm3, zmm7
    vmovdqa64 zmm0, zmm4
    vmovdqa64 zmm1, zmm5
    vmovdqa64 zmm2, zmm6
    vmovdqa64 zmm3, zmm7
  .else
    vmovdqu64 zmm4, ZMMWORD PTR [rdx]
    vmovdqu64 zmm5, ZMMWORD PTR [rdx + 64]
    vmovdqu64 zmm6, ZMMWORD PTR [rdx + 128]
    vmovdqu64 zmm7, ZMMWORD PTR [rdx + 192]
    vpxord zmm0, zmm0, zmm4
    vpxord zmm1, zmm1, zmm5
    vpxord zmm2, zmm2, zmm6
    vpxord zmm3, zmm3, zmm7
    vmovdqu64 ZMMWORD PTR [rdx], zmm0
    vmovdqu64 ZMMWORD PTR [rdx + 64], zmm1
    vmovdqu64 ZMMWORD PTR [rdx + 128], zmm2
    vmovdqu64 ZMMWORD PTR [rdx + 192], zmm3
  .endif

  GHASH16

  .if \open
    vmovdqu64 ZMMWORD PTR [rdx], zmm8
    vmovdqu64 ZMMWORD PTR [rdx + 64], zmm9
    vmovdqu64 ZMMWORD PTR [rdx + 128], zmm10
    vmovdqu64 ZMMWORD PTR [rdx + 192], zmm11
  .endif

  add rdx, 256
  sub rcx, 256
  add r12d, 16
  add QWORD PTR [r9 + 24], 256
  cmp rcx, 256
  jae .L\name\()_loop

.L\name\()_done:
  mov DWORD PTR [r9 + 16], r12d
  vzeroupper
  add rsp, 256
  pop r12
  ret
  .size \name, . - \name
.endm

AES_GCM_16X_FUNC rscrypto_aes128_gcm_seal_16x_vaes512_x86_64_linux, 0, 0
AES_GCM_16X_FUNC rscrypto_aes128_gcm_open_16x_vaes512_x86_64_linux, 0, 1
AES_GCM_16X_FUNC rscrypto_aes256_gcm_seal_16x_vaes512_x86_64_linux, 1, 0
AES_GCM_16X_FUNC rscrypto_aes256_gcm_open_16x_vaes512_x86_64_linux, 1, 1

.section .rodata.cst64,"aM",@progbits,64
.p2align 6
.Lrscrypto_x86_gcm_bswap:
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
