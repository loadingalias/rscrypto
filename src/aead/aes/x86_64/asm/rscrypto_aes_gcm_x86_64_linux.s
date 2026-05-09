.text

.macro GCM_COUNTERS_16
  vmovd xmm25, r10d
  vpbroadcastd zmm25, xmm25

  vbroadcasti32x4 zmm0, XMMWORD PTR [rsi]
  vmovdqa64 zmm1, zmm0
  vmovdqa64 zmm2, zmm0
  vmovdqa64 zmm3, zmm0

  vmovdqa32 zmm24, zmm25
  vpaddd zmm24, zmm24, ZMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_z0]
  vpshufb zmm24, zmm24, zmm26
  vmovdqu32 zmm0{{k1}}, zmm24

  vmovdqa32 zmm24, zmm25
  vpaddd zmm24, zmm24, ZMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_z1]
  vpshufb zmm24, zmm24, zmm26
  vmovdqu32 zmm1{{k1}}, zmm24

  vmovdqa32 zmm24, zmm25
  vpaddd zmm24, zmm24, ZMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_z2]
  vpshufb zmm24, zmm24, zmm26
  vmovdqu32 zmm2{{k1}}, zmm24

  vpaddd zmm25, zmm25, ZMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_z3]
  vpshufb zmm25, zmm25, zmm26
  vmovdqu32 zmm3{{k1}}, zmm25
.endm

.macro GCM_COUNTERS_8
  vmovd xmm25, r10d
  vpbroadcastd ymm25, xmm25

  vbroadcasti32x4 ymm0, XMMWORD PTR [rsi]
  vmovdqa64 ymm1, ymm0
  vmovdqa64 ymm2, ymm0
  vmovdqa64 ymm3, ymm0

  vmovdqa32 ymm24, ymm25
  vpaddd ymm24, ymm24, YMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_y0]
  vpshufb ymm24, ymm24, ymm26
  vmovdqu32 ymm0{{k1}}, ymm24

  vmovdqa32 ymm24, ymm25
  vpaddd ymm24, ymm24, YMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_y1]
  vpshufb ymm24, ymm24, ymm26
  vmovdqu32 ymm1{{k1}}, ymm24

  vmovdqa32 ymm24, ymm25
  vpaddd ymm24, ymm24, YMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_y2]
  vpshufb ymm24, ymm24, ymm26
  vmovdqu32 ymm2{{k1}}, ymm24

  vpaddd ymm25, ymm25, YMMWORD PTR [rip + .Lrscrypto_x86_ctr_inc_y3]
  vpshufb ymm25, ymm25, ymm26
  vmovdqu32 ymm3{{k1}}, ymm25
.endm

.macro AES_ROUND off
  vbroadcasti32x4 zmm27, XMMWORD PTR [rdi + \off]
  vaesenc zmm0, zmm0, zmm27
  vaesenc zmm1, zmm1, zmm27
  vaesenc zmm2, zmm2, zmm27
  vaesenc zmm3, zmm3, zmm27
.endm

.macro AES_START
  vbroadcasti32x4 zmm27, XMMWORD PTR [rdi]
  vpxord zmm0, zmm0, zmm27
  vpxord zmm1, zmm1, zmm27
  vpxord zmm2, zmm2, zmm27
  vpxord zmm3, zmm3, zmm27
.endm

.macro AES_LAST off
  vbroadcasti32x4 zmm27, XMMWORD PTR [rdi + \off]
  vaesenclast zmm0, zmm0, zmm27
  vaesenclast zmm1, zmm1, zmm27
  vaesenclast zmm2, zmm2, zmm27
  vaesenclast zmm3, zmm3, zmm27
.endm

.macro AES_ENCRYPT_16 aes256
  AES_START
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

.macro AES_Y_ROUND off
  vbroadcasti32x4 ymm27, XMMWORD PTR [rdi + \off]
  vaesenc ymm0, ymm0, ymm27
  vaesenc ymm1, ymm1, ymm27
  vaesenc ymm2, ymm2, ymm27
  vaesenc ymm3, ymm3, ymm27
.endm

.macro AES_Y_START
  vbroadcasti32x4 ymm27, XMMWORD PTR [rdi]
  vpxord ymm0, ymm0, ymm27
  vpxord ymm1, ymm1, ymm27
  vpxord ymm2, ymm2, ymm27
  vpxord ymm3, ymm3, ymm27
.endm

.macro AES_Y_LAST off
  vbroadcasti32x4 ymm27, XMMWORD PTR [rdi + \off]
  vaesenclast ymm0, ymm0, ymm27
  vaesenclast ymm1, ymm1, ymm27
  vaesenclast ymm2, ymm2, ymm27
  vaesenclast ymm3, ymm3, ymm27
.endm

.macro AES_ENCRYPT_8Y aes256
  AES_Y_START
  AES_Y_ROUND 16
  AES_Y_ROUND 32
  AES_Y_ROUND 48
  AES_Y_ROUND 64
  AES_Y_ROUND 80
  AES_Y_ROUND 96
  AES_Y_ROUND 112
  AES_Y_ROUND 128
  AES_Y_ROUND 144
  .if \aes256
    AES_Y_ROUND 160
    AES_Y_ROUND 176
    AES_Y_ROUND 192
    AES_Y_ROUND 208
    AES_Y_LAST 224
  .else
    AES_Y_LAST 160
  .endif
.endm

.macro GHASH_FOLD data, h
  vpclmulqdq zmm21, \data, \h, 0x00
  vpclmulqdq zmm22, \data, \h, 0x11

  vpshufd zmm23, \data, 0x4e
  vpxord zmm23, zmm23, \data
  vpshufd zmm24, \h, 0x4e
  vpxord zmm24, zmm24, \h
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

.macro GHASH_FOLD_Y data, h
  vpclmulqdq ymm21, \data, \h, 0x00
  vpclmulqdq ymm22, \data, \h, 0x11

  vpshufd ymm23, \data, 0x4e
  vpxord ymm23, ymm23, \data
  vpshufd ymm24, \h, 0x4e
  vpxord ymm24, ymm24, \h
  vpclmulqdq ymm23, ymm23, ymm24, 0x00
  vpxord ymm23, ymm23, ymm21
  vpxord ymm23, ymm23, ymm22

  vpslldq ymm24, ymm23, 8
  vpxord ymm21, ymm21, ymm24
  vpsrldq ymm24, ymm23, 8
  vpxord ymm22, ymm22, ymm24

  vpxord ymm28, ymm28, ymm21
  vpxord ymm29, ymm29, ymm22
.endm

.macro GHASH_PREP d0, d1, d2, d3
  vpshufb \d0, \d0, zmm31
  vpshufb \d1, \d1, zmm31
  vpshufb \d2, \d2, zmm31
  vpshufb \d3, \d3, zmm31

  vpxord zmm30, zmm30, zmm30
  vmovdqu64 xmm30, XMMWORD PTR [r9]
  vpxord \d0, \d0, zmm30

  vpxord zmm28, zmm28, zmm28
  vpxord zmm29, zmm29, zmm29
.endm

.macro GHASH_PREP_Y d0, d1, d2, d3
  vpshufb \d0, \d0, ymm31
  vpshufb \d1, \d1, ymm31
  vpshufb \d2, \d2, ymm31
  vpshufb \d3, \d3, ymm31

  vpxord ymm30, ymm30, ymm30
  vmovdqu64 xmm30, XMMWORD PTR [r9]
  vpxord \d0, \d0, ymm30

  vpxord ymm28, ymm28, ymm28
  vpxord ymm29, ymm29, ymm29
.endm

.macro GHASH_REDUCE
  vmovdqa64 ymm20, ymm28
  vextracti64x4 ymm21, zmm28, 1
  vpxorq ymm20, ymm20, ymm21
  vextracti64x2 xmm21, ymm20, 1
  vpxorq xmm20, xmm20, xmm21

  vmovdqa64 ymm22, ymm29
  vextracti64x4 ymm21, zmm29, 1
  vpxorq ymm22, ymm22, ymm21
  vextracti64x2 xmm21, ymm22, 1
  vpxorq xmm22, xmm22, xmm21

  vpsllq xmm21, xmm20, 63
  vpsllq xmm23, xmm20, 62
  vpxorq xmm21, xmm21, xmm23
  vpsllq xmm23, xmm20, 57
  vpxorq xmm21, xmm21, xmm23
  vpslldq xmm21, xmm21, 8
  vpxorq xmm20, xmm20, xmm21

  vpsrlq xmm21, xmm20, 1
  vpxorq xmm21, xmm21, xmm20
  vpsrlq xmm23, xmm20, 2
  vpxorq xmm21, xmm21, xmm23
  vpsrlq xmm23, xmm20, 7
  vpxorq xmm21, xmm21, xmm23

  vpsllq xmm23, xmm20, 63
  vpsllq xmm24, xmm20, 62
  vpxorq xmm23, xmm23, xmm24
  vpsllq xmm24, xmm20, 57
  vpxorq xmm23, xmm23, xmm24
  vpsrldq xmm23, xmm23, 8

  vpxorq xmm21, xmm21, xmm23
  vpxorq xmm21, xmm21, xmm22
  vmovdqu64 XMMWORD PTR [r9], xmm21
.endm

.macro GHASH_REDUCE_Y
  vmovdqa64 xmm16, xmm28
  vextracti64x2 xmm17, ymm28, 1
  vpxorq xmm16, xmm16, xmm17

  vmovdqa64 xmm19, xmm29
  vextracti64x2 xmm17, ymm29, 1
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

.macro GHASH16_REG d0, d1, d2, d3
  GHASH16_REG_H \d0, \d1, \d2, \d3, zmm12, zmm13, zmm14, zmm15
.endm

.macro GHASH16_REG_H d0, d1, d2, d3, h0, h1, h2, h3
  GHASH_PREP \d0, \d1, \d2, \d3
  GHASH_FOLD \d0, \h0
  GHASH_FOLD \d1, \h1
  GHASH_FOLD \d2, \h2
  GHASH_FOLD \d3, \h3
  GHASH_REDUCE
.endm

.macro GHASH16
  GHASH16_REG zmm0, zmm1, zmm2, zmm3
.endm

.macro GHASH32_PREP d0, d1, d2, d3, d4, d5, d6, d7
  vpshufb \d0, \d0, zmm31
  vpshufb \d1, \d1, zmm31
  vpshufb \d2, \d2, zmm31
  vpshufb \d3, \d3, zmm31
  vpshufb \d4, \d4, zmm31
  vpshufb \d5, \d5, zmm31
  vpshufb \d6, \d6, zmm31
  vpshufb \d7, \d7, zmm31

  vpxord zmm30, zmm30, zmm30
  vmovdqu64 xmm30, XMMWORD PTR [r9]
  vpxord \d0, \d0, zmm30

  vpxord zmm28, zmm28, zmm28
  vpxord zmm29, zmm29, zmm29
.endm

.macro GHASH32_REG d0, d1, d2, d3, d4, d5, d6, d7
  GHASH32_PREP \d0, \d1, \d2, \d3, \d4, \d5, \d6, \d7
  GHASH_FOLD \d0, zmm12
  GHASH_FOLD \d1, zmm13
  GHASH_FOLD \d2, zmm14
  GHASH_FOLD \d3, zmm15
  GHASH_FOLD \d4, zmm16
  GHASH_FOLD \d5, zmm17
  GHASH_FOLD \d6, zmm18
  GHASH_FOLD \d7, zmm19
  GHASH_REDUCE
.endm

.macro GHASH8Y_REG d0, d1, d2, d3
  GHASH_PREP_Y \d0, \d1, \d2, \d3
  GHASH_FOLD_Y \d0, ymm12
  GHASH_FOLD_Y \d1, ymm13
  GHASH_FOLD_Y \d2, ymm14
  GHASH_FOLD_Y \d3, ymm15
  GHASH_REDUCE_Y
.endm

.macro AES_ENCRYPT_16_WITH_GHASH aes256, d0, d1, d2, d3
  AES_START
  GHASH_PREP \d0, \d1, \d2, \d3
  AES_ROUND 16
  GHASH_FOLD \d0, zmm12
  AES_ROUND 32
  GHASH_FOLD \d1, zmm13
  AES_ROUND 48
  GHASH_FOLD \d2, zmm14
  AES_ROUND 64
  GHASH_FOLD \d3, zmm15
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
    GHASH_REDUCE
    AES_LAST 224
  .else
    GHASH_REDUCE
    AES_LAST 160
  .endif
.endm

.macro AES_ENCRYPT_16_WITH_GHASH32 aes256, d0, d1, d2, d3, d4, d5, d6, d7
  GHASH32_PREP \d0, \d1, \d2, \d3, \d4, \d5, \d6, \d7
  AES_START
  GHASH_FOLD \d0, zmm12
  AES_ROUND 16
  GHASH_FOLD \d1, zmm13
  AES_ROUND 32
  GHASH_FOLD \d2, zmm14
  AES_ROUND 48
  GHASH_FOLD \d3, zmm15
  AES_ROUND 64
  GHASH_FOLD \d4, zmm16
  AES_ROUND 80
  GHASH_FOLD \d5, zmm17
  AES_ROUND 96
  GHASH_FOLD \d6, zmm18
  AES_ROUND 112
  GHASH_FOLD \d7, zmm19
  AES_ROUND 128
  AES_ROUND 144
  .if \aes256
    AES_ROUND 160
    AES_ROUND 176
    AES_ROUND 192
    AES_ROUND 208
    GHASH_REDUCE
    AES_LAST 224
  .else
    GHASH_REDUCE
    AES_LAST 160
  .endif
.endm

.macro AES_ENCRYPT_8Y_WITH_GHASH aes256, d0, d1, d2, d3
  AES_Y_START
  GHASH_PREP_Y \d0, \d1, \d2, \d3
  AES_Y_ROUND 16
  GHASH_FOLD_Y \d0, ymm12
  AES_Y_ROUND 32
  GHASH_FOLD_Y \d1, ymm13
  AES_Y_ROUND 48
  GHASH_FOLD_Y \d2, ymm14
  AES_Y_ROUND 64
  GHASH_FOLD_Y \d3, ymm15
  AES_Y_ROUND 80
  AES_Y_ROUND 96
  AES_Y_ROUND 112
  AES_Y_ROUND 128
  AES_Y_ROUND 144
  .if \aes256
    AES_Y_ROUND 160
    AES_Y_ROUND 176
    AES_Y_ROUND 192
    AES_Y_ROUND 208
    GHASH_REDUCE_Y
    AES_Y_LAST 224
  .else
    GHASH_REDUCE_Y
    AES_Y_LAST 160
  .endif
.endm

.macro AES_GCM_STORE_Z open, d0, d1, d2, d3
  .if \open
    vmovdqu64 \d0, ZMMWORD PTR [rdx]
    vmovdqu64 \d1, ZMMWORD PTR [rdx + 64]
    vmovdqu64 \d2, ZMMWORD PTR [rdx + 128]
    vmovdqu64 \d3, ZMMWORD PTR [rdx + 192]
    vpxord zmm20, zmm0, \d0
    vpxord zmm21, zmm1, \d1
    vpxord zmm22, zmm2, \d2
    vpxord zmm23, zmm3, \d3
    vmovdqu64 ZMMWORD PTR [rdx], zmm20
    vmovdqu64 ZMMWORD PTR [rdx + 64], zmm21
    vmovdqu64 ZMMWORD PTR [rdx + 128], zmm22
    vmovdqu64 ZMMWORD PTR [rdx + 192], zmm23
  .else
    vmovdqu64 \d0, ZMMWORD PTR [rdx]
    vmovdqu64 \d1, ZMMWORD PTR [rdx + 64]
    vmovdqu64 \d2, ZMMWORD PTR [rdx + 128]
    vmovdqu64 \d3, ZMMWORD PTR [rdx + 192]
    vpxord \d0, \d0, zmm0
    vpxord \d1, \d1, zmm1
    vpxord \d2, \d2, zmm2
    vpxord \d3, \d3, zmm3
    vmovdqu64 ZMMWORD PTR [rdx], \d0
    vmovdqu64 ZMMWORD PTR [rdx + 64], \d1
    vmovdqu64 ZMMWORD PTR [rdx + 128], \d2
    vmovdqu64 ZMMWORD PTR [rdx + 192], \d3
  .endif
.endm

.macro AES_GCM_16X_FUNC name, aes256, open
  .p2align 5
  .globl \name
  .type \name, @function
\name:
  mov r10d, DWORD PTR [r9 + 16]

  cmp rcx, 256
  jb .L\name\()_done

  vmovdqu64 zmm31, ZMMWORD PTR [rip + .Lrscrypto_x86_gcm_bswap]
  vmovdqu64 zmm26, ZMMWORD PTR [rip + .Lrscrypto_x86_dword_bswap]
  mov eax, 0x8888
  kmovw k1, eax

  cmp rcx, 512
  jb .L\name\()_single_16

  vmovdqu64 zmm12, ZMMWORD PTR [r8]
  vmovdqu64 zmm13, ZMMWORD PTR [r8 + 64]
  vmovdqu64 zmm14, ZMMWORD PTR [r8 + 128]
  vmovdqu64 zmm15, ZMMWORD PTR [r8 + 192]
  vmovdqu64 zmm16, ZMMWORD PTR [r8 + 256]
  vmovdqu64 zmm17, ZMMWORD PTR [r8 + 320]
  vmovdqu64 zmm18, ZMMWORD PTR [r8 + 384]
  vmovdqu64 zmm19, ZMMWORD PTR [r8 + 448]

  GCM_COUNTERS_16

  AES_ENCRYPT_16 \aes256
  AES_GCM_STORE_Z \open, zmm4, zmm5, zmm6, zmm7

  add rdx, 256
  sub rcx, 256
  add r10d, 16
  add QWORD PTR [r9 + 24], 256

  GCM_COUNTERS_16

  AES_ENCRYPT_16 \aes256
  AES_GCM_STORE_Z \open, zmm8, zmm9, zmm10, zmm11

  add rdx, 256
  sub rcx, 256
  add r10d, 16
  add QWORD PTR [r9 + 24], 256
  cmp rcx, 512
  jb .L\name\()_final_32

  .p2align 5
.L\name\()_loop_32:
  GCM_COUNTERS_16

  AES_ENCRYPT_16_WITH_GHASH32 \aes256, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11
  AES_GCM_STORE_Z \open, zmm4, zmm5, zmm6, zmm7

  add rdx, 256
  sub rcx, 256
  add r10d, 16
  add QWORD PTR [r9 + 24], 256

  GCM_COUNTERS_16

  AES_ENCRYPT_16 \aes256
  AES_GCM_STORE_Z \open, zmm8, zmm9, zmm10, zmm11

  add rdx, 256
  sub rcx, 256
  add r10d, 16
  add QWORD PTR [r9 + 24], 256
  cmp rcx, 512
  jae .L\name\()_loop_32

.L\name\()_final_32:
  GHASH32_REG zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11
  cmp rcx, 256
  jb .L\name\()_done

  GCM_COUNTERS_16

  AES_ENCRYPT_16 \aes256
  AES_GCM_STORE_Z \open, zmm4, zmm5, zmm6, zmm7

  add rdx, 256
  sub rcx, 256
  add r10d, 16
  add QWORD PTR [r9 + 24], 256
  GHASH16_REG_H zmm4, zmm5, zmm6, zmm7, zmm16, zmm17, zmm18, zmm19
  jmp .L\name\()_done

.L\name\()_single_16:
  vmovdqu64 zmm12, ZMMWORD PTR [r8 + 256]
  vmovdqu64 zmm13, ZMMWORD PTR [r8 + 320]
  vmovdqu64 zmm14, ZMMWORD PTR [r8 + 384]
  vmovdqu64 zmm15, ZMMWORD PTR [r8 + 448]

  GCM_COUNTERS_16

  AES_ENCRYPT_16 \aes256
  AES_GCM_STORE_Z \open, zmm4, zmm5, zmm6, zmm7

  add rdx, 256
  sub rcx, 256
  add r10d, 16
  add QWORD PTR [r9 + 24], 256
  GHASH16_REG zmm4, zmm5, zmm6, zmm7

.L\name\()_done:
  mov DWORD PTR [r9 + 16], r10d
  vzeroupper
  ret
  .size \name, . - \name
.endm

.macro AES_GCM_8Y_FUNC name, aes256, open
  .p2align 5
  .globl \name
  .type \name, @function
\name:
  mov r10d, DWORD PTR [r9 + 16]

  cmp rcx, 128
  jb .L\name\()_done

  vmovdqu64 ymm31, YMMWORD PTR [rip + .Lrscrypto_x86_gcm_bswap]
  vmovdqu64 ymm26, YMMWORD PTR [rip + .Lrscrypto_x86_dword_bswap]
  vmovdqu64 ymm12, YMMWORD PTR [r8]
  vmovdqu64 ymm13, YMMWORD PTR [r8 + 32]
  vmovdqu64 ymm14, YMMWORD PTR [r8 + 64]
  vmovdqu64 ymm15, YMMWORD PTR [r8 + 96]
  mov eax, 0x88
  kmovw k1, eax

  GCM_COUNTERS_8

  AES_ENCRYPT_8Y \aes256

  .if \open
    vmovdqu64 ymm4, YMMWORD PTR [rdx]
    vmovdqu64 ymm5, YMMWORD PTR [rdx + 32]
    vmovdqu64 ymm6, YMMWORD PTR [rdx + 64]
    vmovdqu64 ymm7, YMMWORD PTR [rdx + 96]
    vpxord ymm8, ymm0, ymm4
    vpxord ymm9, ymm1, ymm5
    vpxord ymm10, ymm2, ymm6
    vpxord ymm11, ymm3, ymm7
    vmovdqu64 YMMWORD PTR [rdx], ymm8
    vmovdqu64 YMMWORD PTR [rdx + 32], ymm9
    vmovdqu64 YMMWORD PTR [rdx + 64], ymm10
    vmovdqu64 YMMWORD PTR [rdx + 96], ymm11
  .else
    vmovdqu64 ymm4, YMMWORD PTR [rdx]
    vmovdqu64 ymm5, YMMWORD PTR [rdx + 32]
    vmovdqu64 ymm6, YMMWORD PTR [rdx + 64]
    vmovdqu64 ymm7, YMMWORD PTR [rdx + 96]
    vpxord ymm4, ymm4, ymm0
    vpxord ymm5, ymm5, ymm1
    vpxord ymm6, ymm6, ymm2
    vpxord ymm7, ymm7, ymm3
    vmovdqu64 YMMWORD PTR [rdx], ymm4
    vmovdqu64 YMMWORD PTR [rdx + 32], ymm5
    vmovdqu64 YMMWORD PTR [rdx + 64], ymm6
    vmovdqu64 YMMWORD PTR [rdx + 96], ymm7
  .endif

  add rdx, 128
  sub rcx, 128
  add r10d, 8
  add QWORD PTR [r9 + 24], 128
  cmp rcx, 128
  jb .L\name\()_final_ghash

  .p2align 5
.L\name\()_loop:
  GCM_COUNTERS_8

  AES_ENCRYPT_8Y_WITH_GHASH \aes256, ymm4, ymm5, ymm6, ymm7

  .if \open
    vmovdqu64 ymm4, YMMWORD PTR [rdx]
    vmovdqu64 ymm5, YMMWORD PTR [rdx + 32]
    vmovdqu64 ymm6, YMMWORD PTR [rdx + 64]
    vmovdqu64 ymm7, YMMWORD PTR [rdx + 96]
    vpxord ymm8, ymm0, ymm4
    vpxord ymm9, ymm1, ymm5
    vpxord ymm10, ymm2, ymm6
    vpxord ymm11, ymm3, ymm7
    vmovdqu64 YMMWORD PTR [rdx], ymm8
    vmovdqu64 YMMWORD PTR [rdx + 32], ymm9
    vmovdqu64 YMMWORD PTR [rdx + 64], ymm10
    vmovdqu64 YMMWORD PTR [rdx + 96], ymm11
  .else
    vmovdqu64 ymm4, YMMWORD PTR [rdx]
    vmovdqu64 ymm5, YMMWORD PTR [rdx + 32]
    vmovdqu64 ymm6, YMMWORD PTR [rdx + 64]
    vmovdqu64 ymm7, YMMWORD PTR [rdx + 96]
    vpxord ymm4, ymm4, ymm0
    vpxord ymm5, ymm5, ymm1
    vpxord ymm6, ymm6, ymm2
    vpxord ymm7, ymm7, ymm3
    vmovdqu64 YMMWORD PTR [rdx], ymm4
    vmovdqu64 YMMWORD PTR [rdx + 32], ymm5
    vmovdqu64 YMMWORD PTR [rdx + 64], ymm6
    vmovdqu64 YMMWORD PTR [rdx + 96], ymm7
  .endif

  add rdx, 128
  sub rcx, 128
  add r10d, 8
  add QWORD PTR [r9 + 24], 128
  cmp rcx, 128
  jae .L\name\()_loop

.L\name\()_final_ghash:
  GHASH8Y_REG ymm4, ymm5, ymm6, ymm7

.L\name\()_done:
  mov DWORD PTR [r9 + 16], r10d
  vzeroupper
  ret
  .size \name, . - \name
.endm

AES_GCM_16X_FUNC rscrypto_aes128_gcm_seal_16x_vaes512_x86_64_linux, 0, 0
AES_GCM_16X_FUNC rscrypto_aes128_gcm_open_16x_vaes512_x86_64_linux, 0, 1
AES_GCM_16X_FUNC rscrypto_aes256_gcm_seal_16x_vaes512_x86_64_linux, 1, 0
AES_GCM_16X_FUNC rscrypto_aes256_gcm_open_16x_vaes512_x86_64_linux, 1, 1
AES_GCM_8Y_FUNC rscrypto_aes128_gcm_seal_8x_vaes256_x86_64_linux, 0, 0
AES_GCM_8Y_FUNC rscrypto_aes128_gcm_open_8x_vaes256_x86_64_linux, 0, 1
AES_GCM_8Y_FUNC rscrypto_aes256_gcm_seal_8x_vaes256_x86_64_linux, 1, 0
AES_GCM_8Y_FUNC rscrypto_aes256_gcm_open_8x_vaes256_x86_64_linux, 1, 1

.section .rodata.cst64,"aM",@progbits,64
.p2align 6
.Lrscrypto_x86_gcm_bswap:
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

.p2align 6
.Lrscrypto_x86_dword_bswap:
  .byte 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12
  .byte 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12
  .byte 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12
  .byte 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12

.p2align 6
.Lrscrypto_x86_ctr_inc_z0:
  .long 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3
.Lrscrypto_x86_ctr_inc_z1:
  .long 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7
.Lrscrypto_x86_ctr_inc_z2:
  .long 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 10, 0, 0, 0, 11
.Lrscrypto_x86_ctr_inc_z3:
  .long 0, 0, 0, 12, 0, 0, 0, 13, 0, 0, 0, 14, 0, 0, 0, 15

.p2align 5
.Lrscrypto_x86_ctr_inc_y0:
  .long 0, 0, 0, 0, 0, 0, 0, 1
.Lrscrypto_x86_ctr_inc_y1:
  .long 0, 0, 0, 2, 0, 0, 0, 3
.Lrscrypto_x86_ctr_inc_y2:
  .long 0, 0, 0, 4, 0, 0, 0, 5
.Lrscrypto_x86_ctr_inc_y3:
  .long 0, 0, 0, 6, 0, 0, 0, 7
