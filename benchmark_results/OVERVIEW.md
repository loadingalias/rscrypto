# Benchmark Overview

Source: CI run [#24942749587](https://github.com/loadingalias/rscrypto/actions/runs/24942749587) on 2026-04-25, plus local macOS Apple Silicon results from 2026-04-26.
Commit: `7d2d8119622af1f2f77a5c2d2a9bd9457dc76298`

Scope: eight Linux CI runners plus local macOS aarch64. RISC-V is omitted because the RISE runner for the CI run is scalar `rv64imafdcsu` and failed an RVV forced-kernel diagnostic bench.

Speedup is `competitor_time / rscrypto_time`. Values above `1.00x` mean `rscrypto` is faster. Losses are cases below `0.95x`, matching the 5% loss budget.

## Platforms

| Short | Runner | Architecture | Source |
|---|---|---|---|
| Zen4 | AMD EPYC (Zen 4) | x86-64 | ci |
| Zen5 | AMD EPYC (Zen 5) | x86-64 | ci |
| SPR | Intel Xeon (Sapphire Rapids) | x86-64 | ci |
| ICL | Intel Xeon (Ice Lake) | x86-64 | ci |
| Grav3 | AWS Graviton 3 | aarch64 | ci |
| Grav4 | AWS Graviton 4 | aarch64 | ci |
| s390x | IBM Z s390x | s390x | ci |
| POWER10 | IBM POWER10 | ppc64le | ci |
| macOS-A64 | macOS Apple Silicon | aarch64 | local |

## Overall Scoreboard

**3820W / 1599T / 327L = 66% win rate** (5746 comparisons)

## By Category

| Category | W | T | L | Total | Win % | Worst | Platform | Group | Case |
|---|---|---|---|---|---|---|---|---|---|
| Checksums | 821 | 138 | 31 | 990 | 83% | 0.56x | ICL | crc64-nvme vs crc64fast-nvme | 256 |
| SHA-2 | 317 | 204 | 10 | 531 | 60% | 0.01x | SPR | sha256/streaming vs sha2 | 64B |
| SHA-3 | 366 | 41 | 7 | 414 | 88% | 0.87x | s390x | sha3-512 vs sha3 | 0 |
| SHAKE | 190 | 12 | 7 | 209 | 91% | 0.76x | macOS-A64 | shake128 vs sha3 | 65536 |
| BLAKE2 | 426 | 437 | 19 | 882 | 48% | 0.78x | macOS-A64 | blake2/host-stream-overhead/blake2s256 vs rustcrypto | 64 |
| BLAKE3 | 242 | 159 | 31 | 432 | 56% | 0.73x | s390x | blake3/xof vs blake3 | 1024 |
| Ascon | 144 | 54 | 0 | 198 | 73% | 0.99x | Grav3 | ascon-xof128 vs ascon-hash | 262144 |
| XXH3 | 113 | 52 | 33 | 198 | 57% | 0.37x | Zen5 | xxh3-64 vs xxhash-rust | 256 |
| RapidHash | 69 | 240 | 87 | 396 | 17% | 0.53x | s390x | rapidhash-v3-64 vs rapidhash | 1 |
| Auth | 332 | 156 | 18 | 506 | 66% | 0.01x | SPR | hmac-sha256/streaming vs rustcrypto | 64B |
| AEAD | 800 | 106 | 84 | 990 | 81% | 0.41x | SPR | aes-256-gcm/encrypt vs rustcrypto | 32 |

## By Platform

| Platform | W | T | L | Total | Win % | Worst | Category | Group | Case |
|---|---|---|---|---|---|---|---|---|---|
| Zen4 | 486 | 117 | 29 | 632 | 77% | 0.42x | AEAD | aes-256-gcm/encrypt vs rustcrypto | 32 |
| Zen5 | 417 | 174 | 41 | 632 | 66% | 0.37x | XXH3 | xxh3-64 vs xxhash-rust | 256 |
| SPR | 499 | 93 | 40 | 632 | 79% | 0.01x | SHA-2 | sha256/streaming vs sha2 | 64B |
| ICL | 469 | 136 | 27 | 632 | 74% | 0.43x | AEAD | aes-256-gcm/encrypt vs rustcrypto | 32 |
| Grav3 | 346 | 247 | 39 | 632 | 55% | 0.76x | AEAD | aegis-256/encrypt vs aegis-crate | 0 |
| Grav4 | 351 | 249 | 32 | 632 | 56% | 0.65x | RapidHash | rapidhash-v3-128 vs rapidhash | 64 |
| s390x | 517 | 69 | 46 | 632 | 82% | 0.53x | RapidHash | rapidhash-v3-64 vs rapidhash | 1 |
| POWER10 | 350 | 259 | 23 | 632 | 55% | 0.71x | RapidHash | rapidhash-128 vs rapidhash | 1 |
| macOS-A64 | 385 | 255 | 50 | 690 | 56% | 0.76x | XXH3 | xxh3-64 vs xxhash-rust | 32 |

## By Primitive Group

| Category | Group | W | T | L | Total | Win % | Worst | Platform | Case |
|---|---|---|---|---|---|---|---|---|---|
| Checksums | crc16-ccitt vs crc | 98 | 0 | 1 | 99 | 99% | 0.69x | s390x | 1 |
| Checksums | crc16-ibm vs crc | 98 | 0 | 1 | 99 | 99% | 0.78x | s390x | 1 |
| Checksums | crc24-openpgp vs crc | 97 | 1 | 1 | 99 | 98% | 0.64x | s390x | 64 |
| Checksums | crc32 vs crc-fast | 61 | 31 | 7 | 99 | 62% | 0.90x | macOS-A64 | 256 |
| Checksums | crc32 vs crc32fast | 92 | 3 | 4 | 99 | 93% | 0.73x | ICL | 256 |
| Checksums | crc32c vs crc-fast | 55 | 35 | 9 | 99 | 56% | 0.82x | Grav4 | 256 |
| Checksums | crc32c vs crc32c | 96 | 2 | 1 | 99 | 97% | 0.89x | s390x | 1 |
| Checksums | crc64-nvme vs crc-fast | 63 | 35 | 1 | 99 | 64% | 0.61x | ICL | 256 |
| Checksums | crc64-nvme vs crc64fast-nvme | 79 | 16 | 4 | 99 | 80% | 0.56x | ICL | 256 |
| Checksums | crc64-xz vs crc64fast | 82 | 15 | 2 | 99 | 83% | 0.79x | POWER10 | 64 |
| SHA-2 | sha224 vs sha2 | 57 | 41 | 1 | 99 | 58% | 0.80x | macOS-A64 | 32 |
| SHA-2 | sha256 vs sha2 | 57 | 42 | 0 | 99 | 58% | 0.95x | Grav4 | 256 |
| SHA-2 | sha256/streaming vs sha2 | 2 | 10 | 6 | 18 | 11% | 0.01x | SPR | 64B |
| SHA-2 | sha384 vs sha2 | 67 | 31 | 1 | 99 | 68% | 0.92x | macOS-A64 | 256 |
| SHA-2 | sha512 vs sha2 | 63 | 35 | 1 | 99 | 64% | 0.92x | macOS-A64 | 256 |
| SHA-2 | sha512-256 vs sha2 | 63 | 35 | 1 | 99 | 64% | 0.90x | macOS-A64 | 256 |
| SHA-2 | sha512/streaming vs sha2 | 8 | 10 | 0 | 18 | 44% | 0.96x | Grav3 | 64B |
| SHA-3 | sha3-224 vs sha3 | 88 | 11 | 0 | 99 | 89% | 0.99x | POWER10 | 4096 |
| SHA-3 | sha3-256 vs sha3 | 88 | 11 | 0 | 99 | 89% | 0.96x | macOS-A64 | 0 |
| SHA-3 | sha3-256/streaming vs sha3 | 14 | 2 | 2 | 18 | 78% | 0.90x | macOS-A64 | 4096B |
| SHA-3 | sha3-384 vs sha3 | 89 | 7 | 3 | 99 | 90% | 0.91x | macOS-A64 | 0 |
| SHA-3 | sha3-512 vs sha3 | 87 | 10 | 2 | 99 | 88% | 0.87x | s390x | 0 |
| SHAKE | cshake256 vs tiny-keccak | 11 | 0 | 0 | 11 | 100% | 1.18x | macOS-A64 | 64 |
| SHAKE | shake128 vs sha3 | 89 | 3 | 7 | 99 | 90% | 0.76x | macOS-A64 | 65536 |
| SHAKE | shake256 vs sha3 | 90 | 9 | 0 | 99 | 91% | 0.98x | macOS-A64 | 0 |
| BLAKE2 | blake2/blake2b256 vs rustcrypto | 45 | 53 | 1 | 99 | 45% | 0.90x | macOS-A64 | 1 |
| BLAKE2 | blake2/blake2b512 vs rustcrypto | 36 | 63 | 0 | 99 | 36% | 0.96x | Zen4 | 1048576 |
| BLAKE2 | blake2/blake2s128 vs rustcrypto | 56 | 43 | 0 | 99 | 57% | 0.98x | macOS-A64 | 0 |
| BLAKE2 | blake2/blake2s256 vs rustcrypto | 48 | 51 | 0 | 99 | 48% | 0.95x | Zen5 | 1 |
| BLAKE2 | blake2/host-keyed-overhead/blake2b256 vs rustcrypto | 1 | 5 | 0 | 6 | 17% | 0.97x | macOS-A64 | 16 |
| BLAKE2 | blake2/host-keyed-overhead/blake2s256 vs rustcrypto | 6 | 0 | 0 | 6 | 100% | 1.03x | macOS-A64 | 128 |
| BLAKE2 | blake2/host-overhead/blake2b256 vs rustcrypto | 1 | 5 | 0 | 6 | 17% | 0.99x | macOS-A64 | 1 |
| BLAKE2 | blake2/host-overhead/blake2s256 vs rustcrypto | 1 | 5 | 0 | 6 | 17% | 0.97x | macOS-A64 | 0 |
| BLAKE2 | blake2/host-stream-overhead/blake2b256 vs rustcrypto | 0 | 1 | 5 | 6 | 0% | 0.89x | macOS-A64 | 32 |
| BLAKE2 | blake2/host-stream-overhead/blake2s256 vs rustcrypto | 0 | 0 | 6 | 6 | 0% | 0.78x | macOS-A64 | 64 |
| BLAKE2 | blake2/keyed/blake2b256 vs rustcrypto | 48 | 48 | 3 | 99 | 48% | 0.92x | POWER10 | 32 |
| BLAKE2 | blake2/keyed/blake2b512 vs rustcrypto | 53 | 43 | 3 | 99 | 54% | 0.92x | POWER10 | 32 |
| BLAKE2 | blake2/keyed/blake2s128 vs rustcrypto | 55 | 43 | 1 | 99 | 56% | 0.94x | POWER10 | 1 |
| BLAKE2 | blake2/keyed/blake2s256 vs rustcrypto | 55 | 44 | 0 | 99 | 56% | 0.95x | POWER10 | 1 |
| BLAKE2 | blake2/streaming/blake2b256 vs rustcrypto | 9 | 18 | 0 | 27 | 33% | 0.95x | POWER10 | 64B |
| BLAKE2 | blake2/streaming/blake2s256 vs rustcrypto | 12 | 15 | 0 | 27 | 44% | 0.98x | Zen5 | 65536B |
| BLAKE3 | blake3 vs blake3 | 47 | 45 | 7 | 99 | 47% | 0.75x | s390x | 1024 |
| BLAKE3 | blake3/derive-key vs blake3 | 85 | 12 | 2 | 99 | 86% | 0.82x | s390x | 1024 |
| BLAKE3 | blake3/keyed vs blake3 | 45 | 45 | 9 | 99 | 45% | 0.75x | s390x | 1024 |
| BLAKE3 | blake3/streaming vs blake3 | 12 | 20 | 4 | 36 | 33% | 0.86x | Zen5 | 64B |
| BLAKE3 | blake3/xof vs blake3 | 53 | 37 | 9 | 99 | 54% | 0.73x | s390x | 1024 |
| Ascon | ascon-hash256 vs ascon-hash | 64 | 35 | 0 | 99 | 65% | 0.99x | Grav3 | 32 |
| Ascon | ascon-xof128 vs ascon-hash | 80 | 19 | 0 | 99 | 81% | 0.99x | Grav3 | 262144 |
| XXH3 | xxh3-128 vs xxhash-rust | 53 | 31 | 15 | 99 | 54% | 0.68x | SPR | 256 |
| XXH3 | xxh3-64 vs xxhash-rust | 60 | 21 | 18 | 99 | 61% | 0.37x | Zen5 | 256 |
| RapidHash | rapidhash-128 vs rapidhash | 10 | 73 | 16 | 99 | 10% | 0.71x | POWER10 | 1 |
| RapidHash | rapidhash-64 vs rapidhash | 13 | 76 | 10 | 99 | 13% | 0.66x | Zen5 | 32 |
| RapidHash | rapidhash-v3-128 vs rapidhash | 28 | 44 | 27 | 99 | 28% | 0.65x | Grav4 | 64 |
| RapidHash | rapidhash-v3-64 vs rapidhash | 18 | 47 | 34 | 99 | 18% | 0.53x | s390x | 1 |
| Auth | ed25519/keypair-from-secret vs dalek | 9 | 0 | 0 | 9 | 100% | 1.27x | Grav4 | rscrypto |
| Auth | ed25519/public-key-from-secret vs dalek | 9 | 0 | 0 | 9 | 100% | 1.29x | Grav4 | rscrypto |
| Auth | ed25519/sign vs dalek | 36 | 0 | 0 | 36 | 100% | 1.12x | ICL | 16384 |
| Auth | ed25519/verify vs dalek | 26 | 7 | 3 | 36 | 72% | 0.92x | Zen5 | 1024 |
| Auth | hkdf-sha256/expand vs rustcrypto | 29 | 5 | 2 | 36 | 81% | 0.81x | macOS-A64 | 32 |
| Auth | hkdf-sha384/expand vs rustcrypto | 32 | 4 | 0 | 36 | 89% | 0.99x | macOS-A64 | 48 |
| Auth | hmac-sha256 vs rustcrypto | 40 | 54 | 5 | 99 | 40% | 0.73x | SPR | 1024 |
| Auth | hmac-sha256/streaming vs rustcrypto | 2 | 10 | 6 | 18 | 11% | 0.01x | SPR | 64B |
| Auth | hmac-sha384 vs rustcrypto | 62 | 37 | 0 | 99 | 63% | 0.97x | macOS-A64 | 1024 |
| Auth | hmac-sha512 vs rustcrypto | 59 | 38 | 2 | 99 | 60% | 0.92x | macOS-A64 | 256 |
| Auth | kmac256 vs tiny-keccak | 11 | 0 | 0 | 11 | 100% | 1.11x | macOS-A64 | 0 |
| Auth | x25519/diffie-hellman vs dalek | 8 | 1 | 0 | 9 | 89% | 1.02x | SPR | rscrypto |
| Auth | x25519/public-key-from-secret vs dalek | 9 | 0 | 0 | 9 | 100% | 1.16x | macOS-A64 | rscrypto |
| AEAD | aegis-256/decrypt vs aegis-crate | 70 | 18 | 11 | 99 | 71% | 0.86x | Grav3 | 1048576 |
| AEAD | aegis-256/encrypt vs aegis-crate | 43 | 27 | 29 | 99 | 43% | 0.67x | Grav4 | 0 |
| AEAD | aes-256-gcm-siv/decrypt vs rustcrypto | 99 | 0 | 0 | 99 | 100% | 1.13x | SPR | 32 |
| AEAD | aes-256-gcm-siv/encrypt vs rustcrypto | 99 | 0 | 0 | 99 | 100% | 1.05x | ICL | 32 |
| AEAD | aes-256-gcm/decrypt vs rustcrypto | 83 | 4 | 12 | 99 | 84% | 0.53x | SPR | 32 |
| AEAD | aes-256-gcm/encrypt vs rustcrypto | 82 | 1 | 16 | 99 | 83% | 0.41x | SPR | 32 |
| AEAD | chacha20-poly1305/decrypt vs rustcrypto | 83 | 12 | 4 | 99 | 84% | 0.62x | Zen5 | 256 |
| AEAD | chacha20-poly1305/encrypt vs rustcrypto | 79 | 16 | 4 | 99 | 80% | 0.62x | Zen5 | 256 |
| AEAD | xchacha20-poly1305/decrypt vs rustcrypto | 82 | 13 | 4 | 99 | 83% | 0.67x | Zen5 | 256 |
| AEAD | xchacha20-poly1305/encrypt vs rustcrypto | 80 | 15 | 4 | 99 | 81% | 0.66x | Zen5 | 256 |

## Losses Over 5 Percent

| Speedup | Platform | Category | Group | Case | Competitor |
|---|---|---|---|---|---|
| 0.01x | SPR | SHA-2 | sha256/streaming vs sha2 | 64B | sha2 |
| 0.01x | SPR | Auth | hmac-sha256/streaming vs rustcrypto | 64B | rustcrypto |
| 0.01x | SPR | SHA-2 | sha256/streaming vs sha2 | 4096B | sha2 |
| 0.01x | SPR | Auth | hmac-sha256/streaming vs rustcrypto | 4096B | rustcrypto |
| 0.37x | Zen5 | XXH3 | xxh3-64 vs xxhash-rust | 256 | xxhash-rust |
| 0.41x | SPR | AEAD | aes-256-gcm/encrypt vs rustcrypto | 32 | rustcrypto |
| 0.42x | Zen4 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 32 | rustcrypto |
| 0.43x | ICL | AEAD | aes-256-gcm/encrypt vs rustcrypto | 32 | rustcrypto |
| 0.45x | Zen5 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 32 | rustcrypto |
| 0.46x | ICL | AEAD | aes-256-gcm/encrypt vs rustcrypto | 0 | rustcrypto |
| 0.47x | SPR | AEAD | aes-256-gcm/encrypt vs rustcrypto | 0 | rustcrypto |
| 0.48x | Zen4 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 0 | rustcrypto |
| 0.53x | ICL | XXH3 | xxh3-64 vs xxhash-rust | 64 | xxhash-rust |
| 0.53x | s390x | RapidHash | rapidhash-v3-64 vs rapidhash | 1 | rapidhash |
| 0.53x | SPR | AEAD | aes-256-gcm/decrypt vs rustcrypto | 32 | rustcrypto |
| 0.55x | SPR | XXH3 | xxh3-64 vs xxhash-rust | 64 | xxhash-rust |
| 0.56x | ICL | Checksums | crc64-nvme vs crc64fast-nvme | 256 | crc64fast-nvme |
| 0.56x | ICL | AEAD | aes-256-gcm/encrypt vs rustcrypto | 1 | rustcrypto |
| 0.59x | Zen5 | AEAD | aes-256-gcm/decrypt vs rustcrypto | 32 | rustcrypto |
| 0.59x | Zen5 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 0 | rustcrypto |
| 0.61x | Zen4 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 1 | rustcrypto |
| 0.61x | ICL | Checksums | crc64-nvme vs crc-fast | 256 | crc-fast |
| 0.62x | Zen5 | AEAD | chacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.62x | SPR | AEAD | aes-256-gcm/encrypt vs rustcrypto | 1 | rustcrypto |
| 0.62x | s390x | XXH3 | xxh3-64 vs xxhash-rust | 32 | xxhash-rust |
| 0.62x | Zen5 | AEAD | chacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.63x | Zen4 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 64 | rustcrypto |
| 0.63x | Zen4 | AEAD | aes-256-gcm/decrypt vs rustcrypto | 32 | rustcrypto |
| 0.64x | s390x | Checksums | crc24-openpgp vs crc | 64 | crc |
| 0.65x | Grav4 | RapidHash | rapidhash-v3-128 vs rapidhash | 64 | rapidhash |
| 0.65x | ICL | AEAD | aes-256-gcm/encrypt vs rustcrypto | 64 | rustcrypto |
| 0.66x | Zen5 | AEAD | xchacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.66x | Zen5 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 1 | rustcrypto |
| 0.66x | Zen5 | RapidHash | rapidhash-64 vs rapidhash | 32 | rapidhash |
| 0.66x | ICL | AEAD | chacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.67x | SPR | AEAD | chacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.67x | Zen5 | AEAD | xchacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.67x | Grav4 | RapidHash | rapidhash-v3-128 vs rapidhash | 32 | rapidhash |
| 0.67x | Grav4 | AEAD | aegis-256/encrypt vs aegis-crate | 0 | aegis-crate |
| 0.68x | SPR | XXH3 | xxh3-128 vs xxhash-rust | 256 | xxhash-rust |
| 0.68x | Zen4 | AEAD | chacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.68x | ICL | AEAD | aes-256-gcm/decrypt vs rustcrypto | 32 | rustcrypto |
| 0.68x | SPR | AEAD | xchacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.69x | ICL | AEAD | chacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.69x | s390x | Checksums | crc16-ccitt vs crc | 1 | crc |
| 0.69x | ICL | AEAD | xchacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.70x | SPR | AEAD | xchacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.71x | POWER10 | RapidHash | rapidhash-128 vs rapidhash | 1 | rapidhash |
| 0.71x | Zen5 | XXH3 | xxh3-64 vs xxhash-rust | 1024 | xxhash-rust |
| 0.71x | ICL | AEAD | xchacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.72x | Zen4 | AEAD | xchacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.72x | SPR | AEAD | chacha20-poly1305/encrypt vs rustcrypto | 256 | rustcrypto |
| 0.73x | SPR | Auth | hmac-sha256 vs rustcrypto | 1024 | rustcrypto |
| 0.73x | s390x | BLAKE3 | blake3/xof vs blake3 | 1024 | blake3 |
| 0.73x | Zen5 | XXH3 | xxh3-128 vs xxhash-rust | 256 | xxhash-rust |
| 0.73x | SPR | AEAD | aes-256-gcm/encrypt vs rustcrypto | 64 | rustcrypto |
| 0.73x | ICL | Checksums | crc32 vs crc32fast | 256 | crc32fast |
| 0.74x | SPR | AEAD | aes-256-gcm/decrypt vs rustcrypto | 0 | rustcrypto |
| 0.75x | Zen4 | AEAD | chacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.75x | s390x | BLAKE3 | blake3/keyed vs blake3 | 1024 | blake3 |
| 0.75x | SPR | Auth | hmac-sha256 vs rustcrypto | 4096 | rustcrypto |
| 0.75x | s390x | BLAKE3 | blake3 vs blake3 | 1024 | blake3 |
| 0.75x | Zen5 | AEAD | aes-256-gcm/encrypt vs rustcrypto | 64 | rustcrypto |
| 0.75x | Zen4 | XXH3 | xxh3-64 vs xxhash-rust | 256 | xxhash-rust |
| 0.76x | Grav4 | AEAD | aegis-256/encrypt vs aegis-crate | 32 | aegis-crate |
| 0.76x | macOS-A64 | XXH3 | xxh3-64 vs xxhash-rust | 32 | xxhash-rust |
| 0.76x | POWER10 | RapidHash | rapidhash-v3-128 vs rapidhash | 32 | rapidhash |
| 0.76x | macOS-A64 | SHAKE | shake128 vs sha3 | 65536 | sha3 |
| 0.76x | s390x | RapidHash | rapidhash-v3-64 vs rapidhash | 0 | rapidhash |
| 0.76x | Grav3 | AEAD | aegis-256/encrypt vs aegis-crate | 0 | aegis-crate |
| 0.77x | s390x | Checksums | crc64-nvme vs crc64fast-nvme | 1 | crc64fast-nvme |
| 0.77x | Zen4 | AEAD | xchacha20-poly1305/decrypt vs rustcrypto | 256 | rustcrypto |
| 0.78x | SPR | AEAD | aes-256-gcm/decrypt vs rustcrypto | 1 | rustcrypto |
| 0.78x | s390x | Checksums | crc16-ibm vs crc | 1 | crc |
| 0.78x | macOS-A64 | BLAKE2 | blake2/host-stream-overhead/blake2s256 vs rustcrypto | 64 | rustcrypto |
| 0.78x | Grav4 | AEAD | aegis-256/encrypt vs aegis-crate | 1 | aegis-crate |
| 0.79x | POWER10 | Checksums | crc64-xz vs crc64fast | 64 | crc64fast |
| 0.79x | POWER10 | Checksums | crc64-nvme vs crc64fast-nvme | 64 | crc64fast-nvme |
| 0.79x | Grav3 | AEAD | aegis-256/encrypt vs aegis-crate | 32 | aegis-crate |
| 0.79x | Zen5 | BLAKE3 | blake3/keyed vs blake3 | 0 | blake3 |

Showing worst 80 of 327 losses over 5 percent.

## BLAKE2 Detail

| Group | W | T | L | Total | Win % | Worst | Platform | Case |
|---|---|---|---|---|---|---|---|---|
| blake2/blake2b256 vs rustcrypto | 45 | 53 | 1 | 99 | 45% | 0.90x | macOS-A64 | 1 |
| blake2/blake2b512 vs rustcrypto | 36 | 63 | 0 | 99 | 36% | 0.96x | Zen4 | 1048576 |
| blake2/blake2s128 vs rustcrypto | 56 | 43 | 0 | 99 | 57% | 0.98x | macOS-A64 | 0 |
| blake2/blake2s256 vs rustcrypto | 48 | 51 | 0 | 99 | 48% | 0.95x | Zen5 | 1 |
| blake2/host-keyed-overhead/blake2b256 vs rustcrypto | 1 | 5 | 0 | 6 | 17% | 0.97x | macOS-A64 | 16 |
| blake2/host-keyed-overhead/blake2s256 vs rustcrypto | 6 | 0 | 0 | 6 | 100% | 1.03x | macOS-A64 | 128 |
| blake2/host-overhead/blake2b256 vs rustcrypto | 1 | 5 | 0 | 6 | 17% | 0.99x | macOS-A64 | 1 |
| blake2/host-overhead/blake2s256 vs rustcrypto | 1 | 5 | 0 | 6 | 17% | 0.97x | macOS-A64 | 0 |
| blake2/host-stream-overhead/blake2b256 vs rustcrypto | 0 | 1 | 5 | 6 | 0% | 0.89x | macOS-A64 | 32 |
| blake2/host-stream-overhead/blake2s256 vs rustcrypto | 0 | 0 | 6 | 6 | 0% | 0.78x | macOS-A64 | 64 |
| blake2/keyed/blake2b256 vs rustcrypto | 48 | 48 | 3 | 99 | 48% | 0.92x | POWER10 | 32 |
| blake2/keyed/blake2b512 vs rustcrypto | 53 | 43 | 3 | 99 | 54% | 0.92x | POWER10 | 32 |
| blake2/keyed/blake2s128 vs rustcrypto | 55 | 43 | 1 | 99 | 56% | 0.94x | POWER10 | 1 |
| blake2/keyed/blake2s256 vs rustcrypto | 55 | 44 | 0 | 99 | 56% | 0.95x | POWER10 | 1 |
| blake2/streaming/blake2b256 vs rustcrypto | 9 | 18 | 0 | 27 | 33% | 0.95x | POWER10 | 64B |
| blake2/streaming/blake2s256 vs rustcrypto | 12 | 15 | 0 | 27 | 44% | 0.98x | Zen5 | 65536B |

## Raw Results

- `benchmark_results/2026-04-25/linux/amd-zen4/results.txt`
- `benchmark_results/2026-04-25/linux/amd-zen5/results.txt`
- `benchmark_results/2026-04-25/linux/intel-spr/results.txt`
- `benchmark_results/2026-04-25/linux/intel-icl/results.txt`
- `benchmark_results/2026-04-25/linux/graviton3/results.txt`
- `benchmark_results/2026-04-25/linux/graviton4/results.txt`
- `benchmark_results/2026-04-25/linux/ibm-s390x/results.txt`
- `benchmark_results/2026-04-25/linux/ibm-power10/results.txt`
- `benchmark_results/2026-04-26/macos/aarch64/results.txt`
