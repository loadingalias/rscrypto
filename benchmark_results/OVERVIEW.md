# Benchmark overview

This file holds the current benchmark record. A ratio divides the fastest external library's median
time by the `rscrypto` median time for the same work, so a ratio above 1.00x favors `rscrypto`. A
tie is 0.95x through 1.05x. [Benchmarking](../docs/benchmarking.md) gives the method, the comparison
rules, and the commands. Earlier campaigns remain in the
[history of this file](https://github.com/loadingalias/rscrypto/blob/50c708b29905bb3221e7b22b9c7cce8fcc700818/benchmark_results/OVERVIEW.md).

## November 25, 2026 BLAKE3 comparison

On Graviton4, rscrypto led upstream `blake3` 1.8.7 with a **1.072x geometric mean** across 26 single-threaded comparisons.
Both libraries computed one-shot hashes with and without a key at 13 sizes from 0 B through 1 MiB.
rscrypto led in 10 cases, tied in 12, and trailed in 4. It led in both modes at every measured size from 4 KiB through 1 MiB.

The times below are medians in microseconds. Graviton4 used revision `1b950e04`, which includes the SVE2 kernel. Graviton5 and M1 Pro used the v0.13 release source.

| Host | Operation | 4 KiB rscrypto | 4 KiB upstream | 64 KiB rscrypto | 64 KiB upstream |
| --- | --- | ---: | ---: | ---: | ---: |
| Graviton4 | Hash | 2.32 | 2.87 | 34.24 | 44.42 |
| Graviton4 | Keyed hash | 2.37 | 2.87 | 34.87 | 44.42 |
| Graviton5 | Hash | 2.26 | 2.41 | 33.61 | 37.87 |
| Graviton5 | Keyed hash | 2.29 | 2.42 | 34.25 | 37.86 |
| Apple M1 Pro | Hash | 2.50 | 2.42 | 38.93 | 37.91 |
| Apple M1 Pro | Keyed hash | 2.59 | 2.46 | 45.02 | 49.63 |

<details>
<summary>All 26 Graviton4 comparisons and the reproduction command</summary>

Each ratio divides upstream time by rscrypto time. The tie band is 0.95x through 1.05x.

| Input | Hash ratio | Keyed hash ratio |
| --- | ---: | ---: |
| 0 B | 1.028x | 0.968x |
| 1 B | 0.978x | 0.925x |
| 32 B | 0.999x | 0.944x |
| 64 B | 1.046x | 0.976x |
| 65 B | 0.988x | 0.970x |
| 256 B | 0.989x | 0.977x |
| 1 KiB | 0.974x | 0.968x |
| 1,025 B | 0.948x | 0.875x |
| 4 KiB | 1.237x | 1.211x |
| 16 KiB | 1.285x | 1.277x |
| 64 KiB | 1.297x | 1.274x |
| 256 KiB | 1.243x | 1.234x |
| 1 MiB | 1.251x | 1.241x |

Run this command three times on Graviton4 with revision `1b950e04`.

```bash
just bench blake3_serial 'filter=^blake3/serial/(hash|keyed)/(rscrypto|blake3)/[0-9]+$'
```

</details>

One four-vCPU i8g.xlarge ran the Graviton4 suite three times. Each Graviton4 time is the median of the three run medians.
Each run used 20 samples per case, 100 ms of warm-up, and a 400 ms measurement window.
The three family scores ranged from 1.071x to 1.072x. All 156 output checks passed.
Run `20261125T222010Z-90457` retains the plans and raw samples. [Benchmark method](../docs/benchmarking.md#scorecard).

The v0.13 release led upstream on POWER10 (1.48x) and IBM Z (1.34x). Upstream led on Intel Windows (0.92x).
The other six release hosts fell within the tie band, including Graviton5 (1.02x) and Apple M1 Pro (0.96x).
[Release scorecard](https://github.com/loadingalias/rscrypto/actions/runs/44681056190/artifacts/14621750310) · [Release results by host](#v013-release-source).

Against rscrypto v0.13.0, the SVE2 kernel cut Graviton4 unkeyed hash latency by 13% at 4 KiB and 14% at 64 KiB.
[Sealed-record measurements](#november-25-2026-sealed-records).

## November 25, 2026 sealed records

On Graviton4, hashing 64 KiB took 39.73–39.75 µs at p50 with rscrypto 0.13.0.
The four-chunk SVE2 kernel took 34.17–34.19 µs. At 4 KiB, the corresponding medians were 2.71 and 2.36 µs.
The compiler-generated XOR/rotate variant took 43.74–43.77 µs at 64 KiB.
On Apple M1 Pro, register transposition and lane-load experiments did not improve both record sizes.
Graviton4 run `20261125T211643Z-16580` passed 28 BLAKE3 unit tests using the final kernel.

Each cell gives p50 / p99 in microseconds for one plaintext copy, AEGIS-256 seal, and whole-ciphertext BLAKE3 root.
The Intel and Apple rows use the release control. Graviton4 uses the SVE2 kernel.
Each reported run used 2,001 warmups and 2,001 samples.

| Host | 1 KiB | 4 KiB | 64 KiB |
| --- | ---: | ---: | ---: |
| Xeon Platinum 8559C | 1.11 / 1.13 | 1.68 / 1.82 | 19.50 / 27.42 |
| Graviton4 | 1.62 / 1.65 | 3.03 / 3.04 | 44.19 / 46.98 |
| Apple M1 Pro | 1.46 / 1.54 | 3.13 / 3.33 | 46.58 / 58.79 |

On Graviton4, two independent 64 KiB records took 88.64 µs serially and 55.55 µs with a persistent worker.
The worker retained each record's ciphertext, detached tag, and root.
For one 64 KiB root spanning two existing seals, the worker took 46.29 µs. The serial operation took 44.51 µs.
On M1 Pro, a streaming AEGIS-256 reference also reproduced rscrypto's one-shot ciphertext, tag, and root with one nonce at all three sizes.
Its 64 KiB serial median was 42.6–42.7 µs. Overlapping the first half's hash with the second half's seal took 47.75–48.67 µs.
These rows measure crypto work in memory. The storage receipt and durability-class floor belong to Via.
[Workloads and reproduction](../docs/benchmarking.md#sealed-ciphertext-records).

## November 24, 2026 performance measurements

### TLS 1.3 and QUIC

Each cell gives the median time in microseconds for rscrypto and the fastest external peer. Lower times are better.
Each measurement uses AES-128-GCM. The handshake rows use P-256 credentials.
The TLS endpoints authenticate each other. The QUIC handshake authenticates the server.

| Workload | Intel Linux | Graviton4 | Apple M1 Pro |
| --- | ---: | ---: | ---: |
| TLS full, X25519 | 292 / 327 (AWS-LC) | 294 / 369 (AWS-LC) | 250 / 304 (AWS-LC) |
| TLS resumed, X25519 | 86.0 / 93.1 (AWS-LC) | 104 / 119 (AWS-LC) | 94.9 / 102 (AWS-LC) |
| TLS full, X25519MLKEM768 | 342 / 373 (AWS-LC) | 356 / 418 (AWS-LC) | 289 / 337 (AWS-LC) |
| TLS resumed, X25519MLKEM768 | 134 / 141 (AWS-LC) | 167 / 169 (AWS-LC) | 137 / 138 (AWS-LC) |
| TLS record seal, 64 B | 0.080 / 0.109 (AWS-LC) | 0.179 / 0.137 (AWS-LC) | 0.174 / 0.094 (AWS-LC) |
| TLS record open, 64 B | 0.074 / 0.103 (ring) | 0.126 / 0.137 (ring) | 0.077 / 0.095 (ring) |
| TLS record seal, 16 KiB | 0.932 / 0.987 (AWS-LC) | 2.34 / 2.41 (AWS-LC) | 2.01 / 2.14 (AWS-LC) |
| TLS record open, 16 KiB | 1.02 / 1.00 (AWS-LC) | 2.48 / 2.41 (AWS-LC) | 2.07 / 2.16 (AWS-LC) |
| TLS application transfer, 1 MiB | 248 / 251 (OpenSSL) | 447 / 415 (OpenSSL) | 426 / 417 (AWS-LC) |
| QUIC full handshake, X25519 | 203 / 212 (AWS-LC) | 214 / 246 (AWS-LC) | 198 / 202 (AWS-LC) |
| QUIC packet seal, 1,200 B | 0.151 / 0.182 (AWS-LC) | 0.299 / 0.305 (AWS-LC) | 0.251 / 0.263 (AWS-LC) |
| QUIC packet open, 1,200 B | 0.147 / 0.180 (ring) | 0.290 / 0.296 (AWS-LC) | 0.233 / 0.298 (AWS-LC) |

The Intel hybrid handshake took 342 µs with rscrypto, 373 µs with AWS-LC and 660 µs with OpenSSL 4.0.3.
On Graviton4, the same handshake took 356 µs, 418 µs and 729 µs, respectively.
The historical Intel result was 392 µs with rscrypto and 367 µs with AWS-LC.
Graviton4 previously measured 392 µs and 418 µs.
The two campaigns used different builds and hosts.
[Historical provider measurements](https://github.com/loadingalias/rscrypto/blob/50c708b29905bb3221e7b22b9c7cce8fcc700818/benchmark_results/OVERVIEW.md#tls-13-and-quic-through-rscrypto-rustls).

Graviton4 sealed a 64-byte TLS record in 179 ns. AWS-LC took 137 ns.
OpenSSL transferred 1 MiB on Graviton4 in 415 µs. The rscrypto provider took 447 µs.

### IBM Z ECDH

The native runners reported IBM machine type 8562. The ratio divides the fastest external peer’s median time by the rscrypto median time.
The headline includes scalar import, public derivation and parsed-peer agreement for each curve.

| Operation | Release rscrypto | Current rscrypto | Fastest peer | Peer / rscrypto |
| --- | ---: | ---: | ---: | ---: |
| P-256 scalar import | 31.3 ns | 30.0 ns | 51.2 ns (RustCrypto) | 1.70x |
| P-256 public derivation | 1.966 ms | 46.1 µs | 92.7 µs (crrl) | 2.01x |
| P-256 agreement, parsed peer | 10.757 ms | 38.7 µs | 176 µs (crrl) | 4.54x |
| P-256 agreement, SEC1 bytes | 10.761 ms | 52.5 µs | 178 µs (crrl) | 3.38x |
| P-384 scalar import | 29.7 ns | 30.6 ns | 60.0 ns (RustCrypto) | 1.96x |
| P-384 public derivation | 9.856 ms | 96.6 µs | 449 µs (RustCrypto) | 4.64x |
| P-384 agreement, parsed peer | 25.507 ms | 77.1 µs | 1,902 µs (RustCrypto) | 24.66x |
| P-384 agreement, SEC1 bytes | 25.417 ms | 134 µs | 1,771 µs (RustCrypto) | 13.25x |

The six-point family geometric mean was 3.89x. The release recorded 0.13x.
The P-256 and P-384 fixed-versus-random scalar timing tests passed for public derivation and agreement.
[Native timing evidence](https://github.com/loadingalias/rscrypto/actions/runs/45317719955).

| IBM Z mutual TLS, P-256 credentials | Full handshake | Resumed handshake |
| --- | ---: | ---: |
| secp256r1, µs | 10,446 / 7,122 (AWS-LC) | 299 / 2,461 (ring) |
| secp384r1, µs | 11,351 / 12,279 (AWS-LC) | 492 / 7,773 (AWS-LC) |

### Ed25519 on Intel Linux

Each cell gives the rscrypto and fastest external peer times in microseconds. The control used the same Intel host.
The reusable-key workload prepares each public key before timing. The import workload includes decoding the public key.

| Message | Sign | Verify with a reusable key | Import and verify |
| --- | ---: | ---: | ---: |
| 0 B | 6.95 / 7.24 (AWS-LC) | 17.8 / 30.6 (AWS-LC) | 20.9 / 30.2 (dryoc) |
| 32 B | 6.95 / 7.24 (AWS-LC) | 17.7 / 30.6 (AWS-LC) | 20.7 / 29.8 (dryoc) |
| 1,024 B | 9.80 / 10.0 (AWS-LC) | 19.0 / 32.0 (AWS-LC) | 22.1 / 31.2 (dryoc) |
| 16,384 B | 53.2 / 51.0 (AWS-LC) | 40.8 / 52.5 (AWS-LC) | 43.9 / 52.6 (AWS-LC) |

The 12-point family geometric mean was 1.30x. The release recorded 0.88x.
For a 32-byte message on Intel, reusable-key verification went from 35.3 to 17.7 µs. Import and verify went from 40.1 to 20.7 µs.
For a 32-byte message on M1 Pro, reusable-key verification went from 25.6 to 25.7 µs. Import and verify went from 30.2 to 28.7 µs.

### Measurement scope

Each point is the median of three independent Criterion run medians. Each run used 20 samples, 100 ms of warm-up and a 400 ms measurement window.
The tie band is 0.95x through 1.05x. Each peer performed the same catalog operation and passed its output check.
The Intel c8i.xlarge used a Xeon 6975P-C. The c8g.xlarge used Graviton4. Both had four vCPUs.
Both AWS hosts built OpenSSL 4.0.3 from the release tarball with SHA-256 `325b5c806167c13b40b1ffeadfe0248197c00eccc4cf123ec1e28d2d2fd216d9`.
OpenSSL participated in the full and resumed TLS handshake rows and the 1 MiB transfer on AWS. Record primitives and QUIC compare rustls providers.
The Mac and IBM Z ran the rustls-only selection. The Apple M1 Pro was an active workstation.
Linux used the default target CPU. The Mac used `-C target-cpu=native`. All used `nightly-2026-10-24`.

The Intel, Graviton4 and M1 provider binaries used clean revision `be3a5a7d`. IBM Z repeated the same ECDH and provider code across the linked runs.
The Intel Ed25519 control used `1271cf58`. Its comparison build applied the public square and IFMA optimizations to `be3a5a7d`.
A later change removed the public square, because it changed the RISC-V signing machine code. No benchmark measures Ed25519 verification after that change.
The M1 Ed25519 candidate used the local `41cc59f9` checkout with the performance changes.
The M1 Ed25519 control reused the verified `42040818` executable. Its SHA-256 was `8a51630ce61a7b062f7b64727bc56fbfb955b3ad9870be99f8191c796d9d95b3`.

IBM Z benchmark runs retain the raw execution plans and Criterion samples. [First run](https://github.com/loadingalias/rscrypto/actions/runs/45316809839) · [Second run](https://github.com/loadingalias/rscrypto/actions/runs/45318238388) · [Third run](https://github.com/loadingalias/rscrypto/actions/runs/45319366428).

## v0.13 release source

Toolchain `nightly-2026-10-24`, the default target CPU, and the `.config/criterion.json` settings:
20 samples, 100 ms warm-up, and 400 ms measurement. Each row checks its output before timing. Each
point comes from one run on a cloud or partner host with no load or thermal checks, so a difference
of a few percent is not confirmed.

The Apple M1 Pro lane comes from one local `just bench all` at `42040818` on a MacBook Pro
(MacBookPro18,3). It uses the same toolchain and settings, with `-C target-cpu=native`, the local
macOS default. The host had no load or thermal checks, so its points have more noise than the CI
points.

Bench 44681056190 measured every primitive. Each later run replaced the points that a later change
can move, one for one.

| Bench run | Revision | Scope | Points kept |
| --- | --- | --- | ---: |
| [44681056190](https://github.com/loadingalias/rscrypto/actions/runs/44681056190) | `5d35d69a` | Every primitive on every lane | 12221 |
| [44776079627](https://github.com/loadingalias/rscrypto/actions/runs/44776079627) | `78cb3bd9` | AES-GCM-SIV on Windows | 96 |
| [44959988285](https://github.com/loadingalias/rscrypto/actions/runs/44959988285) | `1d0409e5` | X25519 and Ed25519 on x86-64 | 64 |
| [44979516803](https://github.com/loadingalias/rscrypto/actions/runs/44979516803) | `50c708b2` | ChaCha20-Poly1305, XChaCha20-Poly1305, Poly1305, AES-GCM, ECDSA, ML-KEM, X-Wing, and HPKE | 1694 |
| [44984909542](https://github.com/loadingalias/rscrypto/actions/runs/44984909542) | `50c708b2` | Composite ML-DSA-65 | 56 |

The measured source is the same from `1d0409e5` to `42040818`. The commits after `5d35d69a` that
no run repeats do not change the code that these hosts run, with one exception: SLH-DSA key
generation now fills its key in place (`9ca32923`), which moves where a 64- to 128-byte key is
written.

The headline holds each primitive's primary operations. The other points cover streaming, batch,
construction, and parsing.

| Lane | Points | Faster | Tie | Slower | Headline | Faster | Tie | Slower | Headline median | Headline geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intel Linux | 1784 | 981 | 492 | 311 | 1032 | 675 | 251 | 106 | 1.10x | 1.37x |
| AMD Linux | 1784 | 991 | 542 | 251 | 1032 | 676 | 290 | 66 | 1.13x | 1.39x |
| Intel Windows | 1784 | 1048 | 406 | 330 | 1032 | 728 | 204 | 100 | 1.19x | 1.46x |
| AMD Windows | 1784 | 1115 | 408 | 261 | 1032 | 774 | 190 | 68 | 1.15x | 1.48x |
| Graviton5 | 1750 | 1017 | 475 | 258 | 1032 | 650 | 300 | 82 | 1.10x | 1.35x |
| POWER10 | 1750 | 957 | 548 | 245 | 1032 | 530 | 404 | 98 | 1.05x | 1.48x |
| IBM Z | 1745 | 1270 | 226 | 249 | 1029 | 792 | 130 | 107 | 2.13x | 2.84x |
| RISC-V | 1750 | 842 | 511 | 397 | 1032 | 584 | 292 | 156 | 1.07x | 1.14x |
| Apple M1 Pro | 1750 | 765 | 544 | 441 | 1032 | 486 | 354 | 192 | 1.04x | 1.27x |

Headline geomean by family. The points column gives the headline points on each lane.

| Family | Points | Intel Linux | AMD Linux | Intel Windows | AMD Windows | Graviton5 | POWER10 | IBM Z | RISC-V | Apple M1 Pro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CRC | 77 | 7.38x | 6.22x | 7.31x | 6.21x | 5.84x | 13.62x | 11.85x | 1.60x | 5.78x |
| Adler-32 | 15 | 1.41x | 1.64x | 1.51x | 1.64x | 1.52x | 4.73x | 2.72x | 1.28x | 1.48x |
| XXH3 | 34 | 0.94x | 0.95x | 0.93x | 0.96x | 0.96x | 1.19x | 1.42x | 0.78x | 0.98x |
| XXH32 | 13 | 0.95x | 1.05x | 0.92x | 1.01x | 1.02x | 1.27x | 0.97x | 0.86x | 0.81x |
| XXH64 | 12 | 0.99x | 0.98x | 0.99x | 0.99x | 0.96x | 0.99x | 0.96x | 0.97x | 0.96x |
| RapidHash | 11 | 0.98x | 1.04x | 1.19x | 1.04x | 1.11x | 1.19x | 1.11x | 0.80x | 0.89x |
| SipHash | 45 | 1.05x | 1.12x | 1.04x | 1.09x | 1.08x | 1.16x | 0.99x | 1.12x | 1.25x |
| SHA-2 | 65 | 1.03x | 1.07x | 1.06x | 1.05x | 1.06x | 1.00x | 4.68x | 1.03x | 1.11x |
| SHA-3 | 52 | 1.11x | 1.08x | 1.15x | 1.15x | 1.10x | 1.59x | 6.79x | 1.37x | 1.05x |
| SHAKE | 26 | 1.08x | 1.05x | 1.14x | 1.11x | 1.09x | 0.97x | 3.60x | 1.13x | 1.02x |
| cSHAKE | 22 | 1.19x | 1.18x | 1.21x | 1.19x | 1.23x | 1.03x | 4.47x | 1.22x | 1.48x |
| BLAKE2 | 104 | 1.15x | 1.12x | 1.14x | 1.13x | 1.22x | 0.99x | 1.01x | 1.08x | 0.88x |
| BLAKE3 | 26 | 0.95x | 0.95x | 0.92x | 1.01x | 1.02x | 1.48x | 1.34x | 0.96x | 0.96x |
| Ascon | 22 | 1.06x | 1.03x | 1.05x | 1.06x | 1.04x | 1.01x | 1.00x | 1.02x | 1.06x |
| HMAC-SHA-2 | 33 | 1.03x | 1.03x | 0.97x | 1.07x | 1.03x | 0.97x | 4.97x | 1.03x | 1.00x |
| HMAC-SHA-3 | 44 | 1.04x | 1.01x | 1.22x | 1.21x | 1.06x | 1.34x | 10.71x | 1.17x | 1.04x |
| KMAC | 22 | 1.20x | 1.19x | 1.22x | 1.19x | 1.25x | 1.06x | 4.03x | 1.23x | 1.49x |
| Poly1305 | 7 | 1.74x | 1.76x | 1.88x | 1.79x | 0.92x | 1.46x | 2.02x | 1.50x | 1.28x |
| HKDF | 16 | 1.14x | 1.14x | 1.06x | 1.17x | 1.13x | 1.02x | 2.52x | 1.09x | 1.04x |
| PBKDF2 | 14 | 1.20x | 1.17x | 1.10x | 1.22x | 1.08x | 1.05x | 2.57x | 1.16x | 1.00x |
| Argon2 | 10 | 1.37x | 1.41x | 1.34x | 1.41x | 1.09x | 0.75x | 0.87x | 1.13x | 1.09x |
| scrypt | 5 | 0.96x | 0.98x | 0.93x | 1.00x | 1.14x | 1.17x | 1.41x | 1.60x | 1.13x |
| AEAD | 186 | 1.38x | 1.51x | 1.58x | 1.65x | 1.31x | 1.37x | 5.21x | 1.08x | 1.17x |
| X25519 | 2 | 1.02x | 1.02x | 1.42x | 1.69x | 1.01x | 0.58x | 0.87x | 0.86x | 1.00x |
| ECDH | 6 | 1.35x | 1.21x | 2.67x | 2.71x | 1.17x | 1.40x | 0.13x | 0.67x | 1.14x |
| ML-KEM | 15 | 1.59x | 1.61x | 1.89x | 1.62x | 1.43x | 1.11x | 1.55x | 1.20x | 1.07x |
| X-Wing | 5 | 2.23x | 2.48x | 2.31x | 2.50x | 2.57x | 1.29x | 1.47x | 1.77x | 2.32x |
| HPKE | 14 | 2.02x | 2.18x | 2.97x | 3.53x | 1.97x | 1.75x | 2.30x | 1.92x | 2.04x |
| Ed25519 | 12 | 0.88x | 0.98x | 0.97x | 1.16x | 0.98x | 0.79x | 1.05x | 0.95x | 0.99x |
| ECDSA | 24 | 1.31x | 1.29x | 2.15x | 2.22x | 1.54x | 2.10x | 0.58x | 1.36x | 1.48x |
| RSA | 28 | 1.12x | 1.27x | 1.24x | 1.39x | 1.03x | 1.05x | 2.02x | 0.94x | 1.14x |
| ML-DSA | 12–15 | 1.13x | 1.16x | 1.51x | 1.42x | 1.24x | 1.38x | 1.43x | 0.82x | 1.17x |
| Composite ML-DSA | 7 | 3.75x | 4.72x | 3.92x | 4.82x | 5.78x | 4.44x | 2.90x | 3.04x | 6.00x |
| SLH-DSA | 36 | 1.68x | 1.71x | 1.65x | 1.62x | 1.66x | 1.47x | 1.94x | 1.65x | 1.40x |
| ChaCha20Rng | 7 | 1.43x | 1.49x | 1.24x | 1.35x | 1.21x | 1.32x | 1.84x | 0.66x | 1.16x |

Since `dbc1afa5`, Graviton5 standalone Poly1305 is slower at short inputs: 9.0 to 11.0 ns at 0 B
and 9.5 to 15.9 ns at 16 B, while the external rows stayed within 1%. No other lane slowed.

QUIC-LB has no external implementation in Rust, so it has no ratio. Median time per operation, with
the cipher built once, from the base run and the Apple run:

| Lane | encrypt, 16 octets | decrypt, 16 octets | encrypt, 12 octets | decrypt, 12 octets | encrypt, 19 octets | decrypt, 19 octets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intel Linux | 6.5 ns | 8.9 ns | 67.7 ns | 59.3 ns | 67.8 ns | 59.1 ns |
| AMD Linux | 4.5 ns | 6.2 ns | 64.4 ns | 54.7 ns | 64.4 ns | 54.3 ns |
| Intel Windows | 6.5 ns | 8.8 ns | 63.8 ns | 55.8 ns | 63.8 ns | 55.9 ns |
| AMD Windows | 3.8 ns | 6.6 ns | 66.6 ns | 56.1 ns | 66.5 ns | 56.5 ns |
| Graviton5 | 14.1 ns | 16.4 ns | 55.4 ns | 49.2 ns | 55.4 ns | 49.7 ns |
| POWER10 | 27.0 ns | 34.6 ns | 258.2 ns | 233.6 ns | 258.1 ns | 233.4 ns |
| IBM Z | 43.0 ns | 53.8 ns | 482.9 ns | 461.5 ns | 459.7 ns | 435.8 ns |
| RISC-V | 1.46 µs | 1.94 µs | 6.15 µs | 4.77 µs | 6.15 µs | 4.77 µs |
| Apple M1 Pro | 9.5 ns | 15.6 ns | 66.1 ns | 54.8 ns | 66.4 ns | 55.9 ns |

### IBM Z ECDH operations

The six headline points cover private-scalar import and agreement with a parsed or encoded peer on each curve.
The public-key row measures derivation from an existing private scalar. It has a separate score.
All times are medians from the [release scorecard](https://github.com/loadingalias/rscrypto/actions/runs/44681056190/artifacts/14621750310).

| Curve and operation | rscrypto | Fastest peer | Peer / rscrypto |
| --- | ---: | ---: | ---: |
| P-256 private-scalar import | 31.3 ns | RustCrypto, 51.7 ns | 1.65x |
| P-256 public-key derivation | 1.966 ms | crrl, 71.7 µs | 0.036x |
| P-256 agreement, parsed peer | 10.757 ms | crrl, 169.8 µs | 0.016x |
| P-256 agreement, encoded peer | 10.761 ms | crrl, 170.8 µs | 0.016x |
| P-384 private-scalar import | 29.7 ns | RustCrypto, 58.8 ns | 1.98x |
| P-384 public-key derivation | 9.856 ms | RustCrypto, 523.6 µs | 0.053x |
| P-384 agreement, parsed peer | 25.507 ms | RustCrypto, 1.750 ms | 0.069x |
| P-384 agreement, encoded peer | 25.417 ms | RustCrypto, 1.764 ms | 0.069x |

## Not in this record

- The v0.13 TLS, QUIC and OpenSSL comparison targets required OpenSSL 4.0.3.
  The release CI and Apple campaigns omitted them. Their previous full measurements appear in the [v0.12.0
  record](https://github.com/loadingalias/rscrypto/blob/50c708b29905bb3221e7b22b9c7cce8fcc700818/benchmark_results/OVERVIEW.md#2026-10-23-v0120-release-benchmark-run).
- The BLAKE3 file readers, and the diagnostic targets: `aead_kernels`, `aead_diag`,
  `mldsa_portable`, `slhdsa_portable`, and `structural`.
