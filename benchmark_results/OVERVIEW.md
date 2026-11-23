# Benchmark overview

This file holds the current benchmark record. A ratio divides the fastest external library's median
time by the `rscrypto` median time for the same work, so a ratio above 1.00x favors `rscrypto`. A
tie is 0.95x through 1.05x. [Benchmarking](../docs/benchmarking.md) gives the method, the comparison
rules, and the commands. Earlier campaigns remain in the
[history of this file](https://github.com/loadingalias/rscrypto/blob/50c708b29905bb3221e7b22b9c7cce8fcc700818/benchmark_results/OVERVIEW.md).

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

## Not in this record

- TLS 1.3 and QUIC through `rscrypto-rustls`, and the OpenSSL comparisons: these targets need
  OpenSSL 4.0.3, which CI does not provision. The Apple run does not include them. Their last
  full run is the [v0.12.0
  record](https://github.com/loadingalias/rscrypto/blob/50c708b29905bb3221e7b22b9c7cce8fcc700818/benchmark_results/OVERVIEW.md#2026-10-23-v0120-release-benchmark-run).
- The BLAKE3 file readers, and the diagnostic targets: `aead_kernels`, `aead_diag`,
  `mldsa_portable`, `slhdsa_portable`, and `structural`.
