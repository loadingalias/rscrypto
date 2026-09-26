# ML-DSA implementation

The `ml-dsa` feature provides original Rust implementations of ML-DSA-44,
ML-DSA-65, and ML-DSA-87. It enables `sha3` and does not require `alloc`, `std`,
OS entropy, or an external implementation. `signatures`, `auth`, and `full`
include it. RustCrypto ML-DSA 0.1.1 is pinned only as a development oracle.

**The portable implementation and local baseline are complete.** Target qualification
continues with macOS AArch64, Linux x86-64/AArch64/IBM Z/POWER, and RISC-V kernels. The evidence
and limits below do not assert universal performance leadership or whole-operation
constant time.

## Standard and provenance

The implementation derives its arithmetic and encodings from
[FIPS 204](https://csrc.nist.gov/pubs/fips/204/final), final 2024-08-13.
The [potential corrections](https://csrc.nist.gov/files/pubs/fips/204/final/docs/fips-204-potential-updates.xlsx),
updated 2026-07-31, were checked on 2026-09-21. NIST labels them potential
corrections, not an amended final publication.

| Artifact | SHA-256 |
| --- | --- |
| Final FIPS 204 PDF | `57239b9f84c03227eda3ca0991204dc7764c79af9ce2e6824eda774918d46b6b` |
| Potential corrections spreadsheet | `5bc93ce63bc647e6d1d456cb2d3a171426c15aca4a7a0e0edd40d08b7a34c793` |

The signing cap is 821 attempts, incorporating the correction from 814.
Challenge hashing uses `mu || w1`. Hint subtraction wraps to the high-part
modulus minus one. The implementation uses its own unsigned Montgomery
representation with multiplication operands below 2q; it does not implement the
spreadsheet's signed-representative pseudocode. Its reduction bound is proved
at the multiplication routine. Roots are generated from the standard's 1753
root and eight-bit index reversal, not imported from an implementation table.

All 615 retained ACVP cases and their provenance are described in
[the vector manifest](../testdata/mldsa/acvp/README.md). No third-party
implementation source is copied, translated, or bundled into the primitive.

## API contract

Each parameter set has distinct public-key, secret-key, signature, prepared
secret-key, and prepared public-key types, plus caller-owned storage for each
prepared key. Raw FIPS encodings have these sizes:

| Parameter set | Public key | Expanded secret key | Signature |
| --- | ---: | ---: | ---: |
| ML-DSA-44 | 1312 | 2560 | 2420 |
| ML-DSA-65 | 1952 | 4032 | 3309 |
| ML-DSA-87 | 2592 | 4896 | 4627 |

`keypair_from_seed` borrows a 32-byte seed. Callers can retain it in
`SecretBytes<32>`; the key does not retain a second seed. Expanded import checks
noise coefficient ranges, reconstructs the public key, and verifies the
redundant low polynomial and public-key hash. The signing seed K has no
independent consistency check in the expanded format. Public-key encodings
accept every bit pattern of the prescribed length.

Signatures require canonical hints, zero unused hint bytes, and an admissible
response norm. Structural parsing is separate from authentication. Verification
returns the existing opaque `VerificationError`, including for an oversized
context. Contexts are byte strings of at most 255 bytes.

`sign_deterministic` supplies the standard's all-zero randomness. `sign_with`
requests exactly 32 bytes of caller-provided cryptographic randomness.
`try_sign` and `try_generate_keypair` use OS entropy only with `getrandom`.
Context validation precedes entropy acquisition. A callback error publishes no
signature or key; existing keys remain usable. Callback buffers have zeroizing
owners before the callback begins. Successful callbacks must fill every byte.

`MlDsaPrehash` binds a borrowed digest to its algorithm and checks its length.
The caller computes the digest over the original message. All twelve FIPS 204
hash identifiers are supported: SHA-224/256/384/512, SHA-512/224, SHA-512/256,
SHA3-224/256/384/512, SHAKE128 with 32 output bytes, and SHAKE256 with 64.
The prehash algorithm identifier, digest, context, and pure/prehash domain are
authenticated. The hash's collision strength can limit the chosen parameter
set: 224-, 256-, 384-, and 512-bit digests supply at most 112, 128, 192, and
256 bits respectively; the supported SHAKE outputs supply 128 and 256 bits.

The `Verifier` trait uses pure ML-DSA with an empty context. Signing methods
keep randomness and context explicit. No implicit `Signer` mode is selected.
The public API does not accept an unbound external message representative;
the private ACVP harness exercises the standard's internal interface.

Public keys and signatures support `serde`. Secret serialization additionally
requires `serde-secrets`. Deserialization uses the same strict import checks,
rejects extra bytes, and guards partial secret sequence reads. Secret owners
are neither `Clone` nor `Copy`, have redacted `Debug`, and export explicitly
into `SecretBytes`. A prepared secret key borrows its originating key and its
storage, and clears the transformed secret polynomials when dropped; a failed
preparation clears them before returning. The secret storage clears them again
when it is dropped.

## Work and memory

Operations allocate no heap memory and do not copy messages into an intermediate
message-sized buffer. Pure signing absorbs the borrowed message; prehash callers
can hash incrementally before signing. The compact path expands matrix rows as
needed. `prepare(&mut storage)` fills caller-owned storage with the complete
matrix and transformed key for repeated work; it does not change signature
bytes or validation. The caller chooses where storage lives: stack, static, or
heap. `new()` is a `const fn`, so storage can also be a static initializer.
The module's rustdoc example shows the flow. Preparation writes the storage in
place and returns a handle that borrows the key and the storage. Storage can be
reused after its handle drops.

| Parameter set | Prepared secret-key storage | Prepared public-key storage |
| --- | ---: | ---: |
| ML-DSA-44 | 28,672 B | 20,544 B |
| ML-DSA-65 | 48,128 B | 36,928 B |
| ML-DSA-87 | 80,896 B | 65,600 B |

Measured whole-call stack bounds, in bytes, excluding the caller's own storage:

| Operation | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
| --- | ---: | ---: | ---: |
| Key generation | 24,976 | 34,688 | 42,000 |
| Expanded secret-key import | 27,024 | 36,384 | 45,536 |
| Compact signing (pure, hedged, prehash) | 37,488 | 48,480 | 63,408 |
| Prepared signing | 25,088 | 30,960 | 39,744 |
| Compact verification (pure, prehash) | 12,496 | 13,504 | 15,568 |
| Prepared verification | 12,304 | 13,312 | 15,376 |
| Secret-key `prepare` | 1,504 | 1,504 | 1,552 |
| Public-key `prepare` | 4,000 | 4,000 | 4,000 |
| Storage `new` | 320 | 320 | 320 |

Each value is the largest static bound across native and `portable-only`
release builds (fat LTO, one codegen unit) for s390x, POWER, RV64, x86-64, and
AArch64 Linux GNU, and for RV32, Thumb, x86-64, and AArch64 bare metal, built
with the pinned nightly compiler. Values include the returned key or
signature. They exclude one-time runtime capability detection, which adds up to
4.4 KiB on first use in Linux native builds. Linux values also exclude libc
memory routines. QEMU stack painting on RV32 and Cortex-M stays within every
static bound. Other compilers, profiles, and caller inlining can change these
values; they do not bound Windows, macOS, or WASM builds. Compact signing
decodes the secret polynomials on the stack; prepared signing reads them from
storage. A core-only build does not establish that signing fits a particular
microcontroller.

Matrix sampling considers at most 298 candidates (894 SHAKE bytes). Secret-noise
sampling consumes all 481 bytes and compacts accepted coefficients with fixed
addresses. Unpublished-challenge sampling consumes all 221 bytes and uses fixed
scans. Verification uses direct sampling of its public challenge. Signing
tries at most 821 candidates, with at most 5747 mask nonces for ML-DSA-87.
Every exhausted sampler or signing loop returns `MlDsaError::RejectionLimit`.
There is no fallback signature or partially returned private output.

## Security evidence and qualification limits

The existing [threat model](../THREAT_MODEL.md) remains unchanged. This work
does not make a new whole-operation constant-time claim. Polynomial arithmetic
uses fixed coefficient traversals. Signing rejects aggregate norms; candidate
hints are packed only after every rejection check succeeds. Secret polynomial,
SHAKE, byte-buffer, and rejected-output owners use existing volatile cleanup.
After each private SHAKE256 call, a fixed 2 KiB volatile scrub clears the dead
stack below the hashing helper. It reaches compiler-created Keccak spill slots
only when the linked frames fit within that bound, so each target needs frame
review. These source properties require target-specific compiler and timing evidence.

Target qualification must preserve these boundaries:

1. Qualify secret-sensitive arithmetic and fixed-work samplers in the final
   artifact. Standard rejection sampling returns the first accepted candidate.
   Whole signing therefore has variable execution time and is not claimed as
   strict constant-time under the CT policy. Matching an external library does
   not prove side-channel resistance. Machine-code review and native timing
   evidence must identify the operations, compiler, and target they cover.
   `ct.toml` keeps whole ML-DSA signing best-effort and separately requires
   evidence for its secret-sensitive kernels. Nineteen timing cases cover
   forward/inverse transforms, products, accumulation, norm checks, rounding,
   secret samplers, and valid-key preparation across all parameter sets.
   Separate inverse-NTT and Montgomery cases execute portable arithmetic.
   Retained C ABI roots support linked-code and call-closure review. The bounded
   BINSEC root covers only the portable Montgomery leaf, not complete transforms,
   accelerated kernels, samplers, or signing. Registration defines required
   evidence; each target still needs passing results for the exact candidate.
2. Reduce signing stack for constrained targets and collect device evidence
   for the measured bounds. Prepared storage is written in place and its secret
   polynomials are cleared on handle drop, failed preparation, and storage drop.
   Compiler-created move, register, and spill copies remain outside the general
   cleanup claim.
3. Exercise bounded-sampler and signing exhaustion through production paths.
   Extend fuzz/corpus coverage and execute the remaining native/device targets.
   The initial native/portable suites, feature/MSRV matrix, packaging consumers,
   and Wasmtime scalar/SIMD128 vector lanes pass; they do not qualify every target.
4. Complete comparable prepared and unprepared workloads, representative seeds
   and signing tails, pure/prehash modes, and portable-only measurements. Retain
   exact source, compiler, CPU, statistics, and profile artifacts. One repeated
   deterministic signature does not characterize signing latency distributions.

The macOS and Linux AArch64 builds use original NEON forward and inverse NTTs
and matrix-product accumulation when compile-time NEON is enabled and
`portable-only` is absent. Linux x86-64 builds without `portable-only` select
original AVX2 arithmetic after cached CPU and OS capability detection.
Linux IBM Z and little-endian POWER builds without `portable-only` select
original z/Vector or POWER8 vector transforms, products, and accumulation after
cached CPU and OS capability detection. They share one four-lane transform
schedule. The widening products and masked reductions use original register-only
assembly to prevent compiler scalarization and coefficient-dependent reduction branches.
RISC-V32 and RISC-V64 builds with compile-time M support and without `portable-only`
use original scalar Montgomery assembly within the common scalar transform
schedule. These kernels require no vector extension. Register barriers protect
scalar reduction and selection on IBM Z and RISC-V, including `portable-only`.
Other configurations retain portable arithmetic. All paths use the same canonical
coefficients and generated roots.
The Linux NEON inverse stage stays out of line to avoid coefficient spills found
in the initial optimized artifact. This is a compiler-specific mitigation, not
a general register or stack-cleanup guarantee.
The portable implementation defines the semantics and remains the fallback.
Architecture qualification and comparative performance work remain open.
On macOS AArch64 without `portable-only`, public matrix expansion batches
adjacent columns through the existing paired SHAKE backend. Other targets and
`portable-only` retain scalar expansion. The paired path uses two public
polynomial buffers per row instead of one; prepared operations retain the same
key-owner sizes. Existing SHAKE dispatch is reused.
`portable-only` suppresses runtime capability selection, but a build that enables
SHA-3 instructions at compile time can still use them in shared SHAKE. Benchmark
claims must therefore record compiler target features as well as Cargo features.
