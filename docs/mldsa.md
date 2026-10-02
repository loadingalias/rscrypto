# ML-DSA implementation

The `ml-dsa` feature supplies original Rust implementations of ML-DSA-44, ML-DSA-65, and ML-DSA-87.
It enables `sha3`.
It does not need `alloc`, `std`, operating-system entropy, or an external implementation.
`signatures`, `auth`, and `full` include it.
RustCrypto ML-DSA 0.1.1 is pinned only as a development oracle.

**The portable implementation and the local baseline are complete.**
Target qualification continues for macOS AArch64, Linux x86-64, AArch64, IBM Z, and POWER,
and the RISC-V kernels.
The evidence and limits below do not claim universal performance leadership or constant time
for whole operations.

## Standard and provenance

The arithmetic and encodings come from [FIPS 204](https://csrc.nist.gov/pubs/fips/204/final), final, 2024-08-13.
The [potential corrections](https://csrc.nist.gov/files/pubs/fips/204/final/docs/fips-204-potential-updates.xlsx), updated 2026-07-31, were checked on 2026-09-21.
NIST calls them potential corrections, not an amended final publication.

| Artifact                          | SHA-256 |
| --------------------------------- | ------- |
| Final FIPS 204 PDF                | `57239b9f84c03227eda3ca0991204dc7764c79af9ce2e6824eda774918d46b6b` |
| Potential corrections spreadsheet | `5bc93ce63bc647e6d1d456cb2d3a171426c15aca4a7a0e0edd40d08b7a34c793` |

- The signing cap is 821 attempts.
  This includes the correction from 814.
- Challenge hashing uses `mu || w1`.
- Hint subtraction wraps to the high-part modulus minus one.
- The implementation uses its own unsigned Montgomery representation,
  with multiplication operands below 2q.
  It does not implement the spreadsheet's signed-representative pseudocode.
  The multiplication routine carries the proof of its reduction bound.
- The roots come from the standard's root 1753 and eight-bit index reversal.
  They are not imported from another implementation's table.

[The ACVP manifest](../testdata/mldsa/acvp/README.md) describes all 615 retained ACVP cases and their provenance.
[The Wycheproof manifest](../testdata/mldsa/wycheproof/README.md) describes all 1,138 pinned Wycheproof signing and verification cases.
No third-party implementation source is copied, translated, or bundled into the primitive.

## API contract

Each parameter set has its own types for the public key, secret key, signature, prepared secret key,
and prepared public key.
Each prepared key also has storage that the caller owns.
The raw FIPS encodings have these sizes, in bytes:

| Parameter set | Public key | Expanded secret key | Signature |
| ------------- | ---------: | ------------------: | --------: |
| ML-DSA-44     |       1312 |                2560 |      2420 |
| ML-DSA-65     |       1952 |                4032 |      3309 |
| ML-DSA-87     |       2592 |                4896 |      4627 |

### Keys

- `keypair_from_seed` borrows a 32-byte seed.
  Callers can keep the seed in `SecretBytes<32>`.
  The key does not keep a second copy of the seed.
- Expanded import checks the noise coefficient ranges, rebuilds the public key,
  and verifies the redundant low polynomial and the public-key hash.
- The expanded format has no independent consistency check for the signing seed K.
- Public-key decoding accepts every bit pattern of the correct length.

### Signatures and contexts

- A signature needs canonical hints, zero unused hint bytes, and an admissible response norm.
- Structural parsing is separate from authentication.
- Verification returns the opaque `VerificationError`, also for a context that is too long.
- A context is a byte string of at most 255 bytes.

### Randomness

- `sign_deterministic` uses the standard's all-zero randomness.
- `sign_with` asks the caller for exactly 32 bytes of cryptographic randomness.
- `try_sign` and `try_generate_keypair` use operating-system entropy, and only with `getrandom`.
- Context validation happens before entropy acquisition.
- If a callback fails, no signature or key is published, and existing keys stay usable.
- Callback buffers have zeroizing owners before the callback starts.
- A successful callback must fill every byte.

### Prehash mode

`MlDsaPrehash` binds a borrowed digest to its algorithm and checks the digest length.
The caller computes the digest over the original message.
All twelve FIPS 204 hash identifiers are supported: SHA-224/256/384/512, SHA-512/224, SHA-512/256,
SHA3-224/256/384/512, SHAKE128 with 32 output bytes, and SHAKE256 with 64 output bytes.
The signature authenticates the prehash algorithm identifier, the digest, the context,
and the pure or prehash domain.

The collision strength of the hash can limit the security of the parameter set.
Digests of 224, 256, 384, and 512 bits supply at most 112, 128, 192, and 256 bits.
The supported SHAKE outputs supply 128 and 256 bits.

### Traits and Serde

- The `Verifier` trait uses pure ML-DSA with an empty context.
- Signing methods keep randomness and context explicit.
  No implicit `Signer` mode exists.
- The public API does not accept an unbound external message representative.
  The private ACVP harness tests the standard's internal interface.
- Public keys and signatures support `serde`.
  Secret serialization also needs `serde-secrets`.
- Deserialization uses the same strict import checks, rejects extra bytes,
  and guards partial reads of secret sequences.
- Secret owners are not `Clone` or `Copy`, redact `Debug`, and export explicitly into `SecretBytes`.
- A prepared secret key borrows its source key and its storage.
  It clears the transformed secret polynomials on drop.
  A failed preparation clears them before it returns.
  The secret storage clears them again on drop.

## Work and memory

Operations allocate no heap memory, except the explicit `*_in` constructors below.
They do not copy messages into an intermediate buffer of message size.
Pure signing absorbs the borrowed message.
Prehash callers can hash incrementally before signing.

The compact path expands matrix rows when it needs them.
For repeated work, `prepare(&mut storage)` fills storage that the caller owns with the complete matrix
and the transformed key.
It does not change signature bytes or validation.
The caller selects where the storage lives: stack, static, or heap.
`new()` is a `const fn`, so a static initializer can also hold the storage.
The module's Rustdoc example shows the flow.
Preparation writes the storage in place and returns a handle that borrows the key and the storage.
The storage can be used again after its handle drops.

| Parameter set | Prepared secret-key storage | Prepared public-key storage |
| ------------- | --------------------------: | --------------------------: |
| ML-DSA-44     |                    28,672 B |                    20,544 B |
| ML-DSA-65     |                    48,128 B |                    36,928 B |
| ML-DSA-87     |                    80,896 B |                    65,600 B |

Measured stack bounds for a whole call, in bytes, without the caller's own storage:

| Operation                               | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
| --------------------------------------- | --------: | --------: | --------: |
| Key generation                          |    24,976 |    34,688 |    42,000 |
| Expanded secret-key import              |    27,024 |    36,384 |    45,536 |
| Compact signing (pure, hedged, prehash) |    37,488 |    48,480 |    63,408 |
| Prepared signing                        |    25,088 |    30,960 |    39,744 |
| Compact verification (pure, prehash)    |    12,496 |    13,504 |    15,568 |
| Prepared verification                   |    12,304 |    13,312 |    15,376 |
| Secret-key `prepare`                    |     1,504 |     1,504 |     1,552 |
| Public-key `prepare`                    |     4,000 |     4,000 |     4,000 |
| Storage `new`                           |       320 |       320 |       320 |

How these values were measured:

- Each value is the largest static bound across native and `portable-only` release builds
  (fat LTO, one codegen unit), built with the pinned nightly compiler, for:
  s390x, POWER, RV64, x86-64, and AArch64 Linux GNU;
  and RV32, Thumb, x86-64, and AArch64 bare metal.
- The values include the returned key or signature.
- They do not include one-time runtime capability detection,
  which adds up to 4.4 KiB on first use in Linux native builds.
  Linux values also do not include libc memory routines.
- QEMU stack painting on RV32 and Cortex-M stays within every static bound.
- Other compilers, profiles, and caller inlining can change these values.
  They do not bound Windows, macOS, or WASM builds.
- Compact signing decodes the secret polynomials on the stack.
  Prepared signing reads them from storage.
- A core-only build does not prove that signing fits a specific microcontroller.

### Secret keys in caller-selected memory

When a function returns a secret key by value, the bytes move,
and a Rust move leaves the old bytes behind.
On QEMU RV32 and Cortex-M, a caller that generates a key, signs,
and drops the key keeps one unwiped copy of the expanded key in its dead stack frame.

With the `alloc` feature, these functions take an `Allocator` and return `Box<SecretKey, A>`: `keypair_from_seed_in`, `generate_keypair_in`, `try_generate_keypair_in`, and `SecretKey::try_from_slice_in`.

- They generate or copy the key directly into that allocation,
  so moving the box moves only a pointer.
- The box clears the key on drop.
  A failed call clears it before it returns.
- In the same measurement, these constructors leave no copy of the key on the stack
  or in the freed allocation.
- Allocation failure behaves as it does for `Box::new_in`.
- Pass `Global`, or an allocator that supplies locked or dump-excluded memory.
  `rscrypto` does not supply such an allocator.
- The by-value constructors stay for core-only use.
  `serde-secrets` deserialization still builds a by-value key.

### Sampling limits

- Matrix sampling examines at most 298 candidates (894 SHAKE bytes).
- Secret-noise sampling consumes all 481 bytes.
  It compacts accepted coefficients with fixed addresses.
- Unpublished-challenge sampling consumes all 221 bytes and uses fixed scans.
- Verification samples its public challenge directly.
- Signing tries at most 821 candidates, with at most 5747 mask nonces for ML-DSA-87.
- Every exhausted sampler or signing loop returns `MlDsaError::RejectionLimit`.
  There is no fallback signature and no partial private output.

## Security evidence and qualification limits

The [threat model](../THREAT_MODEL.md) does not change.
This work makes no new constant-time claim for whole operations.

- Polynomial arithmetic uses fixed coefficient traversals.
- Signing rejects aggregate norms.
  It packs candidate hints only after every rejection check passes.
- Secret polynomial, SHAKE, byte-buffer,
  and rejected-output owners use the existing volatile cleanup.
- After each private SHAKE256 call,
  a fixed 2 KiB volatile scrub clears the dead stack below the hashing helper.
  ML-KEM shares this helper.
  Capability detection runs before the helper starts.
- The scrub reaches the Keccak spill slots that the compiler makes only
  if the linked frames fit within that bound.
  Each target therefore needs frame review (`just stack-frames`).
- These source properties need compiler and timing evidence for each target.

Target qualification must keep these boundaries:

1. **Qualify the secret-sensitive arithmetic and the fixed-work samplers in the final artifact.**
   - Standard rejection sampling returns the first accepted candidate.
     Whole signing therefore has a variable execution time.
     It has no strict constant-time claim under the CT policy.
   - Agreement with an external library does not prove side-channel resistance.
     Machine-code review and native timing evidence must name the operations, compiler,
     and target they cover.
   - `ct.toml` keeps whole ML-DSA signing best effort,
     and requires separate evidence for its secret-sensitive kernels.
     Its timing cases cover forward and inverse transforms, products, accumulation, norm checks,
     rounding, secret samplers, and valid-key preparation for all parameter sets.
   - Separate inverse-NTT and Montgomery cases run portable arithmetic.
   - Retained C ABI roots support review of the linked code and its call closure.
   - The bounded BINSEC root covers only the portable Montgomery leaf.
     It does not cover full transforms, accelerated kernels, samplers, or signing.
   - Registration defines the required evidence.
     Each target still needs passing results for the exact candidate.
1. **Reduce the signing stack for constrained targets, and collect device evidence for the measured
   bounds.**
   Prepared storage is written in place.
   Its secret polynomials are cleared on handle drop, on failed preparation, and on storage drop.
   Move, register, and spill copies that the compiler makes stay outside the general cleanup claim.
1. **Test bounded-sampler and signing exhaustion through production paths.**
   Extend the fuzz and corpus coverage, and run the remaining native and device targets.
   The first native and portable suites, the feature and MSRV matrix, the packaging consumers,
   and the Wasmtime scalar and SIMD128 vector lanes pass.
   They do not qualify every target.
1. **Complete comparable workloads:** prepared and unprepared,
   representative seeds and signing tails, pure and prehash modes, and `portable-only` measurements.
   Keep the exact source, compiler, CPU, statistics, and profile artifacts.
   One repeated deterministic signature does not describe the distribution of signing latency.

## Backends

The portable implementation defines the behavior and stays the fallback.
All paths use the same canonical coefficients and generated roots.

| Build                                                          | Backend |
| -------------------------------------------------------------- | ------- |
| macOS and Linux AArch64, compile-time NEON, no `portable-only` | Original NEON forward and inverse NTTs and matrix-product accumulation. |
| Linux x86-64, no `portable-only`                               | Original AVX2 arithmetic, after cached CPU and OS capability detection. |
| Linux IBM Z and little-endian POWER, no `portable-only`        | Original z/Vector or POWER8 vector transforms, products, and accumulation, after cached CPU and OS capability detection. |
| RISC-V32 and RISC-V64, compile-time M, no `portable-only`      | Original scalar Montgomery assembly within the common scalar transform schedule. No vector extension is needed. |
| All other configurations                                       | Portable arithmetic. |

- IBM Z and POWER share one four-lane transform schedule.
  The widening products and masked reductions use original register-only assembly.
  This prevents compiler scalarization and reduction branches that depend on coefficients.
- Register barriers protect scalar reduction and selection on IBM Z and RISC-V,
  also with `portable-only`.
- The Linux NEON inverse stage stays out of line.
  This avoids coefficient spills found in the first optimized artifact.
  It is a compiler-specific mitigation, not a general register or stack-cleanup guarantee.
- On macOS AArch64 without `portable-only`, public matrix expansion batches adjacent columns through the existing
  paired SHAKE backend.
  Other targets, and `portable-only`, keep scalar expansion.
  The paired path uses two public polynomial buffers for each row instead of one.
  Prepared operations keep the same key-owner sizes.
  Existing SHAKE dispatch is reused.
- `portable-only` stops runtime capability selection.
  A build that enables SHA-3 instructions at compile time can still use them in shared SHAKE.
  Benchmark claims must therefore record the compiler target features as well as the Cargo features.

Architecture qualification and comparative performance work are still open.
