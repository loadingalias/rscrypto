# Password-hashing memory APIs

Use `verify_password_with_memory` on `Argon2idPassword` or `ScryptPassword`
to verify PHC records in reusable caller memory. Argon2 also provides
`verify_password_with_context_and_memory` for borrowed pepper and associated data.
These methods keep bounded record approval, opaque failures, and cleanup inside
rscrypto. The existing `argon2`, `scrypt`, and `phc-strings` features still enable
`alloc`.

The workspace is a mutable block slice. It can borrow storage from `Vec<Block, A>`
using the allocator API supported by the crate's Rust compatibility floor.
Allocate once with `Vec::new_in(allocator)`, use `try_reserve_exact` to handle
allocation failure, initialize the blocks, and reuse the slice across operations.
The [password-hashing example](../examples/password_hashing.rs) runs this pattern.
The allocator stays with the caller, who controls pooling and concurrency; the
verifier needs no allocator parameter or allocation during sequential execution.

## Current contract

[`Cargo.toml`](../Cargo.toml) makes `argon2`, `scrypt`, and `phc-strings` enable
`alloc`. Disabling default features removes `std`, but these selections still
enable `alloc`. The `pbkdf2` feature does not require either.

The current production paths have these storage owners:

| Operation | Large workspace | Computed digest |
| --- | --- | --- |
| Raw `derive` | Allocated per call | Caller output slice |
| Raw `derive_with_memory` | Caller block slice | Caller output slice |
| Raw `verify` | Allocated per call | Allocated to the expected digest length |
| Raw `verify_with_memory` | Caller block slice | Allocated to the expected digest length |
| PHC `verify_password` | Allocated after record approval | Fixed 32-byte stack owner, cleared on drop |
| PHC `verify_password_with_memory` | Caller block slice | Fixed 32-byte stack owner, cleared on drop |

Argon2's context variants follow the corresponding storage contract. PHC
parsing and resource approval use borrowed text and fixed arrays; rejected
records do not allocate a workspace. The raw verifiers allocate their digest
before invoking derivation, so even a short caller workspace can incur that
allocation.

`Argon2Params::memory_blocks()` returns the required count of 1,024-byte
`Argon2Block`s, including the specified rounding by lane count.
`ScryptParams::memory_blocks()` checks the address-space bounds and returns the
count of 64-byte `ScryptBlock`s. Its count is `(N + 2*p + 1) * 2*r`, including
setup and scratch storage. Use these methods instead of reproducing the formulas
in consumers. Argon2 also checks address-space bounds during derivation.

The derivation paths ignore the initial contents of the required prefix, clear
that prefix before returning after use, and leave any surplus blocks unchanged.
Input and sizing failures leave the workspace untouched. The block slice may
come from any valid caller storage; stack capacity is the caller's constraint,
not a restriction imposed by the slice API.

With `parallel` enabled, eligible Argon2 calls enter the existing Rayon path.
Providing work memory does not establish an allocation-free scheduling contract.
Memory reuse therefore does not imply either a build without `alloc` or an
operation that never reaches an allocator.

## PHC methods

The caller-memory methods have these signatures:

```text
Argon2idPassword::verify_password_with_memory(
    &self, password: &[u8], encoded: &str, memory: &mut [Argon2Block],
) -> Result<PasswordStatus, VerificationError>

Argon2idPassword::verify_password_with_context_and_memory(
    &self, password: &[u8], encoded: &str,
    context: Argon2Context<'_>, memory: &mut [Argon2Block],
) -> Result<PasswordStatus, VerificationError>

ScryptPassword::verify_password_with_memory(
    &self, password: &[u8], encoded: &str, memory: &mut [ScryptBlock],
) -> Result<PasswordStatus, VerificationError>
```

The contract is:

1. Approve the entire record with the existing private parser and verifier
   limits before deriving or touching caller memory. Keep the accepted
   algorithms, versions, canonical parameter/base64 rules, salt lengths, and
   fixed 32-byte digest unchanged.
2. Size the workspace for the **accepted record's parameters**. For a pool that
   accepts every permitted record, retain the profile used to construct its
   verification limits with `for_profile` and size from `memory_blocks()`.
   If `with_limits` permits a larger profile, sizing only for the generation
   profile is insufficient. No public parser or new sizing API is needed.
3. On malformed or over-limit input, invalid context/input lengths, or short
   memory, return the existing opaque `VerificationError` without touching the
   workspace. Once derivation uses it, clear the required prefix on success or
   mismatch and preserve the surplus. Keep the existing unwind cleanup.
4. Keep the computed digest in the existing fixed-size zeroizing owner. Compare
   all 32 bytes with the existing comparison primitive. Return `Current` or
   `NeedsRehash` only after a match, using the same generation-profile and salt
   rules. Do not expose a separate sizing error through verification.
5. Borrow password, encoded record, pepper, and associated data without changing
   their ownership. The mutable workspace borrow excludes concurrent reuse.
   Use the current derivation backends and `parallel` admission policy. Promise
   reuse of the workspace and no heap digest, without promising that Rayon
   scheduling is allocation-free.

## Feature decision and alternatives

The caller-memory PHC extension retains the current feature graph. Removing
`alloc` from the existing leaves would require gating out allocating methods and would break
existing `default-features = false` callers that select those leaves. It would
also remove other APIs they currently obtain through the transitive `alloc`
feature. PHC workspace reuse does not require that compatibility change.

An additive core-feature split could preserve those callers: old features would
keep enabling allocation, and new features would expose only borrowed-memory
operations. That is feasible, but it creates additional public feature and
re-export contracts. PHC parsing would also need separating from allocating
generation, because `phc-strings` currently enables `alloc`.

Raw `verify_with_memory` has a further obstacle: its output length is chosen by
the caller. A fixed 32-byte scratch buffer would narrow the existing contract.
A future allocation-free verifier must either accept separate digest scratch
space or compare output incrementally through the production finalization path.
The former adds a caller resource contract; the latter changes cryptographic
finalization and its proof obligations. Neither is needed by the fixed-size PHC
methods. Allocator-free raw verification and a core-feature split remain outside
this API; they require a separate consumer contract and compatibility decision.

The allocating methods remain available for applications that accept per-call
workspace allocation. Asking consumers to parse PHC records themselves to reach
the raw memory APIs would duplicate canonical parsing, resource approval, and
rehash policy. The caller-memory methods retain those responsibilities in rscrypto
and add no public record representation, allocator abstraction, or dependency.

## Validation and limits

Owned and borrowed PHC verification share each algorithm's private approval,
derivation, and comparison path. Both retain the fixed-size zeroizing digest;
neither routes through the raw verifier's heap digest. The PHC entry points hold
the DIT guard across approval, derivation, and comparison. This does not change
the algorithms' best-effort timing classification in [`ct.toml`](../ct.toml).

The unit tests use RFC 9106 and RustCrypto-derived digests to check the borrowed
path, including profiles larger than the generation policy, rounded Argon2
memory costs, parallel-eligible Argon2, wrong password/context, short storage,
dirty initial contents, cleared used blocks, and unchanged surplus blocks.
The PHC integration tests observe allocations with positive controls on the
allocating methods. Caller-owned `Vec<Block, System>` exercises allocator-backed
storage. The PHC fuzz target compares owned and borrowed verification on hostile
records, including Argon2 context.

No throughput improvement is claimed. Workspace reuse removes the per-call
workspace allocation but preserves initialization, derivation, and clearing.
Argon2 scheduling with `parallel` remains outside the no-allocation observation.

## Evaluation and qualification

The 2026-10-05 evaluation selected the PHC extension and retained the existing
feature graph. The user approved that scope and allocator integration the same
day. Allocator-free raw verification was evaluated and excluded for the
compatibility and digest-storage reasons above; it is not an unfinished part of
this extension.

Implementation qualification used the effective working tree based on
`d4045559cd84e3c6673a7b2c2aa3897bf31a0361`, including pending portable-backend and
compatibility changes. Host execution used `aarch64-apple-darwin` and
`rustc 1.101.0-nightly (5c543b0b8 2026-09-29)`. The maintained evidence lives in
the [Argon2 tests](../src/auth/argon2/mod.rs), [scrypt tests](../src/auth/scrypt.rs),
[PHC integration tests](../tests/phc_roundtrip.rs),
[minimum-feature consumer tests](../tests/phc_external_entropy.rs), and
[PHC fuzz target](../fuzz/target_impls/auth_phc.rs).

Passed checks:

- `just ci-check`: formatting, native and portable host Clippy, independent
  workspace lints, and all-feature Rustdoc. This read-only gate preserved the
  checkout's pre-existing edits.
- `just test --release --lib -E 'test(auth::argon2) | test(auth::scrypt) | test(slow_secret_operations)'`:
  49 tests passed. `just test --release --test phc_roundtrip --test phc_external_entropy`:
  9 tests passed. The same selected library/integration scope with `--portable`
  passed all 58 tests. These use the repository's production-auto and
  portable-only feature selections, respectively.
- The internal evidence wrapper ran `cargo test --release --locked --no-default-features --features argon2,scrypt,phc-strings,diag --lib traits::ct::tests`:
  8 tests passed, including public guard entry, state restoration, and pure
  proof leaves staying outside the guard.
- `cargo +1.100.0-beta.1 test --release --locked --no-default-features --features argon2,phc-strings --test phc_external_entropy`,
  repeated with `scrypt,phc-strings`: 2 tests passed per selection. The library
  selections enable neither `std` nor `getrandom`; the host test harness uses
  `std` to exercise `Vec<Block, System>`.
- `cargo check --lib --locked --no-default-features --features argon2,scrypt,phc-strings`
  with `--target` for every entry in the [target catalog](../.config/target-matrix.json):
  all 16 targets compiled, including bare-metal and WebAssembly.
- `just test-examples`: all 10 examples passed, including allocator-backed
  Argon2 workspace reuse. `just test-fuzz auth_phc`: 2,020,260 inputs in the
  configured 60-second campaign, with no failure.
- All 7 compiler API snapshots gained exactly the 3 approved methods;
  `just ct-validate --manifest-only` passed. The existing best-effort timing
  classifications remain unchanged.

The allocation test includes allocating-verifier positive controls and observes
zero allocations for sequential borrowed verification on success, mismatch, and
short storage. Allocation and cleanup assertions are regression evidence, not
throughput measurements or a new whole-algorithm timing claim. Cross-target
compilation does not establish execution or timing on those targets.
