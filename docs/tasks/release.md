# v0.1.0 Release Audit

## Public API Audit

### Verdict

`rscrypto` is already strong on root UX/DX. The root surface is small, the core traits are coherent, and the crate now has a much better README and item-level docs story than it had before.

The remaining release-quality gaps are structural, not cosmetic:

1. hidden-but-public semver surface
2. naming law not fully settled below the root
3. advanced configuration story is fragmented and env-heavy

Those are fixable. They should be treated as one public-surface cleanup project, not three unrelated nits.

### What Is Strong

- Root exports are curated instead of dumping every alias and backend detail into `rscrypto::*`.
- `Checksum`, `Digest`, and `Mac` share the same mental model: `new` -> `update` -> `finalize` -> `reset`.
- Advanced surfaces are at least mostly explicit: `checksum::config`, `checksum::introspect`, `hashes::introspect`, `hashes::fast`, `platform`.
- The root API is guarded by compile and doctest coverage instead of relying on convention alone.

### What Still Needs Work

#### 1. Hidden-but-public surface is still real surface

Files like [`src/checksum/mod.rs`](../../src/checksum/mod.rs) and [`src/hashes/mod.rs`](../../src/hashes/mod.rs) still publish internal modules behind `#[doc(hidden)] pub` or equivalent public wiring.

That is bad boundary hygiene.

`pub` means semver commitment whether rustdoc shows it or not.

**Target state**

- `pub` means stable supported API
- `pub(crate)` means internal
- `#[doc(hidden)] pub` should be near-zero or zero

Anything that is only for internal benchmarking, dispatch plumbing, or test support should stop being public.

#### 2. Naming is good at the root but not fully governed deeper down

Current examples:

- `Xxh3` vs `Xxh3_64`
- `RapidHash` vs `RapidHash64`
- `AsconXofReader` vs `Shake256Xof`
- `Blake3Xof` vs the clearer reader-role naming used elsewhere

These names are not disastrous. They are just not ruled by one clean law yet.

**Target state**

- Root aliases stay short and opinionated
  - `Xxh3`, `RapidHash`, `Crc32`, `Crc64`
- Explicit algorithm/module names stay explicit
  - `Xxh3_64`, `Xxh3_128`, `RapidHash64`, `RapidHash128`
- Non-canonical tuned variants stay module-only
  - `RapidHashFast64`, `RapidHashFast128`
- Finalized variable-output reader types use one suffix everywhere
  - prefer `*XofReader`

That would make the crate feel intentional instead of historically accumulated.

#### 3. Advanced config is not yet a clean suite

The current per-algorithm force/config modules work, but they do not form a clean system. They are scattered and lean too hard on env vars as the practical control plane.

That is acceptable for internal tuning. It is not ideal as the long-term public power-user story.

**Target state**

Introduce one explicit policy layer:

- `rscrypto::policy::Policy`
- `rscrypto::policy::ChecksumPolicy`
- `rscrypto::policy::HashPolicy`
- `rscrypto::policy::BackendOverride`

Then make env parsing an adapter, not the primary API:

- explicit code policy first
- `from_env()` or `policy::env` second
- old per-algorithm config modules become compatibility views during migration

This turns a bag of knobs into a coherent suite.

## Recommended Plan

### Phase 1. Boundary Pass

- Inventory every public item.
- Classify each one as:
  - root stable
  - advanced stable
  - internal
- Convert internal `pub` items to `pub(crate)` or private.
- Remove `#[doc(hidden)] pub` surfaces unless they are intentionally supported.
- Add compile-fail tests for imports that must remain private.

**Why first**

If the real boundary is still muddy, every later naming or policy change just renames accidental baggage.

### Phase 2. Naming Pass

- Write down the naming law in crate docs and README.
- Keep root aliases short and explicit.
- Standardize finalized variable-output reader types on `*XofReader`.
- Deprecate old names for one release window instead of hard-breaking them.
- Remove compatibility aliases that no longer earn their keep after the transition.

**Why second**

Once the boundary is real, naming gets much easier. You only name what you actually mean to support.

### Phase 3. Policy Pass

- Introduce a unified `rscrypto::policy` module.
- Make explicit typed policy the primary configuration API.
- Keep env parsing as an adapter layer.
- Preserve requested-vs-effective override visibility for diagnostics.
- Map old config entry points onto the new policy during migration.

**Why third**

Configuration should sit on top of a stable naming and visibility model, not try to define it.

### Phase 4. Compatibility Pass

- Add deprecations for renamed types and superseded config entry points.
- Provide migration notes in README and release notes.
- Keep behavior stable while surface cleanup lands.

### Phase 5. Surface Suite

Build a dedicated public-surface test suite that checks:

- visibility discipline
- naming discipline
- deprecation coverage
- policy clamping behavior
- README and doctest accuracy

This is the difference between “we cleaned it up once” and “the surface stays clean.”

## Strong-Suite Definition

The crate should eventually have a release-facing suite that enforces all of this:

### Visibility Discipline

- No hidden-public growth
- No accidental public internal modules
- Public API snapshot reviewed in CI

### Naming Discipline

- Canonical root aliases stay canonical
- Concrete names stay explicit
- Reader suffixes stay uniform
- Deprecated aliases remain until scheduled removal

### Policy Discipline

- Requested override vs effective override is deterministic
- Capability clamping is table-driven
- Env adapter behavior is tested separately from core policy behavior

### Docs Discipline

- Root examples compile
- Advanced examples compile
- README stays aligned with actual exports
- Release docs describe current behavior, not intent or future plans

## Release Readiness

## Verdict

`rscrypto` looks mechanically releasable as `v0.1.0`.

I did **not** find a correctness blocker in the current tree. The crate has already passed a strong validation set, including:

- `cargo test`
- `cargo test --all-features`
- `cargo test --doc --all-features`
- `just test-feature-matrix`
- `cargo check --examples --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo doc --no-deps --all-features`
- `cargo package --allow-dirty`

The main constraint is not correctness. It is surface discipline and release positioning.

## Positioning Constraint

If the release claim is:

> `rscrypto` is a correctness-focused, zero-dependency-by-default Rust checksum and hashing crate with strong checksum coverage, solid SHA-2 support, broad algorithm coverage, and optional parallel BLAKE3 support.

that claim is supportable.

If the release claim is:

> fastest drop-in replacement for `blake3`, `sha3`, `xxh3`, `rapidhash`, and every CRC crate on every target

that claim is still too broad.

Source benchmark context: [`docs/bench/BENCHMARKS.md`](../bench/BENCHMARKS.md)

## Remaining Release Tasks

- Trim the published crate contents in [`Cargo.toml`](../../Cargo.toml) with a real `include` or `exclude` policy.
- Fix stale docs under [`docs/`](..).
- Decide the intended public boundary before tagging if you want `0.1.0` to feel disciplined instead of provisional.
- Do one downstream migration smoke test in your own codebase before publish.
- Run release verification serially, not concurrently, against the same `target/` directory.

## Release Checklist

- [ ] Add `include` / `exclude` rules to [`Cargo.toml`](../../Cargo.toml).
- [ ] Clean stale references in [`docs/README.md`](../README.md).
- [ ] Ensure [`docs/bench/README.md`](../bench/README.md) and [`docs/bench/BENCHMARKS.md`](../bench/BENCHMARKS.md) do not contradict each other.
- [ ] Audit hidden-public modules and either privatize them or explicitly bless them as supported advanced API.
- [ ] Write and adopt the naming law for aliases, concrete types, and XOF reader types.
- [ ] Design the unified `policy` surface before expanding more per-algorithm config APIs.
- [ ] Run one downstream replacement branch and validate tests plus any hot-path benches.
- [ ] Cut the release only after the serial verification pass is clean.

## Serial Release Commands

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --all-features
cargo test --doc --all-features
just test-feature-matrix
cargo check --examples --all-features
cargo package
cargo publish --dry-run
```

If all of that is clean:

```bash
git tag v0.1.0
git push origin main --tags
cargo publish
```

## Bottom Line

If your bar is:

- correct
- broad
- dependency-light
- usable in your own codebase

then `v0.1.0` is reasonable.

If your bar is:

- perfectly disciplined public boundary
- fully unified advanced configuration
- fully settled naming law
- honest claim of universal upstream replacement

then there is still one serious cleanup pass to do.
