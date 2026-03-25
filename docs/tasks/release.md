# v0.1.0 Release Readiness

## Verdict

`rscrypto` looks mechanically releasable as `v0.1.0`.

I did **not** find a correctness blocker in the current tree. The crate passes:

- `cargo test`
- `cargo test --all-features`
- `cargo test --doc --all-features`
- `just test-feature-matrix`
- `cargo check --examples --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo doc --no-deps --all-features`
- `cargo package --allow-dirty`

The real constraint is **positioning**. If the release claim is:

> “This replaces `blake3`, `crc-fast`, `xxhash-rust` / XXH3, `rapidhash`, `sha3`, etc. and is the better choice across the board.”

that claim is too broad today.

## Actual Issues

### 1. Performance claim is not ready for a blanket “replace everything” story

Your own benchmark record says the current 2026-03-24 cross-platform score is:

- Overall: `62%` win rate
- Checksums: `87%`
- SHA-2: `71%`
- SHA-3 / SHAKE: `35%`
- Blake3: `56%`
- Fast Hashes: `33%`

Source: [`docs/bench/BENCHMARKS.md`](../bench/BENCHMARKS.md)

That means:

- Checksums are ready to stand on their own.
- SHA-2 looks credible.
- BLAKE3 is mixed, not dominant.
- SHA-3 / SHAKE and the fast hashes are still materially behind too often to market this as a universal performance replacement.

**Release implication:** publish `0.1.0` if you want a correctness-first, zero-dependency-by-default consolidation crate. Do **not** sell it as “strictly better than all upstream crates” yet.

### 2. Packaging is noisier than it needs to be

`cargo package --allow-dirty --list` shows the crate tarball currently ships:

- `.github/*`
- `.config/*`
- `scripts/*`
- `docs/bench/*`
- `deny.toml`
- `justfile`
- internal benchmark notes and baseline files

Current package size:

- `316` files
- `1.3 MiB` compressed

This is not a publish blocker, but it is sloppy. Crates.io should ship the crate, not your CI plant.

**Release implication:** add a tight `include = [...]` whitelist or an `exclude = [...]` list in [`Cargo.toml`](../../Cargo.toml) before publishing.

### 3. `docs/README.md` is stale

[`docs/README.md`](../README.md) currently:

- says this directory supports the “`v1` release story”
- links to `docs/tasks/sha3-perf-next.md`, which does not exist

That is straightforward stale documentation.

There is also a date mismatch:

- [`docs/bench/README.md`](../bench/README.md) says the canonical report is from `2026-03-22`
- [`docs/bench/BENCHMARKS.md`](../bench/BENCHMARKS.md) is from `2026-03-24`

**Release implication:** clean the docs before tagging so the repo does not contradict itself.

### 4. Local release commands should be run serially

I saw doctest failures only when `cargo test` and `cargo package --verify` were running at the same time against the same `target/` directory. Run release validation serially.

This is a workflow issue, not a library correctness issue.

## What You Need To Do

- Decide the release claim.
- Trim the packaged file set.
- Fix stale docs under `docs/`.
- Do one downstream smoke migration in your own codebase before publish.
- Publish only after the serial verification pass succeeds.

## Recommended Release Positioning

Use a claim like this:

> `rscrypto` v0.1.0 is a correctness-focused, zero-dependency-by-default Rust checksum and hashing crate with strong checksum coverage, solid SHA-2 support, broad algorithm coverage, and optional parallel BLAKE3 support.

Do **not** use a claim like this yet:

> drop-in fastest replacement for `blake3`, `sha3`, `xxh3`, `rapidhash`, and every CRC crate on every target

That second claim is not supported by the current benchmark record.

## Release Checklist

- [ ] Add `include` / `exclude` rules to [`Cargo.toml`](../../Cargo.toml) so the published crate is minimal.
- [ ] Fix stale references in [`docs/README.md`](../README.md).
- [ ] Update [`docs/bench/README.md`](../bench/README.md) so its canonical-report date matches [`docs/bench/BENCHMARKS.md`](../bench/BENCHMARKS.md).
- [ ] Decide whether `README.md` should explicitly state the current performance posture instead of implying blanket replacement.
- [ ] In your downstream codebase, replace the target crates with `rscrypto` on a branch and run its full test suite and any hot-path benches.
- [ ] Cut the release only if the downstream migration result is acceptable.

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

If your bar is **correct, broad, dependency-light, and usable in your own codebase**, `v0.1.0` is reasonable.

If your bar is **honestly replacing every upstream crate on performance-sensitive paths with no caveats**, it is not ready yet. The benchmark record does not support that story.
