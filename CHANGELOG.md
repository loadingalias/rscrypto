# Changelog









## [0.7.6](https://github.com/loadingalias/rscrypto/compare/v0.7.5...v0.7.6) - 2026-07-15

- Bound promoted CT manifest and tool-lock hashes to the recorded evidence commit instead of the release commit's version-only lockfiles.


## [0.7.5](https://github.com/loadingalias/rscrypto/compare/v0.7.3...v0.7.5) - 2026-07-15

- Allowed successful CT and RSA evidence to be promoted across a mechanically verified release-tooling-only delta while preserving exact-commit evidence for every runtime-affecting change.

### 📦 Other Changes

- sync CT tool locks for release ([bef01a7](https://github.com/loadingalias/rscrypto/commit/bef01a7136def5dbdd9306719108a87bb4d7d9ea))


## [0.7.4](https://github.com/loadingalias/rscrypto/compare/v0.7.3...v0.7.4) - 2026-07-15

- Fixed release CT evidence packaging under GitHub Actions by isolating matrix discovery from workflow step outputs.


## [0.7.3](https://github.com/loadingalias/rscrypto/compare/v0.7.2...v0.7.3) - 2026-07-14

- Hardened portable P-256 and P-384 caller-blinded fixed-base multiplication so
  projective randomization is established before secret-selected points enter
  field arithmetic. P-256 also randomizes the scalar representative with a group
  order multiple. Corrected the timing harness's random-secret distribution and
  extended RISC-V assembly screening to conditional branches.
  
  Made release tagging require successful CT and RSA evidence from the exact
  candidate commit. Release publication now promotes that Weekly run's raw CT
  artifacts instead of launching duplicate multi-hour evidence workflows, and RSA
  workflow concurrency can no longer cancel its reusable-workflow caller.
  
  Removed redundant ECDSA keypair timing cases that spent their entire RISC-V
  evidence budgets precomputing public keys without collecting samples. The
  required 20,000-sample P-256/P-384 signing cases and constant-time thresholds
  remain unchanged.

### 📦 Other Changes

- remove zero-sample ECDSA keypair timing cases ([31a9894](https://github.com/loadingalias/rscrypto/commit/31a9894355f2f7d09535cea47bdd4c568035f2dc))
- sync CT tool locks for release ([fbf34a2](https://github.com/loadingalias/rscrypto/commit/fbf34a2a968a383230601714dbdf72cffa603476))
- blind portable ECDSA before secret-selected field arithmetic ([60f4dc2](https://github.com/loadingalias/rscrypto/commit/60f4dc20313587ead5bf75575522188b04f13945))


## [0.7.3](https://github.com/loadingalias/rscrypto/compare/v0.7.2...v0.7.3) - 2026-07-14

- Hardened portable P-256 and P-384 caller-blinded fixed-base multiplication so
  projective randomization is established before secret-selected points enter
  field arithmetic. P-256 also randomizes the scalar representative with a group
  order multiple. Corrected the timing harness's random-secret distribution and
  extended RISC-V assembly screening to conditional branches.
  
  Made release tagging require successful CT and RSA evidence from the exact
  candidate commit. Release publication now promotes that Weekly run's raw CT
  artifacts instead of launching duplicate multi-hour evidence workflows, and RSA
  workflow concurrency can no longer cancel its reusable-workflow caller.

### 📦 Other Changes

- blind portable ECDSA before secret-selected field arithmetic ([60f4dc2](https://github.com/loadingalias/rscrypto/commit/60f4dc20313587ead5bf75575522188b04f13945))


## [0.7.2](https://github.com/loadingalias/rscrypto/compare/v0.7.1...v0.7.2) - 2026-07-14

- Stopped installing the unused BINSEC toolchain on physical RISC-V constant-time
  lanes and gave the full 20,000-sample P-384 evidence cases enough time to
  complete without weakening their release gate.


## [0.7.1](https://github.com/loadingalias/rscrypto/compare/v0.7.0...v0.7.1) - 2026-07-14

- Made release preflight consume the exact-commit Cargo graph assurance gate instead
  of repeating the exhaustive target and feature sweep. Duplicate runs for the same
  release tag now collapse immediately while signed-tag, audit, SemVer, constant-time,
  RSA, package-integrity, provenance, and approval gates remain blocking.


## [0.7.0](https://github.com/loadingalias/rscrypto/compare/v0.6.4...v0.7.0) - 2026-07-13

- Consolidated CI validation into explicit quality, feature-contract, native,
  cross-target, and supply-chain owners. The complete compile and executable
  feature matrices remain blocking, MUSL targets are now passed explicitly, and
  IBM Z, POWER10, and RISC-V runners retain native all-features correctness and
  backend-equivalence coverage without repeating architecture-independent work.
  
  Updated repository automation to cargo-rail 0.17.0 and cargo-rail-action v5.
  Configuration sync now records the public open-world consumer boundary, CI
  checks the compiler-backed unified Cargo graph, and pull requests require
  change-file coverage from the planner's resolved base reference.
  
  Made consolidated CI lanes self-contained by removing the ownership checker's
  ripgrep dependency and pinning the MUSL cross-target lane to the same verified
  Zig build used locally.
  
  Separated fast Quality checks from exhaustive compiler-backed Cargo graph
  assurance. The dedicated blocking lane receives a realistic timeout, retained
  proof artifacts, an identity-validated compiler-evidence cache, planner-based
  PR selection, and unconditional main, Weekly, and release execution.
  
  Consolidated SemVer ownership under cargo-rail's version-aware release planning
  and one hard check of the finalized tag. Release preflight now builds and
  validates the crate once; the publish job verifies and attests that transferred
  artifact instead of repeating the complete preflight.

- Removed variable-latency scalar multiplication from ECDSA P-256/P-384 secret
  arithmetic on s390x and RISC-V while preserving signing support, and restored
  the affected native timing cases as required release gates.
  
  Made constant-time assembly triage fail closed on target-sensitive arithmetic,
  grouped every reachable finding with exact review-bound waivers, and bound
  release evidence to the complete lane set, source commit, toolchain, target
  configuration, generated artifacts, timing results, and BINSEC results. Weekly
  and release publication now both require the dedicated RSA evidence workflow.
  Cross-platform checks now keep host-only feature-matrix flags out of target
  crate selection and keep target-only diagnostics plus CPUID's MSRV safety
  transition lint-clean across the full matrix.
  
  Corrected the public constant-time boundary: `ct.toml` records intent, while a
  claim exists only for configurations with passing evidence in the matching
  attested release bundle.
  
  Removed secret-fed overflow panic branches from the fixed-work ECDSA limb
  multiply-accumulate used on s390x and RISC-V. Blinded signing now additively
  masks the private scalar during `r·d`, with direct full-width carry equivalence
  tests and regenerated target evidence.

- Replaced heap-buffered RapidHash and XXH3 `Hasher` implementations with fixed-size,
  allocation-free state. Added the collection-oriented `RapidHasher`, concatenating
  `RapidStreamHasher`, deterministic `RapidBuildHasher`, and XXH3-128 streaming.
  Deterministic builder documentation now restricts it to trusted keys.
  
  Accelerated long XXH3 streams with AVX2, AVX-512, NEON, VSX, and z/Vector, and
  added seeded partition fuzzing, RapidHash collection properties, backend
  equivalence tests, and explicit allocation coverage.

- Closed native API gaps with HKDF-SHA512, HMAC-SHA3, KMAC128, standalone
  Poly1305, `getrandom` key-generation helpers, AEAD `*_to_vec` helpers, generic
  signing/verification traits, and dedicated AEAD, signature, RSA, and ML-KEM
  examples.
  
  Hardened release and security automation with versioned constant-time evidence
  bundles, release-path audit and semver checks, weekly ASan fuzz-corpus replay,
  OpenSSF Scorecard, CODEOWNERS, CONTRIBUTING guidance, and committed minimized
  fuzz seeds.
  
  Pruned duplicated public Markdown, made `THREAT_MODEL.md` the canonical audit
  entry point, folded advisory readiness into `SECURITY.md`, and kept
  maintainer-only release and benchmark evidence out of README onboarding.

- Private RSA DER exports now return `SecretVec`, which wipes its allocation on
  drop and requires `into_unprotected_vec()` for ordinary extraction. Secret
  parsing, Serde deserialization, and generation use zeroizing temporary guards.
  Secret keys, shared secrets, keypairs, and AEAD cipher contexts no longer
  implement `Clone`; use `duplicate_secret()` where an additional owned secret is
  required.

### 📦 Other Changes

- make RapidHash and XXH3 streaming allocation-free ([822c865](https://github.com/loadingalias/rscrypto/commit/822c865ad909e51d0ab64338c5513db15c24170a))
- isolate ECDSA keypair timing setup ([c8cc2d2](https://github.com/loadingalias/rscrypto/commit/c8cc2d20e740aff9a45bc4bb11e1b5fad7c697e7))
- blind fixed-work ECDSA scalar products ([0fb9929](https://github.com/loadingalias/rscrypto/commit/0fb9929dbc33fb0ee497dcfb381f465dffe59baf))
- harden secret lifecycles and explicit secret duplication ci: consolidate validation and adopt cargo-rail 0.17 planning ([1837da7](https://github.com/loadingalias/rscrypto/commit/1837da704e566c2f672de2b4f9f6eaeb890bb3e6))
- remove variable-latency ECDSA multiplication on s390x and RISC-V ([2a8a11c](https://github.com/loadingalias/rscrypto/commit/2a8a11c53aed7b829a2bcf7cd4aa4d7b5f6f07f5))


## [0.6.4](https://github.com/loadingalias/rscrypto/compare/v0.6.3...v0.6.4) - 2026-07-07

- Updated release workflow validation for cargo-rail 0.15.
- Refreshed the dev dependency lockfile to avoid the crossbeam-epoch advisory in test tooling.

### 📦 Other Changes

- make ECDSA oracles tolerate p256 update ([ac62d9a](https://github.com/loadingalias/rscrypto/commit/ac62d9a937eb2aa4cdd25a4c1722687f4aedfd72))



## [0.6.3](https://github.com/loadingalias/rscrypto/compare/v0.6.2...v0.6.3) - 2026-07-05

### 📦 Other Changes

- auth: format argon2 docs ([0b455cf](https://github.com/loadingalias/rscrypto/commit/0b455cfb6217449707df4f03675441dfcf9f9e60))
- workspace: add trusted release publishing gate ([4de9cbe](https://github.com/loadingalias/rscrypto/commit/4de9cbea87f9031bce37f343d4161e20fba2eb73))



## [0.6.2](https://github.com/loadingalias/rscrypto/compare/v0.6.1...v0.6.2) - 2026-07-04

### 🏗️ Build

- route CT tooling through a Python compatibility shim ci: refresh runs-on action pin workspace: refresh fast hash and AWS-LC dependency locks benchmarks: refresh performance overview artifacts crypto: fix ChaCha20-Poly1305 lint and aarch64 ML-KEM asm tests ([6f9dae8](https://github.com/loadingalias/rscrypto/commit/6f9dae8bb4c14bc46cd3ba189a5dba36a63c508e))
- decouple asm ownership check from task notes ([c31d405](https://github.com/loadingalias/rscrypto/commit/c31d40564a9723c9301369f7a7a88c7bb70dbf69))
- scope dev-machine recipes and ignore task notebooks ([7236a1f](https://github.com/loadingalias/rscrypto/commit/7236a1f23f8c460ef6007889de563d11ffcb1a8b))

### 👷 CI

- pin Wasmtime install for no_std wasm suite auth: restore aarch64 ML-KEM basemul asm test helper ([86fac6f](https://github.com/loadingalias/rscrypto/commit/86fac6f1db53e30aa3ad8f06a3cebf028c792e77))

### 📦 Other Changes

- benchmarks: refresh July 4 performance overview and chart ([878999c](https://github.com/loadingalias/rscrypto/commit/878999cbb7c48f368e1dc8d753cf88ff9b536732))
- auth: harden RSA private CRT arithmetic aead: gate forced ChaCha20 kernels on runtime caps ([596498f](https://github.com/loadingalias/rscrypto/commit/596498f0e07e869eac71fd31c157aa1b22186239))
- auth: fix aarch64 ML-KEM bounded SampleNTT stores ([73eb536](https://github.com/loadingalias/rscrypto/commit/73eb53611f762ab2c6dadad1c81a07f47fa4972b))
- crypto: revert aarch64 ML-KEM matrix XOF split route ([6e0a871](https://github.com/loadingalias/rscrypto/commit/6e0a87146c9a187cf5c9d6225ae4452d06504edf))
- crypto: route aarch64 ML-KEM matrix XOF around x3 hybrid ([7cefb0d](https://github.com/loadingalias/rscrypto/commit/7cefb0d50464b392fdc9d312316026c83de75e9f))
- Revert "crypto: feed aarch64 ML-KEM sampler from triple SHAKE state" ([bc21ad6](https://github.com/loadingalias/rscrypto/commit/bc21ad6cfe15e840183ec02ab218e71701852d1c))
- auth: batch aarch64 ML-KEM fused chunk accumulation ([81f2184](https://github.com/loadingalias/rscrypto/commit/81f2184d9dadbe0e3706320a48b4430c04d95e17))
- hashes: revert regressing aarch64 Keccak x3 assembly ([342356e](https://github.com/loadingalias/rscrypto/commit/342356e10248e643248b2d412b17d128c5a648bc))
- hashes: add owned aarch64 Keccak x3 assembly ([481105a](https://github.com/loadingalias/rscrypto/commit/481105a8b86c43c4d43cfeb8ad74ef41ac02d763))
- hashes: restore aarch64 Keccak hybrid batch dispatch ([4887c16](https://github.com/loadingalias/rscrypto/commit/4887c161439b91e0eb7183811df1c69c5cd769ed))
- hashes: route aarch64 Keccak batches through paired SHA3 lanes ([bad0bc6](https://github.com/loadingalias/rscrypto/commit/bad0bc68f80456bd096445ebb83368588493149f))
- auth: batch aarch64 ML-KEM triple rejection parsing ([dae2b42](https://github.com/loadingalias/rscrypto/commit/dae2b42e90eb5345379b22e97b7921026cda45d5))
- auth: keep AArch64 ML-KEM SampleNTT asm on Linux ([883f255](https://github.com/loadingalias/rscrypto/commit/883f2550443dd85038ad462173d3ca24ff744247))
- auth: add bounded AArch64 ML-KEM SampleNTT tail parser bench: expose ML-KEM three-block sampler diagnostics ([a74f9cb](https://github.com/loadingalias/rscrypto/commit/a74f9cb1ad64b6bc16bcfe4f19b61463a98d6fcd))
- aead: restore x86 and POWER ChaCha20-Poly1305 open fast paths bench: expose x86 ChaCha20-Poly1305 open diagnostic row ([e05a270](https://github.com/loadingalias/rscrypto/commit/e05a27056a951faec4607c7bbf58e58685e3d192))
- aead: add measured Linux ChaCha20-Poly1305 fast paths ([f26ef5d](https://github.com/loadingalias/rscrypto/commit/f26ef5d6a15a4931df278b1ae7d1bd0736945e63))
- aead: gate x86 ChaCha20-Poly1305 asm by measured thresholds ([57fe253](https://github.com/loadingalias/rscrypto/commit/57fe253fecf58353ac2e86225a46e88e123b6c06))
- auth: route x86 P-384 signing through complete comb ([1bcc2d1](https://github.com/loadingalias/rscrypto/commit/1bcc2d19cf7b2df93ad3e814a1763ab9ce7e84b4))
- aead: batch owned AArch64 ChaCha20-Poly1305 par4 bulk ([b09e389](https://github.com/loadingalias/rscrypto/commit/b09e38924a432aac293b311c9a182aac24d125d2))
- aead: add owned AArch64 ChaCha20-Poly1305 par4 diagnostics bench: compare owned AArch64 ChaCha20-Poly1305 par4 rows ([1ffcf00](https://github.com/loadingalias/rscrypto/commit/1ffcf00ea01574d9a59818948c2f864aa8c6d5e7))
- auth: expose Ed25519 verify phase diagnostics bench: add Ed25519 verify phase rows ([dbeff3d](https://github.com/loadingalias/rscrypto/commit/dbeff3d1c363732f26352d17da3a47deb8988a82))
- auth: split ML-KEM-1024 row sampling by aarch64 platform bench: expose ML-KEM-1024 quad row sampler comparator ([a6315a2](https://github.com/loadingalias/rscrypto/commit/a6315a21afa57e7b54da80acfd88b10174d82c78))
- auth: batch ML-KEM-1024 materialized row sampling through quad XOF bench: compare ML-KEM-1024 row sampler schedules ([77e0543](https://github.com/loadingalias/rscrypto/commit/77e0543011f589c164719cd1b28ff00f7d792a78))
- auth: expose ML-KEM sampler phase diagnostics bench: add ML-KEM sampler phase rows ([1b3ec7c](https://github.com/loadingalias/rscrypto/commit/1b3ec7c7ef40f90f61713a086745cd335c05b011))
- auth: expose ML-KEM-1024 materialized keygen split diagnostics bench: add ML-KEM-1024 materialized matrix split rows ([2f103b9](https://github.com/loadingalias/rscrypto/commit/2f103b97a63decff488d762d36aa92fe43182037))
- auth: expose forced ML-KEM keygen matrix diagnostics bench: add ML-KEM keygen matrix split rows ([9843d86](https://github.com/loadingalias/rscrypto/commit/9843d8645c73eb374f4aa15a14e700abc883e700))
- auth: batch aarch64 ML-KEM matrix sampling through triple XOF hashes: add Linux aarch64 Keccak x3 permutation support bench: expose ML-KEM triple matrix sampling rows ([22a3fda](https://github.com/loadingalias/rscrypto/commit/22a3fda51838e07f070a8d7b9f31724546803eaf))
- auth: expose ML-KEM keygen phase diagnostics and gate K2 aarch64 accumulation bench: add ML-KEM keygen phase rows ci: add ML-KEM keygen phase selector ([1379d30](https://github.com/loadingalias/rscrypto/commit/1379d306836cc614c4c290f80aa7db4f1b7978bf))
- auth: tune aarch64 ML-KEM keygen finalization ([7775a8d](https://github.com/loadingalias/rscrypto/commit/7775a8d847d971d550f6807242df82a9c90674c4))
- auth: route aarch64 ML-KEM NTT through owned NEON ([816e919](https://github.com/loadingalias/rscrypto/commit/816e9190864ab49e8ed94a5b5fc58a6f746808c5))
- auth: shortcut P-384 signing r path and delete dead x86 asm ([2a54d93](https://github.com/loadingalias/rscrypto/commit/2a54d933ed46c752988344223d7609d2ffd6b75b))
- hashes: clear x86 Blake3 and introspection clippy lints ([ae22843](https://github.com/loadingalias/rscrypto/commit/ae22843cd938313784cd1f4967f55cd68eaffc9a))
- aead: tune owned AArch64 ChaCha20-Poly1305 path hashes: clear BLAKE3 diag clippy lints docs: record assembly replacement tuning decisions ([c8f6bfb](https://github.com/loadingalias/rscrypto/commit/c8f6bfb91b6fe7cb8c22f4cb9728a98a8abcf47e))
- hashes: delete Darwin SHA-2 assembly for owned aarch64 kernels aead: expose owned ChaCha20-Poly1305 diagnostic path auth: widen public RSA CIOS threshold and add portable diagnostics bench: split AEAD and RSA assembly replacement benchmarks benchmarks: refresh BLAKE3 overview and README perf chart docs: track assembly ownership notebook decisions ([76f4fa3](https://github.com/loadingalias/rscrypto/commit/76f4fa343412c39c8c6a3bd1b7f3e866a42513aa))
- hashes: retarget BLAKE3 x86 compression and expose diag kernels aead: mark AES-GCM assembly as rscrypto-owned auth: mark RSA assembly as rscrypto-owned bench: add BLAKE3 diagnostic benches and gap report build: gate assembly provenance with ledger check ([e643fe5](https://github.com/loadingalias/rscrypto/commit/e643fe55af7311aa2ccf17f18613e8a63871d08a))



## [0.6.1](https://github.com/loadingalias/rscrypto/compare/v0.6.0...v0.6.1) - 2026-06-24

### 📦 Other Changes

- crypto: validate platform overrides and gate s390x AEGIS ([83bcde8](https://github.com/loadingalias/rscrypto/commit/83bcde865334691387c2d7abb5242f1cfc3479e1))



## [0.6.0](https://github.com/loadingalias/rscrypto/compare/v0.5.0...v0.6.0) - 2026-06-23

### 👷 CI

- bump checkout and tool action pins workspace: refresh getrandom lockfiles benchmarks: refresh Linux benchmark scorecard ([5e34396](https://github.com/loadingalias/rscrypto/commit/5e343967375d40eaac2c49d887f159b9b22e2a6a))

### 📦 Other Changes

- auth: add Darwin aarch64 ML-KEM assembly paths ([7832c94](https://github.com/loadingalias/rscrypto/commit/7832c944cc8b6456654b2d7de338a88c16ca0532))
- Revert "auth: add aarch64 ML-KEM quad rejection parser" ([133627a](https://github.com/loadingalias/rscrypto/commit/133627a0b1494eec746c2c434affc968f7113f9a))
- auth: port aarch64 ML-KEM k3 basemul schedule ([2d65c61](https://github.com/loadingalias/rscrypto/commit/2d65c6116a30e468c03c04f49691e7fba55684f2))
- auth: port aarch64 ML-KEM k4 basemul schedule ([abcf8d8](https://github.com/loadingalias/rscrypto/commit/abcf8d86943167ed98c40c17c5ada1b480d517bb))
- Revert "auth: reschedule aarch64 ML-KEM basemul accumulation" ([5bd2052](https://github.com/loadingalias/rscrypto/commit/5bd2052457b7f95bfa1f79dbe88f07ba8ed6e7f2))
- auth: tighten aarch64 ML-KEM inverse final scale ([551f1b6](https://github.com/loadingalias/rscrypto/commit/551f1b6c6fd62ed649b792b4c8067da4a6bd79a4))
- auth: unroll aarch64 ML-KEM inverse NTT asm ([2d725d0](https://github.com/loadingalias/rscrypto/commit/2d725d0152bd40f4a819575e6ea91b56eaffcbbe))
- auth: precompute aarch64 ML-KEM inverse reducers ([7271015](https://github.com/loadingalias/rscrypto/commit/72710156bd95b3b47a9315c330e403fa1e8c3b9e))
- auth: include inverse add asm in aarch64 gate filter ([766132e](https://github.com/loadingalias/rscrypto/commit/766132e858136ba3659c5b825dd4969d09bd6091))
- auth: add Linux aarch64 ML-KEM inverse NTT asm diagnostics ([133b8b6](https://github.com/loadingalias/rscrypto/commit/133b8b62bfc7f7b740ce0a4ceeee4b6629b22029))
- Revert "auth: fuse aarch64 ML-KEM inverse NTT final scale" ([5c79304](https://github.com/loadingalias/rscrypto/commit/5c79304898cf740bcdf473682c2e12a077ae0b20))
- auth: dispatch fused ML-KEM1024 PKE matrix path ([369e422](https://github.com/loadingalias/rscrypto/commit/369e4220553f14730bcad23acde835bb3ab5ba74))
- auth: compact ML-KEM fused rejection into NEON chunks ([9708c1d](https://github.com/loadingalias/rscrypto/commit/9708c1d51bf5707137c89d750ccea4238b3339a9))
- auth: add aarch64 K2 ML-KEM row-dot path ([e28eaee](https://github.com/loadingalias/rscrypto/commit/e28eaee8ad1e5ccb7d9ea828706210eab099d1b5))
- hashes: use paired SHA3 fallback for aarch64 quad Keccak ([d712967](https://github.com/loadingalias/rscrypto/commit/d71296723386e7ce8d7ab5cb53cd46013acdb6ad))
- crypto: add aarch64 triple SHAKE path for ML-KEM sampling ([2f00fa7](https://github.com/loadingalias/rscrypto/commit/2f00fa79bf24c67d1f44fdbbd75f1b01bec4eb58))
- auth: feed fused ML-KEM products from compact aarch64 rejection ([a05a23b](https://github.com/loadingalias/rscrypto/commit/a05a23b1917215f46ea65324656f5d2a8681197a))
- auth: compact aarch64 ML-KEM rejection lanes ([13a50c8](https://github.com/loadingalias/rscrypto/commit/13a50c85bc64de3774cd57b5f31834f67e2fb14c))
- auth: add aarch64 ML-KEM rejection parser ([951be2c](https://github.com/loadingalias/rscrypto/commit/951be2c86d846afcaee2003453768a6596fd9070))
- Revert "hashes: route aarch64 Keccak x4 through SHA3 pairs" ([7e81d7f](https://github.com/loadingalias/rscrypto/commit/7e81d7f75df0b86c83d6b4e52133fc36d1985c7d))
- auth: keep aarch64 ML-KEM sample candidates in registers ([5a94fbe](https://github.com/loadingalias/rscrypto/commit/5a94fbe4e256a923cf0b92ee98e2e5796d3f49ee))
- auth: parse aarch64 ML-KEM sample tails from XOF state ([75b1551](https://github.com/loadingalias/rscrypto/commit/75b155165683ec34555a7f170b83ead8e6a93005))
- hashes: schedule Linux aarch64 ML-KEM batching ([9c6ecb9](https://github.com/loadingalias/rscrypto/commit/9c6ecb9b24b949c9f3b501542df6ff1497b177a9))
- hashes: add Linux aarch64 SVE2-SHA3 Keccak x4 ([315cc93](https://github.com/loadingalias/rscrypto/commit/315cc93f02762662d038e8e904bb954fd369e737))
- auth: parse aarch64 ML-KEM quad samples from XOF state ([56e234a](https://github.com/loadingalias/rscrypto/commit/56e234ab9fd14497265e21300fae59e292996997))
- auth: add aarch64 ML-KEM SampleNTT NEON extractor ([c7c2d89](https://github.com/loadingalias/rscrypto/commit/c7c2d89294cd40135c312d1db2ef114346e0b66f))
- auth: fuse aarch64 ML-KEM K-way accumulate ([95ea2e5](https://github.com/loadingalias/rscrypto/commit/95ea2e5da68a0f4597d6d5dcd505845174449c83))
- auth: tighten aarch64 ML-KEM product-domain reduction ([bf5bfe5](https://github.com/loadingalias/rscrypto/commit/bf5bfe5b3b1e7c3606c3d80435ce8c77dc9229f0))
- auth: vectorize aarch64 ML-KEM product-domain conversion ([c665306](https://github.com/loadingalias/rscrypto/commit/c66530630cee3c9e02c6800123ec1896dd158045))
- auth: fix aarch64 ML-KEM NTT canonicalization ([1e41083](https://github.com/loadingalias/rscrypto/commit/1e41083b9bb0f95177c5e04bde36b674c17c3fbd))
- auth: dispatch Linux aarch64 ML-KEM NTT asm ([67cee6d](https://github.com/loadingalias/rscrypto/commit/67cee6d5276d7d6fef0e38f9fd94f756eb9034b4))
- auth: add fused aarch64 ML-KEM basemul diagnostics ([67caab7](https://github.com/loadingalias/rscrypto/commit/67caab7a540ceb96f80612e8d3098b2afa9d5923))
- auth: vectorize s390x ML-KEM product-domain conversion ([6453f04](https://github.com/loadingalias/rscrypto/commit/6453f042b8fa5a3a4d4bb6bd5f018e533218dcd3))
- auth: batch s390x ML-KEM dot products under CT roots ([4b8f7f4](https://github.com/loadingalias/rscrypto/commit/4b8f7f4c10a2d5a26cfd11877acd84465de86a32))
- auth: prove s390x ML-KEM vector kernels in CT artifacts ([26f80ac](https://github.com/loadingalias/rscrypto/commit/26f80ac29ebdf5dbefaa47a7d971882b226e0bc1))
- auth: add s390x z/Vector ML-KEM NTT kernels ([39cfe62](https://github.com/loadingalias/rscrypto/commit/39cfe62aefef6b8f5df92e7574deb787b2b59a61))
- auth: use materialized ML-KEM matrix path on s390x ([420517b](https://github.com/loadingalias/rscrypto/commit/420517b69742b176fb7ce27657a21f76460021e9))
- auth: fix s390x ML-KEM barrier build mode ([75e7d0a](https://github.com/loadingalias/rscrypto/commit/75e7d0a8821a4cdaa89082e1fef3e00f9334fcd5))
- auth: harden s390x ML-KEM constant-time arithmetic ([446e3d4](https://github.com/loadingalias/rscrypto/commit/446e3d46fc2b3298b79d502904bab4bb0131e1c8))



## [0.5.0](https://github.com/loadingalias/rscrypto/compare/v0.4.1...v0.5.0) - 2026-06-14

### 📝 Documentation

- prepare public docs for v0.5.0 release ([1ba0795](https://github.com/loadingalias/rscrypto/commit/1ba0795447f755ab820d3514cfcd0e06debef5e2))

### 📦 Other Changes

- auth: fix RSA-2048 leakage fixture policy ([c8f6886](https://github.com/loadingalias/rscrypto/commit/c8f6886f343c3771d7bdab14261b703b7277b8a4))
- crypto: harden secret handling and CT validation paths ci: scope CT evidence to required primitives and repair macOS RSA fixtures docs: align migration guidance with hardened verification defaults bench: refresh crypto benches for typed APIs checksum: clarify CRC64 reference constants workspace: align feature metadata and lockfiles for CT tooling ([30ddfb6](https://github.com/loadingalias/rscrypto/commit/30ddfb6632d2d364574e9c4379775cb669219f53))



## [0.4.1](https://github.com/loadingalias/rscrypto/compare/v0.4.0...v0.4.1) - 2026-06-13

### 📝 Documentation

- make public docs user-facing and add ECDSA migration guides benchmarks: refresh 2026-06-12 benchmark evidence ([4a3f4e8](https://github.com/loadingalias/rscrypto/commit/4a3f4e88ba04c324f215785b26b934b7f6133e22))

### 📦 Other Changes

- auth: route RSA blinding inverse through fixed scratch ([a33fc67](https://github.com/loadingalias/rscrypto/commit/a33fc67613dde3640139863b3668d808ad53d71f))
- auth: route macOS aarch64 HKDF-SHA256 through SHA2 compression hashes: batch Apple SHA3 Keccak absorb blocks bench: scale README perf chart axis from benchmark data benchmarks: refresh 2026-06-12 benchmark evidence ([f9ab35f](https://github.com/loadingalias/rscrypto/commit/f9ab35f19f3601b8f32b64efec4dd0d738975284))
- auth: harden HMAC pads against AArch64 SVE division ci: disable native RISC-V Rust cache restore ([62be628](https://github.com/loadingalias/rscrypto/commit/62be628d1ceb3d347ffcbb7ef3af67f046bc22ac))
- auth: harden ECDSA P-256/P-384 CT backends ci: add ECDSA DudeCT diagnostics and target-scoped CT policy ([82db892](https://github.com/loadingalias/rscrypto/commit/82db8924e88136ac3070e892602cb38f4a25d620))
- auth: add ECDSA P-256/P-384 signing and CT coverage ([f24375d](https://github.com/loadingalias/rscrypto/commit/f24375d31784527ee0964b9dd871d64d5c0a6991))



## [0.4.0](https://github.com/loadingalias/rscrypto/compare/v0.3.1...v0.4.0) - 2026-06-09

### 🏗️ Build

- add light and full push preflight commands ci: harden BINSEC solver setup and CT diagnostics ([5a8c2eb](https://github.com/loadingalias/rscrypto/commit/5a8c2ebeaa48989b0fe788b661ab1f5922f17148))

### 👷 CI

- load BINSEC proof relocation sections ([6421da9](https://github.com/loadingalias/rscrypto/commit/6421da978e6f6f87238fc488d39d50014834a2ac))
- build BINSEC proof harnesses as non-PIE ([783eac4](https://github.com/loadingalias/rscrypto/commit/783eac422c783d8170640898786e6b496f298402))
- preinstall BINSEC solver system packages ([df71e54](https://github.com/loadingalias/rscrypto/commit/df71e540e83cb99885fd43539b0f59dfda53e412))
- harden manual CT DudeCT filters ([55ca702](https://github.com/loadingalias/rscrypto/commit/55ca702d86e1a222ccbb3f52d3e11f5de04432b2))
- add s390x AES AEAD DudeCT trace cases ([53812a5](https://github.com/loadingalias/rscrypto/commit/53812a5fae16995dd72f0df29a3d886725d7cd5c))
- add AES-GCM-SIV DudeCT trace cases ([5e6a24f](https://github.com/loadingalias/rscrypto/commit/5e6a24fa1dc76de5709abf2303a30e60b6ce7987))
- add DudeCT filters for targeted CT runs ([fcc326e](https://github.com/loadingalias/rscrypto/commit/fcc326e1b1f00e8dfd3dc59d3e147094564d8a64))
- scope RSA CT evidence and pass BINSEC SMT timeout auth: harden RSA modular import fixed-width output ([c93dc79](https://github.com/loadingalias/rscrypto/commit/c93dc79f6d1c6df3b7aadcc5846e86c3cd17b74b))

### 📦 Other Changes

- workspace: refresh release package metadata, ignore rules, and lockfile pins ci: bump action pins and harden check, coverage, and fuzz scripts docs: align release docs, CT policy, examples, and module snippets with 0.4.0 benchmarks: refresh 2026-06-09 overview and README perf chart ([147c747](https://github.com/loadingalias/rscrypto/commit/147c747b17ce61d7bb3ea3e46ded2a74875bb002))
- aead: align aegis256 AES helper cfgs on POWER and s390x ci: repair CT asm heuristic parsing and RISC-V BINSEC policy docs: narrow RISC-V CT evidence claims ([7dbf097](https://github.com/loadingalias/rscrypto/commit/7dbf097a8db537fc85943f5ae6b2fd2dcc06342b))
- crypto: harden asm dispatch and backend equivalence gates ([643dd44](https://github.com/loadingalias/rscrypto/commit/643dd44a9ebecefa5de99f73bc4934fa86846144))
- aead: batch s390x AES-GCM-SIV CTR keystream blocks ci: route AES AEAD CT evidence through secret-only probes ([e9676b7](https://github.com/loadingalias/rscrypto/commit/e9676b746eca6885fde17962c004f80c8aca9b9e))
- hashes: fix Blake2b diag multiblock oracle ([053c810](https://github.com/loadingalias/rscrypto/commit/053c8106401ea1cc0078e869f8883ab6059d23fc))
- auth: clear CT helper slice lints hashes: clear Blake2b diagnostic slice lints ([32f0e12](https://github.com/loadingalias/rscrypto/commit/32f0e12c5cfb5b0a15e72651d8aff57af8c66a9a))
- auth: align RSA keygen with FIPS 186-5 A.1.3 ([5ceb703](https://github.com/loadingalias/rscrypto/commit/5ceb703cea4b5355eff022fbb1013f2bdcf30e19))



## [0.3.1](https://github.com/loadingalias/rscrypto/compare/v0.3.0...v0.3.1) - 2026-06-01

### 📦 Other Changes

- workspace: enable cargo-rail release publishing ([bb7ec88](https://github.com/loadingalias/rscrypto/commit/bb7ec88d59a2cce916d82af9b182f4905be6600b))
- bench: add Ascon coverage and refresh HMAC measurement shape ([b06b946](https://github.com/loadingalias/rscrypto/commit/b06b946d217248d634c43688211c9ebe5c2692e8))



## [0.3.0](https://github.com/loadingalias/rscrypto/compare/v0.2.0...v0.3.0) - 2026-05-28

### 🏗️ Build

- allow dev-only RustCrypto rsa audit oracle ([ea83045](https://github.com/loadingalias/rscrypto/commit/ea8304559ea2b4f917c30f462376f49c55481d7c))

### 👷 CI

- focus Miri on UB-risk coverage ([c358b72](https://github.com/loadingalias/rscrypto/commit/c358b72cd897df3e5ff10d9c58b07fe6bd03f5a1))
- harden weekly validation timeouts and SHA3 fuzz build ([0705bfd](https://github.com/loadingalias/rscrypto/commit/0705bfdc2a4026f7d11437f90e89e175d91d55b8))
- scope workflow cancellation and widen native lane timeouts ([26845c8](https://github.com/loadingalias/rscrypto/commit/26845c85530d90725d3ad90c7871d7a474d80d27))
- ignore dev-only rsa audit advisory and refresh action pins workspace: refresh dependency lockfile ([9ea0db1](https://github.com/loadingalias/rscrypto/commit/9ea0db181d9cd1072a5df0bcb0b61cdba4f3fdd2))

### 📦 Other Changes

- auth: drop brittle RSA scratch allocation setup count ([ee4c7fb](https://github.com/loadingalias/rscrypto/commit/ee4c7fb2b136ff8fd11a9e5f5cf8401d0ad2add3))
- workspace: refresh RSA package metadata and docs benchmarks: refresh 2026-05-27 Linux scorecard bench: add RSA scorecard row to README chart auth: clear RSA public clippy lints ([e65cb1c](https://github.com/loadingalias/rscrypto/commit/e65cb1c125f87633254e16895b72b4a2793aa458))
- auth: clear RSA portable-only clippy lints ([cc12c7c](https://github.com/loadingalias/rscrypto/commit/cc12c7cdaa89e69216c0c5714cf39d0434eb4698))
- auth: defer RSA public Montgomery precompute and widen 8192-bit verify backends ([3b2a991](https://github.com/loadingalias/rscrypto/commit/3b2a9911ec7a50654b01ef2ee18aadc239062e9e))
- auth: complete RSA private ops, protocols, and assembly backends ([218d15c](https://github.com/loadingalias/rscrypto/commit/218d15c7c4bf154a90f86c036d777d4875472c96))
- workspace: allow dev-only RustCrypto rsa advisory oracle ([8a7d608](https://github.com/loadingalias/rscrypto/commit/8a7d608e8a7cc81071fce980e42f9ce9b94c2543))
- auth: add RSA verifier, vectors, fuzzing, and Ed25519 assembly backends bench: add RSA verification benchmarks and fixtures workspace: wire RSA dependencies and tracked fuzz corpora ([a3d1e79](https://github.com/loadingalias/rscrypto/commit/a3d1e7920776c7bd9dc751de137f223016673451))
- workspace: refresh README release snippets and local asset ignores ([1d838fe](https://github.com/loadingalias/rscrypto/commit/1d838febe1e8d2d751b8c38d32d838b9a5c2db04))
- workspace: fix fuzz support path dependency and refresh locks ci: refresh weekly action pins ([a906007](https://github.com/loadingalias/rscrypto/commit/a906007a4b91c691c99ada30883243c714061cd2))



## [0.2.0](https://github.com/loadingalias/rscrypto/compare/v0.1.1...v0.2.0) - 2026-05-17

### 📦 Other Changes

- benchmarks: refresh 2026-05-17 release scorecard ([3a3a479](https://github.com/loadingalias/rscrypto/commit/3a3a47946d3aa7b3ff0de979a802bfba3fb7a972))
- aead: fuse POWER AES-GCM CTR and GHASH paths hashes: route x86 SHA-256 native streaming before compile-time kernels bench: reuse HMAC state in SHA-384 and SHA-512 benches build: parameterize Linux dev machine just aliases ([1c30a68](https://github.com/loadingalias/rscrypto/commit/1c30a68b667f964982a519a938065f98989322b2))
- bench: cfg-gate aead kernel bench by arch ([2cc1000](https://github.com/loadingalias/rscrypto/commit/2cc1000751a68ec3613a38b078f20f1441d42fb8))
- benchmarks: refresh overview and harden chart parsing ([36b252b](https://github.com/loadingalias/rscrypto/commit/36b252be035f4a589c8813a68cf38afc32b5d0b2))
- aead: extend aarch64 AEAD assembly to Linux auth: add s2n-bignum X25519 assembly backends hashes: remove AES hash primitive from fast hashes bench: drop gxhash competitor benches and fix X25519 aws-lc output benchmarks: refresh benchmark overview after full extraction workspace: drop AES-hash feature and competitor deps ([59b2b37](https://github.com/loadingalias/rscrypto/commit/59b2b374815c063263f50ddebf7870d07cf65581))
- aead: fuse aarch64 ChaCha20-Poly1305 and batch Poly1305 NEON ([7609841](https://github.com/loadingalias/rscrypto/commit/7609841719881f8387f8453fe51581e42ffecbf1))
- aead: fuse x86 AES-GCM assembly with vector counters ([56f75c5](https://github.com/loadingalias/rscrypto/commit/56f75c5f8679325bc6eba89cdf69d5cf5f050ded))
- aead: add AArch64 and x86-64 AES-GCM assembly kernels benchmarks: refresh AES-GCM status ([11ebc4a](https://github.com/loadingalias/rscrypto/commit/11ebc4aed956710955f83100aa438c435b3aaaf9))
- crypto: drop s390x target-feature inline hints ([ff3060e](https://github.com/loadingalias/rscrypto/commit/ff3060e2c0a696884e531a6c18616df828457a5a))
- aead: drop aarch64 AES-GCM scheduling barrier ([4702023](https://github.com/loadingalias/rscrypto/commit/4702023bc56edef99b04b0e7492af8c1d10d7a9e))
- aead: format aarch64 AES-GCM chunk helpers ([31339f2](https://github.com/loadingalias/rscrypto/commit/31339f2cc954ab3e3a066b0bb7c48e5b28a60659))
- aead: batch AES-GCM and GCM-SIV arch kernels ([2088e3f](https://github.com/loadingalias/rscrypto/commit/2088e3f8fd731640772e7162594f6bfe45b493b4))
- bench: fix AEAD required feature gate build: restore s390x inline target-feature gate ([48c75d6](https://github.com/loadingalias/rscrypto/commit/48c75d61782ed186e48ca084fad853b2c0efe8be))
- aead: route AES-GCM through backend-wide GHASH and s390x CTR batching workspace: gate examples and benches by required features ci: include AES-128 GCM benches in CI selectors benchmarks: refresh benchmark overview for expanded AEAD coverage ([16983ef](https://github.com/loadingalias/rscrypto/commit/16983ef467eb2c839df9dd0b8542a82cbeeab64c))
- workspace: gate gxhash dev-dependency on AES SIMD targets bench: skip gxhash competitor rows on unsupported targets ([dc3cf94](https://github.com/loadingalias/rscrypto/commit/dc3cf94f11e0c73f50ad989039c16b617cf1d414))
- aead: ship AES-128-GCM and AES-128-GCM-SIV with full SIMD kernel coverage hashes: add Blake3KeyedHash type with constant-time equality bench: add aws-lc-rs, ring, dryoc, gxhash, ahash, and foldhash competitor rows workspace: wire competitor dev-deps, ungate internal hex module, refresh docs ([0a39948](https://github.com/loadingalias/rscrypto/commit/0a39948a73df05c816163dc7c49b043efc976326))
- workspace: sharpen adoption docs and publish migration guides ([30f9846](https://github.com/loadingalias/rscrypto/commit/30f9846d52a8e6ce0a3d39adf9e0dacfda5cd5f4))



## [0.1.1](https://github.com/loadingalias/rscrypto/compare/v0.1.0...v0.1.1) - 2026-05-02

### 🏗️ Build

- trim tests/testdata/benches from published crate include list docs: fix README quick start imports and add Xxh3 FastHash trait workspace: wire README into doctest harness via ReadmeDoctests hook ([f16a44f](https://github.com/loadingalias/rscrypto/commit/f16a44f5db5527f77ad294e717c64f1b12930bdb))



## [0.1.0](https://github.com/loadingalias/rscrypto/releases/tag/v0.1.0) - 2026-05-02

### 🐛 Bug Fixes

- cicd issues around Blake3 kernel selection and unused s390x gating ([621e80b](https://github.com/loadingalias/rscrypto/commit/621e80b7e8a179e2a56270545a3f296236716864))
- AESE for subword during key expansion ([09f2ace](https://github.com/loadingalias/rscrypto/commit/09f2ace9f39d0433249648294d58f6508687774a))
- removing the dead code 'square_wide' in favor the the 'square_and_negate_d_wide'. ([8f4dc6c](https://github.com/loadingalias/rscrypto/commit/8f4dc6c23994d9991bd28f4e4b08fbb75f7c02d9))
- removing the dead code around IFMA cleanup ([04c3c41](https://github.com/loadingalias/rscrypto/commit/04c3c415c340ee43f38a7fa30922ce68b878bf1d))
- **auth**: reduce before second mul in AVX2 double — vpmuludq ×19 overflow ([bda61b6](https://github.com/loadingalias/rscrypto/commit/bda61b6791666cc50a6d384f48c367643f42e2f7))
- **auth**: revert AVX2 verify dispatch — AVX2 Straus has a latent bug ([af32c11](https://github.com/loadingalias/rscrypto/commit/af32c11c43cf4ecae97d1ee62806ca73def1afd5))
- **auth**: remove broken IFMA field arithmetic — vpmadd52luq truncates products to 52 bits ([4880a42](https://github.com/loadingalias/rscrypto/commit/4880a42b8df6e0b7b77f685459f94c854c18c823))
- **aead**: emit raw VCIPH bytes on s390x — LLVM lacks the mnemonic and .insn vrr ignores %v<N> operands ([f161381](https://github.com/loadingalias/rscrypto/commit/f1613819b2bb36a0ddead7f6c4d69331253799ec))
- s390x accel: vciph {out}, {block}, {rk} with vreg register class ([81f5125](https://github.com/loadingalias/rscrypto/commit/81f5125d10b223094afe7340e76285689e2d67ac))
- **aead**: pin AEGIS-256 s390x VCIPH registers to V0-V2 ([4b15ae5](https://github.com/loadingalias/rscrypto/commit/4b15ae5f884b458fb40f8770be08c5f1be114b10))
- wrap x86_64 intrinsic bodies in unsafe blocks ([3c8602c](https://github.com/loadingalias/rscrypto/commit/3c8602c2e3fc39a8ebbb9bb27fc0905df83da3e2))
- gate platform-specific test helpers to silence warnings on s390x/power aead: add aes-256-gcm-siv with portable constant-time aes and polyval ([7d4ef59](https://github.com/loadingalias/rscrypto/commit/7d4ef59d192416f219c5efd8e0d5454a43a9b724))
- strict_* arithmetic across checksum module ([a11c7cd](https://github.com/loadingalias/rscrypto/commit/a11c7cd759f7cef4a57b2f8015e8464ae3092c92))
- add #[cfg_attr(miri, ignore)] to hashes prefetch inline-asm tests ([824c2c2](https://github.com/loadingalias/rscrypto/commit/824c2c2aca4cc826b9f6ba692c38f2a4db2f3ff5))
- gate bench streaming_dispatch_info behind parallel feature ([e10f9b1](https://github.com/loadingalias/rscrypto/commit/e10f9b1d74dbfa5496cd8eb07916122953682ebc))
- IBM runner clippy lint fix ([fe78b48](https://github.com/loadingalias/rscrypto/commit/fe78b48f9ca6a2971c21bfda4ec60cc6b90a0f25))
- tuning workflow, shape ([03ed73e](https://github.com/loadingalias/rscrypto/commit/03ed73e62877aeed7c3ffb8ce6f0ce5b38b753aa))
- tuning workflow ([3230e85](https://github.com/loadingalias/rscrypto/commit/3230e85a2e9eeecee8fd8ec48e3a9cc94dc923aa))
- ci - rustfmt fix ([660a8af](https://github.com/loadingalias/rscrypto/commit/660a8aff3e10f1b6392e9ccb207d70a13e22291a))
- **checksum**: add missing extern crate imports in crc32 x86_64 tests ([08dc6e4](https://github.com/loadingalias/rscrypto/commit/08dc6e459a333bfb92762c4de8c672a46ed7bc39))
- **platform/backend**: refactor platform && add backend; fix UB in no_std dispatcher ([dbf2e3b](https://github.com/loadingalias/rscrypto/commit/dbf2e3b66810d78b16ef513159a98964faed0430))
- **checksum**: align x86_64 SIMD CRC implementations with ARM approach ([a088d2d](https://github.com/loadingalias/rscrypto/commit/a088d2d1b9ae5b82b459cdc9c33545473d238efb))

### 🏗️ Build

- pin nightly-2026-04-18; raise MSRV to 1.95.0 and trim stale nightly gates checksum: use cold_path for tiny CRC oneshots ([def943a](https://github.com/loadingalias/rscrypto/commit/def943a55a376755b43bf18008a9df97c9e64539))

### 👷 CI

- drop coverage smoke and capture fuzz workspace coverage via per-invocation cargo-llvm-cov build: trim rail.toml comments ([89b2c72](https://github.com/loadingalias/rscrypto/commit/89b2c721f14c474d0ac017c270cba953023a205f))
- anchor pre-push hook on repository root ([1290bb1](https://github.com/loadingalias/rscrypto/commit/1290bb1e38545a310b3f8f086c3fc9ce76d7f489))
- harden action upgrades, pre-push checks, and fuzz coverage build: route just update through action refs and fuzz manifests ([f382065](https://github.com/loadingalias/rscrypto/commit/f3820659c59fb235f1811f1cb646bd3a142ebdbb))
- restore all weekly fuzz corpora for total Codecov coverage ([483906e](https://github.com/loadingalias/rscrypto/commit/483906e897841c33769649a93e12b23317446637))
- upload total llvm-cov coverage and widen weekly timeouts workspace: make fuzz targets replayable under llvm-cov checksum: skip CRC32 VPCLMUL tests under Miri ([fb8b926](https://github.com/loadingalias/rscrypto/commit/fb8b92692909ccb814129e75ed052056026c84df))
- harden weekly validation and scoped fuzz coverage ([cb06972](https://github.com/loadingalias/rscrypto/commit/cb069723fd5753dfe098c34c3a4b32eee0f8a16e))
- harden weekly validation and scoped fuzz coverage ([101da93](https://github.com/loadingalias/rscrypto/commit/101da93dc47b636bfcfcf2b184ff4ff961562c75))
- keep native quality lanes on the shipped library surface ([c8cdce5](https://github.com/loadingalias/rscrypto/commit/c8cdce5f1aa179fc387a8a6cd8b37f5fdd86ae33))
- constrain riscv64 runner concurrency and skip rustdoc ([0fedbdf](https://github.com/loadingalias/rscrypto/commit/0fedbdf53ef808cb40a6b4b946201f32fca4a4f4))
- bump cargo-rail to 0.13.0; adopt unknown file policy; update docs perf: batch riscv AES CTR, restore blake3 small-XOF path, and retune aarch64 sha256 checksum: split CRC oneshots into inline-always fast path + cold dispatch ([6b61463](https://github.com/loadingalias/rscrypto/commit/6b614631a0a97ca7b4dd0a72ba476a6528fa7f38))
- surface real feature-matrix compile errors ([f2c2b50](https://github.com/loadingalias/rscrypto/commit/f2c2b503f6e6cfeb716401795a63b7d4cbcd72d0))
- adopt cargo-rail 0.11 scope-driven planner flow ([face19d](https://github.com/loadingalias/rscrypto/commit/face19d912a6c811698054532e503da0cd7b8e38))
- remove redundant unsafe blocks across all SIMD modules, fix ppc64 doc comment and s390x dead code hashes: split blake3 control modules and cut non-kernel update overhead checksums: improve the usage of the 'strict_*' for all algys ([06ced8f](https://github.com/loadingalias/rscrypto/commit/06ced8f334ef2ba138dcc4e4deecf58b85513c7c))
- fix blake3 selector planning for comp/kernels bench scopes ([3ee8585](https://github.com/loadingalias/rscrypto/commit/3ee8585f0794373cd7f5d651e6edf21072b7554e))
- block unscoped hashes/comp bench runs and selector fallback ([3cbfa0c](https://github.com/loadingalias/rscrypto/commit/3cbfa0c6dfab55de3f8f2af57f8a4c670ce54973))
- installing the weekly.yaml toolchains and tools (miri/cargo fuzz) ([85524dd](https://github.com/loadingalias/rscrypto/commit/85524dd383a852604bbfcfd311072c5b662a3e2b))

### 📝 Documentation

- rewrite release.md with full test/security inventory and focused v0.1.0 plan ([6850028](https://github.com/loadingalias/rscrypto/commit/68500288551fc6cd3782f547ed879e1c5ecec329))

### 📦 Other Changes

- workspace: isolate fuzz replay from release package build: report fuzz coverage from replay manifests ([46cff74](https://github.com/loadingalias/rscrypto/commit/46cff74cc4c6845251e27f9920e5d6f9508114de))
- checksum: gate aarch64 PMULL EOR3 tables out of Miri ([7d50e3b](https://github.com/loadingalias/rscrypto/commit/7d50e3b7239bbb0f4f7a097f770193cb9dae1230))
- workspace: satisfy fuzz support dependency audit ([d9593fb](https://github.com/loadingalias/rscrypto/commit/d9593fb27abc5d6b18fda2ad8c2dad6bbd3e0e1f))
- workspace: gate fuzz replay tests and forward AEAD support ([268860c](https://github.com/loadingalias/rscrypto/commit/268860c94b97198add71aa57745a3cc17b092601))
- auth: skip AVX-512 IFMA tests under Miri ci: preserve fuzz coverage profiles and skip empty fuzz Codecov uploads ([347693d](https://github.com/loadingalias/rscrypto/commit/347693d188e3158c0e9304e3d1a23d90ec94327b))
- workspace: prepare release metadata and split README reference docs build: harden coverage reports and refresh local validation lanes auth: adapt scrypt oracle params and simplify curve25519 square roots hashes: remove dead Ascon dispatch adapter bench: align password hashing bench with scrypt 0.12 benchmarks: prune stale raw runs and refresh overview ci: refresh pinned action SHAs ([d7f65e6](https://github.com/loadingalias/rscrypto/commit/d7f65e6c07e0ce19affcc816f059375304135d12))
- hashes: format Keccak platform cfg ([dff5817](https://github.com/loadingalias/rscrypto/commit/dff5817834fa412ed08e9ddf62a018e2e7821379))
- aead: tighten leaf-feature cfgs for target-policy helpers auth: skip Argon2 oracle scaffolding under Miri hashes: route Keccak Miri builds through the portable permuter workspace: fix introspect features and refresh release docs benchmarks: refresh 2026-04-28 Linux CI overview ([76b243d](https://github.com/loadingalias/rscrypto/commit/76b243d85897d122041f4f66c56c22e598fd50d7))
- crypto: split secret serde and enforce checked bit-length framing aead: narrow dispatch introspection and drop riscv64 AEGIS table fallback auth: return PHC entropy failures from random-salt hashing checksum: document riscv64 CRC unsafe blocks hashes: fast-path RapidHash empty input and inline Blake2 one-shots workspace: refresh docs, examples, and feature-matrix coverage benchmarks: refresh overview from 2026-04-28 Linux CI ([0ea1b73](https://github.com/loadingalias/rscrypto/commit/0ea1b73b2a40fd164b702e8b927d5930a996f846))
- hashes: elide public Blake2 wipes and preserve keyed cleanup bench: remove Blake2 forced-kernel diagnostic benches ci: fix generic bench-quick and keep Blake2 benches off diag benchmarks: refresh overview from 2026-04-28 results ([90043b7](https://github.com/loadingalias/rscrypto/commit/90043b7aeb9d55819ce54f13b22a90c8f0cc10aa))
- hashes: split Blake2b counters and prune stale XXH3 diagnostics ([f1a6e96](https://github.com/loadingalias/rscrypto/commit/f1a6e967a4653abdf66b55d712ec6f61e2e4c6b7))
- hashes: harden Blake2 zeroization and add fast-hash diagnostics bench: add CRC, SHA2, AUTH, and XXH3 diagnostic probes ci: enable diag features for SHA2, AUTH, and XXH3 benches benchmarks: refresh release benchmark overview ([0da9bca](https://github.com/loadingalias/rscrypto/commit/0da9bca9fa61ac0b4dae37541c17e7b4027a4f1d))
- aead: add x86 ChaCha20 SSSE3 path and tighten AES-GCM fast paths hashes: batch Blake2 compression and tune fast-hash dispatch benchmarks: refresh 2026-04-26 CI overview ([e65388d](https://github.com/loadingalias/rscrypto/commit/e65388d85f72ac813d0fd29a28e01c8f1a623f70))
- aead: fix AEGIS aead-only cfg on s390x and riscv64 hashes: clear POWER RapidHash clippy lint ([0047046](https://github.com/loadingalias/rscrypto/commit/004704622b3e4be7f851254f434e1c8e22e0bc15))
- aead: add riscv64 fixslice AES fallback and tighten GCM tag path hashes: use referenced RapidHash v3 secrets on s390x and POWER benchmarks: refresh overview from 2026-04-26 CI and macOS benches ([6703515](https://github.com/loadingalias/rscrypto/commit/6703515f25eb36a85764d94fdede4fb88079f55a))
- workspace: gate riscv64 blake3 portable SIMD feature ([f5c6ff7](https://github.com/loadingalias/rscrypto/commit/f5c6ff7cb80b3cd168998dd152e934751817ec0d))
- Update mod.rs ([65a3c32](https://github.com/loadingalias/rscrypto/commit/65a3c322e5cfe68ff44a4ec6a72acf26373f95f5))
- aead: retune AEGIS and AES-GCM dispatch across target backends hashes: fast-path RapidHash defaults and medium inputs auth: add random-salt PHC helpers and verification policy docs bench: gate RISC-V Blake2 diagnostics and refresh bench scripts benchmarks: refresh overview from latest CI and macOS benches workspace: clean public docs, examples, package include, and security guidance ([e24d79a](https://github.com/loadingalias/rscrypto/commit/e24d79a971a82d3f0ea004dea782c925ea9086ba))
- hashes: mark losing Blake2 kernels diagnostic-only ([7d2d811](https://github.com/loadingalias/rscrypto/commit/7d2d8119622af1f2f77a5c2d2a9bd9457dc76298))
- hashes: gate losing Blake2 SIMD and add riscv64 diagnostics crypto: clear feature-gated CT and dispatch warning paths bench: add Blake2 forced-kernel comparisons ci: include Blake2 forced-kernel diagnostics workspace: narrow stale RISC-V nightly gates ([b62a082](https://github.com/loadingalias/rscrypto/commit/b62a082f3c836e3c927fc2b9e2e46a9200ba9410))
- hashes: fix AVX-512 Blake2 compile-time dispatch ci: make CPU diagnostics pipefail-safe ([70989b0](https://github.com/loadingalias/rscrypto/commit/70989b035bcf49b67af87812f1257a1edc32d284))
- auth: add Argon2 family, scrypt, and PHC string format hashes: tune sha256, blake3, keccak, sha3, sha384/512, and xxh3 dispatch aead: harden chacha20 ppc64 VSX and AEAD nonce counter bench: add kmac/cshake and password-hashing benches ci: consolidate workflows and drop runson/warpbuild adapters workspace: bump deps and surface Argon2, scrypt, and PHC on the root build: refresh justfile and cargo lanes for the new test layout ([2a76963](https://github.com/loadingalias/rscrypto/commit/2a7696349cea4bffe212cfe4cb3b70a431da6444))
- hashes: fix POWER sha256 correctness and ascon cfg linting ([a70c659](https://github.com/loadingalias/rscrypto/commit/a70c659672a2e3fd0bf7808a99cda8544db36e1c))
- hashes: restore ppc64 sha256 and tighten ascon hot paths ([abfbb8f](https://github.com/loadingalias/rscrypto/commit/abfbb8ff1ce19ce1b1dd6def85dfd3c324c56a14))
- workspace: split crypto backends and harden validation ([1a0bf2d](https://github.com/loadingalias/rscrypto/commit/1a0bf2dbbb2f535ad5a2ef97775576dd377a95cd))
- auth: fix pbkdf2 oracle digest-version mismatch ([7792145](https://github.com/loadingalias/rscrypto/commit/77921456944302f9ab745ebdfba713dd24a2aee5))
- checksum: keep riscv64 crc64 auto portable and drop stale riscv gates workspace: bump rayon, getrandom, and pbkdf2 deps ([07a38d0](https://github.com/loadingalias/rscrypto/commit/07a38d04bb073529cd8ae7342d3eda20306f81a5))
- clippy ([98eb8df](https://github.com/loadingalias/rscrypto/commit/98eb8df84bcf07d2d4ef35f3bd800fbaca33078a))
- auth: use fixed-base Edwards path for x25519 public keys ([84674be](https://github.com/loadingalias/rscrypto/commit/84674bee13bc5ef9d26a304869d43204d9a38333))
- blake3: byte-pack short root output and refresh benchmark overview ([e8c4b66](https://github.com/loadingalias/rscrypto/commit/e8c4b66d5a1432387aedabecdefaf2de69586ce3))
- hashes: fast-path one-block blake3 xof reads checksum: retune 64B dispatch thresholds on riscv64, s390x, and power10 aead: cache riscv64 aes-256-gcm-siv master key expansion bench: normalize extracted CI results and refresh overview ([70856ea](https://github.com/loadingalias/rscrypto/commit/70856ea1c10cb66493904419b03a7f317f973c71))
- aead: batch riscv64 vperm AES ECB blocks checksum: widen riscv64 crc64 zbc folding to 8-way hashes: fast-path xxh3-64 empty input ([21a84f1](https://github.com/loadingalias/rscrypto/commit/21a84f1fe582e3ae7a5beeb3edffd2845456a6f9))
- benchmarks: 2026-04-18 CI results, regenerate OVERVIEW, add BENCHMARKS.md ([a165fd9](https://github.com/loadingalias/rscrypto/commit/a165fd9508555535e3467587ba97218d520d4cbe))
- hashes/aead: fix s390x/ppc64le rotate direction bug in Blake2s and ChaCha20 ([d873e91](https://github.com/loadingalias/rscrypto/commit/d873e91debb6a8be4ea1efa0ad2cfd5fc71e6a0d))
- hashes: clear Blake2 params clippy lints ([b56684d](https://github.com/loadingalias/rscrypto/commit/b56684da2b494fb431b8c219fa3b9d5508cdbfd8))
- clippy ([7572ff7](https://github.com/loadingalias/rscrypto/commit/7572ff7d5b16c52fb70f4555f4e1b99743aedf88))
- hashes: align Blake2 API, add params block, tune Blake2s NEON ror8 ([3e1ea72](https://github.com/loadingalias/rscrypto/commit/3e1ea728935f95fec88dcf816fa61f7575e8ab03))
- hashes: fix s390x Blake2b diagonal lane permutation ([d4de4ee](https://github.com/loadingalias/rscrypto/commit/d4de4ee507c0c99569a123a8414885f95de2b825))
- hashes: restore ppc64le Blake2 POWER kernels ci: upgrade cargo-rail to 0.12.0 and cargo-rail-action to v4.1.1 ([ab17790](https://github.com/loadingalias/rscrypto/commit/ab17790dd6d9d1cc96ec27990a39558d996ec33b))
- hashes: disable ppc64le Blake2 VSX and fix s390x test unsafe ([5bc3d90](https://github.com/loadingalias/rscrypto/commit/5bc3d9012bc561cdae222c66bc858e05e2f6c1f6))
- clippy ([6642a7c](https://github.com/loadingalias/rscrypto/commit/6642a7c166aa3cd1f75ab0f05f68a3a2fc6a0d03))
- crypto: harden Blake2/PBKDF2 and tighten AEAD fast paths ([e4b7ff6](https://github.com/loadingalias/rscrypto/commit/e4b7ff6a6b70bb89df7ec2f9b104b016e10f0489))
- hashes: complete Blake2 backends and harden PBKDF2 quality gates ([2453e09](https://github.com/loadingalias/rscrypto/commit/2453e09141aedd229442ddd400b5e707cb6c7b48))
- fix AEAD/BLAKE2 warnings and cross-target unsafe lints ([106b35e](https://github.com/loadingalias/rscrypto/commit/106b35ecdb3cc04d62e582a9d83d7bdb0b1e3cd0))
- pbkdf fix ([a807fdc](https://github.com/loadingalias/rscrypto/commit/a807fdc4b3dda8f03dbb6e55365359061cb80bbf))
- clippy ([fdf620e](https://github.com/loadingalias/rscrypto/commit/fdf620e56317d1b51188fa32ffe5f25637be7fc0))
- auth: add PBKDF2-HMAC-SHA256/SHA512 with precomputed prefix states, direct compress loops, and constant-time verify APIs aead: add Hamburg vperm AES backends for s390x and riscv64 V, route AEGIS-256 and AES-256-GCM-SIV through the new RISC-V backend tier, and tighten fallback security notes hashes: add Blake2b/Blake2s implementations with portable + multi-arch kernel dispatch; optimize Blake3 XOF/oneshot block paths; reorder aarch64 SHA-256 schedule batching; bypass RVV xxh3 long-path dispatch overhead on riscv64 workspace: add `blake2` dependency, wire `pbkdf2`/`blake2b`/`blake2s` features into bundles, and re-export new auth/hash types ([4e24772](https://github.com/loadingalias/rscrypto/commit/4e24772a54fef3f6932e171f815dadb7cd0e3e8d))
- auth: zeroize HMAC on Drop, black_box ct barriers in KMAC/X25519, fix HKDF-SHA256 zeroize scope hashes: remove ppc64 stubs and superseded SHA-512 kernel variants, prune id_from_name, constant_time_eq in Blake3 key cache aead: add wrong-nonce/buffer-zeroed/wrong-AAD oracle tests, separate AES-128 dispatch ci: riscv64 check-only feature matrix, coverage merge summary workspace: rewrite README, self-contained trait doc examples, add serde + getrandom smoke tests ([722b99f](https://github.com/loadingalias/rscrypto/commit/722b99f6d4fef7689384747109bb73604d516443))
- api: harden key/tag security contracts, close no_std feature-matrix gaps, add oracle tests and trait docs ([2c1f7c0](https://github.com/loadingalias/rscrypto/commit/2c1f7c0a4873afea27e1dede028622d7a00c5314))
- bench: infra, results, scripts, workflow, docs reorganization api: add ConstantTimeEq, to_vec, Read, BuildHasher; also optional getrandom, serde deps ([1744581](https://github.com/loadingalias/rscrypto/commit/1744581ff55fda81b8308961abc7b7f222b9c5dc))
- fuzz: align scoped harnesses with the feature model ([9a054b0](https://github.com/loadingalias/rscrypto/commit/9a054b00e1e656d039ce0438bef3f5e0dcaeda08))
- aead: fix cross-arch test import warnings ([b240ab7](https://github.com/loadingalias/rscrypto/commit/b240ab78d38531924607cf39d1d85783f101ebb4))
- clippy ([3abe206](https://github.com/loadingalias/rscrypto/commit/3abe2063f83da98745df47b0f30dbe863712c2ba))
- clippy ([6067e26](https://github.com/loadingalias/rscrypto/commit/6067e2628ff5095a38f0195d8567ef4b7eebea31))
- workspace: harden feature graph and release contracts via refining the feature selection into real leaf and bundle boundaries, split shared Curve25519 internals so x25519 stands alone, tighten checksum leaf builds, and add public API/error contract coverage. ([0d4afb2](https://github.com/loadingalias/rscrypto/commit/0d4afb237f43986f5bfe2874a662c71a26c38450))
- aead: remove inline(always) from target_feature helpers ([251ae70](https://github.com/loadingalias/rscrypto/commit/251ae70210163537e24effca0fe542e4e42ca13d))
- checksum: add real riscv crc ladders and split crc64 strategy hashes: blake3 fast-path short xof first-block emission aead: enforce riscv backend ladder for aes-gcm-siv ([57059da](https://github.com/loadingalias/rscrypto/commit/57059dab3fbf37c586811ca620c4ac3fb841bd66))
- revert xxh3 riscv work ([e810843](https://github.com/loadingalias/rscrypto/commit/e8108437b6ff481451a00f4e90525ffe65cc8f00))
- riscv: fuse XXH3 RVV stripe loop and demote CRC64 Zbc to portable ([5e5da61](https://github.com/loadingalias/rscrypto/commit/5e5da61ded11dda98e9089595465f9b44aadbb88))
- riscv: add XXH3 RVV kernel and retune CRC/AEAD dispatch thresholds ([9e4ca10](https://github.com/loadingalias/rscrypto/commit/9e4ca10acc47557554f109e28534e348a7ecb666))
- riscv: add scalar crypto backends and enable full target validation ([b6f0b27](https://github.com/loadingalias/rscrypto/commit/b6f0b272658f29988369f903f8d1e5d2b030ff47))
- lib: expose std in test builds for no_std feature-matrix lanes ([bc71974](https://github.com/loadingalias/rscrypto/commit/bc71974eb4e10f854967c6b3aa56b9605ca52c0d))
- auth: qualify ISA detection macros and silence reduced-feature Ascon warnings ([412a511](https://github.com/loadingalias/rscrypto/commit/412a511464923a88a66381b6b81aae5e1f9d75a7))
- hashes: allow dead code for Ascon x86 batch kernels in no_std builds ([d6aedaf](https://github.com/loadingalias/rscrypto/commit/d6aedaf570bc1902f22776d4844cb65bf10e19bc))
- workspace: add fuzzing infrastructure and close auth performance gaps ([a2d544f](https://github.com/loadingalias/rscrypto/commit/a2d544fd8662309d5f61430d6eabf9c88ed5a31a))
- sha256, hmac, xxh3: eliminate dispatch overhead at near-boundary sizes ([ce78300](https://github.com/loadingalias/rscrypto/commit/ce7830095bc63ec63d2f06e3b9433952ff85428f))
- hkdf: raw-state expand loop to eliminate per-iteration Sha256 overhead hmac-sha256: merge compress calls and batch zeroization for small inputs docs: bench updates for Blake3 small/keyed and the ([b4cf488](https://github.com/loadingalias/rscrypto/commit/b4cf488ef5fa0f25fc76885e03c24299615fbc31))
- aegis256: VEX-encode x86 AES rounds and widen all HW loops to 4-block ([2cfa192](https://github.com/loadingalias/rscrypto/commit/2cfa1923ff8d3cc7743e706ca23a95ab3b4fbed2))
- blake3: use avx-512 compress for all x86 size classes on avx-512 cpus ([2b8f600](https://github.com/loadingalias/rscrypto/commit/2b8f6001b21a42372a923883c8e856fc80bd77a4))
- blake3: use assembly compress for all x86 oneshot blocks ([0da76b4](https://github.com/loadingalias/rscrypto/commit/0da76b4eed8683563efe337d3ddc5521617bfa05))
- sha2: align kernel coverage with reachable dispatch backends ([fb571ed](https://github.com/loadingalias/rscrypto/commit/fb571ed66c1c84687546c770fb90fb02a49028ab))
- sha2: restrict kernel tests to runtime-reachable backends ([342141b](https://github.com/loadingalias/rscrypto/commit/342141b14850348a63aa2487e83d8a7122a0f2aa))
- sha2: fix truncated sha512 x86 dispatch and restore kernel coverage ([602efeb](https://github.com/loadingalias/rscrypto/commit/602efeba9dc2b6ff5f4a55fda356dc5fc84dff24))
- blake3: fix x86 XOF root-output crash and restore kernel coverage ([c24b255](https://github.com/loadingalias/rscrypto/commit/c24b25590b49c5c3de3d5bb88f6d8129708f753b))
- ci,auth,aead: remove namespace setup and fix cross-target validation ([a801469](https://github.com/loadingalias/rscrypto/commit/a801469d977d178e9c7d28055c7643552655ca36))
- auth,hashes: tighten Ed25519 verify and speed up Ascon hash/xof ([e6c36cb](https://github.com/loadingalias/rscrypto/commit/e6c36cb2718009e9767b1c1e8d3c1c8afda2d725))
- cicd: fixing the cargo-rail interaction to trigger full builds on main ([2790f5a](https://github.com/loadingalias/rscrypto/commit/2790f5a3665667820a21002cffd2e44ae2a15365))
- ascon: eliminate dispatch indirection and unroll permutation ([7fecdee](https://github.com/loadingalias/rscrypto/commit/7fecdee2bc40127b623cc001c6dd9a3355e868f8))
- bench: fixing the formatting; adding crc16-ibm and aegis workspace: bumping competitors to latest versions; adding ASCON competitor; fixing APIs for hmac/hkdf updated deps ([b85cc4b](https://github.com/loadingalias/rscrypto/commit/b85cc4be10f256a7ecec838a458051ee7fbcbd5b))
- auth/ed25519: switch x86 AVX2 verify fallback to wNAF Straus ([8a8d894](https://github.com/loadingalias/rscrypto/commit/8a8d894c142d8a729badb1344d15328653010cf3))
- auth: reduce ed25519 verify and hkdf expand x86 overhead ([f05b393](https://github.com/loadingalias/rscrypto/commit/f05b3936dd2b8cb922ddfeda1d90691250c01035))
- benches: cleaning up ([44905e4](https://github.com/loadingalias/rscrypto/commit/44905e439734982d6249c41d5d787fe75f163034))
- cicd: fixing the cicd failures ([129b0b5](https://github.com/loadingalias/rscrypto/commit/129b0b554cb33718861e0379e7dad4a70c5b6bba))
- clippy ([c6c2f96](https://github.com/loadingalias/rscrypto/commit/c6c2f9686d3e6be74a1b03c3f461c7adad792fc1))
- clippy fix ([d21aad8](https://github.com/loadingalias/rscrypto/commit/d21aad85ac704370ece0044ad8d849212ecb7b0c))
- clippy ([3ea14ff](https://github.com/loadingalias/rscrypto/commit/3ea14ff5a26818c02b66b728dfd98714c51e4ddb))
- diagnostics for the IBM 390x accel ([8d11a36](https://github.com/loadingalias/rscrypto/commit/8d11a36d924ab0e099354986e5a0fa1776fdbacb))
- aead: fix AEGIS-256 s390x byte order (swap i64x2 doubleword halves) ([69f16a7](https://github.com/loadingalias/rscrypto/commit/69f16a76489b6a8ce37d09efcdb15b26f0cce57a))
- aead: fix AEGIS-256 s390x VCIPH by pinning registers to V0-V15 ([00e3f5b](https://github.com/loadingalias/rscrypto/commit/00e3f5be3b5724507bb8ed29f53c68c4f6d39871))
- clippy ([315b66b](https://github.com/loadingalias/rscrypto/commit/315b66bde22491e884affdca81b9c26cea1963d4))
- clippy ([e5b7ae0](https://github.com/loadingalias/rscrypto/commit/e5b7ae07dc0c2ae931f575a53cd29360904e7b45))
- aead: fix AEGIS-256 s390x SIGSEGV (i64x2 vreg); keccak: aarch64 SHA3-CE single-state kernel ([7aa5a9a](https://github.com/loadingalias/rscrypto/commit/7aa5a9a1c01d4ad2667ff584d01bf41e21479b9f))
- aead: hardware acceleration for AEGIS-256 on s390x/ppc64; optimize XChaCha20 + GCM-SIV small sizes ([81b8ac5](https://github.com/loadingalias/rscrypto/commit/81b8ac5ae48cc22f45b48d38908825c3e0ee9cc9))
- aead: fix POWER8 AES register encoding for VSX instructions ([d259d6a](https://github.com/loadingalias/rscrypto/commit/d259d6ad74c0e1050cf4c97fbc41a923e2a41b50))
- aead: fix powerpc64le AES/POLYVAL endian handling ([88ac4c3](https://github.com/loadingalias/rscrypto/commit/88ac4c308d8ffccb3f4b026bc6570773ac7e2e26))
- docs ([ab7280c](https://github.com/loadingalias/rscrypto/commit/ab7280ca512a9a7939f10319807696a8397fc737))
- docs ([16612ae](https://github.com/loadingalias/rscrypto/commit/16612ae606e012d9881312405625327d0162745c))
- aead: hardware AES + CLMUL on s390x, powerpc64, riscv64; Apple M5 detection ([05e3a3e](https://github.com/loadingalias/rscrypto/commit/05e3a3e6b43f55faba51a9ee6b6556e664a4af4e))
- clippy/rustfmt ([a41a620](https://github.com/loadingalias/rscrypto/commit/a41a620aba8c915eaacbff68c4313653e007d4a0))
- aead: AEGIS-256 with AES-NI/AES-CE; ChaCha20 x86 SIMD transpose + vprold/vpshufb ([15e89d5](https://github.com/loadingalias/rscrypto/commit/15e89d5b66857056d9111706857c47111df02f71))
- aead: add Ascon-128 AEAD, VAES-512/VPCLMULQDQ wide paths, and POWER/s390x SIMD backends ([a43fa65](https://github.com/loadingalias/rscrypto/commit/a43fa6545b386e8b342aba426edd6d5fbbb092e1))
- clippy ([e6476d9](https://github.com/loadingalias/rscrypto/commit/e6476d9b8fce8cf6372ee3fe3f587249fc5522ec))
- clippy/unsafe comments ([f5ee551](https://github.com/loadingalias/rscrypto/commit/f5ee5513992606e8ccd358e3a15e5d6a836ef068))
- clippy ([5b55ed6](https://github.com/loadingalias/rscrypto/commit/5b55ed6f59229ac4f6863274c003dd1ed8897b7b))
- aead: add aes-256-gcm with ghash; ed25519: optimize field arithmetic ([8d4b65c](https://github.com/loadingalias/rscrypto/commit/8d4b65c8a169bf973a3fe37d3df80624c21a43f9))
- core: add chacha20-poly1305, cshake256, kmac256, ascon-cxof128; harden ed25519 ([481cf8a](https://github.com/loadingalias/rscrypto/commit/481cf8ae9b4d72f4aebfd4e83966a9b9b6481fec))
- core: remove dead dispatch and bench scaffolding from production paths ([e14540f](https://github.com/loadingalias/rscrypto/commit/e14540fb48e8cdf2e1dde94fc3d152ba6545a3bc))
- hashes: xxh3: fix register pressure — remove empty early-return, align mix32_b with xxhash-rust ([75e7337](https://github.com/loadingalias/rscrypto/commit/75e733764884463a5a4edc618b7ac919644780a8))
- hashes: fused absorb-permute for Keccak — eliminate aarch64 write-then-reload round-trip ([5953ebf](https://github.com/loadingalias/rscrypto/commit/5953ebf87e14dea0925941623672dc4adb813101))
- docs + cleanup: update XXH3 bench data (37L→23L), gate arch-specific dead code ([e93491c](https://github.com/loadingalias/rscrypto/commit/e93491c7a8a283902971442048212d649586be0c))
- hashes: XXH3 flat dispatch + compile-time SIMD — eliminate branch/indirect-call overhead ([9fe0cba](https://github.com/loadingalias/rscrypto/commit/9fe0cba303cdceabf3cf31bfd4df93ead753b385))
- cleanup: remove dead code, redundant extern crate alloc, unused prefetch variants ([3ba7578](https://github.com/loadingalias/rscrypto/commit/3ba7578a2c4c5f2d4f1268d31878949b0b2f6218))
- hashes: XXH3 0B fast-return + SHA-3 sponge absorb bypass ([4cc2228](https://github.com/loadingalias/rscrypto/commit/4cc2228675395f5bd45608972ac3ba7f54b7d046))
- clippy ([3f88c3e](https://github.com/loadingalias/rscrypto/commit/3f88c3eb08a186dc63ef8125d149d1bcad064034))
- hashes: XXH3 NEON accumulate — vuzpq deinterleave, broken dep chain, stripe prefetch ([961330d](https://github.com/loadingalias/rscrypto/commit/961330d03371a51a62e323c9ab409ac1b36e386d))
- auth: HMAC-SHA256 oneshot bypass — direct state + single dispatch resolve ([f2dc11e](https://github.com/loadingalias/rscrypto/commit/f2dc11eb22e1c82a295bd526a49c6943c0cd3ebd))
- clippy ([6bc5e60](https://github.com/loadingalias/rscrypto/commit/6bc5e60f09fb7792b0c19f5dde1d9306ed54a16b))
- checksum: CRC32/CRC32C aarch64 3-tier dispatch — EOR3 kernel for large buffers, hardware CRC for small ([cda6a87](https://github.com/loadingalias/rscrypto/commit/cda6a87e04c6d20c7727c62d94e3852beba46c4f))
- clippy ([e83cfba](https://github.com/loadingalias/rscrypto/commit/e83cfba20a84599fad8484b11aaa4b9fd1a50e8c))
- clippy ([5c04df1](https://github.com/loadingalias/rscrypto/commit/5c04df115e5e5c752c76bae482471a499aa3baac))
- hashes: add RapidHashFast64/128, fix benchmark, optimize codegen (no-avalanche variants) ([eeb8f64](https://github.com/loadingalias/rscrypto/commit/eeb8f64c70c70792f3eb8fd7c2ab7397d1b13c50))
- cicd: wiring the auth bench flow ([8024f9c](https://github.com/loadingalias/rscrypto/commit/8024f9c0d592de51b645bf78105f5c272a5d0f82))
- auth: add HMAC-SHA256, HKDF-SHA256, and Ed25519 + benches hashes: tune SHA-512 x86 dispatch for Zen 5 and add std-round kernels docs: add auth, aead, pqe-pqc, and release roadmaps ([eb7fb87](https://github.com/loadingalias/rscrypto/commit/eb7fb87d3eaaeb7fc782a1632eda7a2c574b1f99))
- hashes: xxh3 IBM POWER10 fix ([0b82f15](https://github.com/loadingalias/rscrypto/commit/0b82f150ace871cfddb596a7509d5be10299089d))
- xxh3: SIMD kernels for NEON, AVX2, AVX-512, POWER VSX, and s390x z/Vector ([3781e87](https://github.com/loadingalias/rscrypto/commit/3781e87e2f78b835214c2dee4a10e29f913cd4db))
- checksum: override Checksum::checksum() to bypass hasher construction ([4e9d67b](https://github.com/loadingalias/rscrypto/commit/4e9d67bccf4bf9581bf60d2903061c9cf0ad3c3c))
- keccak: cfg-gate portable permutation by register file width ([1822dd6](https://github.com/loadingalias/rscrypto/commit/1822dd61a67803a3bc8409e53658f10446c4bab3))
- keccak: rewrite portable permutation with array-based state access ([4dce302](https://github.com/loadingalias/rscrypto/commit/4dce30296662912ae2758d305a92685b8f92cdcc))
- sha512: stitched dual-block kernels with zero portable fallback ([da307b4](https://github.com/loadingalias/rscrypto/commit/da307b4092a8888af82547047188410830343e15))
- cicd: fixing avx2 compilation issue ([dc81286](https://github.com/loadingalias/rscrypto/commit/dc81286b8b7230855991e67ea7a9d5b41f044e2f))
- sha512: prefer AVX2 over AVX-512VL on all x86-64 vendors ([627dc00](https://github.com/loadingalias/rscrypto/commit/627dc006c8e38273b7fe1b5d9627075917681133))
- sha512: rewrite AVX2 kernel with stitched dual-block architecture ([5d966a2](https://github.com/loadingalias/rscrypto/commit/5d966a2ee11501a2c35a63ed24a62f25e69f8860))
- workspace: clearing out old scripts that are dead checksums: lower bytewise fast-path threshold from 64 B to 7 B ([da6797f](https://github.com/loadingalias/rscrypto/commit/da6797f6321409126a615e3d8062266bb2c74a78))
- checksums: inline bytewise fast-path for inputs ≤ 64 B ([22cd7b7](https://github.com/loadingalias/rscrypto/commit/22cd7b752744c5584ec22a0257778e35892a8999))
- rscrypto: simplify public API and make dispatch introspection size-explicit ([14bdbce](https://github.com/loadingalias/rscrypto/commit/14bdbce65a86fbe6fbdf9407eae821d15700620e))
- IBM s390x ([ba41ca5](https://github.com/loadingalias/rscrypto/commit/ba41ca52ad985c7f61802ea72658e8bf77f334a8))
- fixing clippy & cicd ([0f47c9c](https://github.com/loadingalias/rscrypto/commit/0f47c9c2ee0d009ce5a337aea8bab393542c66ce))
- checksums: fixing the cicd issues ([0b6521a](https://github.com/loadingalias/rscrypto/commit/0b6521a8c969d0ea8afd6eb5937cc7a8202016aa))
- checksums: fixing the cicd issues on tests ([bb2ee15](https://github.com/loadingalias/rscrypto/commit/bb2ee15cf1cc3b4581f419ddc07ed60997df3fc4))
- workspace: harden release gates and unify blake3 admission ([a9de1e6](https://github.com/loadingalias/rscrypto/commit/a9de1e68829aa50bc95426d65dc0a084ca686358))
- hashes: oneshot SHA-3 fast-path + scalar SHA3 CE kernel; blake3: fix XOF clippy three XOF small-output perf fix ([da1fb95](https://github.com/loadingalias/rscrypto/commit/da1fb95659e93bb0235d148eb0fa5d39509e86cd))
- Clippy ([b32e60e](https://github.com/loadingalias/rscrypto/commit/b32e60e3ed1daf48ea709176e299e64172540e2e))
- hashes: use portable kernel for single-state keccak on aarch64 ([7e117b1](https://github.com/loadingalias/rscrypto/commit/7e117b195186ac07cac56ca14ac469a9f8480793))
- hashes: add raw keccakf1600 isolation bench for sponge overhead diagnosis ([250a73f](https://github.com/loadingalias/rscrypto/commit/250a73f94384af339b16ae41c060ca61cd07e550))
- clippy ([42f4129](https://github.com/loadingalias/rscrypto/commit/42f4129927354fc21c7b1ebacca2e91992f988cb))
- hashes: replace keccak dispatch with direct-call permuters ([950698a](https://github.com/loadingalias/rscrypto/commit/950698ad37bb6550b87f6b8518aef9a3aabce4c3))
- ascon: add true batched SIMD hash/xof paths ([8571219](https://github.com/loadingalias/rscrypto/commit/85712193be666195010a63e1bb1dc978c24d332e))
- hashes: add SHA-256 s390x KIMD and ppc64 vshasigmaw kernels; rewrite bench suite; convert keccakf_portable from full unroll to loop ([47b0bc6](https://github.com/loadingalias/rscrypto/commit/47b0bc637e9affcb2d7a0acc7d3a4f93bcc906f7))
- hashes: general code efficiency ([21f6cba](https://github.com/loadingalias/rscrypto/commit/21f6cbab32039b2eeb665d5c3d819ea15b77f244))
- hashes: remove blake3 special-case short-xof prefix path hashes: add s390x KIMD absorb kernel, 2-state interleaved aarch64 SHA3 CE kernel, and Sha3_256::digest_pair ([3e023b4](https://github.com/loadingalias/rscrypto/commit/3e023b4457b80fa9c1dabd2f6add8f567c329ec4))
- hashes: add x86_64 AVX-512 Keccak-f[1600] kernel; add native blake3 root-output emitters across non-x86 backends ([494ebd0](https://github.com/loadingalias/rscrypto/commit/494ebd012342e541409959dc81a6a01e47161351))
- hashes: narrow blake3 xof emit retuning to large outputs ([b373504](https://github.com/loadingalias/rscrypto/commit/b373504f3566541755d68983eda1f8f5ebad0c20))
- hashes: decouple blake3 xof emission from streaming kernel choice ([c4d8701](https://github.com/loadingalias/rscrypto/commit/c4d87013c4eb623a5dd39296ea8a4ce0e498355d))
- checksums: fixing the POWER10 runner issue in CICD ([a13b9d7](https://github.com/loadingalias/rscrypto/commit/a13b9d7fcec3f0674319d40759f5d6ee0e177e32))
- checksum: remove redundant inner unsafe blocks from CRC-16 and CRC-24 x86_64 SIMD modules ([ffd128a](https://github.com/loadingalias/rscrypto/commit/ffd128a23e2b5cfcebcac15348ade35d08989513))
- checksum: remove redundant inner unsafe blocks from CRC-16 and CRC-24 x86_64 SIMD modules ([662fcfb](https://github.com/loadingalias/rscrypto/commit/662fcfb86ce4f839ea02ff573a28838f4ea012da))
- checksum: add missing unsafe_op_in_unsafe_fn allow to x86_64 and power SIMD modules ([6d39f4b](https://github.com/loadingalias/rscrypto/commit/6d39f4bae426c2418f4c1bad3796f6d796fa6128))
- hashes: SHA-512 family peak hardware dispatch — AMD vendor-aware, s390x KIMD, ppc64 vshasigmad hashes: fixing Blake3 regressions on 4096B perf checksums: clean up ([724e93d](https://github.com/loadingalias/rscrypto/commit/724e93d08bff3d0e3e77c80650b00afa165bcfd1))
- hashes: wire SHA-512 family per-kernel benchmarks and SHA-384/SHA-512-256 streaming ([389b6ae](https://github.com/loadingalias/rscrypto/commit/389b6ae12fb94f37531eae7db4f5d3a4dc23e1fb))
- hashes: perf: cascade AVX-512 sub-degree hash_many tail to SSE4.1 for ≤4 chunks - blake3 hashes: unify SHA-512 family compression; add aarch64 FEAT_SHA512 hardware kernel ([de1fbb0](https://github.com/loadingalias/rscrypto/commit/de1fbb09d051a676a5dc9a4ee54c88df9c0162f2))
- hashes: fix the little-endian gat for the subtree_cv on IBM runners ([154032f](https://github.com/loadingalias/rscrypto/commit/154032f8776c6fe05c8946a8177e3c3cd99e2d37))
- Clippy ([2230eff](https://github.com/loadingalias/rscrypto/commit/2230eff35438eb3be4edd83abcc9e96e6ca2d36c))
- hashes: wired sha224/256 wasm/risc-v; upgraded the blake3 XOF to improve streaming perf ([c10739b](https://github.com/loadingalias/rscrypto/commit/c10739b5f02125cbfedee19b5653678121f4f43a))
- benches: fixing the run-bench.sh script to fire w/ the 'parallel' feature for clean bench comparisons docs: update readme.md ([1f6aa14](https://github.com/loadingalias/rscrypto/commit/1f6aa140b63cb86ffbf9336be9f4f390deb71b7c))
- clippy ([1b2bf2c](https://github.com/loadingalias/rscrypto/commit/1b2bf2cfd7006877c4adec9433899d80b38c4052))
- cicd: replace Python CI scripts with shell+jq ([9b16f36](https://github.com/loadingalias/rscrypto/commit/9b16f36a58a807ab4e4f7d884dfbac4cb397528e))
- cicd: fixing the wasm runner/smoke ([2fd31c6](https://github.com/loadingalias/rscrypto/commit/2fd31c68bf2651cf4639f3567356949cfe4a2b93))
- clippy fixes ([8e7f6d5](https://github.com/loadingalias/rscrypto/commit/8e7f6d5b49f24f40ef0a2e481b21b0bfa0dc775c))
- clippy fixes ([04a853c](https://github.com/loadingalias/rscrypto/commit/04a853cc2a3afbf58c5733942c85c600a662e44e))
- rscrypto: Bump MSRV to 1.94 and add Debug impls ([8e5d481](https://github.com/loadingalias/rscrypto/commit/8e5d48146e3fafbb5d879078ae9551c6b1648bc8))
- flatten: merge workspace into single publishable crate ([f54f983](https://github.com/loadingalias/rscrypto/commit/f54f983065dcda14c636dad22601365b88a27e41))
- hashes: tighten blake3 one-chunk prefix compression workspace: update gitignore ([23e6fc8](https://github.com/loadingalias/rscrypto/commit/23e6fc89d357bbc6402706a533972d50a9ff3fd2))
- Update mod.rs ([5df7450](https://github.com/loadingalias/rscrypto/commit/5df7450ee8e1dc98bab87876b3fbbe88592823db))
- hashes: tighten blake3 short-chunk and large-input admission ([8f22f83](https://github.com/loadingalias/rscrypto/commit/8f22f836e7b950fd4ce402feefc89a9176320bd2))
- checksum,hashes: restore runtime dispatch caps and wire blake3 exact-chunk aarch64 path ([e1884a9](https://github.com/loadingalias/rscrypto/commit/e1884a94e0016a62572fd703d00f9def6fc40ad6))
- hashes: tighten blake3 little-endian word loads ([fe4cb88](https://github.com/loadingalias/rscrypto/commit/fe4cb881b1f33c27d3e157530f765832df1e1a44))
- hashes: simplify exact blake3 chunk update paths ([a93cd32](https://github.com/loadingalias/rscrypto/commit/a93cd32f782a04cb28bc8d40eca1a19da379812a))
- hashes: update exact blake3 x86 chunk prefix in place ([fa4fb5b](https://github.com/loadingalias/rscrypto/commit/fa4fb5bc2a57c8577afceb941ff79b89a97010ff))
- hashes: narrow blake3 update hot path back to frontier-only fast case ([8281724](https://github.com/loadingalias/rscrypto/commit/8281724bce37a15c344b2dc22806e67208afc4db))
- hashes: split blake3 digest update slow path from hot short path ([afaadd0](https://github.com/loadingalias/rscrypto/commit/afaadd0b37a2121b1be1ed6f42cfb5a74303017d))
- hashes: keep short blake3 streaming updates in chunk state ([7c82edc](https://github.com/loadingalias/rscrypto/commit/7c82edc4372e675a441d0053c2ce176f39ffdec1))
- hashes: revert blake3 exact-chunk start-flag loop reshaping ([541f1b8](https://github.com/loadingalias/rscrypto/commit/541f1b8e46473826b592541455e7390723c7152b))
- hashes: remove per-block start-flag branching from exact blake3 updates checksum,hashes: replace stale dead_code suppressions with cfgs ([89eb909](https://github.com/loadingalias/rscrypto/commit/89eb909a0639ea8c14947170119a4f9ff9faae19))
- hashes: remove tuple return from blake3 exact-chunk x86 helper workspace: remove tune workflow and flatten hash facade ([a03f378](https://github.com/loadingalias/rscrypto/commit/a03f378c7048a03e234cbcc66094568cedbf186f))
- hashes: add upstream-shaped portable exact-chunk blake3 update path platform: cleaning up ([f236375](https://github.com/loadingalias/rscrypto/commit/f23637506c144af6934c5d1610ba5f76c7da5f2c))
- hashes: remove redundant blake3 stream kernel and flags state ([0b5513d](https://github.com/loadingalias/rscrypto/commit/0b5513d004f0b7b21f34df27ce4f11141303255a))
- tune: replace TuneKind with explicit blake3 profiles and drop dead generators hashes: split blake3 empty-frontier update from short hot path ([a491791](https://github.com/loadingalias/rscrypto/commit/a491791d2286e94e32c06960aaad5a9bbc4e3347))
- platform,tune: remove platform::tune from runtime architecture hashes: add direct exact-chunk blake3 chunk-state update path ([93f0b51](https://github.com/loadingalias/rscrypto/commit/93f0b51c8253eb2f45c6f8bbfe1ecacbe710f08e))
- platform,tune: remove runtime tune from detection and dispatch hashes: keep blake3 exact-chunk streaming fast path x86-only ([2a86f4b](https://github.com/loadingalias/rscrypto/commit/2a86f4b8ac303898f1b6d586003d7c0bd2f5923d))
- hashes: restore simple aarch64 exact-chunk blake3 update policy workspace: fixing the facade ([7530777](https://github.com/loadingalias/rscrypto/commit/7530777e966e278b6b2cde8ddeda09bb25c5df28))
- hashes: write blake3 exact-chunk state directly into chunk state ([715da43](https://github.com/loadingalias/rscrypto/commit/715da43b0f256b4f3789428f8b8e55b33a68dc05))
- hashes: split blake3 chunk update hot path from general path ([8bc8349](https://github.com/loadingalias/rscrypto/commit/8bc8349d869a68bd1bd79c48f260ec8cf3629adb))
- hashes: narrow blake3 aarch64 exact-chunk streaming update path ([dab5e66](https://github.com/loadingalias/rscrypto/commit/dab5e66b6db7832756c66e79cabb5ab18669f90b))
- hashes: make blake3 aarch64 exact-chunk asm path input-alignment agnostic ([6b3d5d9](https://github.com/loadingalias/rscrypto/commit/6b3d5d9185e73473ad062d75407172ba05dfa7e3))
- hashes: cache blake3 stream kernel id for hasher construction ([d676420](https://github.com/loadingalias/rscrypto/commit/d676420584b75a6364a170e2570d0d3316292772))
- hashes: remove blake3 per-hasher dispatch snapshot ([def58c8](https://github.com/loadingalias/rscrypto/commit/def58c80cba54e0fdfd31f858219dbabe969aec3))
- hashes: simplify blake3 aarch64 exact-chunk update path ([eb9cf4a](https://github.com/loadingalias/rscrypto/commit/eb9cf4a382ea667eb6f9d9f9c960f97fa9200eae))
- hashes: collapse blake3 exact-chunk update dispatch to one match ([0136b52](https://github.com/loadingalias/rscrypto/commit/0136b52a74f5d88e16a18420682d6916fea61047))
- hashes: make blake3 aarch64 exact-chunk asm path output-alignment agnostic ([07d527d](https://github.com/loadingalias/rscrypto/commit/07d527d91855be6b31cf47f6b84bc7fa84d06016))
- hashes: restore correct neon exact-chunk blake3 update path ([2fe90ab](https://github.com/loadingalias/rscrypto/commit/2fe90ab359e9d89885f93868a3fd20665e8568c9))
- hashes: align exact-chunk blake3 update with upstream compressor policy ([d15569a](https://github.com/loadingalias/rscrypto/commit/d15569ab3673e5b8a362f39f773bbe9fd3279bea))
- hashes: remove blake3 exact-chunk streaming detour ([4bf5f58](https://github.com/loadingalias/rscrypto/commit/4bf5f58f0a40e0389282afcd3a277615c1409e14))
- hashes: store blake3 streaming kernels as ids in hasher state ([2f17f3f](https://github.com/loadingalias/rscrypto/commit/2f17f3fc084632d522ec739f32f687425f93034d))
- hashes: narrow blake3 exact-chunk xof update hot path ([53f18c8](https://github.com/loadingalias/rscrypto/commit/53f18c806554c80d002585026f5e11e399dccc3b))
- hashes: use size-class kernel for exact one-chunk blake3 xof updates ([a9131b4](https://github.com/loadingalias/rscrypto/commit/a9131b4b896ea9786e49c5edcd0ed0557514f876))
- hashes: split blake3 xof short-root finalize from slow merge path ([27eb0c2](https://github.com/loadingalias/rscrypto/commit/27eb0c288abd4e7bef32c9f02f1679798435b745))
- hashes: cleaning up the unnecessary hashes covered either by the Rust std lib or too niche/superseded to matter ([53d1b8f](https://github.com/loadingalias/rscrypto/commit/53d1b8f1283e881366ac1f68273edde732c2d4a3))
- clippy fixes; adding some kind of order to the hashes dev plan ([b1c13ad](https://github.com/loadingalias/rscrypto/commit/b1c13ad90a57f18b9b912dbde21bac4cca93bcd0))
- hashes: tighten blake3 xof short-root state and finalize path ([5accb4c](https://github.com/loadingalias/rscrypto/commit/5accb4c12be13495b4b3689122b56b64c6485b21))
- hashes: fix blake3 BP root emitter cross-target gating ([c0a783f](https://github.com/loadingalias/rscrypto/commit/c0a783f8605858d7950605f8c903f650992c0626))
- hashes: add dedicated blake3 root emitter xof path ([fa0f792](https://github.com/loadingalias/rscrypto/commit/fa0f792d6ae5f8650e7e1dbf9bde04e329d94b24))
- hashes: revert blake3 streaming-only frontier and record rejection ([6bf1013](https://github.com/loadingalias/rscrypto/commit/6bf10132ddd13110fb45e412f963a97bea75710a))
- hashes: extend blake3 short-update frontier across chunk rollover ([988d4f5](https://github.com/loadingalias/rscrypto/commit/988d4f558556ee26b82a5c505ae40cbed02a0941))
- hashes: revert blake3 frontier follow-up and record rejection ([8f979bb](https://github.com/loadingalias/rscrypto/commit/8f979bbe0e2717cb3e4b7030066791ad1165756f))
- hashes: extend blake3 frontier mode across short streaming and xof setup ([39de4b5](https://github.com/loadingalias/rscrypto/commit/39de4b529dc92a82b7f7271eb8476464c6861149))
- hashes: add blake3 frontier mode for short streaming and xof ([5f80d1a](https://github.com/loadingalias/rscrypto/commit/5f80d1a79208b1a0f4ebf4ec0ee5b70b34ceede8))
- hashes: record blake3 lazy tree sidecar rejection ([b79d5b5](https://github.com/loadingalias/rscrypto/commit/b79d5b5a4d26f6b4b2bce6ecec1b0a4a8741814f))
- hashes: record blake3 serial-ops snapshot rejection ([13700dd](https://github.com/loadingalias/rscrypto/commit/13700dd153c44f541378657bb6c726760b043da1))
- hashes: split blake3 serial short-update hot path from cold control flow ([b6bae5c](https://github.com/loadingalias/rscrypto/commit/b6bae5cddca6a46c8d06d79e9c62012e40909b0a))
- hashes: split blake3 serial short-update hot path from cold control flow ([4ae6bc3](https://github.com/loadingalias/rscrypto/commit/4ae6bc3fae8a8e67e080b83d25f8bf9a5954263d))
- hashes: rewrite blake3 serial streaming and xof core - reversion commit ([74a6245](https://github.com/loadingalias/rscrypto/commit/74a6245efff14f02172dad3ab3627022af25766d))
- hashes: rewrite blake3 serial streaming and xof core ([675f96a](https://github.com/loadingalias/rscrypto/commit/675f96a94de868e844711732a4b045bfff342fd2))
- hashes: revert blake3 serial xof rewrite and simplify failure map ([13e5e77](https://github.com/loadingalias/rscrypto/commit/13e5e77ef29915731a3210a3288752e7f064651c))
- hashes: rewrite blake3 serial xof setup path and add repeated-update diagnostics ([d5374db](https://github.com/loadingalias/rscrypto/commit/d5374db738898c2fd730a9989415c87b7de62cae))
- hashes: revert blake3 setup-state and bulk-dispatch candidate ([be10196](https://github.com/loadingalias/rscrypto/commit/be101962b2af98c55716938352c211580641a4e8))
- hashes: shrink blake3 setup state and localize bulk dispatch ([04ab0e8](https://github.com/loadingalias/rscrypto/commit/04ab0e8e54d847bd660354dbd1557258c607c51b))
- hashes: revert blake3 hot-state and single-chunk xof candidate ([05f3517](https://github.com/loadingalias/rscrypto/commit/05f3517a2a09daa3989d9e9f471b73572af48c90))
- hashes: shrink blake3 hot state and fast-path single-chunk xof ([1e5778d](https://github.com/loadingalias/rscrypto/commit/1e5778deeb593464e16fd2412a66c2d05a1777cc))
- hashes: revert blake3 compact xof reader experiment ([26afe13](https://github.com/loadingalias/rscrypto/commit/26afe135abd36b64a16d0092feac66ff3d80ddfd))
- hashes: Document Blake3 candidate BA and reject checksum: fix crc64 aarch64 aes feature contracts ([ea0734c](https://github.com/loadingalias/rscrypto/commit/ea0734c80f5a9120ea38e1a9b69972d4124bf0fb))
- hashes: revert blake3 short-read and in-chunk streaming candidate && bumping latest toolchain ([d9d7051](https://github.com/loadingalias/rscrypto/commit/d9d7051505053e3dfa0837054e4203392a5cf390))
- hashes: fix blake3 xof helper unsafe block documentation ([ab1bf11](https://github.com/loadingalias/rscrypto/commit/ab1bf115c6346b9115fec85a831c1802f79b74cc))
- hashes: tighten blake3 xof short reads and in-chunk streaming updates ([539d0de](https://github.com/loadingalias/rscrypto/commit/539d0dec80ed6fd2ffd7627d652d07be9cc79e07))
- hashes: refactor blake3 parallel batch API and fix bench clippy ([dde901a](https://github.com/loadingalias/rscrypto/commit/dde901a1cdefe17edca6eba9a47a955a1e5a5cda))
- hashes: slim blake3 hasher state and localize parallel scratch buffers ([3a5438b](https://github.com/loadingalias/rscrypto/commit/3a5438bee44d784b11f89f8efbaba2ec013f49fa))
- hashes: record AZ bench results and reject candidate ([df9138b](https://github.com/loadingalias/rscrypto/commit/df9138bc2ee780a1d5bc6d89410838b91133775c))
- Revert "hashes: keep aligned blake3 pending subtrees and fast-path single-block" ([f2a2cb6](https://github.com/loadingalias/rscrypto/commit/f2a2cb6875e411addae75299ee10afe7a186d3e3))
- Revert "hashes: reduce blake3 constructor dispatch overhead with shared hasher" ([3b5f51d](https://github.com/loadingalias/rscrypto/commit/3b5f51d70afd61f88d83e8525f7c84ddd46eb975))
- hashes: split blake3 xof-phase into new finalize and drop costs ([c5d3171](https://github.com/loadingalias/rscrypto/commit/c5d3171c4294c0b3d05355b4d6bcb7c0929b1f97))
- Revert "hashes: trim blake3 update/root-output fixed overhead on streaming+xof" ([0b2345b](https://github.com/loadingalias/rscrypto/commit/0b2345bc0c664877b848edd82a5ac32ff4c4a906))
- hashes: add phase-split blake3 xof benchmarks for update finalize and squeeze ([d5d9198](https://github.com/loadingalias/rscrypto/commit/d5d9198bc279333e2621dc8a729b5eaa14f5d008))
- hashes: revert AX intel avx2 stream policy and record regression ([5982369](https://github.com/loadingalias/rscrypto/commit/59823696277b732eb1b1e3fd550d971e92c55ca0))
- hashes: set blake3 intel spr/icl stream kernel to avx2 ([6ba2b95](https://github.com/loadingalias/rscrypto/commit/6ba2b95ce63a065906b6657c6979f75674c6fe28))
- hashes: revert AW xof first-read fast path and record run results ([2a63390](https://github.com/loadingalias/rscrypto/commit/2a6339038a92877f45789a19530cea0501f5fc91))
- hashes: optimize blake3 xof short first-read squeeze path ([4097712](https://github.com/loadingalias/rscrypto/commit/409771272c712c8799fd4dea64475f0474e58052))
- hashes: revert AV pending_chunk_cv removal and record bench regression ([6b9285c](https://github.com/loadingalias/rscrypto/commit/6b9285ccf8e74a59ac371b3887d5f7cdbf4ce543))
- hashes: revert blake3 candidate AU runtime-path simplification hashes: simplify blake3 streaming/xof hot path by removing deferred chunk CV ([abc1a2f](https://github.com/loadingalias/rscrypto/commit/abc1a2f07c21b249e469e1002c8f4c9e75f98891))
- hashes: simplify blake3 streaming/xof runtime paths ([6d508e6](https://github.com/loadingalias/rscrypto/commit/6d508e6a030582bb441bc1e7dbef78f8f660ae57))
- hashes/blake3: revert candidate AT after run 22645562535 regression ([dc522c2](https://github.com/loadingalias/rscrypto/commit/dc522c26c008c99013fedc7d750a50b293155c50))
- hashes/blake3: lock x86 tiny first-update path to stream kernel ([eb61536](https://github.com/loadingalias/rscrypto/commit/eb61536401be7d19433c0da87bcbd99578f790a8))
- hashes/blake3: revert candidate AS after run 22638507919 regression ([0e66733](https://github.com/loadingalias/rscrypto/commit/0e6673377c03af16da4622bdfcdc3df1c6ba9842))
- hashes/blake3: retune x86 stream policy and bypass x86 first-update defer ([6de31a0](https://github.com/loadingalias/rscrypto/commit/6de31a0999444bd66cc481fc5cc38f91fb1c968c))
- hashes/blake3: revert candidate AR after run 22637943751 regression ([25c6364](https://github.com/loadingalias/rscrypto/commit/25c63644a33cccec1c6a9178ea2b5af3f63dee08))
- hashes/blake3: simplify chunk-state streaming update control path ([1107ec6](https://github.com/loadingalias/rscrypto/commit/1107ec6f009a9d329b2aa5c51ccbf19d6f9ea77e))
- hashes/blake3: revert candidate AQ after run 22636554814 regression ([abf3ab8](https://github.com/loadingalias/rscrypto/commit/abf3ab874d6bad537b559186490962f0c583f277))
- hashes/blake3: add kernel-aware xof first-block root-hash fast path ([bf58df9](https://github.com/loadingalias/rscrypto/commit/bf58df99569450e3088650298e4947dc2140a4bf))
- hashes/blake3: revert candidate AP after run 22634499864 regression ([cd597ac](https://github.com/loadingalias/rscrypto/commit/cd597acfb8d09b22dea240da401312807f092014))
- - hashes: refactor blake3 xof_many helper arity ([a780ad1](https://github.com/loadingalias/rscrypto/commit/a780ad164e636ef81769ba685b80943f79e98c12))
- hashes/blake3: split xof single-block vs many-block dispatch and simplify output reader ([686773a](https://github.com/loadingalias/rscrypto/commit/686773a464368f26b3482e8291176f8ca0d92640))
- hashes: revert blake3 xof reader hot-path simplification ([d4ffe54](https://github.com/loadingalias/rscrypto/commit/d4ffe5427240cb1b5506a0508cbf7162912dc2ed))
- hashes: simplify blake3 xof reader hot path and cut short-output overhead ([ec296b8](https://github.com/loadingalias/rscrypto/commit/ec296b859d8ffe85c4de27ffe3c200e174c9b3df))
- hashes/blake3: revert AN and record run 22601614792 xof/streaming regression ([661da62](https://github.com/loadingalias/rscrypto/commit/661da62a39c33cdae751439455251e4f8b84a604))
- hashes/blake3: simplify streaming/xof hot paths and retune x86 stream kernels to avx512 ([56485ac](https://github.com/loadingalias/rscrypto/commit/56485accb7097ca3430e55d7c08ff996f5b404d0))
- hashes/blake3: revert AM linux sse41 asm wiring and record run 22600518259 regression ([097edf6](https://github.com/loadingalias/rscrypto/commit/097edf6ccecb60cfc09ae26d69f9a8a468a81d73))
- hashes/blake3: fix linux sse41 asm data section directive ([8885536](https://github.com/loadingalias/rscrypto/commit/8885536ac33f7124d622cd4a893286f3ed35cafa))
- hashes: wire Linux SSE4.1 asm into blake3 streaming/xof hot paths ([5a7df00](https://github.com/loadingalias/rscrypto/commit/5a7df005e72089ba6c3922a47682ea9e43157776))
- hashes/blake3: revert AL and record run 22593524598 regression ([71d45a1](https://github.com/loadingalias/rscrypto/commit/71d45a1b587fd070ae2b1d82712bafbf5043716f))
- hashes: simplify blake3 xof kernel path and add x86 one-chunk streaming fast path ([e7fbaf9](https://github.com/loadingalias/rscrypto/commit/e7fbaf99651169e2542b8a26402cccfb59606297))
- hashes/blake3: record AK xof/streaming regression from run 22588630907 ([9d32656](https://github.com/loadingalias/rscrypto/commit/9d326560d72f84ca610a2de2a6a70a99a6252f9d))
- Revert "hashes/blake3: debranch streaming chunk compression and simplify xof" ([b0c088b](https://github.com/loadingalias/rscrypto/commit/b0c088b48e262a4abc234a5cf62f2bdaadca1372))
- hashes/blake3: document AJ bench regression and reject via 70e7519 ([ea7f2c5](https://github.com/loadingalias/rscrypto/commit/ea7f2c5c9b6413e561bdbe01174e3c2602d74918))
- Revert "hashes/blake3: use kernel-aware xof root-hash path and retune x86" ([70e7519](https://github.com/loadingalias/rscrypto/commit/70e7519ec2670b53f46f32b0f347bb572bca1477))
- hashes/blake3: record AI regression from run 22559324946 and revert  decision ([ddf5c5a](https://github.com/loadingalias/rscrypto/commit/ddf5c5ac25e440f5a26a46bb2e3b892d42f2a353))
- Revert "hashes: inline blake3 OutputState compress dispatch" ([a787d48](https://github.com/loadingalias/rscrypto/commit/a787d4811af15ed848dcf633a8b0cbde8f069e1e))
- hashes/blake3: revert AH lazy tiny xof tail-hint path; record 22556963869 regression ([800445d](https://github.com/loadingalias/rscrypto/commit/800445deda3f0634bfce48bbd1389a2b4fe9034a))
- hashes/blake3: lazily optimize tiny xof init+read with single-chunk tail hint ([03d9943](https://github.com/loadingalias/rscrypto/commit/03d9943e50fad6b5c7bbe5f9e09ace4d0fa6e7d8))
- hashes/blake3: revert AG xof cached-root precompute; record 22555765324 regression ([26ada2f](https://github.com/loadingalias/rscrypto/commit/26ada2f26aa63bf1f344871036c74592dbc35ce1))
- hashes/blake3: cache single-chunk xof root hash for tiny init+read path ([b1f25ea](https://github.com/loadingalias/rscrypto/commit/b1f25ea441b5ec0c20a7573342dcceb867788f9a))
- hashes/blake3: revert AF xof/streaming kernel changes; record 22554313539 full loss ([427a819](https://github.com/loadingalias/rscrypto/commit/427a81959af51f6c2c6fa5122215f56a2db13a46))
- hashes: retune blake3 xof finalize path and full-chunk streaming kernel selection ([c3eaa72](https://github.com/loadingalias/rscrypto/commit/c3eaa7220f9e1500543b2124d88f77c72d43a93e))
- hashes/blake3: revert AE xof/streaming kernel changes; record streaming regression ([e3c6960](https://github.com/loadingalias/rscrypto/commit/e3c6960418b1d673bfbced154c9dbbec72aca8e4))
- hashes/blake3: lift xof portable finalize path and switch x86 streaming to avx2 ([dd8be8e](https://github.com/loadingalias/rscrypto/commit/dd8be8e3be00132dc052685ea4bfad6f1564b280))
- ci/bench: make BENCH_FILTER authoritative over BENCH_ONLY matching ([6378a21](https://github.com/loadingalias/rscrypto/commit/6378a215e4fc634d9390c01a93533185e95ca19c))
- hashes/blake3: narrow single-chunk xof kernel override to Intel portable states ([853fd53](https://github.com/loadingalias/rscrypto/commit/853fd5334a61dac914e275ca33b0b823d5d304e0))
- ashes/blake3: revert AC single-chunk xof kernel override; record AB/AC rejects ([450d782](https://github.com/loadingalias/rscrypto/commit/450d782d1d5e17ae45011d0c86845064bc1c80fc))
- hashes: select size-class kernel for single-chunk finalize_xof ([aea12be](https://github.com/loadingalias/rscrypto/commit/aea12be21542ba146af042819f0dd4546c4ec62a))
- ashes: revert Blake3Xof lazy scratch-buffer initialization ([19bc72b](https://github.com/loadingalias/rscrypto/commit/19bc72b286c43140a48497b9f06d1fa92753e156))
- hashes: lazily initialize Blake3Xof scratch buffers ([bd61c53](https://github.com/loadingalias/rscrypto/commit/bd61c5392b35289685ab277815aefa1414f93f83))
- hashes/blake3: optimize short streaming/xof paths and retune 64KiB+  parallel admission ([b493be1](https://github.com/loadingalias/rscrypto/commit/b493be1a0bb220ee01703aa9ba8f9c6d0e84acdc))
- checksum: add aarch64 crc16 pmull+eor3 kernels and wire graviton dispatch ([b34a9b9](https://github.com/loadingalias/rscrypto/commit/b34a9b9e0ede0b02661aef95847fe8a8006b9b5a))
- hashes: promote streaming kernel after first full chunk ([b160f5a](https://github.com/loadingalias/rscrypto/commit/b160f5acda892e6f4f4aa879bff7ed2736f2f2d5))
- checksum/hashes: remove hot-path overhead in arm crc16 2way and blake3 dispatch accessors ([171d60d](https://github.com/loadingalias/rscrypto/commit/171d60de8df636e0040257965ec5784fb07aed19))
- cicd: fixing the filtering issue for bench.yaml and updating blake3 work ([742a11d](https://github.com/loadingalias/rscrypto/commit/742a11d5e95737fe62f96c525554a540c913e8cd))
- checksum: revert crc16 aarch64 2-way pointer-walk loop rewrite ([aa2979c](https://github.com/loadingalias/rscrypto/commit/aa2979ce3c7c77781e2e076e6d86be59fe5dd8a6))
- checksum: optimize aarch64 crc16 PMULL hot loops with pointer-walk folding ([8137efe](https://github.com/loadingalias/rscrypto/commit/8137efee91a3d211286b34bfc80d701051ea65b8))
- Update dispatch_tables.rs ([b6e2065](https://github.com/loadingalias/rscrypto/commit/b6e2065591683ad52f563295fcf3d36566c1783d))
- checksum: cache crc16 dispatch/table in hasher state ([b43a088](https://github.com/loadingalias/rscrypto/commit/b43a088f46b004ff40bf707550164835c880d22d))
- checksum: retune zen4 crc64 l dispatch with xz hybrid and nvme 2way ([96d25b3](https://github.com/loadingalias/rscrypto/commit/96d25b3432c1800c106096f0c3d8d00f370ffed7))
- checksum,hashes: cache crc64 dispatch in hasher state, retune x86 crc tables, and add s390x short-batch blake3 fallback ([2420ac8](https://github.com/loadingalias/rscrypto/commit/2420ac80552cfb829d4b0df83daef8a9aa644175))
- hashes: revert generic one-chunk compress-inline candidate ([8e78bd2](https://github.com/loadingalias/rscrypto/commit/8e78bd252d5781ea65498d809f777680574e2b00))
- hashes: inline generic one-chunk final compress dispatch ([99874ad](https://github.com/loadingalias/rscrypto/commit/99874ade34df2210a5745b5100968271a71c961e))
- hashes: remove BLAKE3 exact-tree copy-back in oneshot reduction ([3e9bac9](https://github.com/loadingalias/rscrypto/commit/3e9bac930dfadc7edf12bce2ff8898f2fe5dc2f5))
- hashes: trim BLAKE3 oneshot exact-tree overhead for small inputs ([5c52746](https://github.com/loadingalias/rscrypto/commit/5c5274615caf479f7fc71c607c5cb5da8030f645))
- checksum: add graviton3 crc16/ccitt l hybrid dispatch ([a94a8e7](https://github.com/loadingalias/rscrypto/commit/a94a8e7d192ba259fc8fd6791f697b08541ca9d2))
- Revert " checksum: unroll+prefetch aarch64 crc16 pmull 1-way fold loop" ([91a8f07](https://github.com/loadingalias/rscrypto/commit/91a8f07c5f591494dd99884fc11584142e9d1c88))
- checksum: retune graviton crc16 dispatch (g3 ibm hybrid, g4 m/l 2way) ([04009ce](https://github.com/loadingalias/rscrypto/commit/04009ce07f922f8bc3e43df328f2bb50ae6c6c7c))
- checksum: split graviton3/4 crc16 l-xl dispatch tables ([ee3aeb0](https://github.com/loadingalias/rscrypto/commit/ee3aeb0b6effd97ff8880c9ed2a74a9e5aefc5a6))
- checksum: retune graviton crc16 l/xl dispatch to pmull-2way ([a5a2c85](https://github.com/loadingalias/rscrypto/commit/a5a2c85ef5c03ae7f0ba3ee9d7db2064ce4de777))
- checksum: retune intel-icl crc32/ieee s dispatch to vpclmul-2way ([19a04e1](https://github.com/loadingalias/rscrypto/commit/19a04e1b778eda4005beed986c0b6cd68a2fada2))
- checksum: retune intel crc32 s dispatch (spr 2way, icl 8way) ([4eab23c](https://github.com/loadingalias/rscrypto/commit/4eab23c03474eb6cad5f55bad48862a271e868db))
- checksum: cache crc32/crc32c auto dispatch table in hasher state ([ee5eee4](https://github.com/loadingalias/rscrypto/commit/ee5eee46e46757de8ccddf2611b368b496c0fbb1))
- checksum: restore icl crc32 s vpclmul and rework aarch64 crc32 xs/s hot path ([db96691](https://github.com/loadingalias/rscrypto/commit/db96691e940af2ae1f19a0e81dc0859df6f5c84f))
- checksum: retune intel-icl crc32 s to vpclmul-2way and document focused gap plan ([3421e2a](https://github.com/loadingalias/rscrypto/commit/3421e2a0b335add88358965299cf0fc1740ec65e))
- checksum: revert intel-icl crc32 s dispatch to vpclmul ([362bbce](https://github.com/loadingalias/rscrypto/commit/362bbce3252326827a21965322c21d6e17f6f41e))
- checksum: revert Graviton4 hwcrc xs/s retune and record failed bench ([2e6f938](https://github.com/loadingalias/rscrypto/commit/2e6f938ea51796aaefd348b4ee8fdfa8f8cb5f6d))
- checksum: split Graviton4 table and retune crc32/ieee xs/s ([b0b9de7](https://github.com/loadingalias/rscrypto/commit/b0b9de75534b476cba9241f22ea142a0825655ae))
- Revert "checksum: cache CRC32/CRC32C dispatch in hasher state and mapping" ([1c171f1](https://github.com/loadingalias/rscrypto/commit/1c171f19246836f550f8ccb2794da66385214e23))
- hashes: revert rayon-aware blake3 parallel admission and 1MiB parallel gate ([8e070d1](https://github.com/loadingalias/rscrypto/commit/8e070d16fcec8074a7df4e46ff9ed840b8c33f43))
- hashes: make blake3 parallel admission Rayon-aware and add 1MiB parallel gate ([fede7b2](https://github.com/loadingalias/rscrypto/commit/fede7b2c772b9b7ef1fe428d768c8237f42bd46d))
- hashes: gate avx2 one-chunk hash_many fast path by x86 tune kind ([21f9c68](https://github.com/loadingalias/rscrypto/commit/21f9c682e63b9642a3a0c689d54f51187e395f1d))
- hashes: revert global avx2 one-chunk blake3 fast path and record candidate V regressions ([42d10e7](https://github.com/loadingalias/rscrypto/commit/42d10e78cda64f1d92792b4d5615dda946192229))
- hashes: add avx2 one-chunk exact-block hash_many fast path for blake3 oneshot ([b7c7984](https://github.com/loadingalias/rscrypto/commit/b7c7984361d190231e7a411f890fac3ee3120f37))
- hashes: route intel-spr short oneshot to avx512 and add avx512 one-chunk asm hash_many fast path ([01f80aa](https://github.com/loadingalias/rscrypto/commit/01f80aad26a3eb7db1f8dafca73ee758fb087bcf))
- hashes: revert candidate T avx512 short-path and record failed intel bench ([f2c61ad](https://github.com/loadingalias/rscrypto/commit/f2c61adb7635d920441e2d90b0ff41073fda8801))
- hashes: narrow blake3 avx512 short-block fast path to <=4 blocks ([77997e4](https://github.com/loadingalias/rscrypto/commit/77997e45507ec7b7eae49bc41f27fbd42794bbf4))
- hashes: revert candidate S short-block avx512/neon chunk-compress fast paths ([43dbe5c](https://github.com/loadingalias/rscrypto/commit/43dbe5c5544800c39bb7cc32e0164c3b0766b5d1))
- hashes: add blake3 short-block fast paths for avx512 and neon chunk compression ([9f4459a](https://github.com/loadingalias/rscrypto/commit/9f4459a89ce5f9fa5b3ab6c0afe05db6994b4ef4))
- hashes: revert power10 dispatch and one-chunk helper devirtualization ([086f99b](https://github.com/loadingalias/rscrypto/commit/086f99b49bc4d219ef0ce741bc43006cc1a7ed24))
- hashes: tune power10 short dispatch and devirtualize blake3 one-chunk x86/aarch64 paths ([7c0b8f5](https://github.com/loadingalias/rscrypto/commit/7c0b8f5de3f385741ce1004c48fe2922e6d4e492))
- hashes: optimize generic BLAKE3 one-chunk path with inline dispatch and zero-copy aligned tails ([ae6b775](https://github.com/loadingalias/rscrypto/commit/ae6b775c2f606331e975e5466a3ea8de44b78b19))
- hashes: add shared one-chunk BLAKE3 oneshot fast path and log candidate P results ([e28c9b9](https://github.com/loadingalias/rscrypto/commit/e28c9b9223b056cf87a94a24655248397f8a6084))
- hashes: enable BLAKE3 vector bulk dispatch for IBM Z and POWER profiles ([3a57016](https://github.com/loadingalias/rscrypto/commit/3a5701628d43a769fad3913a443788fa3b789107))
- hashes: revert noinline oneshot split and lock blake3 perf loop plan ([c445215](https://github.com/loadingalias/rscrypto/commit/c44521528b069a094cddb9267d5bf6c55cefb7ae))
- hashes: split blake3 oneshot hot paths into noinline helpers ([0b12fe3](https://github.com/loadingalias/rscrypto/commit/0b12fe36b99dcba32e6a0d5b959ba84c644e1cff))
- hashes: keep short blake3 inputs portable on aarch64 server profile ([99ce277](https://github.com/loadingalias/rscrypto/commit/99ce2779b54b4b1f09e745ac93f61943bb94ec38))
- hashes: reverting the attempt to optimize blake3 clone by copying only initialized cv stack ([649e58b](https://github.com/loadingalias/rscrypto/commit/649e58b24665372f003ede13aa46259767eda843))
- hashes: optimize blake3 clone by copying only initialized cv stack ([c0d736b](https://github.com/loadingalias/rscrypto/commit/c0d736ba289049a8a2d2cd518eac5970c0676256))
- Revert "hashes: remove aarch64 one-chunk helper kernel-id dispatch" ([c4ae105](https://github.com/loadingalias/rscrypto/commit/c4ae10564ab0a706cd29e39fe01ffd99f5c30a75))
- hashes: split blake3 oneshot fallback into cold helper ([1434fa2](https://github.com/loadingalias/rscrypto/commit/1434fa22c1f32b50a9e2ae824eaac4340fe5b935))
- hashes: stop inlining blake3 root_output_oneshot into digest_oneshot_words ([d21f59a](https://github.com/loadingalias/rscrypto/commit/d21f59a021bf934aff7cf262588d5250bea8067d))
- hashes: revert blake3 work on boundaries/policy and lock kernel-first perf plan ([6bb3279](https://github.com/loadingalias/rscrypto/commit/6bb32793b0129b0d3473d5a472f3fd8fad157ea1))
- hashes: narrow blake3 plain 1024 first-update override to intel avx512 ([84f8f07](https://github.com/loadingalias/rscrypto/commit/84f8f07d302ac6c590e3448a35c6c0ee96f43bce))
- hashes: add plain-mode x86 short split policy for blake3 oneshot/update ([da160b9](https://github.com/loadingalias/rscrypto/commit/da160b950b2ed1c965362812f86efb0cd485b1ad))
- Revert "hashes: add x86 exact-chunk fast path in blake3 ChunkState update" ([c6b36ed](https://github.com/loadingalias/rscrypto/commit/c6b36ed1d1ccdd0931913d6bc89e7974691d139e))
- Revert "hashes: add block-aligned single-chunk fast path for blake3 short" ([5b7497e](https://github.com/loadingalias/rscrypto/commit/5b7497ee031d1ae7f978d3b492701ce07bc0bb0b))
- hashes: revert blake3 x86 one-chunk inline path ([3f6ae9a](https://github.com/loadingalias/rscrypto/commit/3f6ae9a046e77fa6345bb5c3175b5f9375f1ade4))
- hashes: inline x86 one-chunk pre-final block compression in blake3 oneshot path ([e364abb](https://github.com/loadingalias/rscrypto/commit/e364abb64998208528559332ca158533f78ad086))
- hashes: unify blake3 public oneshot path for plain keyed and derive ([0f1b1b8](https://github.com/loadingalias/rscrypto/commit/0f1b1b82aa9d97a877836125d7954dad6b087711))
- hashes: add blake3 oneshot apples-to-apples attribution benches ([c54a494](https://github.com/loadingalias/rscrypto/commit/c54a49492dbf6638bbd4ddaacd05092435c2d95c))
- hashes: add first-update single-chunk fast path and drop s390x-specific finalize fallback ([b4f785f](https://github.com/loadingalias/rscrypto/commit/b4f785f7dbdffa1ef33e9be2eb1f0e14f71af54e))
- hashes: s390x fallback for 1024 single-chunk finalize ([6469d31](https://github.com/loadingalias/rscrypto/commit/6469d315d3b5146277a405541b1a56b955c65b91))
- hashes: candidate F fast-path blake3 single-chunk finalize root tail ([934ccde](https://github.com/loadingalias/rscrypto/commit/934ccdedefa14aa67339ea9bb374f82b3c438b2a))
- hashes: candidate E revert blake3 short-update escalation keep tiny finalize cleanup ([62668f2](https://github.com/loadingalias/rscrypto/commit/62668f20d11eeae29057287bc7e44e1009a25779))
- - hashes: candidate D optimize blake3 pristine short streaming path ([b55ad17](https://github.com/loadingalias/rscrypto/commit/b55ad178723dcb51f2d78edf02b8528fbd2a7374))
- hashes: revert blake3 x86 candidate-b and add candidate-c final-block fast path ([8b9f3d1](https://github.com/loadingalias/rscrypto/commit/8b9f3d1e4895cc3556529a2c8cd20ec429f8a5db))
- hashes: optimize blake3 x86 one-chunk short-path compression loop ([a39e33a](https://github.com/loadingalias/rscrypto/commit/a39e33a04ec4a6133114829540e2e220d095778f))
- hashes: simplify blake3 short-input one-chunk hot paths ([857485f](https://github.com/loadingalias/rscrypto/commit/857485f2f846865df509124c947500ba38d8d65d))
- hashes: route server aarch64 blake3 short oneshot classes to neon ([66457ee](https://github.com/loadingalias/rscrypto/commit/66457eee2d201c8c980b11f4bb43eb35b68ff4c5))
- hashes: add blake3 short-input attribution bench and baseline update ([664401e](https://github.com/loadingalias/rscrypto/commit/664401e7b6dc80c71e48b182b4aca52a5116d786))
- hashes: enable full blake3 gate diagnostics and record baseline tables ([4d35d53](https://github.com/loadingalias/rscrypto/commit/4d35d5307422e4572d471e418a64fd6ba05e53d5))
- bench: cleaning the workflows post-tuning cleanout for the blake3 efficacy benches ([fffdc9a](https://github.com/loadingalias/rscrypto/commit/fffdc9ac876d3fb03a2fe6fe3cda5d132c69337b))
- feature: adding to the tuning system in CICD; adjusting the Namespace runner to RunsOn runners. Adding guards and ensuring it's all useful. ([a8ee12e](https://github.com/loadingalias/rscrypto/commit/a8ee12ed69db13080b96887da8609670b786e92a))
- rscrypto: cache busting ([fe99daf](https://github.com/loadingalias/rscrypto/commit/fe99daf335e04d63d98601bb9e9ab5a715f2de55))
- rscrypto: tuning updates to apply and applied ([c05b449](https://github.com/loadingalias/rscrypto/commit/c05b449abcd5fe93393b845d8086abc2dbb82cc2))
- hashes,tune: optimize blake3 SIMD/tiny paths and calibrate tune contracts/reporting hashes: bind block/final compressors in Kernel to remove hot-path id matching; optimize tiny-path dispatch; reduce RVV vsetivli churn; streamline IBM/RISC-V parent SIMD lane loading. tune: add strict vs informational BLAKE3 contract modes, suppress false MISS for informational surfaces, and fix peak-throughput fallback when threshold-only data would report 0.00 GiB/s. workspace: update cargo-rail to v0.10.8 and refresh cargo-rail-action wiring in dev workflow. ([f3a2f1b](https://github.com/loadingalias/rscrypto/commit/f3a2f1b40eec9aa96fa131f897e97bc9fb3da680))
- rscrypto: fixing the massive stack allocation issue ([e47feee](https://github.com/loadingalias/rscrypto/commit/e47feee7b178a7534b119f0e89fca297d80e2630))
- rscrypto: expanding the tuning engine for Blake3 (w/ extensibility in mind for future hashes) to prioritize real workloads - latency - instead of ONLY throughput. ([100d5b5](https://github.com/loadingalias/rscrypto/commit/100d5b5924ebb545e10155158f7a7c13105cf704))
- rscrypto: profiling w/ cargo-asm, cargo-llvm-lines, and samply to try to get to the bottom of the perf story for Blake3. We're getting closer. ([3c8e19d](https://github.com/loadingalias/rscrypto/commit/3c8e19d757abbedcbf2f6503b805634012d68c63))
- rscrypto: hardening the test vectors to get a better output during failures. Cleaning up. Added a few lints to improve the perf/correctness/etc. improved the check-all command locally. ([2f8a4f8](https://github.com/loadingalias/rscrypto/commit/2f8a4f8aceeb2f78888ca0c1c93d4e8a97b8580f))
- rscrypto: tuning applied ([0144149](https://github.com/loadingalias/rscrypto/commit/0144149a180b064a6ddb3549c226ef4f13ff01dd))
- rscrypto: updating the tuning and apply pipelines to be more accurate and efficient. ([6a9677b](https://github.com/loadingalias/rscrypto/commit/6a9677b77ffed2b160e1ac4d9368b695f3eb5c26))
- rscrypto: updating the bench/tune workflows and manual UI triggers for clarity and consistency. ([bccd468](https://github.com/loadingalias/rscrypto/commit/bccd46842657f6b68cf1f6768e58f60055a8d461))
- rscrypto: committing the latest tuning for the full wheel on Blake3 so far ([7ec3f49](https://github.com/loadingalias/rscrypto/commit/7ec3f49792efc3ae4bcaf59454f5524a5346154f))
- rscrypto: applying the Mac M1 tuning results ([6f820be](https://github.com/loadingalias/rscrypto/commit/6f820bef276b2f79b31536eb4c74fe9e66dfaedb))
- rscrypto: ci/tuning/platform: unify targeted lanes and harden tuning/apply pipeline ([d3f8af3](https://github.com/loadingalias/rscrypto/commit/d3f8af39956dc3dea9aa5d06ad8c2739829c69f5))
- rscrypto: fixing the license issue w/ cargo-audit ([a22dfbd](https://github.com/loadingalias/rscrypto/commit/a22dfbd029a3a2925255626bc5c152fa171face2))
- rscrypto: fixing the tuning timeouts - fucking annoying, expensive, and sloppy on my part. ([43969b4](https://github.com/loadingalias/rscrypto/commit/43969b4c713b1ee16845eddfaad7beb75007ffce))
- rscrypto: fixing tuning engine again ([2fc38aa](https://github.com/loadingalias/rscrypto/commit/2fc38aad281c3997fd0ebb3b1bbd2ad798bd5035))
- rscrypto: fixing the tuning split ([0759071](https://github.com/loadingalias/rscrypto/commit/07590711717ef998d7cfa82fa54d00038b26792f))
- rscrypto: tuning improvements - cleaner split, better artifacts. ([0cc25be](https://github.com/loadingalias/rscrypto/commit/0cc25be1360cc9ab66e9ca41c607ef086e6ca553))
- rscrypto: split measurement/policy in the tuning engine to improve accuracy and efficiency. ([3227b16](https://github.com/loadingalias/rscrypto/commit/3227b1688ed7790eac05f6b35336b341fba6ede2))
- rscrypto: updating the tuning engine to hard error when the 'quick' flag is set to true. ([c4e2998](https://github.com/loadingalias/rscrypto/commit/c4e2998207daac1dbc97178c0c0b6a29d535c11a))
- rscrypto: fixing the tuning engine efficacy and updating the cross-arch stream misses issue w/ blake3 during tuning runs.So ([37a3aa8](https://github.com/loadingalias/rscrypto/commit/37a3aa83c96c985117d4a774c7ee838201490a83))
- rscrypto: added Sapphire Rapids to the tuning workflow in CI ([a18fc09](https://github.com/loadingalias/rscrypto/commit/a18fc09e75e404abf06176eb5d9945586e33598b))
- rscrypto: addressing a few more gaps in perf for the Blake3 impl across arches ([28c9969](https://github.com/loadingalias/rscrypto/commit/28c996909a0726e3d7a777aa19ae856b261b1308))
- rscrypto: fixing the broken sse4.1 SIMD paths for x86-64 ([e241e4e](https://github.com/loadingalias/rscrypto/commit/e241e4e574bc721d6deedc87b06f5424a9f75e19))
- rscrypto: wired the ASM for AVX512; brought parity to the Graviton/Neoverse Blake3 impls becasue they're targeting server chips; started to fix the 'reuse' in critical paths issue. ([ddf900a](https://github.com/loadingalias/rscrypto/commit/ddf900ac9c84a00964416da94bb542ad1b45c47f))
- rscrypto: updating the benches for parity across the 'rscrypto/official/official-rayon' impls. Added the correct Blake3 tuning options so that we're not forcing one-shot on everything foolishly. Updated the hardcoded streamingTable definitions. ([33ef210](https://github.com/loadingalias/rscrypto/commit/33ef210b1471e4b277c688436d06a42900ba7861))
- rscrypto: addressing the weekly.yaml failures and fuzzing configuration. Added the dedicated x86-64 arches fast path for tiny XOF paths. ([3bb04ec](https://github.com/loadingalias/rscrypto/commit/3bb04eca92e94982ceab838734c9d79fb7c2a127))
- rscrypto: fixing the Clippy lints for the Blake3 keyed/derive ([07f4433](https://github.com/loadingalias/rscrypto/commit/07f4433837819b2b5996e2ec78b28c98cb282e91))
- rscrypto: fixing the tiny/small keyed/derive perf for Blake3 ([8c5050b](https://github.com/loadingalias/rscrypto/commit/8c5050b063c67921bfc70541cbd54877a05c780f))
- rscrypto: fixing the buffered benches w/ 8KB. ([eecfaaf](https://github.com/loadingalias/rscrypto/commit/eecfaafb7ee963c0cf930acd809218ceb37c0fba))
- rscrypto: updating the machines from AWS for CRC parity w/ the crc-fast crate. ([16bed15](https://github.com/loadingalias/rscrypto/commit/16bed157089e48d0a24ea4b325d46accd0a0f2e4))
- rscrypto: adding the s390x/power arches to the zig cross-compilation checks and fixing the issues. ([85da199](https://github.com/loadingalias/rscrypto/commit/85da199707d9b0424bad17e4f3d5862c69159711))
- rscrypto: fixing the IBM clippy warnings and test failures ([61fe940](https://github.com/loadingalias/rscrypto/commit/61fe940ed4f7c3c365e08ae76013a569217ab17c))
- rscrypto: improving the IBM cache hits/speed, hopefully. ([e621ace](https://github.com/loadingalias/rscrypto/commit/e621ace5495f2d9ceb74edf36fa3b5129c688ba7))
- rscrypto: fixing the IBM issues ([f9589d0](https://github.com/loadingalias/rscrypto/commit/f9589d0ca2ee27ac954ab3670b87d758e019a64b))
- rscrypto: fixing parallel streaming and parallel admission policy; updating the comparitive deps. ([2c3b714](https://github.com/loadingalias/rscrypto/commit/2c3b714f65ba278bb2ce049802552d7a2f98a4f7))
- rscrypto: fixing the IBM specific implementations to avoid the compiler issues w/ s390x detection failures and Clippy lints ([f929005](https://github.com/loadingalias/rscrypto/commit/f92900530cc92ed65de84dfeb6754c8fcf24ef39))
- rscrypto: updating the Blake3 tuning ([4b85dbb](https://github.com/loadingalias/rscrypto/commit/4b85dbb33dc9ade156f1f21d38965a428c81e9d8))
- rscrypto: updating the GHA shas and updating the codebase's tooling. ([ff135a8](https://github.com/loadingalias/rscrypto/commit/ff135a81ca881655c9aa89b097c04868d8e86cd4))
- rscrypto: improving the bench.yaml ([605fc60](https://github.com/loadingalias/rscrypto/commit/605fc602503f2ac62f4fdb418362af161f344c03))
- rscrypto: fixing the IBM caches and removing the unnecessary unsafe code in the x86-64 code. ([0cbcf39](https://github.com/loadingalias/rscrypto/commit/0cbcf39b1c5bade96720819f02db26e6c4be2e2d))
- rscrypto: fixing the rust toolchain for the s390x detection via the std lib on nightly. Removing the P9 in favor of P10/s390x being enough coverage. Fixing the types issue for P10 compilation. ([59c83a1](https://github.com/loadingalias/rscrypto/commit/59c83a13024d80d456b39cb36d0998c71f085d20))
- rscrypto: fixing the tuning ([e08541b](https://github.com/loadingalias/rscrypto/commit/e08541b70bab22e0f5e0ec639b6c872488cd5ef6))
- rscrypto: more CICD cleaning becasue AI is taking all of our jobs. ([2e2e6af](https://github.com/loadingalias/rscrypto/commit/2e2e6af215a1e5d60c28e6434d7f8dcc1e671696))
- rscrypto: fixing CICD ([d779592](https://github.com/loadingalias/rscrypto/commit/d77959246b860fd52bb34bdf65c6126c4bf06fa1))
- rscrypto: fixing again ([7f8c21a](https://github.com/loadingalias/rscrypto/commit/7f8c21acbc8d97f032f6c51d2f994dc9c621e66d))
- rscrypto: fixing CICD - again ([b255bf4](https://github.com/loadingalias/rscrypto/commit/b255bf4ca385ad8fac72f674a4322c0128d201ad))
- rscrypto: removing the full Windows runs in the commit.yaml workflow. It's far too slow/expensive. They'll run in weekly.yaml now. ([d586755](https://github.com/loadingalias/rscrypto/commit/d5867551db982839d3f499cf1fde506f680a0fb2))
- rscrypto: CICD fixes : ( ([c281e38](https://github.com/loadingalias/rscrypto/commit/c281e38d1be63e0c9334bdd4b53c030405184075))
- rscrypto: CICD fixes ([81a9d70](https://github.com/loadingalias/rscrypto/commit/81a9d70de844b3f5e52907978b6a00def10d1424))
- rscrypto: fixing the CICD ([9af7eff](https://github.com/loadingalias/rscrypto/commit/9af7effa2e14d04e449a51366a033bb0b45e48a0))
- rscrypto: major infrastructure updates to esure the codebase can handle scaling and improvements over time. ([f8a993e](https://github.com/loadingalias/rscrypto/commit/f8a993eaef79cddd6dfbab966b77e7e5fc40c3ef))
- rscrypto: updating the dispatch, tuning, scripts/justfile. ([f00bbdb](https://github.com/loadingalias/rscrypto/commit/f00bbdb6cd3bc4b8fbbe033bbca9db575b665f70))
- rscrypto: improved the global detection and tuning systems. ([e298823](https://github.com/loadingalias/rscrypto/commit/e298823b7e9b2a582d7364b0f6a0f81872cda142))
- rscrypto: finally cleaning up the CICD pipes and improving the workflows/setups/infra becasue it's starting to drift and get messy. ([dc48a94](https://github.com/loadingalias/rscrypto/commit/dc48a9474cb0c8e8662786ae1eb23fd72ddafd29))
- rscrypto: tuning improvements ([6fc7931](https://github.com/loadingalias/rscrypto/commit/6fc79317d4c058e758b99e0ce6509e9a9d9bf6aa))
- rscrypto: fixing tuning ([80ebf3b](https://github.com/loadingalias/rscrypto/commit/80ebf3b759d11649be97ec315b23baab0c47f254))
- rscrypto: swapping tuning ([1b5b2c7](https://github.com/loadingalias/rscrypto/commit/1b5b2c7ade0cb3b673b998a55ce7fcf27fcc72f3))
- rscrypto: hashes: make blake3 parallel policy truly tuned and remove legacy dispatch/streaming overhead ([54673f5](https://github.com/loadingalias/rscrypto/commit/54673f5f7f9bf6acb03d5a9f985b11f5543610d6))
- rscrypto: blake3/aarch64: optimize tails and keep asm chunk-compress hot on unaligned input ([dd6730e](https://github.com/loadingalias/rscrypto/commit/dd6730e9787874013d16d4d2802d79fa9935387a))
- rscrypto: hashes/platform/backend/tune: unify BLAKE3 dispatch+policy, tighten SIMD paths, and harden ARM64 detection ([2a6d906](https://github.com/loadingalias/rscrypto/commit/2a6d9061a56b1b8eb7bc3e27b48ddd58eb1215f7))
- rscrypto: fixing the CI issues. ([38a8104](https://github.com/loadingalias/rscrypto/commit/38a8104e0d86315fdc92d4cc93dc1ae17c012038))
- rscrypto: fixing CICD and fixing Linux ARM64 ASM issues w/ alignment ([39b6ee1](https://github.com/loadingalias/rscrypto/commit/39b6ee1b68c325a0688f7ae8f46efe2f670357e7))
- rscrypto: fixing CI; fixing Linux ARM64 ASM ([11d937f](https://github.com/loadingalias/rscrypto/commit/11d937f06c30b592763e8dc46e98fab8e431458b))
- rscrypto: fixing the IBM CICD runners setup. Fixing the ASM issue for the Linux ARM64 runner ([bf72dd3](https://github.com/loadingalias/rscrypto/commit/bf72dd3bf60012ba850342db0c8d84b8ccd87687))
- rscrypto: fallback to NEON when alignment isn't assured for ASM ([44d6b14](https://github.com/loadingalias/rscrypto/commit/44d6b1494b3ac081e5d3629c3bbbe7991a41e4e7))
- rscrypto: fixing the CICD issues across targets for blake3 ([d0daeaa](https://github.com/loadingalias/rscrypto/commit/d0daeaa7a0d9ff02c2d3add1211d1836ea33c7e5))
- rscrypto: blake3 cleanup and perf wins ([2ee7ef6](https://github.com/loadingalias/rscrypto/commit/2ee7ef67648ca04d30664f9459d2e9d1afa71699))
- rscrypto: blake3: major performance updates to the MacOS aarch64 and general NEON paths. Added 'rayon' dep to parallelize the Blake3 multi-core run for now - a single, well audited dep isn't going to kill us in the v1. In the future, we'll look to improve on it, but for now - it's got to be. ([b71e9ff](https://github.com/loadingalias/rscrypto/commit/b71e9ff5df6a2c719faa71a157b74d09dd51a0d8))
- rscrypto: fixing the aarch64 fast paths and adding the keyed/derive streaming tiny-size fast path for x86-64. ([d7d3abb](https://github.com/loadingalias/rscrypto/commit/d7d3abb675dccda71571c99b543a75b8071c980c))
- rscrypto: fixing the small inputs for aarch64 Blake3 ([39d3bc8](https://github.com/loadingalias/rscrypto/commit/39d3bc83abd281efc24e68cf91acb19c4cbdaa78))
- rscrypto: fixing the Blake3 streaming vs streaming benches ([f73682b](https://github.com/loadingalias/rscrypto/commit/f73682b29a8eff4e9857987f65692b8ad0143ce1))
- rscrypto: Blake3 XOF constructor for output-size-aware w/ kernel storage. ([5c02d5d](https://github.com/loadingalias/rscrypto/commit/5c02d5d788f893937846d85ac33e19b8bae53a09))
- rscrypto: cleaning up the feature flags and removing the tuning engine from prod builds (dev-only). Added the std::io UX/DX helpers for the hashes and checksums - extensible for the future work, as well. ([eed0a30](https://github.com/loadingalias/rscrypto/commit/eed0a3066bbb3da201c7824934cfd9a2e3c3c587))
- rscrypto: fixing the compressor block and avx512 XOF asm path. ([eb6bd0c](https://github.com/loadingalias/rscrypto/commit/eb6bd0cc4002fc23c2e14e1e96a501361ea3e8ac))
- rscrypto: asm improvements for the Blake3 on aarch64 ([258c7cc](https://github.com/loadingalias/rscrypto/commit/258c7ccbf82f5da40e372fd4107f7b10f57b1f50))
- rscrypto: added a dedicated global_asm! for aarch64 kernels (root + cv) to close the len == 1024 oneshot gaps on aarch64. ([795be60](https://github.com/loadingalias/rscrypto/commit/795be60dd55e2b97dc142ef986ab4897066d492c))
- rscrypto: adding the IBM s390x/p9-p10 runners via IBM's generosity ([4709c53](https://github.com/loadingalias/rscrypto/commit/4709c53cd1df156e06bb2c39ef27c2a97ea2d6fb))
- rscrypto: blake3 updates and fixes + MacOS tuning apply ([7299b4d](https://github.com/loadingalias/rscrypto/commit/7299b4d65d3c968f48096e8035df51a716c76562))
- rscrypto: tuning updates ([4eb5288](https://github.com/loadingalias/rscrypto/commit/4eb52884eddf75dcdf5da1f5170e3a363e4a74cb))
- rcrypto: updating Blake3 ([57780e7](https://github.com/loadingalias/rscrypto/commit/57780e72a9f6d2fc36d11148c24ac77bb0524321))
- rscrypto: work on Blake3 ([42bb3e8](https://github.com/loadingalias/rscrypto/commit/42bb3e8c5efecb3662020712cbdd6813eeee384e))
- rscrypto: applying tuning results ([a9b42a7](https://github.com/loadingalias/rscrypto/commit/a9b42a7e6c4cce5b0c5038b5016c69c520d8a0f9))
- rscrypto: CI fixes ([b3f8e81](https://github.com/loadingalias/rscrypto/commit/b3f8e81e08590a41f2ffb766dc63ac417f8c2fac))
- rscrypto: tuning failures in bench ([a6db6a7](https://github.com/loadingalias/rscrypto/commit/a6db6a7e945a874a85e3740362132695c7d1a13a))
- rscrypto: fixing tuning and working on the Blake3 x86-64 ([45a314f](https://github.com/loadingalias/rscrypto/commit/45a314f918c6493e55f4802302c3ed47b816f9fb))
- rscrypto: tuning work for Linux/Windows ([aa0e4ea](https://github.com/loadingalias/rscrypto/commit/aa0e4ea46617e3595427de68f73ad6939f602c40))
- rscrypto: extending the asm for Blake3; regen script in place. massive tuning update and efficiency update ([170babe](https://github.com/loadingalias/rscrypto/commit/170babe1b567a00c0a8b07057f37ee0ebc1c3c00))
- rscrypto: tune apply: per-size-class winners, cross-arch stream mapping, and self-check; some cleaning and unification; implementing the s390x && risc-v runtime detection ([280e22c](https://github.com/loadingalias/rscrypto/commit/280e22c25eb386be0987e36dc2e382a79fda1fac))
- rscrypto: fixing the blake3 tuning w/ the 2D approach and improved the tuning globally for the differences between checksum/hashes. ([d024b64](https://github.com/loadingalias/rscrypto/commit/d024b64c00d549e5c7a07bce49e87af25d9b2c66))
- rscrypto: fixing the tuning engine for Blake3 ([48c0560](https://github.com/loadingalias/rscrypto/commit/48c0560f364e65c5c943e54d58cb7f9c83422a54))
- rscrypto: tuning adjustments ([c63b248](https://github.com/loadingalias/rscrypto/commit/c63b248ce3af863a44a0739bd1def6c62995718f))
- rscrypto: fixing the gating for AVX512 ([afdce09](https://github.com/loadingalias/rscrypto/commit/afdce09b57ea88a35dd615ca6319e932914c3ce2))
- rscrypto: fixing the parent folding on AVX512 ([6d9f3f1](https://github.com/loadingalias/rscrypto/commit/6d9f3f1db4553965454ba866aaf9dd28e090e068))
- rscrypto: fixed the ASM gating where AVX512_READY was too strict. ([71d52dc](https://github.com/loadingalias/rscrypto/commit/71d52dc9194a406ce4eabb8b054c774c4756cc10))
- rscrypto: namespaced the rodata symbols to avoid inlining a single blob ([07295d2](https://github.com/loadingalias/rscrypto/commit/07295d2e9e07aec907dc5936436c9019c74dd7f0))
- rscrypto: adding the asm for Blake3 ([94c9d42](https://github.com/loadingalias/rscrypto/commit/94c9d42f4c773ec766796bf9c44955ad45915483))
- rscrypto: replaced the per-block compressor path for Blake3 ([a3e1194](https://github.com/loadingalias/rscrypto/commit/a3e1194e8b0a75d0765e0e435029fca05945daf5))
- rscrypto: blake3 fixes ([d0ef134](https://github.com/loadingalias/rscrypto/commit/d0ef134ac3f55d4c201cf6116e4f974672979051))
- rscrypto: eliminated the SIMD cliffs and fixed x86_64 dispatch defaults ([c41a1ce](https://github.com/loadingalias/rscrypto/commit/c41a1ce9a4c5da14fe2d60c7b5dbf20d69155469))
- rscrypto: updating the avx512 impl for Blake3 ([4512301](https://github.com/loadingalias/rscrypto/commit/4512301df1da8acd04deb2cf555c535242d8e763))
- rscrypto: x86-64 updates ([262a6c5](https://github.com/loadingalias/rscrypto/commit/262a6c55bc74a98fcedf9095723077aaf3c4bc0e))
- rscrypto: wiring the SSE4.1 SIMD to Blake3 dispatches ([ce31c7c](https://github.com/loadingalias/rscrypto/commit/ce31c7c484e5a478ef472e82b2c8872821b67b94))
- rscrypto: updating the sse4.1 blake3 ([029361a](https://github.com/loadingalias/rscrypto/commit/029361a4fff785a0a8883666550b039e6eda5e38))
- rscrypto: fixing clippy surrounding Blake3 updates ([aaab2e0](https://github.com/loadingalias/rscrypto/commit/aaab2e01e996c5d23f8e14e8e54d726d436a6077))
- rscrypto: working through the blake3 perf optimizations finally. ([32254e1](https://github.com/loadingalias/rscrypto/commit/32254e18a85e433ae9a43733f45da746c710c99d))
- rscrypto: cleaning, streamlining, and preparing the codebase (checksums, hashes/crypto, and hashes/fast) for the SIMD/HW instructions/accel work. Added official test vectors for all hashes and deterministic testing for all algys ([db29973](https://github.com/loadingalias/rscrypto/commit/db299736d852884a0c901fd7b25fe3a3228d0a0c))
- rscrypto: adding the baseline hashes/crypto algys and hashes/fast algys; fuzzing and deterministic testing ([bce02ec](https://github.com/loadingalias/rscrypto/commit/bce02ec6e6e0b255fce1ee9c1c350c8e894dd080))
- rscrypto: fixing clippy issues ([095cdd1](https://github.com/loadingalias/rscrypto/commit/095cdd12d2c01e9cdd3ebd53ad1a2f9905ebe69a))
- rscrypto: standing up the first version of the crypto-side hashes - pre-SIMD/HW instructions ([d1da75c](https://github.com/loadingalias/rscrypto/commit/d1da75c482c9be3946e451081e6c0328bb6193ea))
- rscrypto: adding the vectored CRC APIs and fuzz-target for it. Ran tuning and updated the codebase. ([314ff77](https://github.com/loadingalias/rscrypto/commit/314ff7776eb7e6e566b08b630f7619695b573781))
- rscrypto: adding the LICENSES; cleaning the codebase up after reverting to a monorepo. ([a428e14](https://github.com/loadingalias/rscrypto/commit/a428e14a5b2fc20883fc6e9416c9128d34eb4b27))
- rscrypto: adding the Intel IceLake runner to populate that table ([b8f2e0d](https://github.com/loadingalias/rscrypto/commit/b8f2e0d7ff03bf72724f9fb89abbe67026fa01ce))
- rscrypto: updating the bench.yaml for the Zen5 runner. ([a61c7c9](https://github.com/loadingalias/rscrypto/commit/a61c7c9ff6633d18e38f2a9170e57f1c90c9b24b))
- rscrypto: adding the risc-v benchmarking runner ([24da72e](https://github.com/loadingalias/rscrypto/commit/24da72ef605fd628cc4ef5cc786fae67ac8cabe4))
- rscrypto: cicd updates to bust the cargo-audit cache for Windows ARM64. ([f9871cf](https://github.com/loadingalias/rscrypto/commit/f9871cfc31f0bc9c741cb731b1323ef4366a56c6))
- rscrypto: update to bust the cache/bin again ([e9ed114](https://github.com/loadingalias/rscrypto/commit/e9ed11438b2c348de1c4fc7d56c268ce5b083c10))
- rscrypto: fixing the caching issues for the Windows ARM64 runner ([b14b4b8](https://github.com/loadingalias/rscrypto/commit/b14b4b807b050265355b090e12a55d8e06519f23))
- rscrypto: fixing the caching issue w/ the windows arm64 runner ([7b1a94b](https://github.com/loadingalias/rscrypto/commit/7b1a94b77f756026005f7d07d20e4f8d24db544d))
- rscrypto: tuning ([b62a733](https://github.com/loadingalias/rscrypto/commit/b62a73395d5f65a133543206baa5efedcf7c1eae))
- rscrypto: fixing the risc-v runner label, hopefully. Looking at the results from the tuning to determine what's going on w/ the tuning engine. busting the cache for the windows arm64. ([49b6e60](https://github.com/loadingalias/rscrypto/commit/49b6e60b6add44aeaa630488e1b4468e1d1a61a7))
- rscrypto: updating the bench.yaml and adding the RISC-V runners ([b0bde6f](https://github.com/loadingalias/rscrypto/commit/b0bde6fca86c420d190ad9cace35474c60d3b93c))
- rscrypto: cleaning ([3fd4b6e](https://github.com/loadingalias/rscrypto/commit/3fd4b6e8ea2a7129a2cdac62854048cdd849dc10))
- rscrypto: added examples, contributing.md, and updated the readme.md; added introspection and hid the slice-by-4 code. updated the ENV usage and cleaned the codebase for release. added the proper AWS instances, Namespace instances, Github instances, and RISC-V instances for CICD/benching. updated the GHA pins. ([2c2f176](https://github.com/loadingalias/rscrypto/commit/2c2f176290eeadded14607f4858205e02ee7f14e))
- rscrypto: reducing overhead ([81dd7ec](https://github.com/loadingalias/rscrypto/commit/81dd7ec05a9ad79b7aaa6bbf4fe0d01966b6f4bb))
- rscrypto: fixing the off-by-one error in the double-unrolled loops ([7f73666](https://github.com/loadingalias/rscrypto/commit/7f736668b9758b9896f30a4e9f2c2e6eaf5bb839))
- rscrypto: improving the gen scripts and tuning caps safety. ([e3d7efa](https://github.com/loadingalias/rscrypto/commit/e3d7efa396cbc0449bcfde9f5da596f60d3778d7))
- rscrypto: fixing via pre-push hook ([4834a3e](https://github.com/loadingalias/rscrypto/commit/4834a3e7fb6343830ce70a8f4f77e9a6859dd9a5))
- rscrypto: adding the double-unroll and prefetching to the kernels it fits. ([0d08dec](https://github.com/loadingalias/rscrypto/commit/0d08dec8a346539edca9448cefb3dc7a103b9271))
- rscrypto: updating the tuned defaults via dispatch now that we've removed the tuned_defaults overhead and run all kernels/buffers ([5cafd14](https://github.com/loadingalias/rscrypto/commit/5cafd14d9657f5ba3f83d95149290bd876e25426))
- rscrypto: updating the codebase via removal of the old dispatch overhead and solving some tuning issues. ([702c440](https://github.com/loadingalias/rscrypto/commit/702c4407cb697c739478f86aaa6d50fd453a52af))
- rscrypto: fixing the unused issue for x86-64; gating the extern crate alloc. ([2fa87bb](https://github.com/loadingalias/rscrypto/commit/2fa87bbf64a481f9c1b8f5567f4453c6c57804bb))
- rscrypto: fixing the tests ([88ec68c](https://github.com/loadingalias/rscrypto/commit/88ec68c5a8aaa900d7ce9cfcd5ee919b23c39f63))
- rscrypto: tuning ([0938d5e](https://github.com/loadingalias/rscrypto/commit/0938d5e1b49873484193ef6fe62e310679a32fab))
- rscrypto: update for CI ([eef34c9](https://github.com/loadingalias/rscrypto/commit/eef34c9702c82b66dde8148b2c1b2c68c0a5e36b))
- rscrypto: refactor: pre-release cleanup for v0.1.0 - De-macro CRC-64 types for better auditability Replace   define_crc64_type! macro with explicit implementations matching the   CRC-32 pattern - Fix repository URLs in CONTRIBUTING.md - Document tuning coverage in README.md - Add platform coverage matrix   showing measured vs inferred presets and list hardware we need   contributions for - Remove speculative flag from Tune struct. The field added complexity   without practical benefit; tuning docs now indicate extrapolated   values in comments instead ([f89284d](https://github.com/loadingalias/rscrypto/commit/f89284db0f44f71fe3eae5df4b1c50193bf7f28b))
- rscrypto: fixing the tuning engine to use the KernelSet and update the existing tables in dispatch.rs ([ce9b4a5](https://github.com/loadingalias/rscrypto/commit/ce9b4a51e844f40d2f8187d929e1576ed6674195))
- rscrypto: improved performance everywhere via an architectural update. dropped the policy/runtime code for the determinism of compile-time - based on benches. ([89a5dcf](https://github.com/loadingalias/rscrypto/commit/89a5dcfbf7a4969623987f115446a052f0b5e1c7))
- rscrypto: tuning ([179c144](https://github.com/loadingalias/rscrypto/commit/179c1443ceece7b81ce4dae8f006803a292f5935))
- rscrypto: improving the crc32 code and improving the bench runtime speed ([c077292](https://github.com/loadingalias/rscrypto/commit/c07729223fc61c0a83d6cc2e68d5b8b12f321c28))
- rscrypto: fixing the generated tuning defaults ([b591256](https://github.com/loadingalias/rscrypto/commit/b591256276c99b90e2d996757213dd441dfd2fe2))
- rscrypto: added the diagnostics for kernel selection to be certain we know what's being run and why. ([f7ab9c8](https://github.com/loadingalias/rscrypto/commit/f7ab9c8c2f80ad3e0cb3e72c72a53d1e040b41d2))
- rscrypto: pushing the tuned updates ([67ee55b](https://github.com/loadingalias/rscrypto/commit/67ee55be1e7db95ef2fdf2b4361d5efd13af30a9))
- rscrypto: added the improvements to the comp bench and DRYed out the codebase, prepping for the hashes/aead improvements. ([0749f44](https://github.com/loadingalias/rscrypto/commit/0749f4496ab1e6799c15cd4e6442259273d50720))
- rscrypto: fix(backend,checksum): unbreak Arm EOR3 selection; remove CRC64 dispatch overhead ([ffb5b47](https://github.com/loadingalias/rscrypto/commit/ffb5b47b6d9cfb9ba8ba0dce2e0da32d4107cdb7))
- rscrypto: tune: add tunable cutoff for CRC64 small kernel. Add `small_kernel_max_bytes` threshold to CRC64 config, allowing the tuning engine to find the optimal crossover point between the small single-lane SIMD kernel and the multi-stream folding kernels. ([9d19bd4](https://github.com/loadingalias/rscrypto/commit/9d19bd4f261228df83303ded3720335b4f2c5c2c))
- rscrypto: prepping for release to OSS; fixing the dispatch issues ([1f87a1f](https://github.com/loadingalias/rscrypto/commit/1f87a1fed67e4b5cb2c640742ba6cf0e5dc31270))
- rscrypto: adding the needed CONTRIBUTING.md for the community to turn in their tuning results or tuned_presets updates to cover all the arches. ensured the doctests actually ran. ([94f9c60](https://github.com/loadingalias/rscrypto/commit/94f9c60391f82af1a9ef383f818f5bde842fd7df))
- rscrypto: added Intel tuning best guesses ([d96958f](https://github.com/loadingalias/rscrypto/commit/d96958fbbf25a8dc235b8df5ecf389e2ced10bfd))
- rscrypto: fixing the crc32 smoke test ([88732c2](https://github.com/loadingalias/rscrypto/commit/88732c204268fb1dd61631c30d8670f26fa0f555))
- rscrypto: removed the 'powerpc64' naming conventions all over and replaced it w/ simply 'power'. improved the multi-stream for the risc-v, power, and s390x arches. improved the small buffer sizes. also, completed tuned defaults (with what we have access to) + tuning/apply pipeline; add multi-stream + POWER naming; fix CRC32 thresholds and small-buffer selection ([9da27a9](https://github.com/loadingalias/rscrypto/commit/9da27a98727292611b2465b958cec35cd46438b7))
- rscrypto: fixing the Windows/Linux gating of the aarch64 code in CICD. ([7f2d2b5](https://github.com/loadingalias/rscrypto/commit/7f2d2b5e915a07739883838493e2f134e9dd6289))
- rscrypto: expanded crc16citt/ibm, crc24openpgp, and the tuning system. updated the former algos w/ the risc-v, power, and s390x accelerated versions. added tests; scalability. ([25d1242](https://github.com/loadingalias/rscrypto/commit/25d12428d15ab40fd09b147ff437ab63551f6911))
- rscrypto: fixing the PCLMUL/VPCLMUL force modes for CI; fixed the streams noop issues across x86-64 and ARM64. improved the tuning engine's accuracy via stream awareness across algorithms/variants ([f6a52e7](https://github.com/loadingalias/rscrypto/commit/f6a52e76ec11e8a95d481773e2669e2bf068bbbf))
- rscrypto: improving the mid-range crc32 variants on ARM64 ([2b76bb5](https://github.com/loadingalias/rscrypto/commit/2b76bb555e29b96ec3ce8e4a221d18cf1edb2c4b))
- rscrypto: updating the tuned_defaults for the systems we know. ([d0908eb](https://github.com/loadingalias/rscrypto/commit/d0908eb725cc087328f60fc261eba99d7d4e8113))
- rscrypto: tuning updates ([17478ba](https://github.com/loadingalias/rscrypto/commit/17478ba8cf1f08e1996f7eb8e42f2659c35926f4))
- rscrypto: fixing the crc32 (ieee) implementations acceleration over multi-streams ([b919fad](https://github.com/loadingalias/rscrypto/commit/b919fad82da6281dbf26edc58f1e85b27987006d))
- rscrypto: fixing the CI ([8a2514b](https://github.com/loadingalias/rscrypto/commit/8a2514b654ff051c20fab94f0d9bd503c9d45674))
- rscrypto: fixing the tuning stats ([92f10d2](https://github.com/loadingalias/rscrypto/commit/92f10d24cb087f3d2ffd1cf43c99385bcb2474e8))
- rscrypto: cleaning up the bench.yaml manual triggers ([8b205a3](https://github.com/loadingalias/rscrypto/commit/8b205a3c8edb34eb186f2e90e1be94e2b0ae2992))
- rscrypto: building a proper fucking tuning engine for Rust crypto libraries and it's outstanding ([d1131d5](https://github.com/loadingalias/rscrypto/commit/d1131d524d5b7da99a655f61967b2899fa5d980c))
- rscrypto: checksum: add kernel equivalence fuzzing for CRC16/CRC24/CRC32 - Extends the CRC64 kernel_test pattern to all checksum variants, ensuring all SIMD backends produce identical results to the bitwise reference. ([3e18faf](https://github.com/loadingalias/rscrypto/commit/3e18fafbea1e1fddf19228900267c0ea91f31ca0))
- rscrypto: added the helper functions needed to clear CICD ([68a22db](https://github.com/loadingalias/rscrypto/commit/68a22db07b6d161a0acec1d71db980fbe99dfb20))
- rscrypto: unifying the forced selection thought policy/kernels ([408e4a0](https://github.com/loadingalias/rscrypto/commit/408e4a0adcb8f5c9fd6c83742ba9f47c03dca84f))
- rscrypto: fixing the imports ([c608487](https://github.com/loadingalias/rscrypto/commit/c608487f76d35c21f1fd559cb0c094a65ed71900))
- rscrypto: backend: reduce unsafe surface in dispatch/cache subsystem - major refactor ([8513517](https://github.com/loadingalias/rscrypto/commit/8513517f92e301c3fe76ef706800d24bb1df31e4))
- rscrypto: critical safety fixes for x86 AVX/AVX-512 detection and hybrid Intel; restore aarch64 multi-stream tuning contract ([04217f1](https://github.com/loadingalias/rscrypto/commit/04217f16abc17937a0a1cc37e8192c4d985ab47f))
- rscrypto: fixing the constant assertions causing issues via Clippy; removed the CRC64 Smoke Tests from CI ([e3bd5d7](https://github.com/loadingalias/rscrypto/commit/e3bd5d7550b31710294be31f4c7d549e7b265e22))
- rscrypto: updated the tuning engine significantly. added per-lane byte matching the hardware; added explicit flags and clear override chains. addressed the memory-bound crc32c on Zen4/NEON. the tuning engine offers a fully tunable checksum system... and it's benchmark driven now. fixing the tuning infrastructure in the justfile/bench.yaml, as well. ([abfe822](https://github.com/loadingalias/rscrypto/commit/abfe8228a527bbb8989c41d14d313e79fd9cd096))
- rscrypto: refactored the backend/ to be much cleaner; more performant; more maintainable. updated the checksum crate to use the new architecture. rmeoved overhead. ([61b3278](https://github.com/loadingalias/rscrypto/commit/61b32782c2c817b0faecab75d8e4d93eae9b07d9))
- rscrypto: refactored the property tests into the proper, uniform, centralized structure they should have been from day one. refactored the kernels and dispatch to improve the readability and uniformity across the codebase. ([add55d1](https://github.com/loadingalias/rscrypto/commit/add55d14fb91e64864b3a4060d7969898e9b73c0))
- rscrypto: checksum: add cross-kernel equivalence fuzz target and testing infrastructure; executed the fuzzer and the miri tests, along w/ full test suite to validate the safety/soundness across the crate. all is well. gated a handful of property tests for Miri execution. ([133b61b](https://github.com/loadingalias/rscrypto/commit/133b61bcba9ecd5fe25fb893aa9e3345263b03cd))
- rscrypto: added common/proptests.rs with rigorous property-based tests for all 7 CRC variants. Tests prove two fundamental invariants against our bitwise reference: ([10ac73a](https://github.com/loadingalias/rscrypto/commit/10ac73ae31349fe29e6de24a35dce1bd0cde2ba7))
- rscrypto: improving the tuning/selection logic; it sets the tone for all other implementations in the codebase - including hashes, aead, etc. ([94fa770](https://github.com/loadingalias/rscrypto/commit/94fa770d4a8d7a654e9b68d5d12b00b2a7c358a4))
- rscrypto: fixing the combine overhead holding back the crc32c impl ([fab0489](https://github.com/loadingalias/rscrypto/commit/fab04899ca12ee31e116648303705243116703bd))
- rscrypto: cleaning the bench.yaml workflow ([02603c0](https://github.com/loadingalias/rscrypto/commit/02603c0da823d2781f651322d67aeb4214e6d660))
- rscrypto: fixing the bench.yaml for diagnostics and tuning ([e8b7bc7](https://github.com/loadingalias/rscrypto/commit/e8b7bc7a0b1df65e2c6b066ed873c4997f2aa70b))
- rscrypto: tuning the checksum implementations ([773a767](https://github.com/loadingalias/rscrypto/commit/773a767387bf75babf670d3df7344a0ac0ac0d56))
- rscrypto: fix CRC32/CRC32C x86 dispatch + multistream combine; add comp-report toggle ([ea792d4](https://github.com/loadingalias/rscrypto/commit/ea792d484bff1f4f1e9fe68c6e0dcfc218a21158))
- rscrypto: scaffolded hash/ crate and blake3 module. Updated the crc bench to be sensible. ([a46f892](https://github.com/loadingalias/rscrypto/commit/a46f8928f7b3be394aea6cd90586a621a785c4fe))
- rscrypto: tuning the checksum crate for performance and efficiency ([1948c29](https://github.com/loadingalias/rscrypto/commit/1948c29e88eba28a691c36d8d18d471913321218))
- rscrypto: fixing w/ pre-push ([64e867c](https://github.com/loadingalias/rscrypto/commit/64e867c7ade3058e66d969b5e307c88d3a9aec64))
- rscrypto: adding the CRC16CITT, CRC16IBM, and CRC24PGP initial impls. ([a0091b7](https://github.com/loadingalias/rscrypto/commit/a0091b7e2a22d21fdf91b799d96fb766d775cd32))
- rscrypto: alignment and parity between the CRC64 reference impls and the CRC32 impls. Checking API shapes; UX/DX; streaming. Ensuring the tests/benches are solid and preparing for the CRC16 (2) and CRC24 (pgp) impls. ([f7dabe0](https://github.com/loadingalias/rscrypto/commit/f7dabe08df47678c63def351ed3ad6d66e120fdc))
- rscrypto: checksum - add dedicated crc32 fuzz target; fuzz+miri weekly on x86_64+aarch64 ([bf6a046](https://github.com/loadingalias/rscrypto/commit/bf6a046c9efca6e0c67b5526424b62af0768fa9e))
- rscrypto: checksum/crc32 - add POWER VPMSUM, s390x VGFM, riscv64 Zbc/Zvbc backends ([8fd3d58](https://github.com/loadingalias/rscrypto/commit/8fd3d58f2650c74a51e0890d4646457a463a4c30))
- rscrypto: removing the unused return ([3a6ce64](https://github.com/loadingalias/rscrypto/commit/3a6ce64d3a786bc4abe331e4d51e876346660b45))
- rscrypto: crc32: add small-buffer + aarch64 fusion parity; add crc32-tune tooling ([a20c7b9](https://github.com/loadingalias/rscrypto/commit/a20c7b9709b5cee3dbaa69ae236e5670fffabd00))
- rscrypto: fixing the clippy issue w/ the needless return in ci ([13cdc24](https://github.com/loadingalias/rscrypto/commit/13cdc24fe68c96ad78d3e5dac9308ac4357f0f5d))
- rscrypto: improving the aarch64 and x86-64 perf. cut tiny-update overhead; add SVE2-PMULL force tier. Cache CRC32/CRC32C dispatch params in std builds; remove per-update() config/caps checks. Fix aarch64 PMULL/PMULL+EOR3 stream slot mapping; add fusion min-size gate ([4827b87](https://github.com/loadingalias/rscrypto/commit/4827b8730ccaf117e360c54d6bbe27489f1b4717))
- rscrypto: adding fuzzing for the crc32/crc32c and cleaned crc64 bench that's too heavy. ([d478f2b](https://github.com/loadingalias/rscrypto/commit/d478f2b09ae2e05574d2e8e05194ddc70de87d2b))
- rscrypto: fixing the fold constants in the CRC32 (IEEE) and gated the AVX512 tests ([9868cc1](https://github.com/loadingalias/rscrypto/commit/9868cc16f8978ef28e6ef9ce148ff126a13bfc3b))
- rscrypto: removed the Windows ARM64 runner - it's not a Windows ARM64 runner. repaired a real x86-64 CI issue in the crc32 sanity check that was comparing the 16B fold-coefficient in the wrong order (high, low). ([a1a6831](https://github.com/loadingalias/rscrypto/commit/a1a6831924c5109178f4309e120e0c5d64b812fd))
- rscrypto: added the per-stream kernels and wired into the dispatch. added sse4.2 stream wrappers and aarch64 crc extensions. updated the stream selection to engage for large buffers and added correctness tests for the folding constant gen + multi-stream variants ([e98cda3](https://github.com/loadingalias/rscrypto/commit/e98cda312059112cc6c0c897971e97bd3943a048))
- rscrypto: fixed the cicd issue w/ crc32 ([0044792](https://github.com/loadingalias/rscrypto/commit/0044792fd8dd4a14b4e3474a3a89a8b7d5bb3f41))
- rscrypto: fixing the crc32 issue from CI ([6c9f20e](https://github.com/loadingalias/rscrypto/commit/6c9f20e4943724cf01534cfad5f81a731a6e37ad))
- rscrypto: finished wiring x86-64 wiring and clamped only when the CRC32C actually runs. The audot select for PCLMUL is present. The comp bench was updated w/ crc-fast (crc-fast-rust) comparison. ([9fbe330](https://github.com/loadingalias/rscrypto/commit/9fbe3309318360f88acb9375104e8c4b1b173b05))
- rscrypto: added HWCRC impl for CRC32 and CRC32C on aarch64; updated the x86-64 impls w/ fusion tiers as well. auto-selection wired; portable slice-by-16 software integrated ([2322ea3](https://github.com/loadingalias/rscrypto/commit/2322ea347ae16d1f9bec012b5af1d362cfd50491))
- rscrypto: adding the CRC32 and CRC32C to the fold ([856c25f](https://github.com/loadingalias/rscrypto/commit/856c25f3cd3f5222cd89708a5dd764ff5c87ab77))
- rscrypto: updating the bench.yaml to allow for tuning scripts to run w/o the need to run a full benchmark. ([2f9f1a4](https://github.com/loadingalias/rscrypto/commit/2f9f1a4c2ff81290f6abc853b49d520ee4ea041a))
- rscrypto: fixing the bits/bytes mistake for folding distances in the new VPCLMUL 4x512-bit kernel ([5ef6f90](https://github.com/loadingalias/rscrypto/commit/5ef6f90df64006b01bbcf57adcea6eba94752fb2))
- rscrypto: fixing the AVX512 smoke and investigated the ARM64 Windows; no real solution, though ([f432d41](https://github.com/loadingalias/rscrypto/commit/f432d4167b06b3543977816089ca6d8d42924616))
- rscrypto: fixing the tuning engine/scripts; adding the 4x512-bit processing kernel. ([96e81fd](https://github.com/loadingalias/rscrypto/commit/96e81fd510b0a858f47b232c1355923c0333dd2d))
- rscrypto: adjusting thresholds for the Intel SPR triggers; adding the CPU print to the comp benches. ([44cf09a](https://github.com/loadingalias/rscrypto/commit/44cf09aa9c7972c321bcd78e5bd093a68bbc6001))
- rscrypto: adding the 8-way folding impl for the x86-64. ([9551be4](https://github.com/loadingalias/rscrypto/commit/9551be4dcf69f6b62adee0e9ce6c8059c8282368))
- rscrypto: made adjustments to the fuzzing infrastructure for more control/containment; improved the target-triples selection. ([2a5d6cb](https://github.com/loadingalias/rscrypto/commit/2a5d6cb57200e150bb8415016524dc82b4e372be))
- rscrypto: fixing the fuzzer in cicd ([8f01899](https://github.com/loadingalias/rscrypto/commit/8f0189979b1f668dbd3e03abffad513aa75eab75))
- rscrypto: fixing the fuzzing/bench/weekly workflows. ([ab12365](https://github.com/loadingalias/rscrypto/commit/ab12365fd15703ee1ff39d24da67a981d074a21d))
- rscrypto: cleaning out the nightmare that was the CRC32 and CRC32C impls; adding crc-fast to the property testing suite; improving the testing/validation of the existing CRC64XZ/CRC64NVME before adding the 8-way fold we're missing. ([7f5b158](https://github.com/loadingalias/rscrypto/commit/7f5b158d03a930fdb7f14d58addc431c8955d91c))
- rscrypto: fixing the -1 offset (CRC32) still broken ([a5d1fac](https://github.com/loadingalias/rscrypto/commit/a5d1facea5415b6480e366f9df8158868d1d38e6))
- rscrypto: fix the CRC-32 CLMUL fold coefficient ordering (it was backwards and wasn't allowing the larger exponent to 'shift up'. ([a4635a7](https://github.com/loadingalias/rscrypto/commit/a4635a7fc5a487a19619b01529bdf498a8ba757e))
- rscrypto: fixing the CRC32 CLMUL fold coefficient exponent formula ([cfa1cd6](https://github.com/loadingalias/rscrypto/commit/cfa1cd621f200e244b292f73fdf6c74054706493))
- rscrypto:fix(crc32): correct CLMUL fold coefficient exponents ([bcf51a7](https://github.com/loadingalias/rscrypto/commit/bcf51a733515b0ea067a05db060bdbddeb40d2d1))
- rscrypto: feat(crc32): implement x86_64 VPCLMUL kernels for CRC32/CRC32C ([b236b61](https://github.com/loadingalias/rscrypto/commit/b236b615b7a1215401215857037d1baa6c38d546))
- rscrypto: fixing crc32 again ([c444098](https://github.com/loadingalias/rscrypto/commit/c44409838a12dcdcaeabb48eb6fc57ee94a2bc58))
- rscrypto: fixing the module inception ([336f3b9](https://github.com/loadingalias/rscrypto/commit/336f3b92bfd00c7cd51191276b5e1ffdaf00b724))
- rscrypto: cleaning up the architecture, crc32/crc32c, and the testing. ([77ce7a3](https://github.com/loadingalias/rscrypto/commit/77ce7a3beb06a3f79886d2596b61d6d9e588e535))
- rscrypto: fix(crc32): use correct CLMUL selectors (0x10/0x01) for CRC-32 fold_16 ([d6d189e](https://github.com/loadingalias/rscrypto/commit/d6d189e00e02a811c2fb77c1048ba78143024c70))
- rscrypto: fixing the crc32 issue ([6ebe02e](https://github.com/loadingalias/rscrypto/commit/6ebe02eca37bd00ef25bc52688bd110b8de0d6cc))
- rscrypto: fixing the fold_8 w/ fold_width and updating the Crc32ClmulConstants for CRC32/CRC32C. ([1baa646](https://github.com/loadingalias/rscrypto/commit/1baa646fc236326e24331dec2db3d1d87f1949e7))
- rscrypto: fixing the crc32/crc32c algorithms ([71195ab](https://github.com/loadingalias/rscrypto/commit/71195ab911eb3a57e06887170ad5329088eba132))
- rscrypto: fixing the CRC32/CRC32C mu tests on Linux/Windows x86-64. ([f2e5dc9](https://github.com/loadingalias/rscrypto/commit/f2e5dc9994bc5be7bbcd674d64726a4e106893f5))
- rscrypto: crc32: fix Barrett reduction CLMUL imm8 operands in x86_64 PCLMUL ([efb0c0b](https://github.com/loadingalias/rscrypto/commit/efb0c0b8f0d6479891dcf70a9c425b375ced0e1d))
- rscrypto: updated the GHA SHAs and pinning ([5aff4a4](https://github.com/loadingalias/rscrypto/commit/5aff4a41b43f321e4ba448acfbab48a480cd35c8))
- rscrypto: crc32: add 7-way PCLMUL kernel, completing x86_64 PCLMUL set ([d620d56](https://github.com/loadingalias/rscrypto/commit/d620d567185925d883e59e4d7d85bbc002f9d0d2))
- rscrypto: added powerpc VPMSUM; s390x VGFM 4-way; riscv ZVBC vector, ZBC scalar; removed the 'loongarch' and planned to update w/ Chobra. CRC64/xz-nvme is getting closer to the 'reference' impl we're aiming for ([1303e1f](https://github.com/loadingalias/rscrypto/commit/1303e1f16c4339d4ae71aa5b8845754ecb0acf4f))
- rscrypto: added powerpc64; updated target-triples and zig-cc scripts ([6e0ce61](https://github.com/loadingalias/rscrypto/commit/6e0ce6162119286fc58e24ca565e96033a4d5e2e))
- rscrypto: improving the crc64 tables (8>16) and the tuning scripts/cli workflow ([e8731f4](https://github.com/loadingalias/rscrypto/commit/e8731f4f7b88cb99c826a17b92c2614446e70dee))
- rscrypto: bench and tune refinements; working on the tuning scripts. ([ebfc373](https://github.com/loadingalias/rscrypto/commit/ebfc373ad4031b4008d4935b9661078fec99e9e9))
- rscrypto: added the EO3 (XOR) and multi-stream update to the aarch impl; added VPTERNLOGD impl for the x86-64. ([a7d54c1](https://github.com/loadingalias/rscrypto/commit/a7d54c139c1ee7b47e2cd1fd012e6d2dd8276df4))
- rscrypto: added the EOR3 kernel to automatically run on any ARMv8 CPU w/ SHA3 extensions for buffers larger than 128 bytes. ([fe2127b](https://github.com/loadingalias/rscrypto/commit/fe2127b646ab7fa1b9e50e2a0f98a84f308aa77c))
- rscrypto: added 3-way fold to the arm64 arches ([7f43652](https://github.com/loadingalias/rscrypto/commit/7f43652485a489bad1aa68902ae74c66aee8fd99))
- rscrypto: fmt; fuzz updates ([23584ea](https://github.com/loadingalias/rscrypto/commit/23584ea8958a9250774293f3c016e049f3c0fb33))
- rscrypto: fixing the check.sh script to actually cover the codebas's targets. fixed the Windows x86-64 benches to use the free runners. fixing the fuzz testing issues. ([5f0e021](https://github.com/loadingalias/rscrypto/commit/5f0e021ad0c6ed76e3fb466dbc5067adef9b0728))
- rscrypto: fmt - cicd fixes ([ede4a12](https://github.com/loadingalias/rscrypto/commit/ede4a121a729e11dc807478bddc756958f7b5a56))
- rscrypto: more x86-64 fixes. ([5d7784d](https://github.com/loadingalias/rscrypto/commit/5d7784daa1216e7668cbfe105840ae6a8d98f9d4))
- rscrypto: fixing the fmt again; cicd fixed ([1a5d9c0](https://github.com/loadingalias/rscrypto/commit/1a5d9c0c20c146a21730018a82dc932bcf926889))
- rscrypto: fixes to the x86/arm64 Linux CICD ([17ceb99](https://github.com/loadingalias/rscrypto/commit/17ceb9912a6ceaffbfb528c2f1e7f2fb2e84a57b))
- rscrypto: added reference implementation for the CRC64XZ/NVME + backend/platform/traits integration + ci: full SIMD tier coverage, aarch64 Miri, tune discovery, regression detection ([15f95ce](https://github.com/loadingalias/rscrypto/commit/15f95ceda6c92b42c01aece31170ee7ceef55665))
- rscrypto: fix: Miri failure & test-fuzz.sh script fix. ([a50b0ae](https://github.com/loadingalias/rscrypto/commit/a50b0aef057520872e5ce9d27fa4c31b76c6efa9))
- rscrypto: fix: replaced the manual 'caps = caps| x w/ idomatic caps |= x compound assignment operator ([19e1483](https://github.com/loadingalias/rscrypto/commit/19e1483addb5bd185e8af20e4500ceba0d4ce7df))
- rscrypto: fix: resolve CI failures across all targets ([286f503](https://github.com/loadingalias/rscrypto/commit/286f503c0ff65f0fb2b8f2e88a7ddbc191261cbf))
- rscrypto: refactor is required for us to build anything worth a shit; checksum and hashes removed; added the callback macro w/o pulling in procmacro crate ([4381375](https://github.com/loadingalias/rscrypto/commit/438137531ccf3e8a9bed371d2fe417c808ad7c70))
- rscrypto: backend/platform dispatch work; updated the 'indexing_slicing' lint. ([1cc2adc](https://github.com/loadingalias/rscrypto/commit/1cc2adcbda01eee4d9fb2d439494062a08831409))
- rscrypto: fix: no_std sentinel cfg issues w/ 64-bit atomics ([8922f46](https://github.com/loadingalias/rscrypto/commit/8922f46e2856e395a22dde83422f9aef9c8a8cf7))
- rscrypto: fix: lints/cfg issues ([ae97e0d](https://github.com/loadingalias/rscrypto/commit/ae97e0d816ad9b8183dda27d0a34187edd619805))
- rscrypto: fixed the Miri cfg issues; fixed the runtime detection && PCLMULQDQ enabled in benches ([f7ef78c](https://github.com/loadingalias/rscrypto/commit/f7ef78cd3726d16d2e6655231ff8bcf98734b243))
- rscrypto: fixing Miri stupidity and the cpu-native ([1532ecb](https://github.com/loadingalias/rscrypto/commit/1532ecb52bf7ba8b555ee9609834e82a415ac15c))
- rscrypto: added the weekly.yaml for fuzzing/etc. ([9081378](https://github.com/loadingalias/rscrypto/commit/9081378dcf5170dab22c5fbae90d45b2e6cdda96))
- rscrypto: fixing the Barrett reduction order; it was backwards. ([1f4b4ab](https://github.com/loadingalias/rscrypto/commit/1f4b4ab5fbc43a7bd46085567874a5defca127e9))
- rscrypto: added new no_std targets; removed the 'std' gate because it's just not needed and then added hand-rolled errors ([51283a0](https://github.com/loadingalias/rscrypto/commit/51283a05f6f19dfb0aaff944eea48e3dfec89fa5))
- rscrypto: ci: fix CI failures and expand no_std target coverage; setup  new Namespace 'profiles' for rscrypto, too. ([625044e](https://github.com/loadingalias/rscrypto/commit/625044e71d469de09f80b5d7d7a7f11a6f4fd4ca))
- rscrypto: fix the cicd issues; simple ([632a18a](https://github.com/loadingalias/rscrypto/commit/632a18ac4779e3939d95c0cb9b1ec59d1a4e3925))
- rscrypto: wiring the 'pre-push' hook/script. ([e76634f](https://github.com/loadingalias/rscrypto/commit/e76634f87b5a88a2b97216f9ac4935e0d05789e2))
- rscrypto: initial commit ([3d06ff9](https://github.com/loadingalias/rscrypto/commit/3d06ff975c0baaed6b2067df041b11266bca696b))

### ⚡ Performance

- **ed25519**: eliminate IFMA overflow corrections in verify hot path ([6434ebf](https://github.com/loadingalias/rscrypto/commit/6434ebf72c5323c7f197b90b7b38425abd1a3f39))
- **aead**: fuse GCM-SIV + AEGIS paths, add wide POLYVAL for ARM/POWER/s390x ([4423c21](https://github.com/loadingalias/rscrypto/commit/4423c213546f1e92dcc91c123bea27d44b4e0caf))
- **aead**: fuse AEGIS-256 init/aad/encrypt/finalize into single target_feature scope ([7534fd1](https://github.com/loadingalias/rscrypto/commit/7534fd1cd0a2050f21a405162e1afb66bc9d0574))
- **auth**: asymmetric IFMA mul eliminates reduce in double ([7c741f4](https://github.com/loadingalias/rscrypto/commit/7c741f44bf5a7d9fc88b4490b0ea8f79df20989e))
- close 4 acceleration gaps (KECCAK-2/4, SHA-2, IFMA) ([aa31440](https://github.com/loadingalias/rscrypto/commit/aa3144098e9c8ebb27cf1e41304c8a145fa6b63b))
- **auth**: eliminate reduce in IFMA double + static basepoint table ([df53f0b](https://github.com/loadingalias/rscrypto/commit/df53f0bb6d1e0b3e6e8b563abeae6f9db74e1784))
- **auth**: route Ed25519 verify to AVX2 — IFMA is structurally slower ([380005b](https://github.com/loadingalias/rscrypto/commit/380005bebbfb8d6f695d27a6eb1749b143dae5b9))
- rapidhash native-endian reads, precomputed seed; SHA-2 volatile K, aarch64 compact loop ([0034627](https://github.com/loadingalias/rscrypto/commit/0034627c60dd55a1bd98a1e65be7e7b9b07a5184))
- CRC32 aarch64 dispatch bypass, SHA-3 sponge output extraction; XXH3 small-input codegen overhaul — cold dispatch + typed mix16_b ([09f0dc2](https://github.com/loadingalias/rscrypto/commit/09f0dc2f44b055c1fb88ca5a97713415cd996536))
- SHA-512 single-block rotation schedule + vector K addition ([a291d28](https://github.com/loadingalias/rscrypto/commit/a291d28a7fcdb6a8538d09ceb7c23cc5c61fbed0))
- SHA-512 Zen5 dispatch fix, Keccak θ rewrite, RapidHash codegen tune ([8a23f40](https://github.com/loadingalias/rscrypto/commit/8a23f40b5ad8313de53a80dc169ed0a2d575a3c8))
- RapidHashFast inner core, HKDF midstate cache, Ed25519 field/point optimizations RapidHash: - RapidHashFast64/128 now uses a dedicated inner-algorithm core instead   of   V3-no-avalanche. Size-tuned dispatch: 3-stream (49-400B), 7-stream   (>400B),   cold-path separation for codegen quality. Oracle:   rapidhash::fast::RapidHasher. - RapidHash64/128 (standard V3) unchanged.   Auth: - HKDF-SHA256: cache HmacSha256 keyed with PRK at extract time. expand()   now   resets (1 memcpy) instead of re-creating (2 SHA-256 compressions) per   chunk. - Ed25519 field: dedicated squaring — 15 wide muls vs 25 (40% fewer). - Ed25519 point: dedicated dbl-2008-hwcd doubling — 4 sq + 4 mul, no D2   multiply. - Ed25519 point: precomputed 16-entry basepoint table with 4-bit   windowed   scalar mul — adds drop from ~128 to ~60. - Ed25519 verify: Straus/Shamir interleaved [s]B + [-h]A in one 256-bit   scan,   halving doublings from 512 to 256. ([89dbfc2](https://github.com/loadingalias/rscrypto/commit/89dbfc2c29dd27feb37736e155f2d57996560953))
- bypass dispatch overhead for small fast-hash inputs ([e12085c](https://github.com/loadingalias/rscrypto/commit/e12085c77afc42cceb70d63102a3c1a2ee7ea95a))
- cascade AVX-512 sub-degree tails to AVX2 for BLAKE3 hash_many and parent compression ([17f47d6](https://github.com/loadingalias/rscrypto/commit/17f47d6a52df857a03bce7a5473965f70b962d66))
- fix BLAKE3 XOF ~250ns Drop overhead, upgrade zeroize to word-sized writes feat: add SHA-NI/SHA2 CE hardware acceleration for SHA-256/SHA-224 ([9f76def](https://github.com/loadingalias/rscrypto/commit/9f76defed3d793386ae9d058df3bdd7e9a08945d))
- extend direct-assembly compress to update_general() for 128B-512B streaming ([c509ce3](https://github.com/loadingalias/rscrypto/commit/c509ce31b948e558b513a5191fdb15d5ac9770e3))
- flatten BLAKE3 streaming compress call chain, eliminate overflow branch ([f7a551a](https://github.com/loadingalias/rscrypto/commit/f7a551ab14a8f9e1bdfe6006b75b20be21bbb427))
- remove pending_chunk_cv gate from BLAKE3 streaming hot path ([b168ad0](https://github.com/loadingalias/rscrypto/commit/b168ad0f3ac24f60b4c1531e15c918dd5d7ca4ae))
- optimize BLAKE3 streaming control path for sub-chunk updates ([331bac3](https://github.com/loadingalias/rscrypto/commit/331bac3b35106b94c3682b1c5dee0a97f4f53d61))
- remove cold/inline-never from blake3 streaming and XOF hot paths fix: fixing the SSSE3 issue in the weekly.yaml run/testing ([9eeb7bc](https://github.com/loadingalias/rscrypto/commit/9eeb7bcc7b0265f4f7bc90d8d4ca0b1235ae0e91))
- add XOF fast path and remove hot-path cold/inline barriers; ([76aa96b](https://github.com/loadingalias/rscrypto/commit/76aa96b791869dc774ec647b5dd565f875e78a65))
- remove legacy tune pipeline and add Blake3 CI gap gates ([0549cee](https://github.com/loadingalias/rscrypto/commit/0549cee9df15942470d045a72cbecd40e05b4391))

### 💄 Styling

- rustfmt ([1c28ea2](https://github.com/loadingalias/rscrypto/commit/1c28ea2c1142013360ea446ad3bc5a6c04c29543))



All notable changes to rscrypto will be documented in this file.
