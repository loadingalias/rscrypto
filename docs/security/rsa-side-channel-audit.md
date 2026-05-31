# RSA Side-Channel Audit

This document is the closure record for RSA private-operation side-channel
evidence. It is intentionally narrow: it covers timing/control-flow leakage in
private signing, private decryption, CRT recombination, blinding, Montgomery
arithmetic, padding failure opacity, and the diagnostic evidence gates that
must pass before a release claim.

## Release Claim Boundary

RSA is releasable only when all of the following are true for the release
commit:

1. `just test --all` passes.
2. `just test-fuzz --all` passes.
3. `just test-miri --rsa` passes on the Linux x64 RSA workflow lane.
4. `just test-rsa-leakage` passes on Linux x64 and Linux arm64 RSA workflow
   lanes.
5. The RSA workflow artifacts are retained with the release evidence.

The leakage gate is first-order timing evidence. It is not a formal proof of
constant time, and it is not a substitute for this audit. It is a regression
gate for large observable timing separation on the hosted runners.

## Constant-Time Design

Private exponentiation uses fixed-width, fixed-window Montgomery arithmetic.
The exponent scan covers the full modulus byte width, including leading-zero
padding for shorter CRT exponents. Window selection scans every table entry and
uses masks instead of secret-indexed table loads.

CRT recombination computes both prime-side exponentiations, reduces both
residues, multiplies the masked difference by `qInv`, and recombines at fixed
output width. Caller-visible output is copied only after the public fault check
passes.

Montgomery final reduction, modular add/double, and reduction import now use
branchless conditional subtraction/addition. Carry propagation in private
product paths walks the full remaining limb tail before checking overflow.

OAEP and RSAES-PKCS1-v1_5 same-width decryption failures clear caller output.
PSS and RSASSA-PKCS1-v1_5 signing failures clear caller output. Padding
decoders scan the full encoded block before returning the opaque public error.

## Findings Resolved In Code

- Secret-bit modular import no longer branches on `(byte >> bit) & 1`; it
  always executes the add path with a masked carry bit.
- Montgomery CIOS/product/comba final subtraction no longer branches on high
  limbs or `cmp_limbs`; subtraction is attempted unconditionally and restored
  with a mask when not needed.
- Modular add/double carry handling no longer exits early.
- Private product carry propagation no longer exits on `carry == 0`; it walks
  the full available tail and checks overflow after the fixed loop.
- Decrement helpers used by blinding inverse and key arithmetic no longer exit
  early after the borrow clears.
- The RSA leakage test has a `diag,getrandom`-only blinding inverse hook so the
  inverse derivation path is measured directly rather than inferred from a
  larger operation.

## Accepted Non-Constant Behavior

These paths are not part of online private signing/decryption constant-time
claims:

- Public-key parsing, public exponentiation, public representative range
  rejection, and public-operation backend dispatch branch on public data.
- DER private-key import validation may branch while rejecting malformed key
  material before a key is accepted. This is an offline key-loading boundary,
  not a remote signing/decryption oracle.
- Key generation branches on random candidate primality, rejection sampling,
  Miller-Rabin outcomes, and generated prime relationships. This leaks
  generation randomness and candidate structure during local key generation; it
  is not part of the deployed private-operation timing claim.
- OS-backed blinding-factor generation uses rejection sampling. Rejection
  timing may depend on fresh blinding randomness. The leakage gate still
  measures an end-to-end OS-blinded signing path to catch large regressions in
  the deployed API, but the stricter inverse-derivation measurement uses
  caller-supplied factors.
- Final public fault-check equality branches only after the private result has
  been unblinded and re-encrypted. Failure is reported as an opaque private-op
  error and caller output is cleared.

## Leakage Gate

`scripts/test/test-rsa-leakage.sh` runs an ignored release test under
`--release` with `rsa,diag,getrandom`. Each case collects fixed-vs-random timing
samples in randomized order and applies a Welch t-test. The default threshold is
`|t| < 8.0`; release evidence may lower it, but must not raise it after seeing
results.

Measured cases:

- RSA-2048 RSASSA-PKCS1-v1_5 signing with fixed caller-supplied blinding.
- RSA-2048 RSASSA-PSS signing with fixed salt and fixed caller-supplied
  blinding.
- RSA-2048 RSAES-OAEP decryption with fixed caller-supplied blinding.
- RSA-2048 RSAES-PKCS1-v1_5 decryption with fixed caller-supplied blinding.
- RSA-2048 blinding-factor inverse derivation.
- RSA-2048 RSASSA-PKCS1-v1_5 signing through the OS-blinded public API.

The gate is manual in CI because noisy shared runners are not laboratory
equipment. Passing on both Linux x64 and Linux arm64 is required release
evidence; failure or inconclusive reruns mean the RSA release claim remains
open.
