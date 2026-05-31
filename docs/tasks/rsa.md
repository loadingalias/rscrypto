# RSA End-To-End Closure Ledger

Single task: close RSA so the implementation is complete, secure, sound, and
stable. Completed work is deleted from this file. Do not keep historical status
notes, victory claims, or already-closed implementation tasks here.

The current RSA surface is end-to-end: public verify, public encrypt, private
sign, private decrypt, strict public/private key parsing, PKCS#1/PKCS#8 export,
and key generation all exist. The remaining blockers are proof and release
evidence, not feature scaffolding.

RSA security comes before RSA speed. A fast RSA implementation that leaks
secret-dependent timing, exposes padding oracles, mishandles private material,
or relies on optional evidence is not acceptable.

Current public-operation performance evidence is positive on the active host
targets: focused RSA-2048/3072/4096 public operation, PSS verify, and
PKCS#1 v1.5 verify benches are ahead of AWS-LC, ring, and OpenSSL on the local
macOS M1 Pro host and the Linux x86_64 Sapphire Rapids runner. This is not a
release claim for private RSA performance, side-channel behavior, or broader
platform coverage.

## Security And Correctness Proof

- [ ] Review the current RSA implementation for usability, security,
      stability, and integration into the codebase. Remove or simplify every
      script, fixture, diagnostic hook, ASM wrapper, benchmark helper, or
      abstraction that is not justified by correctness, security, portability,
      or measured performance evidence.
- [ ] Run the manual RSA workflow for the release commit and retain artifacts:
      Linux x64 `just test-miri --rsa`, Linux x64 `just test-rsa-leakage`, and
      Linux arm64 `just test-rsa-leakage`.
- [ ] Keep RSA-specific scripts, fixtures, fuzz seeds, scorecard tools, and
      diagnostic hooks minimal. Each artifact must be justified by security
      evidence, side-channel evidence, oracle coverage, or benchmark evidence
      that the normal `just check-all`, `just test`, and `just bench rsa` lanes
      cannot provide.

## Performance Evidence

- [ ] Preserve the current public RSA lead while simplifying the implementation.
      Any cleanup that touches public-operation arithmetic, byte/limb
      conversion, ASM dispatch, or scratch layout must re-run focused
      RSA-2048/3072/4096 benches against AWS-LC, ring, and OpenSSL on macOS
      AArch64 and Linux x86_64.
- [ ] After the review and security proof items above are closed, tune and
      benchmark RSA import, private operations, blinding, CRT recombination,
      key generation, allocation behavior, and broader cross-target
      performance with `just bench rsa` and RSA scorecards.
