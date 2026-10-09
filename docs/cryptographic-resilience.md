# Choosing cryptography under changing assumptions

Choose a protocol profile by the property it must protect, the assumptions it relies on,
and the evidence available for the implementation you will deploy.
Evaluate authentication and confidentiality separately.
Faster execution, larger parameters, and additional algorithms answer different questions.

The [primitive inventory](../README.md#primitives-and-features) and
[feature graph](../Cargo.toml) identify implemented capabilities.
The [threat model](../THREAT_MODEL.md) separates library and caller responsibilities.
This guide does not establish a new security or qualification claim.

## Identify the assumption you need to change

| Family | Role and assumption | Consequence for selection |
| --- | --- | --- |
| ECDSA, Ed25519, and elliptic-curve key agreement | Signatures or key agreement based on elliptic-curve problems. | Changing curves or increasing their size does not remove dependence on this family of problems. |
| RSA | Signatures and encryption whose security relies on RSA-related number-theoretic assumptions. | Larger keys preserve those assumptions. |
| ML-KEM and ML-DSA | Key establishment and signatures based on module-lattice problems. | They offer post-quantum alternatives to classical public-key schemes, with implementation and protocol requirements of their own. |
| SLH-DSA | Signatures based on hash-function properties. | It offers a different foundation from curves and lattices. It is planned for rscrypto and is not currently implemented. |
| Symmetric encryption, MACs, and cryptographic hashes | Protect data using their construction-specific assumptions, keys, and usage bounds. | They remain necessary after public-key migration; they do not by themselves replace public-key authentication or key establishment. |

[FIPS 203](https://csrc.nist.gov/pubs/fips/203/final),
[FIPS 204](https://csrc.nist.gov/pubs/fips/204/final), and
[FIPS 205](https://csrc.nist.gov/pubs/fips/205/final) define ML-KEM, ML-DSA, and SLH-DSA respectively.
SLH-DSA supports signatures, not general-purpose encryption or key exchange.
Supporting SHA-2 or SHAKE does not itself supply an SLH-DSA implementation.

Select parameters and prehashes together with the protocol's security target.
For example, the prehash can limit the collision strength of an
[ML-DSA profile](mldsa.md#prehash-mode).
An algorithm's post-quantum analysis does not establish resistance to every future mathematical
advance. No current rscrypto evidence supports an unconditional "AI-proof" claim.

## Plan the migration at the protocol boundary

Record the algorithms, parameter sets, specification revisions, credential formats, trust
anchors, and required versus optional capabilities before migrating.
Distribute replacement trust anchors through a channel that still provides the required
authentication. Document how existing keys, signatures, and encrypted data remain usable.
Replacing an algorithm does not retroactively protect already exposed secrets.

Use the composition specified for the protocol. For a dual-signature profile that requires
both components, require both verifications and the specified binding of keys, algorithm
identifiers, contexts, and messages. Accepting either component does not provide that property.
For hybrid key establishment, use its specified combiner, validation, and authenticated
negotiation. A signature combiner and a KEM combiner solve different problems.
Review drafts against the exact revision selected for the integration; see the
[IETF hybrid-signature work](https://www.ietf.org/archive/id/draft-prabel-cfrg-suf-hybrid-sigs-02.html)
for an example of explicit key and context binding, not an rscrypto API contract.

Keep required capabilities fail-closed under negotiation and credential processing.
Test rotation, rollback, old records, malformed inputs, entropy failure, and incompatible peers.
Key custody, nonce policy, certificate validation, and recovery remain caller responsibilities.

## Compare the work you will deploy

Measure the complete operation, including applicable import, validation, randomness,
preparation, output handling, and cleanup. Separate compact and prepared keys.
Report latency distributions, throughput, stack, heap, allocations, artifact size, and bytes
transmitted or stored. Match parameters and security properties between comparison rows.
Use the [benchmarking guide](benchmarking.md) and [retained results](../benchmark_results/OVERVIEW.md)
for the measurement contract and its current evidence.

Hash-based signatures make short hash inputs, independent chains, tree traversal, and signature
size important. Bulk hash throughput alone does not measure those operations.
For a proof-system integration, also measure proving time, verifier cost, proof size, and memory
through the actual integration. Native hash speed does not determine proof-system cost.

SLH-DSA is stateless. Other hash-based signatures, including LMS and XMSS, require careful
persistent-state management. Backups, concurrent signers, and rollback can violate their
one-time-key requirements; see [NIST's stateful-signature guidance](https://www.nist.gov/news-events/news/2020/10/recommendation-stateful-hash-based-signature-schemes-nist-sp-800-208).

## Inspect the assurance boundary

For the selected operation, record the source revision, compiler, features, target, backend,
and retained evidence. Review [test coverage](test-vector-coverage.md),
[constant-time boundaries](constant-time.md), [secret lifecycle](secret-lifecycle.md),
and [platform qualification](platforms.md) separately. Configured checks are requirements;
only the corresponding retained results establish what passed.

Formal verification establishes specified properties under its model and assumptions.
Functional correctness, source-level secret independence, compiled timing, and cleanup are
different properties. A proof of one does not establish the others or prove that the underlying
cryptographic assumption will withstand future attacks.

rscrypto has not had a third-party security audit and does not claim whole-crate formal
verification. Read the [assurance statement](../README.md#assurance) and
[compliance boundary](compliance.md) when evaluating an integration.
