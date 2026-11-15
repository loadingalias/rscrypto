---
"rscrypto" = "patch"
---

Accept valid Ed25519 signatures under public keys with a small-order component. Verification
computed [S]B + [L - k]A, which differs from [S]B - [k]A when A has a small-order component, so it
rejected every such signature. It now checks [S]B = R + [k]A on the whole group on every backend.
It still rejects small-order and non-canonical A and R, S of at least L, and signatures that only
the cofactored check accepts.
