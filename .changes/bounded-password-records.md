---
"rscrypto" = "minor"
---

Argon2 and scrypt password verification now reject noncanonical or over-budget
PHC records before decoding, allocating, or running a KDF. The pre-1.0 API is
split into verifier-owned `Argon2idPassword` and `ScryptPassword` records and
raw `derive` operations with valid-by-construction parameters; legacy public
PHC parsing, unbounded verification, builder, version-selection, and
caller-supplied-salt password helpers have been removed.
