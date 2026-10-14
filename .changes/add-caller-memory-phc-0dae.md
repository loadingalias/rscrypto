---
"rscrypto" = "patch"
---

Add `Argon2idPassword::verify_password_with_memory`, `Argon2idPassword::verify_password_with_context_and_memory`, and `ScryptPassword::verify_password_with_memory`. They verify PHC records in a reusable block slice that the caller owns, from any allocator, with the same record limits, opaque failures, and workspace cleanup.
