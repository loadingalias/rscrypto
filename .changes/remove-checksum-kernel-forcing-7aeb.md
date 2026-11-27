---
"rscrypto" = "minor"
---

CRC and Adler-32 now select their kernels from the process CPU capabilities.
Remove `CrcForce`, `CrcConfig`, `Adler32Force`, `Adler32Config`, and their `config()` accessors.
Remove the `RSCRYPTO_CRC*_FORCE` and `RSCRYPTO_ADLER32_FORCE` environment settings.
Move read-only reports from `checksum::diag` to `checksum::introspect`.
The CRC reports drop `SelectionReason` and the `reason` and `effective_force` fields.
Use `kernel_for::<T>(len)` to inspect the selected kernel.
To restrict CPU capabilities, call `platform::expert::try_set_override` before detection in a fresh process.

Remove the `diag` Cargo feature. Hash and AEAD introspection use their primitive features.
Use `auth::argon2::active_kernel()` to report the Argon2 compression kernel.
