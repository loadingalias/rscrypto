---
"rscrypto" = "patch"
---

Clear native AES-GCM authentication state at its Rust ABI boundary and erase the Linux x86-64
GHASH power workspaces before releasing their stack frames. Native saved-register slots are
cleared after restoring the caller's registers. This does not claim erasure of every register
or compiler-created copy.
