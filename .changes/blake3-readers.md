---
"rscrypto" = "minor"
---

Add buffered BLAKE3 input reading under `std`, including bounded readers for
caller-scheduled subtrees. Preserve successfully read input on I/O errors and
clear the owned input buffer before deallocation.

With `parallel`, plain readers on little-endian Linux AArch64 can hash complete
aligned buffers through the current Rayon pool.
