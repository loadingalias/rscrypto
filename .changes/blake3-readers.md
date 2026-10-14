---
"rscrypto" = "minor"
---

Add `Blake3::update_reader` and `Blake3Subtree::update_reader` under `std`:
buffered BLAKE3 input reading, with bounded readers for caller-scheduled subtrees. Preserve successfully read input on I/O errors and
clear the owned input buffer before deallocation.

With `parallel`, plain readers on little-endian Linux AArch64 can hash complete
aligned buffers through the current Rayon pool.
