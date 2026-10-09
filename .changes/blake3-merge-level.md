---
"rscrypto" = "minor"
---

Add allocation-free `Blake3Tree::merge_level` for batched non-root parent merges
in every BLAKE3 mode. Validate all child pairs before changing output and clear
the bounded child and parent scratch on return or unwind.
