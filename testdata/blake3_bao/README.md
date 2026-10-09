# Bao combined-stream vectors

Unmodified test vectors from Bao 0.13.1:

- Source: https://raw.githubusercontent.com/oconnor663/bao/0.13.1/tests/test_vectors.json
- License: CC0-1.0 or Apache-2.0 (upstream Bao license choice).
- SHA-256: `e819b403a62e5fef51764d00783c3f750a78f8cb72e646f8a9add0c521333de3`.

`tests/blake3_bao.rs` uses the combined `encode` cases. Input words are
incrementing little-endian u32 values starting at one. The locked upstream
encoder supplies bytes; the official length, root and encoding hash independently
anchor those bytes before the production decoder runs. Other sections remain
unmodified for provenance and do not imply support for outboard/slice/seeking.
