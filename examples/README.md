# Examples

Runnable examples for the public API.

| Example | Command | Purpose |
|---------|---------|---------|
| `basic` | `cargo run --example basic --features full` | Core API patterns across primitive families |
| `password_hashing` | `cargo run --example password_hashing --features password-hashing,getrandom` | PHC strings with bounded verification |
| `introspect` | `cargo run --example introspect --features checksums,hashes,diag` | Runtime dispatch inspection |
| `parallel` | `cargo run --example parallel --features checksums` | CRC chunk combining |

These examples are intentionally small. Use the item-level docs on docs.rs for
API details.
