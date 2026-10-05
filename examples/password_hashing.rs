//! Hash and verify a password with the bounded Argon2id policy.

use std::alloc::System;

use rscrypto::{Argon2Block, Argon2Params, Argon2idPassword};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let password = b"correct horse battery staple";

  let params = Argon2Params::default();
  let policy = Argon2idPassword::new(params)?;
  let record = policy.hash_password(password)?;

  // Allocate once with a caller-selected allocator, then reuse the workspace.
  // With separate verification limits, size from their profile instead.
  let mut memory = Vec::new_in(System);
  let blocks = usize::try_from(params.memory_blocks())?;
  memory.try_reserve_exact(blocks)?;
  memory.resize(blocks, Argon2Block::ZERO);

  policy.verify_password_with_memory(password, &record, &mut memory)?;
  if policy
    .verify_password_with_memory(b"wrong password", &record, &mut memory)
    .is_ok()
  {
    return Err(std::io::Error::other("Argon2id accepted the wrong password").into());
  }

  println!("{record}");
  Ok(())
}
