#[inline]
#[must_use]
pub(crate) fn is_crc32_algorithm(name: &str) -> bool {
  matches!(name, "crc32-ieee" | "crc32c")
}
