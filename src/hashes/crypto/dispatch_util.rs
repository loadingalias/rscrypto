//! Internal helpers for tuned size-class dispatch.
//!
//! BLAKE3 caches four size classes `{xs, s, m, l}` with three boundaries.

#[derive(Clone, Copy)]
pub(crate) struct SizeClassDispatch<T: Copy> {
  pub(crate) boundaries: [usize; 3],
  pub(crate) xs: T,
  pub(crate) s: T,
  pub(crate) m: T,
  pub(crate) l: T,
}

impl<T: Copy> SizeClassDispatch<T> {
  #[inline]
  #[must_use]
  pub(crate) fn select(self, len_hint: usize) -> T {
    let [xs_max, s_max, m_max] = self.boundaries;
    if len_hint <= xs_max {
      self.xs
    } else if len_hint <= s_max {
      self.s
    } else if len_hint <= m_max {
      self.m
    } else {
      self.l
    }
  }
}
