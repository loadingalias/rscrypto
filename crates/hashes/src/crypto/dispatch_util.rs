//! Internal helpers for tuned size-class dispatch.
//!
//! Many algorithms in this crate use the same 4-way size-class model:
//! `{xs, s, m, l}` with three boundaries. This module centralizes the
//! selection logic so hashers can cache dispatch tables locally without
//! duplicating the boundary checks.

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

#[inline]
#[must_use]
pub(crate) fn len_hint_from_u64(v: u64) -> usize {
  if (v as usize) as u64 == v {
    v as usize
  } else {
    usize::MAX
  }
}

#[inline]
#[must_use]
pub(crate) fn len_hint_from_u128(v: u128) -> usize {
  if (v as usize) as u128 == v {
    v as usize
  } else {
    usize::MAX
  }
}
