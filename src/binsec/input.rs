/// Read the symbolic bytes or limbs of a BINSEC input in address order.
///
/// # Safety
///
/// `source` must point to an initialized array of volatile-compatible scalars.
#[inline(always)]
pub(crate) unsafe fn array_from_global<T: Copy, const N: usize>(source: *const [T; N]) -> [T; N] {
  core::array::from_fn(|index| {
    // SAFETY: The caller supplies N initialized scalars. The array bounds the index.
    unsafe { source.cast::<T>().add(index).read_volatile() }
  })
}
