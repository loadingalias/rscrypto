use core::ptr;

#[unsafe(no_mangle)]
#[inline(never)]
/// C-compatible volatile byte fill for the analysis harness.
///
/// # Safety
///
/// `dst` must be valid and writable for `len` bytes.
pub(crate) unsafe extern "C" fn memset(dst: *mut u8, value: i32, len: usize) -> *mut u8 {
  let mut offset = 0usize;
  while offset < len {
    // SAFETY: C callers require `dst..dst+len` to be writable. Volatile writes
    // keep zeroization visible to BINSEC and prevent this shim from becoming a
    // recursive compiler intrinsic.
    unsafe { ptr::write_volatile(dst.add(offset), value.to_le_bytes()[0]) };
    offset = offset.strict_add(1);
  }
  dst
}

#[unsafe(no_mangle)]
#[inline(never)]
/// C-compatible non-overlapping byte copy for the analysis harness.
///
/// # Safety
///
/// `src` must be readable and `dst` writable for `len` bytes, and the two
/// ranges must not overlap.
pub(crate) unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
  // SAFETY: `memcpy` has the same validity contract as `memmove`, with the
  // additional non-overlap precondition. The `memmove` implementation is valid
  // for both cases and keeps the harness small.
  unsafe { memmove(dst, src, len) }
}

#[unsafe(no_mangle)]
#[inline(never)]
/// C-compatible overlapping byte copy for the analysis harness.
///
/// # Safety
///
/// `src` must be readable and `dst` writable for `len` bytes. Both pointers
/// must satisfy Rust's pointer provenance requirements for those ranges.
pub(crate) unsafe extern "C" fn memmove(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
  if (dst as usize) <= (src as usize) {
    let mut offset = 0usize;
    while offset < len {
      // SAFETY: C callers require source and destination ranges to be valid.
      let byte = unsafe { ptr::read(src.add(offset)) };
      // SAFETY: C callers require source and destination ranges to be valid.
      unsafe { ptr::write(dst.add(offset), byte) };
      offset = offset.strict_add(1);
    }
  } else {
    let mut remaining = len;
    while remaining != 0 {
      remaining = remaining.strict_sub(1);
      // SAFETY: C callers require source and destination ranges to be valid.
      let byte = unsafe { ptr::read(src.add(remaining)) };
      // SAFETY: C callers require source and destination ranges to be valid.
      unsafe { ptr::write(dst.add(remaining), byte) };
    }
  }
  dst
}

#[unsafe(no_mangle)]
#[inline(never)]
/// C-compatible byte comparison for the analysis harness.
///
/// # Safety
///
/// `lhs` and `rhs` must each be readable for `len` bytes.
pub(crate) unsafe extern "C" fn bcmp(lhs: *const u8, rhs: *const u8, len: usize) -> i32 {
  let mut acc = 0u8;
  let mut offset = 0usize;
  while offset < len {
    // SAFETY: C callers require both ranges to be readable for `len` bytes.
    let l = unsafe { ptr::read(lhs.add(offset)) };
    // SAFETY: C callers require both ranges to be readable for `len` bytes.
    let r = unsafe { ptr::read(rhs.add(offset)) };
    acc |= l ^ r;
    offset = offset.strict_add(1);
  }
  i32::from(acc)
}

/// The page that the `mmap` shim returns, page-aligned and zero-filled.
#[cfg(target_os = "linux")]
#[repr(C, align(4096))]
struct ProcessOriginPage([u8; 4096]);

#[cfg(target_os = "linux")]
static mut CT_BINSEC_PROCESS_ORIGIN_PAGE: ProcessOriginPage = ProcessOriginPage([0; 4096]);

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
/// Harness-only `mmap` for the analysis harness, which BINSEC never runs
/// natively. No production build defines this symbol.
///
/// BINSEC loads the initial memory from the file, so the first
/// `rscrypto::platform::ProcessOrigin::current()` in an analyzed leaf runs
/// the process setup. The setup maps one page and advises it. This shim and
/// [`madvise`] model the Linux production path: `mmap` returns one
/// page-aligned, zero-filled page, the probe with invalid advice fails, and
/// `MADV_WIPEONFORK` succeeds. So the analyzed path is the page-mode path, and
/// it stays inside the binary, where BINSEC can follow it. The prototype is
/// `void *mmap(void *, size_t, int, int, int, off_t)`.
///
/// # Safety
///
/// Callers must use the returned page only as the one private page that the
/// process setup asks for.
pub(crate) unsafe extern "C" fn mmap(
  _addr: *mut core::ffi::c_void,
  _len: usize,
  _prot: i32,
  _flags: i32,
  _fd: i32,
  _offset: i64,
) -> *mut core::ffi::c_void {
  core::ptr::addr_of_mut!(CT_BINSEC_PROCESS_ORIGIN_PAGE).cast()
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
/// Harness-only `madvise` for the analysis harness: it accepts
/// `MADV_WIPEONFORK` (18) and refuses every other advice, as Linux does for the
/// invalid advice -1. The prototype is `int madvise(void *, size_t, int)`. See
/// [`mmap`].
///
/// # Safety
///
/// It reads and writes no memory, so every call is defined.
pub(crate) unsafe extern "C" fn madvise(_addr: *mut core::ffi::c_void, _len: usize, advice: i32) -> i32 {
  if advice == 18 { 0 } else { -1 }
}
