//! The process that created a value, for owners of state that a fork must not copy.

use core::fmt;

/// The process that created a value.
///
/// A fork copies the memory of a process. It therefore also copies each value
/// that must stay unique to one process, such as a nonce counter or a random
/// generator. [`ProcessOrigin::current`] records the calling process. In a copy
/// that a fork made, [`ProcessOrigin::is_current`] returns `false`. An owner of
/// unique state records its origin when it is created and refuses to continue
/// when the origin is not current.
///
/// ```
/// use rscrypto::platform::ProcessOrigin;
///
/// let origin = ProcessOrigin::current();
/// assert!(origin.is_current());
/// ```
///
/// # Detection
///
/// - Linux and Android on x86-64, AArch64, 64-bit POWER, 64-bit RISC-V, and
///   IBM Z use a page that the kernel gives as zeros to a fork child
///   (`MADV_WIPEONFORK`, Linux 4.14). The check is one function call with one
///   atomic load. If the kernel refuses the page, as before Linux 4.14, under
///   an emulator that ignores the request, or under a seccomp filter, the
///   process compares process IDs.
/// - Apple targets and other Unix targets compare process IDs.
/// - Targets that cannot fork, such as Windows, WebAssembly, and bare-metal
///   targets, always report the current process.
///
/// # Limits
///
/// - Without the `std` feature, no fork is detected, and every origin reports
///   the current process.
/// - A process-ID comparison accepts a copy in a descendant that receives the
///   recycled process ID of a dead ancestor, or the same ID in a new PID
///   namespace. The page detection does not depend on process IDs.
/// - A copied memory image, such as a restored virtual-machine snapshot, is
///   not detected.
#[derive(Clone, Copy)]
pub struct ProcessOrigin(imp::Origin);

impl ProcessOrigin {
  /// The origin of a value that the calling process creates now.
  #[inline]
  #[must_use]
  pub fn current() -> Self {
    Self(imp::Origin::current())
  }

  /// Whether the calling process created this origin.
  ///
  /// A fork between [`current`](Self::current) and this call makes it
  /// `false` in the child. See [Limits](Self#limits).
  #[inline]
  #[must_use]
  pub fn is_current(&self) -> bool {
    self.0.is_current()
  }
}

impl fmt::Debug for ProcessOrigin {
  // The page address would reveal the address-space layout, so `Debug` shows no field.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ProcessOrigin").finish_non_exhaustive()
  }
}

// The detection mechanism of each target: the wipe-on-fork page, a process-ID comparison, or none.
core::cfg_select! {
  all(
    feature = "std",
    not(miri),
    any(target_os = "linux", target_os = "android"),
    target_pointer_width = "64",
    any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "riscv64",
      target_arch = "s390x"
    )
  ) => {
    mod imp {
      use core::ffi::{c_int, c_void};
      use core::ptr;
      use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

      // Linux v6.12 uapi. `include/uapi/asm-generic/mman-common.h` defines PROT_READ 0x1,
      // PROT_WRITE 0x2, MAP_ANONYMOUS 0x20, and MADV_WIPEONFORK 18. `include/uapi/linux/mman.h`
      // defines MAP_PRIVATE 0x02. The x86 and arm64 `asm/mman.h` include `asm-generic/mman.h`, the
      // powerpc `asm/mman.h` includes `asm-generic/mman-common.h` and overrides none of these names,
      // and riscv and s390 have no `asm/mman.h`, so they get the mandatory generic header
      // (`include/uapi/asm-generic/Kbuild`). MAP_FAILED is `(void *) -1` in glibc 2.39, musl 1.2.5,
      // and bionic.
      const PROT_READ: c_int = 0x1;
      const PROT_WRITE: c_int = 0x2;
      const MAP_PRIVATE: c_int = 0x02;
      const MAP_ANONYMOUS: c_int = 0x20;
      const MADV_WIPEONFORK: c_int = 18;
      /// The kernel rounds the length up to one page (`mm/mmap.c` and `mm/madvise.c`, `PAGE_ALIGN`).
      const LEN: usize = size_of::<AtomicU64>();

      // glibc 2.39 `misc/sys/mman.h`, musl 1.2.5 `include/sys/mman.h`, and bionic
      // `libc/include/sys/mman.h` declare `void *mmap(void *, size_t, int, int, int, off_t)` and
      // `int madvise(void *, size_t, int)`. `off_t` is 64 bits on every 64-bit target this module
      // admits.
      unsafe extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
        fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
      }

      /// A sentinel for "compare process IDs": address 8, the alignment of `AtomicU64`. `mmap`
      /// without `MAP_FIXED` returns only page-aligned addresses, and address 8 is not page-aligned,
      /// so no page is ever this sentinel.
      const PID_MODE: *mut AtomicU64 = ptr::dangling_mut();

      /// Null until this process image completes its setup. Then the wipe-on-fork page, or
      /// [`PID_MODE`]. A fork copies the value, so a child of a process with a page keeps the page,
      /// which the kernel gives the child as zeros.
      static STATE: AtomicPtr<AtomicU64> = AtomicPtr::new(ptr::null_mut());
      /// The latest generation that this lineage took. A fork copies it, so a child continues above
      /// every generation of its ancestors.
      static LAST: AtomicU64 = AtomicU64::new(0);

      #[derive(Clone, Copy)]
      pub(super) struct Origin {
        /// The page whose value was `value` when this origin was recorded, or `None` to compare
        /// process IDs.
        page: Option<&'static AtomicU64>,
        value: u64,
      }

      impl Origin {
        #[inline]
        pub(super) fn current() -> Self {
          let Some(page) = page() else {
            return Self {
              page: None,
              value: u64::from(std::process::id()),
            };
          };
          let generation = page.load(Ordering::Relaxed);
          let value = if generation != 0 {
            generation
          } else {
            new_generation(page)
          };
          Self {
            page: Some(page),
            value,
          }
        }

        // Out of line on purpose. Inlined, this check made a caller such as a `NonceCounter` seal so
        // large that LLVM no longer inlined the caller into its loop: a 16-byte AES-256-GCM seal took
        // 13% longer on x86-64. As one call, it takes 1.5% longer.
        #[inline(never)]
        pub(super) fn is_current(&self) -> bool {
          match self.page {
            // The kernel gives the page as zeros to a fork child, and every later generation in that
            // child is higher than `value` (see `new_generation`).
            Some(page) => page.load(Ordering::Relaxed) == self.value,
            None => u64::from(std::process::id()) == self.value,
          }
        }
      }

      /// The generation of a fork child whose page the kernel gave as zeros.
      #[cold]
      #[inline(never)]
      fn new_generation(page: &AtomicU64) -> u64 {
        // Take the generation from `LAST` before the CAS publishes it. A fork between the two steps
        // then copies a `LAST` that already holds it, so the grandchild takes a higher value.
        let next = LAST.fetch_add(1, Ordering::Relaxed).strict_add(1);
        // Racing threads of this child each take a value. The first CAS wins, and the others adopt the
        // winner's value, so every origin of this process records one generation.
        match page.compare_exchange(0, next, Ordering::Relaxed, Ordering::Relaxed) {
          Ok(_) => next,
          Err(winner) => winner,
        }
      }

      /// The wipe-on-fork page of this process, or `None` to compare process IDs.
      #[inline]
      fn page() -> Option<&'static AtomicU64> {
        let state = STATE.load(Ordering::Acquire);
        let state = if state.is_null() { setup() } else { state };
        if state == PID_MODE {
          return None;
        }
        // SAFETY: Every non-null `STATE` other than `PID_MODE` came from `setup`, which publishes only
        // a non-null page that `map_wipe_on_fork_page` mapped readable and writable for the rest of the
        // process, page-aligned (so aligned for `AtomicU64`), and zero-filled before its first
        // generation (all bit patterns are valid `AtomicU64` values). The page is never unmapped, so
        // the reference is valid for `'static`. Every access is atomic, so shared references may mutate
        // it. A fork child keeps the mapping at the same address.
        Some(unsafe { &*state })
      }

      /// Map and advise a page, then publish it, or publish [`PID_MODE`] if the kernel refuses.
      ///
      /// There is no "setup in progress" state. Each racing thread maps its own page, and the first
      /// CAS publishes one result. A losing thread adopts the winner and leaks its own page: one page,
      /// once, in a rare race. A fork during the setup copies a null `STATE`, because nothing is
      /// published before the CAS. The child then runs the setup itself, and it never waits for a
      /// thread that the fork did not copy.
      #[cold]
      #[inline(never)]
      fn setup() -> *mut AtomicU64 {
        let mine = match map_wipe_on_fork_page() {
          Some(page) => {
            let first = LAST.fetch_add(1, Ordering::Relaxed).strict_add(1);
            page.store(first, Ordering::Relaxed);
            ptr::from_ref(page).cast_mut()
          }
          None => PID_MODE,
        };
        // Release publishes the first generation with the page.
        match STATE.compare_exchange(ptr::null_mut(), mine, Ordering::AcqRel, Ordering::Acquire) {
          Ok(_) => mine,
          Err(winner) => winner,
        }
      }

      /// One private anonymous page with `MADV_WIPEONFORK`, or `None` if the kernel refuses.
      fn map_wipe_on_fork_page() -> Option<&'static AtomicU64> {
        // SAFETY: `mmap` with a null hint and no `MAP_FIXED` cannot replace an existing mapping. A
        // private anonymous mapping takes no file descriptor (`-1`) and offset 0. The kernel returns a
        // new page-aligned, zero-filled, readable and writable mapping, or `MAP_FAILED`.
        let page = unsafe {
          mmap(
            ptr::null_mut(),
            LEN,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
          )
        };
        // Refuse a null page too. Only a root-lowered `vm.mmap_min_addr` permits one. A null page can
        // back no reference, and publishing it would leave `STATE` unset.
        if page.is_null() || page.addr() == usize::MAX {
          return None;
        }
        // SAFETY: `page` is the start of the page that `mmap` just mapped for `LEN` bytes. `madvise`
        // only checks or changes the advice of that page and touches no memory. Advice -1 is invalid
        // (`mm/madvise.c`, `madvise_behavior_valid`), so a kernel that accepts it, such as an emulator
        // that accepts all advice, is not trusted. Only then does a zero result for MADV_WIPEONFORK
        // prove that the kernel will give the page as zeros to a fork child.
        let wipes_on_fork = unsafe { madvise(page, LEN, -1) != 0 && madvise(page, LEN, MADV_WIPEONFORK) == 0 };
        if !wipes_on_fork {
          // The unused page stays mapped: one page, once for each process.
          return None;
        }
        // SAFETY: `page` is a live, non-null, page-aligned, zero-filled mapping that this module never
        // unmaps; see `page` for the reference proof.
        Some(unsafe { &*page.cast::<AtomicU64>() })
      }
    }
  }
  all(feature = "std", unix) => {
    mod imp {
      /// The ID of the process that recorded the origin. Apple's libSystem caches the ID and sets it
      /// again in a fork child, so the comparison costs no system call there.
      #[derive(Clone, Copy)]
      pub(super) struct Origin(u32);

      impl Origin {
        #[inline]
        pub(super) fn current() -> Self {
          Self(std::process::id())
        }

        #[inline]
        pub(super) fn is_current(&self) -> bool {
          std::process::id() == self.0
        }
      }
    }
  }
  _ => {
    mod imp {
      /// The target cannot fork, or the build has no `std` to detect a fork.
      #[derive(Clone, Copy)]
      pub(super) struct Origin;

      impl Origin {
        #[inline]
        pub(super) fn current() -> Self {
          Self
        }

        #[inline]
        pub(super) const fn is_current(&self) -> bool {
          true
        }
      }
    }
  }
}
