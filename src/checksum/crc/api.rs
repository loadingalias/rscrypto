//! The public CRC types, generated from one definition.

/// Inputs below this length run the portable kernel inline, without the kernel call.
pub(in crate::checksum) const INLINE_MAX: usize = 15;

/// Buffer size of the buffered CRC wrappers.
#[cfg(feature = "alloc")]
pub(in crate::checksum) const BUFFER_SIZE: usize = 512;

/// The update length from which a buffered CRC wrapper calls the kernel directly.
#[cfg(feature = "alloc")]
pub(in crate::checksum) const fn buffered_threshold() -> usize {
  64
}

/// Defines one public CRC type, its buffered wrapper, and their trait implementations.
macro_rules! define_crc {
  (
    $(#[$meta:meta])*
    pub struct $name:ident;
    word: $word:ty,
    output_size: $output_size:expr,
    spec: $spec:expr,
    hw: $hw:expr,
    env: $env:literal,
    buffered: $buffered:ident,
  ) => {
    $(#[$meta])*
    #[derive(Clone, Copy)]
    pub struct $name {
      /// Register in the word domain.
      register: $word,
      kernel: $crate::checksum::crc::dispatch::Kernel<$word>,
    }

    const _: () = {
      use $crate::checksum::crc::{Algo, Crc, Hw, Spec, api::INLINE_MAX, dispatch, mask, portable};

      const SPEC: Spec = $spec;
      static DATA: Crc<$word> = $crate::checksum::crc::crc_data!($word, SPEC);
      static KERNEL: $crate::backend::cache::OnceCache<dispatch::Kernel<$word>> =
        $crate::backend::cache::OnceCache::new();
      static SHIFT: $crate::backend::cache::OnceCache<dispatch::Shift> = $crate::backend::cache::OnceCache::new();
      #[cfg(feature = "std")]
      static CONFIG: $crate::backend::cache::OnceCache<dispatch::CrcConfig> =
        $crate::backend::cache::OnceCache::new();

      /// Shift from a register in the specification's form to the word domain.
      const WORD_SHIFT: u32 = if SPEC.reflected { 0 } else { <$word>::BITS.strict_sub(SPEC.width) };

      /// The low bits of a 64-bit value, as a word.
      const fn narrow(value: u64) -> $word {
        match value.to_le_bytes().first_chunk::<{ core::mem::size_of::<$word>() }>() {
          Some(low) => <$word>::from_le_bytes(*low),
          None => 0,
        }
      }

      /// A word, zero-extended.
      const fn widen(word: $word) -> u64 {
        let bytes = word.to_le_bytes();
        let mut wide = [0u8; 8];
        let mut i = 0usize;
        while i < bytes.len() {
          wide[i] = bytes[i];
          i = i.strict_add(1);
        }
        u64::from_le_bytes(wide)
      }

      /// The word-domain register for a specification-form register.
      const fn to_word(register: u64) -> $word {
        narrow((register & mask(SPEC.width)) << WORD_SHIFT)
      }

      /// The finalized CRC for a word-domain register.
      const fn finish(register: $word) -> $word {
        narrow((widen(register) >> WORD_SHIFT) ^ SPEC.xorout)
      }

      const INIT: $word = to_word(SPEC.init);

      impl Algo for $name {
        type W = $word;
        const CRC: &'static Crc<$word> = &DATA;
        const HW: Hw = $hw;
      }

      fn config() -> dispatch::CrcConfig {
        #[cfg(feature = "std")]
        {
          CONFIG.get_or_init(|| dispatch::config($env, <$name as Algo>::HW))
        }
        #[cfg(not(feature = "std"))]
        {
          dispatch::config($env, <$name as Algo>::HW)
        }
      }

      fn tier() -> dispatch::CrcForce {
        dispatch::tier(config().effective_force, <$name as Algo>::HW)
      }

      fn kernel() -> dispatch::Kernel<$word> {
        KERNEL.get_or_init(|| dispatch::kernel::<$name>(tier()))
      }

      /// Resolves the kernel on first use, for hashers made in `const` context.
      fn lazy(register: $word, data: &[u8]) -> $word {
        kernel()(register, data)
      }

      impl $name {
        /// Creates a hasher that continues from a previous finalized CRC value.
        #[inline]
        #[must_use]
        pub const fn resume(crc: $word) -> Self {
          Self {
            register: to_word(widen(crc) ^ SPEC.xorout),
            kernel: lazy,
          }
        }

        /// Returns the effective force configuration of this CRC.
        #[must_use]
        pub fn config() -> $crate::checksum::config::CrcConfig {
          config()
        }

        /// Returns the name of the kernel that runs for an input of `len` bytes.
        #[must_use]
        pub fn kernel_name_for_len(len: usize) -> &'static str {
          dispatch::kernel_name(tier(), <$name as Algo>::HW, len)
        }

        /// Creates a buffering wrapper that coalesces short updates before kernel dispatch.
        #[cfg(feature = "alloc")]
        #[must_use]
        pub fn buffered() -> $buffered {
          $buffered::new()
        }
      }

      impl $crate::traits::Checksum for $name {
        const OUTPUT_SIZE: usize = $output_size;
        type Output = $word;

        #[inline]
        fn new() -> Self {
          Self {
            register: INIT,
            kernel: kernel(),
          }
        }

        #[inline]
        fn with_initial(initial: $word) -> Self {
          Self {
            register: to_word(widen(initial) ^ SPEC.xorout),
            kernel: kernel(),
          }
        }

        #[inline]
        fn update(&mut self, data: &[u8]) {
          self.register = if data.len() <= INLINE_MAX {
            portable::update(&DATA, self.register, data)
          } else {
            (self.kernel)(self.register, data)
          };
        }

        #[inline]
        fn finalize(&self) -> $word {
          finish(self.register)
        }

        #[inline]
        fn reset(&mut self) {
          self.register = INIT;
        }

        #[inline]
        fn checksum(data: &[u8]) -> $word {
          if data.len() <= INLINE_MAX {
            finish(portable::update(&DATA, INIT, data))
          } else {
            finish(kernel()(INIT, data))
          }
        }
      }

      impl $crate::traits::ChecksumCombine for $name {
        fn combine(crc_a: $word, crc_b: $word, len_b: usize) -> $word {
          let shift = SHIFT.get_or_init(|| dispatch::shift::<$name>(tier()));
          narrow(DATA.combine(widen(crc_a), widen(crc_b), len_b, shift))
        }
      }

      #[cfg(feature = "diag")]
      impl $crate::checksum::introspect::KernelIntrospect for $name {
        fn kernel_name_for_len(len: usize) -> &'static str {
          $name::kernel_name_for_len(len)
        }
      }
    };

    impl core::fmt::Debug for $name {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(stringify!($name)).finish_non_exhaustive()
      }
    }

    impl Default for $name {
      fn default() -> Self {
        <Self as $crate::traits::Checksum>::new()
      }
    }

    impl_std_io_write_for_checksum!($name);

    #[cfg(feature = "alloc")]
    define_buffered_crc! {
      #[doc = concat!("A buffering wrapper around [`", stringify!($name), "`] for streams of short updates.")]
      pub struct $buffered<$name> {
        buffer_size: $crate::checksum::crc::api::BUFFER_SIZE,
        threshold_fn: $crate::checksum::crc::api::buffered_threshold,
      }
    }
  };
}
pub(in crate::checksum) use define_crc;
