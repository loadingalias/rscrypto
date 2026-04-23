macro_rules! define_unit_error {
  (
    $(#[$meta:meta])*
    $vis:vis struct $name:ident;
    $display:literal
  ) => {
    $(#[$meta])*
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    $vis struct $name;

    impl $name {
      #[inline]
      #[must_use]
      pub const fn new() -> Self {
        Self
      }
    }

    impl Default for $name {
      #[inline]
      fn default() -> Self {
        Self::new()
      }
    }

    impl core::fmt::Display for $name {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str($display)
      }
    }

    impl core::error::Error for $name {}
  };
}

#[cfg(all(
  any(target_arch = "aarch64", target_arch = "riscv64", target_arch = "x86_64"),
  any(feature = "chacha20poly1305", feature = "xchacha20poly1305")
))]
macro_rules! define_target_feature_forwarder {
  (
    $vis:vis fn $name:ident($($arg:ident : $arg_ty:ty),* $(,)?) $(-> $ret:ty)? {
      feature = $feature:literal;
      outer_safety = $outer_safety:literal;
      inner_safety = $inner_safety:literal;
      call = $call:expr;
    }
  ) => {
    #[inline]
    $vis fn $name($($arg : $arg_ty),*) $(-> $ret)? {
      // SAFETY: $outer_safety
      unsafe { __target_feature_forwarder_impl($($arg),*) }
    }

    #[target_feature(enable = $feature)]
    unsafe fn __target_feature_forwarder_impl($($arg : $arg_ty),*) $(-> $ret)? {
      // SAFETY: $inner_safety
      unsafe { $call }
    }
  };
}

#[cfg(feature = "sha2")]
macro_rules! define_sha_family_dispatch {
  (
    kernel_id: $kernel_id:ty,
    compress_fn_ty: $compress_fn_ty:ty,
    portable_kernel: $portable_kernel:path,
    compress_fn: $compress_fn:path,
    required_caps: $required_caps:path,
    runtime_table: $runtime_table:path,
    output_len: $output_len:expr,
    word_bytes: $word_bytes:expr,
    total_bits_ty: $total_bits_ty:ty,
    length_offset: $length_offset:expr,
    h0: $h0:path,
    compile_time: {
      hw: $compile_time_hw:expr,
      name: $compile_time_name:expr,
      best: $compile_time_best:expr,
    },
  ) => {
    #[derive(Clone, Copy)]
    struct Entry {
      compress_blocks: $compress_fn_ty,
      #[cfg(any(test, feature = "diag"))]
      #[allow(dead_code)]
      name: &'static str,
    }

    #[derive(Clone, Copy)]
    struct ActiveDispatch {
      boundaries: [usize; 3],
      xs: Entry,
      s: Entry,
      m: Entry,
      l: Entry,
    }

    static ACTIVE: crate::backend::cache::OnceCache<ActiveDispatch> = crate::backend::cache::OnceCache::new();

    #[inline]
    #[must_use]
    fn resolve(id: $kernel_id, caps: crate::platform::Caps) -> $kernel_id {
      if caps.has($required_caps(id)) {
        id
      } else {
        $portable_kernel
      }
    }

    #[inline]
    #[must_use]
    fn active() -> ActiveDispatch {
      ACTIVE.get_or_init(|| {
        let caps = crate::platform::caps();
        let table = $runtime_table(caps);

        let xs_id = resolve(table.xs, caps);
        let s_id = resolve(table.s, caps);
        let m_id = resolve(table.m, caps);
        let l_id = resolve(table.l, caps);

        ActiveDispatch {
          boundaries: table.boundaries,
          xs: Entry {
            compress_blocks: $compress_fn(xs_id),
            #[cfg(any(test, feature = "diag"))]
            name: xs_id.as_str(),
          },
          s: Entry {
            compress_blocks: $compress_fn(s_id),
            #[cfg(any(test, feature = "diag"))]
            name: s_id.as_str(),
          },
          m: Entry {
            compress_blocks: $compress_fn(m_id),
            #[cfg(any(test, feature = "diag"))]
            name: m_id.as_str(),
          },
          l: Entry {
            compress_blocks: $compress_fn(l_id),
            #[cfg(any(test, feature = "diag"))]
            name: l_id.as_str(),
          },
        }
      })
    }

    #[inline]
    #[must_use]
    fn select(d: &ActiveDispatch, len: usize) -> Entry {
      let [xs_max, s_max, m_max] = d.boundaries;
      if len <= xs_max {
        d.xs
      } else if len <= s_max {
        d.s
      } else if len <= m_max {
        d.m
      } else {
        d.l
      }
    }

    #[cfg(any(test, feature = "diag"))]
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub fn kernel_name_for_len(len: usize) -> &'static str {
      if $compile_time_hw {
        return $compile_time_name;
      }
      let d = active();
      select(&d, len).name
    }

    #[inline]
    #[must_use]
    pub fn digest(data: &[u8]) -> [u8; $output_len] {
      if $compile_time_hw {
        return digest_oneshot(data, $compile_time_best);
      }
      let d = active();
      let compress = select(&d, data.len()).compress_blocks;
      digest_oneshot(data, compress)
    }

    #[inline]
    fn digest_oneshot(data: &[u8], compress_blocks: $compress_fn_ty) -> [u8; $output_len] {
      let mut state = $h0;

      let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
      if !blocks.is_empty() {
        compress_blocks(&mut state, &data[..blocks.len().strict_mul(BLOCK_LEN)]);
      }

      let total_bits = (data.len() as $total_bits_ty).strict_mul(8);

      let mut block = [0u8; BLOCK_LEN];
      block[..rest.len()].copy_from_slice(rest);
      block[rest.len()] = 0x80;

      if rest.len() >= $length_offset {
        compress_blocks(&mut state, &block);
        block = [0u8; BLOCK_LEN];
      }

      block[$length_offset..BLOCK_LEN].copy_from_slice(&total_bits.to_be_bytes());
      compress_blocks(&mut state, &block);

      let mut out = [0u8; $output_len];
      for (chunk, &word) in out.chunks_exact_mut($word_bytes).zip(state.iter()) {
        chunk.copy_from_slice(&word.to_be_bytes());
      }
      out
    }

    #[inline]
    #[must_use]
    pub(crate) fn compress_dispatch() -> crate::hashes::crypto::dispatch_util::SizeClassDispatch<$compress_fn_ty> {
      if $compile_time_hw {
        let f = $compile_time_best;
        return crate::hashes::crypto::dispatch_util::SizeClassDispatch {
          boundaries: [usize::MAX; 3],
          xs: f,
          s: f,
          m: f,
          l: f,
        };
      }
      let d = active();
      crate::hashes::crypto::dispatch_util::SizeClassDispatch {
        boundaries: d.boundaries,
        xs: d.xs.compress_blocks,
        s: d.s.compress_blocks,
        m: d.m.compress_blocks,
        l: d.l.compress_blocks,
      }
    }
  };
}

#[cfg(any(feature = "blake2b", feature = "blake2s"))]
macro_rules! define_blake2_dispatch {
  (
    kernel_id: $kernel_id:ty,
    compress_fn_ty: $compress_fn_ty:ty,
    portable_kernel: $portable_kernel:path,
    compress_fn: $compress_fn:path,
    required_caps: $required_caps:path,
    candidates: [$($candidates:tt)*],
  ) => {
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    #[derive(Clone, Copy)]
    struct Resolved {
      compress: $compress_fn_ty,
      #[cfg(any(test, feature = "diag"))]
      #[allow(dead_code)]
      name: &'static str,
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    static ACTIVE: crate::backend::cache::OnceCache<Resolved> = crate::backend::cache::OnceCache::new();

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    fn resolve() -> Resolved {
      let caps = crate::platform::caps();
      let candidates: &[$kernel_id] = &[$($candidates)*];

      for &id in candidates {
        if caps.has($required_caps(id)) {
          return Resolved {
            compress: $compress_fn(id),
            #[cfg(any(test, feature = "diag"))]
            name: id.as_str(),
          };
        }
      }

      Resolved {
        compress: $compress_fn($portable_kernel),
        #[cfg(any(test, feature = "diag"))]
        name: $portable_kernel.as_str(),
      }
    }

    #[inline]
    #[must_use]
    pub(crate) fn compress_dispatch() -> $compress_fn_ty {
      if super::kernels::COMPILE_TIME_HW {
        return super::kernels::compile_time_best();
      }
      #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
      {
        $compress_fn($portable_kernel)
      }
      #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
      ACTIVE.get_or_init(resolve).compress
    }

    #[cfg(any(test, feature = "diag"))]
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub fn kernel_name_for_len(_len: usize) -> &'static str {
      if super::kernels::COMPILE_TIME_HW {
        return compile_time_name();
      }
      #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
      {
        $portable_kernel.as_str()
      }
      #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
      ACTIVE.get_or_init(resolve).name
    }

    #[cfg(any(test, feature = "diag"))]
    #[allow(dead_code)]
    const fn compile_time_name() -> &'static str {
      if cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl"
      )) {
        "x86/avx512vl"
      } else if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        "x86/avx2"
      } else if cfg!(all(
        target_arch = "aarch64",
        target_feature = "neon",
        not(target_os = "macos")
      )) {
        "aarch64/neon"
      } else {
        "portable"
      }
    }
  };
}

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
macro_rules! define_aead_key_type {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    #[derive(Clone)]
    pub struct $name([u8; Self::LENGTH]);

    impl PartialEq for $name {
      fn eq(&self, other: &Self) -> bool {
        crate::traits::ct::constant_time_eq(&self.0, &other.0)
      }
    }

    impl Eq for $name {}

    impl $name {
      /// Key length in bytes.
      pub const LENGTH: usize = $len;

      /// Construct a typed key from raw bytes.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Explicitly extract the key bytes into a zeroizing wrapper.
      #[inline]
      #[must_use]
      pub fn expose_secret(&self) -> crate::SecretBytes<{ Self::LENGTH }> {
        crate::SecretBytes::new(self.0)
      }

      /// Borrow the key bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl_ct_eq!($name);

    impl core::fmt::Debug for $name {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}(****)", stringify!($name))
      }
    }

    impl $name {
      #[doc = concat!(
                                                    "Construct a key by filling bytes from the provided closure.\n\n",
                                                    "```rust\n",
                                                    "# use rscrypto::",
                                                    stringify!($name),
                                                    ";\n",
                                                    "let key = ",
                                                    stringify!($name),
                                                    "::generate(|buf| buf.fill(0xA5));\n",
                                                    "assert_eq!(key.as_bytes(), &[0xA5; ",
                                                    stringify!($name),
                                                    "::LENGTH]);\n",
                                                    "```"
                                                  )]
      #[inline]
      #[must_use]
      pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
        let mut bytes = [0u8; Self::LENGTH];
        fill(&mut bytes);
        Self(bytes)
      }

      impl_getrandom!();
    }

    impl_hex_fmt_secret!($name);
    impl_serde_bytes!($name);

    impl Drop for $name {
      fn drop(&mut self) {
        crate::traits::ct::zeroize(&mut self.0);
      }
    }
  };
}

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
macro_rules! define_aead_tag_type {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct $name([u8; Self::LENGTH]);

    impl $name {
      /// Tag length in bytes.
      pub const LENGTH: usize = $len;

      /// Construct a typed tag from raw bytes.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Return the tag bytes.
      #[inline]
      #[must_use]
      pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
        self.0
      }

      /// Borrow the tag bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }
    }

    impl Default for $name {
      #[inline]
      fn default() -> Self {
        Self([0u8; Self::LENGTH])
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl_ct_eq!($name);

    impl core::fmt::Debug for $name {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}(", stringify!($name))?;
        crate::hex::fmt_hex_lower(&self.0, f)?;
        write!(f, ")")
      }
    }

    impl_hex_fmt!($name);
    impl_serde_bytes!($name);
  };
}
