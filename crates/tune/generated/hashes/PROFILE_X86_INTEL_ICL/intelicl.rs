// Family Profile: X86_INTEL_ICL
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_INTEL_ICL: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 64, 4096],
    xs: KernelId::X86Avx512,
    s: KernelId::X86Sse41,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Avx512,
    bulk: KernelId::X86Avx512,
  },
  parallel: ParallelTable {
    min_bytes: 131072,
    min_chunks: 128,
    max_threads: 8,
    spawn_cost_bytes: 1,
    merge_cost_bytes: 678766,
    bytes_per_core_small: 24576,
    bytes_per_core_medium: 111761,
    bytes_per_core_large: 963729,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 18446744073709551615,
    min_chunks: 18446744073709551615,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_INTEL_ICL: FamilyProfile = default_kind_profile();
