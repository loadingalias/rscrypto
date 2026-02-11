// Family Profile: AARCH64_SERVER_NEON
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_SERVER_NEON: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 4095, 4096],
    xs: KernelId::Portable,
    s: KernelId::Portable,
    m: KernelId::Aarch64Neon,
    l: KernelId::Aarch64Neon,
  },
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: KernelId::Aarch64Neon,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 8,
    spawn_cost_bytes: 1,
    merge_cost_bytes: 65536,
    bytes_per_core_small: 16384,
    bytes_per_core_medium: 122879,
    bytes_per_core_large: 1040383,
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
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_SERVER_NEON: FamilyProfile = default_kind_profile();
