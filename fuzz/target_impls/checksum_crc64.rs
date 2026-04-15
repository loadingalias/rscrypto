
use libfuzzer_sys::fuzz_target;
use rscrypto::{Checksum, ChecksumCombine, Crc64, Crc64Nvme};
use rscrypto_fuzz::{
    FuzzInput, assert_checksum_chunked, assert_checksum_combine, split_at_ratio, some_or_return,
};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    assert_checksum_combine::<Crc64>(data, split);
    assert_checksum_combine::<Crc64Nvme>(data, split);

    assert_checksum_chunked::<Crc64>(data, split);
    assert_checksum_chunked::<Crc64Nvme>(data, split);

    let ours_64 = Crc64::checksum(data);
    let oracle_64 = crc::Crc::<u64>::new(&crc::CRC_64_XZ).checksum(data);
    assert_eq!(ours_64, oracle_64, "crc64/xz oracle mismatch");

    let ours_nvme = Crc64Nvme::checksum(data);
    let (a, b) = split_at_ratio(data, split);
    let combined = Crc64Nvme::combine(Crc64Nvme::checksum(a), Crc64Nvme::checksum(b), b.len());
    assert_eq!(ours_nvme, combined, "crc64/nvme combine vs checksum mismatch");
});
