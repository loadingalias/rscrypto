
use libfuzzer_sys::fuzz_target;
use rscrypto::{Checksum, Crc32, Crc32C};
use rscrypto_fuzz::{FuzzInput, assert_checksum_chunked, assert_checksum_combine, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    assert_checksum_combine::<Crc32>(data, split);
    assert_checksum_combine::<Crc32C>(data, split);

    assert_checksum_chunked::<Crc32>(data, split);
    assert_checksum_chunked::<Crc32C>(data, split);

    let ours_32 = Crc32::checksum(data);
    let oracle_32 = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC).checksum(data);
    assert_eq!(ours_32, oracle_32, "crc32 oracle mismatch");

    let ours_32c = Crc32C::checksum(data);
    let oracle_32c = crc::Crc::<u32>::new(&crc::CRC_32_ISCSI).checksum(data);
    assert_eq!(ours_32c, oracle_32c, "crc32c oracle mismatch");
});
