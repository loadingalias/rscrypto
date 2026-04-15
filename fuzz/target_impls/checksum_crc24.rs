
use libfuzzer_sys::fuzz_target;
use rscrypto::{Checksum, Crc24OpenPgp};
use rscrypto_fuzz::{FuzzInput, assert_checksum_chunked, assert_checksum_combine, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    assert_checksum_combine::<Crc24OpenPgp>(data, split);
    assert_checksum_chunked::<Crc24OpenPgp>(data, split);

    let ours = Crc24OpenPgp::checksum(data);
    let oracle = crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP).checksum(data);
    assert_eq!(ours, oracle, "crc24/openpgp oracle mismatch");
});
