
use libfuzzer_sys::fuzz_target;
use rscrypto::{Checksum, Crc16Ccitt, Crc16Ibm};
use rscrypto_fuzz::{FuzzInput, assert_checksum_chunked, assert_checksum_combine, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    assert_checksum_combine::<Crc16Ccitt>(data, split);
    assert_checksum_combine::<Crc16Ibm>(data, split);

    assert_checksum_chunked::<Crc16Ccitt>(data, split);
    assert_checksum_chunked::<Crc16Ibm>(data, split);

    let ours_ccitt = Crc16Ccitt::checksum(data);
    let oracle_ccitt = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC).checksum(data);
    assert_eq!(ours_ccitt, oracle_ccitt, "crc16/ccitt oracle mismatch");

    let ours_ibm = Crc16Ibm::checksum(data);
    let oracle_ibm = crc::Crc::<u16>::new(&crc::CRC_16_ARC).checksum(data);
    assert_eq!(ours_ibm, oracle_ibm, "crc16/ibm oracle mismatch");
});
