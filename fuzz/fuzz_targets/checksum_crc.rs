#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{
    Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64,
    Crc64Nvme,
};
use rscrypto_fuzz::{
    FuzzInput, assert_checksum_chunked, assert_checksum_combine, split_at_ratio, some_or_return,
};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    // Property: combine equivalence for all CRC variants
    assert_checksum_combine::<Crc16Ccitt>(data, split);
    assert_checksum_combine::<Crc16Ibm>(data, split);
    assert_checksum_combine::<Crc24OpenPgp>(data, split);
    assert_checksum_combine::<Crc32>(data, split);
    assert_checksum_combine::<Crc32C>(data, split);
    assert_checksum_combine::<Crc64>(data, split);
    assert_checksum_combine::<Crc64Nvme>(data, split);

    // Property: streaming equivalence for all CRC variants
    assert_checksum_chunked::<Crc16Ccitt>(data, split);
    assert_checksum_chunked::<Crc16Ibm>(data, split);
    assert_checksum_chunked::<Crc24OpenPgp>(data, split);
    assert_checksum_chunked::<Crc32>(data, split);
    assert_checksum_chunked::<Crc32C>(data, split);
    assert_checksum_chunked::<Crc64>(data, split);
    assert_checksum_chunked::<Crc64Nvme>(data, split);

    // Differential: rscrypto ↔ crc crate
    {
        let ours_32 = Crc32::checksum(data);
        let oracle_32 = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC).checksum(data);
        assert_eq!(ours_32, oracle_32, "crc32 oracle mismatch");

        let ours_32c = Crc32C::checksum(data);
        let oracle_32c = crc::Crc::<u32>::new(&crc::CRC_32_ISCSI).checksum(data);
        assert_eq!(ours_32c, oracle_32c, "crc32c oracle mismatch");

        let ours_64 = Crc64::checksum(data);
        let oracle_64 = crc::Crc::<u64>::new(&crc::CRC_64_XZ).checksum(data);
        assert_eq!(ours_64, oracle_64, "crc64/xz oracle mismatch");

        let ours_16ccitt = Crc16Ccitt::checksum(data);
        let oracle_16ccitt =
            crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC).checksum(data);
        assert_eq!(ours_16ccitt, oracle_16ccitt, "crc16/ccitt oracle mismatch");

        let ours_16ibm = Crc16Ibm::checksum(data);
        let oracle_16ibm = crc::Crc::<u16>::new(&crc::CRC_16_ARC).checksum(data);
        assert_eq!(ours_16ibm, oracle_16ibm, "crc16/ibm oracle mismatch");

        let ours_24 = Crc24OpenPgp::checksum(data);
        let oracle_24 =
            crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP).checksum(data);
        assert_eq!(ours_24, oracle_24, "crc24/openpgp oracle mismatch");

        let ours_64nvme = Crc64Nvme::checksum(data);
        // CRC-64/NVME not in crc-catalog; verify combine consistency instead.
        let (a, b) = split_at_ratio(data, split);
        let combined = Crc64Nvme::combine(Crc64Nvme::checksum(a), Crc64Nvme::checksum(b), b.len());
        assert_eq!(ours_64nvme, combined, "crc64/nvme combine vs checksum mismatch");
    }
});
