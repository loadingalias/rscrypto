#![no_main]

use std::io::{self, IoSlice, IoSliceMut, Read, Write};

use libfuzzer_sys::fuzz_target;
use rscrypto::{Blake3, Checksum as _, Crc32C, Digest as _};
use rscrypto_fuzz::{FuzzInput, split_at_ratio, some_or_return};

#[derive(Clone, Debug)]
struct PartialReader {
    data: Vec<u8>,
    pos: usize,
    max_per_call: usize,
}

impl PartialReader {
    #[inline]
    fn new(data: &[u8], max_per_call: usize) -> Self {
        Self {
            data: data.to_vec(),
            pos: 0,
            max_per_call,
        }
    }
}

impl Read for PartialReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = &self.data[self.pos..];
        let n = remaining.len().min(buf.len()).min(self.max_per_call);
        buf[..n].copy_from_slice(&remaining[..n]);
        self.pos = self.pos.strict_add(n);
        Ok(n)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let remaining = &self.data[self.pos..];
        let capacity = bufs.iter().fold(0usize, |sum, buf| sum.strict_add(buf.len()));
        let n = remaining.len().min(capacity).min(self.max_per_call);

        let mut copied = 0usize;
        for buf in bufs {
            if copied == n {
                break;
            }
            let take = buf.len().min(n.strict_sub(copied));
            if take == 0 {
                continue;
            }
            buf[..take].copy_from_slice(&remaining[copied..copied.strict_add(take)]);
            copied = copied.strict_add(take);
        }

        self.pos = self.pos.strict_add(n);
        Ok(n)
    }
}

#[derive(Clone, Debug, Default)]
struct PartialWriter {
    inner: Vec<u8>,
    flushes: usize,
    max_per_call: usize,
}

impl PartialWriter {
    #[inline]
    fn new(max_per_call: usize) -> Self {
        Self {
            inner: Vec::new(),
            flushes: 0,
            max_per_call,
        }
    }
}

impl Write for PartialWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = buf.len().min(self.max_per_call);
        self.inner.extend_from_slice(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flushes = self.flushes.strict_add(1);
        Ok(())
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut remaining = self.max_per_call;
        let mut written = 0usize;

        for buf in bufs {
            let take = remaining.min(buf.len());
            if remaining == 0 {
                break;
            }
            if take == 0 {
                continue;
            }
            self.inner.extend_from_slice(&buf[..take]);
            written = written.strict_add(take);
            remaining = remaining.strict_sub(take);
        }

        Ok(written)
    }
}

#[inline]
fn scalar_len(control: u8) -> usize {
    usize::from(control >> 2)
}

#[inline]
fn vectored_lens(control: u8) -> [usize; 3] {
    [
        usize::from(control & 0x0F),
        usize::from((control.rotate_left(3) & 0x0F).strict_add(1)),
        usize::from(control.rotate_left(6) & 0x0F),
    ]
}

#[inline]
fn extend_prefix(dst: &mut Vec<u8>, chunks: &[&[u8]], mut remaining: usize) {
    for chunk in chunks {
        if remaining == 0 {
            break;
        }
        let take = remaining.min(chunk.len());
        if take == 0 {
            continue;
        }
        dst.extend_from_slice(&chunk[..take]);
        remaining = remaining.strict_sub(take);
    }
}

#[inline]
fn sequential_chunks(data: &[u8], lens: [usize; 3]) -> [&[u8]; 3] {
    let first_len = lens[0].min(data.len());
    let (first, rest) = data.split_at(first_len);
    let second_len = lens[1].min(rest.len());
    let (second, rest) = rest.split_at(second_len);
    let third_len = lens[2].min(rest.len());
    let (third, _) = rest.split_at(third_len);
    [first, second, third]
}

fn reader_step<R: Read>(reader: &mut R, seen: &mut Vec<u8>, control: u8) -> io::Result<()> {
    match control % 3 {
        0 => {
            let mut buf = vec![0u8; scalar_len(control)];
            let n = reader.read(&mut buf)?;
            seen.extend_from_slice(&buf[..n]);
        }
        1 => {
            let lens = vectored_lens(control);
            let mut first = vec![0u8; lens[0]];
            let mut second = vec![0u8; lens[1]];
            let mut third = vec![0u8; lens[2]];
            let n = {
                let mut bufs = [
                    IoSliceMut::new(first.as_mut_slice()),
                    IoSliceMut::new(second.as_mut_slice()),
                    IoSliceMut::new(third.as_mut_slice()),
                ];
                reader.read_vectored(&mut bufs)?
            };
            let chunks = [first.as_slice(), second.as_slice(), third.as_slice()];
            extend_prefix(seen, &chunks, n);
        }
        _ => {
            let n = reader.read(&mut [])?;
            assert_eq!(n, 0, "zero-length read must report zero");
        }
    }

    Ok(())
}

fn writer_step<W: Write>(
    writer: &mut W,
    data: &[u8],
    cursor: &mut usize,
    accepted: &mut Vec<u8>,
    control: u8,
) -> io::Result<()> {
    match control % 4 {
        0 => {
            let request = scalar_len(control).min(data.len().strict_sub(*cursor));
            let end = cursor.strict_add(request);
            let n = writer.write(&data[*cursor..end])?;
            accepted.extend_from_slice(&data[*cursor..cursor.strict_add(n)]);
            *cursor = cursor.strict_add(n);
        }
        1 => {
            let chunks = sequential_chunks(&data[*cursor..], vectored_lens(control));
            let bufs = [
                IoSlice::new(chunks[0]),
                IoSlice::new(chunks[1]),
                IoSlice::new(chunks[2]),
            ];
            let n = writer.write_vectored(&bufs)?;
            extend_prefix(accepted, &chunks, n);
            *cursor = cursor.strict_add(n);
        }
        2 => {
            let n = writer.write(&[])?;
            assert_eq!(n, 0, "zero-length write must report zero");
        }
        _ => writer.flush()?,
    }

    Ok(())
}

fn fuzz_checksum_reader(data: &[u8], ops: &[u8], max_per_call: usize) {
    let mut reader = Crc32C::reader(PartialReader::new(data, max_per_call));
    let mut seen = Vec::with_capacity(data.len());

    reader_step(&mut reader, &mut seen, 0xFF).expect("checksum reader zero-length read");
    reader_step(&mut reader, &mut seen, 0x95).expect("checksum reader vectored read");
    assert_eq!(
        reader.checksum(),
        Crc32C::checksum(&seen),
        "checksum reader prelude mismatch"
    );

    for &op in ops.iter().take(32) {
        reader_step(&mut reader, &mut seen, op).expect("checksum reader step");
        assert_eq!(
            reader.checksum(),
            Crc32C::checksum(&seen),
            "checksum reader step mismatch"
        );
    }

    loop {
        let mut buf = [0u8; 17];
        let n = reader.read(&mut buf).expect("checksum reader drain");
        seen.extend_from_slice(&buf[..n]);
        assert_eq!(
            reader.checksum(),
            Crc32C::checksum(&seen),
            "checksum reader drain mismatch"
        );
        if n == 0 {
            break;
        }
    }

    assert_eq!(seen, data, "checksum reader returned different bytes");

    let (inner, checksum) = reader.into_parts();
    assert_eq!(inner.pos, data.len(), "checksum reader inner position mismatch");
    assert_eq!(inner.data, data, "checksum reader inner payload mutated");
    assert_eq!(checksum, Crc32C::checksum(data), "checksum reader final mismatch");
}

fn fuzz_digest_reader(data: &[u8], ops: &[u8], max_per_call: usize) {
    let mut reader = Blake3::reader(PartialReader::new(data, max_per_call));
    let mut seen = Vec::with_capacity(data.len());

    reader_step(&mut reader, &mut seen, 0xFF).expect("digest reader zero-length read");
    reader_step(&mut reader, &mut seen, 0x59).expect("digest reader vectored read");
    assert_eq!(
        reader.digest(),
        Blake3::digest(&seen),
        "digest reader prelude mismatch"
    );

    for &op in ops.iter().take(32) {
        reader_step(&mut reader, &mut seen, op).expect("digest reader step");
        assert_eq!(
            reader.digest(),
            Blake3::digest(&seen),
            "digest reader step mismatch"
        );
    }

    loop {
        let mut buf = [0u8; 19];
        let n = reader.read(&mut buf).expect("digest reader drain");
        seen.extend_from_slice(&buf[..n]);
        assert_eq!(
            reader.digest(),
            Blake3::digest(&seen),
            "digest reader drain mismatch"
        );
        if n == 0 {
            break;
        }
    }

    assert_eq!(seen, data, "digest reader returned different bytes");

    let (inner, digest) = reader.into_parts();
    assert_eq!(inner.pos, data.len(), "digest reader inner position mismatch");
    assert_eq!(inner.data, data, "digest reader inner payload mutated");
    assert_eq!(digest, Blake3::digest(data), "digest reader final mismatch");
}

fn fuzz_checksum_writer(data: &[u8], ops: &[u8], max_per_call: usize) {
    let mut writer = Crc32C::writer(PartialWriter::new(max_per_call));
    let mut accepted = Vec::with_capacity(data.len());
    let mut cursor = 0usize;

    writer_step(&mut writer, data, &mut cursor, &mut accepted, 0x02).expect("checksum writer zero-length write");
    writer_step(&mut writer, data, &mut cursor, &mut accepted, 0x03).expect("checksum writer flush");
    assert_eq!(
        writer.checksum(),
        Crc32C::checksum(&accepted),
        "checksum writer prelude mismatch"
    );

    for &op in ops.iter().take(32) {
        writer_step(&mut writer, data, &mut cursor, &mut accepted, op).expect("checksum writer step");
        assert_eq!(
            writer.checksum(),
            Crc32C::checksum(&accepted),
            "checksum writer step mismatch"
        );
    }

    while cursor < data.len() {
        let end = data.len().min(cursor.strict_add(max_per_call.strict_add(1)));
        let n = writer.write(&data[cursor..end]).expect("checksum writer drain");
        assert!(n > 0, "checksum writer made no progress");
        accepted.extend_from_slice(&data[cursor..cursor.strict_add(n)]);
        cursor = cursor.strict_add(n);
        assert_eq!(
            writer.checksum(),
            Crc32C::checksum(&accepted),
            "checksum writer drain mismatch"
        );
    }

    assert_eq!(writer.write(&[]).expect("checksum writer final empty write"), 0);
    writer.flush().expect("checksum writer final flush");
    assert_eq!(accepted, data, "checksum writer accepted different bytes");

    let (inner, checksum) = writer.into_parts();
    assert_eq!(inner.inner, data, "checksum writer inner payload mismatch");
    assert!(inner.flushes > 0, "checksum writer flush was not forwarded");
    assert_eq!(checksum, Crc32C::checksum(data), "checksum writer final mismatch");
}

fn fuzz_digest_writer(data: &[u8], ops: &[u8], max_per_call: usize) {
    let mut writer = Blake3::writer(PartialWriter::new(max_per_call));
    let mut accepted = Vec::with_capacity(data.len());
    let mut cursor = 0usize;

    writer_step(&mut writer, data, &mut cursor, &mut accepted, 0x02).expect("digest writer zero-length write");
    writer_step(&mut writer, data, &mut cursor, &mut accepted, 0x03).expect("digest writer flush");
    assert_eq!(
        writer.digest(),
        Blake3::digest(&accepted),
        "digest writer prelude mismatch"
    );

    for &op in ops.iter().take(32) {
        writer_step(&mut writer, data, &mut cursor, &mut accepted, op).expect("digest writer step");
        assert_eq!(
            writer.digest(),
            Blake3::digest(&accepted),
            "digest writer step mismatch"
        );
    }

    while cursor < data.len() {
        let end = data.len().min(cursor.strict_add(max_per_call.strict_add(1)));
        let n = writer.write(&data[cursor..end]).expect("digest writer drain");
        assert!(n > 0, "digest writer made no progress");
        accepted.extend_from_slice(&data[cursor..cursor.strict_add(n)]);
        cursor = cursor.strict_add(n);
        assert_eq!(
            writer.digest(),
            Blake3::digest(&accepted),
            "digest writer drain mismatch"
        );
    }

    assert_eq!(writer.write(&[]).expect("digest writer final empty write"), 0);
    writer.flush().expect("digest writer final flush");
    assert_eq!(accepted, data, "digest writer accepted different bytes");

    let (inner, digest) = writer.into_parts();
    assert_eq!(inner.inner, data, "digest writer inner payload mismatch");
    assert!(inner.flushes > 0, "digest writer flush was not forwarded");
    assert_eq!(digest, Blake3::digest(data), "digest writer final mismatch");
}

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let reader_limit = usize::from(some_or_return!(input.byte()) % 32).strict_add(1);
    let writer_limit = usize::from(some_or_return!(input.byte()) % 32).strict_add(1);
    let split = some_or_return!(input.byte());
    let (ops, payload) = split_at_ratio(input.rest(), split);

    fuzz_checksum_reader(payload, ops, reader_limit);
    fuzz_digest_reader(payload, ops, reader_limit);
    fuzz_checksum_writer(payload, ops, writer_limit);
    fuzz_digest_writer(payload, ops, writer_limit);
});
