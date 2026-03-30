#![allow(dead_code)]

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Error {
  InvalidVlq,
  InvalidIndex,
  UnexpectedEnd,
  NotEnoughElements,
}

const NEXT_MASK: u8 = 0b1000_0000;
const VAL_MASK: u8 = 0b0111_1111;

fn read_vlq(data: &[u8], pos: &mut usize) -> Result<usize, Error> {
  let b = *data.get(*pos).ok_or(Error::UnexpectedEnd)?;
  *pos += 1;
  let mut next = b & NEXT_MASK;
  let mut val = (b & VAL_MASK) as usize;

  macro_rules! step {
    () => {
      if next == 0 {
        return Ok(val);
      }
      let b = *data.get(*pos).ok_or(Error::UnexpectedEnd)?;
      *pos += 1;
      next = b & NEXT_MASK;
      let t = (b & VAL_MASK) as usize;
      val = ((val + 1) << 7) + t;
    };
  }

  step!();
  step!();
  step!();

  if next != 0 {
    return Err(Error::InvalidVlq);
  }

  Ok(val)
}

pub struct BlobIterator<'a> {
  data: &'a [u8],
  dedup: Box<[&'a [u8]]>,
  pos: usize,
}

impl<'a> BlobIterator<'a> {
  pub fn new(data: &'a [u8]) -> Result<Self, Error> {
    let mut pos = 0;
    let dedup_n = read_vlq(data, &mut pos)?;

    let mut dedup = vec![&[][..]; dedup_n];
    for entry in &mut dedup {
      let len = read_vlq(data, &mut pos)?;
      let end = pos.checked_add(len).ok_or(Error::UnexpectedEnd)?;
      *entry = data.get(pos..end).ok_or(Error::UnexpectedEnd)?;
      pos = end;
    }

    Ok(Self {
      data: &data[pos..],
      dedup: dedup.into_boxed_slice(),
      pos: 0,
    })
  }

  fn read(&mut self) -> Result<&'a [u8], Error> {
    let val = read_vlq(self.data, &mut self.pos)?;
    let is_ref = (val & 1) != 0;
    let val = val >> 1;

    if is_ref {
      return self.dedup.get(val).copied().ok_or(Error::InvalidIndex);
    }

    let start = self.pos;
    let end = start.checked_add(val).ok_or(Error::UnexpectedEnd)?;
    self.pos = end;
    self.data.get(start..end).ok_or(Error::UnexpectedEnd)
  }

  fn error_block(&mut self) {
    self.pos = self.data.len();
  }
}

impl<'a> Iterator for BlobIterator<'a> {
  type Item = Result<&'a [u8], Error>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.pos >= self.data.len() {
      return None;
    }

    let value = self.read();
    if value.is_err() {
      self.error_block();
    }
    Some(value)
  }
}

macro_rules! blob_iter {
  ($name:ident, $n:expr) => {
    pub struct $name<'a> {
      inner: BlobIterator<'a>,
    }

    impl<'a> $name<'a> {
      pub fn new(data: &'a [u8]) -> Result<Self, Error> {
        BlobIterator::new(data).map(|inner| Self { inner })
      }
    }

    impl<'a> Iterator for $name<'a> {
      type Item = Result<[&'a [u8]; $n], Error>;

      fn next(&mut self) -> Option<Self::Item> {
        let mut out = [&[][..]; $n];

        for (i, slot) in out.iter_mut().enumerate() {
          *slot = match self.inner.next() {
            Some(Ok(value)) => value,
            Some(Err(err)) => return Some(Err(err)),
            None if i == 0 => return None,
            None => {
              self.inner.error_block();
              return Some(Err(Error::NotEnoughElements));
            }
          };
        }

        Some(Ok(out))
      }
    }
  };
}

blob_iter!(Blob2Iterator, 2);
blob_iter!(Blob6Iterator, 6);
