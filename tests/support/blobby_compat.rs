#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub(crate) enum Error {
  InvalidVlq,
  InvalidIndex,
  UnexpectedEnd,
  NotEnoughElements,
}

const NEXT_MASK: u8 = 0b1000_0000;
const VAL_MASK: u8 = 0b0111_1111;

fn read_vlq(data: &[u8], pos: &mut usize) -> Result<usize, Error> {
  let b = *data.get(*pos).ok_or(Error::UnexpectedEnd)?;
  *pos = (*pos).strict_add(1);
  let mut next = b & NEXT_MASK;
  let mut val = usize::from(b & VAL_MASK);

  macro_rules! step {
    () => {
      if next == 0 {
        return Ok(val);
      }
      let b = *data.get(*pos).ok_or(Error::UnexpectedEnd)?;
      *pos = (*pos).strict_add(1);
      next = b & NEXT_MASK;
      let t = usize::from(b & VAL_MASK);
      val = val.strict_add(1).strict_mul(128).strict_add(t);
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

pub(crate) struct BlobIterator<'a, const N: usize> {
  data: &'a [u8],
  dedup: Box<[&'a [u8]]>,
  pos: usize,
}

impl<'a, const N: usize> BlobIterator<'a, N> {
  pub(crate) fn new(data: &'a [u8]) -> Result<Self, Error> {
    if N == 0 {
      return Err(Error::NotEnoughElements);
    }

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
      data: data.get(pos..).ok_or(Error::UnexpectedEnd)?,
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

impl<'a, const N: usize> Iterator for BlobIterator<'a, N> {
  type Item = Result<[&'a [u8]; N], Error>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.pos >= self.data.len() {
      return None;
    }

    let mut out = [&[][..]; N];
    for slot in &mut out {
      if self.pos >= self.data.len() {
        self.error_block();
        return Some(Err(Error::NotEnoughElements));
      }
      *slot = match self.read() {
        Ok(value) => value,
        Err(err) => {
          self.error_block();
          return Some(Err(err));
        }
      };
    }
    Some(Ok(out))
  }
}
