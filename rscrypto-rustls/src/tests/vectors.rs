use alloc::{collections::BTreeMap, vec::Vec};

pub(crate) fn hex(text: &str) -> Vec<u8> {
  assert!(text.len().is_multiple_of(2));
  text
    .as_bytes()
    .as_chunks::<2>()
    .0
    .iter()
    .map(|pair| {
      let high = char::from(pair[0]).to_digit(16).expect("hex digit");
      let low = char::from(pair[1]).to_digit(16).expect("hex digit");
      u8::try_from((high << 4) | low).expect("two hexadecimal digits fit a byte")
    })
    .collect()
}

pub(crate) fn fields(text: &str) -> BTreeMap<&str, Vec<u8>> {
  text
    .lines()
    .filter(|line| !line.is_empty() && !line.starts_with('#'))
    .map(|line| {
      let (name, value) = line.split_once(" = ").expect("vector field");
      (name, hex(value))
    })
    .collect()
}
