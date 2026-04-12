#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{AsconCxof128, Xof};
use rscrypto_fuzz::{FuzzInput, split_at_ratio, some_or_return};

fn extend_to_overlong(input: &[u8], control: u8) -> Vec<u8> {
    let target_len = AsconCxof128::MAX_CUSTOMIZATION_LEN
        .strict_add(1)
        .strict_add(usize::from(control >> 4));
    let mut out = Vec::with_capacity(target_len);
    if input.is_empty() {
        out.resize(target_len, control);
        return out;
    }

    while out.len() < target_len {
        out.extend_from_slice(input);
    }
    out.truncate(target_len);
    out
}

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let customization_split: u8 = some_or_return!(input.byte());
    let message_split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let squeeze_split: u8 = some_or_return!(input.byte());
    let mode: u8 = some_or_return!(input.byte());
    let (customization, message) = split_at_ratio(input.rest(), customization_split);
    let out_len = out_len_byte as usize;
    let split_out = if out_len == 0 {
        0
    } else {
        out_len.strict_mul(squeeze_split as usize) / 255
    };

    let customization = if (mode & 1) != 0 && customization.len() <= AsconCxof128::MAX_CUSTOMIZATION_LEN {
        extend_to_overlong(customization, mode)
    } else {
        customization.to_vec()
    };

    if customization.len() > AsconCxof128::MAX_CUSTOMIZATION_LEN {
        let mut out = vec![0u8; out_len];
        assert!(
            AsconCxof128::new(&customization).is_err(),
            "cxof accepted overlong customization"
        );
        assert!(
            AsconCxof128::xof(&customization, message).is_err(),
            "cxof one-shot accepted overlong customization"
        );
        assert!(
            AsconCxof128::hash_into(&customization, message, &mut out).is_err(),
            "cxof hash_into accepted overlong customization"
        );
        return;
    }

    let (a, b) = split_at_ratio(message, message_split);
    let mut expected = vec![0u8; out_len];
    AsconCxof128::hash_into(&customization, message, &mut expected).expect("cxof hash_into");

    let mut h = AsconCxof128::new(&customization).expect("cxof new");
    h.update(a);
    h.update(b);
    h.update(&[]);
    let mut reader = h.finalize_xof();

    let mut got = vec![0u8; out_len];
    reader.squeeze(&mut got[..split_out]);
    reader.squeeze(&mut got[split_out..]);
    reader.squeeze(&mut []);
    assert_eq!(expected, got, "cxof streaming mismatch");

    h.reset();
    h.update(message);
    let mut reset = vec![0u8; out_len];
    h.finalize_xof().squeeze(&mut reset);
    assert_eq!(expected, reset, "cxof reset mismatch");

    let mut oneshot = AsconCxof128::xof(&customization, message).expect("cxof one-shot");
    let mut oneshot_out = vec![0u8; out_len];
    oneshot.squeeze(&mut oneshot_out);
    assert_eq!(expected, oneshot_out, "cxof oneshot mismatch");
});
