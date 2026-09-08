// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CtSummary {
    pub max_t: f64,
    pub max_tau: f64,
    pub sample_size: usize,
}

impl CtSummary {
    pub fn fmt(&self) -> String {
        let &CtSummary {
            max_t,
            max_tau,
            sample_size,
        } = self;
        format!(
            "n == {:+0.3}M, max t = {:+0.5}, max tau = {:+0.5}, (5/tau)^2 = {}",
            (sample_size as f64) / 1_000_000f64,
            max_t,
            max_tau,
            (5f64 / max_tau).powi(2) as usize
        )
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct CtTest {
    means: (f64, f64),
    sq_diffs: (f64, f64),
    sizes: (usize, usize),
}

#[derive(Default)]
pub struct CtCtx {
    tests: Vec<CtTest>,
    percentiles: Vec<f64>,
}

// NaNs are smaller than everything
fn local_cmp(x: f64, y: f64) -> cmp::Ordering {
    use std::cmp::Ordering::{Equal, Greater, Less};
    if y.is_nan() {
        Greater
    } else if x.is_nan() || x < y {
        Less
    } else if x == y {
        Equal
    } else {
        Greater
    }
}

/// Helper function: extract a value representing the `pct` percentile of a sorted sample-set,
/// using linear interpolation. If samples are not sorted, return nonsensical value.
fn percentile_of_sorted(sorted_samples: &[f64], pct: f64) -> f64 {
    assert!(!sorted_samples.is_empty());
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let zero = 0f64;
    assert!(zero <= pct);
    let hundred = 100f64;
    assert!(pct <= hundred);
    let length = (sorted_samples.len() - 1) as f64;
    let rank = (pct / hundred) * length;
    let lrank = rank.floor();
    let d = rank - lrank;
    let n = lrank as usize;
    let lo = sorted_samples[n];
    let hi = sorted_samples[n + 1];
    lo + (hi - lo) * d
}

/// Return the percentiles at f(1), f(2), ..., f(100) of the runtime distribution, where
/// `f(k) = 1 - 0.5^(10k / 100)`
pub fn prepare_percentiles(durations: &[u64]) -> Vec<f64> {
    let sorted: Vec<f64> = {
        let mut v = durations.to_vec();
        v.sort();
        v.into_iter().map(|d| d as f64).collect()
    };

    // Collect all the percentile values
    (0..100)
        .map(|i| {
            let pct = {
                let exp = f64::from(10 * (i + 1)) / 100f64;
                1f64 - 0.5f64.powf(exp)
            };
            percentile_of_sorted(&sorted, 100f64 * pct)
        })
        .collect()
}

pub fn update_ct_stats(
    ctx: Option<CtCtx>,
    (left_samples, right_samples): &(Vec<u64>, Vec<u64>),
) -> (CtSummary, CtCtx) {
    // Only construct the context (that is, percentiles and test structs) on the first run
    let (mut tests, percentiles) = match ctx {
        Some(c) => (c.tests, c.percentiles),
        None => {
            let all_samples = {
                let mut v = left_samples.clone();
                v.extend_from_slice(right_samples);
                v
            };
            let pcts = prepare_percentiles(&all_samples);
            let tests = vec![CtTest::default(); 101];

            (tests, pcts)
        }
    };

    let left_samples: Vec<f64> = left_samples.iter().map(|&n| n as f64).collect();
    let right_samples: Vec<f64> = right_samples.iter().map(|&n| n as f64).collect();

    for &left_sample in left_samples.iter() {
        update_test_left(&mut tests[0], left_sample);
    }
    for &right_sample in right_samples.iter() {
        update_test_right(&mut tests[0], right_sample);
    }

    for (test, &pct) in tests.iter_mut().skip(1).zip(percentiles.iter()) {
        let left_cropped = left_samples.iter().filter(|&&x| x < pct);
        let right_cropped = right_samples.iter().filter(|&&x| x < pct);

        for &left_sample in left_cropped {
            update_test_left(test, left_sample);
        }
        for &right_sample in right_cropped {
            update_test_right(test, right_sample);
        }
    }

    let (max_t, max_tau, sample_size) = {
        // Get the test with the maximum t
        let max_test = tests
            .iter()
            .max_by(|&x, &y| local_cmp(compute_t(x).abs(), compute_t(y).abs()))
            .unwrap();
        let sample_size = max_test.sizes.0 + max_test.sizes.1;
        let max_t = compute_t(max_test);
        let max_tau = max_t / (sample_size as f64).sqrt();

        (max_t, max_tau, sample_size)
    };

    let new_ctx = CtCtx { tests, percentiles };
    let summ = CtSummary {
        max_t,
        max_tau,
        sample_size,
    };

    (summ, new_ctx)
}

fn compute_t(test: &CtTest) -> f64 {
    let &CtTest {
        means,
        sq_diffs,
        sizes,
    } = test;
    let num = means.0 - means.1;
    let n0 = sizes.0 as f64;
    let n1 = sizes.1 as f64;
    let var0 = sq_diffs.0 / (n0 - 1f64);
    let var1 = sq_diffs.1 / (n1 - 1f64);
    let den = (var0 / n0 + var1 / n1).sqrt();

    num / den
}

fn update_test_left(test: &mut CtTest, datum: f64) {
    test.sizes.0 += 1;
    let diff = datum - test.means.0;
    test.means.0 += diff / (test.sizes.0 as f64);
    test.sq_diffs.0 += diff * (datum - test.means.0);
}

fn update_test_right(test: &mut CtTest, datum: f64) {
    test.sizes.1 += 1;
    let diff = datum - test.means.1;
    test.means.1 += diff / (test.sizes.1 as f64);
    test.sq_diffs.1 += diff * (datum - test.means.1);
}
