# dudect-bencher
[![Version](https://img.shields.io/crates/v/dudect-bencher.svg)](https://crates.io/crates/dudect-bencher)
[![Docs](https://docs.rs/dudect-bencher/badge.svg)](https://docs.rs/dudect-bencher)

This crate implements the [DudeCT](https://eprint.iacr.org/2016/1123.pdf) statistical methods for testing whether functions are constant-time. It is based loosely off of the [`bencher`](https://github.com/bluss/bencher) benchmarking framework.

In general, it is not possible to prove that a function always runs in constant time. The purpose of this tool is to find non-constant-timeness when it exists. This is not easy, and it requires the user to think very hard about where the non-constant-timeness might be.

# Import and features

To import this crate, put the following line in your `Cargo.toml`:
```toml
dudect-bencher = "0.7"
```

# Usage

This framework builds a standalone binary. So you must define a `main.rs`, or a file in your `src/bin` directory, or a separate binary crate that pulls in the library you want to test.

At a high, level you test a function `f` by first defining two sets inputs to `f`, called Right and Left. The way you pick these is highly subjective. You need to already have an idea of what might cause non-constant-time behavior. You then fill in the Left and Right sets such that (you think) `f(l)` and `f(r)` will take a different amount of time to run, on average, where `l` comes from Left and `r` from Right. Finally, you run the benchmarks and label which set is which.

Here is an example of testing the equality function `v == u` where `v` and `u` are `Vec<u8>` of the same length. This is clearly not a constant time function. We define the left distribution to be a set of `(v, u)` where `v == u`, and the right distribution to be the set of `(v, u)` where `v[6] != u[6]`.

```rust
use dudect_bencher::{ctbench_main, BenchRng, Class, CtRunner};
use rand::{Rng, RngExt};

// Return a random vector of length len
fn rand_vec(len: usize, rng: &mut BenchRng) -> Vec<u8> {
    let mut arr = vec![0u8; len];
    rng.fill(arr.as_mut_slice());
    arr
}

// Benchmark for equality of vectors. This does an early return when it finds an
// inequality, so it should be very much not constant-time
fn vec_eq(runner: &mut CtRunner, rng: &mut BenchRng) {
    // Make vectors of size 100
    let vlen = 100;
    let mut inputs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut classes = Vec::new();

    // Make 100,000 random pairs of vectors
    for _ in 0..100_000 {
        // Flip a coin. If true, make a pair of vectors that are equal to each
        // other and put it in the Left distribution
        if rng.random::<bool>() {
            let v1 = rand_vec(vlen, rng);
            let v2 = v1.clone();
            inputs.push((v1, v2));
            classes.push(Class::Left);
        }
        // Otherwise, make a pair of vectors that differ at the 6th element and
        // put it in the right distribution
        else {
            let v1 = rand_vec(vlen, rng);
            let mut v2 = v1.clone();
            v2[5] = 7;
            inputs.push((v1, v2));
            classes.push(Class::Right);
        }
    }

    for (class, (u, v)) in classes.into_iter().zip(inputs.into_iter()) {
        // Now time how long it takes to do a vector comparison
        runner.run_one(class, || u == v);
    }
}

// Crate the main function to include the bench for vec_eq
ctbench_main!(vec_eq);
```

This is a portion of the example code in [`examples/ctbench-foo.rs`](examples/). To run the example, run

```shell
cargo run --release --example ctbench-foo
```

See more command line arguments [below](#command-line-arguments)

## Bencher output

The program output looks like

```ignore
bench array_eq ... : n == +0.046M, max t = +61.61472, max tau = +0.28863, (5/tau)^2 = 300
```

It is interpreted as follows. Firstly note that the runtime distributions are cropped at different percentiles and about 100 t-tests are performed. Of these t-tests, the one that produces the largest absolute t-value is printed as `max_t`. The other values printed are

 * `n`, indicating the number of samples used in computing this t-value
 * `max_tau`, which is the t-value scaled for the samples size (formally, `max_tau = max_t / sqrt(n)`)
 * `(5/tau)^2`, which indicates the number of measurements that would be needed to distinguish the two distributions with t > 5

t-values greater than 5 are generally considered a good indication that the function is not constant time. t-values less than 5 does not necessarily imply that the function is constant-time, since there may be other input distributions under which the function behaves significantly differently.

## Command line arguments

* `--filter` runs a subset of the benchmarks whose name contains a specific string. Example:
```shell
cargo run --release --example ctbench-foo -- --filter ar
```
will run only the benchmarks with the substring `ar` in it, i.e., `arith`, and not `vec_eq`.
* `--continuous` run a benchmark continuously, collecting more samples as it goes along. Example:
```shell
cargo run --release --example ctbench-foo -- --continuous vec_eq
```
will run the `vec_eq` benchmark continuously.

* `--out` outputs raw runtimes in CSV format. Example:
```shell
cargo run --release --example ctbench-foo -- --out data.csv
```
will output all the benchmarks in `ctbench-foo.rs` to `data.csv`.

# MSRV

The current minimum supported Rust version (MSRV) is 1.85.0 (2025-02-20).

# Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes made throughout past versions.

# License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE))
 * MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
