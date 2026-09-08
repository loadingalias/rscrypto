// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc = include_str!("../README.md")]

// TODO: More comments
// TODO: Do "higher order preprocessing" from the paper

pub mod ctbench;
#[doc(hidden)]
pub mod macros;
mod stats;

// Re-export the rand dependency
pub use rand;

#[doc(inline)]
pub use ctbench::{BenchRng, Class, CtRunner};
