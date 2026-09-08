use crate::stats;

use std::{
    fs::{File, OpenOptions},
    hint::black_box,
    io::{self, BufWriter, Write},
    path::PathBuf,
    process,
    sync::{
        Arc,
        atomic::{self, AtomicBool},
    },
    time::Instant,
};

use ctrlc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

/// Just a static str representing the name of a function
#[derive(Copy, Clone)]
pub struct BenchName(pub &'static str);

impl BenchName {
    fn padded(&self, column_count: usize) -> String {
        let mut name = self.0.to_string();
        let pad_len = column_count.saturating_sub(name.len());
        let pad = " ".repeat(pad_len);
        name.push_str(&pad);

        name
    }
}

/// A random number generator implementing [`rand::SeedableRng`]. This is given to every
/// benchmarking function to use as a source of randomness.
pub type BenchRng = ChaChaRng;

/// A function that is to be benchmarked. This crate only supports statically-defined functions.
pub type BenchFn = fn(&mut CtRunner, &mut BenchRng);

// TODO: Consider giving this a lifetime so we don't have to copy names and vecs into it
#[derive(Clone)]
enum BenchEvent {
    ContStart,
    Begin(Vec<BenchName>),
    Wait(BenchName),
    Result(MonitorMsg),
    Seed(u64, BenchName),
}

type MonitorMsg = (BenchName, stats::CtSummary);

/// CtBencher is the primary interface for benchmarking. All setup for function inputs should be
/// doen within the closure supplied to the `iter` method.
struct CtBencher {
    samples: (Vec<u64>, Vec<u64>),
    order: Vec<Class>,
    next_sequence: usize,
    ctx: Option<stats::CtCtx>,
    file_out: Option<BufWriter<File>>,
    rng: BenchRng,
}

impl CtBencher {
    /// Creates and returns a new empty `CtBencher` whose `BenchRng` is zero-seeded
    pub fn new() -> CtBencher {
        CtBencher {
            samples: (Vec::new(), Vec::new()),
            order: Vec::new(),
            next_sequence: 0,
            ctx: None,
            file_out: None,
            rng: BenchRng::seed_from_u64(0u64),
        }
    }

    /// Runs the bench function and returns the CtSummary
    fn go(&mut self, f: BenchFn) -> stats::CtSummary {
        // This populates self.samples
        let mut runner = CtRunner::default();
        f(&mut runner, &mut self.rng);
        self.samples = runner.runtimes;
        self.order = runner.order;

        // Replace the old CtCtx with an updated one
        let old_self = ::std::mem::replace(self, CtBencher::new());
        let (summ, new_ctx) = stats::update_ct_stats(old_self.ctx, &old_self.samples);

        // Copy the old stuff back in
        self.samples = old_self.samples;
        self.order = old_self.order;
        self.next_sequence = old_self.next_sequence;
        self.file_out = old_self.file_out;
        self.ctx = Some(new_ctx);
        self.rng = old_self.rng;

        summ
    }

    /// Returns a random seed
    fn rand_seed() -> u64 {
        rand::rng().next_u64()
    }

    /// Reseeds the internal RNG with the given seed
    pub fn seed_with(&mut self, seed: u64) {
        self.rng = BenchRng::seed_from_u64(seed);
    }

    /// Clears out all sample and contextual data
    fn clear_data(&mut self) {
        self.samples = (Vec::new(), Vec::new());
        self.order.clear();
        self.next_sequence = 0;
        self.ctx = None;
    }
}

/// Represents a single benchmark to conduct
pub struct BenchMetadata {
    pub name: BenchName,
    pub seed: Option<u64>,
    pub benchfn: BenchFn,
}

/// Benchmarking options.
///
/// When `continuous` is set, it will continuously set the first (alphabetically) of the benchmarks
/// after they have been optionally filtered.
///
/// When `filter` is set and `continuous` is not set, only benchmarks whose names contain the
/// filter string as a substring will be executed.
///
/// `file_out` is optionally the filename where CSV output of raw runtime data should be written
#[derive(Default)]
pub struct BenchOpts {
    pub continuous: bool,
    pub filter: Option<String>,
    pub file_out: Option<PathBuf>,
}

struct ConsoleBenchState {
    // Number of columns to fill when aligning names
    max_name_len: usize,
}

impl ConsoleBenchState {
    fn write_plain(&mut self, s: &str) -> io::Result<()> {
        let mut stdout = io::stdout();
        stdout.write_all(s.as_bytes())?;
        stdout.flush()
    }

    fn write_bench_start(&mut self, name: &BenchName) -> io::Result<()> {
        let name = name.padded(self.max_name_len);
        self.write_plain(&format!("bench {} ... ", name))
    }

    fn write_seed(&mut self, seed: u64, name: &BenchName) -> io::Result<()> {
        let name = name.padded(self.max_name_len);
        self.write_plain(&format!("bench {} seeded with 0x{:016x}\n", name, seed))
    }

    fn write_run_start(&mut self, len: usize) -> io::Result<()> {
        let noun = if len != 1 { "benches" } else { "bench" };
        self.write_plain(&format!("\nrunning {} {}\n", len, noun))
    }

    fn write_continuous_start(&mut self) -> io::Result<()> {
        self.write_plain("running 1 benchmark continuously\n")
    }

    fn write_result(&mut self, summ: &stats::CtSummary) -> io::Result<()> {
        self.write_plain(&format!(": {}\n", summ.fmt()))
    }

    fn write_run_finish(&mut self) -> io::Result<()> {
        self.write_plain("\ndudect benches complete\n\n")
    }
}

/// Runs the given benches under the given options and prints the output to the console
pub fn run_benches_console(opts: BenchOpts, benches: Vec<BenchMetadata>) -> io::Result<()> {
    // TODO: Consider making this do screen updates in continuous mode
    // TODO: Consider making this run in its own thread
    fn callback(event: &BenchEvent, st: &mut ConsoleBenchState) -> io::Result<()> {
        match (*event).clone() {
            BenchEvent::ContStart => st.write_continuous_start(),
            BenchEvent::Begin(ref filtered_benches) => st.write_run_start(filtered_benches.len()),
            BenchEvent::Wait(ref b) => st.write_bench_start(b),
            BenchEvent::Result(msg) => {
                let (_, summ) = msg;
                st.write_result(&summ)
            }
            BenchEvent::Seed(seed, ref name) => st.write_seed(seed, name),
        }
    }

    let mut st = ConsoleBenchState {
        max_name_len: benches.iter().map(|t| t.name.0.len()).max().unwrap_or(0),
    };

    run_benches(&opts, benches, |x| callback(&x, &mut st))?;
    st.write_run_finish()
}

/// Returns an atomic bool that indicates whether Ctrl-C was pressed
fn setup_kill_bit() -> Arc<AtomicBool> {
    let x = Arc::new(AtomicBool::new(false));
    let y = x.clone();

    ctrlc::set_handler(move || y.store(true, atomic::Ordering::SeqCst))
        .expect("Error setting Ctrl-C handler");

    x
}

fn run_benches<F>(opts: &BenchOpts, benches: Vec<BenchMetadata>, mut callback: F) -> io::Result<()>
where
    F: FnMut(BenchEvent) -> io::Result<()>,
{
    let filter = &opts.filter;
    let filtered_benches = filter_benches(filter, benches);
    let filtered_names = filtered_benches.iter().map(|b| b.name).collect();

    // Write the CSV header line to the file if the file is defined
    let mut file_out = opts.file_out.as_ref().map(|filename| {
        OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(filename)
            .unwrap_or_else(|e| panic!("Could not open file '{:?}' for writing: {e}", filename))
    });
    let mut file_out = file_out.take().map(BufWriter::new);
    if let Some(f) = file_out.as_mut() {
        f.write_all(b"benchname,sequence,class,runtime_ns\n")
            .expect("Error writing CSV header to file");
    }

    // Make a bencher with the optional file output specified
    let mut cb: CtBencher = {
        let mut d = CtBencher::new();
        d.file_out = file_out;
        d
    };

    if opts.continuous {
        callback(BenchEvent::ContStart)?;

        if filtered_benches.is_empty() {
            match *filter {
                Some(ref f) => panic!("No benchmark matching '{}' was found", f),
                None => return Ok(()),
            }
        }

        // Get a bit that tells us when we've been killed
        let kill_bit = setup_kill_bit();

        // Continuously run the first matched bench we see
        let mut filtered_benches = filtered_benches;
        let bench = filtered_benches.remove(0);

        // If a seed was specified for this bench, use it. Otherwise, use a random seed
        let seed = bench.seed.unwrap_or_else(CtBencher::rand_seed);
        cb.seed_with(seed);
        callback(BenchEvent::Seed(seed, bench.name))?;

        loop {
            callback(BenchEvent::Wait(bench.name))?;
            let msg = run_bench_with_bencher(&bench.name, bench.benchfn, &mut cb);
            callback(BenchEvent::Result(msg))?;

            // Check if the program has been killed. If so, exit
            if kill_bit.load(atomic::Ordering::SeqCst) {
                process::exit(0);
            }
        }
    } else {
        callback(BenchEvent::Begin(filtered_names))?;

        // Run different benches
        for bench in filtered_benches {
            // Clear the data out from the previous bench, but keep the CSV file open
            cb.clear_data();

            // If a seed was specified for this bench, use it. Otherwise, use a random seed
            let seed = bench.seed.unwrap_or_else(CtBencher::rand_seed);
            cb.seed_with(seed);
            callback(BenchEvent::Seed(seed, bench.name))?;

            callback(BenchEvent::Wait(bench.name))?;
            let msg = run_bench_with_bencher(&bench.name, bench.benchfn, &mut cb);
            callback(BenchEvent::Result(msg))?;
        }
        Ok(())
    }
}

fn run_bench_with_bencher(name: &BenchName, benchfn: BenchFn, cb: &mut CtBencher) -> MonitorMsg {
    let summ = cb.go(benchfn);

    // Write the runtime samples out
    if let Some(f) = cb.file_out.as_mut() {
        write_samples(f, name, &cb.samples, &cb.order, &mut cb.next_sequence)
            .expect("Error writing data to file");
        f.flush().expect("Error flushing data to file");
    }

    (*name, summ)
}

fn write_samples(
    out: &mut impl Write,
    name: &BenchName,
    samples: &(Vec<u64>, Vec<u64>),
    order: &[Class],
    next_sequence: &mut usize,
) -> io::Result<()> {
    let mut left = samples.0.iter();
    let mut right = samples.1.iter();
    for class in order {
        let (label, runtime) = match class {
            Class::Left => (0, left.next()),
            Class::Right => (1, right.next()),
        };
        let runtime = runtime.expect("sample order must match recorded runtimes");
        writeln!(out, "{},{},{},{}", name.0, *next_sequence, label, runtime)?;
        *next_sequence = next_sequence
            .checked_add(1)
            .expect("sample sequence overflow");
    }
    assert!(left.next().is_none() && right.next().is_none());
    Ok(())
}

fn filter_benches(filter: &Option<String>, bs: Vec<BenchMetadata>) -> Vec<BenchMetadata> {
    let mut filtered = bs;

    // Remove benches that don't match the filter
    filtered = match *filter {
        None => filtered,
        Some(ref filter) => filtered
            .into_iter()
            .filter(|b| b.name.0.contains(&filter[..]))
            .collect(),
    };

    // Sort them alphabetically
    filtered.sort_by(|b1, b2| b1.name.0.cmp(b2.name.0));

    filtered
}

/// Specifies the distribution that a particular run belongs to
#[derive(Copy, Clone)]
pub enum Class {
    Left,
    Right,
}

/// Used for timing single operations at a time
#[derive(Default)]
pub struct CtRunner {
    // Runtimes of left and right distributions in nanoseconds
    runtimes: (Vec<u64>, Vec<u64>),
    order: Vec<Class>,
}

impl CtRunner {
    /// Runs and times a single operation whose constant-timeness is in question
    pub fn run_one<T, F>(&mut self, class: Class, f: F)
    where
        F: Fn() -> T,
    {
        let start = Instant::now();
        black_box(f());
        let end = Instant::now();

        let runtime = {
            let dur = end.duration_since(start);
            dur.as_secs() * 1_000_000_000 + u64::from(dur.subsec_nanos())
        };

        self.record(class, runtime);
    }

    // Both consumers retain the same duration; ordering work stays outside the timed closure.
    fn record(&mut self, class: Class, runtime: u64) {
        match class {
            Class::Left => self.runtimes.0.push(runtime),
            Class::Right => self.runtimes.1.push(runtime),
        }
        self.order.push(class);
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;

    #[test]
    fn unequal_classes_preserve_every_duration_and_execution_order() {
        let mut runner = CtRunner::default();
        for (class, runtime) in [
            (Class::Right, 31),
            (Class::Left, 11),
            (Class::Right, 32),
            (Class::Right, 33),
            (Class::Left, 12),
        ] {
            runner.record(class, runtime);
        }
        let mut out = Vec::new();
        let mut sequence = 0;
        write_samples(
            &mut out,
            &BenchName("unequal"),
            &runner.runtimes,
            &runner.order,
            &mut sequence,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "unequal,0,1,31\nunequal,1,0,11\nunequal,2,1,32\nunequal,3,1,33\nunequal,4,0,12\n"
        );
        assert_eq!(sequence, 5);

        let mut out = Vec::new();
        write_samples(
            &mut out,
            &BenchName("unequal"),
            &(vec![41, 42], vec![]),
            &[Class::Left, Class::Left],
            &mut sequence,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "unequal,5,0,41\nunequal,6,0,42\n"
        );
    }

    #[test]
    fn timed_samples_roundtrip_without_changing_statistics() {
        fn bench(runner: &mut CtRunner, _: &mut BenchRng) {
            for i in 0..101 {
                let class = if i % 3 == 0 {
                    Class::Left
                } else {
                    Class::Right
                };
                runner.run_one(class, || black_box(i));
            }
        }
        let mut cb = CtBencher::new();
        let summary = cb.go(bench);
        assert_eq!(cb.samples.0.len(), 34);
        assert_eq!(cb.samples.1.len(), 67);
        let mut out = Vec::new();
        write_samples(
            &mut out,
            &BenchName("timed"),
            &cb.samples,
            &cb.order,
            &mut cb.next_sequence,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        let mut reconstructed = (Vec::new(), Vec::new());
        for (index, line) in text.lines().enumerate() {
            let fields: Vec<_> = line.split(',').collect();
            assert_eq!(fields[0], "timed");
            assert_eq!(fields[1].parse::<usize>().unwrap(), index);
            let expected_label = if index % 3 == 0 { "0" } else { "1" };
            assert_eq!(fields[2], expected_label);
            let runtime = fields[3].parse::<u64>().unwrap();
            if fields[2] == "0" {
                reconstructed.0.push(runtime);
            } else {
                reconstructed.1.push(runtime);
            }
        }
        assert_eq!(reconstructed, cb.samples);
        let (replayed, _) = stats::update_ct_stats(None, &reconstructed);
        assert_eq!(replayed.sample_size, summary.sample_size);
        assert_eq!(replayed.max_t.to_bits(), summary.max_t.to_bits());
        assert_eq!(replayed.max_tau.to_bits(), summary.max_tau.to_bits());
        cb.go(bench);
        assert_eq!(cb.next_sequence, 101);
        cb.clear_data();
        assert_eq!(cb.next_sequence, 0);
        assert!(cb.order.is_empty());
    }
}
