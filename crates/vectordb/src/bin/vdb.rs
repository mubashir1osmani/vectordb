//! Recall and latency harness.
//!
//! ```text
//! vdb bench --n 5000 --dim 64 --queries 100 --k 10 --index hnsw --metric l2
//! ```
//!
//! The flat index is the ground truth. Report recall@k plus p50 and p99
//! search latency for the index under test.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vectordb::{
    percentile, recall_at_k, Engine, EngineConfig, HnswParams, IndexParams, Metric, SyncPolicy,
};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("bench") => match bench(&args.collect::<Vec<_>>()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::from(1)
            }
        },
        Some("help") | Some("-h") | Some("--help") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(cmd) => {
            eprintln!("unknown command `{cmd}`");
            print_help();
            ExitCode::from(2)
        }
    }
}

fn print_help() {
    eprintln!(
        "\
vectordb {version}

Usage:
  vdb bench [--n N] [--dim D] [--queries Q] [--k K]
            [--index flat|hnsw] [--metric l2|cosine|ip]
            [--m M] [--ef-construction N] [--ef-search N] [--seed N]

Builds an in-memory index, measures search latency, and scores recall
against an exact flat scan of the same vectors.
",
        version = env!("CARGO_PKG_VERSION")
    );
}

struct BenchOpts {
    n: usize,
    dim: usize,
    queries: usize,
    k: usize,
    index: String,
    metric: Metric,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
}

fn bench(args: &[String]) -> Result<(), String> {
    let opts = parse_opts(args)?;
    if opts.k == 0 {
        return Err("--k must be >= 1".into());
    }
    if opts.queries == 0 || opts.n == 0 {
        return Err("--n and --queries must be >= 1".into());
    }

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut vectors = Vec::with_capacity(opts.n);
    for _ in 0..opts.n {
        let vector: Vec<f32> = (0..opts.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        vectors.push(vector);
    }
    let queries: Vec<Vec<f32>> = (0..opts.queries)
        .map(|_| (0..opts.dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let hnsw_params = HnswParams {
        m: opts.m,
        ef_construction: opts.ef_construction,
        ef_search: opts.ef_search,
        seed: opts.seed,
    };
    let config = match opts.index.as_str() {
        "flat" => EngineConfig {
            sync: SyncPolicy::Never,
            ..EngineConfig::flat(opts.dim, opts.metric)
        },
        "hnsw" => EngineConfig {
            sync: SyncPolicy::Never,
            ..EngineConfig::hnsw(opts.dim, opts.metric, hnsw_params)
        },
        other => return Err(format!("unknown --index `{other}` (use flat or hnsw)")),
    };
    if let IndexParams::Hnsw(params) = &config.index {
        params.validate()?;
    }

    let build_start = Instant::now();
    let mut engine = Engine::open_memory(config).map_err(|e| e.to_string())?;
    for (id, vector) in vectors.iter().enumerate() {
        engine
            .upsert(id as u64, vector)
            .map_err(|e| e.to_string())?;
    }
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    // Ground truth always comes from an exact scan, even when the index under
    // test is itself flat (recall should then be 1).
    let mut flat = Engine::open_memory(EngineConfig::flat(opts.dim, opts.metric))
        .map_err(|e| e.to_string())?;
    for (id, vector) in vectors.iter().enumerate() {
        flat.upsert(id as u64, vector).map_err(|e| e.to_string())?;
    }

    let mut latencies = Vec::with_capacity(opts.queries);
    let mut recall = 0.0f32;
    for query in &queries {
        let truth: Vec<u64> = flat
            .search(query, opts.k)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|hit| hit.id)
            .collect();
        let started = Instant::now();
        let approx: Vec<u64> = engine
            .search(query, opts.k)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|hit| hit.id)
            .collect();
        latencies.push(started.elapsed().as_micros());
        recall += recall_at_k(&truth, &approx);
    }
    recall /= opts.queries as f32;
    latencies.sort_unstable();
    let p50 = percentile(&latencies, 50.0);
    let p99 = percentile(&latencies, 99.0);

    println!("index            {}", opts.index);
    println!("metric           {}", opts.metric.name());
    println!("n                {}", opts.n);
    println!("dim              {}", opts.dim);
    println!("queries          {}", opts.queries);
    println!("k                {}", opts.k);
    if opts.index == "hnsw" {
        println!("m                {}", opts.m);
        println!("ef_construction  {}", opts.ef_construction);
        println!("ef_search        {}", opts.ef_search);
    }
    println!("build_ms         {build_ms:.2}");
    println!("recall@{:<10} {:.4}", opts.k, recall);
    println!("p50_us           {p50}");
    println!("p99_us           {p99}");
    Ok(())
}

fn parse_opts(args: &[String]) -> Result<BenchOpts, String> {
    let mut opts = BenchOpts {
        n: 2_000,
        dim: 64,
        queries: 50,
        k: 10,
        index: "hnsw".into(),
        metric: Metric::L2,
        m: 16,
        ef_construction: 200,
        ef_search: 128,
        seed: 1,
    };
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("{key} needs a value"))?;
        match key {
            "--n" => opts.n = parse_usize(value, key)?,
            "--dim" => opts.dim = parse_usize(value, key)?,
            "--queries" => opts.queries = parse_usize(value, key)?,
            "--k" => opts.k = parse_usize(value, key)?,
            "--m" => opts.m = parse_usize(value, key)?,
            "--ef-construction" => opts.ef_construction = parse_usize(value, key)?,
            "--ef-search" => opts.ef_search = parse_usize(value, key)?,
            "--seed" => opts.seed = parse_u64(value, key)?,
            "--index" => opts.index = value.clone(),
            "--metric" => {
                opts.metric =
                    Metric::parse(value).ok_or_else(|| format!("unknown metric `{value}`"))?;
            }
            other => return Err(format!("unknown flag `{other}`")),
        }
        i += 2;
    }
    Ok(opts)
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an integer, got `{value}`"))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an integer, got `{value}`"))
}
