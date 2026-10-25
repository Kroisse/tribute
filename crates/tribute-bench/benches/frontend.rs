//! Frontend compile time: parse through specialization, from a fresh
//! database, for workloads that stress one frontend stage each.
//!
//! ```text
//! cargo bench -p tribute-bench --bench frontend -- --save-baseline <name>
//! cargo bench -p tribute-bench --bench frontend -- --baseline <name>
//! ```
//!
//! - `frontend/specialization/<types>`: a program instantiating a generic
//!   struct and three generic functions, one calling the others, at every
//!   one of `types` distinct structs.

use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

use tribute_bench::programs::{Program, WasmSupport};
use tribute_bench::stages;

/// The source of a program that instantiates the generic items at `types`
/// structs.
fn generic_instances_source(types: usize) -> String {
    let mut source = String::from(
        "struct Box(a) { value: a }\n\
         fn wrap(value: a) -> Box(a) { Box { value: value } }\n\
         fn unwrap(boxed: Box(a)) -> a { boxed.value }\n\
         fn round_trip(value: a) -> a { unwrap(wrap(unwrap(wrap(value)))) }\n",
    );
    for index in 0..types {
        source.push_str(&format!("struct T{index} {{ n: Int }}\n"));
    }
    source.push_str("fn main() -> Nil {\n");
    for index in 0..types {
        source.push_str(&format!(
            "    let _ = round_trip(T{index} {{ n: +{index} }})\n"
        ));
    }
    source.push_str("}\n");
    source
}

fn specialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("frontend/specialization");
    for types in [16, 64] {
        let program = Program {
            name: "generic_instances",
            source: String::leak(generic_instances_source(types)),
            stdin: b"",
            wasm: WasmSupport::Supported,
        };
        stages::frontend(&program).expect("the program compiles");
        group.bench_function(types.to_string(), |b| {
            b.iter_batched(
                || (),
                |()| stages::frontend(&program).expect("frontend"),
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = config();
    targets = specialization
}
criterion_main!(benches);
