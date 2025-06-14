//! Quick performance benchmarks for TrString system
//!
//! Lightweight benchmarks to quickly measure TrString performance

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tribute_runtime::memory::*;
use tribute_runtime::string_ops::*;

fn bench_string_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_modes");

    // Test the three different string modes
    group.bench_function("inline_string_5_bytes", |b| {
        b.iter(|| {
            let handle = tr_value_from_string(black_box("hello".as_ptr()), black_box(5));
            tr_value_free(black_box(handle));
        });
    });

    group.bench_function("heap_string_20_bytes", |b| {
        b.iter(|| {
            let handle =
                tr_value_from_string(black_box("this is a long string".as_ptr()), black_box(20));
            tr_value_free(black_box(handle));
        });
    });

    group.bench_function("static_string", |b| {
        b.iter(|| {
            let handle = tr_value_from_static_string(black_box(0), black_box(10));
            tr_value_free(black_box(handle));
        });
    });

    group.finish();
}

fn bench_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");

    // Create test strings
    let hello = tr_value_from_string("hello".as_ptr(), 5);
    let world = tr_value_from_string("world".as_ptr(), 5);

    group.bench_function("concat_short_strings", |b| {
        b.iter(|| {
            let result = tr_string_concat(black_box(hello), black_box(world));
            tr_value_free(black_box(result));
        });
    });

    group.bench_function("string_length", |b| {
        b.iter(|| {
            let len = tr_string_length(black_box(hello));
            black_box(len);
        });
    });

    group.bench_function("string_equality", |b| {
        b.iter(|| {
            let equal = tr_value_equals(black_box(hello), black_box(world));
            black_box(equal);
        });
    });

    // Cleanup
    tr_value_free(hello);
    tr_value_free(world);

    group.finish();
}

criterion_group!(benches, bench_string_modes, bench_string_operations);
criterion_main!(benches);
