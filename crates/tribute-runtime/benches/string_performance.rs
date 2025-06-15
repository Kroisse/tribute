//! Performance benchmarks for TrString system
//!
//! Tests the performance of all three TrString modes:
//! - Inline: strings ≤ 15 bytes
//! - Static: compile-time strings in .rodata
//! - Heap: runtime strings in AllocationTable

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tribute_runtime::memory::*;
use tribute_runtime::string_ops::*;

fn bench_string_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_creation");

    // Test different string lengths to trigger different modes
    let test_cases = vec![
        ("short_5", "hello"),                   // Inline mode
        ("medium_10", "hello world"),           // Inline mode
        ("exact_15", "exactly_15_char"),        // Inline mode boundary
        ("long_20", "this is a longer string"), // Heap mode
        (
            "very_long_50",
            "this is a much longer string that definitely goes to heap allocation table",
        ), // Heap mode
    ];

    for (name, text) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("inline_or_heap", name),
            &text,
            |b, text| {
                b.iter(|| {
                    let handle =
                        tr_value_from_string(black_box(text.as_ptr()), black_box(text.len()));
                    tr_value_free(black_box(handle));
                });
            },
        );

        // Test static string creation (for comparison)
        group.bench_with_input(BenchmarkId::new("static", name), &text, |b, text| {
            b.iter(|| {
                // Static strings use offset/length (simulated here)
                let handle =
                    tr_value_from_static_string(black_box(0), black_box(text.len() as u32));
                tr_value_free(black_box(handle));
            });
        });
    }

    group.finish();
}

fn bench_string_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_access");

    // Create test strings in different modes
    let short_handle = tr_value_from_string("hello".as_ptr(), 5); // Inline
    let long_handle = tr_value_from_string("this is a long string".as_ptr(), 20); // Heap
    let static_handle = tr_value_from_static_string(0, 10); // Static

    group.bench_function("inline_access", |b| {
        b.iter(|| {
            let mut len = 0usize;
            let ptr = tr_string_as_ptr(black_box(short_handle), &mut len);
            let total_len = tr_string_length(black_box(short_handle));
            black_box((ptr, len, total_len));
        });
    });

    group.bench_function("heap_access", |b| {
        b.iter(|| {
            let mut len = 0usize;
            let ptr = tr_string_as_ptr(black_box(long_handle), &mut len);
            let total_len = tr_string_length(black_box(long_handle));
            black_box((ptr, len, total_len));
        });
    });

    group.bench_function("static_access", |b| {
        b.iter(|| {
            let mut len = 0usize;
            let ptr = tr_string_as_ptr(black_box(static_handle), &mut len);
            let total_len = tr_string_length(black_box(static_handle));
            black_box((ptr, len, total_len));
        });
    });

    // Cleanup
    tr_value_free(short_handle);
    tr_value_free(long_handle);
    tr_value_free(static_handle);

    group.finish();
}

fn bench_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");

    // Test string concatenation performance
    let hello_handle = tr_value_from_string("hello".as_ptr(), 5);
    let world_handle = tr_value_from_string("world".as_ptr(), 5);

    group.bench_function("concat_inline", |b| {
        b.iter(|| {
            let result = tr_string_concat(black_box(hello_handle), black_box(world_handle));
            tr_value_free(black_box(result));
        });
    });

    // Test equality comparison
    let hello2_handle = tr_value_from_string("hello".as_ptr(), 5);

    group.bench_function("equals_inline", |b| {
        b.iter(|| {
            let result = tr_value_equals(black_box(hello_handle), black_box(hello2_handle));
            black_box(result);
        });
    });

    // Cleanup
    tr_value_free(hello_handle);
    tr_value_free(world_handle);
    tr_value_free(hello2_handle);

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Compare memory allocation patterns
    group.bench_function("create_1000_short_strings", |b| {
        b.iter(|| {
            let mut handles = Vec::new();
            for i in 0..1000 {
                let text = format!("str{}", i % 100); // Mix of short strings
                let handle = tr_value_from_string(text.as_ptr(), text.len());
                handles.push(handle);
            }
            // Cleanup
            for handle in handles {
                tr_value_free(handle);
            }
        });
    });

    group.bench_function("create_1000_long_strings", |b| {
        b.iter(|| {
            let mut handles = Vec::new();
            for i in 0..1000 {
                let text = format!("this is a longer string number {} for testing", i);
                let handle = tr_value_from_string(text.as_ptr(), text.len());
                handles.push(handle);
            }
            // Cleanup
            for handle in handles {
                tr_value_free(handle);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_string_creation,
    bench_string_access,
    bench_string_operations,
    bench_memory_usage
);
criterion_main!(benches);
