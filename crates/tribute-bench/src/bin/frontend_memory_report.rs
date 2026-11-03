//! Size and copy time of the frontend's whole-module values, next to the
//! per-item values they duplicate.
//!
//! ```text
//! cargo run --release -p tribute-bench --bin frontend_memory_report
//! ```
//!
//! A value's size is the heap its deep clone allocates, so it counts the
//! vectors, strings, and maps the value owns. Interned Salsa values and
//! reference-counted symbols are shared, so their storage is not counted.
//! The copy time is the mean time of that clone, an estimate of what
//! assembling or merging the value costs on each edit. The span map, which
//! has no `Clone`, is measured by merging the items' spans again, as each
//! parse does.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

use tribute_bench::editing::{EditSession, workloads};
use tribute_front::analysis;
use tribute_front::astgen::{ParsedItem, lower_source_to_parsed_ast};

/// Tracks the bytes currently allocated.
struct LiveAllocator;

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for LiveAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE_BYTES.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: LiveAllocator = LiveAllocator;

/// The heap a value `make` builds holds, and the mean time `make` takes.
fn measure<T>(make: impl Fn() -> T) -> (isize, f64) {
    let before = LIVE_BYTES.load(Ordering::Relaxed);
    let value = make();
    let bytes = LIVE_BYTES.load(Ordering::Relaxed) - before;
    drop(value);
    const RUNS: u32 = 200;
    let start = Instant::now();
    for _ in 0..RUNS {
        black_box(make());
    }
    (bytes, start.elapsed().as_secs_f64() * 1e6 / f64::from(RUNS))
}

fn row(name: &str, (bytes, micros): (isize, f64)) {
    println!(
        "| {name} | {:.1} KiB | {micros:.1} µs |",
        bytes as f64 / 1024.0
    );
}

/// `items` and the items of their inline modules.
fn all_items<'db>(db: &'db dyn salsa::Database, items: &[ParsedItem<'db>]) -> Vec<ParsedItem<'db>> {
    items
        .iter()
        .flat_map(|&item| std::iter::once(item).chain(all_items(db, item.children(db))))
        .collect()
}

fn main() {
    for workload in workloads() {
        let edit = workload
            .edits
            .iter()
            .find(|edit| edit.name == "body")
            .expect("every workload has a body edit");
        let session = EditSession::new(edit);
        let db: &dyn salsa::Database = session.db();
        let source = session.source();
        let table = tribute::root_environment(db, source);
        let typed = tribute::parse_and_lower_ast(db, source).expect("the program parses");

        println!("\n### {}\n", workload.name);
        println!("| Value | Heap | Copy |");
        println!("|---|---:|---:|");

        let parsed = lower_source_to_parsed_ast(db, source).expect("the program parses");
        let items = all_items(db, parsed.items(db));
        row("`ParsedAst::module`", measure(|| parsed.module(db).clone()));
        row(
            "`ParsedItem::decls`, all items",
            measure(|| {
                items
                    .iter()
                    .map(|item| item.decls(db).clone())
                    .collect::<Vec<_>>()
            }),
        );
        row(
            "`ParsedAst::items_by_owner`",
            measure(|| parsed.items_by_owner(db).clone()),
        );

        let resolved = analysis::resolved_module(db, source, table).expect("the program resolves");
        row(
            "`ResolvedModule::module`",
            measure(|| resolved.module.clone()),
        );
        row(
            "`ResolvedFunction::decl`, all functions",
            measure(|| {
                resolved
                    .functions
                    .iter()
                    .map(|function| function.decl(db).clone())
                    .collect::<Vec<_>>()
            }),
        );

        row(
            "`TypeCheckOutput::module`",
            measure(|| typed.module(db).clone()),
        );
        row(
            "`TypeCheckOutput::expression_types`",
            measure(|| typed.expression_types(db).clone()),
        );
    }
}
