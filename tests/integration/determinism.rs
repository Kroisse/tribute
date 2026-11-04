//! Output determinism (`new-plans/implementation.md`, 출력 결정성).
//!
//! The same input compiled in two fresh databases, or by two processes,
//! yields the same boundary-exit IR text for both targets, the same native
//! object, and the same Wasm module. Before its second compilation each
//! check perturbs what an order-dependent pass could read: the global symbol
//! interner and heap addresses in one process, and the environment's size
//! across processes. `TRIBUTE_DETERMINISM_REPEAT` repeats each comparison to
//! chase a difference that shows up rarely.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use itertools::Itertools;
use proptest::prelude::*;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::{
    CompilationConfig, OptimizationOptions, compile_to_native_binary, compile_to_wasm_binary,
    dump_ir,
};
use tribute_front::SourceCst;
use tribute_prop::program::program;
use trunk_ir::Symbol;

use crate::lang_examples::{EXPECTED_FAILURES, examples};

/// What one compilation produces. A target that fails to compile has no
/// artifact; its diagnostics are not compared.
struct Artifacts {
    native_ir: Option<String>,
    wasm_ir: Option<String>,
    native_object: Option<Vec<u8>>,
    wasm: Option<Vec<u8>>,
}

fn compile(db: &dyn salsa::Database, source: SourceCst) -> Artifacts {
    let config = CompilationConfig::new(db, false, OptimizationOptions::production());
    Artifacts {
        native_ir: dump_ir(db, source, true).ok().map(str::to_owned),
        wasm_ir: dump_ir(db, source, false).ok().map(str::to_owned),
        native_object: compile_to_native_binary(db, source, config).map(<[u8]>::to_vec),
        wasm: compile_to_wasm_binary(db, source).ok().map(<[u8]>::to_vec),
    }
}

fn compile_fresh(load: &impl Fn(&TributeDatabaseImpl) -> SourceCst) -> Artifacts {
    let db = TributeDatabaseImpl::default();
    let source = load(&db);
    db.attach(|db| compile(db, source))
}

/// The number of times each check compiles its input again: a positive
/// `TRIBUTE_DETERMINISM_REPEAT`, or 1.
fn repeat() -> usize {
    std::env::var("TRIBUTE_DETERMINISM_REPEAT")
        .ok()
        .and_then(|repeat| repeat.parse().ok())
        .filter(|&repeat| repeat > 0)
        .unwrap_or(1)
}

/// The artifacts an input must produce, so that a check never passes by
/// comparing two failed compilations.
#[derive(Clone, Copy)]
struct Required {
    ir: bool,
    native_object: bool,
    wasm_module: bool,
}

impl Required {
    const EVERYTHING: Self = Self {
        ir: true,
        native_object: true,
        wasm_module: true,
    };
}

/// Examples that lower for both targets but build no Wasm module.
const NO_WASM_MODULE: &[&str] = &[
    // `List` has no Wasm lowering.
    "let-destructuring.trb",
    // `std::io::read_line` has no Wasm lowering.
    "native_calculator.trb",
];

fn required(example: &str) -> Required {
    if EXPECTED_FAILURES.iter().any(|(name, _)| *name == example) {
        Required {
            ir: false,
            native_object: false,
            wasm_module: false,
        }
    } else {
        Required {
            wasm_module: !NO_WASM_MODULE.contains(&example),
            ..Required::EVERYTHING
        }
    }
}

/// The required artifacts a compilation did not produce. `dump_ir` gives
/// empty text when the frontend fails, so IR must also be nonempty.
fn missing(artifacts: &Artifacts, required: Required) -> Option<String> {
    let has_ir = |ir: &Option<String>| ir.as_deref().is_some_and(|ir| !ir.is_empty());
    let missing = [
        ("native IR", required.ir && !has_ir(&artifacts.native_ir)),
        ("Wasm IR", required.ir && !has_ir(&artifacts.wasm_ir)),
        (
            "native object",
            required.native_object && artifacts.native_object.is_none(),
        ),
        (
            "Wasm module",
            required.wasm_module && artifacts.wasm.is_none(),
        ),
    ]
    .into_iter()
    .filter_map(|(what, missing)| missing.then_some(what))
    .collect_vec();
    (!missing.is_empty()).then(|| format!("produced no {}", missing.iter().format(", ")))
}

/// Symbols and allocations that a later compilation runs beside, so the
/// interner's contents and the addresses it allocates differ from the first
/// compilation's.
struct Perturbation {
    _symbols: Vec<Symbol>,
    _blocks: Vec<Vec<u8>>,
}

impl Perturbation {
    fn new(round: usize) -> Self {
        Self {
            _symbols: (0..512)
                .map(|i| Symbol::new(&format!("determinism_perturbation_{round}_{i}")))
                .collect(),
            _blocks: (0..64).map(|i| vec![0; 24 + 40 * i + round]).collect(),
        }
    }
}

/// Compile the input in a fresh database, then again in another fresh
/// database beside a perturbation, and report the first difference.
fn check_deterministic(
    label: &str,
    required: Required,
    load: impl Fn(&TributeDatabaseImpl) -> SourceCst,
) -> Result<(), String> {
    let first = compile_fresh(&load);
    if let Some(missing) = missing(&first, required) {
        return Err(format!("{label}: {missing}"));
    }
    for round in 0..repeat() {
        let perturbation = Perturbation::new(round);
        let again = compile_fresh(&load);
        drop(perturbation);
        if let Some(difference) = difference(&first, &again) {
            return Err(format!("{label}: {difference}"));
        }
    }
    Ok(())
}

fn difference(first: &Artifacts, again: &Artifacts) -> Option<String> {
    text_difference("native IR", &first.native_ir, &again.native_ir)
        .or_else(|| text_difference("Wasm IR", &first.wasm_ir, &again.wasm_ir))
        .or_else(|| bytes_difference("native object", &first.native_object, &again.native_object))
        .or_else(|| bytes_difference("Wasm module", &first.wasm, &again.wasm))
}

fn outcome_difference<T>(what: &str, first: &Option<T>, again: &Option<T>) -> Option<String> {
    (first.is_some() != again.is_some()).then(|| {
        let outcome = |artifact: &Option<T>| {
            if artifact.is_some() {
                "succeeded"
            } else {
                "failed"
            }
        };
        format!(
            "{what}: the first compilation {}, the second {}",
            outcome(first),
            outcome(again)
        )
    })
}

/// The first line where two texts differ, with the lines around it.
fn text_difference(what: &str, first: &Option<String>, again: &Option<String>) -> Option<String> {
    if let Some(difference) = outcome_difference(what, first, again) {
        return Some(difference);
    }
    let (first, again) = (first.as_deref()?, again.as_deref()?);
    if first == again {
        return None;
    }
    let (first, again) = (first.lines().collect_vec(), again.lines().collect_vec());
    let line = first
        .iter()
        .zip(&again)
        .position(|(a, b)| a != b)
        .unwrap_or(first.len().min(again.len()));
    let context = |lines: &[&str]| {
        let start = line.saturating_sub(2);
        let end = (line + 3).min(lines.len());
        lines[start..end]
            .iter()
            .enumerate()
            .format_with("\n", |(i, text), f| {
                f(&format_args!("{:>6} | {text}", start + i + 1))
            })
            .to_string()
    };
    Some(format!(
        "{what} differs at line {}\n--- first\n{}\n--- second\n{}",
        line + 1,
        context(&first),
        context(&again)
    ))
}

/// The first byte where two binaries differ, and their lengths.
fn bytes_difference(
    what: &str,
    first: &Option<Vec<u8>>,
    again: &Option<Vec<u8>>,
) -> Option<String> {
    if let Some(difference) = outcome_difference(what, first, again) {
        return Some(difference);
    }
    let (first, again) = (first.as_deref()?, again.as_deref()?);
    if first == again {
        return None;
    }
    let offset = first
        .iter()
        .zip(again)
        .position(|(a, b)| a != b)
        .unwrap_or(first.len().min(again.len()));
    Some(format!(
        "{what} differs at byte {offset} (lengths {} and {})",
        first.len(),
        again.len()
    ))
}

fn lang_examples() -> (PathBuf, Vec<PathBuf>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("lang-examples");
    let mut files = Vec::new();
    examples(&root, &root, &mut files);
    files.sort();
    assert!(!files.is_empty());
    (root, files)
}

fn example_name(file: &Path) -> String {
    file.to_string_lossy().replace('\\', "/")
}

/// Every example compiles to the same artifacts in two fresh databases.
/// The examples reach abilities, closures, enums, and modules, which the
/// generated programs do not.
#[test]
fn lang_examples_compile_deterministically() {
    let (root, files) = lang_examples();
    let failures = files
        .iter()
        .filter_map(|file| {
            let name = example_name(file);
            check_deterministic(&name, required(&name), |db| {
                db.input(root.join(file)).expect("example source")
            })
            .err()
        })
        .collect_vec();
    assert!(failures.is_empty(), "{}", failures.iter().format("\n\n"));
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        max_shrink_iters: 32,
        ..ProptestConfig::default()
    })]

    /// Every generated program compiles to the same artifacts in two fresh
    /// databases.
    #[test]
    fn generated_programs_compile_deterministically(program in program()) {
        let source = program.source();
        check_deterministic("generated.trb", Required::EVERYTHING, |db| {
            SourceCst::from_source_str(db, "generated.trb", &source)
        })
        .map_err(|error| TestCaseError::fail(format!("{error}\n{source}")))?;
    }
}

/// Compile `file` with the CLI. `padding` sets the size of an unused
/// environment variable, which moves the process's stack and heap. The CLI
/// writes no native object, so the result has none.
fn cli_run(file: &Path, out: &Path, padding: usize) -> Artifacts {
    let compile = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_tribute"))
            .arg("compile")
            .args(args)
            .arg(file)
            .env("TRIBUTE_DETERMINISM_PADDING", "x".repeat(padding))
            .output()
            .expect("invoke tribute compile")
    };
    let _ = std::fs::remove_file(out);
    let out_arg = out.to_str().expect("UTF-8 temporary path");
    let dump = |output: Output| {
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    };
    let wasm_status = compile(&["--target", "wasm", "-o", out_arg]).status;
    Artifacts {
        native_ir: dump(compile(&["--target", "native", "--dump-ir"])),
        wasm_ir: dump(compile(&["--target", "wasm", "--dump-ir"])),
        native_object: None,
        wasm: wasm_status
            .success()
            .then(|| std::fs::read(out).expect("read the Wasm module")),
    }
}

/// Every example compiles to the same IR dumps and Wasm module in two CLI
/// processes. The native object is not compared across processes: the CLI
/// writes only a linked executable, which carries linker output such as a
/// Mach-O `LC_UUID`.
#[test]
fn lang_examples_compile_deterministically_across_processes_slow() {
    let (root, files) = lang_examples();
    let temp = tempfile::tempdir().expect("temporary directory");
    let out = temp.path().join("module.wasm");
    let failures = files
        .iter()
        .filter_map(|file| {
            let (name, path) = (example_name(file), root.join(file));
            let first = cli_run(&path, &out, 0);
            let required = Required {
                native_object: false,
                ..required(&name)
            };
            if let Some(missing) = missing(&first, required) {
                return Some(format!("{name}: {missing}"));
            }
            (0..repeat()).find_map(|round| {
                let again = cli_run(&path, &out, 4096 + 1024 * round);
                difference(&first, &again).map(|difference| format!("{name}: {difference}"))
            })
        })
        .collect_vec();
    assert!(failures.is_empty(), "{}", failures.iter().format("\n\n"));
}
