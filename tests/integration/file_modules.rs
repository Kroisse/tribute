//! A package's file modules (`new-plans/modules.md` § 모듈 선언): each `mod`
//! without a body reads its file, and the package is checked as if its
//! modules were written inline.

use std::process::Command;

use tribute::pipeline::{compile_frontend, compile_with_diagnostics, parse_and_lower_ast};
use tribute_front::SourceCst;
use tribute_front::source_tree::source_tree_of;
use tribute_front::testing::LoggingDb;
use trunk_ir::printer::print_module;

const ROOT: &str = "/pkg/src/main.trb";

const MAIN: &str = "\
mod math
mod util {
    pub mod text
}

fn run() -> Nat { util::text::twice(math::square(3)) }

fn main() -> Nil { }
";

const MATH: &str = "\
pub mod ops

pub fn square(n: Nat) -> Nat { ops::times(n, n) }
";

const OPS: &str = "\
pub fn times(a: Nat, b: Nat) -> Nat { a * b }
";

const TEXT: &str = "\
use pkg::math::ops::times

pub fn twice(n: Nat) -> Nat { times(n, 2) }
";

const INLINE: &str = "\
mod math {
    pub mod ops {
        pub fn times(a: Nat, b: Nat) -> Nat { a * b }
    }

    pub fn square(n: Nat) -> Nat { ops::times(n, n) }
}
mod util {
    pub mod text {
        use pkg::math::ops::times

        pub fn twice(n: Nat) -> Nat { times(n, 2) }
    }
}

fn run() -> Nat { util::text::twice(math::square(3)) }

fn main() -> Nil { }
";

/// The package of `MAIN` and its file modules.
fn package(db: &mut LoggingDb) -> SourceCst {
    db.file("/pkg/src/math.trb", MATH);
    db.file("/pkg/src/math/ops.trb", OPS);
    db.file("/pkg/src/util/text.trb", TEXT);
    db.file(ROOT, MAIN)
}

/// The diagnostics of the package rooted at `root`, each with the path of
/// the file it is in.
fn located_diagnostics(db: &LoggingDb, root: SourceCst) -> Vec<String> {
    let tree = source_tree_of(db, root).expect("the root parses");
    compile_with_diagnostics(db, root)
        .diagnostics
        .iter()
        .map(|diagnostic| {
            let span = diagnostic.inner.span;
            let file = tree.file(db, span.source).expect("a file of the package");
            format!(
                "{}:{}..{}: {}",
                file.uri(db).path(),
                span.start,
                span.end,
                diagnostic.inner.message
            )
        })
        .collect()
}

#[test]
fn file_modules_lower_like_inline_modules() {
    let mut db = LoggingDb::default();
    let root = package(&mut db);
    assert_eq!(located_diagnostics(&db, root), Vec::<String>::new());
    let (ctx, module) = compile_frontend(&db, root).expect("frontend output");
    let files = print_module(&ctx, module.op());

    let mut db = LoggingDb::default();
    let root = db.file(ROOT, INLINE);
    let (ctx, module) = compile_frontend(&db, root).expect("frontend output");
    let inline = print_module(&ctx, module.op());

    assert_eq!(files, inline);
}

#[test]
fn editing_a_file_module_parses_only_that_file() {
    let mut db = LoggingDb::default();
    let root = package(&mut db);
    let analyze = |db: &LoggingDb| {
        parse_and_lower_ast(db, root).expect("the package type checks");
    };
    let baseline = db.record(analyze);
    assert_eq!(baseline.count("lower_file_module"), 3, "{baseline:?}");

    db.file(
        "/pkg/src/math/ops.trb",
        "pub fn times(a: Nat, b: Nat) -> Nat { b * a }\n",
    );
    let edited = db.record(analyze);
    assert_eq!(edited.count("lower_file_module"), 1, "{edited:?}");
    assert_eq!(edited.count("parse_cst"), 1, "{edited:?}");
    edited.assert_reuse(
        &baseline,
        &["source_tree"],
        &["lower_source_to_parsed_ast_with_module_path"],
    );
}

#[test]
fn creating_a_missing_file_module_loads_it() {
    let mut db = LoggingDb::default();
    let root = db.file(ROOT, "mod util\n\nfn main() -> Nil { util::run() }\n");
    assert_eq!(
        located_diagnostics(&db, root),
        [
            "/pkg/src/main.trb:0..8: cannot find the file `/pkg/src/util.trb` of module `util`",
            "/pkg/src/main.trb:29..38: unresolved name `util::run`",
        ]
    );

    db.file("/pkg/src/util.trb", "pub fn run() -> Nil { }\n");
    assert_eq!(located_diagnostics(&db, root), Vec::<String>::new());

    db.remove_file("/pkg/src/util.trb");
    assert_eq!(located_diagnostics(&db, root).len(), 2);
}

#[test]
fn diag_file_module_missing() {
    let mut db = LoggingDb::default();
    let root = db.file(ROOT, "mod util\n\nfn main() -> Nil { }\n");
    insta::assert_yaml_snapshot!(located_diagnostics(&db, root));
}

#[test]
fn diag_file_module_with_a_body() {
    let mut db = LoggingDb::default();
    db.file("/pkg/src/util.trb", "pub fn run() -> Nil { }\n");
    let root = db.file(
        ROOT,
        "mod util {\n    pub fn run() -> Nil { }\n}\n\nfn main() -> Nil { util::run() }\n",
    );
    insta::assert_yaml_snapshot!(located_diagnostics(&db, root));
}

#[test]
fn diag_file_module_reached_twice() {
    let mut db = LoggingDb::default();
    let root = db.file(ROOT, "mod main\n\nfn main() -> Nil { }\n");
    insta::assert_yaml_snapshot!(located_diagnostics(&db, root));
}

#[test]
fn diag_file_module_declared_twice() {
    let mut db = LoggingDb::default();
    db.file("/pkg/src/util.trb", "pub fn run() -> Nil { }\n");
    let root = db.file(ROOT, "mod util\nmod util\n\nfn main() -> Nil { }\n");
    insta::assert_yaml_snapshot!(located_diagnostics(&db, root));
}

#[test]
fn diag_file_module_errors_name_their_file() {
    let mut db = LoggingDb::default();
    db.file(
        "/pkg/src/util.trb",
        "pub mod deep\n\npub fn run() -> Nat { missing }\n",
    );
    db.file("/pkg/src/util/deep.trb", "pub fn f( -> Nat { 1 }\n");
    let root = db.file(ROOT, "mod util\n\nfn main() -> Nil { }\n");
    insta::assert_yaml_snapshot!(located_diagnostics(&db, root));
}

/// `tribute compile` reads the file modules of the file it compiles, whose
/// directory acts as the package's `src/`.
#[test]
fn the_cli_compiles_and_runs_a_file_module_package() {
    let root = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/lang-examples/modules_file/main.trb"
    );
    let output_dir = tempfile::tempdir().expect("temporary output directory");
    let binary = output_dir.path().join("modules_file");
    let compiled = Command::new(env!("CARGO_BIN_EXE_tribute"))
        .args(["compile", "--target", "native", root, "-o"])
        .arg(&binary)
        .output()
        .expect("invoke tribute compile");
    assert!(
        compiled.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary).output().expect("run the binary");
    assert!(run.status.success(), "{run:?}");
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "Testing file-based modules:\nBasic math:\n15\n5\nAdvanced math:\n256\nUtilities:\nhahaha\nFile-based modules work!\n"
    );
}

/// `tribute compile` prints a diagnostic in a file module against that file.
#[test]
fn the_cli_reports_file_module_errors_against_their_file() {
    let package = tempfile::tempdir().expect("temporary package directory");
    let root = package.path().join("main.trb");
    std::fs::write(&root, "mod util\n\nfn main() -> Nil { }\n").expect("write main.trb");
    std::fs::write(
        package.path().join("util.trb"),
        "pub fn run() -> Nat { missing }\n\npub fn run() -> Nat { 1 }\n",
    )
    .expect("write util.trb");
    let compiled = Command::new(env!("CARGO_BIN_EXE_tribute"))
        .args(["compile", "--target", "native"])
        .arg(&root)
        .arg("-o")
        .arg(package.path().join("out"))
        .output()
        .expect("invoke tribute compile");
    let stderr = String::from_utf8_lossy(&compiled.stderr);
    assert!(!compiled.status.success(), "stderr: {stderr}");
    assert!(
        stderr.contains("unresolved name `missing`"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("util.trb:1:"), "stderr: {stderr}");
    assert!(stderr.contains("first defined here"), "stderr: {stderr}");
}

/// `tribute debug --show-env` builds the environment of the whole package.
#[test]
fn the_cli_debugs_a_file_module_package() {
    let root = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/lang-examples/modules_file/main.trb"
    );
    let debugged = Command::new(env!("CARGO_BIN_EXE_tribute"))
        .args(["debug", "--show-env", root])
        .output()
        .expect("invoke tribute debug");
    let stdout = String::from_utf8_lossy(&debugged.stdout);
    assert!(debugged.status.success(), "{debugged:?}");
    assert!(stdout.contains("No errors"), "stdout: {stdout}");
    assert!(stdout.contains("advanced"), "stdout: {stdout}");
}
