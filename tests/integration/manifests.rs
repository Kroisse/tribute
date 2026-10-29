//! Packages loaded from their manifests (`new-plans/modules.md` §
//! 매니페스트 (`tribute.toml`)): `tribute compile` builds a package
//! directory, and the manifest paths it follows do not depend on the working
//! directory.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tribute::pipeline::{compile_frontend, compile_with_diagnostics};
use tribute_front::testing::LoggingDb;
use trunk_ir::printer::print_module;

const GEOMETRY_MANIFEST: &str = "[package]\nname = \"geometry\"\n";

const GEOMETRY: &str = "\
pub struct Rect { w: Nat, h: Nat }

pub fn area(rect: Rect) -> Nat { rect.w * rect.h }
";

const APP_MANIFEST: &str = "\
[package]
name = \"app\"

[dependencies]
geometry = { path = \"../geometry\" }
";

const APP: &str = "\
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    print_line(geometry::area(geometry::Rect { w: 3, h: 4 }).to_string)
}
";

/// A workspace directory holding `app`, which depends on `geometry`.
fn workspace() -> tempfile::TempDir {
    let temp = tempfile::tempdir().expect("temporary directory");
    write(temp.path(), "geometry/tribute.toml", GEOMETRY_MANIFEST);
    write(temp.path(), "geometry/src/lib.trb", GEOMETRY);
    write(temp.path(), "app/tribute.toml", APP_MANIFEST);
    write(temp.path(), "app/src/main.trb", APP);
    temp
}

fn write(root: &Path, path: &str, text: &str) {
    let path = root.join(path);
    std::fs::create_dir_all(path.parent().expect("a parent directory"))
        .expect("create the directory");
    std::fs::write(path, text).expect("write the file");
}

/// Run `tribute compile` in `cwd` with `args`.
fn compile(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tribute"))
        .current_dir(cwd)
        .arg("compile")
        .args(args)
        .output()
        .expect("invoke tribute compile")
}

#[track_caller]
fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// The app's IR has the dependency's functions at the identity paths the
/// dependency has on its own, beside the app's own `pkg::` paths.
#[test]
fn an_app_lowers_its_dependency_at_its_identity_paths() {
    let mut db = LoggingDb::default();
    db.file("/ws/geometry/tribute.toml", GEOMETRY_MANIFEST);
    db.file("/ws/geometry/src/lib.trb", GEOMETRY);
    db.file("/ws/app/tribute.toml", APP_MANIFEST);
    let app = db.file("/ws/app/src/main.trb", APP);
    let messages: Vec<_> = compile_with_diagnostics(&db, app)
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.inner.message)
        .collect();
    assert_eq!(messages, Vec::<String>::new());
    let (ctx, module) = compile_frontend(&db, app).expect("frontend output");
    let ir = print_module(&ctx, module.op());
    for function in ["geometry::area", "pkg::main"] {
        assert!(
            ir.contains(&format!("tribute_control.func @\"{function}\"")),
            "missing `{function}`:\n{ir}"
        );
    }
}

/// `tribute compile <dir>` builds and runs a package with a path
/// dependency, from two working directories that spell the package
/// differently.
#[test]
fn the_cli_compiles_a_package_from_any_working_directory() {
    let temp = workspace();
    let ws = temp.path();
    let output_dir = tempfile::tempdir().expect("temporary output directory");
    let runs: [(PathBuf, &str); 2] = [
        (ws.to_owned(), "app"),
        (ws.join("geometry/src"), "../../app/"),
    ];
    for (index, (cwd, package)) in runs.iter().enumerate() {
        let binary = output_dir.path().join(format!("app{index}"));
        let compiled = compile(
            cwd,
            &[
                "--target",
                "native",
                package,
                "-o",
                binary.to_str().expect("a UTF-8 path"),
            ],
        );
        assert_success(&compiled);
        let run = Command::new(&binary).output().expect("run the binary");
        assert!(run.status.success(), "{run:?}");
        assert_eq!(String::from_utf8_lossy(&run.stdout), "12\n", "from {cwd:?}");
    }
}

/// Without `-o`, the executable is named after the package, in its
/// directory.
#[test]
fn the_cli_names_a_packages_executable_after_it() {
    let temp = workspace();
    let compiled = compile(temp.path(), &["--target", "native", "app"]);
    assert_success(&compiled);
    assert!(temp.path().join("app/app").is_file());
}

/// A package with only `src/lib.trb` is checked as a library, and builds
/// no executable.
#[test]
fn the_cli_checks_a_library_package() {
    let temp = workspace();
    let checked = compile(temp.path(), &["--target", "none", "geometry"]);
    assert_success(&checked);

    let built = compile(temp.path(), &["--target", "native", "geometry"]);
    assert!(!built.status.success());
    assert!(
        stderr(&built).contains("package `geometry` has no `src/main.trb`"),
        "{}",
        stderr(&built)
    );

    write(
        temp.path(),
        "geometry/src/lib.trb",
        "pub fn f() -> Nat { missing }\n",
    );
    let checked = compile(temp.path(), &["--target", "none", "geometry"]);
    assert!(!checked.status.success());
    assert!(
        stderr(&checked).contains("unresolved name `missing`"),
        "{}",
        stderr(&checked)
    );
}

/// A directory without `tribute.toml`, spelled exactly, is no package.
#[test]
fn the_cli_reports_a_missing_or_misspelled_manifest() {
    let temp = workspace();
    let missing = compile(temp.path(), &["--target", "none", "app/src"]);
    assert!(!missing.status.success());
    assert!(
        stderr(&missing).contains("cannot find `tribute.toml` in"),
        "{}",
        stderr(&missing)
    );

    std::fs::rename(
        temp.path().join("app/tribute.toml"),
        temp.path().join("app/Tribute.toml"),
    )
    .expect("rename the manifest");
    let misspelled = compile(temp.path(), &["--target", "none", "app"]);
    assert!(!misspelled.status.success());
    assert!(
        stderr(&misspelled)
            .contains("the manifest must be named `tribute.toml`, not `Tribute.toml`"),
        "{}",
        stderr(&misspelled)
    );
}

/// Manifest errors are printed against the manifest that has them.
#[test]
fn the_cli_reports_manifest_errors_against_their_manifest() {
    let temp = workspace();
    write(
        temp.path(),
        "app/tribute.toml",
        "[package]\nname = \"app\"\n\n[dependencies]\ngeo = { path = \"../geometry\" }\n",
    );
    let compiled = compile(temp.path(), &["--target", "native", "app"]);
    let stderr = stderr(&compiled);
    assert!(!compiled.status.success(), "{stderr}");
    assert!(
        stderr.contains("dependency `geo` names the package `geometry`"),
        "{stderr}"
    );
    assert!(stderr.contains("app/tribute.toml:5:1"), "{stderr}");

    write(
        temp.path(),
        "geometry/tribute.toml",
        "[package]\nname = \"std\"\n",
    );
    let compiled = compile(temp.path(), &["--target", "none", "geometry"]);
    let stderr = String::from_utf8_lossy(&compiled.stderr);
    assert!(!compiled.status.success(), "{stderr}");
    assert!(stderr.contains("geometry/tribute.toml:2:8"), "{stderr}");

    std::fs::remove_file(temp.path().join("geometry/src/lib.trb")).expect("remove lib.trb");
    write(temp.path(), "geometry/tribute.toml", GEOMETRY_MANIFEST);
    let compiled = compile(temp.path(), &["--target", "none", "geometry"]);
    let stderr = String::from_utf8_lossy(&compiled.stderr);
    assert!(!compiled.status.success(), "{stderr}");
    assert!(
        stderr.contains("the package has neither `src/lib.trb` nor `src/main.trb`"),
        "{stderr}"
    );
}
