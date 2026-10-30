//! Packages loaded from their manifests (`new-plans/modules.md` §
//! 매니페스트 (`tribute.toml`)), without a file system.

use tribute_core::diagnostic::Diagnostic;
use tribute_front::analysis::{self, DependencyTable, Package};
use tribute_front::manifest::{self, source_dependencies};
use tribute_front::resolve::PackageName;
use tribute_front::source_file::{FileEntry, SourceFiles, file_at, uri_to_path};
use tribute_front::testing::LoggingDb;
use tribute_front::{SourceCst, path_to_uri};
use trunk_ir::Symbol;

const PRELUDE: &str = include_str!("../../../lib/std/prelude.trb");

const GEOMETRY: &str = "\
pub struct Rect { w: Nat, h: Nat }

pub fn area(rect: Rect) -> Nat { rect.w * rect.h }
";

const APP: &str = "\
fn run() -> Nat { geometry::area(geometry::Rect { w: 3, h: 4 }) }
";

/// A workspace of packages in memory, analyzed against package `std`.
struct Workspace {
    db: LoggingDb,
    std: DependencyTable,
}

impl Workspace {
    fn new() -> Self {
        let db = LoggingDb::default();
        let std = Package::new(
            &db,
            PackageName::std(),
            db.source("prelude:///std/prelude", PRELUDE),
            DependencyTable::new(&db, Vec::new()),
        );
        let std = DependencyTable::new(&db, vec![std]);
        Self { db, std }
    }

    /// `app` depending on `geometry`, both under `/ws`.
    fn with_app() -> (Self, SourceCst) {
        let mut ws = Self::new();
        ws.file(
            "/ws/geometry/tribute.toml",
            r#"[package]
name = "geometry"
"#,
        );
        ws.file("/ws/geometry/src/lib.trb", GEOMETRY);
        ws.file(
            "/ws/app/tribute.toml",
            r#"[package]
name = "app"

[dependencies]
geometry = { path = "../geometry" }
"#,
        );
        let app = ws.file("/ws/app/src/main.trb", APP);
        (ws, app)
    }

    fn file(&mut self, path: &str, text: &str) -> SourceCst {
        self.db.file(path, text)
    }

    fn entry(&self, path: &str) -> FileEntry {
        file_at(&self.db, &path_to_uri(path.as_ref())).expect("the database has a provider")
    }

    /// The packages `source` is analyzed against, by name.
    fn table(&self, source: SourceCst) -> Vec<String> {
        source_dependencies(&self.db, source, self.std)
            .packages(&self.db)
            .iter()
            .map(|package| {
                package
                    .name(&self.db)
                    .name()
                    .expect("a library")
                    .to_string()
            })
            .collect()
    }

    /// The diagnostics analyzing `source` reports, before its own source.
    fn source_errors(&self, source: SourceCst) -> String {
        source_dependencies(&self.db, source, self.std);
        self.located(source_dependencies::accumulated::<Diagnostic>(
            &self.db, source, self.std,
        ))
    }

    /// The diagnostics loading the package of the manifest at `path`
    /// reports.
    fn load_errors(&self, path: &str) -> String {
        let file = self.entry(path);
        manifest::load_package(&self.db, file, self.std);
        self.located(manifest::load_package::accumulated::<Diagnostic>(
            &self.db, file, self.std,
        ))
    }

    /// Each diagnostic with the path of the file it is in and the text it
    /// points at.
    fn located(&self, diagnostics: Vec<&Diagnostic>) -> String {
        let provider = SourceFiles::get(&self.db).provider(&self.db).clone();
        let mut lines: Vec<String> = diagnostics
            .into_iter()
            .map(|diagnostic| {
                let span = diagnostic.inner.span;
                let file = provider
                    .find(&self.db, span.source)
                    .expect("a diagnostic in a file the build read");
                let text = file.text(&self.db).to_string();
                format!(
                    "{}:{}..{} `{}`: {}",
                    uri_to_path(file.uri(&self.db))
                        .expect("a file path")
                        .display(),
                    span.start,
                    span.end,
                    &text[span.range()],
                    diagnostic.inner.message
                )
            })
            .collect();
        lines.sort();
        lines.join("\n")
    }

    /// The paths of the functions `package` exports, sorted.
    fn exported_functions(&self, package: Package) -> Vec<String> {
        let checked = analysis::checked_package(&self.db, package).expect("the package checks");
        let mut paths: Vec<String> = checked
            .exports
            .function_types(&self.db)
            .iter()
            .map(|(id, _)| id.path(&self.db).to_string())
            .collect();
        paths.sort();
        paths
    }
}

#[test]
fn an_app_is_analyzed_against_its_manifests_dependencies() {
    let (ws, app) = Workspace::with_app();
    assert_eq!(ws.table(app), ["geometry", "std"]);
    assert_eq!(ws.source_errors(app), "");
    let messages: Vec<_> =
        analysis::diagnostics(&ws.db, app, *source_dependencies(&ws.db, app, ws.std))
            .into_iter()
            .map(|diagnostic| diagnostic.inner.message.clone())
            .collect();
    assert_eq!(messages, Vec::<String>::new());
}

#[test]
fn a_source_outside_a_package_depends_on_std_alone() {
    let mut ws = Workspace::new();
    ws.file(r#"/ws/app/tribute.toml"#, "[package]\nname = \"app\"\n");
    let lib = ws.file("/ws/app/src/lib.trb", GEOMETRY);
    let loose = ws.file("/ws/loose/src/main.trb", APP);
    assert_eq!(ws.table(lib), ["std"]);
    assert_eq!(ws.table(loose), ["std"]);
}

/// The dependency's identity paths are the same whether it is loaded as a
/// dependency, loaded alone, or built from its root file by hand.
#[test]
fn a_dependency_has_the_identity_paths_it_has_alone() {
    let (ws, app) = Workspace::with_app();
    let table = *source_dependencies(&ws.db, app, ws.std);
    let dependency = table.packages(&ws.db)[0];
    let alone = manifest::library_package(&ws.db, ws.entry("/ws/geometry/tribute.toml"), ws.std)
        .expect("geometry loads");
    assert_eq!(dependency, alone, "one directory is one package");

    let by_hand = Package::new(
        &ws.db,
        PackageName::library(Symbol::new("geometry")),
        ws.db.source("/elsewhere/lib.trb", GEOMETRY),
        ws.std,
    );
    let expected = ["geometry::area"];
    assert_eq!(ws.exported_functions(dependency), expected);
    assert_eq!(ws.exported_functions(by_hand), expected);
}

/// Two packages that name one directory, spelled differently, share it.
#[test]
fn dependencies_on_one_directory_share_one_package() {
    let mut ws = Workspace::new();
    ws.file(
        "/ws/geometry/tribute.toml",
        r#"[package]
name = "geometry"
"#,
    );
    ws.file("/ws/geometry/src/lib.trb", GEOMETRY);
    ws.file(
        "/ws/shapes/tribute.toml",
        r#"[package]
name = "shapes"

[dependencies]
geometry = { path = "../geometry" }
"#,
    );
    ws.file("/ws/shapes/src/lib.trb", "");
    ws.file(
        "/ws/app/tribute.toml",
        r#"[package]
name = "app"

[dependencies]
geometry = { path = "./../geometry/" }
shapes = { path = "../shapes" }
"#,
    );
    let app = ws.file("/ws/app/src/main.trb", APP);
    let table = *source_dependencies(&ws.db, app, ws.std);
    let packages = table.packages(&ws.db);
    let shapes = packages[1];
    assert_eq!(shapes.dependencies(&ws.db).packages(&ws.db)[0], packages[0]);
    assert_eq!(ws.source_errors(app), "");
}

#[test]
fn editing_a_manifest_reloads_only_its_dependents() {
    let (mut ws, app) = Workspace::with_app();
    let analyze = |ws: &Workspace| {
        ws.db.record(|db| {
            let table = *source_dependencies(db, app, ws.std);
            analysis::analyze(db, app, table).expect("app parses");
        })
    };
    let baseline = analyze(&ws);

    // A comment changes no manifest value.
    ws.file(
        "/ws/geometry/tribute.toml",
        r#"# shapes
[package]
name = "geometry"
"#,
    );
    analyze(&ws).assert_reuse(
        &baseline,
        &["manifest"],
        &[
            "library_package",
            "dependency_table",
            "checked_package",
            "analyze",
        ],
    );

    // A dependency that does not load leaves the table as it was.
    ws.file(
        "/ws/app/tribute.toml",
        r#"[package]
name = "app"

[dependencies]
geometry = { path = "../geometry" }
extra = { path = "../extra" }
"#,
    );
    analyze(&ws).assert_reuse(
        &baseline,
        &["manifest", "loaded_dependencies"],
        &["dependency_table", "checked_package", "analyze"],
    );
}

#[test]
fn creating_a_missing_dependency_loads_it() {
    let (mut ws, app) = Workspace::with_app();
    ws.db.remove_file("/ws/geometry/tribute.toml");
    assert_eq!(ws.table(app), ["std"]);
    ws.file(
        "/ws/geometry/tribute.toml",
        r#"[package]
name = "geometry"
"#,
    );
    assert_eq!(ws.table(app), ["geometry", "std"]);
}

fn app_with(manifest: &str) -> (Workspace, SourceCst) {
    let mut ws = Workspace::new();
    ws.file("/ws/app/tribute.toml", manifest);
    let app = ws.file("/ws/app/src/main.trb", APP);
    (ws, app)
}

#[test]
fn diag_manifest_package_names() {
    let mut report = Vec::new();
    for manifest in [
        r#"[dependencies]
"#,
        r#"[package]
version = "1"
"#,
        r#"[package]
name = 1
"#,
        r#"[package]
name = "pkg"
"#,
        r#"[package]
name = "std"
"#,
        r#"[package]
name = "Geometry"
"#,
        r#"[package]
name = "my-app"
"#,
        r#"[package
name = "app"
"#,
    ] {
        let (ws, app) = app_with(manifest);
        report.push(format!("{manifest}=> {}\n", ws.source_errors(app)));
    }
    insta::assert_snapshot!(report.join("\n"));
}

#[test]
fn diag_manifest_dependency_entries() {
    let mut ws = Workspace::new();
    ws.file(
        "/ws/geometry/tribute.toml",
        r#"[package]
name = "geometry"
"#,
    );
    ws.file("/ws/geometry/src/lib.trb", GEOMETRY);
    ws.file(
        "/ws/misspelled/Tribute.toml",
        r#"[package]
name = "misspelled"
"#,
    );
    ws.file("/ws/misspelled/src/lib.trb", "");
    ws.file(
        "/ws/app_only/tribute.toml",
        r#"[package]
name = "app_only"
"#,
    );
    ws.file("/ws/app_only/src/main.trb", "");
    ws.file(
        "/ws/app/tribute.toml",
        r#"[package]
name = "app"

[dependencies]
missing = { path = "../missing" }
misspelled = { path = "../misspelled" }
geo = { path = "../geometry" }
app_only = { path = "../app_only" }
std = { path = "../std" }
versioned = { path = "../geometry", version = "1" }
bare = "../geometry"
pathless = {}
numbered = { path = 1 }
"#,
    );
    let app = ws.file("/ws/app/src/main.trb", APP);
    insta::assert_snapshot!(ws.source_errors(app));
    assert_eq!(ws.table(app), ["std"]);
}

#[test]
fn diag_manifest_dependency_cycle() {
    let mut ws = Workspace::new();
    for (name, dependency) in [("a", "b"), ("b", "c"), ("c", "b")] {
        ws.file(
            &format!("/ws/{name}/tribute.toml"),
            &format!(
                r#"[package]
name = "{name}"

[dependencies]
{dependency} = {{ path = "../{dependency}" }}
"#
            ),
        );
        ws.file(&format!("/ws/{name}/src/lib.trb"), "");
    }
    ws.file(
        "/ws/app/tribute.toml",
        r#"[package]
name = "app"

[dependencies]
a = { path = "../a" }
itself = { path = "." }
"#,
    );
    let app = ws.file("/ws/app/src/main.trb", "");
    insta::assert_snapshot!(ws.source_errors(app));
    // `a` reaches the cycle `b -> c -> b` at `b`, whose entry for `c`
    // closes it, so `b` depends on `std` alone.
    let table = *source_dependencies(&ws.db, app, ws.std);
    let a = table.packages(&ws.db)[0];
    let b = a.dependencies(&ws.db).packages(&ws.db)[0];
    assert_eq!(b.name(&ws.db).name().expect("a library"), "b");
    assert_eq!(
        b.dependencies(&ws.db).packages(&ws.db)[..],
        ws.std.packages(&ws.db)[..]
    );
}

#[test]
fn diag_manifest_package_without_root_files() {
    let mut ws = Workspace::new();
    ws.file(r#"/ws/empty/tribute.toml"#, "[package]\nname = \"empty\"\n");
    ws.file("/ws/empty/main.trb", "");
    insta::assert_snapshot!(ws.load_errors("/ws/empty/tribute.toml"));
}
