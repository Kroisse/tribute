//! Tests that an edit reruns only the queries whose inputs it changed.

use tribute_front::SourceCst;
use tribute_front::ast::{ItemLayout, ItemSpans};
use tribute_front::astgen::ParsedItem;
use tribute_front::query::{func_names, parsed_ast, parsed_module, resolved_module};
use tribute_front::testing::{Executed, LoggingDb};

const PATH: &str = "reuse.trb";

const PROGRAM: &str = "\
fn f(x: Int) -> Int { x }
fn g(y: Int) -> Int { y }
";

/// A query that depends on the function names only.
#[salsa::tracked(returns(copy))]
fn function_count(db: &dyn salsa::Database, source: SourceCst) -> usize {
    func_names(db, source).len()
}

/// A database holding `PROGRAM` with the queries under test computed, the
/// source, and the record of that computation.
fn computed() -> (LoggingDb, SourceCst, Executed) {
    let db = LoggingDb::default();
    let source = db.source(PATH, PROGRAM);
    let baseline = db.record(|db| {
        function_count(db, source);
        resolved_module(db, source);
    });
    (db, source, baseline)
}

/// Move every declaration with a comment and blank lines, and `f`'s body
/// with a comment inside it.
fn move_code(db: &mut LoggingDb, source: SourceCst) {
    db.replace(
        source,
        "fn f(x: Int) -> Int { x }",
        "\
// A comment and blank lines move every declaration.

fn f(x: Int) -> Int {
    // and one inside a body
    x
}",
    );
}

#[test]
fn moving_code_without_changing_it_reuses_the_parsed_module() {
    let (mut db, source, baseline) = computed();
    move_code(&mut db, source);
    db.record(|db| {
        function_count(db, source);
    })
    .assert_reuse(
        &baseline,
        &[],
        &["parsed_ast", "func_names", "function_count"],
    );
}

/// A body edit changes the module, so `func_names` reruns, but its result
/// is unchanged, so what depends on it is reused.
#[test]
fn a_body_edit_keeps_the_function_names() {
    let (mut db, source, baseline) = computed();
    db.replace(source, "{ x }", "{ x + 1 }");
    db.record(|db| assert_eq!(function_count(db, source), 2))
        .assert_reuse(&baseline, &["func_names"], &["function_count"]);
}

#[test]
fn adding_a_function_reruns_what_depends_on_the_names() {
    let (mut db, source, baseline) = computed();
    db.replace(source, "{ y }\n", "{ y }\nfn h() -> Int { 1 }\n");
    db.record(|db| assert_eq!(function_count(db, source), 3))
        .assert_reuse(&baseline, &["func_names", "function_count"], &[]);
}

/// Name resolution reads the span map only to report a diagnostic, so
/// moving code that resolves without one reuses it.
#[test]
fn moving_code_reuses_name_resolution() {
    let (mut db, source, baseline) = computed();
    move_code(&mut db, source);
    db.record(|db| {
        resolved_module(db, source);
    })
    .assert_reuse(&baseline, &[], &["resolved_module"]);
}

/// A program with every kind of item, including one in an inline module.
const ITEMS: &str = "\
use std::io::{Io, print_line}

struct Point { x: Int }

ability Counter {
    op tick() -> Nil
}

mod inner {
    fn h() -> Int { 1 }
}

fn f(x: Int) -> Int { x }
";

/// A query that depends on the parse's relative spans only.
#[salsa::tracked(returns(copy))]
fn has_item_spans(db: &dyn salsa::Database, source: SourceCst) -> bool {
    parsed_ast(db, source).is_some_and(|parsed| *parsed.item_spans(db) != ItemSpans::default())
}

/// A query that depends on the parse's item layout only.
#[salsa::tracked(returns(copy))]
fn has_item_layout(db: &dyn salsa::Database, source: SourceCst) -> bool {
    parsed_ast(db, source).is_some_and(|parsed| *parsed.item_layout(db) != ItemLayout::default())
}

/// Moving every item without changing it changes only the item layout: the
/// module and every node's span relative to its item are reused.
#[test]
fn moving_items_changes_only_their_layout() {
    let db = LoggingDb::default();
    let source = db.source(PATH, ITEMS);
    let baseline = db.record(|db| {
        parsed_module(db, source);
        has_item_spans(db, source);
        has_item_layout(db, source);
    });
    let mut db = db;
    db.replace(source, "use ", "// A comment moves every item.\n\nuse ");
    db.record(|db| {
        parsed_module(db, source);
        assert!(has_item_spans(db, source));
        assert!(has_item_layout(db, source));
    })
    .assert_reuse(
        &baseline,
        &[
            "lower_source_to_parsed_ast_with_module_path",
            "has_item_layout",
        ],
        &["parsed_ast", "has_item_spans"],
    );
}

/// The owner keys of every item of `source`, inline modules' items after
/// their module, read outside any query.
fn item_owners(db: &LoggingDb, source: SourceCst) -> Vec<u64> {
    fn visit(db: &LoggingDb, items: &[ParsedItem<'_>], owners: &mut Vec<u64>) {
        for item in items {
            owners.push(*item.owner(db));
            visit(db, item.children(db), owners);
        }
    }
    let mut owners = Vec::new();
    let parsed = parsed_ast(db, source).expect("the source parses");
    visit(db, parsed.items(db), &mut owners);
    owners
}

/// The fields of the items of `source` that `owners` name, found without
/// reading any other item's declarations or spans.
fn read_items(db: &dyn salsa::Database, source: SourceCst, owners: &[u64]) -> usize {
    fn visit(db: &dyn salsa::Database, items: &[ParsedItem<'_>], owners: &[u64]) -> usize {
        items
            .iter()
            .map(|item| {
                let own = if owners.contains(item.owner(db)) {
                    item.decls(db).len() + usize::from(*item.spans(db) != ItemSpans::default())
                } else {
                    0
                };
                own + visit(db, item.children(db), owners)
            })
            .sum()
    }
    parsed_ast(db, source).map_or(0, |parsed| visit(db, parsed.items(db), owners))
}

/// A query that reads the item `h`, in `ITEMS`'s inline module.
#[salsa::tracked(returns(copy))]
fn edited_item(db: &dyn salsa::Database, source: SourceCst, owner: u64) -> usize {
    read_items(db, source, &[owner])
}

/// A query that reads every other item of `ITEMS`.
#[salsa::tracked(returns(copy))]
fn other_items(db: &dyn salsa::Database, source: SourceCst, owners: Vec<u64>) -> usize {
    read_items(db, source, &owners)
}

/// `ITEMS` with both queries computed, the owner of `h` and of the other
/// items, and the record of that computation.
fn items_computed() -> (LoggingDb, SourceCst, u64, Vec<u64>, Executed) {
    let db = LoggingDb::default();
    let source = db.source(PATH, ITEMS);
    let mut others = item_owners(&db, source);
    // `use`, `Point`, `Counter`, `inner`, then `h` inside it, then `f`.
    let h = others.remove(4);
    let baseline = db.record(|db| {
        edited_item(db, source, h);
        other_items(db, source, others.clone());
    });
    (db, source, h, others, baseline)
}

/// Editing one item's body changes only that item's fields.
#[test]
fn a_body_edit_changes_only_its_item() {
    let (mut db, source, h, others, baseline) = items_computed();
    db.replace(source, "{ 1 }", "{ 1 + 2 }");
    db.record(|db| {
        edited_item(db, source, h);
        other_items(db, source, others.clone());
    })
    .assert_reuse(&baseline, &["edited_item"], &["other_items"]);
}

/// Moving every item without changing it leaves every item's fields
/// unchanged.
#[test]
fn moving_items_keeps_every_item() {
    let (mut db, source, h, others, baseline) = items_computed();
    db.replace(source, "use ", "// A comment moves every item.\n\nuse ");
    db.record(|db| {
        edited_item(db, source, h);
        other_items(db, source, others.clone());
    })
    .assert_reuse(&baseline, &[], &["edited_item", "other_items"]);
}

/// A query that reads one item's fields, keyed by the item.
#[salsa::tracked(returns(copy))]
fn item_fields<'db>(db: &'db dyn salsa::Database, item: ParsedItem<'db>) -> usize {
    item.decls(db).len() + usize::from(*item.spans(db) != ItemSpans::default())
}

/// The item of `source` that `owner` names, read through `item_fields`.
#[salsa::tracked(returns(copy))]
fn item_by_owner(db: &dyn salsa::Database, source: SourceCst, owner: u64) -> usize {
    let parsed = parsed_ast(db, source).expect("the source parses");
    let item = parsed
        .items(db)
        .iter()
        .find(|item| *item.owner(db) == owner)
        .expect("the item exists");
    item_fields(db, *item)
}

/// The Salsa ids of the top-level items of `source`, by owner key.
fn item_ids(db: &LoggingDb, source: SourceCst) -> Vec<(u64, salsa::Id)> {
    use salsa::plumbing::AsId;
    let parsed = parsed_ast(db, source).expect("the source parses");
    let mut ids: Vec<_> = parsed
        .items(db)
        .iter()
        .map(|item| (*item.owner(db), item.as_id()))
        .collect();
    ids.sort();
    ids
}

/// Reordering items keeps each item's identity and fields; only the list of
/// items changes.
#[test]
fn reordering_items_keeps_every_item() {
    let db = LoggingDb::default();
    let source = db.source(PATH, ITEMS);
    let ids = item_ids(&db, source);
    let read_all = |db: &LoggingDb| {
        for &(owner, _) in &ids {
            item_by_owner(db, source, owner);
        }
    };
    let baseline = db.record(read_all);
    let mut db = db;
    db.replace(source, "struct Point { x: Int }\n\n", "");
    db.replace(
        source,
        "fn f(x: Int) -> Int { x }\n",
        "fn f(x: Int) -> Int { x }\n\nstruct Point { x: Int }\n",
    );
    assert_eq!(item_ids(&db, source), ids, "every item keeps its identity");
    db.record(read_all)
        .assert_reuse(&baseline, &["item_by_owner"], &["item_fields"]);
}
