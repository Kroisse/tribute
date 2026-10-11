//! Resolution of `use` paths: diagnostics for paths that do not resolve,
//! and the identity of what a resolved import names.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "use_imports.trb", text))
}

#[salsa_test]
fn resolvable_imports_are_accepted(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn add(a: Nat, b: Nat) -> Nat { a + b }
}
mod outer {
    mod inner {
        pub fn one() -> Nat { 1 }
    }
    use inner::one
}
use basic
use basic::add
use basic::add as plus
use outer::inner
use helper

fn helper() -> Nat { 1 }

fn main() -> Nil {
    let _ = add(1, plus(2, 3))
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

#[salsa_test]
fn unresolved_imports_are_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn add(a: Nat, b: Nat) -> Nat { a + b }
}
use basic::sub
use nowhere::thing
use nothing_here

fn main() -> Nil {}
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved import `basic::sub`",
            "unresolved import `nowhere::thing`",
            "unresolved import `nothing_here`",
        ],
    );
}

/// A `use` path may start with a path keyword.
#[salsa_test]
fn path_keyword_imports_resolve(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn add(a: Nat, b: Nat) -> Nat { a + b }
}
use self::basic::add

fn main() -> Nil {
    let _ = add(1, 2)
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// A module named where a pattern expects a constructor is rejected before
/// lowering instead of crashing there.
#[salsa_test]
fn module_in_pattern_is_reported(db: &salsa::DatabaseImpl) {
    for (name, pattern) in [("tuple-style", "M(x)"), ("struct-style", "M { x }")] {
        let errors = errors(
            db,
            &format!(
                r#"
mod M {{ pub fn f() -> Nat {{ 1 }} }}
enum E {{ A(Nat) }}

fn g(e: E) -> Nat {{
    case e {{
        {pattern} -> x
        _ -> 0
    }}
}}
"#
            ),
        );
        assert_eq!(
            errors,
            ["expected a constructor, found module `M`"],
            "{name}"
        );
    }
}

/// A module named as a record literal's type is rejected before lowering.
#[salsa_test]
fn module_in_record_literal_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod M { pub fn f() -> Nat { 1 } }

fn main() -> Nil {
    let _ = M { x: 1 }
}
"#,
    );
    assert_eq!(errors, ["expected a constructor, found module `M`"]);
}

/// A type imported by a path written relative to the enclosing inline module
/// keeps the identity name resolution found; type checking does not reread the
/// path from the package root.
#[salsa_test]
fn relative_type_imports_keep_the_resolved_type(db: &salsa::DatabaseImpl) {
    for import in [
        "use a::P",
        "use self::a::P",
        "use pkg::outer::a::P",
        "use a::P as Q",
    ] {
        let annotation = if import.ends_with("as Q") { "Q" } else { "P" };
        let errors = errors(
            db,
            &format!(
                r#"
mod outer {{
    mod a {{
        pub struct P {{ x: Nat }}
    }}
    {import}

    pub fn get(p: {annotation}) -> Nat {{
        p.x
    }}
}}

fn main() -> Nil {{
    let _ = outer::get(outer::a::P {{ x: 1 }})
}}
"#
            ),
        );
        assert!(errors.is_empty(), "{import}: {errors:#?}");
    }
}

/// A path inside an inline module is read from that module, never from the
/// package root, even when the package root names the same path.
#[salsa_test]
fn inline_module_paths_start_from_the_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod a {
    pub struct P { root: Nat }
}
mod outer {
    mod a {
        pub struct P { nested: Nat }
    }
    use a::P

    pub fn get(p: P) -> Nat {
        p.nested
    }
}

fn main() -> Nil {
    let _ = outer::get(outer::a::P { nested: 1 })
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

/// Imports declared inside an inline module are in scope in that module's
/// body, whether the path is written from the package root or relative to the
/// module, aliased, or names a module used as a path prefix.
#[salsa_test]
fn inline_module_imports_are_in_scope_in_the_module(db: &salsa::DatabaseImpl) {
    for (import, call) in [
        ("use pkg::outer::a::one", "one()"),
        ("use a::one", "one()"),
        ("use a::one as uno", "uno()"),
        ("use a", "a::one()"),
    ] {
        let errors = errors(
            db,
            &format!(
                r#"
mod outer {{
    mod a {{
        pub fn one() -> Nat {{ 1 }}
    }}
    {import}

    pub fn get() -> Nat {{
        {call}
    }}
}}

fn main() -> Nil {{
    let _ = outer::get()
}}
"#
            ),
        );
        assert!(errors.is_empty(), "{import}: {errors:#?}");
    }
}

/// Constructors and abilities imported inside an inline module resolve in
/// expressions, effect annotations, handler arms, and unqualified operation
/// calls under an effect annotation.
#[salsa_test]
fn inline_module_imports_cover_constructors_and_abilities(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod a {
        pub struct P { x: Nat }
    }
    mod fx {
        pub ability Tick {
            op tick() -> Nat
        }
    }
    use a::P
    use fx::Tick

    pub fn make() -> P {
        P { x: 1 }
    }

    pub fn count() ->{Tick} Nat {
        Tick::tick()
    }

    pub fn count_unqualified() ->{Tick} Nat {
        tick()
    }

    pub fn run() -> Nat {
        let _ = handle count_unqualified() {
            do result { result }
            op Tick::tick() { resume 2 }
        }
        handle count() {
            do result { result }
            op Tick::tick() { resume make().x }
        }
    }
}

fn main() -> Nil {
    let _ = outer::run()
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

/// An inline module's import is not a member of the module.
#[salsa_test]
fn inline_module_imports_are_not_visible_outside_the_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod a {
        pub fn one() -> Nat { 1 }
    }
    use a::one
}

fn main() -> Nil {
    let _ = outer::one()
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.starts_with("unresolved name `outer::one`")),
        "{errors:#?}"
    );
}

/// An import of an inline module that gives a path prefix a name hides the
/// package-root namespace of that name, even for members the import lacks.
#[salsa_test]
fn inline_module_import_hides_the_root_namespace_it_shadows(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod a {
    pub fn one() -> Nat { 1 }
}
mod outer {
    mod b {
        pub fn two() -> Nat { 2 }
    }
    use outer::b as a

    pub fn get() -> Nat {
        a::one()
    }
}

fn main() -> Nil {
    let _ = outer::get()
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.starts_with("unresolved name `a::one`")),
        "{errors:#?}"
    );
}

/// An effect annotation may reach an ability through a module an inline
/// module imports under an alias, and its operations are callable unqualified.
#[salsa_test]
fn inline_module_import_prefixes_qualified_effect_annotations(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod fx {
        pub ability Tick {
            op tick() -> Nat
        }
    }
    use fx as effects

    pub fn count() ->{effects::Tick} Nat {
        tick()
    }

    pub fn run() -> Nat {
        handle count() {
            do result { result }
            op effects::Tick::tick() { resume 1 }
        }
    }
}

fn main() -> Nil {
    let _ = outer::run()
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

/// Effect annotations follow the same precedence as other names: an ability
/// the inline module defines itself wins over one it imports under the same
/// name.
#[salsa_test]
fn inline_module_definition_wins_over_an_imported_ability(db: &salsa::DatabaseImpl) {
    use tribute_front::ast::{Decl, TypeAnnotationKind};

    let source = SourceCst::from_source_str(
        db,
        "use_imports.trb",
        r#"
mod outer {
    mod fx {
        pub ability Tick {
            op tick() -> Nat
        }
    }
    pub ability Tick {
        op other() -> Nat
    }
    use fx::Tick

    pub fn noop() ->{Tick} Nat {
        1
    }
}
"#,
    );
    let module = tribute_front::query::resolved_module(db, source).expect("module resolves");
    let effects = module
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Module(outer) => outer.body.as_ref(),
            _ => None,
        })
        .and_then(|body| {
            body.iter().find_map(|decl| match decl {
                Decl::Function(function) if function.name == "noop" => function.effects.clone(),
                _ => None,
            })
        })
        .expect("noop declares effects");
    assert!(
        matches!(&effects[0].kind, TypeAnnotationKind::Named(name) if *name == "Tick"),
        "{effects:?}"
    );
}

/// An inline module may import a package-root ability under an alias; the
/// alias names the package-root ability in effect annotations, unqualified
/// operation calls, and handler arms.
#[salsa_test]
fn inline_module_import_aliases_a_root_ability(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
ability Tick {
    op tick() -> Nat
}
mod outer {
    use super::Tick as T

    pub fn count() ->{T} Nat {
        tick()
    }

    pub fn run() -> Nat {
        handle count() {
            do result { result }
            op T::tick() { resume 1 }
        }
    }
}

fn main() -> Nil {
    let _ = outer::run()
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

#[salsa_test]
fn imported_abilities_resolve_inside_function_type_annotations(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod abilities {
    pub ability Ask {
        fn ask() -> Int
    }
}
use abilities::Ask
use std::io::{Io, print_line}

struct Hook { run: fn(String) ->{Io} String }

fn io_once(f: fn(String) ->{Io} String, s: String) ->{Io} String { f(s) }

fn asking(g: fn(Int) ->{Ask} Int) ->{Ask} Int { g(+5) }

fn echo(s: String) ->{Io} String {
    print_line(s)
    s
}

fn hooked(hook: Hook) ->{Io} String {
    io_once(hook.run, "hook")
}

fn main() ->{Io} Nil {
    print_line(io_once(echo, "io"))
    print_line(hooked(Hook { run: echo }))
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

const SIZES: &str = r#"
struct A { n: Nat }
struct B { n: Nat }
mod a {
    pub fn size(x: super::A) -> Nat { x.n }
    pub fn make() -> Nat { 1 }
}
mod b {
    pub fn size(x: super::B) -> Nat { x.n + 100 }
    pub fn make() -> Nat { 2 }
}
"#;

/// A name several `use`s give different functions is called with the one
/// whose first parameter takes the first argument, like a method call.
#[salsa_test]
fn a_call_selects_among_imported_functions_by_its_first_argument(db: &salsa::DatabaseImpl) {
    for body in [
        // At the package root, also for a first argument typed after solving.
        "use a::size\nuse b::size\n\
         fn main() -> Nil {\n    \
             let _ = size(A { n: 1 }) + size(B { n: 2 }) + A { n: 3 }.size\n    \
             let late = fn(x) { size(x) }\n    \
             let _ = late(B { n: 4 })\n}\n",
        // In an inline module.
        "mod inner {\n    use super::a::size\n    use super::b::size\n    \
             pub fn both(x: super::A, y: super::B) -> Nat { size(x) + size(y) }\n}\n\
         fn main() -> Nil {\n    let _ = inner::both(A { n: 1 }, B { n: 2 })\n}\n",
        // A declaration or a local of that name is what the name means.
        "use a::size\nuse b::size\n\
         fn size(x: Nat) -> Nat { x }\n\
         fn main() -> Nil {\n    let _ = size(1)\n}\n",
        "use a::size\nuse b::size\n\
         fn main() -> Nil {\n    let size = fn(x: Nat) { x }\n    let _ = size(1)\n}\n",
        // An alias is the name the functions are imported under.
        "use a::size as measure\nuse b::size as measure\n\
         fn main() -> Nil {\n    let _ = measure(A { n: 1 }) + measure(B { n: 2 })\n}\n",
        // One function imported twice is one function.
        "use a::make\nuse a::make\n\
         fn main() -> Nil {\n    let _ = make()\n}\n",
    ] {
        let errors = errors(db, &format!("{SIZES}{body}"));
        assert!(errors.is_empty(), "{body}: {errors:#?}");
    }
}

/// Without a first argument whose type selects one, a name that imports
/// several functions has to be written as a path.
#[salsa_test]
fn an_unselected_imported_function_is_reported(db: &salsa::DatabaseImpl) {
    for (body, expected) in [
        (
            "use a::make\nuse b::make\n\
             fn main() -> Nil {\n    let _ = make()\n}\n",
            "`make` imports several functions and is called without an argument, \
             so no argument type selects one; name the function by its path",
        ),
        (
            "use a::size\nuse b::size\n\
             fn main() -> Nil {\n    let _ = size\n}\n",
            "`size` imports several functions and is not called, \
             so no argument type selects one; name the function by its path",
        ),
        // A call with the wrong number of arguments still names the function
        // its arguments select.
        (
            "use a::size\nuse b::size\n\
             fn main() -> Nil {\n    let _ = size(A { n: 1 }, 2)\n}\n",
            "call arity mismatch: expected 1 argument, found 2",
        ),
        (
            "use a::size\nuse b::size\n\
             fn main() -> Nil {\n    let _ = size(1)\n}\n",
            "unresolved path `size`: no function it names takes arguments of types (`Nat`)",
        ),
    ] {
        assert_eq!(errors(db, &format!("{SIZES}{body}")), [expected], "{body}");
    }
}

const PAIRS: &str = r#"
struct A { n: Nat }
mod a {
    pub fn pair(x: super::A, y: Nat) -> Nat { x.n + y }
    pub fn pick(x: Nat) -> Nat { x }
    pub fn count(x: super::A) -> Nat { x.n }
}
mod b {
    pub fn pair(x: super::A, y: String) -> Nat { x.n }
    pub fn pick(x: Nat) -> String { "picked" }
    pub fn count(x: super::A, y: Nat) -> Nat { x.n + y }
}
use a::pair
use b::pair
use a::count
use b::count
use a::pick
use b::pick
fn text(s: String) -> String { s }
"#;

/// Every argument of a call and the type its result is used at select the
/// function, in either call syntax.
#[salsa_test]
fn a_call_selects_a_function_by_all_its_arguments_and_its_result(db: &salsa::DatabaseImpl) {
    for body in [
        // The second argument, whose first parameter the functions share.
        "let _ = pair(A { n: 1 }, 2) + pair(A { n: 1 }, \"two\")",
        "let _ = A { n: 1 }.pair(2) + A { n: 1 }.pair(\"two\")",
        // An argument typed after solving waits for the others to select.
        "let late = fn(x) { pair(x, 5) }\n    let _ = late(A { n: 1 })",
        // The number of arguments.
        "let _ = count(A { n: 1 }) + count(A { n: 1 }, 2)",
        // The result, used as a `String` and as a `Nat`.
        "let _ = text(pick(1))\n    let _ = pick(7) + 1",
    ] {
        let errors = errors(db, &format!("{PAIRS}fn main() -> Nil {{\n    {body}\n}}\n"));
        assert!(errors.is_empty(), "{body}: {errors:#?}");
    }
}

/// A call that no argument or use of its result decides, or that no
/// function takes, is reported with the types it has.
#[salsa_test]
fn a_call_its_types_do_not_decide_is_reported(db: &salsa::DatabaseImpl) {
    for (body, expected) in [
        (
            "let _ = pick(1)",
            "ambiguous path `pick` for arguments of types (`Nat`): it names `a::pick`, `b::pick`",
        ),
        (
            "let _ = pair(A { n: 1 }, True)",
            "unresolved path `pair`: no function it names takes arguments of types (`A`, `Bool`)",
        ),
    ] {
        assert_eq!(
            errors(db, &format!("{PAIRS}fn main() -> Nil {{\n    {body}\n}}\n")),
            [expected],
            "{body}"
        );
    }
}

/// A method call and the call it stands for select the same function and
/// report the same errors.
#[salsa_test]
fn method_syntax_and_call_syntax_are_checked_alike(db: &salsa::DatabaseImpl) {
    const DECLARATIONS: &str = r#"
struct A { n: Nat }
struct B { n: Nat }
fn pair(x: A, y: Nat) -> Nat { x.n + y }
mod a {
    pub fn tag(x: super::A, s: String) -> Nat { x.n }
}
mod b {
    pub fn tag(x: super::B, s: String) -> Nat { x.n }
}
use a::tag
use b::tag
"#;
    let check = |body: &str| {
        errors(
            db,
            &format!("{DECLARATIONS}fn main() -> Nil {{\n    {body}\n}}\n"),
        )
    };
    // A receiver not typed yet is an argument like the others: the rest of
    // the call selects, and the receiver's type follows.
    for body in [
        "let late = fn(x) { x.pair(5) }\n    let _ = late(A { n: 1 })",
        "let late = fn(x) { pair(x, 5) }\n    let _ = late(A { n: 1 })",
        "let late = fn(x) { x.tag(\"t\") }\n    let _ = late(B { n: 1 })",
        "let late = fn(x) { tag(x, \"t\") }\n    let _ = late(B { n: 1 })",
        // Nothing else types these receivers.
        "let _ = fn(x) { x.pair(5) }",
        "let _ = fn(x) { pair(x, 5) }",
    ] {
        let errors = check(body);
        assert!(errors.is_empty(), "{body}: {errors:#?}");
    }
    for (method, call) in [
        ("let _ = A { n: 1 }.pair()", "let _ = pair(A { n: 1 })"),
        (
            "let _ = A { n: 1 }.pair(1, 2)",
            "let _ = pair(A { n: 1 }, 1, 2)",
        ),
        ("let _ = 1.pair(2)", "let _ = pair(1, 2)"),
        (
            "let _ = A { n: 1 }.pair(True)",
            "let _ = pair(A { n: 1 }, True)",
        ),
    ] {
        let reported = check(method);
        assert!(!reported.is_empty(), "{method}");
        assert_eq!(reported, check(call), "{method}");
    }
}

/// Reading a field is calling its getter: `x.f` selects like any other
/// call, also before its receiver is typed, and a struct's own getter comes
/// before another function of the field's name.
#[salsa_test]
fn a_field_read_is_a_call_of_its_getter(db: &salsa::DatabaseImpl) {
    const DECLARATIONS: &str = r#"
struct Name { text: String }
struct Label { text: String, size: Nat }
fn text(name: Name) -> String { name.text }
fn width(s: String) -> Nat { 1 }
"#;
    let check = |body: &str| {
        errors(
            db,
            &format!("{DECLARATIONS}fn main() -> Nil {{\n    {body}\n}}\n"),
        )
    };
    for body in [
        // One struct has the field, so the getter types the receiver.
        "let _ = fn(l) { l.size }",
        "let size = fn(l) { l.size }\n    let _ = size(Label { text: \"a\", size: 1 })",
        // Two structs and a function share the name; the receiver decides.
        "let read = fn(n) { n.text }\n    let _ = read(Label { text: \"a\", size: 1 })",
        "let _ = Name { text: \"a\" }.text",
        "let _ = text(Name { text: \"a\" })",
    ] {
        let errors = check(body);
        assert!(errors.is_empty(), "{body}: {errors:#?}");
    }
}
