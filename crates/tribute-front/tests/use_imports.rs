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
    pub mod inner {
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

/// A name an unresolved `use` imports is reported once, at the `use`.
#[salsa_test]
fn names_of_unresolved_imports_are_not_reported_again(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use nowhere::thing

fn main() -> Nil {
    let _ = thing()
}
"#,
    );
    assert_eq!(errors, ["unresolved import `nowhere::thing`"]);
}

/// A `use` of one segment names the package root's own module, which is
/// not a value.
#[salsa_test]
fn a_one_segment_import_names_the_root_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn one() -> Nat { 1 }
}
use basic

fn main() -> Nil {
    let _ = basic
}
"#,
    );
    assert_eq!(errors, ["expected a value, found module `basic`"]);
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
    pub mod a {{
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
    pub mod a {
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
/// expressions, ability annotations, handler arms, and unqualified operation
/// calls under an ability annotation.
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

/// An ability annotation may reach an ability through a module an inline
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

/// Ability annotations follow the same precedence as other names: an ability
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
        .entries
        .iter()
        .find_map(|entry| match &entry.decl {
            Decl::Function(function)
                if entry.module == ["pkg", "outer"] && function.name == "noop" =>
            {
                function.effects.clone()
            }
            _ => None,
        })
        .expect("noop declares effects");
    assert!(
        matches!(&effects[0].kind, TypeAnnotationKind::Named(name) if *name == "Tick"),
        "{effects:?}"
    );
}

/// An inline module may import a package-root ability under an alias; the
/// alias names the package-root ability in ability annotations, unqualified
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
    use super::A
    pub fn size(x: super::A) -> Nat { x.n }
    pub fn make() -> Nat { 1 }
}
mod b {
    use super::B
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
        // A method is a declared function, whatever a local is called.
        "use a::size\nuse b::size\n\
         fn first() -> Nat {\n    let size = 1\n    A { n: 1 }.size\n}\n\
         fn second() -> Nat { A { n: 1 }.size + B { n: 2 }.size }\n\
         fn main() -> Nil { }\n",
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

/// A name several `use`s give different functions, used as a value, is the
/// function whose type the value is used at.
#[salsa_test]
fn a_function_value_is_selected_by_the_type_it_is_used_at(db: &salsa::DatabaseImpl) {
    const USES: &str = "use a::size\nuse b::size\n\
        fn apply(x: t, f: fn(t) ->{e} u) ->{e} u { f(x) }\n\
        fn on_b(f: fn(B) ->{e} Nat) ->{e} Nat { f(B { n: 2 }) }\n";
    for body in [
        // The parameter's type, also one solved from another argument.
        "fn main() -> Nil {\n    let _ = on_b(size)\n}\n",
        "fn main() -> Nil {\n    let _ = apply(A { n: 1 }, size) + apply(B { n: 2 }, size)\n}\n",
        // The declared result.
        "fn pick() -> fn(A) -> Nat { size }\nfn main() -> Nil { }\n",
        // A local takes the type its use gives it.
        "fn main() -> Nil {\n    let f = size\n    let _ = f(A { n: 1 })\n}\n",
        "fn main() -> Nil {\n    let f = size\n    let _ = apply(A { n: 1 }, f)\n}\n",
    ] {
        let errors = errors(db, &format!("{SIZES}{USES}{body}"));
        assert!(errors.is_empty(), "{body}: {errors:#?}");
    }
    for (body, expected) in [
        (
            "fn main() -> Nil {\n    let _ = apply(1, size)\n}\n",
            "unresolved path `size`: no function it names has type `fn(Nat) -> _`",
        ),
        (
            "fn main() -> Nil {\n    let _ = size\n}\n",
            "ambiguous path `size` used as a value: it names `a::size`, `b::size`",
        ),
        // The result has to fit too; no function is picked by its
        // parameters alone.
        (
            "fn text(f: fn(A) ->{e} String) ->{e} String { f(A { n: 1 }) }\n\
             fn main() -> Nil {\n    let _ = text(size)\n}\n",
            "unresolved path `size`: no function it names has type `fn(A) -> std::String`",
        ),
        // A function type both fit selects neither.
        (
            "fn any(f: fn(t) ->{e} Nat) -> Nat { 0 }\n\
             fn main() -> Nil {\n    let _ = any(size)\n}\n",
            "ambiguous path `size` used as a value of type `fn(_) -> Nat`: \
             it names `a::size`, `b::size`",
        ),
        // A local is one function, selected by its first use.
        (
            "fn main() -> Nil {\n    let f = size\n    let _ = f(A { n: 1 }) + f(B { n: 2 })\n}\n",
            "type error at call site in function 'main': expected `A`, found `B`",
        ),
    ] {
        assert_eq!(
            errors(db, &format!("{SIZES}{USES}{body}")),
            [expected],
            "{body}"
        );
    }
}

/// Where no type of the call selects one, a name that imports several
/// functions has to be written as a path.
#[salsa_test]
fn an_unselected_imported_function_is_reported(db: &salsa::DatabaseImpl) {
    for (body, expected) in [
        (
            "use a::make\nuse b::make\n\
             fn main() -> Nil {\n    let _ = make()\n}\n",
            "ambiguous path `make` for no arguments: it names `a::make`, `b::make`",
        ),
        (
            "use a::size\nuse b::size\n\
             fn main() -> Nil {\n    let _ = size\n}\n",
            "ambiguous path `size` used as a value: it names `a::size`, `b::size`",
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
    use super::A
    pub fn pair(x: super::A, y: Nat) -> Nat { x.n + y }
    pub fn pick(x: Nat) -> Nat { x }
    pub fn count(x: super::A) -> Nat { x.n }
}
mod b {
    use super::A
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
    use super::A
    pub fn tag(x: super::A, s: String) -> Nat { x.n }
}
mod b {
    use super::B
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
    // The same errors, each counted the way its call is written.
    for (body, expected) in [
        (
            "let _ = A { n: 1 }.pair()",
            "call arity mismatch: expected 1 argument after the receiver, found 0",
        ),
        (
            "let _ = pair(A { n: 1 })",
            "call arity mismatch: expected 2 arguments, found 1",
        ),
        (
            "let _ = A { n: 1 }.pair(1, 2)",
            "call arity mismatch: expected 1 argument after the receiver, found 2",
        ),
        (
            "let _ = pair(A { n: 1 }, 1, 2)",
            "call arity mismatch: expected 2 arguments, found 3",
        ),
    ] {
        assert_eq!(check(body), [expected], "{body}");
    }
    for (method, call) in [
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
/// call, also before its receiver is typed. The struct's own module cannot declare
/// the getter's name again; elsewhere it is another function.
#[salsa_test]
fn a_field_read_is_a_call_of_its_getter(db: &salsa::DatabaseImpl) {
    const DECLARATIONS: &str = r#"
struct Name { text: String }
struct Label { text: String, size: Nat }
fn shown(name: Name) -> String { name.text }
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
        // Two structs have the field; the receiver decides.
        "let read = fn(n) { n.text }\n    let _ = read(Label { text: \"a\", size: 1 })",
        "let _ = Name { text: \"a\" }.text",
        "let _ = shown(Name { text: \"a\" })",
    ] {
        let errors = check(body);
        assert!(errors.is_empty(), "{body}: {errors:#?}");
    }
    // The getter is `Name::text`, so the struct's own module cannot declare
    // that name again, whatever the function takes.
    assert_eq!(
        errors(
            db,
            "struct Name { text: String }\n\
             pub mod Name {\n\
                 pub fn text(name: Name, suffix: String) -> String { suffix }\n\
             }\n\
             fn main() -> Nil { }\n",
        ),
        ["function `Name::text` is already declared by field `text` of struct `Name`"],
    );
    // Another module's function is another name. Where it takes the struct
    // too, a path says which of the two a read means.
    const AUDITED: &str = "struct Name { text: String }\n\
         mod audit {\n\
             pub fn text(name: pkg::Name) -> String { \"audit\" }\n\
             pub fn size(text: String) -> Nat { 1 }\n\
         }\n";
    for body in [
        "let _ = Name { text: \"a\" }.Name::text",
        "let _ = Name { text: \"a\" }.audit::text",
        "let _ = \"a\".size",
    ] {
        let errors = errors(db, &format!("{AUDITED}fn main() -> Nil {{ {body} }}\n"));
        assert!(errors.is_empty(), "{body}: {errors:#?}");
    }
}
