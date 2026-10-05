//! Generated field setters and modifiers (`T::f::set`, `T::f::modify`) and
//! the qualified UFCS calls that reach them (`x.f::set(v)`).

use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_with_diagnostics;
use tribute_front::SourceCst;

use crate::common::assert_native_output;

#[test]
fn test_native_field_setter_and_modifier() {
    // The examples of `new-plans/syntax.md` and `new-plans/types.md`.
    assert_native_output(
        "field_lenses.trb",
        r#"
use std::io::{Io, print_line}

struct User { name: String, age: Int }

fn show(user: User) ->{Io} Nil {
    print_line(user.name)
    print_line(Int::to_string(user.age))
}

fn main() ->{Io} Nil {
    let user = User { name: "John", age: +30 }
    show(user.name::set("Jane"))
    show(user.age::modify(fn(n) n + +1))
    show(User::name::set(user, "Path"))
    show(User::age::modify(user, fn(n) n - +1))
    show(user)
}
"#,
        "Jane\n30\nJohn\n31\nPath\n30\nJohn\n29\nJohn\n30",
    );
}

#[test]
fn test_native_field_lens_chain() {
    assert_native_output(
        "field_lens_chain.trb",
        r#"
use std::io::{Io, print_line}

struct User { name: String, age: Int }

fn main() ->{Io} Nil {
    let user = User { name: "John", age: +30 }
    let updated = user.name::set("Jane").age::modify(fn(n) n + +1)
    print_line(updated.name)
    print_line(Int::to_string(updated.age))
    print_line(Int::to_string(updated.User::age))
}
"#,
        "Jane\n31\n31",
    );
}

#[test]
fn test_native_generic_struct_field_lenses() {
    // The struct's own parameter `e` shares its name with the row variable a
    // modifier would otherwise use.
    assert_native_output(
        "field_lens_generic.trb",
        r#"
use std::io::{Io, print_line}

struct Either(e, a) { left: e, right: a, count: Int }

fn main() ->{Io} Nil {
    let either = Either { left: "l", right: "r", count: +1 }
    let updated = either.left::set("L").right::modify(fn(r) r <> "R").count::set(+2)
    print_line(updated.left <> updated.right)
    print_line(Int::to_string(updated.count))
}
"#,
        "LrR\n2",
    );
}

#[test]
fn test_native_field_modifier_with_effectful_callback() {
    assert_native_output(
        "field_lens_effects.trb",
        r#"
use std::io::{Io, print_line}

struct Name { text: String }
struct Score { points: Int }

ability Counter {
    fn tick() -> Int
}

fn bump(n: Int) ->{Counter} Int { n + Counter::tick() }

fn main() ->{Io} Nil {
    let name = Name { text: "io" }.text::modify(fn(text) {
        print_line("callback")
        text <> "!"
    })
    print_line(name.text)
    let score = handle Score { points: +1 }.points::modify(bump) {
        do result { result }
        fn Counter::tick() { +41 }
    }
    print_line(Int::to_string(score.points))
}
"#,
        "callback\nio!\n42",
    );
}

#[test]
fn test_native_field_lenses_through_modules() {
    // A field whose type is a sibling of the struct, a call inside the
    // struct's module, a companion module beside the struct, and a
    // call from a module that imports the struct.
    assert_native_output(
        "field_lens_modules.trb",
        r#"
use std::io::{Io, print_line}
use geometry::Line

mod geometry {
    pub struct Point { x: Int, y: Int }
    pub struct Line { start: Point, label: String }

    pub mod Line {
        pub fn describe(line: Line) -> String { line.label }
        pub fn renamed(line: Line) -> Line { line.label::modify(fn(label) label <> "!") }
    }

    pub fn origin() -> Line {
        Line { start: Point { x: +0, y: +0 }, label: "origin" }
    }

    pub fn start_x(line: Line) -> Int { line.start.x }

    pub fn shifted(line: Line) -> Line {
        line.start::modify(fn(point) point.x::set(+7))
    }
}

mod client {
    use pkg::geometry::Line

    pub fn relabel(line: Line) -> Line { line.label::set("client") }
}

fn main() ->{Io} Nil {
    let line = geometry::shifted(geometry::origin()).label::set("moved")
    print_line(Line::describe(line))
    print_line(Int::to_string(geometry::start_x(line)))
    print_line(Line::describe(client::relabel(line)))
    print_line(Line::describe(Line::renamed(line)))
}
"#,
        "moved\n7\nclient\nmoved!",
    );
}

#[test]
fn test_native_field_functions_as_paths_and_values() {
    // A getter called by its path, and a setter and a modifier used as
    // function values.
    assert_native_output(
        "field_function_values.trb",
        r#"
use std::io::{Io, print_line}

struct User { name: String, age: Int }

fn show(user: User) ->{Io} Nil {
    print_line(user.name)
    print_line(Int::to_string(User::age(user)))
}

fn rename(update: fn(User, String) ->{} User, user: User) -> User { update(user, "Passed") }

fn main() ->{Io} Nil {
    let user = User { name: "John", age: +30 }
    let set = User::name::set
    show(set(user, "Bound"))
    show(rename(User::name::set, user))
    let modify = User::age::modify
    show(modify(user, fn(n) n + +1))
}
"#,
        "Bound\n30\nPassed\n30\nJohn\n31",
    );
}

/// The names of the functions with bodies after the shared middle-end.
fn shared_function_names(db: &salsa::DatabaseImpl, code: &str) -> Vec<String> {
    use trunk_ir::dialect::func;
    use trunk_ir::ops::DialectOp;

    let source = SourceCst::from_source_str(db, "test.trb", code);
    let frontend =
        tribute::pipeline::compile_frontend_for_shared_route(db, source).expect("frontend output");
    let (ctx, module) =
        tribute::pipeline::run_shared_middle_end(frontend).expect("shared middle-end");
    module
        .ops(&ctx)
        .iter()
        .filter_map(|&op| func::Func::from_op(&ctx, op).ok())
        .filter(|function| ctx.op_has_regions(function.op_ref()))
        .map(|function| function.sym_name(&ctx).to_owned())
        .collect()
}

#[salsa_test]
fn unreferenced_field_functions_do_not_reach_cps(db: &salsa::DatabaseImpl) {
    let names = shared_function_names(
        db,
        r#"
struct User { name: String, age: Int }

fn unused(user: User) -> User { user.age::set(+1) }

fn main() -> Nil {
    let _ = User { name: "John", age: +30 }.name::set("Jane")
}
"#,
    );
    assert!(
        names.iter().any(|name| name == "User::name::set"),
        "{names:?}"
    );
    for dropped in [
        "User::name::modify",
        "User::age::set",
        "User::age::modify",
        "unused",
    ] {
        assert!(
            !names.iter().any(|name| name == dropped),
            "{dropped}: {names:?}"
        );
    }
    assert!(
        !names
            .iter()
            .any(|name| name.starts_with("std::io::SystemError::")),
        "{names:?}"
    );
}

#[test]
fn test_native_prelude_struct_field_lenses() {
    // A prelude struct's field functions are reachable from user code.
    assert_native_output(
        "field_lens_prelude.trb",
        r#"
use std::io::{Io, SystemError, print_line}

fn main() ->{Io} Nil {
    let error = SystemError { code: +2, message: "missing" }
    let updated = error.message::modify(fn(message) message <> "!").code::set(+3)
    print_line(updated.message)
    print_line(Int::to_string(updated.code))
}
"#,
        "missing!\n3",
    );
}

#[test]
fn test_native_method_path_with_generic_and_structural_receivers() {
    // A type variable takes any receiver; function and tuple parameters take
    // receivers of their shape.
    assert_native_output(
        "method_path_receivers.trb",
        r#"
use std::io::{Io, print_line}

mod util {
    pub fn id(value: a) -> a { value }
    pub fn apply(function: fn(Int) ->{} Int, value: Int) -> Int { function(value) }
    pub fn first(pair: #(Int, String)) -> Int {
        let #(number, _) = pair
        number
    }
}

fn double(n: Int) -> Int { n + n }

fn main() ->{Io} Nil {
    print_line(Int::to_string(+5.util::id()))
    print_line("text".util::id())
    print_line(Int::to_string(double.util::apply(+4)))
    print_line(Int::to_string(#(+7, "seven").util::first()))
}
"#,
        "5\ntext\n8\n7",
    );
}

fn messages(db: &salsa::DatabaseImpl, code: &str) -> Vec<(String, String)> {
    let source = SourceCst::from_source_str(db, "test.trb", code);
    compile_with_diagnostics(db, source)
        .diagnostics
        .into_iter()
        .map(|diagnostic| {
            let span = diagnostic.inner.span;
            (
                diagnostic.inner.message,
                code[span.start..span.end].to_owned(),
            )
        })
        .collect()
}

#[salsa_test]
fn diag_unresolved_method_path(db: &salsa::DatabaseImpl) {
    let diagnostics = messages(
        db,
        r#"
struct User { name: String }

fn main() -> Nil {
    let user = User { name: "John" }
    let _ = user.nope::set("Jane")
    let _ = +1.name::set("Jane")
}
"#,
    );
    assert_eq!(
        diagnostics,
        [
            (
                "unresolved name `nope::set`".to_owned(),
                "nope::set".to_owned()
            ),
            (
                "unresolved path `name::set`: no function it names takes a receiver of type `Int`"
                    .to_owned(),
                "name::set".to_owned()
            ),
        ]
    );
}

#[salsa_test]
fn diag_method_path_outside_use_scope(db: &salsa::DatabaseImpl) {
    // The module that defines the receiver's type is not searched unless it
    // is in the call's scope.
    let diagnostics = messages(
        db,
        r#"
mod account {
    pub struct User { name: String }
    pub fn guest() -> User { User { name: "guest" } }
}

fn main() -> Nil {
    let _ = account::guest().name::set("Jane")
}
"#,
    );
    assert_eq!(
        diagnostics,
        [(
            "unresolved name `name::set`".to_owned(),
            "name::set".to_owned()
        )]
    );
}

#[salsa_test]
fn diag_ambiguous_method_path(db: &salsa::DatabaseImpl) {
    let diagnostics = messages(
        db,
        r#"
struct User { name: String }

mod audit {
    pub mod name {
        pub fn set(user: pkg::User, value: String) -> pkg::User { user }
    }
}
use audit::name

fn main() -> Nil {
    let user = User { name: "John" }
    let _ = user.name::set("Jane")
}
"#,
    );
    assert_eq!(
        diagnostics,
        [(
            "ambiguous path `name::set` for a receiver of type `User`: it names \
             `User::name::set`, `audit::name::set`"
                .to_owned(),
            "name::set".to_owned()
        )]
    );
}

#[salsa_test]
fn diag_ambiguous_getter_path(db: &salsa::DatabaseImpl) {
    // `x.User::name` reads the field only when the getter is the one function
    // its path names for the receiver.
    let diagnostics = messages(
        db,
        r#"
struct User { name: String }

mod audit {
    pub mod User {
        pub fn name(user: pkg::User) -> String { "audit" }
    }
}
use audit

fn main() -> Nil {
    let user = User { name: "John" }
    let _ = user.User::name
}
"#,
    );
    assert_eq!(
        diagnostics,
        [(
            "ambiguous path `User::name` for a receiver of type `User`: it names \
             `User::name`, `audit::User::name`"
                .to_owned(),
            "User::name".to_owned()
        )]
    );
}

#[salsa_test]
fn diag_companion_redefines_field_lens(db: &salsa::DatabaseImpl) {
    let diagnostics = messages(
        db,
        r#"
struct User { name: String, age: Int }

mod User {
    pub mod name {
        pub fn set(user: pkg::User, value: String) -> pkg::User { user }
        pub fn clear(user: pkg::User) -> pkg::User { user }
    }
    pub mod age {
        pub fn modify(user: pkg::User, update: fn(Int) -> Int) -> pkg::User { user }
    }
}

fn main() -> Nil {
    let user = User { name: "John", age: +30 }
    let _ = user.name::clear().name::set("Jane")
}
"#,
    );
    let messages: Vec<_> = diagnostics
        .into_iter()
        .map(|(message, _)| message)
        .collect();
    assert_eq!(
        messages,
        [
            "duplicate definition of `User::name::set`: struct `User` defines it for its field `name`",
            "duplicate definition of `User::age::modify`: struct `User` defines it for its field `age`",
        ]
    );
}
