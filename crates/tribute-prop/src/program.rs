use std::fmt::Write;

use proptest::prelude::*;

use crate::env::{Env, Goal};
use crate::expr::fragment;
use crate::fragment::Fragment;
use crate::value::{Ty, ty};

/// One `fn check_i() -> T` of a [`Program`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    pub name: String,
    pub body: Fragment,
}

/// Independent checks that `main` prints with `print_line`, one per line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub checks: Vec<Check>,
}

impl Program {
    /// The program's source text.
    pub fn source(&self) -> String {
        let mut src = String::from("use std::io::{Io, print_line}\n\n");
        if self.checks.iter().any(|check| check.body.ty() == Ty::Bool) {
            src.push_str(
                "fn bool_text(value: Bool) -> String {\n    \
                 case value {\n        True -> \"True\"\n        False -> \"False\"\n    }\n}\n\n",
            );
        }
        for check in &self.checks {
            write!(
                src,
                "fn {}() -> {} {{\n    {}\n}}\n\n",
                check.name,
                check.body.ty(),
                check.body.src
            )
            .expect("writing to a String");
        }
        src.push_str("fn main() ->{Io} Nil {\n");
        for check in &self.checks {
            let call = format!("{}()", check.name);
            let text = match check.body.ty() {
                Ty::Nat => format!("{call}.to_string"),
                Ty::Bool => format!("bool_text({call})"),
                Ty::String => call,
            };
            writeln!(src, "    print_line({text})").expect("writing to a String");
        }
        src.push_str("}\n");
        src
    }

    /// The exact stdout the program prints.
    pub fn expected_stdout(&self) -> String {
        self.checks
            .iter()
            .map(|check| format!("{}\n", check.body.value))
            .collect()
    }
}

/// Strategy for a program of one to eight checks, each nested at most
/// three levels.
pub fn program() -> impl Strategy<Value = Program> {
    program_with(1..=8, 3)
}

/// Strategy for a program with `checks` checks, each nested at most `depth`
/// levels.
pub fn program_with(
    checks: impl Into<prop::collection::SizeRange>,
    depth: u32,
) -> impl Strategy<Value = Program> {
    let body = ty().prop_flat_map(move |ty| fragment(Env::default(), Goal::of(ty), depth));
    prop::collection::vec(body, checks).prop_map(|bodies| Program {
        checks: bodies
            .into_iter()
            .enumerate()
            .map(|(i, body)| Check {
                name: format!("check_{i}"),
                body,
            })
            .collect(),
    })
}
