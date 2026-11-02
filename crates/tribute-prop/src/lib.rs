//! Typed, value-carrying Tribute source strategies for property-based testing.
//!
//! Each strategy yields a [`Fragment`]: a piece of Tribute source text with
//! the value it evaluates to, and so its type. The strategies emit text
//! directly; there is no term model or printer. Every compound expression is
//! wrapped in `{ … }`, so no precedence handling is needed.
//!
//! Generation is top-down and type-directed: a [`Goal`] names the type to
//! produce, and for `Nat` an upper bound on its value. `+` and `*` choose
//! their right operand within the bound the left one leaves, so every
//! intermediate value stays in the current unsigned 32-bit range
//! (`new-plans/numeric-types.md`) by construction.
//!
//! This crate holds strategies only. Properties live next to the code they
//! test.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt::{self, Write};

use proptest::prelude::*;

/// A type the generator produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Nat,
    Bool,
    String,
}

impl Ty {
    /// The type's name in Tribute source.
    pub fn name(self) -> &'static str {
        match self {
            Ty::Nat => "Nat",
            Ty::Bool => "Bool",
            Ty::String => "String",
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The value a fragment evaluates to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Nat(u32),
    Bool(bool),
    String(String),
}

impl Value {
    pub fn ty(&self) -> Ty {
        match self {
            Value::Nat(_) => Ty::Nat,
            Value::Bool(_) => Ty::Bool,
            Value::String(_) => Ty::String,
        }
    }
}

/// The text a [`Program`] prints for a value: decimal for `Nat`, the
/// constructor name for `Bool`, and the text itself for `String`.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nat(n) => write!(f, "{n}"),
            Value::Bool(true) => f.write_str("True"),
            Value::Bool(false) => f.write_str("False"),
            Value::String(s) => f.write_str(s),
        }
    }
}

/// A piece of source text and the value it evaluates to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fragment {
    pub src: String,
    pub value: Value,
    /// The names the text refers to that it does not bind itself.
    pub free: BTreeSet<String>,
}

impl Fragment {
    pub fn ty(&self) -> Ty {
        self.value.ty()
    }

    fn leaf(src: String, value: Value) -> Self {
        Self {
            src,
            value,
            free: BTreeSet::new(),
        }
    }

    fn compound<'a>(
        src: String,
        value: Value,
        children: impl IntoIterator<Item = &'a Fragment>,
    ) -> Self {
        let free = children
            .into_iter()
            .flat_map(|child| child.free.iter().cloned())
            .collect();
        Self { src, value, free }
    }
}

/// One `let` binding in scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub value: Value,
}

/// The bindings in scope, innermost last.
///
/// [`Env::bind`] names each binding after its depth, so a name is never
/// shadowed and a reference reads the value its binding holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Env {
    bindings: Vec<Binding>,
}

impl Env {
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    pub fn lookup(&self, name: &str) -> Option<&Value> {
        self.bindings
            .iter()
            .rev()
            .find(|binding| binding.name == name)
            .map(|binding| &binding.value)
    }

    /// This env with a fresh name bound to `value`, and that name.
    pub fn bind(&self, value: Value) -> (String, Env) {
        let name = format!("v{}", self.bindings.len());
        let mut env = self.clone();
        env.bindings.push(Binding {
            name: name.clone(),
            value,
        });
        (name, env)
    }

    /// The bindings a reference may use for `goal`.
    fn candidates(&self, goal: Goal) -> Vec<Binding> {
        self.bindings
            .iter()
            .filter(|binding| goal.accepts(&binding.value))
            .cloned()
            .collect()
    }
}

/// What a strategy produces: a type and, for `Nat`, the largest value
/// allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Goal {
    Nat { max: u32 },
    Bool,
    String,
}

impl Goal {
    /// A `Nat` goal over the whole current range.
    pub const NAT: Goal = Goal::Nat { max: u32::MAX };

    pub fn of(ty: Ty) -> Self {
        match ty {
            Ty::Nat => Self::NAT,
            Ty::Bool => Self::Bool,
            Ty::String => Self::String,
        }
    }

    pub fn ty(self) -> Ty {
        match self {
            Goal::Nat { .. } => Ty::Nat,
            Goal::Bool => Ty::Bool,
            Goal::String => Ty::String,
        }
    }

    pub fn accepts(self, value: &Value) -> bool {
        match (self, value) {
            (Goal::Nat { max }, Value::Nat(n)) => *n <= max,
            (Goal::Bool, Value::Bool(_)) | (Goal::String, Value::String(_)) => true,
            _ => false,
        }
    }
}

/// Strategy for a type the generator produces.
pub fn ty() -> impl Strategy<Value = Ty> {
    prop_oneof![Just(Ty::Nat), Just(Ty::Bool), Just(Ty::String)]
}

/// Strategy for a fragment that meets `goal` in `env`, nested at most
/// `depth` levels.
pub fn fragment(env: Env, goal: Goal, depth: u32) -> BoxedStrategy<Fragment> {
    let leaf = leaf(&env, goal);
    if depth == 0 {
        return leaf;
    }
    let depth = depth - 1;
    let compound = match goal {
        Goal::Nat { max } => prop_oneof![
            add(env.clone(), max, depth),
            mul(env.clone(), max, depth),
            block(env.clone(), goal, depth),
            case(env, goal, depth),
        ]
        .boxed(),
        Goal::Bool => prop_oneof![
            compare_nat(env.clone(), depth),
            compare_string(env.clone(), depth),
            block(env.clone(), goal, depth),
            case(env, goal, depth),
        ]
        .boxed(),
        Goal::String => prop_oneof![
            concat(env.clone(), depth),
            nat_to_string(env.clone(), depth),
            block(env.clone(), goal, depth),
            case(env, goal, depth),
        ]
        .boxed(),
    };
    prop_oneof![1 => leaf, 2 => compound].boxed()
}

// ============================================================================
// Leaves
// ============================================================================

fn leaf(env: &Env, goal: Goal) -> BoxedStrategy<Fragment> {
    let literal = literal(goal);
    let candidates = env.candidates(goal);
    if candidates.is_empty() {
        return literal;
    }
    let var = prop::sample::select(candidates)
        .prop_map(|binding| Fragment {
            free: BTreeSet::from([binding.name.clone()]),
            src: binding.name,
            value: binding.value,
        })
        .boxed();
    prop_oneof![literal, var].boxed()
}

fn literal(goal: Goal) -> BoxedStrategy<Fragment> {
    match goal {
        Goal::Nat { max } => prop_oneof![0..=max.min(100), 0..=max]
            .prop_map(|n| Fragment::leaf(n.to_string(), Value::Nat(n)))
            .boxed(),
        Goal::Bool => any::<bool>()
            .prop_map(|b| {
                let value = Value::Bool(b);
                Fragment::leaf(value.to_string(), value)
            })
            .boxed(),
        // Nonempty and free of whitespace and escapes, so printed lines
        // compare exactly.
        Goal::String => "[A-Za-z0-9_]{1,6}"
            .prop_map(|s| Fragment::leaf(format!("\"{s}\""), Value::String(s)))
            .boxed(),
    }
}

// ============================================================================
// Nat arithmetic
// ============================================================================

/// `{ l + r }` with `r` at most what `max` leaves after `l`.
fn add(env: Env, max: u32, depth: u32) -> BoxedStrategy<Fragment> {
    arithmetic(env, max, depth, "+", |max, l| max - l, u32::checked_add)
}

/// `{ l * r }` with `r` at most what `max` allows after `l`.
fn mul(env: Env, max: u32, depth: u32) -> BoxedStrategy<Fragment> {
    arithmetic(
        env,
        max,
        depth,
        "*",
        |max, l| max.checked_div(l).unwrap_or(max),
        u32::checked_mul,
    )
}

/// `{ l op r }`, with `r` at most `rest(max, l)` so that `apply` stays
/// within `max`.
fn arithmetic(
    env: Env,
    max: u32,
    depth: u32,
    op: &'static str,
    rest: fn(u32, u32) -> u32,
    apply: fn(u32, u32) -> Option<u32>,
) -> BoxedStrategy<Fragment> {
    fragment(env.clone(), Goal::Nat { max }, depth)
        .prop_flat_map(move |lhs| {
            let rhs = fragment(
                env.clone(),
                Goal::Nat {
                    max: rest(max, nat(&lhs)),
                },
                depth,
            );
            // TODO: drop this filter once type checking selects `+` and `*`
            // in a `case` arm whose operands are both `case` expressions;
            // it reports them as ambiguous today.
            let lhs_is_case = is_case(&lhs);
            rhs.prop_filter("both operands are case expressions", move |rhs| {
                !(lhs_is_case && is_case(rhs))
            })
            .prop_map(move |rhs| {
                let value =
                    apply(nat(&lhs), nat(&rhs)).expect("the right operand stays within the bound");
                let src = format!("{{ {} {op} {} }}", lhs.src, rhs.src);
                Fragment::compound(src, Value::Nat(value), [&lhs, &rhs])
            })
        })
        .boxed()
}

fn is_case(fragment: &Fragment) -> bool {
    fragment.src.starts_with("{ case ")
}

fn nat(fragment: &Fragment) -> u32 {
    match fragment.value {
        Value::Nat(n) => n,
        _ => panic!("a Nat goal yields a Nat, got {:?}", fragment.value),
    }
}

// ============================================================================
// Comparisons
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Comparison {
    fn symbol(self) -> &'static str {
        match self {
            Comparison::Eq => "==",
            Comparison::Ne => "!=",
            Comparison::Lt => "<",
            Comparison::Le => "<=",
            Comparison::Gt => ">",
            Comparison::Ge => ">=",
        }
    }

    /// Whether the comparison holds of two values of one type.
    fn holds(self, lhs: &Value, rhs: &Value) -> bool {
        let ordering = match (lhs, rhs) {
            (Value::Nat(a), Value::Nat(b)) => a.cmp(b),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => panic!("compared operands share a type: {lhs:?}, {rhs:?}"),
        };
        match self {
            Comparison::Eq => ordering == Ordering::Equal,
            Comparison::Ne => ordering != Ordering::Equal,
            Comparison::Lt => ordering == Ordering::Less,
            Comparison::Le => ordering != Ordering::Greater,
            Comparison::Gt => ordering == Ordering::Greater,
            Comparison::Ge => ordering != Ordering::Less,
        }
    }
}

fn compare(
    env: Env,
    operand: Goal,
    comparisons: &'static [Comparison],
    depth: u32,
) -> BoxedStrategy<Fragment> {
    let side = fragment(env, operand, depth);
    (prop::sample::select(comparisons), side.clone(), side)
        .prop_map(|(comparison, lhs, rhs)| {
            let holds = comparison.holds(&lhs.value, &rhs.value);
            let src = format!("{{ {} {} {} }}", lhs.src, comparison.symbol(), rhs.src);
            Fragment::compound(src, Value::Bool(holds), [&lhs, &rhs])
        })
        .boxed()
}

/// A `Nat` comparison with any of the six operators.
fn compare_nat(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    use Comparison::*;
    compare(env, Goal::NAT, &[Eq, Ne, Lt, Le, Gt, Ge], depth)
}

/// A `String` equality test; `String` declares only `==` and `!=`.
fn compare_string(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    compare(env, Goal::String, &[Comparison::Eq, Comparison::Ne], depth)
}

// ============================================================================
// Strings
// ============================================================================

/// `{ l <> r }`.
fn concat(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    let side = fragment(env, Goal::String, depth);
    (side.clone(), side)
        .prop_map(|(lhs, rhs)| {
            let (Value::String(l), Value::String(r)) = (&lhs.value, &rhs.value) else {
                panic!("a String goal yields a String");
            };
            let value = Value::String(format!("{l}{r}"));
            let src = format!("{{ {} <> {} }}", lhs.src, rhs.src);
            Fragment::compound(src, value, [&lhs, &rhs])
        })
        .boxed()
}

/// `Nat::to_string`, called by path or as a method.
fn nat_to_string(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    (any::<bool>(), fragment(env, Goal::NAT, depth))
        .prop_map(|(method, operand)| {
            let src = if method {
                format!("{{ {{ {} }}.to_string }}", operand.src)
            } else {
                format!("{{ Nat::to_string({}) }}", operand.src)
            };
            let value = Value::String(nat(&operand).to_string());
            Fragment::compound(src, value, [&operand])
        })
        .boxed()
}

// ============================================================================
// Blocks and cases
// ============================================================================

/// `{ let v = rhs` and `body }` on separate lines, with `body` in the env
/// `v` extends. The grammar separates statements by newlines.
fn block(env: Env, goal: Goal, depth: u32) -> BoxedStrategy<Fragment> {
    let outer = env.clone();
    ty().prop_flat_map(move |rhs_ty| fragment(outer.clone(), Goal::of(rhs_ty), depth))
        .prop_flat_map(move |rhs| {
            let (name, inner) = env.bind(rhs.value.clone());
            fragment(inner, goal, depth).prop_map(move |body| {
                let src = format!("{{\nlet {name} = {}\n{}\n}}", rhs.src, body.src);
                let mut free = body.free.clone();
                free.remove(&name);
                free.extend(rhs.free.iter().cloned());
                Fragment {
                    src,
                    value: body.value,
                    free,
                }
            })
        })
        .boxed()
}

/// `{ case s { 3 -> a, 7 -> b, _ -> c } }` on distinct `Nat` literal arms
/// and a final wildcard. The arms sometimes include the scrutinee's value,
/// so both a literal arm and the wildcard are taken.
fn case(env: Env, goal: Goal, depth: u32) -> BoxedStrategy<Fragment> {
    let arms = (prop::collection::btree_set(0u32..=10, 1..=3), any::<bool>());
    (fragment(env.clone(), Goal::NAT, depth), arms)
        .prop_flat_map(move |(scrutinee, (mut literals, matched))| {
            if matched {
                literals.insert(nat(&scrutinee));
            }
            let literals: Vec<u32> = literals.into_iter().collect();
            let bodies =
                prop::collection::vec(fragment(env.clone(), goal, depth), literals.len() + 1);
            bodies.prop_map(move |bodies| {
                let taken = literals
                    .iter()
                    .position(|&literal| literal == nat(&scrutinee))
                    .unwrap_or(literals.len());
                let mut src = format!("{{ case {} {{ ", scrutinee.src);
                for (literal, body) in literals.iter().zip(&bodies) {
                    write!(src, "{literal} -> {}, ", body.src).expect("writing to a String");
                }
                let wildcard = bodies.last().expect("a case has a wildcard arm");
                write!(src, "_ -> {} }} }}", wildcard.src).expect("writing to a String");
                let value = bodies[taken].value.clone();
                Fragment::compound(src, value, std::iter::once(&scrutinee).chain(&bodies))
            })
        })
        .boxed()
}

// ============================================================================
// Programs
// ============================================================================

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
