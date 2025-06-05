use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SimpleSpan {
    pub start: usize,
    pub end: usize,
    pub context: (),
}

impl SimpleSpan {
    pub const fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            context: (),
        }
    }
}

pub type Span = SimpleSpan;
pub type Spanned<T> = (T, Span);

pub type Identifier = String;

#[salsa::tracked]
pub struct Program<'db> {
    #[tracked]
    #[return_ref]
    pub items: Vec<Item<'db>>,
}

#[salsa::tracked]
pub struct Item<'db> {
    pub expr: Spanned<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    Number(i64),
    String(String),
    Identifier(Identifier),
    List(Vec<Spanned<Expr>>),
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::String(s) => write!(f, "\"{}\"", s),
            Expr::Identifier(s) => f.write_str(s),
            Expr::List(exprs) => {
                f.write_str("(")?;
                for (i, (expr, _span)) in exprs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    // This would require database access, so we'll implement it differently
                    write!(f, "{}", expr)?;
                }
                f.write_str(")")
            }
        }
    }
}
