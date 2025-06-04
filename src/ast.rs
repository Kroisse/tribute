use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SimpleSpan {
    pub start: usize,
    pub end: usize,
    pub context: (),
}

impl SimpleSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end, context: () }
    }
}

pub type Span = SimpleSpan;
pub type Spanned<T> = (T, Span);

pub type Identifier = String;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<'src> {
    Number(i64),
    String(&'src str),
    Ident(&'src str),
    ParenOpen,
    ParenClose,
}

impl std::fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Token::Number(n) => write!(f, "{}", n),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Ident(s) => f.write_str(s),
            Token::ParenOpen => f.write_str("("),
            Token::ParenClose => f.write_str(")"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    write!(f, "{}", expr.0)?;
                }
                f.write_str(")")
            }
        }
    }
}

