use chumsky::prelude::*;
use serde::{Serialize, Deserialize};

pub type Span = SimpleSpan<usize>;
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

pub(crate) fn lexer<'a>(
) -> impl Parser<'a, &'a str, Vec<(Token<'a>, Span)>, extra::Err<Rich<'a, char, Span>>> {
    let num = text::int(10)
        .to_slice()
        .from_str()
        .unwrapped()
        .map(Token::Number);
    let string = just('"')
        .ignore_then(none_of('"').repeated())
        .then_ignore(just('"'))
        .to_slice()
        .map(Token::String);
    let ident = text::ascii::ident().map(Token::Ident);
    let paren = one_of("()").map(|c| match c {
        '(' => Token::ParenOpen,
        ')' => Token::ParenClose,
        _ => unreachable!(),
    });

    let token = num.or(string).or(paren).or(ident);

    let comment = just("//")
        .then(any().and_is(just('\n').not()).repeated())
        .padded();

    token
        .map_with(|tok, e| (tok, e.span()))
        .padded_by(comment.repeated())
        .padded()
        .recover_with(skip_then_retry_until(any().ignored(), end()))
        .repeated()
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lexer() {
        let input = r#"123 (abc) "hello my friend // eh" // comment\n"#;
        let tokens = super::lexer().parse(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                (super::Token::Number(123), Span::from(0..3)),
                (super::Token::ParenOpen, Span::from(4..5)),
                (super::Token::Ident("abc"), Span::from(5..8)),
                (super::Token::ParenClose, Span::from(8..9)),
            ]
        );
    }
}
