use crate::ast::{lexer, Expr, Span, Spanned, Token};
use chumsky::prelude::*;

type ParserInput<'tokens, 'src> =
    chumsky::input::SpannedInput<Token<'src>, Span, &'tokens [Spanned<Token<'src>>]>;

fn parser<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    ParserInput<'tokens, 'src>,
    Spanned<Expr>,
    extra::Err<Rich<'tokens, Token<'src>, Span>>,
> {
    recursive(|expr| {
        let value = select! {
            Token::Number(n) => Expr::Number(n),
            Token::String(s) => Expr::String(s.to_owned()),
            Token::Ident(s) => Expr::Identifier(s.to_owned()),
        }
        .map_with(|expr, e| (expr, e.span()));

        let block = expr
            .repeated()
            .collect()
            .delimited_by(just(Token::ParenOpen), just(Token::ParenClose))
            .map_with(|exprs, e| (Expr::List(exprs), e.span()));

        block.or(value)
    })
}

pub fn parse<'a>(input: &'a str) -> Vec<Spanned<Expr>> {
    let (tokens, _) = lexer().parse(input).into_output_errors();
    let ast = if let Some(tokens) = tokens {
        let result = parser()
            .repeated()
            .collect()
            .parse(tokens.as_slice().spanned((input.len()..input.len()).into()));
        result.into_output_errors().0
    } else {
        None
    };
    ast.unwrap_or_default()
}
