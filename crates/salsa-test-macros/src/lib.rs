use proc_macro::TokenStream;
use quote::quote;
use unsynn::{EndOfStream, Group, Parser, ToTokens, TokenStream as UnsTokenStream};

#[proc_macro_attribute]
pub fn salsa_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Err(err) = parse_empty_attr(attr) {
        return compile_error(&err.to_string());
    }

    let input = UnsTokenStream::from(item);
    match transform_fn(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => compile_error(&err.to_string()),
    }
}

fn parse_empty_attr(attr: TokenStream) -> Result<(), Box<unsynn::Error>> {
    let mut tokens = UnsTokenStream::from(attr).to_token_iter();
    EndOfStream::parser(&mut tokens)
        .map(|_| ())
        .map_err(Box::new)
}

fn transform_fn(input: UnsTokenStream) -> Result<UnsTokenStream, Box<unsynn::Error>> {
    let mut tokens = input.to_token_iter();
    let func: unsynn::TokenStream = unsynn::TokenStream::parser(&mut tokens).map_err(Box::new)?;
    let mut func_tokens: Vec<unsynn::TokenTree> = func.into_iter().collect();

    let body_index = func_tokens.iter().rposition(|token| {
        matches!(token, unsynn::TokenTree::Group(group) if group.delimiter() == unsynn::Delimiter::Brace)
    });
    let Some(body_index) = body_index else {
        let empty = UnsTokenStream::new().to_token_iter();
        return unsynn::Error::other::<UnsTokenStream>(
            None,
            &empty,
            "salsa_test expects a function with a brace body".to_string(),
        )
        .map_err(Box::new);
    };

    let params_index = func_tokens[..body_index].iter().rposition(|token| {
        matches!(token, unsynn::TokenTree::Group(group) if group.delimiter() == unsynn::Delimiter::Parenthesis)
    });
    let Some(params_index) = params_index else {
        let empty = UnsTokenStream::new().to_token_iter();
        return unsynn::Error::other::<UnsTokenStream>(
            None,
            &empty,
            "salsa_test expects a function with a parameter list".to_string(),
        )
        .map_err(Box::new);
    };

    func_tokens[params_index] = unsynn::TokenTree::Group(Group::new(
        unsynn::Delimiter::Parenthesis,
        UnsTokenStream::new(),
    ));

    let body_group = match func_tokens.remove(body_index) {
        unsynn::TokenTree::Group(group) => group,
        _ => unreachable!("body_index points to a group"),
    };
    let wrapped_body = quote!({
        salsa::Database::attach(&salsa::DatabaseImpl::default(), |db| {
            #body_group
        });
    });

    func_tokens.insert(
        body_index,
        unsynn::TokenTree::Group(Group::new(unsynn::Delimiter::Brace, wrapped_body)),
    );

    let mut output = UnsTokenStream::new();
    output.extend(quote!(#[test]));
    output.extend(func_tokens);
    Ok(output)
}

fn compile_error(message: &str) -> TokenStream {
    quote!(compile_error!(#message);).into()
}
