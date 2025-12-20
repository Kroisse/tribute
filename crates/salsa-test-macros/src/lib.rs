use proc_macro::TokenStream as ProcTokenStream;
use quote::{format_ident, quote};
use unsynn::{EndOfStream, Group, Parser, ToTokens, TokenStream, unsynn};

unsynn! {
    use unsynn::combinator::Cons;
    use unsynn::Colon;
    use unsynn::{
        BraceGroup, GroupContaining, Ident, LazyVecUntil, NonEmptyOption, TokenTree,
    };

    keyword Mut = "mut";

    enum ParamName {
        Mut(Cons<Mut, Ident>),
        Plain(Ident),
    }

    struct Param {
        name: ParamName,
        colon: Colon,
        ty: TokenStream,
    }

    type ParamsGroup = GroupContaining<NonEmptyOption<Param>>;

    struct SalsaTestFn {
        head: LazyVecUntil<TokenTree, ParamsGroup>,
        params: ParamsGroup,
        tail: LazyVecUntil<TokenTree, BraceGroup>,
        body: BraceGroup,
    }
}

#[proc_macro_attribute]
pub fn salsa_test(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    if let Err(err) = parse_empty_attr(attr) {
        return compile_error(&err.to_string());
    }

    let input = TokenStream::from(item);
    match transform_fn(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => compile_error(&err.to_string()),
    }
}

fn parse_empty_attr(attr: ProcTokenStream) -> Result<(), Box<unsynn::Error>> {
    let mut tokens = TokenStream::from(attr).to_token_iter();
    EndOfStream::parser(&mut tokens)?;
    Ok(())
}

fn transform_fn(input: TokenStream) -> Result<TokenStream, Box<unsynn::Error>> {
    let mut tokens = input.to_token_iter();
    let func: SalsaTestFn = SalsaTestFn::parser(&mut tokens).map_err(Box::new)?;
    let db_ident = db_ident_from_params(&func.params)?;
    let body_group: Group = func.body.into();

    let wrapped_body = quote!({
        salsa::Database::attach(&salsa::DatabaseImpl::default(), |#db_ident| {
            #body_group
        });
    });

    let mut output = TokenStream::new();
    output.extend(quote!(#[test]));
    func.head.to_tokens(&mut output);
    output.extend(quote!(()));
    func.tail.to_tokens(&mut output);
    output.extend(wrapped_body);
    Ok(output)
}

impl ParamName {
    fn ident(&self) -> &unsynn::Ident {
        match self {
            ParamName::Mut(cons) => &cons.second,
            ParamName::Plain(ident) => ident,
        }
    }
}

fn db_ident_from_params(params: &ParamsGroup) -> Result<proc_macro2::Ident, Box<unsynn::Error>> {
    match &params.content.0 {
        None => Ok(format_ident!("db")),
        Some(param) => {
            let ident = param.name.ident();
            if ident == "_" {
                return other_error("salsa_test expects a named parameter, not `_`");
            }
            Ok(ident.clone())
        }
    }
}

fn other_error<T>(message: &str) -> Result<T, Box<unsynn::Error>> {
    let empty = TokenStream::new().to_token_iter();
    Ok(unsynn::Error::other::<T>(
        None,
        &empty,
        message.to_string(),
    )?)
}

fn compile_error(message: &str) -> ProcTokenStream {
    quote!(compile_error!(#message);).into()
}
