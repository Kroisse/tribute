//! `#[op_flag]`: an operation flag declared as a unit struct.
//!
//! The struct is kept unchanged. The macro adds the `static` `OpFlagDef`
//! that identifies the flag and the `OpFlag` impl pointing to it.

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::quote;

/// Expand `#[op_flag]` on a unit struct.
pub fn gen_flag(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    if !attr.is_empty() {
        return Err("`#[op_flag]` takes no arguments".into());
    }
    let mut tokens = item.clone().into_iter().peekable();
    while matches!(tokens.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '#') {
        tokens.next();
        match tokens.next() {
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => {}
            _ => return Err("expected `[...]` after `#`".into()),
        }
    }
    loop {
        match tokens.next() {
            Some(TokenTree::Ident(id)) if id == "struct" => break,
            Some(TokenTree::Ident(id)) if id == "enum" || id == "union" || id == "trait" => {
                return Err(format!(
                    "`#[op_flag]` applies to a unit struct, not to a `{id}`"
                ));
            }
            Some(_) => {}
            None => return Err("`#[op_flag]` applies to a unit struct".into()),
        }
    }
    let name = match tokens.next() {
        Some(TokenTree::Ident(name)) => name,
        _ => return Err("expected the flag's name".into()),
    };
    match (tokens.next(), tokens.next()) {
        (Some(TokenTree::Punct(p)), None) if p.as_char() == ';' => {}
        _ => {
            return Err(format!(
                "operation flag `{name}` must be a unit struct without generics, \
                 as in `struct {name};`"
            ));
        }
    }

    let krate = quote!(::trunk_ir);
    let name_str = name.to_string();
    Ok(quote! {
        #item

        const _: () = {
            static DEF: #krate::op_flag::OpFlagDef = #krate::op_flag::OpFlagDef { name: #name_str };

            impl #krate::op_flag::OpFlag for #name {
                const DEF: &'static #krate::op_flag::OpFlagDef = &DEF;
            }
        };
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_struct_gets_static_and_impl() {
        let out = gen_flag(
            TokenStream::new(),
            quote! {
                /// Doc.
                pub struct Pure;
            },
        )
        .unwrap()
        .to_string();
        assert!(out.contains("pub struct Pure ;"), "got: {out}");
        assert!(
            out.contains(
                "static DEF : :: trunk_ir :: op_flag :: OpFlagDef = :: trunk_ir :: op_flag :: OpFlagDef { name : \"Pure\" }"
            ),
            "got: {out}"
        );
        assert!(
            out.contains("impl :: trunk_ir :: op_flag :: OpFlag for Pure"),
            "got: {out}"
        );
    }

    #[test]
    fn rejects_other_items() {
        for (item, expected) in [
            (
                quote! { pub struct Pure { x: u32 } },
                "must be a unit struct",
            ),
            (quote! { pub struct Pure(u32); }, "must be a unit struct"),
            (quote! { pub struct Pure<T>; }, "must be a unit struct"),
            (quote! { pub enum Pure {} }, "not to a `enum`"),
        ] {
            let err = gen_flag(TokenStream::new(), item).unwrap_err();
            assert!(err.contains(expected), "unexpected error: {err}");
        }
        assert!(
            gen_flag(quote!(x), quote! { struct Pure; })
                .unwrap_err()
                .contains("takes no arguments")
        );
    }
}
