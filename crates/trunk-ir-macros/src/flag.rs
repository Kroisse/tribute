//! `#[op_flag]`: an operation flag declared as a unit struct.
//!
//! The struct is kept unchanged. The macro adds the `static` `OpFlagDef`
//! that identifies the flag and the `OpFlag` impl pointing to it.

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::quote;

/// Expand `#[op_flag]` on a unit struct.
pub fn gen_flag(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    let verify = {
        let mut args = attr.into_iter();
        match (args.next(), args.next()) {
            (None, _) => false,
            (Some(TokenTree::Ident(arg)), None) if arg == "verify" => true,
            _ => return Err("`#[op_flag]` takes no arguments other than `verify`".into()),
        }
    };
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
    let verifier = if verify {
        quote!(::core::option::Option::Some(
            <#name as #krate::op_flag::VerifyFlag>::verify
        ))
    } else {
        quote!(::core::option::Option::None)
    };
    Ok(quote! {
        #item

        const _: () = {
            static DEF: #krate::op_flag::OpFlagDef = #krate::op_flag::OpFlagDef {
                name: #name_str,
                verifier: #verifier,
            };

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
                "static DEF : :: trunk_ir :: op_flag :: OpFlagDef = :: trunk_ir :: op_flag :: OpFlagDef { name : \"Pure\" , verifier : :: core :: option :: Option :: None , }"
            ),
            "got: {out}"
        );
        assert!(
            out.contains("impl :: trunk_ir :: op_flag :: OpFlag for Pure"),
            "got: {out}"
        );
    }

    #[test]
    fn verify_points_to_the_verify_flag_impl() {
        let out = gen_flag(quote!(verify), quote! { pub struct Isolated; })
            .unwrap()
            .to_string();
        assert!(
            out.contains(
                "verifier : :: core :: option :: Option :: Some (< Isolated as :: trunk_ir :: op_flag :: VerifyFlag > :: verify)"
            ),
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
        for args in [quote!(x), quote!(verify, verify)] {
            assert!(
                gen_flag(args, quote! { struct Pure; })
                    .unwrap_err()
                    .contains("takes no arguments other than `verify`")
            );
        }
    }
}
