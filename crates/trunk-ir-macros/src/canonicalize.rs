//! Proc-macro attributes for the canonicalize pass: `#[canonicalize_fold]`
//! and `#[canonicalize_pattern]`. They submit `inventory` entries the
//! pass discovers at startup, so the registration sits next to the
//! function definition rather than in a separate macro_rules! call.

use proc_macro2::{Ident, TokenStream, TokenTree};
use quote::quote;

/// Generate the expanded form for `#[canonicalize_fold(<dialect>.<op>)]`.
pub fn gen_fold(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    let (dialect, op_name) = parse_dialect_op(attr)?;
    let fn_ident = extract_fn_ident(&item)?;
    Ok(quote! {
        #item

        ::inventory::submit! {
            ::trunk_ir::transforms::canonicalize::CanonicalizeFold {
                dialect: #dialect,
                op_name: #op_name,
                fold: #fn_ident,
            }
        }
    })
}

/// Generate the expanded form for `#[canonicalize_pattern]`.
pub fn gen_pattern(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    if !attr.is_empty() {
        return Err("`#[canonicalize_pattern]` does not accept arguments".to_string());
    }
    let fn_ident = extract_fn_ident(&item)?;
    Ok(quote! {
        #item

        ::inventory::submit! {
            ::trunk_ir::transforms::canonicalize::CanonicalizePattern {
                make: #fn_ident,
            }
        }
    })
}

/// Parse `<dialect_ident> . <op_ident>` from the attribute payload.
/// Strips `r#` from raw identifiers so op names like `r#const` register
/// as `"const"`.
fn parse_dialect_op(attr: TokenStream) -> Result<(String, String), String> {
    let mut iter = attr.into_iter();
    let dialect = next_ident_str(&mut iter, "dialect name")?;
    expect_dot(&mut iter)?;
    let op_name = next_ident_str(&mut iter, "op name")?;
    if let Some(extra) = iter.next() {
        return Err(format!(
            "unexpected token after `{dialect}.{op_name}`: `{extra}`"
        ));
    }
    Ok((dialect, op_name))
}

fn next_ident_str(
    iter: &mut impl Iterator<Item = TokenTree>,
    what: &str,
) -> Result<String, String> {
    match iter.next() {
        Some(TokenTree::Ident(id)) => Ok(strip_raw_prefix(&id.to_string())),
        Some(other) => Err(format!("expected {what}, got `{other}`")),
        None => Err(format!("expected {what}")),
    }
}

fn expect_dot(iter: &mut impl Iterator<Item = TokenTree>) -> Result<(), String> {
    match iter.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == '.' => Ok(()),
        Some(other) => Err(format!("expected `.`, got `{other}`")),
        None => Err("expected `.` between dialect and op name".to_string()),
    }
}

fn strip_raw_prefix(s: &str) -> String {
    s.strip_prefix("r#").unwrap_or(s).to_string()
}

/// Pull the function name out of the attributed item by skipping
/// visibility tokens / modifiers and grabbing the ident right after
/// the `fn` keyword.
fn extract_fn_ident(item: &TokenStream) -> Result<Ident, String> {
    let mut iter = item.clone().into_iter();
    let mut saw_fn = false;
    for tt in iter.by_ref() {
        if let TokenTree::Ident(ident) = tt {
            if saw_fn {
                return Ok(ident);
            }
            if ident == "fn" {
                saw_fn = true;
            }
        }
    }
    Err("expected `fn <name>` in attributed item".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_emits_inventory_submit_for_plain_idents() {
        let attr: TokenStream = quote! { arith.addi };
        let item: TokenStream = quote! {
            pub(crate) fn fold_addi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { None }
        };
        let out = gen_fold(attr, item).unwrap().to_string();
        assert!(out.contains("CanonicalizeFold"), "got: {out}");
        assert!(out.contains("\"arith\""), "got: {out}");
        assert!(out.contains("\"addi\""), "got: {out}");
        assert!(out.contains("fold : fold_addi"), "got: {out}");
    }

    #[test]
    fn fold_strips_raw_prefix_from_op_name() {
        // r#const should register as "const".
        let attr: TokenStream = quote! { core.r#const };
        let item: TokenStream = quote! {
            fn fold_const(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { None }
        };
        let out = gen_fold(attr, item).unwrap().to_string();
        assert!(out.contains("\"const\""), "got: {out}");
        assert!(!out.contains("\"r#const\""), "got: {out}");
    }

    #[test]
    fn fold_rejects_missing_dot() {
        let attr: TokenStream = quote! { arith addi };
        let item: TokenStream = quote! { fn f() {} };
        let err = match gen_fold(attr, item) {
            Err(e) => e,
            Ok(_) => panic!("expected parse error"),
        };
        assert!(err.contains("expected `.`"), "got: {err}");
    }

    #[test]
    fn pattern_emits_inventory_submit() {
        let item: TokenStream = quote! {
            fn make_if_const_fold() -> Box<dyn RewritePattern> { Box::new(IfConstFold) }
        };
        let out = gen_pattern(TokenStream::new(), item).unwrap().to_string();
        assert!(out.contains("CanonicalizePattern"), "got: {out}");
        assert!(out.contains("make : make_if_const_fold"), "got: {out}");
    }

    #[test]
    fn pattern_rejects_attr_args() {
        let attr: TokenStream = quote! { foo };
        let item: TokenStream = quote! { fn f() {} };
        let err = match gen_pattern(attr, item) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(err.contains("does not accept arguments"), "got: {err}");
    }
}
