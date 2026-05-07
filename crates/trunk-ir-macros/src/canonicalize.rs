//! Proc-macro attribute for the canonicalize pass:
//! `#[canonicalize_fold]`. It submits an `inventory` entry the pass
//! discovers at startup, so the registration sits next to the function
//! definition rather than in a separate macro_rules! call.

use proc_macro2::{Ident, TokenStream, TokenTree};
use quote::quote;

/// Generate the expanded form for `#[canonicalize_fold(<dialect>.<op>)]`.
pub fn gen_fold(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    let (dialect, op_name) = parse_dialect_op(attr)?;
    let (fn_attrs, fn_ident) = extract_fn_attrs_and_ident(&item)?;
    Ok(quote! {
        #item

        #fn_attrs
        ::trunk_ir::inventory::submit! {
            ::trunk_ir::transforms::canonicalize::CanonicalizeFold {
                dialect: #dialect,
                op_name: #op_name,
                fold: #fn_ident,
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

/// Pull the cfg-gating outer attributes and function name out of the
/// attributed item. Only `#[cfg(...)]` and `#[cfg_attr(...)]` are
/// forwarded onto the generated `inventory::submit!` block — they
/// gate compilation, so the registration must observe the same gates
/// as the function it references (otherwise a `#[cfg(test)] fn
/// fold_...` would emit an inventory entry pointing at a non-existent
/// function outside `cfg(test)`). Other outer attributes (doc
/// comments, `#[allow(...)]`, etc.) stay on the function only —
/// duplicating them onto the submit macro produces spurious
/// warnings.
fn extract_fn_attrs_and_ident(item: &TokenStream) -> Result<(TokenStream, Ident), String> {
    let mut iter = item.clone().into_iter().peekable();
    let mut cfg_attrs = TokenStream::new();
    loop {
        match iter.peek() {
            Some(TokenTree::Punct(p)) if p.as_char() == '#' => {
                let pound = iter.next().expect("peeked");
                let group = match iter.next() {
                    Some(tt @ TokenTree::Group(_)) => tt,
                    Some(other) => {
                        return Err(format!("expected `[...]` after `#`, got `{other}`"));
                    }
                    None => return Err("expected `[...]` after `#`".to_string()),
                };
                if is_cfg_gate(&group) {
                    cfg_attrs.extend([pound, group]);
                }
                // Otherwise (doc comments, lint allows, etc.) just
                // drop them — they stay on the original `#item` and
                // shouldn't be replicated on the submit block.
            }
            Some(TokenTree::Ident(id)) if id == "fn" => {
                iter.next();
                return match iter.next() {
                    Some(TokenTree::Ident(name)) => Ok((cfg_attrs, name)),
                    Some(other) => Err(format!("expected fn name, got `{other}`")),
                    None => Err("expected fn name".to_string()),
                };
            }
            Some(_) => {
                // Visibility (`pub`, `pub(crate)`), modifiers (`unsafe`,
                // `async`, `extern "C"`), or other tokens before `fn`.
                // Skip without capturing.
                iter.next();
            }
            None => return Err("expected `fn <name>` in attributed item".to_string()),
        }
    }
}

/// `true` iff the given `[...]` group is `[cfg(...)]` or
/// `[cfg_attr(...)]` — the only outer attributes that gate compilation
/// of the function they're attached to.
fn is_cfg_gate(group: &TokenTree) -> bool {
    let TokenTree::Group(g) = group else {
        return false;
    };
    let mut inner = g.stream().into_iter();
    matches!(
        inner.next(),
        Some(TokenTree::Ident(id)) if id == "cfg" || id == "cfg_attr",
    )
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
    fn fold_forwards_cfg_to_submit_block() {
        // A `#[cfg(test)] fn fold_x` would only exist under cfg(test).
        // The inventory::submit! must observe the same gate so the
        // registration doesn't reference a non-existent function.
        let attr: TokenStream = quote! { arith.foo };
        let item: TokenStream = quote! {
            #[cfg(test)]
            fn fold_foo(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { None }
        };
        let out = gen_fold(attr, item).unwrap().to_string();
        // Both the original fn and the submit block should carry the
        // cfg, so the literal `# [cfg (test)]` appears twice in the
        // expansion.
        let cfg_count = out.matches("cfg (test)").count();
        assert_eq!(cfg_count, 2, "expected 2 cfg(test) occurrences, got: {out}");
    }

    #[test]
    fn fold_forwards_cfg_attr_to_submit_block() {
        let attr: TokenStream = quote! { arith.foo };
        let item: TokenStream = quote! {
            #[cfg_attr(feature = "x", allow(dead_code))]
            fn fold_foo(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { None }
        };
        let out = gen_fold(attr, item).unwrap().to_string();
        let cfg_count = out.matches("cfg_attr").count();
        assert_eq!(cfg_count, 2, "expected 2 cfg_attr occurrences, got: {out}");
    }

    #[test]
    fn fold_does_not_forward_doc_or_lint_attrs() {
        // Doc comments and `#[allow(...)]` etc. don't gate
        // compilation. Forwarding them onto the submit macro
        // produces spurious `unused_doc_comments` warnings.
        let attr: TokenStream = quote! { arith.foo };
        let item: TokenStream = quote! {
            /// fold doc comment
            #[allow(dead_code)]
            fn fold_foo(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { None }
        };
        let out = gen_fold(attr, item).unwrap().to_string();
        // The original function (`#item`) keeps the doc comment and
        // allow, so they appear once. They must not be duplicated
        // onto the submit block.
        assert_eq!(
            out.matches("fold doc comment").count(),
            1,
            "doc comment must not be duplicated onto submit block: {out}"
        );
        assert_eq!(
            out.matches("allow (dead_code)").count(),
            1,
            "allow(dead_code) must not be duplicated onto submit block: {out}"
        );
    }

    #[test]
    fn fold_emits_trunk_ir_inventory_path() {
        // External consumers shouldn't need a direct `inventory`
        // dependency — the submit path goes through trunk-ir's
        // re-export.
        let attr: TokenStream = quote! { arith.foo };
        let item: TokenStream = quote! {
            fn fold_foo(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { None }
        };
        let out = gen_fold(attr, item).unwrap().to_string();
        assert!(
            out.contains(":: trunk_ir :: inventory :: submit"),
            "expected ::trunk_ir::inventory::submit, got: {out}"
        );
    }
}
