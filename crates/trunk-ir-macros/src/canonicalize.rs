//! Proc-macro attribute for the canonicalize pass:
//! `#[canonicalize_fold]`. It submits an `inventory` entry the pass
//! discovers at startup, so the registration sits next to the function
//! definition rather than in a separate macro_rules! call.

use proc_macro2::{Ident, TokenStream, TokenTree};
use quote::quote;

/// Generate the expanded form for `#[canonicalize_fold(<OpWrapper>)]`.
pub fn gen_fold(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    if attr.is_empty() {
        return Err(
            "expected the operation wrapper type, e.g. `#[canonicalize_fold(Addi)]`".into(),
        );
    }
    let (fn_attrs, fn_ident) = extract_fn_attrs_and_ident(&item)?;
    Ok(quote! {
        #item

        #fn_attrs
        ::trunk_ir::inventory::submit! {
            ::trunk_ir::transforms::canonicalize::CanonicalizeFold::new::<#attr>(#fn_ident)
        }
    })
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
    fn fold_emits_typed_inventory_submit() {
        let attr: TokenStream = quote! { Addi };
        let item: TokenStream = quote! {
            pub(crate) fn fold_addi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { None }
        };
        let out = gen_fold(attr, item).unwrap().to_string();
        assert!(
            out.contains("CanonicalizeFold :: new :: < Addi > (fold_addi)"),
            "got: {out}"
        );
    }

    #[test]
    fn fold_rejects_missing_wrapper() {
        let item: TokenStream = quote! { fn f() {} };
        let err = match gen_fold(TokenStream::new(), item) {
            Err(e) => e,
            Ok(_) => panic!("expected parse error"),
        };
        assert!(err.contains("operation wrapper type"), "got: {err}");
    }

    #[test]
    fn fold_forwards_cfg_to_submit_block() {
        // A `#[cfg(test)] fn fold_x` would only exist under cfg(test).
        // The inventory::submit! must observe the same gate so the
        // registration doesn't reference a non-existent function.
        let attr: TokenStream = quote! { Foo };
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
        let attr: TokenStream = quote! { Foo };
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
        let attr: TokenStream = quote! { Foo };
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
        let attr: TokenStream = quote! { Foo };
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
