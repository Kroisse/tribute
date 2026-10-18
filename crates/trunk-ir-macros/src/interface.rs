//! `#[op_interface]`: an operation interface declared as a trait that
//! operation wrappers implement.
//!
//! The trait keeps its items and gains a hidden registration constant. The
//! macro also generates a vtable with one function pointer per method, its
//! registry, and the `Dyn<Trait>` handle that queries an operation by name.

use heck::ToSnakeCase;
use proc_macro2::{Delimiter, Ident, Spacing, TokenStream, TokenTree};
use quote::{format_ident, quote};

struct InterfaceTrait {
    attrs: Vec<TokenStream>,
    vis: TokenStream,
    name: Ident,
    supertraits: TokenStream,
    items: Vec<TokenStream>,
    methods: Vec<Method>,
}

struct Method {
    docs: Vec<TokenStream>,
    name: Ident,
    lifetimes: TokenStream,
    receiver: Receiver,
    ctx: Param,
    params: Vec<Param>,
    ret: TokenStream,
}

#[derive(Clone, Copy)]
enum Receiver {
    Value,
    Ref,
}

struct Param {
    name: Ident,
    ty: TokenStream,
}

/// Expand `#[op_interface]` on a trait.
pub fn gen_interface(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    if !attr.is_empty() {
        return Err("`#[op_interface]` takes no arguments".into());
    }
    let def = parse_trait(item)?;
    Ok(generate(&def))
}

fn parse_trait(item: TokenStream) -> Result<InterfaceTrait, String> {
    let mut iter = item.into_iter().peekable();
    let attrs = parse_outer_attrs(&mut iter)?;

    let mut vis = TokenStream::new();
    loop {
        match iter.next() {
            Some(TokenTree::Ident(id)) if id == "trait" => break,
            Some(TokenTree::Ident(id)) if id == "unsafe" || id == "auto" => {
                return Err(format!("`#[op_interface]` does not support `{id}` traits"));
            }
            Some(tt) => vis.extend([tt]),
            None => return Err("`#[op_interface]` applies to a trait".into()),
        }
    }
    let name = match iter.next() {
        Some(TokenTree::Ident(name)) => name,
        _ => return Err("expected the interface trait's name".into()),
    };

    let mut supertraits = TokenStream::new();
    let body = loop {
        match iter.next() {
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => break g.stream(),
            Some(TokenTree::Punct(p)) if p.as_char() == '<' && supertraits.is_empty() => {
                return Err(format!("operation interface `{name}` cannot be generic"));
            }
            Some(TokenTree::Ident(id)) if id == "where" => {
                return Err(format!(
                    "operation interface `{name}` cannot have a `where` clause"
                ));
            }
            Some(tt) => supertraits.extend([tt]),
            None => return Err(format!("expected the body of `{name}`")),
        }
    };
    if iter.next().is_some() {
        return Err(format!("unexpected tokens after the body of `{name}`"));
    }

    let mut items = Vec::new();
    let mut methods = Vec::new();
    let mut body = body.into_iter().peekable();
    while body.peek().is_some() {
        let (item, method) = parse_method(&mut body, &name)?;
        items.push(item);
        methods.push(method);
    }

    Ok(InterfaceTrait {
        attrs,
        vis,
        name,
        supertraits,
        items,
        methods,
    })
}

type TokenIter = std::iter::Peekable<proc_macro2::token_stream::IntoIter>;

/// The vtable's `name` field and the handle's generated functions.
const RESERVED_METHOD_NAMES: &[&str] = &["name", "get", "is_implemented", "implementors", "op"];

fn parse_outer_attrs(iter: &mut TokenIter) -> Result<Vec<TokenStream>, String> {
    let mut attrs = Vec::new();
    while matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '#') {
        let pound = iter.next().expect("peeked");
        match iter.next() {
            Some(group @ TokenTree::Group(_)) => attrs.push(TokenStream::from_iter([pound, group])),
            _ => return Err("expected `[...]` after `#`".into()),
        }
    }
    Ok(attrs)
}

/// Parse one trait item, which must be a method. Returns the item's tokens
/// unchanged with its parsed signature.
fn parse_method(iter: &mut TokenIter, interface: &Ident) -> Result<(TokenStream, Method), String> {
    let attrs = parse_outer_attrs(iter)?;
    let docs = attrs
        .iter()
        .filter(|attr| is_doc_attr(attr))
        .cloned()
        .collect();
    let mut item: TokenStream = attrs.into_iter().collect();

    match iter.next() {
        Some(TokenTree::Ident(id)) if id == "fn" => item.extend([TokenTree::Ident(id)]),
        Some(TokenTree::Ident(id)) if id == "const" || id == "type" => {
            return Err(format!(
                "operation interface `{interface}` can only declare methods, not `{id}` items"
            ));
        }
        Some(other) => {
            return Err(format!(
                "operation interface `{interface}` can only declare methods, found `{other}`"
            ));
        }
        None => return Err("expected a method".into()),
    }
    let name = match iter.next() {
        Some(TokenTree::Ident(name)) => {
            item.extend([TokenTree::Ident(name.clone())]);
            name
        }
        _ => return Err("expected a method name".into()),
    };
    if RESERVED_METHOD_NAMES
        .iter()
        .any(|reserved| name == reserved)
    {
        return Err(format!(
            "interface method name `{name}` is reserved by the generated vtable and handle"
        ));
    }

    let mut lifetimes = TokenStream::new();
    if matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '<') {
        let mut depth = 0usize;
        loop {
            let tt = iter
                .next()
                .ok_or_else(|| format!("unterminated generics on `{name}`"))?;
            item.extend([tt.clone()]);
            if let TokenTree::Punct(p) = &tt {
                match p.as_char() {
                    '<' => {
                        depth += 1;
                        if depth == 1 {
                            continue;
                        }
                    }
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
            lifetimes.extend([tt]);
        }
        for generic in split_top_level(lifetimes.clone(), ',') {
            if !matches!(generic.first(), Some(TokenTree::Punct(p)) if p.as_char() == '\'') {
                return Err(format!(
                    "interface method `{name}` can only be generic over lifetimes"
                ));
            }
        }
    }

    let params = match iter.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
            item.extend([TokenTree::Group(g.clone())]);
            g.stream()
        }
        _ => return Err(format!("expected the parameters of `{name}`")),
    };

    let mut ret = TokenStream::new();
    loop {
        match iter.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ';' => {
                item.extend([TokenTree::Punct(p)]);
                break;
            }
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => {
                item.extend([TokenTree::Group(g)]);
                break;
            }
            Some(TokenTree::Ident(id)) if id == "where" => {
                return Err(format!(
                    "interface method `{name}` cannot have a `where` clause"
                ));
            }
            Some(tt) => {
                item.extend([tt.clone()]);
                ret.extend([tt]);
            }
            None => return Err(format!("expected `;` or a body after `{name}`")),
        }
    }

    let mut params = split_top_level(params, ',').into_iter();
    let receiver = match params.next().as_deref() {
        Some([TokenTree::Ident(id)]) if id == "self" => Receiver::Value,
        Some([TokenTree::Ident(m), TokenTree::Ident(id)]) if m == "mut" && id == "self" => {
            Receiver::Value
        }
        Some([TokenTree::Punct(p), TokenTree::Ident(id)]) if p.as_char() == '&' && id == "self" => {
            Receiver::Ref
        }
        _ => {
            return Err(format!(
                "interface method `{name}` must take `self` or `&self`"
            ));
        }
    };
    let mut params = params
        .enumerate()
        .map(|(index, tokens)| parse_param(&name, index, tokens))
        .collect::<Result<Vec<_>, _>>()?;
    if params.is_empty() || !mentions_ir_context(&params[0].ty) {
        return Err(format!(
            "interface method `{name}` must take the IR context (`&IrContext` or \
             `&mut IrContext`) after its receiver"
        ));
    }
    let ctx = params.remove(0);

    Ok((
        item,
        Method {
            docs,
            name,
            lifetimes,
            receiver,
            ctx,
            params,
            ret,
        },
    ))
}

fn parse_param(method: &Ident, index: usize, tokens: Vec<TokenTree>) -> Result<Param, String> {
    let colon = tokens
        .iter()
        .position(|tt| matches!(tt, TokenTree::Punct(p) if p.as_char() == ':'))
        .ok_or_else(|| format!("expected `name: Type` in the parameters of `{method}`"))?;
    let name = match &tokens[..colon] {
        [.., TokenTree::Ident(id)] => {
            let text = id.to_string();
            let text = text.trim_start_matches('_');
            if text.is_empty() {
                format_ident!("arg{index}")
            } else {
                Ident::new(text, id.span())
            }
        }
        _ => format_ident!("arg{index}"),
    };
    Ok(Param {
        name,
        ty: tokens[colon + 1..].iter().cloned().collect(),
    })
}

fn mentions_ir_context(ty: &TokenStream) -> bool {
    ty.clone()
        .into_iter()
        .any(|tt| matches!(tt, TokenTree::Ident(id) if id == "IrContext"))
}

fn is_doc_attr(attr: &TokenStream) -> bool {
    attr.clone().into_iter().any(|tt| match tt {
        TokenTree::Group(g) => {
            matches!(g.stream().into_iter().next(), Some(TokenTree::Ident(id)) if id == "doc")
        }
        _ => false,
    })
}

fn has_on_unimplemented(attrs: &[TokenStream]) -> bool {
    attrs
        .iter()
        .any(|attr| attr.to_string().contains("on_unimplemented"))
}

/// Split a token sequence at top-level `sep` punctuation, treating `<...>`
/// as nesting (but not the `>` of `->`).
fn split_top_level(stream: TokenStream, sep: char) -> Vec<Vec<TokenTree>> {
    let mut parts = vec![Vec::new()];
    let mut depth = 0usize;
    let mut after_joint_minus = false;
    for tt in stream {
        let mut joint_minus = false;
        if let TokenTree::Punct(p) = &tt {
            match p.as_char() {
                '<' => depth += 1,
                '>' if !after_joint_minus => depth = depth.saturating_sub(1),
                '-' => joint_minus = p.spacing() == Spacing::Joint,
                c if c == sep && depth == 0 => {
                    parts.push(Vec::new());
                    after_joint_minus = false;
                    continue;
                }
                _ => {}
            }
        }
        after_joint_minus = joint_minus;
        parts.last_mut().expect("never empty").push(tt);
    }
    if parts.last().is_some_and(Vec::is_empty) {
        parts.pop();
    }
    parts
}

fn generate(def: &InterfaceTrait) -> TokenStream {
    let InterfaceTrait {
        attrs,
        vis,
        name,
        supertraits,
        items,
        methods,
    } = def;
    let krate = quote!(::trunk_ir);
    let vtable = format_ident!("__{}VTable", name);
    let handle = format_ident!("Dyn{}", name);
    let name_str = name.to_string();

    let diagnostic = (!has_on_unimplemented(attrs)).then(|| {
        let message = format!("`{{Self}}` does not implement the `{name_str}` operation interface");
        let label = format!("declared to implement `{name_str}`");
        let note = format!("add `impl {name_str} for {{Self}}`");
        quote! {
            #[diagnostic::on_unimplemented(message = #message, label = #label, note = #note)]
        }
    });

    let fields = methods.iter().map(|m| {
        let field = &m.name;
        let ctx_ty = &m.ctx.ty;
        let tys = m.params.iter().map(|p| &p.ty);
        let ret = &m.ret;
        let lifetimes = &m.lifetimes;
        let binder = (!lifetimes.is_empty()).then(|| quote!(for<#lifetimes>));
        quote! { #field: #binder fn(#ctx_ty, #krate::OpRef #(, #tys)*) #ret }
    });

    let thunk_name = |m: &Method| format_ident!("__{}", m.name.to_string().to_snake_case());
    let inits = methods.iter().map(|m| {
        let field = &m.name;
        let thunk = thunk_name(m);
        quote! { #field: Self::#thunk::<T> }
    });

    let expect_msg = format!("{name_str} registry entries are keyed by the operation's name");
    let thunks = methods.iter().map(|m| {
        let method = &m.name;
        let thunk = thunk_name(m);
        let ctx = &m.ctx.name;
        let ctx_ty = &m.ctx.ty;
        let args = m.params.iter().map(|p| &p.name).collect::<Vec<_>>();
        let tys = m.params.iter().map(|p| &p.ty);
        let ret = &m.ret;
        let lifetimes = &m.lifetimes;
        let lifetimes = (!lifetimes.is_empty()).then(|| quote!(#lifetimes,));
        let this = match m.receiver {
            Receiver::Value => quote!(this),
            Receiver::Ref => quote!(&this),
        };
        quote! {
            fn #thunk<#lifetimes T: #name>(
                #ctx: #ctx_ty,
                op: #krate::OpRef
                #(, #args: #tys)*
            ) #ret {
                let this = <T as #krate::ops::DialectOp>::from_op(&*#ctx, op).expect(#expect_msg);
                <T as #name>::#method(#this, #ctx #(, #args)*)
            }
        }
    });

    let handle_methods = methods.iter().map(|m| {
        let method = &m.name;
        let docs = &m.docs;
        let ctx = &m.ctx.name;
        let ctx_ty = &m.ctx.ty;
        let args = m.params.iter().map(|p| &p.name).collect::<Vec<_>>();
        let tys = m.params.iter().map(|p| &p.ty);
        let ret = &m.ret;
        let lifetimes = &m.lifetimes;
        let generics = (!lifetimes.is_empty()).then(|| quote!(<#lifetimes>));
        quote! {
            #(#docs)*
            pub fn #method #generics(self, #ctx: #ctx_ty #(, #args: #tys)*) #ret {
                (self.vtable.#method)(#ctx, self.op #(, #args)*)
            }
        }
    });

    let duplicate_msg = format!("duplicate {name_str} registration for {{:?}}");
    let vtable_doc = format!("Registration of one operation's [`{name_str}`] implementation.");
    let handle_doc =
        format!("An operation together with its [`{name_str}`] implementation, found by its name.");
    let get_doc = format!("The [`{name_str}`] implementation of `op`'s operation, if it has one.");

    quote! {
        #(#attrs)*
        #diagnostic
        #vis trait #name #supertraits {
            #(#items)*

            #[doc(hidden)]
            const __TRUNK_IR_INTERFACE: #vtable = #vtable::new::<Self>();
        }

        #[doc = #vtable_doc]
        #[doc(hidden)]
        #vis struct #vtable {
            name: #krate::OpName,
            #(#fields,)*
        }

        impl #vtable {
            pub const fn new<T: #name>() -> Self {
                Self {
                    name: <T as #krate::ops::DialectOp>::NAME,
                    #(#inits,)*
                }
            }

            #(#thunks)*

            fn registry() -> &'static #krate::__private::HashMap<#krate::OpName, &'static Self> {
                static REGISTRY: ::std::sync::LazyLock<
                    #krate::__private::HashMap<#krate::OpName, &'static #vtable>,
                > = ::std::sync::LazyLock::new(|| {
                    let mut registry = #krate::__private::HashMap::default();
                    for entry in #krate::__private::inventory::iter::<#vtable> {
                        assert!(
                            registry.insert(entry.name, entry).is_none(),
                            #duplicate_msg,
                            entry.name,
                        );
                    }
                    registry
                });
                &REGISTRY
            }
        }

        #krate::__private::inventory::collect!(#vtable);

        #[doc = #handle_doc]
        #[derive(Clone, Copy)]
        #vis struct #handle {
            op: #krate::OpRef,
            vtable: &'static #vtable,
        }

        impl #handle {
            #[doc = #get_doc]
            pub fn get(ctx: &#krate::IrContext, op: #krate::OpRef) -> Option<Self> {
                #vtable::registry()
                    .get(&ctx.op(op).name)
                    .map(|vtable| Self { op, vtable })
            }

            /// Whether `op`'s operation implements this interface.
            pub fn is_implemented(ctx: &#krate::IrContext, op: #krate::OpRef) -> bool {
                #vtable::registry().contains_key(&ctx.op(op).name)
            }

            /// The names of every operation that implements this interface,
            /// in no particular order.
            pub fn implementors() -> impl Iterator<Item = #krate::OpName> {
                #vtable::registry().keys().copied()
            }

            /// The operation this handle queries.
            pub fn op(self) -> #krate::OpRef {
                self.op
            }

            #(#handle_methods)*
        }

        impl ::std::fmt::Debug for #handle {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.debug_struct(stringify!(#handle))
                    .field("op", &self.op)
                    .field("name", &self.vtable.name)
                    .finish()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(item: TokenStream) -> String {
        gen_interface(TokenStream::new(), item).unwrap().to_string()
    }

    fn expand_err(item: TokenStream) -> String {
        match gen_interface(TokenStream::new(), item) {
            Err(error) => error,
            Ok(tokens) => panic!("expected an error, got: {tokens}"),
        }
    }

    #[test]
    fn method_gets_vtable_slot_thunk_and_handle_method() {
        let out = expand(quote! {
            pub trait Branch: DialectOp {
                fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, Error>;
            }
        });
        assert!(out.contains("pub struct __BranchVTable"), "got: {out}");
        assert!(
            out.contains("successors : fn (& IrContext , :: trunk_ir :: OpRef) -> Result < BranchSuccessors , Error >"),
            "got: {out}"
        );
        assert!(
            out.contains("< T as Branch > :: successors (this , ctx)"),
            "got: {out}"
        );
        assert!(out.contains("pub struct DynBranch"), "got: {out}");
        assert!(
            out.contains("pub fn successors (self , ctx : & IrContext) -> Result < BranchSuccessors , Error > { (self . vtable . successors) (ctx , self . op) }"),
            "got: {out}"
        );
        assert!(
            out.contains(
                "const __TRUNK_IR_INTERFACE : __BranchVTable = __BranchVTable :: new :: < Self > ()"
            ),
            "got: {out}"
        );
        assert!(out.contains("on_unimplemented"), "got: {out}");
    }

    #[test]
    fn marker_interface_has_no_slots() {
        let out = expand(quote! { pub trait Pure: DialectOp {} });
        assert!(
            out.contains("pub struct __PureVTable { name : :: trunk_ir :: OpName , }"),
            "got: {out}"
        );
    }

    #[test]
    fn lifetimes_become_higher_ranked_slots() {
        let out = expand(quote! {
            pub trait Args: DialectOp {
                fn arguments<'a>(self, ctx: &'a IrContext) -> Option<&'a [ValueRef]>;
            }
        });
        assert!(
            out.contains("arguments : for < 'a > fn (& 'a IrContext , :: trunk_ir :: OpRef)"),
            "got: {out}"
        );
        assert!(
            out.contains("fn __arguments < 'a , T : Args >"),
            "got: {out}"
        );
    }

    #[test]
    fn ref_receiver_mut_context_and_extra_params() {
        let out = expand(quote! {
            pub trait Sig: DialectOp {
                fn allowed(&self, _ctx: &IrContext) -> bool { false }
                fn set(self, ctx: &mut IrContext, signature: HashMap<A, B>) -> bool;
            }
        });
        assert!(
            out.contains("< T as Sig > :: allowed (& this , ctx)"),
            "got: {out}"
        );
        assert!(
            out.contains(
                "set : fn (& mut IrContext , :: trunk_ir :: OpRef , HashMap < A , B >) -> bool"
            ),
            "got: {out}"
        );
        assert!(
            out.contains("< T as Sig > :: set (this , ctx , signature)"),
            "got: {out}"
        );
    }

    #[test]
    fn keeps_existing_on_unimplemented() {
        let out = expand(quote! {
            #[diagnostic::on_unimplemented(message = "custom")]
            pub trait Pure: DialectOp {}
        });
        assert_eq!(out.matches("on_unimplemented").count(), 1, "got: {out}");
    }

    #[test]
    fn rejects_unsupported_forms() {
        assert!(expand_err(quote! { trait X<T>: DialectOp {} }).contains("cannot be generic"));
        assert!(
            expand_err(quote! { trait X: DialectOp { const N: u32; } })
                .contains("only declare methods")
        );
        assert!(
            expand_err(quote! { trait X: DialectOp { fn f<T>(self, ctx: &IrContext); } })
                .contains("only be generic over lifetimes")
        );
        assert!(
            expand_err(quote! { trait X: DialectOp { fn f(ctx: &IrContext); } })
                .contains("`self` or `&self`")
        );
        assert!(
            expand_err(quote! { trait X: DialectOp { fn f(self, n: u32); } })
                .contains("IR context")
        );
        assert!(
            expand_err(quote! { trait X: DialectOp { fn op(self, ctx: &IrContext); } })
                .contains("reserved")
        );
    }
}
