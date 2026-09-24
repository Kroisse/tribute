//! Parsing for typed operation signatures. The legacy DSL remains in `parse.rs`.

use super::*;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericArgument, Pat, Path, PathArguments, Type, TypeParamBound};

pub struct TypeVar {
    pub name: String,
    pub bounds: Vec<Path>,
}

#[derive(Clone)]
pub struct Projection {
    pub var: usize,
    pub bound: Option<Path>,
    pub name: String,
}

#[derive(Clone)]
pub enum TypeExpr {
    Any,
    Var(usize),
    Anon(Vec<Path>),
    Proj(Projection),
}

#[derive(Clone)]
pub enum ListExpr {
    Types(Vec<TypeExpr>),
    Proj(Projection),
}

#[derive(Clone)]
pub enum ValueExpr {
    Each(TypeExpr),
    List(ListExpr),
}

/// Whether the signature after the operation name uses the typed syntax:
/// generic parameters, or a typed entity wrapper (`Value<..>`, `Variadic<..>`,
/// `Values<..>`, `Attr<..>`) in the parameters or the return type. Legacy
/// results may be `-> Option<result>`, so `Option` alone is not a marker.
pub(super) fn is_typed_operation(iter: &TokenIter) -> Result<bool, String> {
    if matches!(iter.clone().next(), Some(TokenTree::Punct(p)) if p.as_char() == '<') {
        return Ok(true);
    }
    let mut signature = TokenStream::new();
    for tt in iter.clone() {
        if matches!(&tt, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace) {
            return Ok(has_typed_wrapper(signature));
        }
        signature.extend([tt]);
    }
    Err("expected operation body".into())
}

fn has_typed_wrapper(stream: TokenStream) -> bool {
    let mut prev_wrapper = false;
    for tt in stream {
        match tt {
            TokenTree::Punct(p) if p.as_char() == '<' && prev_wrapper => return true,
            TokenTree::Group(g) if has_typed_wrapper(g.stream()) => return true,
            TokenTree::Ident(ident) => {
                prev_wrapper = ["Value", "Variadic", "Values", "Attr"]
                    .iter()
                    .any(|w| ident == w);
                continue;
            }
            _ => {}
        }
        prev_wrapper = false;
    }
    false
}

pub(super) fn parse_typed_operation(
    iter: &mut TokenIter,
    name_ident: Ident,
    legacy_attrs: Vec<AttrDef>,
    rest_results: bool,
) -> Result<OperationDef, String> {
    if !legacy_attrs.is_empty() || rest_results {
        return Err("cannot mix legacy and new operation syntax".into());
    }
    let mut sig_tokens = TokenStream::new();
    let body = loop {
        let tt = iter.next().ok_or("expected operation body")?;
        if matches!(&tt, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace) {
            break tt;
        }
        sig_tokens.extend([tt]);
    };
    let signature = quote!(fn #name_ident #sig_tokens {});
    let func: syn::ItemFn =
        syn::parse2(signature).map_err(|e| format!("invalid typed operation signature: {e}"))?;
    let mut vars = Vec::<TypeVar>::new();
    for param in &func.sig.generics.params {
        match param {
            syn::GenericParam::Type(tp) => {
                let name = ident_str(&tp.ident);
                if vars.iter().any(|v| v.name == name) {
                    return Err(format!("duplicate type variable `{name}`"));
                }
                vars.push(TypeVar {
                    name,
                    bounds: paths(&tp.bounds)?,
                });
            }
            _ => return Err("only type variables are supported in typed operations".into()),
        }
    }
    if let Some(where_clause) = &func.sig.generics.where_clause {
        for pred in &where_clause.predicates {
            let syn::WherePredicate::Type(pred) = pred else {
                return Err("unsupported where predicate".into());
            };
            let Type::Path(ty) = &pred.bounded_ty else {
                return Err("where bound must name a type variable".into());
            };
            let Some(var) = simple_var(&ty.path, &vars) else {
                return Err("where bound must name a declared type variable".into());
            };
            vars[var].bounds.extend(paths(&pred.bounds)?);
        }
    }
    let mut attrs = Vec::new();
    let mut operands = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut typed_entity = false;
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(arg) = arg else {
            return Err("self parameter is not supported".into());
        };
        if !arg.attrs.is_empty() {
            return Err("cannot mix legacy and new operation syntax".into());
        }
        let Pat::Ident(pat) = &*arg.pat else {
            return Err("parameter must have a name".into());
        };
        let ident = pat.ident.clone();
        let name = ident_str(&ident);
        check_name(&name, &mut names)?;
        if matches!(&*arg.ty, Type::Tuple(t) if t.elems.is_empty()) {
            return Err("cannot mix legacy and new operation syntax".into());
        }
        typed_entity = true;
        let (wrapper, inner, optional) = unwrap_wrapper(&arg.ty)?;
        match wrapper.as_str() {
            "Attr" => {
                let (ty, binds) = parse_attr_kind(inner, &vars)?;
                attrs.push(AttrDef {
                    name,
                    raw_ident: ident,
                    ty,
                    optional,
                    binds,
                });
            }
            "Value" | "Variadic" | "Values" if !optional => {
                if operands.last().is_some_and(|o: &Operand| o.variadic) {
                    return Err("variadic/Values operand must be last; only one is allowed".into());
                }
                let (variadic, constraint) = match wrapper.as_str() {
                    "Value" => (false, ValueExpr::Each(parse_one(inner, &vars)?)),
                    "Variadic" => (true, ValueExpr::Each(parse_one(inner, &vars)?)),
                    _ => (true, ValueExpr::List(parse_list(inner, &vars)?)),
                };
                operands.push(Operand {
                    name,
                    raw_ident: ident,
                    variadic,
                    constraint,
                });
            }
            "Value" if optional => return Err("Option<Value<..>> operand is reserved".into()),
            _ => return Err("expected Value, Variadic, Values, Attr, or Option<Attr>".into()),
        }
    }
    let (results, result_constraint) = match &func.sig.output {
        syn::ReturnType::Default => (ResultDef::None, ValueExpr::Each(TypeExpr::Any)),
        syn::ReturnType::Type(_, ty) => {
            typed_entity = true;
            if matches!(&**ty, Type::Tuple(_)) {
                return Err("tuple results are reserved".into());
            }
            let (wrapper, inner, optional) = unwrap_wrapper(ty)?;
            match (wrapper.as_str(), optional) {
                ("Value", false) => (
                    ResultDef::Single("result".into()),
                    ValueExpr::Each(parse_one(inner, &vars)?),
                ),
                ("Value", true) => (
                    ResultDef::Optional("result".into()),
                    ValueExpr::Each(parse_one(inner, &vars)?),
                ),
                ("Variadic", false) => (
                    ResultDef::Variadic("results".into()),
                    ValueExpr::Each(parse_one(inner, &vars)?),
                ),
                ("Values", false) => (
                    ResultDef::Variadic("results".into()),
                    ValueExpr::List(parse_list(inner, &vars)?),
                ),
                _ => return Err("expected Value, Option<Value>, Variadic, or Values result".into()),
            }
        }
    };
    if !typed_entity && !vars.is_empty() {
        return Err("generics require new operation syntax".into());
    }
    let TokenTree::Group(body) = body else {
        unreachable!()
    };
    let regions = parse_regions(body.stream())?;
    for item in &regions {
        let name = match item {
            RegionOrSuccessor::Region { name, .. } | RegionOrSuccessor::Successor(name) => name,
        };
        check_name(name, &mut names)?;
    }
    Ok(OperationDef {
        name: ident_str(&name_ident),
        raw_ident: name_ident,
        attrs,
        operands,
        results,
        regions,
        syntax: Syntax::Typed,
        type_vars: vars,
        result_constraint,
    })
}

fn check_name(name: &str, names: &mut std::collections::HashSet<String>) -> Result<(), String> {
    if [
        "operands",
        "results",
        "regions",
        "successors",
        "build",
        "builder",
    ]
    .contains(&name)
    {
        return Err(format!(
            "parameter name `{name}` collides with a builder method"
        ));
    }
    if !names.insert(name.into()) {
        return Err(format!("duplicate entity name `{name}`"));
    }
    Ok(())
}

fn paths(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, syn::token::Plus>,
) -> Result<Vec<Path>, String> {
    bounds
        .iter()
        .map(|bound| match bound {
            TypeParamBound::Trait(tr) if tr.lifetimes.is_none() => {
                check_path(&tr.path)?;
                Ok(tr.path.clone())
            }
            _ => Err("bounds must be plain Rust paths".into()),
        })
        .collect()
}

fn check_path(path: &Path) -> Result<(), String> {
    if path
        .segments
        .iter()
        .any(|s| !matches!(s.arguments, PathArguments::None))
    {
        return Err("generic arguments on bound paths are not supported".into());
    }
    Ok(())
}

fn simple_var(path: &Path, vars: &[TypeVar]) -> Option<usize> {
    if path.leading_colon.is_none() && path.segments.len() == 1 {
        let name = path.segments.first()?.ident.to_string();
        vars.iter().position(|v| v.name == name)
    } else {
        None
    }
}

fn unwrap_wrapper(ty: &Type) -> Result<(String, &Type, bool), String> {
    let Type::Path(path) = ty else {
        return Err("expected a typed entity wrapper".into());
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err("expected a typed entity wrapper".into());
    }
    let seg = path.path.segments.first().unwrap();
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err("expected wrapper<..>".into());
    };
    if args.args.len() != 1 {
        return Err("entity wrapper requires one type argument".into());
    }
    let GenericArgument::Type(inner) = args.args.first().unwrap() else {
        return Err("entity wrapper requires a type argument".into());
    };
    if seg.ident == "Option" {
        let (wrapper, inner, nested) = unwrap_wrapper(inner)?;
        if nested {
            return Err("nested Option is not supported".into());
        }
        Ok((wrapper, inner, true))
    } else {
        Ok((seg.ident.to_string(), inner, false))
    }
}

fn parse_attr_kind(ty: &Type, vars: &[TypeVar]) -> Result<(AttrType, Option<usize>), String> {
    if matches!(ty, Type::Infer(_)) {
        return Ok((AttrType::Any, None));
    }
    let Type::Path(path) = ty else {
        return Err("invalid attribute kind".into());
    };
    if path.qself.is_some() {
        return Err("invalid attribute projection".into());
    }
    if path.path.segments.len() == 2 {
        let mut it = path.path.segments.iter();
        let first = it.next().unwrap();
        let second = it.next().unwrap();
        if second.ident == "Type"
            && let Some(var) = vars.iter().position(|v| first.ident == v.name)
        {
            return Ok((AttrType::Type, Some(var)));
        }
        return Err("attribute projection must be V::Type for a declared variable".into());
    }
    if path.path.segments.len() != 1 {
        return Err("invalid attribute kind".into());
    }
    let ident = &path.path.segments.first().unwrap().ident;
    parse_attr_type(ident).map(|kind| (kind, None))
}

fn parse_one(ty: &Type, vars: &[TypeVar]) -> Result<TypeExpr, String> {
    match ty {
        Type::Infer(_) => Ok(TypeExpr::Any),
        Type::Tuple(_) => Err("tuple type is not allowed in Value/Variadic".into()),
        Type::ImplTrait(tr) => Ok(TypeExpr::Anon(paths(&tr.bounds)?)),
        Type::Path(path) => {
            if let Some(proj) = projection(path, vars)? {
                return Ok(TypeExpr::Proj(proj));
            }
            check_path(&path.path)?;
            if let Some(var) = simple_var(&path.path, vars) {
                return Ok(TypeExpr::Var(var));
            }
            Ok(TypeExpr::Anon(vec![path.path.clone()]))
        }
        _ => Err("invalid single-type constraint".into()),
    }
}

fn parse_list(ty: &Type, vars: &[TypeVar]) -> Result<ListExpr, String> {
    match ty {
        Type::Tuple(t) => t
            .elems
            .iter()
            .map(|e| parse_one(e, vars))
            .collect::<Result<_, _>>()
            .map(ListExpr::Types),
        Type::Path(path) => projection(path, vars)?
            .map(ListExpr::Proj)
            .ok_or_else(|| "Values<..> requires a type list".into()),
        _ => Err("Values<..> requires a type list".into()),
    }
}

fn projection(path: &syn::TypePath, vars: &[TypeVar]) -> Result<Option<Projection>, String> {
    let (var, bound, name) = if let Some(qself) = &path.qself {
        let Type::Path(var_ty) = &*qself.ty else {
            return Err("projection needs a declared variable".into());
        };
        let Some(var) = simple_var(&var_ty.path, vars) else {
            return Err("projection needs a declared variable".into());
        };
        let Some(last) = path.path.segments.last() else {
            return Err("invalid projection".into());
        };
        let segments: Vec<_> = path.path.segments.iter().take(qself.position).collect();
        let bound: Path = syn::parse2(quote!(#(#segments)::*))
            .map_err(|_| "invalid qualified projection bound")?;
        (var, Some(bound), last.ident.to_string())
    } else if path.path.segments.len() == 2 {
        let mut it = path.path.segments.iter();
        let first = it.next().unwrap();
        let last = it.next().unwrap();
        let Some(var) = vars.iter().position(|v| first.ident == v.name) else {
            return Ok(None);
        };
        (var, None, last.ident.to_string())
    } else {
        return Ok(None);
    };
    if name == "Type" {
        return Err("V::Type in a value position is reserved; use V".into());
    }
    if let Some(bound) = &bound {
        check_path(bound)?;
        if !vars[var]
            .bounds
            .iter()
            .any(|b| quote!(#b).to_string() == quote!(#bound).to_string())
        {
            return Err("qualified projection bound is not declared on the variable".into());
        }
    }
    Ok(Some(Projection { var, bound, name }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_op(item: TokenStream) -> Result<OperationDef, String> {
        let module = parse_input(quote!(), quote!(mod d { #item }))?;
        match module.items.into_iter().next() {
            Some(DialectItem::Operation(op)) => Ok(op),
            _ => panic!("expected an operation"),
        }
    }

    fn parse_err(item: TokenStream) -> String {
        match parse_op(item) {
            Ok(_) => panic!("expected a parse error"),
            Err(err) => err,
        }
    }

    #[test]
    fn typed_operands_bind_variables_and_results() {
        let op = parse_op(quote! {
            fn addi<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
        })
        .unwrap();
        assert!(op.syntax == Syntax::Typed);
        assert_eq!(op.type_vars.len(), 1);
        assert_eq!(op.type_vars[0].bounds.len(), 1);
        assert!(matches!(
            op.operands[1].constraint,
            ValueExpr::Each(TypeExpr::Var(0))
        ));
        assert!(matches!(op.results, ResultDef::Single(ref name) if name == "result"));
        assert!(matches!(
            op.result_constraint,
            ValueExpr::Each(TypeExpr::Var(0))
        ));
    }

    #[test]
    fn typed_attributes_and_projections() {
        let op = parse_op(quote! {
            fn call_indirect<S: clif::FuncSig>(
                sig: Attr<S::Type>,
                callee: Value<core::Ptr>,
                args: Values<S::Inputs>,
                tag: Option<Attr<Symbol>>,
                raw: Attr<_>,
            ) -> Values<S::Results> {}
        })
        .unwrap();
        assert_eq!(op.attrs.len(), 3);
        assert!(matches!(op.attrs[0].ty, AttrType::Type));
        assert_eq!(op.attrs[0].binds, Some(0));
        assert!(op.attrs[1].optional);
        assert!(matches!(op.attrs[2].ty, AttrType::Any));
        assert!(
            matches!(op.operands[0].constraint, ValueExpr::Each(TypeExpr::Anon(ref b)) if b.len() == 1)
        );
        assert!(op.operands[1].variadic);
        assert!(matches!(
            op.operands[1].constraint,
            ValueExpr::List(ListExpr::Proj(Projection { var: 0, bound: None, ref name })) if name == "Inputs"
        ));
        assert!(matches!(op.results, ResultDef::Variadic(ref name) if name == "results"));
    }

    #[test]
    fn where_clauses_merge_bounds_and_qualify_projections() {
        let op = parse_op(quote! {
            fn call<S: A>(callee: Value<S>, args: Values<<S as func::FuncSig>::Inputs>)
                -> Option<Value<_>>
            where
                S: func::FuncSig,
            {}
        })
        .unwrap();
        assert_eq!(op.type_vars[0].bounds.len(), 2);
        assert!(matches!(
            op.operands[1].constraint,
            ValueExpr::List(ListExpr::Proj(Projection { bound: Some(_), .. }))
        ));
        assert!(matches!(op.results, ResultDef::Optional(_)));
    }

    #[test]
    fn typed_lists_impl_bounds_and_regions() {
        let op = parse_op(quote! {
            fn pack<T>(xs: Values<(T, impl IntegerLike + BoolLike, _)>, rest: ()) {}
        });
        assert!(op.is_err(), "a legacy operand in a typed op is rejected");

        let op = parse_op(quote! {
            fn select(cond: Value<impl BoolLike>, elems: Values<()>) -> Variadic<_> {
                #[region(body)] {}
            }
        })
        .unwrap();
        assert!(matches!(
            op.operands[0].constraint,
            ValueExpr::Each(TypeExpr::Anon(_))
        ));
        assert!(
            matches!(op.operands[1].constraint, ValueExpr::List(ListExpr::Types(ref t)) if t.is_empty())
        );
        assert!(matches!(
            op.result_constraint,
            ValueExpr::Each(TypeExpr::Any)
        ));
        assert_eq!(op.regions.len(), 1);
    }

    #[test]
    fn legacy_optional_results_stay_legacy() {
        let op = parse_op(quote! {
            fn r#if(cond: ()) -> Option<result> {
                #[region(then_region)] {}
            }
        })
        .unwrap();
        assert!(op.syntax == Syntax::Legacy);
        assert!(matches!(op.results, ResultDef::Optional(_)));
    }

    #[test]
    fn typed_syntax_errors() {
        let cases = [
            (
                quote!(
                    #[attr(p: Symbol)]
                    fn f(x: Value<_>) {}
                ),
                "cannot mix legacy and new operation syntax",
            ),
            (
                quote!(
                    fn f(x: Value<_>, #[rest] ys: ()) {}
                ),
                "cannot mix legacy and new operation syntax",
            ),
            (
                quote!(
                    fn f(x: Value<_>) -> result {}
                ),
                "expected wrapper",
            ),
            (
                quote!(
                    fn f(xs: Variadic<_>, y: Value<_>) {}
                ),
                "variadic/Values operand must be last",
            ),
            (
                quote!(
                    fn f<T>(xs: Values<T>) {}
                ),
                "Values<..> requires a type list",
            ),
            (
                quote!(
                    fn f(x: Value<(A, B)>) {}
                ),
                "tuple type is not allowed in Value/Variadic",
            ),
            (
                quote!(
                    fn f(x: Option<Value<_>>) {}
                ),
                "Option<Value<..>> operand is reserved",
            ),
            (
                quote!(
                    fn f() -> (Value<_>, Value<_>) {}
                ),
                "tuple results are reserved",
            ),
            (
                quote!(
                    fn f<T>(x: Value<T::Type>) {}
                ),
                "V::Type in a value position",
            ),
            (
                quote!(
                    fn f<S: A>(x: Values<<S as B>::Xs>) {}
                ),
                "qualified projection bound is not declared",
            ),
            (
                quote!(
                    fn f(t: Attr<U::Type>) {}
                ),
                "attribute projection must be V::Type",
            ),
            (
                quote!(
                    fn f(results: Value<_>) {}
                ),
                "collides with a builder method",
            ),
            (
                quote!(
                    fn f(x: Value<_>, x: Value<_>) {}
                ),
                "duplicate entity name",
            ),
            (
                quote!(
                    fn f<T>(x: Value<_>)
                    where
                        U: A,
                    {
                    }
                ),
                "declared type variable",
            ),
        ];
        for (item, expected) in cases {
            let err = parse_err(item.clone());
            assert!(
                err.contains(expected),
                "`{item}`: expected `{expected}`, got `{err}`"
            );
        }
    }
}
