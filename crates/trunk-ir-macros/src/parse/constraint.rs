//! Parsing for typed operation signatures. The legacy DSL remains in `parse.rs`.
//!
//! The signature after the operation name is parsed into a small type
//! expression tree ([`Ty`]) and then interpreted as entity wrappers, type
//! variables, bounds, and projections.

use super::*;
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote};

// ============================================================================
// Parsed constraint model
// ============================================================================

/// A bound path such as `IntegerLike` or `func::FuncSig`.
#[derive(Clone)]
pub struct BoundPath {
    leading_colon: bool,
    segments: Vec<Ident>,
}

impl BoundPath {
    pub fn span(&self) -> Span {
        self.segments[0].span()
    }

    fn key(&self) -> String {
        let path = self
            .segments
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("::");
        if self.leading_colon {
            format!("::{path}")
        } else {
            path
        }
    }
}

impl ToTokens for BoundPath {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let segments = &self.segments;
        if self.leading_colon {
            tokens.extend(quote!(:: #(#segments)::*));
        } else {
            tokens.extend(quote!(#(#segments)::*));
        }
    }
}

pub struct TypeVar {
    pub name: String,
    pub bounds: Vec<BoundPath>,
}

#[derive(Clone)]
pub struct Projection {
    pub var: usize,
    pub bound: Option<BoundPath>,
    pub name: String,
}

#[derive(Clone)]
pub enum TypeExpr {
    Any,
    Var(usize),
    Anon(Vec<BoundPath>),
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

// ============================================================================
// Type expression syntax
// ============================================================================

/// A type written in a typed operation signature.
#[derive(Clone)]
enum Ty {
    /// `_`
    Infer,
    /// `(A, B)`, `()`
    Tuple(Vec<Ty>),
    /// `impl A + B`
    Impl(Vec<BoundPath>),
    /// `a::B<C>`, `S::X`, `<S as B>::X`
    Path(TyPath),
    /// Any other type form (references, slices, ...), which is always rejected.
    Other,
}

#[derive(Clone)]
struct TyPath {
    /// `<ty as bound>` of a qualified path.
    qself: Option<(Box<Ty>, Option<Box<TyPath>>)>,
    leading_colon: bool,
    segments: Vec<Segment>,
}

#[derive(Clone)]
struct Segment {
    ident: Ident,
    args: Option<Vec<GenericArg>>,
}

#[derive(Clone)]
enum GenericArg {
    Type(Ty),
    Lifetime,
}

fn peek_ident(iter: &TokenIter, name: &str) -> bool {
    matches!(iter.clone().next(), Some(TokenTree::Ident(i)) if i == name)
}

fn eat_path_sep(iter: &mut TokenIter) -> bool {
    let mut look = iter.clone();
    let is_sep = matches!(look.next(), Some(TokenTree::Punct(p)) if p.as_char() == ':')
        && matches!(look.next(), Some(TokenTree::Punct(p)) if p.as_char() == ':');
    if is_sep {
        *iter = look;
    }
    is_sep
}

fn skip_lifetime(iter: &mut TokenIter) {
    iter.next();
    iter.next();
}

fn parse_ty(iter: &mut TokenIter) -> Result<Ty, String> {
    match iter.clone().next() {
        Some(TokenTree::Ident(i)) if i == "_" => {
            iter.next();
            Ok(Ty::Infer)
        }
        Some(TokenTree::Ident(i)) if i == "impl" => {
            iter.next();
            Ok(Ty::Impl(parse_bounds(iter)?))
        }
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
            iter.next();
            parse_ty_list(g.stream()).map(Ty::Tuple)
        }
        Some(TokenTree::Punct(p)) if p.as_char() == '<' => parse_qualified(iter).map(Ty::Path),
        Some(TokenTree::Ident(_)) => parse_ty_path(iter).map(Ty::Path),
        Some(TokenTree::Punct(p)) if p.as_char() == ':' => parse_ty_path(iter).map(Ty::Path),
        Some(_) => {
            skip_ty(iter);
            Ok(Ty::Other)
        }
        None => Err("expected a type".into()),
    }
}

/// Skip the rest of an unsupported type, up to a top-level `,` or `>`.
fn skip_ty(iter: &mut TokenIter) {
    let mut depth = 0usize;
    while let Some(tt) = iter.clone().next() {
        if let TokenTree::Punct(p) = &tt {
            match p.as_char() {
                '<' => depth += 1,
                '>' if depth == 0 => return,
                '>' => depth -= 1,
                ',' if depth == 0 => return,
                _ => {}
            }
        }
        iter.next();
    }
}

fn parse_ty_list(stream: TokenStream) -> Result<Vec<Ty>, String> {
    let mut iter = stream.to_token_iter();
    let mut tys = Vec::new();
    while has_remaining(&iter) {
        tys.push(parse_ty(&mut iter)?);
        if has_remaining(&iter) {
            expect_punct(&mut iter, ',')?;
        }
    }
    Ok(tys)
}

fn parse_ty_path(iter: &mut TokenIter) -> Result<TyPath, String> {
    let leading_colon = eat_path_sep(iter);
    let segments = parse_segments(iter)?;
    Ok(TyPath {
        qself: None,
        leading_colon,
        segments,
    })
}

fn parse_segments(iter: &mut TokenIter) -> Result<Vec<Segment>, String> {
    let mut segments = Vec::new();
    loop {
        let ident = Ident::parser(iter).map_err(|e| format!("expected a path segment: {e}"))?;
        let args = if peek_punct(iter, '<') {
            Some(parse_generic_args(iter)?)
        } else {
            None
        };
        segments.push(Segment { ident, args });
        if !eat_path_sep(iter) {
            return Ok(segments);
        }
    }
}

fn parse_generic_args(iter: &mut TokenIter) -> Result<Vec<GenericArg>, String> {
    expect_punct(iter, '<')?;
    let mut args = Vec::new();
    while !peek_punct(iter, '>') {
        if peek_punct(iter, '\'') {
            skip_lifetime(iter);
            args.push(GenericArg::Lifetime);
        } else {
            args.push(GenericArg::Type(parse_ty(iter)?));
        }
        if !peek_punct(iter, '>') {
            expect_punct(iter, ',')?;
        }
    }
    expect_punct(iter, '>')?;
    Ok(args)
}

/// Parse `<ty as bound>::segments` (or `<ty>::segments`).
fn parse_qualified(iter: &mut TokenIter) -> Result<TyPath, String> {
    expect_punct(iter, '<')?;
    let ty = parse_ty(iter)?;
    let bound = if peek_ident(iter, "as") {
        iter.next();
        Some(Box::new(parse_ty_path(iter)?))
    } else {
        None
    };
    expect_punct(iter, '>')?;
    if !eat_path_sep(iter) {
        return Err("invalid projection".into());
    }
    Ok(TyPath {
        qself: Some((Box::new(ty), bound)),
        leading_colon: false,
        segments: parse_segments(iter)?,
    })
}

fn parse_bounds(iter: &mut TokenIter) -> Result<Vec<BoundPath>, String> {
    let mut bounds = Vec::new();
    loop {
        match iter.clone().next() {
            Some(TokenTree::Ident(_)) => {}
            Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
            _ => return Err("bounds must be plain Rust paths".into()),
        }
        bounds.push(bound_path(&parse_ty_path(iter)?)?);
        if !peek_punct(iter, '+') {
            return Ok(bounds);
        }
        consume_punct(iter)?;
    }
}

/// Convert an unqualified path to a bound; qualified paths are projections.
fn bound_path(path: &TyPath) -> Result<BoundPath, String> {
    debug_assert!(path.qself.is_none());
    if path.segments.iter().any(|s| s.args.is_some()) {
        return Err("generic arguments on bound paths are not supported".into());
    }
    Ok(BoundPath {
        leading_colon: path.leading_colon,
        segments: path.segments.iter().map(|s| s.ident.clone()).collect(),
    })
}

// ============================================================================
// Typed operation parsing
// ============================================================================

/// Whether the signature after the operation name uses the typed syntax:
/// generic parameters, or a typed entity wrapper (`Value<..>`, `Variadic<..>`,
/// `Values<..>`, `Attr<..>`) in the parameters or the return type. Legacy
/// results may be `-> Option<result>`, so `Option` alone is not a marker.
pub(super) fn is_typed_operation(iter: &TokenIter) -> Result<bool, String> {
    if peek_punct(iter, '<') {
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
        match iter.next().ok_or("expected operation body")? {
            TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => break g,
            tt => sig_tokens.extend([tt]),
        }
    };
    let mut sig = sig_tokens.to_token_iter();

    let mut vars = parse_generics(&mut sig)?;
    let params = parse_params(expect_group(&mut sig, Delimiter::Parenthesis)?.stream())?;
    let output = if peek_punct(&sig, '-') {
        expect_punct(&mut sig, '-')?;
        expect_punct(&mut sig, '>')?;
        Some(parse_ty(&mut sig)?)
    } else {
        None
    };
    if peek_ident(&sig, "where") {
        sig.next();
        parse_where_clause(&mut sig, &mut vars)?;
    }
    expect_consumed(&sig, "typed operation signature")?;

    let mut attrs = Vec::new();
    let mut operands = Vec::new();
    let mut names = std::collections::HashSet::new();
    for (ident, ty) in &params {
        let name = ident_str(ident);
        check_name(&name, &mut names)?;
        if matches!(ty, Ty::Tuple(t) if t.is_empty()) {
            return Err("cannot mix legacy and new operation syntax".into());
        }
        let (wrapper, inner, optional) = unwrap_wrapper(ty)?;
        match wrapper.as_str() {
            "Attr" => {
                let (ty, binds) = parse_attr_kind(inner, &vars)?;
                attrs.push(AttrDef {
                    name,
                    raw_ident: ident.clone(),
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
                    raw_ident: ident.clone(),
                    variadic,
                    constraint,
                });
            }
            "Value" if optional => return Err("Option<Value<..>> operand is reserved".into()),
            _ => return Err("expected Value, Variadic, Values, Attr, or Option<Attr>".into()),
        }
    }
    let (results, result_constraint) = match &output {
        None => (ResultDef::None, ValueExpr::Each(TypeExpr::Any)),
        Some(Ty::Tuple(_)) => return Err("tuple results are reserved".into()),
        Some(ty) => {
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
    if params.is_empty() && output.is_none() && !vars.is_empty() {
        return Err("generics require new operation syntax".into());
    }
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

/// Parse `<T: A + B, U>`.
fn parse_generics(iter: &mut TokenIter) -> Result<Vec<TypeVar>, String> {
    let mut vars = Vec::<TypeVar>::new();
    if !peek_punct(iter, '<') {
        return Ok(vars);
    }
    consume_punct(iter)?;
    while !peek_punct(iter, '>') {
        if peek_punct(iter, '\'') || peek_ident(iter, "const") {
            return Err("only type variables are supported in typed operations".into());
        }
        let ident = Ident::parser(iter).map_err(|e| format!("expected type variable: {e}"))?;
        let name = ident_str(&ident);
        if vars.iter().any(|v| v.name == name) {
            return Err(format!("duplicate type variable `{name}`"));
        }
        let bounds = if peek_punct(iter, ':') {
            consume_punct(iter)?;
            parse_bounds(iter)?
        } else {
            Vec::new()
        };
        vars.push(TypeVar { name, bounds });
        if !peek_punct(iter, '>') {
            expect_punct(iter, ',')?;
        }
    }
    consume_punct(iter)?;
    Ok(vars)
}

/// Parse `where T: A, U: B`, merging the bounds into the declared variables.
fn parse_where_clause(iter: &mut TokenIter, vars: &mut [TypeVar]) -> Result<(), String> {
    while has_remaining(iter) {
        if peek_punct(iter, '\'') {
            return Err("unsupported where predicate".into());
        }
        let bounded = parse_ty(iter)?;
        expect_punct(iter, ':')?;
        let bounds = parse_bounds(iter)?;
        let var = match &bounded {
            Ty::Path(path) => {
                simple_var(path, vars).ok_or("where bound must name a declared type variable")?
            }
            _ => return Err("where bound must name a type variable".into()),
        };
        vars[var].bounds.extend(bounds);
        if has_remaining(iter) {
            expect_punct(iter, ',')?;
        }
    }
    Ok(())
}

/// Parse `name: Type` parameters.
fn parse_params(stream: TokenStream) -> Result<Vec<(Ident, Ty)>, String> {
    let mut iter = stream.to_token_iter();
    let mut params = Vec::new();
    while has_remaining(&iter) {
        if peek_punct(&iter, '#') {
            return Err("cannot mix legacy and new operation syntax".into());
        }
        if peek_punct(&iter, '&') || peek_ident(&iter, "self") {
            return Err("self parameter is not supported".into());
        }
        let ident = match iter.next() {
            Some(TokenTree::Ident(ident)) if ident != "mut" => ident,
            _ => return Err("parameter must have a name".into()),
        };
        expect_punct(&mut iter, ':')?;
        params.push((ident, parse_ty(&mut iter)?));
        if has_remaining(&iter) {
            expect_punct(&mut iter, ',')?;
        }
    }
    Ok(params)
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

fn simple_var(path: &TyPath, vars: &[TypeVar]) -> Option<usize> {
    match (&path.qself, path.leading_colon, path.segments.as_slice()) {
        (None, false, [Segment { ident, args: None }]) => {
            vars.iter().position(|v| *ident == v.name)
        }
        _ => None,
    }
}

fn unwrap_wrapper(ty: &Ty) -> Result<(String, &Ty, bool), String> {
    let Ty::Path(path) = ty else {
        return Err("expected a typed entity wrapper".into());
    };
    let (None, false, [seg]) = (&path.qself, path.leading_colon, path.segments.as_slice()) else {
        return Err("expected a typed entity wrapper".into());
    };
    let Some(args) = &seg.args else {
        return Err("expected wrapper<..>".into());
    };
    let [arg] = args.as_slice() else {
        return Err("entity wrapper requires one type argument".into());
    };
    let GenericArg::Type(inner) = arg else {
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

fn parse_attr_kind(ty: &Ty, vars: &[TypeVar]) -> Result<(AttrType, Option<usize>), String> {
    let path = match ty {
        Ty::Infer => return Ok((AttrType::Any, None)),
        Ty::Path(path) => path,
        _ => return Err("invalid attribute kind".into()),
    };
    if path.qself.is_some() {
        return Err("invalid attribute projection".into());
    }
    if path.leading_colon || path.segments.iter().any(|s| s.args.is_some()) {
        return Err("invalid attribute kind".into());
    }
    match path.segments.as_slice() {
        [kind] => parse_attr_type(&kind.ident).map(|kind| (kind, None)),
        [var, proj] => {
            if proj.ident == "Type"
                && let Some(var) = vars.iter().position(|v| var.ident == v.name)
            {
                return Ok((AttrType::Type, Some(var)));
            }
            Err("attribute projection must be V::Type for a declared variable".into())
        }
        _ => Err("invalid attribute kind".into()),
    }
}

fn parse_one(ty: &Ty, vars: &[TypeVar]) -> Result<TypeExpr, String> {
    match ty {
        Ty::Infer => Ok(TypeExpr::Any),
        Ty::Tuple(_) => Err("tuple type is not allowed in Value/Variadic".into()),
        Ty::Impl(bounds) => Ok(TypeExpr::Anon(bounds.clone())),
        Ty::Path(path) => {
            if let Some(proj) = projection(path, vars)? {
                return Ok(TypeExpr::Proj(proj));
            }
            let bound = bound_path(path)?;
            if let Some(var) = simple_var(path, vars) {
                return Ok(TypeExpr::Var(var));
            }
            Ok(TypeExpr::Anon(vec![bound]))
        }
        Ty::Other => Err("invalid single-type constraint".into()),
    }
}

fn parse_list(ty: &Ty, vars: &[TypeVar]) -> Result<ListExpr, String> {
    match ty {
        Ty::Tuple(tys) => tys
            .iter()
            .map(|ty| parse_one(ty, vars))
            .collect::<Result<_, _>>()
            .map(ListExpr::Types),
        Ty::Path(path) => projection(path, vars)?
            .map(ListExpr::Proj)
            .ok_or_else(|| "Values<..> requires a type list".into()),
        _ => Err("Values<..> requires a type list".into()),
    }
}

/// Interpret `V::Name` or `<V as B>::Name` as a projection of a declared
/// variable. Returns `None` for paths that do not start with a variable.
fn projection(path: &TyPath, vars: &[TypeVar]) -> Result<Option<Projection>, String> {
    let (var, bound, name) = if let Some((qself, bound)) = &path.qself {
        let var = match &**qself {
            Ty::Path(var_path) => simple_var(var_path, vars),
            _ => None,
        }
        .ok_or("projection needs a declared variable")?;
        let bound = bound.as_deref().map(bound_path).transpose()?;
        let [Segment { ident, args: None }] = path.segments.as_slice() else {
            return Err("invalid projection".into());
        };
        (var, bound, ident.to_string())
    } else {
        let (false, [first, Segment { ident, args: None }]) =
            (path.leading_colon, path.segments.as_slice())
        else {
            return Ok(None);
        };
        let (None, Some(var)) = (&first.args, vars.iter().position(|v| first.ident == v.name))
        else {
            return Ok(None);
        };
        (var, None, ident.to_string())
    };
    if name == "Type" {
        return Err("V::Type in a value position is reserved; use V".into());
    }
    if let Some(bound) = &bound
        && !vars[var].bounds.iter().any(|b| b.key() == bound.key())
    {
        return Err("qualified projection bound is not declared on the variable".into());
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
    fn qualified_projections_without_as_and_absolute_bounds() {
        let op = parse_op(quote! {
            fn f<S: ::a::FuncSig>(
                x: Value<<S>::Input>,
                ys: Values<<S as ::a::FuncSig>::Inputs>,
            ) {}
        })
        .unwrap();
        assert!(matches!(
            op.operands[0].constraint,
            ValueExpr::Each(TypeExpr::Proj(Projection { bound: None, .. }))
        ));
        assert!(matches!(
            op.operands[1].constraint,
            ValueExpr::List(ListExpr::Proj(Projection { bound: Some(ref b), .. })) if b.key() == "::a::FuncSig"
        ));
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
            (quote!(fn f(x: , y: Value<_>) {}), "expected a type"),
            (
                quote!(
                    fn f(x: &T, y: Value<_>) {}
                ),
                "expected a typed entity wrapper",
            ),
            (
                quote!(fn f<S: B>(x: Value<<S as B>>) {}),
                "invalid projection",
            ),
            (
                quote!(
                    fn f<S: B>(x: Value<<S as B>::X::Y>) {}
                ),
                "invalid projection",
            ),
            (
                quote!(
                    fn f(x: Attr<::Symbol>) {}
                ),
                "invalid attribute kind",
            ),
            (
                quote!(
                    fn f(mut x: Value<_>) {}
                ),
                "parameter must have a name",
            ),
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
            (
                quote!(
                    fn f(x: Value<_>);
                ),
                "expected operation body",
            ),
            (
                quote!(
                    fn f<T, T>(x: Value<T>) {}
                ),
                "duplicate type variable",
            ),
            (
                quote!(
                    fn f<'a>(x: Value<_>) {}
                ),
                "only type variables",
            ),
            (
                quote!(
                    fn f<T>(x: Value<T>)
                    where
                        'a: 'b,
                    {
                    }
                ),
                "unsupported where predicate",
            ),
            (
                quote!(
                    fn f<T>(x: Value<T>)
                    where
                        (T,): A,
                    {
                    }
                ),
                "where bound must name a type variable",
            ),
            (
                quote!(
                    fn f<T>() {}
                ),
                "generics require new operation syntax",
            ),
            (
                quote!(
                    fn f(self, x: Value<_>) {}
                ),
                "self parameter",
            ),
            (
                quote!(
                    fn f((a, b): Value<_>) {}
                ),
                "parameter must have a name",
            ),
            (
                quote!(
                    fn f(x: Foo<_>, y: Value<_>) {}
                ),
                "expected Value, Variadic, Values, Attr",
            ),
            (
                quote!(
                    fn f() -> Option<Variadic<_>> {}
                ),
                "expected Value, Option<Value>",
            ),
            (
                quote!(
                    fn f<T: 'static>(x: Value<T>) {}
                ),
                "bounds must be plain Rust paths",
            ),
            (
                quote!(
                    fn f<T: ?Sized>(x: Value<T>) {}
                ),
                "bounds must be plain Rust paths",
            ),
            (
                quote!(
                    fn f<T: A<B>>(x: Value<T>) {}
                ),
                "generic arguments on bound paths",
            ),
            (
                quote!(
                    fn f(x: &Value<_>) {}
                ),
                "expected a typed entity wrapper",
            ),
            (
                quote!(
                    fn f(x: a::Value<_>) {}
                ),
                "expected a typed entity wrapper",
            ),
            (
                quote!(
                    fn f(x: Value<A, B>) {}
                ),
                "requires one type argument",
            ),
            (
                quote!(
                    fn f(x: Value<'a>) {}
                ),
                "requires a type argument",
            ),
            (
                quote!(
                    fn f(x: Option<Option<Attr<_>>>) {}
                ),
                "nested Option",
            ),
            (
                quote!(
                    fn f(x: Attr<&str>) {}
                ),
                "invalid attribute kind",
            ),
            (
                quote!(
                    fn f<S: B>(x: Attr<<S as B>::Type>) {}
                ),
                "invalid attribute projection",
            ),
            (
                quote!(
                    fn f(x: Attr<a::b::c>) {}
                ),
                "invalid attribute kind",
            ),
            (
                quote!(
                    fn f(x: Attr<Unknown>) {}
                ),
                "unknown attribute type",
            ),
            (
                quote!(
                    fn f<T>(x: Value<&T>) {}
                ),
                "invalid single-type constraint",
            ),
            (
                quote!(
                    fn f<T>(xs: Values<[T]>) {}
                ),
                "Values<..> requires a type list",
            ),
            (
                quote!(
                    fn f(x: Value<<(A,) as B>::X>) {}
                ),
                "projection needs a declared variable",
            ),
            (
                quote!(
                    fn f(x: Value<<U as B>::X>) {}
                ),
                "projection needs a declared variable",
            ),
            (
                quote!(
                    fn f<T>(x: Value<T<U>>) {}
                ),
                "generic arguments on bound paths",
            ),
        ];
        let failures: Vec<String> = cases
            .into_iter()
            .filter_map(|(item, expected)| {
                let err = parse_err(item.clone());
                (!err.contains(expected))
                    .then(|| format!("`{item}`: expected `{expected}`, got `{err}`"))
            })
            .collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
