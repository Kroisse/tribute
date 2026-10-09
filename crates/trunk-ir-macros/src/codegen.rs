//! Code generation for `#[dialect]`.

use heck::{ToShoutySnakeCase, ToSnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};

use crate::parse::{
    AttrDef, AttrKind, BoundPath, DialectItem, DialectModule, ListExpr, Operand, OperationDef,
    Projection, RegionOrSuccessor, RelationDef, RelationExpr, ResultDef, TypeDefData, TypeExpr,
    ValueExpr,
};

/// Generate all code for a dialect module.
pub fn generate(crate_path: &TokenStream, module: &DialectModule) -> TokenStream {
    let dialect_name_fn = gen_dialect_name(crate_path, &module.name);

    let mut items = Vec::new();
    for item in &module.items {
        match item {
            DialectItem::Operation(op) => {
                items.push(gen_operation(crate_path, &module.name, op));
            }
            DialectItem::TypeDef(td) => {
                items.push(gen_type_def(crate_path, &module.name, td));
            }
        }
    }

    quote! {
        #dialect_name_fn
        #(#items)*
    }
}

fn gen_dialect_name(crate_path: &TokenStream, dialect: &str) -> TokenStream {
    quote! {
        #[allow(non_snake_case)]
        #[inline]
        pub fn DIALECT_NAME() -> #crate_path::Symbol {
            #crate_path::Symbol::new(#dialect)
        }
    }
}

fn gen_operation(crate_path: &TokenStream, dialect: &str, op: &OperationDef) -> TokenStream {
    let op_name_fn = gen_op_name_fn(crate_path, &op.name);
    let struct_and_trait = gen_struct_and_trait(crate_path, dialect, op);
    let impl_block = gen_impl_block(crate_path, op);
    let builder = gen_fluent_builder(crate_path, dialect, op);

    quote! {
        #op_name_fn
        #struct_and_trait
        #impl_block
        #builder
    }
}

fn gen_op_name_fn(crate_path: &TokenStream, op_name: &str) -> TokenStream {
    let upper_name = format_ident!("{}", op_name.to_shouty_snake_case());
    quote! {
        #[allow(non_snake_case)]
        #[inline]
        pub fn #upper_name() -> #crate_path::Symbol {
            #crate_path::Symbol::new(#op_name)
        }
    }
}

fn struct_name(op_name: &str) -> proc_macro2::Ident {
    format_ident!("{}", op_name.to_upper_camel_case())
}

fn gen_struct_and_trait(crate_path: &TokenStream, dialect: &str, op: &OperationDef) -> TokenStream {
    let sname = struct_name(&op.name);
    let op_name = &op.name;
    let full_name = format!("{dialect}.{op_name}");
    let def = gen_op_def(crate_path, dialect, op);

    quote! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct #sname(#crate_path::OpRef);

        impl #crate_path::ops::DialectOp for #sname {
            const DIALECT_NAME: &'static str = #dialect;
            const OP_NAME: &'static str = #op_name;
            const DEF: &'static #crate_path::op_def::OpDef = &#def;

            fn from_op(
                ctx: &#crate_path::IrContext,
                op: #crate_path::OpRef,
            ) -> Result<Self, #crate_path::ops::ConversionError> {
                if !Self::matches(ctx, op) {
                    return Err(#crate_path::ops::ConversionError::WrongOperation {
                        expected: #full_name,
                        actual_dialect: ctx.op(op).dialect.clone(),
                        actual_name: ctx.op(op).name.clone(),
                    });
                }
                Ok(Self(op))
            }

            fn op_ref(&self) -> #crate_path::OpRef {
                self.0
            }
        }

        #crate_path::inventory::submit! {
            #crate_path::op_def::OpDefRegistration(
                <#sname as #crate_path::ops::DialectOp>::DEF
            )
        }
    }
}

/// Build the static `OpDef` expression for an operation.
fn gen_op_def(crate_path: &TokenStream, dialect: &str, op: &OperationDef) -> TokenStream {
    let op_name = &op.name;
    let schema_mod = quote!(#crate_path::op_schema);
    let type_vars = op.type_vars.iter().map(|var| {
        let name = &var.name;
        let bounds = gen_bounds(crate_path, &var.bounds);
        quote!(#schema_mod::TypeVarSchema { name: #name, bounds: #bounds })
    });

    let operands = op.operands.iter().map(|operand| {
        let name = &operand.name;
        let arity = if operand.variadic {
            quote!(#schema_mod::Arity::Variadic)
        } else {
            quote!(#schema_mod::Arity::One)
        };
        let constraint = gen_value_expr(crate_path, &operand.constraint, op);
        quote!(#schema_mod::OperandSchema { name: #name, arity: #arity, constraint: #constraint })
    });

    let results = match &op.results {
        ResultDef::None => quote!(#schema_mod::ResultSchema::Fixed(&[])),
        ResultDef::Single(name) => quote!(#schema_mod::ResultSchema::Fixed(&[#name])),
        ResultDef::Variadic(name) => quote!(#schema_mod::ResultSchema::Variadic(#name)),
        ResultDef::Optional(name) => quote!(#schema_mod::ResultSchema::Optional(#name)),
    };
    let result_constraint = gen_value_expr(crate_path, &op.result_constraint, op);

    let attributes = op.attrs.iter().map(|attr| {
        let name = &attr.name;
        let kind = attr_kind_trait(crate_path, attr);
        let kind = quote!(#kind::KIND);
        let optional = attr.optional;
        let binds = match attr.binds {
            Some(i) => quote!(Some(#i)),
            None => quote!(None),
        };
        quote!(#schema_mod::AttributeSchema { name: #name, kind: #kind, optional: #optional, binds: #binds })
    });

    let regions = op.regions.iter().filter_map(|item| match item {
        RegionOrSuccessor::Region { name, optional } => {
            Some(quote!(#schema_mod::RegionSchema { name: #name, optional: #optional }))
        }
        RegionOrSuccessor::Successor { .. } => None,
    });
    let successors = op.regions.iter().filter_map(|item| match item {
        RegionOrSuccessor::Successor { name, variadic } => {
            Some(quote!(#schema_mod::SuccessorSchema { name: #name, variadic: #variadic }))
        }
        RegionOrSuccessor::Region { .. } => None,
    });

    let relations = op
        .relations
        .iter()
        .map(|relation| gen_relation(crate_path, relation, op));

    // The span points a missing `Verify` impl at the `#[verify]` attribute.
    let verifier = match op.verify {
        Some(span) => {
            let mut sname = struct_name(&op.name);
            sname.set_span(span);
            quote_spanned!(span=> Some(|ctx, op| {
                <#sname as #crate_path::ops::__private::DeclaredVerify>::verify_declared(#sname(op), ctx)
            }))
        }
        None => quote!(None),
    };

    quote! {
        #crate_path::op_def::OpDef {
            schema: #schema_mod::OpSchema {
                dialect: #dialect,
                name: #op_name,
                type_vars: &[#(#type_vars),*],
                operands: &[#(#operands),*],
                results: #results,
                result_constraint: #result_constraint,
                attributes: &[#(#attributes),*],
                regions: &[#(#regions),*],
                successors: &[#(#successors),*],
                relations: &[#(#relations),*],
            },
            verifier: #verifier,
        }
    }
}

fn gen_bounds(crate_path: &TokenStream, paths: &[BoundPath]) -> TokenStream {
    let descriptors = paths.iter().map(|path| {
        quote_spanned!(path.span()=> <#path as #crate_path::type_constraint::TypeConstraint>::DESC)
    });
    quote!({
        const B: &[&#crate_path::type_constraint::ConstraintDesc] = &[#(#descriptors),*];
        #crate_path::type_constraint::check_bounds(B)
    })
}

/// Resolve a projection at compile time. With `kind`, the projection must
/// have that kind; without, any kind is accepted.
fn gen_projection(
    crate_path: &TokenStream,
    projection: &Projection,
    op: &OperationDef,
    kind: Option<TokenStream>,
) -> TokenStream {
    let schema = quote!(#crate_path::op_schema);
    let tc = quote!(#crate_path::type_constraint);
    let var = projection.var;
    let name = &projection.name;
    let bounds = gen_bounds(crate_path, &op.type_vars[var].bounds);
    if let Some(bound) = &projection.bound {
        let index = op.type_vars[var]
            .bounds
            .iter()
            .position(|p| quote!(#p).to_string() == quote!(#bound).to_string())
            .unwrap();
        let resolve = match kind {
            Some(kind) => quote!(#tc::projection_in(B[#index], #name, #kind)),
            None => quote!(#tc::projection_in_any(B[#index], #name)),
        };
        quote_spanned!(bound.span()=> {
            const B: &[&#tc::ConstraintDesc] = #bounds;
            const I: usize = #resolve;
            #schema::ProjectionRef { var: #var, bound: #index, index: I }
        })
    } else {
        let resolve = match kind {
            Some(kind) => quote!(#tc::resolve_projection(B, #name, #kind)),
            None => quote!(#tc::resolve_projection_any(B, #name)),
        };
        quote!({
            const B: &[&#tc::ConstraintDesc] = #bounds;
            const P: (usize, usize) = #resolve;
            #schema::ProjectionRef { var: #var, bound: P.0, index: P.1 }
        })
    }
}

fn gen_relation(
    crate_path: &TokenStream,
    relation: &RelationDef,
    op: &OperationDef,
) -> TokenStream {
    let schema = quote!(#crate_path::op_schema);
    let tc = quote!(#crate_path::type_constraint);
    let (kind, target) = match &relation.target {
        RelationExpr::One(expr) => {
            let expr = gen_type_expr(crate_path, expr, op);
            (
                Some(quote!(#tc::ProjectionKind::One)),
                quote!(#schema::RelationTarget::One(#expr)),
            )
        }
        RelationExpr::List(exprs) => {
            let exprs = exprs.iter().map(|expr| gen_type_expr(crate_path, expr, op));
            (
                Some(quote!(#tc::ProjectionKind::List)),
                quote!(#schema::RelationTarget::List(&[#(#exprs),*])),
            )
        }
        RelationExpr::Proj(right) => {
            let left_ref = gen_projection(crate_path, &relation.projection, op, None);
            let right_ref = gen_projection(crate_path, right, op, None);
            let left_bounds = gen_bounds(crate_path, &op.type_vars[relation.projection.var].bounds);
            let right_bounds = gen_bounds(crate_path, &op.type_vars[right.var].bounds);
            (
                None,
                quote!({
                    const L: #schema::ProjectionRef = #left_ref;
                    const R: #schema::ProjectionRef = #right_ref;
                    const _: () = #tc::check_same_kind(
                        #left_bounds, (L.bound, L.index),
                        #right_bounds, (R.bound, R.index),
                    );
                    #schema::RelationTarget::Proj(R)
                }),
            )
        }
    };
    let projection = gen_projection(crate_path, &relation.projection, op, kind);
    quote!(#schema::Relation { projection: #projection, target: #target })
}

fn gen_type_expr(crate_path: &TokenStream, expr: &TypeExpr, op: &OperationDef) -> TokenStream {
    let schema = quote!(#crate_path::op_schema);
    match expr {
        TypeExpr::Any => quote!(#schema::TypeSpec::Any),
        TypeExpr::Var(i) => quote!(#schema::TypeSpec::Var(#i)),
        TypeExpr::Anon(paths) => {
            let bounds = gen_bounds(crate_path, paths);
            quote!(#schema::TypeSpec::Anon(#bounds))
        }
        TypeExpr::Exact(path) => {
            let bounds = gen_bounds(crate_path, std::slice::from_ref(path));
            quote!(#schema::TypeSpec::Anon(#bounds))
        }
        TypeExpr::Proj(proj) => {
            let p = gen_projection(
                crate_path,
                proj,
                op,
                Some(quote!(#crate_path::type_constraint::ProjectionKind::One)),
            );
            quote!(#schema::TypeSpec::Proj(#p))
        }
    }
}

fn gen_value_expr(crate_path: &TokenStream, expr: &ValueExpr, op: &OperationDef) -> TokenStream {
    let schema = quote!(#crate_path::op_schema);
    match expr {
        ValueExpr::Each(ty) => {
            let ty = gen_type_expr(crate_path, ty, op);
            quote!(#schema::ValueConstraint::Each(#ty))
        }
        ValueExpr::List(ListExpr::Types(types)) => {
            let types = types.iter().map(|ty| gen_type_expr(crate_path, ty, op));
            quote!(#schema::ValueConstraint::List(#schema::ListSpec::Types(&[#(#types),*])))
        }
        ValueExpr::List(ListExpr::Proj(proj)) => {
            let p = gen_projection(
                crate_path,
                proj,
                op,
                Some(quote!(#crate_path::type_constraint::ProjectionKind::List)),
            );
            quote!(#schema::ValueConstraint::List(#schema::ListSpec::Proj(#p)))
        }
    }
}

fn gen_impl_block(crate_path: &TokenStream, op: &OperationDef) -> TokenStream {
    let sname = struct_name(&op.name);
    let op_ref_method = quote! {
        pub fn op_ref(&self) -> #crate_path::OpRef {
            self.0
        }
    };
    let operand_accessors = gen_operand_accessors(crate_path, &op.operands);
    let result_accessors = gen_result_accessors(crate_path, &op.results);
    let attr_accessors = gen_attr_accessors(crate_path, &op.attrs);
    let region_accessors = gen_region_accessors(crate_path, &op.regions);

    quote! {
        impl #sname {
            #op_ref_method
            #operand_accessors
            #result_accessors
            #attr_accessors
            #region_accessors
        }
    }
}

// ============================================================================
// Operand accessors
// ============================================================================

fn gen_operand_accessors(crate_path: &TokenStream, operands: &[Operand]) -> TokenStream {
    if operands.is_empty() {
        return quote!();
    }

    // Check if there's only a variadic operand with no fixed operands
    if operands.len() == 1 && operands[0].variadic {
        let name = &operands[0].raw_ident;
        return quote! {
            pub fn #name<'a>(&self, ctx: &'a #crate_path::IrContext) -> &'a [#crate_path::ValueRef] {
                ctx.op_operands(self.0)
            }
        };
    }

    let mut methods = Vec::new();
    let mut fixed_count = 0usize;

    for operand in operands {
        let name = &operand.raw_ident;
        if operand.variadic {
            let idx = fixed_count;
            if idx > 0 {
                methods.push(quote! {
                    pub fn #name<'a>(&self, ctx: &'a #crate_path::IrContext) -> &'a [#crate_path::ValueRef] {
                        &ctx.op_operands(self.0)[#idx..]
                    }
                });
            } else {
                methods.push(quote! {
                    pub fn #name<'a>(&self, ctx: &'a #crate_path::IrContext) -> &'a [#crate_path::ValueRef] {
                        ctx.op_operands(self.0)
                    }
                });
            }
        } else {
            let idx = fixed_count;
            methods.push(quote! {
                pub fn #name(&self, ctx: &#crate_path::IrContext) -> #crate_path::ValueRef {
                    ctx.op_operands(self.0)[#idx]
                }
            });
            fixed_count += 1;
        }
    }

    quote!(#(#methods)*)
}

// ============================================================================
// Result accessors
// ============================================================================

fn gen_result_accessors(crate_path: &TokenStream, results: &ResultDef) -> TokenStream {
    match results {
        ResultDef::None => quote!(),
        ResultDef::Single(name) | ResultDef::Optional(name) => {
            let name_ident = format_ident!("{name}");
            let ty_name = format_ident!("{name}_ty");
            quote! {
                pub fn #name_ident(&self, ctx: &#crate_path::IrContext) -> #crate_path::ValueRef {
                    ctx.op_result(self.0, 0)
                }

                pub fn #ty_name(&self, ctx: &#crate_path::IrContext) -> #crate_path::TypeRef {
                    ctx.op_result_types(self.0)[0]
                }
            }
        }
        ResultDef::Variadic(name) => {
            let name_ident = format_ident!("{name}");
            quote! {
                pub fn #name_ident<'a>(&self, ctx: &'a #crate_path::IrContext) -> &'a [#crate_path::ValueRef] {
                    ctx.op_results(self.0)
                }
            }
        }
    }
}

// ============================================================================
// Attribute accessors
// ============================================================================

fn gen_attr_accessors(crate_path: &TokenStream, attrs: &[AttrDef]) -> TokenStream {
    let methods: Vec<TokenStream> = attrs
        .iter()
        .map(|attr| gen_attr_accessor(crate_path, attr))
        .collect();
    quote!(#(#methods)*)
}

fn gen_attr_accessor(crate_path: &TokenStream, attr: &AttrDef) -> TokenStream {
    gen_map_attr_accessor(crate_path, attr, quote!(ctx.op(self.0).attributes))
}

fn gen_map_attr_accessor(
    crate_path: &TokenStream,
    attr: &AttrDef,
    attrs: TokenStream,
) -> TokenStream {
    let name = &attr.raw_ident;
    let name_str = &attr.name;
    let kind = attr_kind_trait(crate_path, attr);
    let accessor = |method: &proc_macro2::Ident, out: TokenStream, read: TokenStream| {
        if attr.optional {
            quote! {
                pub fn #method<'ctx>(&self, ctx: &'ctx #crate_path::IrContext) -> Option<#out> {
                    #attrs.get(#name_str).map(|attr| #read)
                }
            }
        } else {
            quote! {
                pub fn #method<'ctx>(&self, ctx: &'ctx #crate_path::IrContext) -> #out {
                    let attr = #attrs
                        .get(#name_str)
                        .expect(concat!("missing attribute: ", #name_str));
                    #read
                }
            }
        }
    };
    let value = accessor(
        name,
        quote!(#kind::Out<'ctx>),
        quote!(#kind::read(ctx, attr)),
    );
    if !attr.kind.is_string() {
        return value;
    }

    // A string attribute's text lives in the context's string pool, so the
    // accessor borrows it from `ctx`, as MLIR's `getName()`. `<name>_ref`
    // returns the pooled handle, for copying the value to another operation
    // without borrowing the context.
    let kind_ty = attr_kind_type(crate_path, attr);
    let handle_kind = quote!(<#kind_ty as #crate_path::attr_kind::AttrHandle>);
    let handle = accessor(
        &format_ident!("{}_ref", attr.name),
        quote!(#handle_kind::Handle<'ctx>),
        quote!(#handle_kind::handle(attr)),
    );
    quote!(#value #handle)
}

// ============================================================================
// Region/successor accessors
// ============================================================================

fn gen_region_accessors(crate_path: &TokenStream, regions: &[RegionOrSuccessor]) -> TokenStream {
    let mut region_idx = 0usize;
    let mut succ_idx = 0usize;
    let mut methods = Vec::new();

    for item in regions {
        match item {
            RegionOrSuccessor::Region { name, .. } => {
                let name_ident = format_ident!("{name}");
                let idx = region_idx;
                let missing = format!("missing region `{name}`");
                methods.push(quote! {
                    pub fn #name_ident(&self, ctx: &#crate_path::IrContext) -> #crate_path::RegionRef {
                        ctx.op_region(self.0, #idx)
                            .expect(#missing)
                    }
                });
                region_idx += 1;
            }
            RegionOrSuccessor::Successor {
                name,
                variadic: true,
            } => {
                let name_ident = format_ident!("{name}");
                let idx = succ_idx;
                methods.push(quote! {
                    pub fn #name_ident<'a>(
                        &self,
                        ctx: &'a #crate_path::IrContext,
                    ) -> impl Iterator<Item = #crate_path::BlockRef> + 'a {
                        ctx.op_successors(self.0).skip(#idx)
                    }
                });
            }
            RegionOrSuccessor::Successor { name, .. } => {
                let name_ident = format_ident!("{name}");
                let idx = succ_idx;
                let missing = format!("missing successor `{name}`");
                methods.push(quote! {
                    pub fn #name_ident(&self, ctx: &#crate_path::IrContext) -> #crate_path::BlockRef {
                        ctx.op_successor(self.0, #idx)
                            .expect(#missing)
                    }
                });
                succ_idx += 1;
            }
        }
    }

    quote!(#(#methods)*)
}

// ============================================================================
// Constructor function
// ============================================================================

// ============================================================================
// Fluent builder (typed syntax)
// ============================================================================

/// Generate `Op::operands(..)` and the `OpBuilder` type.
///
/// Inputs are grouped by entity kind: operands start the builder, result
/// types, regions, and successors are each one call, and attributes are set
/// by name. Missing required inputs panic in `build`.
fn gen_fluent_builder(crate_path: &TokenStream, dialect: &str, op: &OperationDef) -> TokenStream {
    let sname = struct_name(&op.name);
    let bname = format_ident!("{}Builder", sname);
    let op_name = &op.name;
    let full_name = format!("{dialect}.{op_name}");
    let value_ref = quote!(#crate_path::ValueRef);
    let type_ref = quote!(#crate_path::TypeRef);

    let mut fields = vec![quote!(operands: ::std::vec::Vec<#value_ref>)];
    let mut field_inits = Vec::new();
    let mut methods = Vec::new();
    let mut build_stmts = vec![quote!(__builder = __builder.operands(self.operands);)];

    // Entry point: all operands in declaration order.
    let mut entry_params = Vec::new();
    let mut entry_stmts = Vec::new();
    for operand in &op.operands {
        let name = &operand.raw_ident;
        if operand.variadic {
            entry_params.push(quote!(#name: impl IntoIterator<Item = #value_ref>));
            entry_stmts.push(quote!(__operands.extend(#name);));
        } else {
            entry_params.push(quote!(#name: #value_ref));
            entry_stmts.push(quote!(__operands.push(#name);));
        }
    }

    // Result types: inferred when uniquely determined, otherwise one call.
    let mut pre_stmts = Vec::new();
    let checks = fixed_result_checks(crate_path, op);
    let results_param = match &op.results {
        ResultDef::None => None,
        _ if results_inferable(op) => {
            let attrs = op.attrs.iter().map(|attr| {
                let name = &attr.name;
                let field = format_ident!("attr_{}", attr.name);
                quote!((#name, #field.as_ref()))
            });
            pre_stmts.push(quote! {
                let __results = <#sname as #crate_path::ops::DialectOp>::DEF
                    .schema
                    .infer_result_types(ctx, &self.operands, &[#(#attrs),*]);
            });
            build_stmts.push(quote!(__builder = __builder.results(__results);));
            None
        }
        ResultDef::Single(_) => Some((quote!(result: #type_ref), quote!(::std::vec![result]))),
        ResultDef::Optional(_) => Some((
            quote!(result: impl Into<Option<#type_ref>>),
            quote!(result.into().into_iter().collect()),
        )),
        ResultDef::Variadic(_) => Some((
            quote!(results: impl IntoIterator<Item = #type_ref>),
            quote!(results.into_iter().collect()),
        )),
    };
    if let Some((param, value)) = results_param {
        fields.push(quote!(results: Option<::std::vec::Vec<#type_ref>>));
        field_inits.push(quote!(results: None,));
        methods.push(quote! {
            /// Set the result types.
            pub fn results(mut self, #param) -> Self {
                self.results = Some(#value);
                self
            }
        });
        let missing = format!("{full_name}: missing result types");
        build_stmts.push(quote! {
            __builder = __builder.results(self.results.expect(#missing));
        });
    }

    // Attributes, one setter each. `build` first turns every set attribute
    // into an `Attribute` local; a string attribute is interned there.
    let mut attr_locals = Vec::new();
    for attr in &op.attrs {
        let name = &attr.raw_ident;
        let name_str = &attr.name;
        let field = format_ident!("attr_{}", attr.name);
        let kind = attr_kind_trait(crate_path, attr);
        let element = attr_element_kind(crate_path, attr);
        let element = quote!(<#element as #crate_path::attr_kind::AttrKind>);
        fields.push(quote!(#field: Option<#kind::In>));
        field_inits.push(quote!(#field: None,));
        attr_locals.push(quote! {
            let #field = self.#field.map(|value| #kind::write(ctx, value));
        });
        // A list is set from its elements. A string takes a handle or text
        // to intern. An optional attribute is absent unless set.
        methods.push(match (attr.list, attr.kind.is_string()) {
            (true, true) => quote! {
                pub fn #name(
                    mut self,
                    values: impl IntoIterator<Item = impl Into<#element::In>>,
                ) -> Self {
                    self.#field = Some(values.into_iter().map(Into::into).collect());
                    self
                }
            },
            (true, false) => quote! {
                pub fn #name(mut self, values: impl IntoIterator<Item = #element::In>) -> Self {
                    self.#field = Some(values.into_iter().collect());
                    self
                }
            },
            (false, true) => quote! {
                pub fn #name(mut self, value: impl Into<#kind::In>) -> Self {
                    self.#field = Some(value.into());
                    self
                }
            },
            (false, false) if attr.optional => quote! {
                pub fn #name(mut self, value: impl Into<Option<#kind::In>>) -> Self {
                    self.#field = value.into();
                    self
                }
            },
            (false, false) => quote! {
                pub fn #name(mut self, value: #kind::In) -> Self {
                    self.#field = Some(value);
                    self
                }
            },
        });
        if attr.optional {
            build_stmts.push(quote! {
                if let Some(value) = #field {
                    __builder = __builder.attr(#crate_path::Symbol::new(#name_str), value);
                }
            });
        } else {
            let missing = format!("{full_name}: missing attribute `{name_str}`");
            build_stmts.push(quote! {
                __builder = __builder.attr(
                    #crate_path::Symbol::new(#name_str),
                    #field.expect(#missing),
                );
            });
        }
    }

    // Regions and successors, one call per kind.
    let regions: Vec<_> = op
        .regions
        .iter()
        .filter_map(|item| match item {
            RegionOrSuccessor::Region { name, optional } => {
                Some((format_ident!("{name}"), *optional))
            }
            RegionOrSuccessor::Successor { .. } => None,
        })
        .collect();
    if !regions.is_empty() {
        let params = regions.iter().map(|(name, optional)| {
            if *optional {
                quote!(#name: impl Into<Option<#crate_path::RegionRef>>)
            } else {
                quote!(#name: #crate_path::RegionRef)
            }
        });
        let pushes = regions.iter().map(|(name, optional)| {
            if *optional {
                quote!(__regions.extend(#name.into());)
            } else {
                quote!(__regions.push(#name);)
            }
        });
        fields.push(quote!(regions: Option<::std::vec::Vec<#crate_path::RegionRef>>));
        field_inits.push(quote!(regions: None,));
        methods.push(quote! {
            /// Set the regions in declaration order.
            pub fn regions(mut self, #(#params),*) -> Self {
                let mut __regions = ::std::vec::Vec::new();
                #(#pushes)*
                self.regions = Some(__regions);
                self
            }
        });
        let missing = format!("{full_name}: missing regions");
        build_stmts.push(quote! {
            for __region in self.regions.expect(#missing) {
                __builder = __builder.region(__region);
            }
        });
    }

    let successors: Vec<_> = op
        .regions
        .iter()
        .filter_map(|item| match item {
            RegionOrSuccessor::Successor { name, variadic } => {
                Some((format_ident!("{name}"), *variadic))
            }
            RegionOrSuccessor::Region { .. } => None,
        })
        .collect();
    if !successors.is_empty() {
        let params = successors.iter().map(|(name, variadic)| {
            if *variadic {
                quote!(#name: impl ::std::iter::IntoIterator<Item = #crate_path::BlockRef>)
            } else {
                quote!(#name: #crate_path::BlockRef)
            }
        });
        let pushes = successors.iter().map(|(name, variadic)| {
            if *variadic {
                quote!(__successors.extend(#name);)
            } else {
                quote!(__successors.push(#name);)
            }
        });
        fields.push(quote!(successors: Option<::std::vec::Vec<#crate_path::BlockRef>>));
        field_inits.push(quote!(successors: None,));
        methods.push(quote! {
            /// Set the successor blocks in declaration order.
            pub fn successors(mut self, #(#params),*) -> Self {
                let mut __successors = ::std::vec::Vec::new();
                #(#pushes)*
                self.successors = Some(__successors);
                self
            }
        });
        let missing = format!("{full_name}: missing successors");
        build_stmts.push(quote! {
            for __successor in self.successors.expect(#missing) {
                __builder = __builder.successor(__successor);
            }
        });
    }

    // Operations without operands start from an empty `operands()` so every
    // builder has the same entry point.
    let operands_init = if op.operands.is_empty() {
        quote!(::std::vec::Vec::new())
    } else {
        quote!({
            let mut __operands = ::std::vec::Vec::new();
            #(#entry_stmts)*
            __operands
        })
    };
    let entry = quote! {
        /// Start building this operation from its operands in declaration
        /// order.
        pub fn operands(#(#entry_params),*) -> #bname {
            #bname { operands: #operands_init, #(#field_inits)* }
        }
    };
    let builder_doc = format!("Builder for `{full_name}`.");

    quote! {
        #(#checks)*

        impl #sname {
            #entry
        }

        #[doc = #builder_doc]
        #[must_use]
        pub struct #bname {
            #(#fields),*
        }

        impl #bname {
            #(#methods)*

            /// Create the operation.
            pub fn build(
                self,
                ctx: &mut #crate_path::IrContext,
                location: #crate_path::Location,
            ) -> #sname {
                #(#attr_locals)*
                #(#pre_stmts)*
                let mut __builder = #crate_path::OperationDataBuilder::new(
                    location,
                    #crate_path::Symbol::new(#dialect),
                    #crate_path::Symbol::new(#op_name),
                );
                #(#build_stmts)*
                let __data = __builder.build(ctx);
                #sname(ctx.create_op(__data))
            }
        }
    }
}

/// Whether a builder can infer every result type: each is a fixed type, a
/// variable bound by a single operand, a required attribute, or a bound
/// constraint on such a variable, or a projection of such a variable. Must
/// agree with `OpSchema::infer_result_types`.
fn results_inferable(op: &OperationDef) -> bool {
    let mut bound: Vec<bool> = (0..op.type_vars.len())
        .map(|var| {
            op.attrs
                .iter()
                .any(|attr| attr.binds == Some(var) && !attr.optional)
                || op.operands.iter().any(|operand| {
                    !operand.variadic
                        && matches!(operand.constraint, ValueExpr::Each(TypeExpr::Var(v)) if v == var)
                })
        })
        .collect();
    loop {
        let mut progressed = false;
        for relation in &op.relations {
            if !bound[relation.projection.var] {
                continue;
            }
            for var in relation.bound_vars() {
                progressed |= !std::mem::replace(&mut bound[var], true);
            }
        }
        if !progressed {
            break;
        }
    }
    let one = |expr: &TypeExpr| match expr {
        TypeExpr::Var(var) => bound[*var],
        TypeExpr::Proj(proj) => bound[proj.var],
        TypeExpr::Exact(_) => true,
        TypeExpr::Any | TypeExpr::Anon(_) => false,
    };
    match (&op.results, &op.result_constraint) {
        (ResultDef::Single(_), ValueExpr::Each(expr)) => one(expr),
        (ResultDef::Variadic(_), ValueExpr::List(ListExpr::Types(exprs))) => exprs.iter().all(one),
        (ResultDef::Variadic(_), ValueExpr::List(ListExpr::Proj(proj))) => bound[proj.var],
        _ => false,
    }
}

/// Compile-time checks that directly named result bounds denote one type,
/// whether or not the builder infers them.
fn fixed_result_checks(crate_path: &TokenStream, op: &OperationDef) -> Vec<TokenStream> {
    let exprs: Vec<&TypeExpr> = match &op.result_constraint {
        ValueExpr::Each(expr) => vec![expr],
        ValueExpr::List(ListExpr::Types(exprs)) => exprs.iter().collect(),
        ValueExpr::List(ListExpr::Proj(_)) => Vec::new(),
    };
    exprs
        .into_iter()
        .filter_map(|expr| match expr {
            TypeExpr::Exact(path) => Some(path),
            _ => None,
        })
        .map(|path| {
            quote_spanned! {path.span()=>
                const _: () = assert!(
                    <#path as #crate_path::type_constraint::TypeConstraint>::DESC.fixed.is_some(),
                    concat!(
                        "result bound `", stringify!(#path), "` does not denote one type; ",
                        "declare the result as `impl ", stringify!(#path),
                        "` and pass it with `.results(..)`"
                    ),
                );
            }
        })
        .collect()
}

// ============================================================================
// Type definition codegen
// ============================================================================

fn gen_type_def(crate_path: &TokenStream, dialect: &str, td: &TypeDefData) -> TokenStream {
    let ir_type_name = td.name.to_snake_case();
    let type_name_fn = gen_type_name_fn(crate_path, &ir_type_name);
    let struct_and_trait = gen_type_struct_and_trait(crate_path, dialect, &ir_type_name, td);
    let impl_block = gen_type_impl_block(crate_path, td);
    let constructor = gen_type_constructor(crate_path, dialect, &ir_type_name, td);
    let constraint = gen_type_constraint(crate_path, dialect, &ir_type_name, td);

    quote! {
        #type_name_fn
        #struct_and_trait
        #impl_block
        #constructor
        #constraint
    }
}

fn gen_type_constraint(
    crate_path: &TokenStream,
    dialect: &str,
    type_name: &str,
    td: &TypeDefData,
) -> TokenStream {
    let sname = struct_name(&td.name);
    let full_name = format!("{dialect}.{type_name}");
    let fixed = td.params.iter().filter(|p| !p.variadic).count();
    // Declared attributes are part of the wrapper invariant: required ones
    // must be present, and every present one must match its kind.
    let attrs_ok = td.attrs.iter().map(|attr| {
        let name = &attr.name;
        let kind = attr_kind_trait(crate_path, attr);
        let kind = quote!(#kind::KIND);
        if attr.optional {
            quote!(attrs.get(#name).is_none_or(|attr| #kind.accepts(attr)))
        } else {
            quote!(attrs.get(#name).is_some_and(|attr| #kind.accepts(attr)))
        }
    });
    let count_ok = if td.params.iter().any(|p| p.variadic) {
        quote!(params.len() >= #fixed)
    } else {
        quote!(params.len() == #fixed)
    };
    let projections = td.params.iter().map(|param| {
        let name = &param.name;
        let kind = if param.variadic {
            quote!(List)
        } else {
            quote!(One)
        };
        quote!(#crate_path::type_constraint::ProjectionDesc {
            name: #name, kind: #crate_path::type_constraint::ProjectionKind::#kind,
        })
    });
    let arms = td.params.iter().enumerate().map(|(i, param)| {
        if param.variadic {
            quote!(#i => Some(#crate_path::type_constraint::Projected::List(&params[#fixed..])),)
        } else {
            quote!(#i => Some(#crate_path::type_constraint::Projected::One(params[#i])),)
        }
    });
    // A type without parameters or attributes has exactly one instance.
    let fixed_type = if td.params.is_empty() && td.attrs.is_empty() {
        quote!(Some(|ctx| {
            ctx.intern_type(#crate_path::TypeDataBuilder::new(#dialect, #type_name).build())
        }))
    } else {
        quote!(None)
    };
    quote! {
        impl #crate_path::type_constraint::TypeConstraint for #sname {
            const DESC: &'static #crate_path::type_constraint::ConstraintDesc =
                &#crate_path::type_constraint::ConstraintDesc {
                    name: #full_name,
                    exact: true,
                    projections: &[#(#projections),*],
                    matches: |ctx, ty| {
                        <Self as #crate_path::ops::DialectType>::matches(ctx, ty)
                            && {
                                let data = ctx.get_type(ty);
                                let params = &data.params;
                                #[allow(unused_variables)]
                                let attrs = &data.attrs;
                                #count_ok #(&& #attrs_ok)*
                            }
                    },
                    project: |ctx, ty, index| {
                        if !(<Self as #crate_path::type_constraint::TypeConstraint>::DESC.matches)(ctx, ty) {
                            return None;
                        }
                        let params = &ctx.get_type(ty).params;
                        match index { #(#arms)* _ => None }
                    },
                    fixed: #fixed_type,
                };
        }
    }
}

fn gen_type_name_fn(crate_path: &TokenStream, type_name: &str) -> TokenStream {
    let upper_name = format_ident!("{}", type_name.to_shouty_snake_case());
    quote! {
        #[allow(non_snake_case)]
        #[inline]
        pub fn #upper_name() -> #crate_path::Symbol {
            #crate_path::Symbol::new(#type_name)
        }
    }
}

fn gen_type_struct_and_trait(
    crate_path: &TokenStream,
    dialect: &str,
    type_name: &str,
    td: &TypeDefData,
) -> TokenStream {
    let sname = struct_name(&td.name);

    quote! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct #sname(#crate_path::TypeRef);

        impl #crate_path::ops::DialectType for #sname {
            const DIALECT_NAME: &'static str = #dialect;
            const TYPE_NAME: &'static str = #type_name;

            fn from_type_ref(
                ctx: &#crate_path::IrContext,
                ty: #crate_path::TypeRef,
            ) -> Option<Self> {
                if !Self::matches(ctx, ty) {
                    return None;
                }
                Some(Self(ty))
            }

            fn as_type_ref(&self) -> #crate_path::TypeRef {
                self.0
            }
        }

        impl From<#sname> for #crate_path::TypeRef {
            fn from(t: #sname) -> Self {
                t.0
            }
        }
    }
}

fn gen_type_impl_block(crate_path: &TokenStream, td: &TypeDefData) -> TokenStream {
    let sname = struct_name(&td.name);

    let as_type_ref_method = quote! {
        pub fn as_type_ref(&self) -> #crate_path::TypeRef {
            self.0
        }
    };

    // Param accessors: each param is accessed by index in ctx.get_type(self.0).params
    let param_accessors = gen_type_param_accessors(crate_path, &td.params);

    // Attr accessors: same pattern as op attrs, but via ctx.get_type(self.0).attrs
    let attr_accessors: Vec<TokenStream> = td
        .attrs
        .iter()
        .map(|attr| gen_type_attr_accessor(crate_path, attr))
        .collect();

    quote! {
        impl #sname {
            #as_type_ref_method
            #param_accessors
            #(#attr_accessors)*
        }
    }
}

fn gen_type_param_accessors(
    crate_path: &TokenStream,
    params: &[crate::parse::TypeParam],
) -> TokenStream {
    if params.is_empty() {
        return quote!();
    }

    // Check if there's only a variadic param with no fixed params
    if params.len() == 1 && params[0].variadic {
        let name = format_ident!("r#{}", params[0].raw_ident.to_string().to_snake_case());
        return quote! {
            pub fn #name<'a>(&self, ctx: &'a #crate_path::IrContext) -> &'a [#crate_path::TypeRef] {
                &ctx.get_type(self.0).params
            }
        };
    }

    let mut methods = Vec::new();
    let mut fixed_count = 0usize;

    for param in params {
        let name = format_ident!("r#{}", param.raw_ident.to_string().to_snake_case());
        if param.variadic {
            let idx = fixed_count;
            if idx > 0 {
                methods.push(quote! {
                    pub fn #name<'a>(&self, ctx: &'a #crate_path::IrContext) -> &'a [#crate_path::TypeRef] {
                        &ctx.get_type(self.0).params[#idx..]
                    }
                });
            } else {
                methods.push(quote! {
                    pub fn #name<'a>(&self, ctx: &'a #crate_path::IrContext) -> &'a [#crate_path::TypeRef] {
                        &ctx.get_type(self.0).params
                    }
                });
            }
        } else {
            let idx = fixed_count;
            methods.push(quote! {
                pub fn #name(&self, ctx: &#crate_path::IrContext) -> #crate_path::TypeRef {
                    ctx.get_type(self.0).params[#idx]
                }
            });
            fixed_count += 1;
        }
    }

    quote!(#(#methods)*)
}

fn gen_type_attr_accessor(crate_path: &TokenStream, attr: &AttrDef) -> TokenStream {
    gen_map_attr_accessor(crate_path, attr, quote!(ctx.get_type(self.0).attrs))
}

fn gen_type_constructor(
    crate_path: &TokenStream,
    dialect: &str,
    type_name: &str,
    td: &TypeDefData,
) -> TokenStream {
    let sname = struct_name(&td.name);

    // Build parameter list
    let mut params = Vec::new();
    let mut body_stmts = Vec::new();

    // Type params become TypeRef parameters
    for param in &td.params {
        let name = format_ident!("r#{}", param.raw_ident.to_string().to_snake_case());
        if param.variadic {
            params.push(quote!(#name: impl IntoIterator<Item = #crate_path::TypeRef>));
            body_stmts.push(quote!(__builder = __builder.params(#name);));
        } else {
            params.push(quote!(#name: #crate_path::TypeRef));
            body_stmts.push(quote!(__builder = __builder.param(#name);));
        }
    }

    // Attrs
    for attr in &td.attrs {
        let name = &attr.raw_ident;
        let name_str = &attr.name;
        let kind = attr_kind_trait(crate_path, attr);
        let rust_ty = quote!(#kind::In);
        let to_attr_expr = |val: TokenStream| quote!(#kind::write(ctx, #val));

        if attr.optional {
            params.push(quote!(#name: Option<#rust_ty>));
            let attr_conv = to_attr_expr(quote!(__attr_val));
            body_stmts.push(quote! {
                if let ::core::option::Option::Some(__attr_val) = #name {
                    __builder = __builder.attr(
                        #crate_path::Symbol::new(#name_str),
                        #attr_conv,
                    );
                }
            });
        } else {
            params.push(quote!(#name: #rust_ty));
            let attr_conv = to_attr_expr(quote!(#name));
            body_stmts.push(quote! {
                __builder = __builder.attr(
                    #crate_path::Symbol::new(#name_str),
                    #attr_conv,
                );
            });
        }
    }

    // Use snake_case for the constructor function name (e.g., "Nil" -> "nil", "Array" -> "array")
    // Use raw ident to handle Rust keywords (e.g., "Ref" -> "r#ref")
    let fn_name_snake = format_ident!("r#{}", td.name.to_snake_case());

    quote! {
        #[allow(clippy::too_many_arguments)]
        pub fn #fn_name_snake(
            ctx: &mut #crate_path::IrContext,
            #(#params),*
        ) -> #sname {
            #[allow(unused_mut)]
            let mut __builder = #crate_path::TypeDataBuilder::new(
                #dialect,
                #type_name,
            );
            #(#body_stmts)*
            let __data = __builder.build();
            let __type_ref = ctx.intern_type(__data);
            #sname(__type_ref)
        }
    }
}

// ============================================================================
// Attribute type helpers
// ============================================================================

/// The Rust type naming a declared attribute's kind, or its element kind
/// for a list.
fn attr_element_kind(crate_path: &TokenStream, attr: &AttrDef) -> TokenStream {
    match &attr.kind {
        AttrKind::Any => quote!(#crate_path::Attribute),
        AttrKind::BoundType => quote!(#crate_path::attr_kind::Type),
        AttrKind::Path(path) => quote!(#path),
    }
}

/// The Rust type naming a declared attribute's kind.
fn attr_kind_type(crate_path: &TokenStream, attr: &AttrDef) -> TokenStream {
    let element = attr_element_kind(crate_path, attr);
    if attr.list {
        quote!([#element])
    } else {
        element
    }
}

/// A declared attribute's kind as its `AttrKind` implementation.
fn attr_kind_trait(crate_path: &TokenStream, attr: &AttrDef) -> TokenStream {
    let kind = attr_kind_type(crate_path, attr);
    // The span points a type that is not a kind at its declaration.
    let span = match &attr.kind {
        AttrKind::Path(path) => path.span(),
        AttrKind::Any | AttrKind::BoundType => proc_macro2::Span::call_site(),
    };
    quote_spanned!(span=> <#kind as #crate_path::attr_kind::AttrKind>)
}
