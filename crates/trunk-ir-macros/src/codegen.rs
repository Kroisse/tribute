//! Code generation for `#[dialect]`.

use heck::{ToShoutySnakeCase, ToSnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};

use crate::parse::{
    AttrDef, AttrType, BoundPath, DialectItem, DialectModule, ListExpr, Operand, OperationDef,
    Projection, RegionOrSuccessor, ResultDef, TypeDefData, TypeExpr, ValueExpr,
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
    let schema = gen_op_schema(crate_path, dialect, op);

    quote! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct #sname(#crate_path::OpRef);

        impl #crate_path::ops::DialectOp for #sname {
            const DIALECT_NAME: &'static str = #dialect;
            const OP_NAME: &'static str = #op_name;
            const SCHEMA: &'static #crate_path::op_schema::OpSchema = &#schema;

            fn from_op(
                ctx: &#crate_path::IrContext,
                op: #crate_path::OpRef,
            ) -> Result<Self, #crate_path::ops::ConversionError> {
                if !Self::matches(ctx, op) {
                    return Err(#crate_path::ops::ConversionError::WrongOperation {
                        expected: #full_name,
                        actual: format!("{}.{}",
                            ctx.op(op).dialect,
                            ctx.op(op).name),
                    });
                }
                Ok(Self(op))
            }

            fn op_ref(&self) -> #crate_path::OpRef {
                self.0
            }
        }

        #crate_path::inventory::submit! {
            #crate_path::op_schema::OpSchemaRegistration(
                <#sname as #crate_path::ops::DialectOp>::SCHEMA
            )
        }
    }
}

/// Build the static `OpSchema` expression for an operation.
fn gen_op_schema(crate_path: &TokenStream, dialect: &str, op: &OperationDef) -> TokenStream {
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
        let kind = attr_kind(crate_path, attr.ty);
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
        RegionOrSuccessor::Successor(_) => None,
    });
    let successors = op.regions.iter().filter_map(|item| match item {
        RegionOrSuccessor::Successor(name) => Some(name),
        RegionOrSuccessor::Region { .. } => None,
    });

    // The span points a missing `verify` method at the `#[verify]` attribute.
    let verifier = match op.verify {
        Some(span) => {
            let mut sname = struct_name(&op.name);
            sname.set_span(span);
            quote_spanned!(span=> Some(|ctx, op| {
                use #crate_path::op_schema::__private::VerifyFallback as _;
                #sname::verify(#sname(op), ctx)
            }))
        }
        None => quote!(None),
    };

    quote! {
        #schema_mod::OpSchema {
            dialect: #dialect,
            name: #op_name,
            type_vars: &[#(#type_vars),*],
            operands: &[#(#operands),*],
            results: #results,
            result_constraint: #result_constraint,
            attributes: &[#(#attributes),*],
            regions: &[#(#regions),*],
            successors: &[#(#successors),*],
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

fn gen_projection(
    crate_path: &TokenStream,
    projection: &Projection,
    op: &OperationDef,
    kind: TokenStream,
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
        quote_spanned!(bound.span()=> {
            const B: &[&#tc::ConstraintDesc] = #bounds;
            const I: usize = #tc::projection_in(B[#index], #name, #kind);
            #schema::ProjectionRef { var: #var, bound: #index, index: I }
        })
    } else {
        quote!({
            const B: &[&#tc::ConstraintDesc] = #bounds;
            const P: (usize, usize) = #tc::resolve_projection(B, #name, #kind);
            #schema::ProjectionRef { var: #var, bound: P.0, index: P.1 }
        })
    }
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
                quote!(#crate_path::type_constraint::ProjectionKind::One),
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
                quote!(#crate_path::type_constraint::ProjectionKind::List),
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
    let rust_ty = attr_rust_type(crate_path, attr.ty);

    if let Some(lookup) = typed_attr_lookup(attr.ty, &attrs, name_str) {
        return if attr.optional {
            quote! {
                pub fn #name(&self, ctx: &#crate_path::IrContext) -> Option<#rust_ty> {
                    #lookup
                }
            }
        } else {
            quote! {
                pub fn #name(&self, ctx: &#crate_path::IrContext) -> #rust_ty {
                    #lookup.expect(concat!("missing attribute: ", #name_str))
                }
            }
        };
    }

    let from_attr = attr_from_attr(crate_path, attr.ty);
    if attr.optional {
        quote! {
            pub fn #name(&self, ctx: &#crate_path::IrContext) -> Option<#rust_ty> {
                #attrs
                    .get(#name_str)
                    .map(|attr| #from_attr)
            }
        }
    } else {
        quote! {
            pub fn #name(&self, ctx: &#crate_path::IrContext) -> #rust_ty {
                let attr = #attrs
                    .get(#name_str)
                    .expect(concat!("missing attribute: ", #name_str));
                #from_attr
            }
        }
    }
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
                methods.push(quote! {
                    pub fn #name_ident(&self, ctx: &#crate_path::IrContext) -> #crate_path::RegionRef {
                        ctx.op(self.0).regions[#idx]
                    }
                });
                region_idx += 1;
            }
            RegionOrSuccessor::Successor(name) => {
                let name_ident = format_ident!("{name}");
                let idx = succ_idx;
                methods.push(quote! {
                    pub fn #name_ident(&self, ctx: &#crate_path::IrContext) -> #crate_path::BlockRef {
                        ctx.op(self.0).successors[#idx]
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
                quote!((#name, self.#field.as_ref()))
            });
            pre_stmts.push(quote! {
                let __results = <#sname as #crate_path::ops::DialectOp>::SCHEMA
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

    // Attributes, one setter each.
    for attr in &op.attrs {
        let name = &attr.raw_ident;
        let name_str = &attr.name;
        let field = format_ident!("attr_{}", attr.name);
        let rust_ty = attr_rust_type(crate_path, attr.ty);
        let conv = attr_to_attr(crate_path, attr.ty, quote!(value));
        fields.push(quote!(#field: Option<#crate_path::Attribute>));
        field_inits.push(quote!(#field: None,));
        if attr.optional {
            methods.push(quote! {
                pub fn #name(mut self, value: impl Into<Option<#rust_ty>>) -> Self {
                    self.#field = value.into().map(|value| #conv);
                    self
                }
            });
            build_stmts.push(quote! {
                if let Some(value) = self.#field {
                    __builder = __builder.attr(#crate_path::Symbol::new(#name_str), value);
                }
            });
        } else {
            methods.push(quote! {
                pub fn #name(mut self, value: #rust_ty) -> Self {
                    self.#field = Some(#conv);
                    self
                }
            });
            let missing = format!("{full_name}: missing attribute `{name_str}`");
            build_stmts.push(quote! {
                __builder = __builder.attr(
                    #crate_path::Symbol::new(#name_str),
                    self.#field.expect(#missing),
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
            RegionOrSuccessor::Successor(_) => None,
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
            RegionOrSuccessor::Successor(name) => Some(format_ident!("{name}")),
            RegionOrSuccessor::Region { .. } => None,
        })
        .collect();
    if !successors.is_empty() {
        fields.push(quote!(successors: Option<::std::vec::Vec<#crate_path::BlockRef>>));
        field_inits.push(quote!(successors: None,));
        methods.push(quote! {
            /// Set the successor blocks in declaration order.
            pub fn successors(mut self, #(#successors: #crate_path::BlockRef),*) -> Self {
                self.successors = Some(::std::vec![#(#successors),*]);
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
/// variable bound by a single operand or a required attribute, or a
/// projection of such a variable. Must agree with
/// `OpSchema::infer_result_types`.
fn results_inferable(op: &OperationDef) -> bool {
    let var_bound = |var: usize| {
        op.attrs
            .iter()
            .any(|attr| attr.binds == Some(var) && !attr.optional)
            || op.operands.iter().any(|operand| {
                !operand.variadic
                    && matches!(operand.constraint, ValueExpr::Each(TypeExpr::Var(v)) if v == var)
            })
    };
    let one = |expr: &TypeExpr| match expr {
        TypeExpr::Var(var) => var_bound(*var),
        TypeExpr::Proj(proj) => var_bound(proj.var),
        TypeExpr::Exact(_) => true,
        TypeExpr::Any | TypeExpr::Anon(_) => false,
    };
    match (&op.results, &op.result_constraint) {
        (ResultDef::Single(_), ValueExpr::Each(expr)) => one(expr),
        (ResultDef::Variadic(_), ValueExpr::List(ListExpr::Types(exprs))) => exprs.iter().all(one),
        (ResultDef::Variadic(_), ValueExpr::List(ListExpr::Proj(proj))) => var_bound(proj.var),
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
        let kind = attr_kind(crate_path, attr.ty);
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
        let rust_ty = attr_rust_type(crate_path, attr.ty);
        let to_attr_expr = |val: TokenStream| attr_to_attr(crate_path, attr.ty, val);

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

fn attr_rust_type(crate_path: &TokenStream, ty: AttrType) -> TokenStream {
    match ty {
        AttrType::Any => quote!(#crate_path::Attribute),
        AttrType::Bool => quote!(bool),
        AttrType::I32 => quote!(i32),
        AttrType::I64 => quote!(i64),
        AttrType::U32 => quote!(u32),
        AttrType::U64 => quote!(u64),
        AttrType::F32 => quote!(f32),
        AttrType::F64 => quote!(f64),
        AttrType::Type => quote!(#crate_path::TypeRef),
        AttrType::String => quote!(::std::string::String),
        AttrType::Symbol | AttrType::QualifiedName => quote!(#crate_path::Symbol),
        AttrType::Bytes => quote!(#crate_path::smallvec::SmallVec<[u8; 16]>),
    }
}

fn attr_kind(crate_path: &TokenStream, ty: AttrType) -> TokenStream {
    let kind = match ty {
        AttrType::Any => quote!(Any),
        AttrType::Bool => quote!(Bool),
        AttrType::I32 => quote!(I32),
        AttrType::I64 => quote!(I64),
        AttrType::U32 => quote!(U32),
        AttrType::U64 => quote!(U64),
        AttrType::F32 => quote!(F32),
        AttrType::F64 => quote!(F64),
        AttrType::Type => quote!(Type),
        AttrType::String => quote!(String),
        AttrType::Symbol => quote!(Symbol),
        AttrType::QualifiedName => quote!(QualifiedName),
        AttrType::Bytes => quote!(Bytes),
    };
    quote!(#crate_path::op_schema::AttributeKind::#kind)
}

fn attr_to_attr(crate_path: &TokenStream, ty: AttrType, val: TokenStream) -> TokenStream {
    match ty {
        AttrType::Any => quote!(#val),
        AttrType::Bool => quote!(#crate_path::Attribute::Bool(#val)),
        AttrType::I32 | AttrType::I64 | AttrType::U32 => {
            quote!(#crate_path::Attribute::Int(#val as i128))
        }
        AttrType::U64 => quote!(#crate_path::Attribute::Int(#val as i128)),
        AttrType::F32 => {
            quote!(#crate_path::Attribute::FloatBits((#val as f64).to_bits()))
        }
        AttrType::F64 => quote!(#crate_path::Attribute::FloatBits(#val.to_bits())),
        AttrType::Type => quote!(#crate_path::Attribute::Type(#val)),
        AttrType::String => quote!(#crate_path::Attribute::String(#val)),
        AttrType::Symbol | AttrType::QualifiedName => {
            quote!(#crate_path::Attribute::Symbol(#val))
        }
        AttrType::Bytes => quote!(#crate_path::Attribute::Bytes(#val)),
    }
}

fn typed_attr_lookup(ty: AttrType, attrs: &TokenStream, name: &str) -> Option<TokenStream> {
    match ty {
        AttrType::Bool => Some(quote!(#attrs.get_bool(#name))),
        AttrType::I32 => Some(quote!(
            #attrs
                .get_i32(#name)
                .expect(concat!("attribute out of range: ", #name))
        )),
        AttrType::I64 => Some(quote!(
            #attrs
                .get_i64(#name)
                .expect(concat!("attribute out of range: ", #name))
        )),
        AttrType::U32 => Some(quote!(
            #attrs
                .get_u32(#name)
                .expect(concat!("attribute out of range: ", #name))
        )),
        AttrType::U64 => Some(quote!(
            #attrs
                .get_u64(#name)
                .expect(concat!("attribute out of range: ", #name))
        )),
        AttrType::Type => Some(quote!(#attrs.get_type(#name))),
        AttrType::String => Some(quote!(
            #attrs.get_str(#name).map(::std::borrow::ToOwned::to_owned)
        )),
        AttrType::Symbol | AttrType::QualifiedName => Some(quote!(#attrs.get_symbol(#name))),
        AttrType::Any | AttrType::F32 | AttrType::F64 | AttrType::Bytes => None,
    }
}

fn attr_from_attr(crate_path: &TokenStream, ty: AttrType) -> TokenStream {
    match ty {
        AttrType::Any => quote!(attr.clone()),
        AttrType::Bool => quote! {
            match attr {
                #crate_path::Attribute::Bool(v) => *v,
                _ => panic!("expected Bool attribute"),
            }
        },
        AttrType::I32 => quote! {
            match attr {
                #crate_path::Attribute::Int(v) => i32::try_from(*v)
                    .expect("Int attribute is out of range for i32"),
                _ => panic!("expected Int attribute"),
            }
        },
        AttrType::I64 => quote! {
            match attr {
                #crate_path::Attribute::Int(v) => i64::try_from(*v)
                    .expect("Int attribute is out of range for i64"),
                _ => panic!("expected Int attribute"),
            }
        },
        AttrType::U32 => quote! {
            match attr {
                #crate_path::Attribute::Int(v) => u32::try_from(*v)
                    .expect("Int attribute is out of range for u32"),
                _ => panic!("expected Int attribute"),
            }
        },
        AttrType::U64 => quote! {
            match attr {
                #crate_path::Attribute::Int(v) => u64::try_from(*v)
                    .expect("Int attribute is out of range for u64"),
                _ => panic!("expected Int attribute"),
            }
        },
        AttrType::F32 => quote! {
            match attr {
                #crate_path::Attribute::FloatBits(v) => f64::from_bits(*v) as f32,
                _ => panic!("expected FloatBits attribute"),
            }
        },
        AttrType::F64 => quote! {
            match attr {
                #crate_path::Attribute::FloatBits(v) => f64::from_bits(*v),
                _ => panic!("expected FloatBits attribute"),
            }
        },
        AttrType::Type => quote! {
            match attr {
                #crate_path::Attribute::Type(v) => *v,
                _ => panic!("expected Type attribute"),
            }
        },
        AttrType::String => quote! {
            match attr {
                #crate_path::Attribute::String(v) => v.clone(),
                _ => panic!("expected String attribute"),
            }
        },
        AttrType::Symbol | AttrType::QualifiedName => quote! {
            match attr {
                #crate_path::Attribute::Symbol(v) => *v,
                _ => panic!("expected Symbol attribute"),
            }
        },
        AttrType::Bytes => quote! {
            match attr {
                #crate_path::Attribute::Bytes(v) => v.clone(),
                _ => panic!("expected Bytes attribute"),
            }
        },
    }
}
