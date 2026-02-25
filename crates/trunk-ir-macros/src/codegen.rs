//! Code generation for `#[arena_dialect]`.

use heck::{ToShoutySnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{
    AttrDef, AttrType, DialectItem, DialectModule, Operand, OperationDef, RegionOrSuccessor,
    ResultDef,
};

/// Generate all code for a dialect module.
pub fn generate(crate_path: &TokenStream, module: &DialectModule) -> TokenStream {
    let dialect_name_fn = gen_dialect_name(crate_path, &module.name);

    let op_items: Vec<TokenStream> = module
        .items
        .iter()
        .filter_map(|item| match item {
            DialectItem::Operation(op) => Some(gen_operation(crate_path, &module.name, op)),
            DialectItem::TypeDef => None,
        })
        .collect();

    quote! {
        #dialect_name_fn
        #(#op_items)*
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
    let constructor = gen_constructor(crate_path, dialect, op);

    quote! {
        #op_name_fn
        #struct_and_trait
        #impl_block
        #constructor
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

    quote! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct #sname(#crate_path::arena::OpRef);

        impl #crate_path::arena::ops::ArenaDialectOp for #sname {
            const DIALECT_NAME: &'static str = #dialect;
            const OP_NAME: &'static str = #op_name;

            fn from_op(
                ctx: &#crate_path::arena::IrContext,
                op: #crate_path::arena::OpRef,
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

            fn op_ref(&self) -> #crate_path::arena::OpRef {
                self.0
            }
        }
    }
}

fn gen_impl_block(crate_path: &TokenStream, op: &OperationDef) -> TokenStream {
    let sname = struct_name(&op.name);
    let op_ref_method = quote! {
        pub fn op_ref(&self) -> #crate_path::arena::OpRef {
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
            pub fn #name<'a>(&self, ctx: &'a #crate_path::arena::IrContext) -> &'a [#crate_path::arena::ValueRef] {
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
                    pub fn #name<'a>(&self, ctx: &'a #crate_path::arena::IrContext) -> &'a [#crate_path::arena::ValueRef] {
                        &ctx.op_operands(self.0)[#idx..]
                    }
                });
            } else {
                methods.push(quote! {
                    pub fn #name<'a>(&self, ctx: &'a #crate_path::arena::IrContext) -> &'a [#crate_path::arena::ValueRef] {
                        ctx.op_operands(self.0)
                    }
                });
            }
        } else {
            let idx = fixed_count;
            methods.push(quote! {
                pub fn #name(&self, ctx: &#crate_path::arena::IrContext) -> #crate_path::arena::ValueRef {
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
        ResultDef::Single(name) => {
            let name_ident = format_ident!("{name}");
            let ty_name = format_ident!("{name}_ty");
            quote! {
                pub fn #name_ident(&self, ctx: &#crate_path::arena::IrContext) -> #crate_path::arena::ValueRef {
                    ctx.op_result(self.0, 0)
                }

                pub fn #ty_name(&self, ctx: &#crate_path::arena::IrContext) -> #crate_path::arena::TypeRef {
                    ctx.op_result_types(self.0)[0]
                }
            }
        }
        ResultDef::Multi(names) => {
            let methods: Vec<TokenStream> = names
                .iter()
                .enumerate()
                .map(|(idx, name)| {
                    let name_ident = format_ident!("{name}");
                    let ty_name = format_ident!("{name}_ty");
                    quote! {
                        pub fn #name_ident(&self, ctx: &#crate_path::arena::IrContext) -> #crate_path::arena::ValueRef {
                            ctx.op_result(self.0, #idx)
                        }

                        pub fn #ty_name(&self, ctx: &#crate_path::arena::IrContext) -> #crate_path::arena::TypeRef {
                            ctx.op_result_types(self.0)[#idx]
                        }
                    }
                })
                .collect();
            quote!(#(#methods)*)
        }
        ResultDef::Variadic(name) => {
            let name_ident = format_ident!("{name}");
            quote! {
                pub fn #name_ident<'a>(&self, ctx: &'a #crate_path::arena::IrContext) -> &'a [#crate_path::arena::ValueRef] {
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
    let name = &attr.raw_ident;
    let name_str = &attr.name;
    let rust_ty = attr_rust_type(crate_path, attr.ty);
    let from_attr = attr_from_attr(crate_path, attr.ty);

    if attr.optional {
        quote! {
            pub fn #name(&self, ctx: &#crate_path::arena::IrContext) -> Option<#rust_ty> {
                ctx.op(self.0).attributes
                    .get(&#crate_path::Symbol::new(#name_str))
                    .map(|attr| #from_attr)
            }
        }
    } else {
        quote! {
            pub fn #name(&self, ctx: &#crate_path::arena::IrContext) -> #rust_ty {
                let attr = ctx.op(self.0).attributes
                    .get(&#crate_path::Symbol::new(#name_str))
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
            RegionOrSuccessor::Region(name) => {
                let name_ident = format_ident!("{name}");
                let idx = region_idx;
                methods.push(quote! {
                    pub fn #name_ident(&self, ctx: &#crate_path::arena::IrContext) -> #crate_path::arena::RegionRef {
                        ctx.op(self.0).regions[#idx]
                    }
                });
                region_idx += 1;
            }
            RegionOrSuccessor::Successor(name) => {
                let name_ident = format_ident!("{name}");
                let idx = succ_idx;
                methods.push(quote! {
                    pub fn #name_ident(&self, ctx: &#crate_path::arena::IrContext) -> #crate_path::arena::BlockRef {
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

fn gen_constructor(crate_path: &TokenStream, dialect: &str, op: &OperationDef) -> TokenStream {
    let sname = struct_name(&op.name);
    let fn_name = &op.raw_ident;
    let op_name = &op.name;

    // Build parameter list
    let mut params = Vec::new();
    let mut body_stmts = Vec::new();

    // Fixed operands
    for operand in &op.operands {
        let name = &operand.raw_ident;
        if operand.variadic {
            params.push(quote!(#name: impl IntoIterator<Item = #crate_path::arena::ValueRef>));
            body_stmts.push(quote!(__builder = __builder.operands(#name);));
        } else {
            params.push(quote!(#name: #crate_path::arena::ValueRef));
            body_stmts.push(quote!(__builder = __builder.operand(#name);));
        }
    }

    // Results
    match &op.results {
        ResultDef::None => {}
        ResultDef::Single(name) => {
            let ty_param = format_ident!("{name}_ty");
            params.push(quote!(#ty_param: #crate_path::arena::TypeRef));
            body_stmts.push(quote!(__builder = __builder.result(#ty_param);));
        }
        ResultDef::Multi(names) => {
            for name in names {
                let ty_param = format_ident!("{name}_ty");
                params.push(quote!(#ty_param: #crate_path::arena::TypeRef));
                body_stmts.push(quote!(__builder = __builder.result(#ty_param);));
            }
        }
        ResultDef::Variadic(_) => {
            params
                .push(quote!(result_types: impl IntoIterator<Item = #crate_path::arena::TypeRef>));
            body_stmts.push(quote!(__builder = __builder.results(result_types);));
        }
    }

    // Attributes
    for attr in &op.attrs {
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

    // Regions and successors
    for item in &op.regions {
        match item {
            RegionOrSuccessor::Region(name) => {
                let name_ident = format_ident!("{name}");
                params.push(quote!(#name_ident: #crate_path::arena::RegionRef));
                body_stmts.push(quote!(__builder = __builder.region(#name_ident);));
            }
            RegionOrSuccessor::Successor(name) => {
                let name_ident = format_ident!("{name}");
                params.push(quote!(#name_ident: #crate_path::arena::BlockRef));
                body_stmts.push(quote!(__builder = __builder.successor(#name_ident);));
            }
        }
    }

    quote! {
        #[allow(clippy::too_many_arguments)]
        pub fn #fn_name(
            ctx: &mut #crate_path::arena::IrContext,
            location: #crate_path::arena::Location,
            #(#params),*
        ) -> #sname {
            #[allow(unused_mut)]
            let mut __builder = #crate_path::arena::OperationDataBuilder::new(
                location,
                #crate_path::Symbol::new(#dialect),
                #crate_path::Symbol::new(#op_name),
            );
            #(#body_stmts)*
            let __data = __builder.build(ctx);
            let __op_ref = ctx.create_op(__data);
            #sname(__op_ref)
        }
    }
}

// ============================================================================
// Attribute type helpers
// ============================================================================

fn attr_rust_type(crate_path: &TokenStream, ty: AttrType) -> TokenStream {
    match ty {
        AttrType::Any => quote!(#crate_path::arena::Attribute),
        AttrType::Bool => quote!(bool),
        AttrType::I32 => quote!(i32),
        AttrType::I64 => quote!(i64),
        AttrType::U32 => quote!(u32),
        AttrType::U64 => quote!(u64),
        AttrType::F32 => quote!(f32),
        AttrType::F64 => quote!(f64),
        AttrType::Type => quote!(#crate_path::arena::TypeRef),
        AttrType::String => quote!(::std::string::String),
        AttrType::Symbol | AttrType::QualifiedName => quote!(#crate_path::Symbol),
    }
}

fn attr_to_attr(crate_path: &TokenStream, ty: AttrType, val: TokenStream) -> TokenStream {
    match ty {
        AttrType::Any => quote!(#val),
        AttrType::Bool => quote!(#crate_path::arena::Attribute::Bool(#val)),
        AttrType::I32 | AttrType::I64 | AttrType::U32 => {
            quote!(#crate_path::arena::Attribute::IntBits(#val as u64))
        }
        AttrType::U64 => quote!(#crate_path::arena::Attribute::IntBits(#val)),
        AttrType::F32 => {
            quote!(#crate_path::arena::Attribute::FloatBits((#val as f64).to_bits()))
        }
        AttrType::F64 => quote!(#crate_path::arena::Attribute::FloatBits(#val.to_bits())),
        AttrType::Type => quote!(#crate_path::arena::Attribute::Type(#val)),
        AttrType::String => quote!(#crate_path::arena::Attribute::String(#val)),
        AttrType::Symbol | AttrType::QualifiedName => {
            quote!(#crate_path::arena::Attribute::Symbol(#val))
        }
    }
}

fn attr_from_attr(crate_path: &TokenStream, ty: AttrType) -> TokenStream {
    match ty {
        AttrType::Any => quote!(attr.clone()),
        AttrType::Bool => quote! {
            match attr {
                #crate_path::arena::Attribute::Bool(v) => *v,
                _ => panic!("expected Bool attribute"),
            }
        },
        AttrType::I32 => quote! {
            match attr {
                #crate_path::arena::Attribute::IntBits(v) => *v as i32,
                _ => panic!("expected IntBits attribute"),
            }
        },
        AttrType::I64 => quote! {
            match attr {
                #crate_path::arena::Attribute::IntBits(v) => *v as i64,
                _ => panic!("expected IntBits attribute"),
            }
        },
        AttrType::U32 => quote! {
            match attr {
                #crate_path::arena::Attribute::IntBits(v) => *v as u32,
                _ => panic!("expected IntBits attribute"),
            }
        },
        AttrType::U64 => quote! {
            match attr {
                #crate_path::arena::Attribute::IntBits(v) => *v,
                _ => panic!("expected IntBits attribute"),
            }
        },
        AttrType::F32 => quote! {
            match attr {
                #crate_path::arena::Attribute::FloatBits(v) => f64::from_bits(*v) as f32,
                _ => panic!("expected FloatBits attribute"),
            }
        },
        AttrType::F64 => quote! {
            match attr {
                #crate_path::arena::Attribute::FloatBits(v) => f64::from_bits(*v),
                _ => panic!("expected FloatBits attribute"),
            }
        },
        AttrType::Type => quote! {
            match attr {
                #crate_path::arena::Attribute::Type(v) => *v,
                _ => panic!("expected Type attribute"),
            }
        },
        AttrType::String => quote! {
            match attr {
                #crate_path::arena::Attribute::String(v) => v.clone(),
                _ => panic!("expected String attribute"),
            }
        },
        AttrType::Symbol | AttrType::QualifiedName => quote! {
            match attr {
                #crate_path::arena::Attribute::Symbol(v) => *v,
                _ => panic!("expected Symbol attribute"),
            }
        },
    }
}
