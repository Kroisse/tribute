//! Function-symbol references at the final Wasm emission boundary.
//!
//! Target lowering can leave well-formed bodyless `wasm.func` declarations in
//! the module: ordinary C helper declarations that survived call rewriting and
//! registered intrinsic declarations whose calls were already removed. A Wasm
//! code entry and body are mandatory, so a bodyless declaration can only be
//! emitted when an explicit import binds its symbol.
//!
//! This module owns the read-only emission disposition described in
//! `new-plans/wasm-backend.md`. It first collects every surviving reference to
//! a function symbol, freshly from the final IR, and then drops the bodyless
//! declarations that nothing uses. Omission is decided here and never mutates
//! the IR, which keeps it distinct from generic DCE reachability and
//! authenticated intrinsic deletion.

use std::collections::HashSet;

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef};

use super::definitions::{FunctionDef, ImportFuncDef};
use crate::{CompilationError, CompilationResult};

/// Attributes that name a function symbol. Every operation that owns one is
/// modeled below; an operation that owns an unmodeled one is rejected rather
/// than silently treated as a non-user.
const FUNCTION_SYMBOL_ATTRIBUTES: [&str; 3] = ["callee", "func_name", "func"];

/// How a surviving operation refers to a function symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FunctionReference {
    /// `wasm.call`
    DirectCall,
    /// `wasm.return_call`
    TailCall,
    /// `wasm.ref_func` outside an element segment
    Value,
    /// `wasm.ref_func` inside a `wasm.elem` funcs region
    Table,
    /// `wasm.export_func`
    Export,
}

impl FunctionReference {
    fn describe(self) -> &'static str {
        match self {
            Self::DirectCall => "direct call",
            Self::TailCall => "proper tail call",
            Self::Value => "first-class function reference",
            Self::Table => "element segment entry",
            Self::Export => "export",
        }
    }
}

/// A surviving function-symbol reference, for diagnostics.
pub(crate) struct ResolvedReference {
    pub symbol: Symbol,
    pub kind: FunctionReference,
}

/// Collect every function-symbol reference from the operations admitted at the
/// final Wasm boundary.
///
/// Container operations are traversed recursively, so an element segment's
/// `funcs` region and nested control regions never hide their children. An
/// operation that owns a function-symbol attribute but is not modeled, or whose
/// modeled attribute is missing, is a malformed boundary user and is rejected.
pub(crate) fn collect_function_references(
    ctx: &IrContext,
    module: Module,
) -> CompilationResult<Vec<ResolvedReference>> {
    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;
    let mut references = Vec::new();
    collect_region(ctx, body, false, &mut references)?;
    Ok(references)
}

fn collect_region(
    ctx: &IrContext,
    region: RegionRef,
    inside_element_segment: bool,
    references: &mut Vec<ResolvedReference>,
) -> CompilationResult<()> {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            collect_op(ctx, op, inside_element_segment, references)?;
        }
    }
    Ok(())
}

fn collect_op(
    ctx: &IrContext,
    op: OpRef,
    inside_element_segment: bool,
    references: &mut Vec<ResolvedReference>,
) -> CompilationResult<()> {
    let (kind, attribute) = if wasm_dialect::Call::matches(ctx, op) {
        (Some(FunctionReference::DirectCall), "callee")
    } else if wasm_dialect::ReturnCall::matches(ctx, op) {
        (Some(FunctionReference::TailCall), "callee")
    } else if wasm_dialect::ExportFunc::matches(ctx, op) {
        (Some(FunctionReference::Export), "func")
    } else if wasm_dialect::RefFunc::matches(ctx, op) {
        let kind = if inside_element_segment {
            FunctionReference::Table
        } else {
            FunctionReference::Value
        };
        (Some(kind), "func_name")
    } else {
        (None, "")
    };

    match kind {
        Some(kind) => {
            let Some(symbol) = ctx.op(op).attributes.get_symbol(attribute) else {
                return Err(CompilationError::invalid_module(format!(
                    "wasm.{} requires a symbol `{}` attribute",
                    ctx.op(op).name,
                    attribute,
                )));
            };
            references.push(ResolvedReference { symbol, kind });
        }
        None => {
            for owned in FUNCTION_SYMBOL_ATTRIBUTES {
                if ctx.op(op).attributes.get(owned).is_some() {
                    return Err(CompilationError::invalid_module(format!(
                        "wasm.{} owns `{}` but is not a modeled function-symbol user",
                        ctx.op(op).name,
                        owned,
                    )));
                }
            }
        }
    }

    let nested_inside_element = inside_element_segment || wasm_dialect::Elem::matches(ctx, op);
    for &nested in &ctx.op(op).regions {
        collect_region(ctx, nested, nested_inside_element, references)?;
    }
    Ok(())
}

/// Apply the read-only emission disposition for bodyless declarations.
///
/// A definition is emitted iff it owns a body region. Bodyless declarations are
/// removed from the emitted definition/index/code lists without touching the
/// IR. Every surviving reference must then resolve to an explicit import (which
/// keeps its import-first index) or to a body-bearing definition; an
/// unsatisfied one is reported precisely instead of failing later as a missing
/// code body.
pub(crate) fn dispose_bodyless_declarations(
    ctx: &IrContext,
    funcs: &mut Vec<FunctionDef>,
    imports: &[ImportFuncDef],
    references: &[ResolvedReference],
) -> CompilationResult<()> {
    let mut bound: HashSet<Symbol> = imports.iter().map(|import| import.sym).collect();
    bound.extend(
        funcs
            .iter()
            .filter(|func| has_body(ctx, func))
            .map(|func| func.name),
    );
    let declarations: HashSet<Symbol> = funcs
        .iter()
        .filter(|func| !has_body(ctx, func))
        .map(|func| func.name)
        .collect();

    funcs.retain(|func| has_body(ctx, func));

    for reference in references {
        if bound.contains(&reference.symbol) {
            continue;
        }
        let symbol = reference.symbol;
        let kind = reference.kind.describe();
        if declarations.contains(&symbol) {
            return Err(CompilationError::invalid_module(format!(
                "bodyless declaration @{symbol} has no import binding and no body \
                 (referenced by {kind})"
            )));
        }
        return Err(CompilationError::invalid_module(format!(
            "reference to @{symbol} is unresolved ({kind})"
        )));
    }

    Ok(())
}

/// A bodyless `wasm.func` declaration owns no region at all. A present region is
/// a definition and must carry an entry block.
fn has_body(ctx: &IrContext, func: &FunctionDef) -> bool {
    !ctx.op(func.op).regions.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Attribute;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use wasmparser::{Parser, Payload, Validator, WasmFeatures};

    /// A bodyless ordinary C helper declaration with an `(i32) -> i32` contract.
    const UNUSED_HELPER: &str = r#"  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<(core.i32) -> core.i32>}"#;

    fn section_count(bytes: &[u8], want_imports: bool) -> u32 {
        Parser::new(0)
            .parse_all(bytes)
            .filter_map(Result::ok)
            .find_map(|payload| match payload {
                Payload::ImportSection(reader) if want_imports => Some(reader.count()),
                Payload::FunctionSection(reader) if !want_imports => Some(reader.count()),
                _ => None,
            })
            .unwrap_or(0)
    }

    fn disposition_error(source: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, source);
        crate::emit_module_to_wasm(&mut ctx, module)
            .err()
            .expect("the unsatisfied function-symbol reference must fail emission")
            .to_string()
    }

    #[test]
    fn omits_unused_extern_but_keeps_imports_and_body_bearing_c_functions() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.import_func {module = @env, name = @run, sym_name = @run, type = wasm.func_sig<(core.i32) -> core.i32>}
  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<(core.i32) -> core.i32>}
  wasm.func {abi = "C", sym_name = @c_helper, type = wasm.func_sig<(core.i32) -> core.i32>} {
    ^bb0(%value: core.i32):
      wasm.return %value
  }
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.i32>} {
    %one = wasm.i32_const {value = 1} : core.i32
    %imported = wasm.call %one {callee = @run} : core.i32
    %local = wasm.call %imported {callee = @c_helper} : core.i32
    wasm.return %local
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let binary = crate::emit_module_to_wasm(&mut ctx, module)
            .expect("an unused ordinary extern requires no body");
        Validator::new_with_features(WasmFeatures::default())
            .validate_all(&binary.bytes)
            .expect("the emitted module must validate");

        assert_eq!(
            section_count(&binary.bytes, true),
            1,
            "imports stay imports"
        );
        assert_eq!(
            section_count(&binary.bytes, false),
            2,
            "only body-bearing definitions are emitted"
        );
        assert_eq!(
            print_module(&ctx, module.op()),
            before,
            "emission disposition must not mutate the IR"
        );
    }

    #[test]
    fn unused_declaration_alone_is_omitted() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<(core.i32) -> core.i32>}
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.nil>} { wasm.return }
}"#,
        );
        let binary = crate::emit_module_to_wasm(&mut ctx, module).expect("unused extern");
        assert_eq!(section_count(&binary.bytes, false), 1);
    }

    #[test]
    fn referenced_bodyless_declarations_report_their_reference_role() {
        let direct_call = format!(
            r#"core.module @test {{
{UNUSED_HELPER}
  wasm.func {{sym_name = @main, type = wasm.func_sig<(core.i32) -> core.i32>}} {{
    %arg = wasm.i32_const {{value = 1}} : core.i32
    %result = wasm.call %arg {{callee = @helper}} : core.i32
    wasm.return %result
  }}
}}"#
        );
        assert!(
            disposition_error(&direct_call)
                .contains("@helper has no import binding and no body (referenced by direct call)"),
            "{}",
            disposition_error(&direct_call)
        );

        let tail_call = r#"core.module @test {
  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<() -> core.i32>}
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.i32>} {
    wasm.return_call {callee = @helper}
  }
}"#;
        assert!(
            disposition_error(tail_call).contains("referenced by proper tail call"),
            "{}",
            disposition_error(tail_call)
        );

        let value = r#"core.module @test {
  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<() -> core.i32>}
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.nil>} {
    %function = wasm.ref_func {func_name = @helper} : wasm.funcref
    wasm.return
  }
}"#;
        assert!(
            disposition_error(value).contains("referenced by first-class function reference"),
            "{}",
            disposition_error(value)
        );

        // The element segment is a late container: its child `wasm.ref_func` use
        // must not be hidden by the container's own lack of a symbol attribute.
        let element = r#"core.module @test {
  wasm.table {reftype = @funcref, min = 1, max = 1}
  wasm.elem {table = 0, offset = 0} {
    wasm.ref_func {func_name = @helper} : wasm.funcref
  }
  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<() -> core.i32>}
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.nil>} { wasm.return }
}"#;
        assert!(
            disposition_error(element).contains("referenced by element segment entry"),
            "{}",
            disposition_error(element)
        );

        let export = r#"core.module @test {
  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<() -> core.i32>}
  wasm.export_func {func = @helper, name = "helper"}
}"#;
        assert!(
            disposition_error(export).contains("referenced by export"),
            "{}",
            disposition_error(export)
        );
    }

    #[test]
    fn unresolved_reference_is_reported_separately_from_an_unbound_declaration() {
        let source = r#"core.module @test {
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.i32>} {
    %result = wasm.call {callee = @missing} : core.i32
    wasm.return %result
  }
}"#;
        assert!(
            disposition_error(source).contains("reference to @missing is unresolved (direct call)"),
            "{}",
            disposition_error(source)
        );
    }

    #[test]
    fn malformed_function_symbol_users_are_rejected() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func {sym_name = @callee, type = wasm.func_sig<() -> core.nil>} { wasm.return }
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.nil>} {
    wasm.call {callee = @callee}
    wasm.return
  }
}"#,
        );
        let body = module.body(&ctx).unwrap();
        let main_block = ctx.region(body).blocks[0];
        let main = ctx.block(main_block).ops[1];
        let inner = ctx.region(ctx.op(main).regions[0]).blocks[0];
        let call = ctx.block(inner).ops[0];
        assert!(wasm_dialect::Call::matches(&ctx, call));
        ctx.op_mut(call).attributes.remove("callee");

        let error = collect_function_references(&ctx, module)
            .err()
            .expect("a modeled op without its symbol attribute is malformed");
        assert!(
            error
                .to_string()
                .contains("wasm.call requires a symbol `callee` attribute"),
            "{error}"
        );

        // An op that owns a function-symbol attribute but is outside the model is
        // rejected rather than silently treated as a non-user.
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func {sym_name = @main, type = wasm.func_sig<() -> core.nil>} { wasm.return }
}"#,
        );
        let body = module.body(&ctx).unwrap();
        let main_block = ctx.region(body).blocks[0];
        let main = ctx.block(main_block).ops[0];
        let inner = ctx.region(ctx.op(main).regions[0]).blocks[0];
        let ret = ctx.block(inner).ops[0];
        assert!(wasm_dialect::Return::matches(&ctx, ret));
        ctx.op_mut(ret).attributes.insert(
            Symbol::new("callee"),
            Attribute::Symbol(Symbol::new("helper")),
        );

        let error = collect_function_references(&ctx, module)
            .err()
            .expect("an unmodeled function-symbol user is rejected");
        assert!(
            error
                .to_string()
                .contains("wasm.return owns `callee` but is not a modeled function-symbol user"),
            "{error}"
        );
    }

    #[test]
    fn present_but_empty_body_is_a_malformed_definition_not_a_declaration() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func {abi = "C", sym_name = @helper, type = wasm.func_sig<() -> core.nil>} {}
}"#,
        );
        let error = crate::emit_module_to_wasm(&mut ctx, module)
            .err()
            .expect("a region without an entry block is not a bodyless declaration");
        assert!(error.to_string().contains("no entry block"), "{error}");
    }
}
