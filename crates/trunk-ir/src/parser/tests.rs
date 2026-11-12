//! Tests of the text format parser.

use rustc_hash::FxHashMap as HashMap;
use smallvec::smallvec;

use super::{parse_module, parse_test_module};
use crate::context::{IrContext, OperationDataBuilder};
use crate::dialect::{arith, clif, core, func};
use crate::ops::{DialectOp, DialectType};
use crate::refs::*;
use crate::rewrite::Module;
use crate::types::*;
use crate::validation;
use crate::{BlockArgData, BlockData, RegionData, Symbol, SymbolPath};

/// Build IR → print → parse → print, verify texts match.
fn assert_roundtrip(ctx: &IrContext, module_op: OpRef) {
    let printed = module_op.display(ctx).to_string();
    let mut ctx2 = IrContext::new();
    let parsed_op = parse_module(&mut ctx2, &printed).unwrap_or_else(|e| {
        panic!(
            "Round-trip parse failed at offset {}:\n  {}\n\nPrinted IR:\n{}",
            e.offset, e.message, printed
        );
    });
    let reprinted = parsed_op.display(&ctx2).to_string();
    assert_eq!(printed, reprinted, "Round-trip mismatch");
}

fn test_location(ctx: &mut IrContext) -> Location {
    let path = ctx.intern_path("test.trb");
    Location::new(path, crate::location::Span::new(0, 0))
}

fn make_i32_type(ctx: &mut IrContext) -> TypeRef {
    TypeDataBuilder::new("core.i32").intern(ctx)
}

fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
    func::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
}

fn wrap_in_module(ctx: &mut IrContext, loc: Location, func_ops: Vec<OpRef>) -> OpRef {
    let mod_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    for op in func_ops {
        ctx.push_op(mod_block, op);
    }
    let mod_region = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![mod_block],
        parent_op: None,
    });
    core::Module::operands()
        .sym_name("test")
        .regions(mod_region)
        .build(ctx, loc)
        .op_ref()
}

// ================================================================
// Round-trip tests
// ================================================================

#[test]
fn test_roundtrip_simple_module() {
    let mut ctx = IrContext::new();
    let loc = test_location(&mut ctx);
    let i32_ty = make_i32_type(&mut ctx);
    let func_ty = make_func_type(&mut ctx, &[], i32_ty);

    let entry = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let c = arith::Constant::operands()
        .value(Attribute::Int(42))
        .results(i32_ty)
        .build(&mut ctx, loc);
    ctx.push_op(entry, c.op_ref());
    let c_val = c.result(&ctx);
    let ret = func::Return::operands([c_val]).build(&mut ctx, loc);
    ctx.push_op(entry, ret.op_ref());
    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry],
        parent_op: None,
    });
    let f = func::Func::operands()
        .sym_name("main")
        .r#type(func_ty)
        .regions(body)
        .build(&mut ctx, loc);
    let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_namespaced_type_attribute_key() {
    let input = r#"
core.module @test {
  %0 = "test.make"() : test.value<{test.marker = 2}>
}
"#;
    let mut ctx = IrContext::new();
    let root = parse_module(&mut ctx, input).expect("namespaced attribute key should parse");
    let printed = root.display(&ctx).to_string();
    assert!(printed.contains("test.marker = 2"));
    assert_roundtrip(&ctx, root);
}

#[test]
fn test_roundtrip_module_attributes() {
    let input = r#"core.module @test {test.flag = 1, test.string = test.value<{name = @Str}>} {
  !Str = test.value<{name = @Str}>

  %0 = "test.make"() : !Str
}
"#;
    let mut ctx = IrContext::new();
    let root = parse_module(&mut ctx, input).expect("module attributes should parse");
    let printed = root.display(&ctx).to_string();
    assert_eq!(printed, input);
    assert_roundtrip(&ctx, root);
}

#[test]
fn regions_without_blocks_are_not_read_as_an_attribute_dictionary() {
    let input = r#"core.module @test {
  "test.op"() ({
  }, {
  })
  "test.op"() ({
  }) {test.flag = 1}
}
"#;
    let mut ctx = IrContext::new();
    let root = parse_module(&mut ctx, input).expect("empty regions parse");
    let module = Module::new(&ctx, root).unwrap();
    let ops = module.ops(&ctx);
    assert_eq!(ctx.op_region_count(ops[0]), 2);
    assert!(ctx.op(ops[0]).attributes.is_empty());
    assert_eq!(ctx.op_region_count(ops[1]), 1);
    assert_eq!(root.display(&ctx).to_string(), input);
    assert_roundtrip(&ctx, root);
}

#[test]
fn test_roundtrip_module_attribute_keys_that_need_quoting() {
    let input = r#"core.module @test {"test.dash-key" = 1, test.plain = 2} {
  %0 = "test.make"() : core.i32
}
"#;
    let mut ctx = IrContext::new();
    let root = parse_module(&mut ctx, input).expect("quoted module attribute keys parse");
    let printed = root.display(&ctx).to_string();
    assert_eq!(printed, input);
    assert_roundtrip(&ctx, root);
}

#[test]
fn test_roundtrip_dict_attributes() {
    let input = r#"
core.module @test {
  %0 = "test.make"() {meta = {b = [{}, {k = @v}], a = core.i32}, nested = test.value<{b = {}}>} : test.value<{param = {x = 1}}>
}
"#;
    let mut ctx = IrContext::new();
    let root = parse_module(&mut ctx, input).expect("dictionary attributes should parse");
    let printed = root.display(&ctx).to_string();
    assert!(
        printed
            .contains("{meta = {a = core.i32, b = [{}, {k = @v}]}, nested = test.value<{b = {}}>}"),
        "dictionary keys print in sorted order:\n{printed}"
    );
    assert!(
        printed.contains("test.value<{param = {x = 1}}>"),
        "{printed}"
    );
    assert_roundtrip(&ctx, root);
}

#[test]
fn test_roundtrip_dict_keys_that_need_quoting() {
    let input = r#"
core.module @test {
  %0 = "test.make"() {meta = {"1st" = 1, "a b" = 2, "q\"uote" = 3, "x::y" = 4, plain.key = 5}} : core.i32
}
"#;
    let mut ctx = IrContext::new();
    let root = parse_module(&mut ctx, input).expect("quoted dictionary keys should parse");
    let printed = root.display(&ctx).to_string();
    assert!(
        printed.contains(
            r#"{meta = {"1st" = 1, "a b" = 2, plain.key = 5, "q\"uote" = 3, "x::y" = 4}}"#
        ),
        "{printed}"
    );
    assert_roundtrip(&ctx, root);
}

#[test]
fn test_roundtrip_parameter_attributes() {
    let input = r#"core.module @test {
  !pair = core.tuple<core.i32, core.ptr {k = @v}>
  !sig = func.func_sig<(core.i32 {a = core.i32}, core.ptr) -> core.i32>
  !plain = core.tuple<core.i32 {}, core.ptr {}>
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("parameter attributes should parse");
    let printed = module.display(&ctx).to_string();
    assert!(
        printed.contains("core.tuple<core.i32, core.ptr {k = @v}>"),
        "{printed}"
    );
    assert!(
        printed.contains("func.func_sig<(core.i32 {a = core.i32}, core.ptr) -> core.i32>"),
        "{printed}"
    );
    let aliases: HashMap<_, _> = ctx
        .type_aliases()
        .iter()
        .map(|(name, ty)| (name.to_string(), *ty))
        .collect();
    let sig = func::FuncSig::from_type_ref(&ctx, aliases["sig"]).unwrap();
    assert_eq!(
        sig.input_attrs(&ctx)
            .filter(|attrs| !attrs.is_empty())
            .count(),
        1
    );
    assert!(sig.result_attrs(&ctx).all(|attrs| attrs.is_empty()));
    assert!(
        !ctx.type_attrs(aliases["plain"])
            .contains_key(crate::types::PARAM_ATTRS_ATTR),
        "all-empty parameter attributes are canonicalized away"
    );
    assert_roundtrip(&ctx, module);
}

#[test]
fn test_explicit_parameter_attributes_are_parse_errors() {
    for spelling in [
        "core.tuple<core.i32, {param_attrs = [{k = 1}]}>",
        "core.tuple<core.i32, {param_attrs = 1}>",
        "func.func_sig<(core.i32) -> (), {param_attrs = [{k = 1}]}>",
    ] {
        let mut ctx = IrContext::new();
        let input = format!("core.module @test {{ !bad = {spelling} }}");
        let error = parse_module(&mut ctx, &input).expect_err(spelling);
        assert!(
            error.message.contains("`param_attrs` is reserved"),
            "{spelling}: {error}"
        );
    }
}

#[test]
fn test_dict_attribute_rejects_duplicate_keys() {
    let input = "core.module @test { %0 = \"test.make\"() {meta = {a = 1, a = 2}} : core.i32 }";
    let mut ctx = IrContext::new();
    let error = parse_module(&mut ctx, input).expect_err("duplicate key must be rejected");
    assert!(
        error
            .message
            .contains("duplicate dictionary attribute key 'a'"),
        "{error}"
    );
}

#[test]
fn test_roundtrip_with_params() {
    let mut ctx = IrContext::new();
    let loc = test_location(&mut ctx);
    let i32_ty = make_i32_type(&mut ctx);
    let func_ty = make_func_type(&mut ctx, &[i32_ty, i32_ty], i32_ty);

    let entry = ctx.create_block(BlockData {
        location: loc,
        args: vec![
            BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            },
        ],
        ops: smallvec![],
        parent_region: None,
    });
    let x = ctx.block_arg(entry, 0);
    let y = ctx.block_arg(entry, 1);
    let add = arith::Addi::operands(x, y).build(&mut ctx, loc);
    ctx.push_op(entry, add.op_ref());
    let add_val = add.result(&ctx);
    let ret = func::Return::operands([add_val]).build(&mut ctx, loc);
    ctx.push_op(entry, ret.op_ref());
    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry],
        parent_op: None,
    });
    let f = func::Func::operands()
        .sym_name("add")
        .r#type(func_ty)
        .regions(body)
        .build(&mut ctx, loc);
    let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_nested_regions() {
    let mut ctx = IrContext::new();
    let loc = test_location(&mut ctx);
    let i32_ty = make_i32_type(&mut ctx);
    let i1_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);
    let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);

    // Entry block with param
    let entry = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: i32_ty,
            attrs: Default::default(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let param = ctx.block_arg(entry, 0);

    // Condition
    let cond = arith::Constant::operands()
        .value(Attribute::Int(1))
        .results(i1_ty)
        .build(&mut ctx, loc);
    ctx.push_op(entry, cond.op_ref());
    let cond_val = cond.result(&ctx);

    // Then region: scf.yield %param
    let then_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let yield_then = OperationDataBuilder::new(loc, "scf.yield")
        .operand(param)
        .build(&mut ctx);
    let yield_then = ctx.create_op(yield_then);
    ctx.push_op(then_block, yield_then);
    let then_region = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![then_block],
        parent_op: None,
    });

    // Else region: const 1, add param+1, yield
    let else_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let c1 = arith::Constant::operands()
        .value(Attribute::Int(1))
        .results(i32_ty)
        .build(&mut ctx, loc);
    ctx.push_op(else_block, c1.op_ref());
    let c1_val = c1.result(&ctx);
    let sum = arith::Addi::operands(param, c1_val).build(&mut ctx, loc);
    ctx.push_op(else_block, sum.op_ref());
    let sum_val = sum.result(&ctx);
    let yield_else = OperationDataBuilder::new(loc, "scf.yield")
        .operand(sum_val)
        .build(&mut ctx);
    let yield_else = ctx.create_op(yield_else);
    ctx.push_op(else_block, yield_else);
    let else_region = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![else_block],
        parent_op: None,
    });

    // scf.if
    let if_op = OperationDataBuilder::new(loc, "scf.if")
        .operand(cond_val)
        .result(i32_ty)
        .region(then_region)
        .region(else_region)
        .build(&mut ctx);
    let if_op = ctx.create_op(if_op);
    ctx.push_op(entry, if_op);
    let if_result = ctx.op_result(if_op, 0);

    let ret = func::Return::operands([if_result]).build(&mut ctx, loc);
    ctx.push_op(entry, ret.op_ref());

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry],
        parent_op: None,
    });
    let f = func::Func::operands()
        .sym_name("choose")
        .r#type(func_ty)
        .regions(body)
        .build(&mut ctx, loc);
    let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_effects() {
    let mut ctx = IrContext::new();
    let loc = test_location(&mut ctx);
    let i32_ty = make_i32_type(&mut ctx);
    let func_ty = make_func_type(&mut ctx, &[], i32_ty);

    let entry = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let c = arith::Constant::operands()
        .value(Attribute::Int(7))
        .results(i32_ty)
        .build(&mut ctx, loc);
    ctx.push_op(entry, c.op_ref());
    let c_val = c.result(&ctx);
    let ret = func::Return::operands([c_val]).build(&mut ctx, loc);
    ctx.push_op(entry, ret.op_ref());
    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry],
        parent_op: None,
    });
    let f = func::Func::operands()
        .sym_name("pure")
        .r#type(func_ty)
        .regions(body)
        .build(&mut ctx, loc);
    let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_multiple_functions() {
    let mut ctx = IrContext::new();
    let loc = test_location(&mut ctx);
    let i32_ty = make_i32_type(&mut ctx);
    let func_ty = make_func_type(&mut ctx, &[], i32_ty);

    let mut funcs = vec![];
    for (name, val) in &[("foo", 1i128), ("bar", 2)] {
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c = arith::Constant::operands()
            .value(Attribute::Int(*val))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry, c.op_ref());
        let c_val = c.result(&ctx);
        let ret = func::Return::operands([c_val]).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::Func::operands()
            .sym_name(*name)
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, loc);
        funcs.push(f.op_ref());
    }
    let module_op = wrap_in_module(&mut ctx, loc, funcs);

    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_func_call() {
    let mut ctx = IrContext::new();
    let loc = test_location(&mut ctx);
    let i32_ty = make_i32_type(&mut ctx);

    // fn callee() -> i32 { return 42 }
    let callee_ty = make_func_type(&mut ctx, &[], i32_ty);
    let entry1 = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let c = arith::Constant::operands()
        .value(Attribute::Int(42))
        .results(i32_ty)
        .build(&mut ctx, loc);
    ctx.push_op(entry1, c.op_ref());
    let c_val = c.result(&ctx);
    let ret1 = func::Return::operands([c_val]).build(&mut ctx, loc);
    ctx.push_op(entry1, ret1.op_ref());
    let body1 = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry1],
        parent_op: None,
    });
    let callee = func::Func::operands()
        .sym_name("callee")
        .r#type(callee_ty)
        .regions(body1)
        .build(&mut ctx, loc);

    // fn main() -> i32 { call @callee, return result }
    let main_ty = make_func_type(&mut ctx, &[], i32_ty);
    let entry2 = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let call = func::Call::operands([])
        .callee(SymbolPath::new(["callee"]))
        .results([i32_ty])
        .build(&mut ctx, loc);
    ctx.push_op(entry2, call.op_ref());
    let call_val = call.results(&ctx)[0];
    let ret2 = func::Return::operands([call_val]).build(&mut ctx, loc);
    ctx.push_op(entry2, ret2.op_ref());
    let body2 = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry2],
        parent_op: None,
    });
    let main_fn = func::Func::operands()
        .sym_name("main")
        .r#type(main_ty)
        .regions(body2)
        .build(&mut ctx, loc);

    let module_op = wrap_in_module(&mut ctx, loc, vec![callee.op_ref(), main_fn.op_ref()]);

    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_bodyless_func_decl() {
    // Body-less func.func should still emit its full signature
    let input = r#"core.module @test {
  func.func @extern_add(%arg0: core.i32, %arg1: core.i32) -> core.i32
}"#;
    let mut ctx = IrContext::new();
    let module_op = parse_module(&mut ctx, input).unwrap();
    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_non_ascii_symbol() {
    // Non-ASCII symbol names must be quoted to survive round-trip
    // (parser only accepts ASCII alphanumeric + '_' for bare symbols)
    let input = r#"core.module @test {
  func.func @"π"() -> core.i32 {
%0 = arith.constant 42 : core.i32
func.return %0
  }
}"#;
    let mut ctx = IrContext::new();
    let module_op = parse_module(&mut ctx, input).unwrap();
    assert_roundtrip(&ctx, module_op);
}

// ================================================================
// Error detection tests
// ================================================================

#[test]
fn test_parse_rejects_malformed_module() {
    // (case, input, expected error message fragment)
    for (case, input, expected) in [
        (
            "undefined operand",
            r#"core.module @test {
  func.func @f() -> core.i32 {
func.return %missing
  }
}"#,
            "undefined value '%missing'",
        ),
        (
            "duplicate block label",
            r#"core.module @test {
  func.func @f() -> core.i32 {
^bb0:
  %0 = arith.constant 1 : core.i32
  func.return %0
^bb0:
  %1 = arith.constant 2 : core.i32
  func.return %1
  }
}"#,
            "duplicate block label",
        ),
        (
            "entry block arity mismatch",
            r#"core.module @test {
  func.func @f(%0: core.i32, %1: core.i32) -> core.i32 {
^bb0(%2: core.i32):
  func.return %2
  }
}"#,
            "entry block has 1 args but function signature has 2 params",
        ),
        (
            "result count mismatch",
            r#"core.module @test {
  func.func @f() -> core.i32 {
%0, %1 = arith.constant 42 : core.i32
func.return %0
  }
}"#,
            "declares 2 result names but 1 result types",
        ),
        (
            "cross block duplicate arg name",
            r#"core.module @test {
  func.func @f() -> core.i32 {
^entry:
  %cond = arith.constant 1 : core.i1
  "scf.br"()[^left]
^left(%x: core.i32):
  "scf.br"()[^right]
^right(%x: core.i32):
  func.return %x
  }
}"#,
            "duplicate SSA name '%x'",
        ),
        (
            "nested region block isolation",
            r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
^entry:
  %cond = arith.constant 1 : core.i1
  %r = scf.if %cond -> (core.i32) {
    "scf.br"()[^entry]
  } else {
    scf.yield %0
  }
  func.return %r
  }
}"#,
            "undefined block '^entry'",
        ),
        (
            "undefined type alias error",
            r#"core.module @test {
  func.func @f(%0: !unknown) -> core.i32 {
func.return %0
  }
}"#,
            "undefined type alias '!unknown'",
        ),
        (
            "type alias rejected in non module region",
            r#"core.module @test {
  func.func @f() -> core.i32 {
!bad = core.i32
%0 = arith.constant 42 : !bad
func.return %0
  }
}"#,
            "type aliases are only allowed in module regions",
        ),
        (
            "duplicate block argument name",
            r#"core.module @test {
  func.func @f() {
    "test.br"()[^next]
  ^next(%a: core.i32, %a: core.i32):
    func.return
  }
}"#,
            "duplicate block argument name 'a' at index 1",
        ),
        (
            "entry block argument type mismatch",
            r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
^bb0(%1: core.i64):
  func.return %1
  }
}"#,
            "entry block arg 0 type mismatch: block has core.i64 but function param has core.i32",
        ),
        (
            "unquoted operation without a custom format",
            r#"core.module @test {
  test.nothing
}"#,
            "expected a quoted name",
        ),
        (
            "trailing input after the module",
            "core.module @a {\n}\ncore.module @b {\n}",
            "trailing input after top-level operation",
        ),
    ] {
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input)
            .err()
            .unwrap_or_else(|| panic!("{case}: expected a parse error"));
        assert!(
            err.message.contains(expected),
            "{case}: expected {expected:?}, got: {}",
            err.message
        );
    }
}

// ================================================================
// Arena-specific tests (validation after parse)
// ================================================================

#[test]
fn test_parse_validates_value_integrity() {
    let input = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
^bb0:
  %1 = arith.constant 1 : core.i32
  %2 = arith.addi %0, %1 : core.i32
  func.return %2
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    let result = validation::validate_value_integrity(&ctx, module);
    assert!(
        result.is_ok(),
        "Parsed module should pass value integrity: {}",
        result
    );
}

#[test]
fn test_parse_validates_use_chains() {
    let input = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
^bb0:
  %1 = arith.constant 1 : core.i32
  %2 = arith.addi %0, %1 : core.i32
  func.return %2
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    let result = validation::validate_use_chains(&ctx, module);
    assert!(
        result.is_ok(),
        "Parsed module should pass use-chain validation: {}",
        result
    );
}

#[test]
fn test_parse_then_validate_all() {
    let input = r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
^bb0:
  %2 = arith.addi %0, %1 : core.i32
  func.return %2
  }
  func.func @main() -> core.i32 {
%0 = arith.constant 40 : core.i32
%1 = arith.constant 2 : core.i32
%2 = func.call @add(%0, %1) : (core.i32, core.i32) -> core.i32
func.return %2
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    let result = validation::validate_all(&ctx, module, &mut Default::default());
    assert!(result.is_ok(), "Full validation should pass: {}", result);
}

#[test]
fn test_parse_test_module_convenience() {
    let input = r#"core.module @test {
  func.func @f() -> core.i32 {
%0 = arith.constant 99 : core.i32
func.return %0
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    // Verify it's a valid module
    let ops = module.ops(&ctx);
    assert_eq!(ops.len(), 1);
    let func_data = ctx.op(ops[0]);
    assert_eq!(func_data.name, crate::dialect::func::Func::NAME);
}

// ================================================================
// Scoping error tests
// ================================================================

// ================================================================
// Type alias tests
// ================================================================

#[test]
fn review_function_result_type_preserves_empty_body() {
    for results in ["()", "(), {effect = core.nil}"] {
        let input = format!(
            "core.module @test {{ func.func @f() -> func.func_sig<() -> {results}> {{}} }}"
        );
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, &input).unwrap();
        let function = ctx
            .block(ctx.region(ctx.op_region(module, 0).unwrap()).blocks[0])
            .ops[0];
        assert_eq!(
            ctx.op_region_count(function),
            1,
            "empty body must not become a declaration"
        );
        assert_roundtrip(&ctx, module);
        let printed = module.display(&ctx).to_string();
        let mut reparsed = IrContext::new();
        let module2 = parse_module(&mut reparsed, &printed).unwrap();
        let function2 = reparsed
            .block(
                reparsed
                    .region(reparsed.op_region(module2, 0).unwrap())
                    .blocks[0],
            )
            .ops[0];
        assert_eq!(reparsed.op_region_count(function2), 1);
        assert_eq!(printed.contains("effect = core.nil"), results != "()");
    }
}

#[test]
fn test_canonical_func_type_roundtrips_all_supported_arities() {
    let input = r#"core.module @test {
  !zero_zero = func.func_sig<() -> ()>
  !many_zero = func.func_sig<(core.i32, core.i64) -> ()>
  !zero_one = func.func_sig<() -> core.i32>
  !many_one = func.func_sig<(core.i32, core.i64) -> core.i32>
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("canonical function types should parse");

    for (name, input_count, result_count) in [
        ("zero_zero", 0, 0),
        ("many_zero", 2, 0),
        ("zero_one", 0, 1),
        ("many_one", 2, 1),
    ] {
        let ty = ctx.type_alias_by_text(name).expect("function alias");
        let function = func::FuncSig::from_type_ref(&ctx, ty).expect("validated func.func_sig");
        assert_eq!(function.inputs(&ctx).len(), input_count);
        assert_eq!(function.results(&ctx).len(), result_count);
    }
    assert_roundtrip(&ctx, module);
}

#[test]
fn generic_shared_func_assembly_remains_parseable() {
    let input = r#"core.module @test {
  !signature = func.func_sig<(core.i32) -> core.i32>
  func.func @generic(%0: core.i32) -> core.i32
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("generic func assembly should parse");
    let function = ctx
        .block(ctx.region(ctx.op_region(module, 0).unwrap()).blocks[0])
        .ops[0];
    assert_eq!(ctx.op_attrs(function).get_str("sym_name"), Some("generic"));
    assert_eq!(
        ctx.op_attrs(function).get_type("type"),
        ctx.type_alias_by_text("signature")
    );
}

#[test]
fn wasm_assembly_keeps_an_explicit_shared_signature_shared() {
    let input = r#"core.module @test {
  "wasm.func"() ({ wasm.return }) {sym_name = "unlowered", type = func.func_sig<() -> core.nil>}
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("explicit assembly should parse");
    let function = ctx
        .block(ctx.region(ctx.op_region(module, 0).unwrap()).blocks[0])
        .ops[0];
    let signature = ctx.op_attrs(function).get_type("type").unwrap();
    assert!(func::FuncSig::from_type_ref(&ctx, signature).is_some());
    assert!(crate::dialect::wasm::FuncSig::from_type_ref(&ctx, signature).is_none());
    assert!(
        module
            .display(&ctx)
            .to_string()
            .contains("type = func.func_sig<() -> core.nil>"),
        "the parser must not apply an implicit target conversion"
    );
}

#[test]
fn clif_syntax_uses_the_target_owned_signature() {
    let input = r#"core.module @test {
  !contract = clif.func_sig<(core.i32) -> (core.i32, core.i64), {tag = @native}>
  clif.func @id(%value: core.i32) -> core.i32 {
clif.return %value
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("native assembly should parse");
    let contract = ctx.type_alias_by_text("contract").unwrap();
    assert!(clif::FuncSig::from_type_ref(&ctx, contract).is_some());
    assert!(func::FuncSig::from_type_ref(&ctx, contract).is_none());
    let function = ctx
        .block(ctx.region(ctx.op_region(module, 0).unwrap()).blocks[0])
        .ops[0];
    let signature = ctx.op_attrs(function).get_type("type").unwrap();
    assert!(clif::FuncSig::from_type_ref(&ctx, signature).is_some());
    assert!(func::FuncSig::from_type_ref(&ctx, signature).is_none());
    assert_roundtrip(&ctx, module);

    let mut rejected = IrContext::new();
    let error = parse_module(
        &mut rejected,
        "core.module @test { !bad = clif.func_sig<() -> (), {num_inputs = 0}> }",
    )
    .expect_err("target count delimiters are constructor-owned");
    assert!(
        error.message.contains("`num_inputs` or `num_results`"),
        "{error}"
    );
}

#[test]
fn test_func_type_preserves_non_reserved_attributes() {
    let input = r#"core.module @test {
  !with_attr = func.func_sig<(core.i32) -> core.i64, {effect = core.nil}>
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("function type attribute should parse");
    let printed = module.display(&ctx).to_string();
    assert!(
        printed.contains("!with_attr = func.func_sig<(core.i32) -> core.i64, {effect = core.nil}>"),
        "{printed}"
    );
    assert!(!printed.contains("num_inputs"), "{printed}");
    assert!(!printed.contains("num_results"), "{printed}");
    assert_roundtrip(&ctx, module);
}

#[test]
fn test_func_type_rejects_reserved_textual_attributes() {
    for reserved in [func::NUM_INPUTS_ATTR, func::NUM_RESULTS_ATTR] {
        let input = format!(
            "core.module @test {{ !bad = func.func_sig<() -> core.nil, {{{reserved} = 0}}> }}"
        );
        let mut ctx = IrContext::new();
        let error = parse_module(&mut ctx, &input).expect_err("reserved key must be rejected");
        assert!(
            error.message.contains("`num_inputs` or `num_results`"),
            "{error}"
        );
    }
}

#[test]
fn test_func_type_accepts_multiple_results() {
    let input = "core.module @test {
  func.func @pair(%a: core.i32) -> (core.i32, core.i64) {
    %b = arith.constant 1 : core.i64
    func.return %a, %b
  }
  func.func @caller(%a: core.i32) -> core.i64 {
    %x, %y = func.call @pair(%a) : (core.i32) -> (core.i32, core.i64)
    func.return %y
  }
}";
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("multi-result signatures parse");
    let printed = module.display(&ctx).to_string();
    assert!(
        printed.contains("func.func @pair(%0: core.i32) -> (core.i32, core.i64)"),
        "{printed}"
    );
    assert!(
        printed.contains("%1, %2 = func.call @pair(%0) : (core.i32) -> (core.i32, core.i64)"),
        "{printed}"
    );
    let mut reparsed = IrContext::new();
    let again = parse_module(&mut reparsed, &printed).expect("printed form parses");
    assert_eq!(again.display(&reparsed).to_string(), printed);
}

#[test]
fn unregistered_func_sig_uses_generic_syntax() {
    let input = r#"core.module @test {
  !foreign = foreign.func_sig<core.i32, core.i64, core.i1, core.i32, {nested = core.array<core.i32>, num_inputs = 2, num_results = 2}>
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("generic storage should parse");
    let printed = module.display(&ctx).to_string();
    assert!(printed.contains("!foreign = foreign.func_sig<core.i32, core.i64, core.i1, core.i32, {nested = core.array<core.i32>, num_inputs = 2, num_results = 2}>"), "{printed}");
    assert_roundtrip(&ctx, module);

    let mut rejected = IrContext::new();
    parse_module(
        &mut rejected,
        "core.module @test { !bad = foreign.func_sig<(core.i32) -> core.i32> }",
    )
    .expect_err("a type without a registered format has no signature syntax");
}

#[test]
fn malformed_func_sig_storage_roundtrips_as_concrete_type() {
    let input = r#"core.module @test {
  !foreign = foreign.func_sig<core.i32, {num_inputs = 2, num_results = 1}>
  !shared = func.func_sig<core.i32, core.i32, core.i32, {num_inputs = 1, num_results = 1}>
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("raw storage should remain parseable");
    let printed = module.display(&ctx).to_string();
    assert!(
        printed
            .contains("!foreign = foreign.func_sig<core.i32, {num_inputs = 2, num_results = 1}>"),
        "{printed}"
    );
    assert!(
        printed.contains(
            "!shared = func.func_sig<core.i32, core.i32, core.i32, {num_inputs = 1, num_results = 1}>"
        ),
        "{printed}"
    );
    assert_roundtrip(&ctx, module);
}

#[test]
fn test_no_arrow_func_op_preserves_zero_results() {
    let input = r#"core.module @test {
  func.func @consume(%value: core.i32) {
func.return
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input).expect("Unit function should parse");
    let printed = module.display(&ctx).to_string();
    assert!(
        printed.contains("func.func @consume(%0: core.i32) {"),
        "{printed}"
    );

    let module_view = crate::rewrite::Module::new(&ctx, module).unwrap();
    let function = func::Func::from_op(&ctx, module_view.ops(&ctx)[0]).unwrap();
    let signature = func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
    assert_eq!(signature.inputs(&ctx).len(), 1);
    assert!(signature.is_resultless(&ctx));
    assert_roundtrip(&ctx, module);
}

#[test]
fn test_roundtrip_type_alias() {
    let input = r#"core.module @test {
  !marker = test.record<core.i32 {name = "ability_id"}, core.i32 {name = "prompt_tag"}, {name = "_Marker"}>

  func.func @foo(%0: core.array<!marker>) -> core.array<!marker> {
func.return %0
  }
}"#;
    let mut ctx = IrContext::new();
    let module_op = parse_module(&mut ctx, input).unwrap_or_else(|e| {
        panic!(
            "Parse failed at offset {}: {}\n\nInput:\n{}",
            e.offset, e.message, input
        );
    });
    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_programmatic_type_alias() {
    let mut ctx = IrContext::new();
    let loc = test_location(&mut ctx);
    let i32_ty = make_i32_type(&mut ctx);
    let tuple_ty = core::Tuple::params([i32_ty, i32_ty])
        .intern(&mut ctx)
        .as_type_ref();

    // Register alias
    ctx.register_type_alias(Symbol::new("point"), tuple_ty);

    // Build a function using that type
    let func_ty = make_func_type(&mut ctx, &[tuple_ty], tuple_ty);
    let entry = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: tuple_ty,
            attrs: Default::default(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let x = ctx.block_arg(entry, 0);
    let ret = func::Return::operands([x]).build(&mut ctx, loc);
    ctx.push_op(entry, ret.op_ref());
    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry],
        parent_op: None,
    });
    let f = func::Func::operands()
        .sym_name("identity")
        .r#type(func_ty)
        .regions(body)
        .build(&mut ctx, loc);
    let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

    let printed = module_op.display(&ctx).to_string();
    // Alias should appear in output
    assert!(
        printed.contains("!point = core.tuple<core.i32, core.i32>"),
        "Expected alias definition in output:\n{printed}",
    );
    assert!(
        printed.contains("!point)"),
        "Expected alias usage in output:\n{printed}",
    );
}

#[test]
fn test_nested_type_alias() {
    let input = r#"core.module @test {
  !inner = core.tuple<core.i32, core.i32>
  !outer = func.func_sig<() -> !inner>

  func.func @foo(%0: !inner) -> !inner {
func.return %0
  }
}"#;
    let mut ctx = IrContext::new();
    let module_op = parse_module(&mut ctx, input).unwrap_or_else(|e| {
        panic!(
            "Parse failed at offset {}: {}\n\nInput:\n{}",
            e.offset, e.message, input
        );
    });
    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_type_alias_no_alias_backward_compat() {
    // Existing IR without aliases should still round-trip
    let input = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
func.return %0
  }
}"#;
    let mut ctx = IrContext::new();
    let module_op = parse_module(&mut ctx, input).unwrap();
    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_quoted_type_alias_roundtrip() {
    let input = r#"core.module @test {
  !"test::MyStruct" = test.record<core.i32 {name = "x"}, core.i32 {name = "y"}, {name = "test::MyStruct"}>

  func.func @foo(%0: !"test::MyStruct") -> !"test::MyStruct" {
func.return %0
  }
}"#;
    let mut ctx = IrContext::new();
    let module_op = parse_module(&mut ctx, input).unwrap_or_else(|e| {
        panic!(
            "Parse failed at offset {}: {}\n\nInput:\n{}",
            e.offset, e.message, input
        );
    });
    assert_roundtrip(&ctx, module_op);
}

#[test]
fn test_roundtrip_bytes_const() {
    let input = r#"core.module @test {
  func.func @f() -> core.bytes {
%0 = "test.bytes_const"() {value = b"hello"} : core.bytes
func.return %0
  }
}"#;
    let mut ctx = IrContext::new();
    let module_op = parse_module(&mut ctx, input).expect("should parse bytes_const");
    assert_roundtrip(&ctx, module_op);
}

/// Printing a module whose operation holds arbitrary bytes, string, and
/// symbol reference attributes, then parsing it, recovers every value.
mod escaped_attribute_roundtrip {
    use proptest::prelude::*;

    use super::*;

    fn text() -> impl Strategy<Value = String> {
        let special = prop::sample::select(vec![
            '"', '\\', '\n', '\t', '\0', '\x01', '\u{85}', ':', '@', ' ', 'é', '😀',
        ]);
        prop_oneof![
            prop::collection::vec(prop_oneof![special, any::<char>()], 0..8)
                .prop_map(|chars| chars.into_iter().collect()),
            "[A-Za-z0-9_]{1,6}",
        ]
    }

    proptest! {
        #[test]
        fn module_attributes_round_trip(
            bytes in prop::collection::vec(any::<u8>(), 0..24),
            string in text(),
            callee in prop::collection::vec(text(), 1..3),
        ) {
            let mut ctx = IrContext::new();
            let loc = test_location(&mut ctx);
            let i32_ty = make_i32_type(&mut ctx);
            let callee = SymbolPath::new(callee.iter().map(String::as_str));
            let string_attr = ctx.string_attr(&string);
            let data = OperationDataBuilder::new(loc, "test.make")
                .attr("bytes", Attribute::Bytes(bytes.as_slice().into()))
                .attr("text", string_attr)
                .attr("callee", Attribute::SymbolRef(callee.clone()))
                .result(i32_ty)
                .build(&mut ctx);
            let op = ctx.create_op(data);
            let module_op = wrap_in_module(&mut ctx, loc, vec![op]);

            let printed = module_op.display(&ctx).to_string();
            let mut parsed_ctx = IrContext::new();
            let parsed = parse_module(&mut parsed_ctx, &printed).map_err(|e| {
                TestCaseError::fail(format!("parse failed: {}\n{printed}", e.message))
            })?;
            let body = parsed_ctx.op_region(parsed, 0).expect("module body");
            let block = parsed_ctx.region(body).blocks[0];
            let parsed_op = parsed_ctx.block(block).ops[0];
            let attrs = &parsed_ctx.op(parsed_op).attributes;
            prop_assert_eq!(
                attrs.view(&parsed_ctx).get("bytes"),
                Some(&Attribute::Bytes(bytes.as_slice().into()))
            );
            prop_assert_eq!(attrs.view(&parsed_ctx).get_str("text"), Some(string.as_str()));
            prop_assert_eq!(attrs.view(&parsed_ctx).get_symbol_ref("callee"), Some(&callee));
            prop_assert_eq!(parsed.display(&parsed_ctx).to_string(), printed);
        }
    }
}

// ================================================================
// Names resolved as they are read
// ================================================================

#[test]
fn a_value_may_be_used_before_its_block_argument_is_defined() {
    let input = r#"core.module @test {
  func.func @f() -> core.i32 {
      "test.use"(%x)
      "test.br"()[^next]
    ^next(%x: core.i32):
      func.return %x
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    let function = module.ops(&ctx)[0];
    let body = ctx.op_region(function, 0).unwrap();
    let [entry, next] = ctx.region(body).blocks[..] else {
        panic!("two blocks")
    };
    let used = ctx.block(entry).ops[0];
    let arg = ctx.block_arg(next, 0);
    assert_eq!(ctx.op_operands(used), [arg]);
    assert_eq!(ctx.uses(arg).len(), 2);
    assert!(validation::validate_use_chains(&ctx, module).is_ok());
    assert!(validation::validate_value_integrity(&ctx, module).is_ok());
    assert_roundtrip(&ctx, module.op());
}

#[test]
fn a_nested_region_may_use_a_later_block_argument_of_an_enclosing_one() {
    let input = r#"core.module @test {
  func.func @f() -> core.i32 {
      "test.wrap"() ({
        "test.use"(%x)
      })
      "test.br"()[^next]
    ^next(%x: core.i32):
      func.return %x
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    let function = module.ops(&ctx)[0];
    let body = ctx.op_region(function, 0).unwrap();
    let next = ctx.region(body).blocks[1];
    assert_eq!(ctx.uses(ctx.block_arg(next, 0)).len(), 2);
    assert!(validation::validate_use_chains(&ctx, module).is_ok());
    assert!(validation::validate_value_integrity(&ctx, module).is_ok());
    assert_roundtrip(&ctx, module.op());
}

#[test]
fn a_type_written_for_a_forward_reference_is_checked_at_its_definition() {
    let input = r#"core.module @test {
  func.func @f() {
      func.call @g(%x) : (core.i64) -> ()
      "test.br"()[^next]
    ^next(%x: core.i32):
      func.return
  }
}"#;
    let mut ctx = IrContext::new();
    let error = parse_module(&mut ctx, input).expect_err("the written type differs");
    assert!(
        error.message.contains(
            "operand '%x' of 'func.call' is written as core.i64, but its type is core.i32"
        ),
        "{error}"
    );
    assert_eq!(error.offset, input.find("core.i64").unwrap());
}

#[test]
fn resolution_errors_report_where_the_name_is_written() {
    for (input, at, expected) in [
        (
            "core.module @test {\n  func.func @f() -> core.i32 {\n    func.return %missing\n  }\n}",
            "%missing",
            "undefined value '%missing' in operation 'func.return'",
        ),
        (
            "core.module @test {\n  \"test.br\"()[^nowhere]\n}",
            "^nowhere",
            "undefined block '^nowhere' in successor list of 'test.br'",
        ),
        (
            "core.module @test {\n  %0 = \"test.a\"() : core.i32\n  %0 = \"test.b\"() : core.i32\n}",
            "%0 = \"test.b\"",
            "duplicate SSA name '0'",
        ),
        (
            "core.module @test {\n  %0 = \"test.a\"() : !missing\n}",
            "!missing",
            "undefined type alias '!missing'",
        ),
    ] {
        let mut ctx = IrContext::new();
        let error = parse_module(&mut ctx, input).expect_err(expected);
        assert!(error.message.contains(expected), "{error}");
        assert_eq!(error.offset, input.find(at).unwrap(), "{error}");
    }
}

#[test]
fn a_value_defined_only_in_a_sibling_region_stays_undefined() {
    let input = r#"core.module @test {
  "test.pair"() ({
    "test.use"(%x)
  }, {
    %x = "test.make"() : core.i32
  })
}"#;
    let mut ctx = IrContext::new();
    let error = parse_module(&mut ctx, input).expect_err("%x is not visible");
    assert!(error.message.contains("undefined value '%x'"), "{error}");
}

#[test]
fn forward_references_from_sibling_regions_meet_at_the_definition() {
    let input = r#"core.module @test {
  func.func @f() -> core.i32 {
      "test.pair"() ({
        "test.use"(%x)
      }, {
        "test.use"(%x)
      })
      "test.br"()[^next]
    ^next(%x: core.i32):
      func.return %x
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    let function = module.ops(&ctx)[0];
    let body = ctx.op_region(function, 0).unwrap();
    let next = ctx.region(body).blocks[1];
    assert_eq!(ctx.uses(ctx.block_arg(next, 0)).len(), 3);
    assert!(validation::validate_use_chains(&ctx, module).is_ok());
    assert!(validation::validate_value_integrity(&ctx, module).is_ok());
    assert_roundtrip(&ctx, module.op());
}

// ================================================================
// Types and attributes on their own
// ================================================================

mod types {
    use super::super::{AsmParser, parse_attribute, parse_type};
    use crate::context::IrContext;
    use crate::types::Attribute;

    #[test]
    fn parameters_carry_their_own_attributes() {
        let mut ctx = IrContext::new();
        let ty = parse_type(
            &mut ctx,
            "core.tuple<core.i32 {k = @v}, core.ptr, {tag = 1}>",
        )
        .expect("should parse");
        let data = ctx.get_type(ty);
        let param_attrs: Vec<_> = data
            .params_with_attrs(&ctx)
            .map(|(_, attrs)| attrs.len())
            .collect();
        assert_eq!(param_attrs, [1, 0]);
        assert!(ctx.type_attrs(ty).contains_key("tag"));
    }

    #[test]
    fn a_dictionary_after_a_type_is_not_the_types() {
        let mut ctx = IrContext::new();
        let mut p = AsmParser::new(&mut ctx, "core.tuple<core.nil, {tag = core.nil}> {");
        p.parse_type().expect("should parse");
        assert_eq!(p.peek(), "{");
    }

    #[test]
    fn malformed_types_are_errors() {
        for text in [
            "core.tuple<{tag = 1}, core.i32>",
            "foreign.func_sig<(core.i32) -> core.i32>",
            "core.tuple(core.i32)",
        ] {
            let mut ctx = IrContext::new();
            assert!(parse_type(&mut ctx, text).is_err(), "{text}");
        }
    }

    #[test]
    fn attribute_values_read_back() {
        let mut ctx = IrContext::new();
        for (text, expected) in [
            ("42", Attribute::Int(42)),
            ("3.25", Attribute::FloatBits(3.25f64.to_bits())),
            ("true", Attribute::Bool(true)),
            ("unit", Attribute::Unit),
        ] {
            assert_eq!(parse_attribute(&mut ctx, text).unwrap(), expected, "{text}");
        }
        let text = parse_attribute(&mut ctx, r#""hello""#).unwrap();
        assert_eq!(text, ctx.string_attr("hello"));
        let symbol = parse_attribute(&mut ctx, "@a::@b").unwrap();
        assert_eq!(
            symbol,
            Attribute::SymbolRef(crate::SymbolPath::new(["a", "b"]))
        );
    }

    #[test]
    fn locations_bytes_and_escapes_read_back() {
        let mut ctx = IrContext::new();
        let Attribute::Location(location) =
            parse_attribute(&mut ctx, r#"loc("main.trb" 3:7)"#).unwrap()
        else {
            panic!("a location")
        };
        assert_eq!(location.span, crate::location::Span::new(3, 7));
        assert_eq!(
            parse_attribute(&mut ctx, r#"b"a\x00\xff""#).unwrap(),
            Attribute::Bytes([b'a', 0, 0xff].into_iter().collect())
        );
        for (text, expected) in [
            (r#""\x41""#, "A"),
            (r#""\xZZ""#, "\\xZZ"),
            (r#""\q""#, "\\q"),
        ] {
            let value = parse_attribute(&mut ctx, text).unwrap();
            assert_eq!(value, ctx.string_attr(expected), "{text}");
        }
    }

    #[test]
    fn a_whole_attribute_text_rejects_trailing_input() {
        let mut ctx = IrContext::new();
        let error = parse_attribute(&mut ctx, "1 2").expect_err("trailing input");
        assert!(
            error.message.contains("trailing input after attribute"),
            "{error}"
        );
        assert_eq!(error.offset, 2);
    }

    #[test]
    fn optional_pieces_leave_the_text_as_it_was_when_absent() {
        let mut ctx = IrContext::new();
        let mut p = AsmParser::new(&mut ctx, "  %x");
        assert!(!p.parse_optional_keyword("attributes"));
        assert!(!p.parse_optional_punct("("));
        assert_eq!(p.parse_optional_symbol_name().unwrap(), None);
        assert_eq!(p.parse_optional_attr_dict().unwrap(), None);
        assert_eq!(p.offset(), 0);
    }
}
