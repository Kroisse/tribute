//! Lower arith dialect operations to clif dialect.
//!
//! This pass converts arithmetic operations to their Cranelift equivalents:
//! - `arith.const` -> `clif.iconst` / `clif.f32const` / `clif.f64const`
//! - `arith.{add,sub,mul,div,rem}` -> `clif.{iadd,isub,imul,sdiv,srem}` / `clif.{fadd,fsub,fmul,fdiv}`
//! - `arith.cmp_*` -> `clif.icmp` / `clif.fcmp` with `cond` attribute
//! - `arith.neg` -> `clif.ineg` / `clif.fneg` (native support, no expansion needed)
//! - `arith.{and,or,xor,shl,shr,shru}` -> `clif.{band,bor,bxor,ishl,sshr,ushr}`
//! - `arith.{cast,trunc,extend,convert}` -> `clif.{ireduce,sextend,fpromote,fdemote,fcvt_*}`

use tracing::warn;

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{arith, clif, core};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol, Type};

/// Lower arith dialect to clif dialect.
///
/// Returns an error if any `arith.*` operations remain after conversion.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Result<Module<'db>, ConversionError> {
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("arith");

    Ok(PatternApplicator::new(type_converter)
        .add_pattern(ArithConstPattern)
        .add_pattern(ArithBinOpPattern)
        .add_pattern(ArithCmpPattern)
        .add_pattern(ArithNegPattern)
        .add_pattern(ArithBitwisePattern)
        .add_pattern(ArithConversionPattern)
        .apply(db, module, target)?
        .module)
}

/// Classify a type into integer vs float category.
/// Unlike wasm, clif ops are size-agnostic (iadd works for i8/i16/i32/i64).
fn type_category<'db>(db: &'db dyn salsa::Database, ty: Option<Type<'db>>) -> &'static str {
    match ty {
        Some(t) => {
            let name = t.name(db);
            if name == Symbol::new("i1")
                || name == Symbol::new("i8")
                || name == Symbol::new("i16")
                || name == Symbol::new("i32")
                || name == Symbol::new("i64")
                || name == Symbol::new("int")
                || name == Symbol::new("nat")
                || name == Symbol::new("bool")
            {
                "int"
            } else if name == Symbol::new("f32") {
                "f32"
            } else if name == Symbol::new("f64") {
                "f64"
            } else if name == Symbol::new("nil") {
                "nil"
            } else {
                #[cfg(debug_assertions)]
                warn!(
                    "Unknown type '{}' in arith_to_clif, defaulting to int",
                    name
                );
                "int"
            }
        }
        None => {
            #[cfg(debug_assertions)]
            warn!("No type in arith_to_clif, defaulting to int");
            "int"
        }
    }
}

/// Pattern for `arith.const` -> `clif.iconst` / `clif.f32const` / `clif.f64const`
struct ArithConstPattern;

impl<'db> RewritePattern<'db> for ArithConstPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(const_op) = arith::Const::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };

        let category = type_category(db, Some(result_ty));
        if category == "nil" {
            // Nil constants have no runtime representation in Cranelift.
            // Use iconst 0 with the nil type to maintain SSA form.
            let new_op = clif::iconst(db, op.location(db), result_ty, 0);
            return RewriteResult::Replace(new_op.as_operation());
        }

        let location = op.location(db);
        let value = const_op.value(db).clone();

        let new_op = match category {
            "f32" => if let Attribute::FloatBits(v) = value {
                clif::f32const(db, location, result_ty, f32::from_bits(v as u32))
            } else {
                clif::f32const(db, location, result_ty, 0.0)
            }
            .as_operation(),
            "f64" => if let Attribute::FloatBits(v) = value {
                clif::f64const(db, location, result_ty, f64::from_bits(v))
            } else {
                clif::f64const(db, location, result_ty, 0.0)
            }
            .as_operation(),
            _ => {
                // All integer types (i1, i8, i16, i32, i64, int, nat, bool)
                if let Attribute::IntBits(v) = value {
                    clif::iconst(db, location, result_ty, v as i64).as_operation()
                } else {
                    clif::iconst(db, location, result_ty, 0).as_operation()
                }
            }
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.{add,sub,mul,div,rem}` -> clif binary ops
struct ArithBinOpPattern;

impl<'db> RewritePattern<'db> for ArithBinOpPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_binop = name == arith::ADD()
            || name == arith::SUB()
            || name == arith::MUL()
            || name == arith::DIV()
            || name == arith::REM();

        if !is_binop {
            return RewriteResult::Unchanged;
        }

        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };
        let operands = op.operands(db);
        let (Some(lhs), Some(rhs)) = (operands.first().copied(), operands.get(1).copied()) else {
            return RewriteResult::Unchanged;
        };
        let category = type_category(db, Some(result_ty));
        let location = op.location(db);

        let new_op = if name == arith::ADD() {
            match category {
                "f32" | "f64" => clif::fadd(db, location, lhs, rhs, result_ty).as_operation(),
                _ => clif::iadd(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::SUB() {
            match category {
                "f32" | "f64" => clif::fsub(db, location, lhs, rhs, result_ty).as_operation(),
                _ => clif::isub(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::MUL() {
            match category {
                "f32" | "f64" => clif::fmul(db, location, lhs, rhs, result_ty).as_operation(),
                _ => clif::imul(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::DIV() {
            match category {
                "f32" | "f64" => clif::fdiv(db, location, lhs, rhs, result_ty).as_operation(),
                _ => clif::sdiv(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::REM() {
            // Only integer remainder in Cranelift (no float remainder)
            clif::srem(db, location, lhs, rhs, result_ty).as_operation()
        } else {
            return RewriteResult::Unchanged;
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.cmp_*` -> `clif.icmp` / `clif.fcmp` with `cond` attribute
struct ArithCmpPattern;

impl<'db> RewritePattern<'db> for ArithCmpPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_cmp = name == arith::CMP_EQ()
            || name == arith::CMP_NE()
            || name == arith::CMP_LT()
            || name == arith::CMP_LE()
            || name == arith::CMP_GT()
            || name == arith::CMP_GE();

        if !is_cmp {
            return RewriteResult::Unchanged;
        }

        let operand_ty = adaptor.operand_type(0);
        let category = type_category(db, operand_ty);

        let result_ty =
            op.results(db).first().copied().unwrap_or_else(|| {
                panic!("arith cmp missing result type at {:?}", op.location(db))
            });
        let location = op.location(db);
        let operands = op.operands(db);
        let lhs = operands[0];
        let rhs = operands[1];

        let is_float = matches!(category, "f32" | "f64");

        let (cond, use_fcmp) = if name == arith::CMP_EQ() {
            ("eq", is_float)
        } else if name == arith::CMP_NE() {
            ("ne", is_float)
        } else if name == arith::CMP_LT() {
            if is_float {
                ("lt", true)
            } else {
                ("slt", false)
            }
        } else if name == arith::CMP_LE() {
            if is_float {
                ("le", true)
            } else {
                ("sle", false)
            }
        } else if name == arith::CMP_GT() {
            if is_float {
                ("gt", true)
            } else {
                ("sgt", false)
            }
        } else if name == arith::CMP_GE() {
            if is_float {
                ("ge", true)
            } else {
                ("sge", false)
            }
        } else {
            return RewriteResult::Unchanged;
        };

        let cond_sym = Symbol::new(cond);
        let new_op = if use_fcmp {
            clif::fcmp(db, location, lhs, rhs, result_ty, cond_sym).as_operation()
        } else {
            clif::icmp(db, location, lhs, rhs, result_ty, cond_sym).as_operation()
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.neg` -> `clif.ineg` / `clif.fneg`
///
/// Unlike wasm, Cranelift has native integer negation (ineg),
/// so no expansion to `0 - x` is needed.
struct ArithNegPattern;

impl<'db> RewritePattern<'db> for ArithNegPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(neg_op) = arith::Neg::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let result_ty = op.results(db).first().copied();
        let category = type_category(db, result_ty);
        let location = op.location(db);
        let operand = neg_op.operand(db);

        let new_op = match category {
            "f32" => {
                let ty = result_ty.unwrap_or_else(|| core::F32::new(db).as_type());
                clif::fneg(db, location, operand, ty).as_operation()
            }
            "f64" => {
                let ty = result_ty.unwrap_or_else(|| core::F64::new(db).as_type());
                clif::fneg(db, location, operand, ty).as_operation()
            }
            _ => {
                let ty = result_ty.unwrap_or_else(|| core::I64::new(db).as_type());
                clif::ineg(db, location, operand, ty).as_operation()
            }
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.{and,or,xor,shl,shr,shru}` -> clif bitwise ops
struct ArithBitwisePattern;

impl<'db> RewritePattern<'db> for ArithBitwisePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_bitwise = name == arith::AND()
            || name == arith::OR()
            || name == arith::XOR()
            || name == arith::SHL()
            || name == arith::SHR()
            || name == arith::SHRU();

        if !is_bitwise {
            return RewriteResult::Unchanged;
        }

        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };
        let operands = op.operands(db);
        let (Some(lhs), Some(rhs)) = (operands.first().copied(), operands.get(1).copied()) else {
            return RewriteResult::Unchanged;
        };
        let location = op.location(db);

        // clif bitwise ops are type-width agnostic (band works for all integer widths)
        let new_op = if name == arith::AND() {
            clif::band(db, location, lhs, rhs, result_ty).as_operation()
        } else if name == arith::OR() {
            clif::bor(db, location, lhs, rhs, result_ty).as_operation()
        } else if name == arith::XOR() {
            clif::bxor(db, location, lhs, rhs, result_ty).as_operation()
        } else if name == arith::SHL() {
            clif::ishl(db, location, lhs, rhs, result_ty).as_operation()
        } else if name == arith::SHR() {
            clif::sshr(db, location, lhs, rhs, result_ty).as_operation()
        } else if name == arith::SHRU() {
            clif::ushr(db, location, lhs, rhs, result_ty).as_operation()
        } else {
            return RewriteResult::Unchanged;
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.{cast,trunc,extend,convert}` -> clif conversion ops
struct ArithConversionPattern;

impl<'db> RewritePattern<'db> for ArithConversionPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_conv = name == arith::CAST()
            || name == arith::TRUNC()
            || name == arith::EXTEND()
            || name == arith::CONVERT();

        if !is_conv {
            return RewriteResult::Unchanged;
        }

        let src_ty = adaptor.operand_type(0);
        let src_cat = type_category(db, src_ty);

        let dst_ty = op.results(db).first().copied().unwrap_or_else(|| {
            panic!(
                "arith conversion missing result type at {:?}",
                op.location(db)
            )
        });
        let dst_cat = type_category(db, Some(dst_ty));

        let location = op.location(db);
        let operand = op.operands(db)[0];

        let new_op = if name == arith::CAST() {
            // cast: integer sign extension/truncation
            match (src_cat, dst_cat) {
                ("int", "int") => {
                    // Determine direction from actual type widths
                    let src_name = src_ty.map(|t| t.name(db));
                    let dst_name = dst_ty.name(db);
                    if is_wider_int(dst_name, src_name) {
                        clif::sextend(db, location, operand, dst_ty).as_operation()
                    } else {
                        clif::ireduce(db, location, operand, dst_ty).as_operation()
                    }
                }
                _ => return RewriteResult::Unchanged,
            }
        } else if name == arith::TRUNC() {
            match (src_cat, dst_cat) {
                ("f32" | "f64", "int") => {
                    clif::fcvt_to_sint(db, location, operand, dst_ty).as_operation()
                }
                ("int", "int") => clif::ireduce(db, location, operand, dst_ty).as_operation(),
                _ => return RewriteResult::Unchanged,
            }
        } else if name == arith::EXTEND() {
            match (src_cat, dst_cat) {
                ("int", "int") => clif::sextend(db, location, operand, dst_ty).as_operation(),
                ("f32", "f64") => clif::fpromote(db, location, operand, dst_ty).as_operation(),
                _ => return RewriteResult::Unchanged,
            }
        } else if name == arith::CONVERT() {
            match (src_cat, dst_cat) {
                // int to float
                ("int", "f32" | "f64") => {
                    clif::fcvt_from_sint(db, location, operand, dst_ty).as_operation()
                }
                // float to int
                ("f32" | "f64", "int") => {
                    clif::fcvt_to_sint(db, location, operand, dst_ty).as_operation()
                }
                // float to float
                ("f32", "f64") => clif::fpromote(db, location, operand, dst_ty).as_operation(),
                ("f64", "f32") => clif::fdemote(db, location, operand, dst_ty).as_operation(),
                _ => return RewriteResult::Unchanged,
            }
        } else {
            return RewriteResult::Unchanged;
        };

        RewriteResult::Replace(new_op)
    }
}

/// Check if `dst` is a wider integer type than `src`.
fn is_wider_int(dst: Symbol, src: Option<Symbol>) -> bool {
    let width = |s: Symbol| -> u8 {
        if s == Symbol::new("i64") {
            64
        } else if s == Symbol::new("i32") || s == Symbol::new("int") || s == Symbol::new("nat") {
            32
        } else if s == Symbol::new("i16") {
            16
        } else if s == Symbol::new("i8") || s == Symbol::new("bool") || s == Symbol::new("i1") {
            8
        } else {
            32 // default
        }
    };

    match src {
        Some(s) => width(dst) > width(s),
        None => true, // assume extension if source unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockId, DialectOp, Location, PathId, Region, Span, idvec};

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Format module operations for snapshot testing
    fn format_module_ops(db: &dyn salsa::Database, module: &Module<'_>) -> String {
        let body = module.body(db);
        let ops = &body.blocks(db)[0].operations(db);
        ops.iter()
            .map(|op| format_op(db, op))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_op(db: &dyn salsa::Database, op: &trunk_ir::Operation<'_>) -> String {
        let name = op.full_name(db);
        let operands = op.operands(db);
        let results = op.results(db);
        let attrs = op.attributes(db);

        let mut parts = vec![name];

        for (key, attr) in attrs.iter() {
            if *key == "value" {
                parts.push(format!("value={}", format_attr(attr)));
            } else if *key == "cond"
                && let Attribute::Symbol(s) = attr
            {
                parts.push(format!("cond={}", s));
            }
        }

        if !operands.is_empty() {
            parts.push(format!("operands={}", operands.len()));
        }

        if !results.is_empty() {
            let result_types: Vec<_> = results.iter().map(|t| t.name(db).to_string()).collect();
            parts.push(format!("-> {}", result_types.join(", ")));
        }

        parts.join(" ")
    }

    fn format_attr(attr: &trunk_ir::Attribute<'_>) -> String {
        match attr {
            trunk_ir::Attribute::IntBits(v) => format!("{}", *v as i64),
            trunk_ir::Attribute::FloatBits(v) => format!("{}", f64::from_bits(*v)),
            _ => "...".to_string(),
        }
    }

    #[salsa::tracked]
    fn make_arith_add_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let const1 = arith::Const::i32(db, location, 1);
        let const2 = arith::Const::i32(db, location, 2);
        let add = arith::add(db, location, const1.result(db), const2.result(db), i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                const1.as_operation(),
                const2.as_operation(),
                add.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn format_lowered_module<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> String {
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
        format_module_ops(db, &lowered)
    }

    #[salsa_test]
    fn test_arith_const_and_add_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_arith_add_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=1 -> i32
        clif.iconst value=2 -> i32
        clif.iadd operands=2 -> i32
        ");
    }

    #[salsa::tracked]
    fn make_convert_i32_to_f64_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let f64_ty = core::F64::new(db).as_type();

        let int_const = arith::Const::i32(db, location, 42);
        let convert = arith::convert(db, location, int_const.result(db), f64_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![int_const.as_operation(), convert.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_convert_i32_to_f64(db: &salsa::DatabaseImpl) {
        let module = make_convert_i32_to_f64_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=42 -> i32
        clif.fcvt_from_sint operands=1 -> f64
        ");
    }

    #[salsa::tracked]
    fn make_extend_i32_to_i64_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        let int_const = arith::Const::i32(db, location, 100);
        let extend = arith::extend(db, location, int_const.result(db), i64_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![int_const.as_operation(), extend.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_extend_i32_to_i64(db: &salsa::DatabaseImpl) {
        let module = make_extend_i32_to_i64_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=100 -> i32
        clif.sextend operands=1 -> i64
        ");
    }

    #[salsa::tracked]
    fn make_trunc_f64_to_i32_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let float_const = arith::Const::f64(db, location, 3.5);
        let trunc = arith::trunc(db, location, float_const.result(db), i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![float_const.as_operation(), trunc.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_trunc_f64_to_i32(db: &salsa::DatabaseImpl) {
        let module = make_trunc_f64_to_i32_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.f64const value=3.5 -> f64
        clif.fcvt_to_sint operands=1 -> i32
        ");
    }

    #[salsa::tracked]
    fn make_neg_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let const_op = arith::Const::i32(db, location, 5);
        let neg = arith::neg(db, location, const_op.result(db), i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![const_op.as_operation(), neg.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_neg_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_neg_module(db);
        let formatted = format_lowered_module(db, module);

        // Unlike wasm, clif has native ineg (no 0-x expansion)
        assert_snapshot!(formatted, @r"
        clif.iconst value=5 -> i32
        clif.ineg operands=1 -> i32
        ");
    }

    #[salsa::tracked]
    fn make_cmp_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let const1 = arith::Const::i32(db, location, 1);
        let const2 = arith::Const::i32(db, location, 2);
        let cmp = arith::cmp_lt(db, location, const1.result(db), const2.result(db), i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                const1.as_operation(),
                const2.as_operation(),
                cmp.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_cmp_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_cmp_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=1 -> i32
        clif.iconst value=2 -> i32
        clif.icmp cond=slt operands=2 -> i32
        ");
    }
}
