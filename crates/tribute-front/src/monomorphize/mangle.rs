use std::fmt;

use tribute_core::fmt::joined_by;
use trunk_ir::Symbol;

use crate::ast::{Type, TypeKind};

/// Generate a mangled symbol for a specialized generic function or type.
///
/// Mangling rules:
/// - `identity + [Int]` → `identity$Int`
/// - `first + [Int, Text]` → `first$Int$Text`
/// - `map + [Int, Option(Int)]` → `map$Int$Option_Int_`
/// - Nested type args are wrapped with `_`: `Option(Int)` → `Option_Int_`
pub fn mangle_name(db: &dyn salsa::Database, base: Symbol, type_args: &[Type<'_>]) -> Symbol {
    let mut buf = String::new();
    base.with_str(|s| buf.push_str(s));
    for ty in type_args {
        buf.push('$');
        write_type_mangled(db, *ty, &mut buf).unwrap();
    }
    Symbol::from_dynamic(&buf)
}

fn write_type_mangled(
    db: &dyn salsa::Database,
    ty: Type<'_>,
    f: &mut impl fmt::Write,
) -> fmt::Result {
    match ty.kind(db) {
        TypeKind::Int => f.write_str("Int"),
        TypeKind::Nat => f.write_str("Nat"),
        TypeKind::Float => f.write_str("Float"),
        TypeKind::Bool => f.write_str("Bool"),
        TypeKind::Bytes => f.write_str("Bytes"),
        TypeKind::Rune => f.write_str("Rune"),
        TypeKind::Nil => f.write_str("Nil"),
        TypeKind::Never => f.write_str("Never"),
        TypeKind::Named { name, args } => {
            name.with_str(|s| f.write_str(s))?;
            if !args.is_empty() {
                write!(
                    f,
                    "_{}_",
                    joined_by("_", args, |arg, f| write_type_mangled(db, *arg, f))
                )?;
            }
            Ok(())
        }
        TypeKind::Func { params, result, .. } => {
            write!(
                f,
                "Fn_{}__",
                joined_by("_", params, |p, f| write_type_mangled(db, *p, f))
            )?;
            write_type_mangled(db, *result, f)?;
            f.write_str("_")
        }
        TypeKind::Tuple(elems) => {
            write!(
                f,
                "Tup_{}_",
                joined_by("_", elems, |e, f| write_type_mangled(db, *e, f))
            )
        }
        TypeKind::BoundVar { index } => write!(f, "T{index}"),
        TypeKind::UniVar { .. } | TypeKind::App { .. } | TypeKind::Continuation { .. } => {
            panic!("mangle_name requires fully-resolved concrete types");
        }
        TypeKind::Error => f.write_str("error"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    #[test]
    fn test_single_primitive() {
        let db = TestDb::default();
        let base = Symbol::new("identity");
        let int_ty = Type::new(&db, TypeKind::Int);
        let result = mangle_name(&db, base, &[int_ty]);
        assert_eq!(result.to_string(), "identity$Int");
    }

    #[test]
    fn test_multiple_primitives() {
        let db = TestDb::default();
        let base = Symbol::new("first");
        let int_ty = Type::new(&db, TypeKind::Int);
        let float_ty = Type::new(&db, TypeKind::Float);
        let result = mangle_name(&db, base, &[int_ty, float_ty]);
        assert_eq!(result.to_string(), "first$Int$Float");
    }

    #[test]
    fn test_named_type_no_args() {
        let db = TestDb::default();
        let base = Symbol::new("wrap");
        let text_ty = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Text"),
                args: vec![],
            },
        );
        let result = mangle_name(&db, base, &[text_ty]);
        assert_eq!(result.to_string(), "wrap$Text");
    }

    #[test]
    fn test_named_type_with_args() {
        let db = TestDb::default();
        let base = Symbol::new("map");
        let int_ty = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );
        let result = mangle_name(&db, base, &[int_ty, option_int]);
        assert_eq!(result.to_string(), "map$Int$Option_Int_");
    }

    #[test]
    fn test_nested_named_types() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let int_ty = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );
        let list_option_int = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![option_int],
            },
        );
        let result = mangle_name(&db, base, &[list_option_int]);
        assert_eq!(result.to_string(), "f$List_Option_Int__");
    }

    #[test]
    fn test_function_type() {
        let db = TestDb::default();
        let base = Symbol::new("apply");
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: bool_ty,
                effect: crate::ast::EffectRow::new(&db, vec![], None),
            },
        );
        let result = mangle_name(&db, base, &[func_ty]);
        assert_eq!(result.to_string(), "apply$Fn_Int__Bool_");
    }

    #[test]
    fn test_tuple_type() {
        let db = TestDb::default();
        let base = Symbol::new("swap");
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let tup_ty = Type::new(&db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        let result = mangle_name(&db, base, &[tup_ty]);
        assert_eq!(result.to_string(), "swap$Tup_Int_Bool_");
    }

    #[test]
    fn test_all_primitives() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let types: Vec<(Type<'_>, &str)> = vec![
            (Type::new(&db, TypeKind::Int), "Int"),
            (Type::new(&db, TypeKind::Nat), "Nat"),
            (Type::new(&db, TypeKind::Float), "Float"),
            (Type::new(&db, TypeKind::Bool), "Bool"),
            (Type::new(&db, TypeKind::Bytes), "Bytes"),
            (Type::new(&db, TypeKind::Rune), "Rune"),
            (Type::new(&db, TypeKind::Nil), "Nil"),
            (Type::new(&db, TypeKind::Never), "Never"),
        ];
        for (ty, expected_suffix) in types {
            let result = mangle_name(&db, base, &[ty]);
            assert_eq!(result.to_string(), format!("f${expected_suffix}"));
        }
    }

    #[test]
    fn test_empty_type_args() {
        let db = TestDb::default();
        let base = Symbol::new("main");
        let result = mangle_name(&db, base, &[]);
        assert_eq!(result.to_string(), "main");
    }

    #[test]
    fn test_bound_var() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let bv = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let result = mangle_name(&db, base, &[bv]);
        assert_eq!(result.to_string(), "f$T0");
    }

    #[test]
    fn test_error_type() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let err_ty = Type::new(&db, TypeKind::Error);
        let result = mangle_name(&db, base, &[err_ty]);
        assert_eq!(result.to_string(), "f$error");
    }

    #[test]
    #[should_panic(expected = "mangle_name requires fully-resolved concrete types")]
    fn test_univar_panics() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let univar_id = crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(0), 0);
        let ty = Type::new(&db, TypeKind::UniVar { id: univar_id });
        mangle_name(&db, base, &[ty]);
    }
}
