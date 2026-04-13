use std::fmt::Write;

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
        write_type_mangled(db, *ty, &mut buf);
    }
    Symbol::from_dynamic(&buf)
}

fn write_type_mangled(db: &dyn salsa::Database, ty: Type<'_>, buf: &mut String) {
    match ty.kind(db) {
        TypeKind::Int => buf.push_str("Int"),
        TypeKind::Nat => buf.push_str("Nat"),
        TypeKind::Float => buf.push_str("Float"),
        TypeKind::Bool => buf.push_str("Bool"),
        TypeKind::Bytes => buf.push_str("Bytes"),
        TypeKind::Rune => buf.push_str("Rune"),
        TypeKind::Nil => buf.push_str("Nil"),
        TypeKind::Never => buf.push_str("Never"),
        TypeKind::Named { name, args } => {
            name.with_str(|s| buf.push_str(s));
            if !args.is_empty() {
                buf.push('_');
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        buf.push('_');
                    }
                    write_type_mangled(db, *arg, buf);
                }
                buf.push('_');
            }
        }
        TypeKind::Func { params, result, .. } => {
            buf.push_str("Fn_");
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    buf.push('_');
                }
                write_type_mangled(db, *p, buf);
            }
            buf.push_str("__");
            write_type_mangled(db, *result, buf);
            buf.push('_');
        }
        TypeKind::Tuple(elems) => {
            buf.push_str("Tup_");
            for (i, e) in elems.iter().enumerate() {
                if i > 0 {
                    buf.push('_');
                }
                write_type_mangled(db, *e, buf);
            }
            buf.push('_');
        }
        TypeKind::BoundVar { index } => {
            let _ = write!(buf, "T{index}");
        }
        TypeKind::UniVar { .. } | TypeKind::App { .. } | TypeKind::Continuation { .. } => {
            buf.push_str("unknown");
        }
        TypeKind::Error => buf.push_str("error"),
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
}
