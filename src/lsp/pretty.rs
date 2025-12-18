//! Type pretty-printing for display in IDE hovers.
//!
//! This module provides a thin wrapper around `tribute_trunk_ir::type_interface::Printable`
//! for use in the LSP server.

use tribute_trunk_ir::Type;
use tribute_trunk_ir::type_interface;

/// Pretty-print a type to a user-friendly string.
pub fn print_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    type_interface::print_type(db, ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::prelude::*;
    use tribute_core::TributeDatabaseImpl;
    use tribute_trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func, I64, Nil};
    use tribute_trunk_ir::{IdVec, Symbol, idvec};

    #[test]
    fn test_print_basic_types() {
        TributeDatabaseImpl::default().attach(|db| {
            // Int (i64)
            let int_ty = *I64::new(db);
            assert_eq!(print_type(db, int_ty), "Int");

            // Nil
            let nil_ty = *Nil::new(db);
            assert_eq!(print_type(db, nil_ty), "()");
        });
    }

    #[test]
    fn test_print_function_type() {
        TributeDatabaseImpl::default().attach(|db| {
            let int_ty = *I64::new(db);

            // fn(Int, Int) -> Int
            let func_ty = *Func::new(db, idvec![int_ty, int_ty], int_ty);
            assert_eq!(print_type(db, func_ty), "fn(Int, Int) -> Int");

            // fn() -> ()
            let nil_ty = *Nil::new(db);
            let unit_func = *Func::new(db, IdVec::new(), nil_ty);
            assert_eq!(print_type(db, unit_func), "fn() -> ()");
        });
    }

    #[test]
    fn test_print_effect_row() {
        TributeDatabaseImpl::default().attach(|db| {
            // Empty row
            let empty = *EffectRowType::empty(db);
            assert_eq!(print_type(db, empty), "{}");

            // Row with ability
            let console = *AbilityRefType::simple(db, Symbol::new(db, "Console"));
            let row = *EffectRowType::concrete(db, idvec![console]);
            assert_eq!(print_type(db, row), "{Console}");

            // Row with tail variable
            let open_row = *EffectRowType::with_tail(db, idvec![console], 4); // 'e' = id 4
            assert_eq!(print_type(db, open_row), "{Console | e}");
        });
    }

    #[test]
    fn test_print_type_var() {
        TributeDatabaseImpl::default().attach(|db| {
            let var_a = tribute_trunk_ir::dialect::ty::var_with_id(db, 0);
            assert_eq!(print_type(db, var_a), "a");

            let var_z = tribute_trunk_ir::dialect::ty::var_with_id(db, 25);
            assert_eq!(print_type(db, var_z), "z");

            let var_t0 = tribute_trunk_ir::dialect::ty::var_with_id(db, 26);
            assert_eq!(print_type(db, var_t0), "t0");
        });
    }
}
