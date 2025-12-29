//! Type pretty-printing for display in IDE hovers.
//!
//! This module provides a thin wrapper around `trunk_ir::type_interface::Printable`
//! for use in the LSP server.

use trunk_ir::Type;
use trunk_ir::type_interface;

/// Pretty-print a type to a user-friendly string.
pub fn print_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    type_interface::print_type(db, ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tribute_ir::dialect::ty::{self, Int};
    use trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func, Nil};
    use trunk_ir::{IdVec, Symbol, idvec};

    #[salsa_test]
    fn test_print_basic_types(db: &salsa::DatabaseImpl) {
        // Int (arbitrary precision)
        let int_ty = *Int::new(db);
        assert_eq!(print_type(db, int_ty), "Int");

        // Nil
        let nil_ty = *Nil::new(db);
        assert_eq!(print_type(db, nil_ty), "()");
    }

    #[salsa_test]
    fn test_print_function_type(db: &salsa::DatabaseImpl) {
        let int_ty = *Int::new(db);

        // fn(Int, Int) -> Int
        let func_ty = *Func::new(db, idvec![int_ty, int_ty], int_ty);
        assert_eq!(print_type(db, func_ty), "fn(Int, Int) -> Int");

        // fn() -> ()
        let nil_ty = *Nil::new(db);
        let unit_func = *Func::new(db, IdVec::new(), nil_ty);
        assert_eq!(print_type(db, unit_func), "fn() -> ()");
    }

    #[salsa_test]
    fn test_print_effect_row(db: &salsa::DatabaseImpl) {
        // Empty row
        let empty = *EffectRowType::empty(db);
        assert_eq!(print_type(db, empty), "{}");

        // Row with ability
        let console = *AbilityRefType::simple(db, Symbol::new("Console"));
        let row = *EffectRowType::concrete(db, idvec![console]);
        assert_eq!(print_type(db, row), "{Console}");

        // Row with tail variable
        let open_row = *EffectRowType::with_tail(db, idvec![console], 4); // 'e' = id 4
        assert_eq!(print_type(db, open_row), "{Console | e}");
    }

    #[salsa_test]
    fn test_print_type_var(db: &salsa::DatabaseImpl) {
        let var_a = ty::var_with_id(db, 0);
        assert_eq!(print_type(db, var_a), "a");

        let var_z = ty::var_with_id(db, 25);
        assert_eq!(print_type(db, var_z), "z");

        let var_t0 = ty::var_with_id(db, 26);
        assert_eq!(print_type(db, var_t0), "t0");
    }
}
