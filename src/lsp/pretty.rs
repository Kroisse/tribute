//! Type pretty-printing for display in IDE hovers.
//!
//! Converts internal IR types to user-friendly notation like:
//! - `Int`, `Float`, `String`, `Bool`
//! - `fn(Int, Int) -> Int`
//! - `fn(a) ->{Console} a`
//! - `List(a)`
//! - `{State(Int), Console | e}`

use std::fmt::{self, Display, Formatter, Write};

use tribute_trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func};
use tribute_trunk_ir::{Attribute, DialectType, Type};

/// Pretty-print a type to a user-friendly string.
pub fn print_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    TypeDisplay::new(db, ty).to_string()
}

/// Wrapper for displaying a type with database context.
pub struct TypeDisplay<'a, 'db> {
    db: &'a dyn salsa::Database,
    ty: Type<'db>,
}

impl<'a, 'db> TypeDisplay<'a, 'db>
where
    'a: 'db,
{
    /// Create a new type display wrapper.
    pub fn new(db: &'a dyn salsa::Database, ty: Type<'db>) -> Self {
        Self { db, ty }
    }
}

impl Display for TypeDisplay<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        TypePrinter::new(self.db).fmt(f, self.ty)
    }
}

/// Type printer with database context.
struct TypePrinter<'a> {
    db: &'a dyn salsa::Database,
}

impl<'a> TypePrinter<'a> {
    fn new(db: &'a dyn salsa::Database) -> Self {
        Self { db }
    }

    fn fmt(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        let dialect = ty.dialect(self.db).text(self.db);
        let name = ty.name(self.db).text(self.db);

        match (dialect, name) {
            // Core types with friendly names
            ("core", "nil") => f.write_str("()"),
            ("core", "never") => f.write_str("Never"),
            ("core", "string") => f.write_str("String"),
            ("core", "bytes") => f.write_str("Bytes"),
            ("core", "ptr") => f.write_str("Ptr"),

            // Function type (must come before float pattern since "func" starts with 'f')
            ("core", "func") => self.fmt_function(f, ty),

            // Integer types
            ("core", name) if name.starts_with('i') => match &name[1..] {
                "1" => f.write_str("Bool"),
                "64" => f.write_str("Int"),
                bits => write!(f, "I{bits}"),
            },

            // Float types
            ("core", name) if name.starts_with('f') => match &name[1..] {
                "64" => f.write_str("Float"),
                bits => write!(f, "F{bits}"),
            },

            // Tuple type
            ("core", "tuple") => self.fmt_tuple(f, ty),

            // Array type
            ("core", "array") => self.fmt_array(f, ty),

            // Effect row
            ("core", "effect_row") => self.fmt_effect_row(f, ty),

            // Ability reference
            ("core", "ability_ref") => self.fmt_ability_ref(f, ty),

            // Reference type
            ("core", "ref_") => self.fmt_ref(f, ty),

            // Type variable
            ("type", "var") => self.fmt_type_var(f, ty),

            // Type error
            ("type", "error") => f.write_str("<error>"),

            // Source-level unresolved types
            ("src", "unresolved_type") => self.fmt_unresolved(f, ty),

            // Generic fallback with params
            _ => self.fmt_generic(f, dialect, name, ty),
        }
    }

    fn fmt_function(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        let Some(func) = Func::from_type(self.db, ty) else {
            return f.write_str("fn(?)");
        };

        let params = func.params(self.db);
        let result = func.result(self.db);
        let effect = func.effect(self.db);

        // Format parameters
        f.write_str("fn(")?;
        for (i, &p) in params.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            self.fmt(f, p)?;
        }
        f.write_char(')')?;

        // Format arrow with effect
        match effect {
            Some(eff) => {
                if let Some(row) = EffectRowType::from_type(self.db, eff) {
                    if row.is_empty(self.db) {
                        f.write_str(" -> ")?;
                    } else {
                        f.write_str(" ->{")?;
                        self.fmt_effect_row_inner(f, &row)?;
                        f.write_str("} ")?;
                    }
                } else {
                    f.write_str(" -> ")?;
                }
            }
            None => f.write_str(" -> ")?,
        }

        self.fmt(f, result)
    }

    fn fmt_tuple(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        let params = ty.params(self.db);
        if params.is_empty() {
            return f.write_str("()");
        }

        // Flatten cons cells into a list
        let mut elements = Vec::new();
        let mut current = ty;

        while current.is_dialect(self.db, "core", "tuple") {
            let params = current.params(self.db);
            if params.len() >= 2 {
                elements.push(params[0]); // head
                current = params[1]; // tail
            } else {
                break;
            }
        }

        // Check if tail is nil (complete tuple)
        let has_tail = !current.is_dialect(self.db, "core", "nil");

        f.write_char('(')?;
        for (i, &elem) in elements.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            self.fmt(f, elem)?;
        }
        if has_tail {
            if !elements.is_empty() {
                f.write_str(", ")?;
            }
            self.fmt(f, current)?;
        }
        f.write_char(')')
    }

    fn fmt_array(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        let params = ty.params(self.db);
        if let Some(&elem) = params.first() {
            f.write_str("Array(")?;
            self.fmt(f, elem)?;
            f.write_char(')')
        } else {
            f.write_str("Array(?)")
        }
    }

    fn fmt_effect_row(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        if let Some(row) = EffectRowType::from_type(self.db, ty) {
            f.write_char('{')?;
            self.fmt_effect_row_inner(f, &row)?;
            f.write_char('}')
        } else {
            f.write_str("{}")
        }
    }

    fn fmt_effect_row_inner(&self, f: &mut Formatter<'_>, row: &EffectRowType<'_>) -> fmt::Result {
        if row.is_empty(self.db) {
            return Ok(());
        }

        let abilities = row.abilities(self.db);
        for (i, &a) in abilities.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            self.fmt(f, a)?;
        }

        if let Some(var_id) = row.tail_var(self.db) {
            if !abilities.is_empty() {
                f.write_str(" | ")?;
            }
            fmt_var_id(f, var_id)?;
        }

        Ok(())
    }

    fn fmt_ability_ref(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        let Some(ability) = AbilityRefType::from_type(self.db, ty) else {
            return f.write_str("?ability");
        };

        let Some(name) = ability.name(self.db) else {
            return f.write_str("?ability");
        };

        let params = ability.params(self.db);
        if params.is_empty() {
            f.write_str(name.text(self.db))
        } else {
            f.write_str(name.text(self.db))?;
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                self.fmt(f, p)?;
            }
            f.write_char(')')
        }
    }

    fn fmt_ref(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        let params = ty.params(self.db);
        let nullable = matches!(
            ty.get_attr(self.db, "nullable"),
            Some(Attribute::Bool(true))
        );

        if let Some(&pointee) = params.first() {
            self.fmt(f, pointee)?;
            if nullable {
                f.write_char('?')?;
            }
            Ok(())
        } else if nullable {
            f.write_char('?')
        } else {
            f.write_str("Ref(?)")
        }
    }

    fn fmt_type_var(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        if let Some(Attribute::IntBits(id)) = ty.get_attr(self.db, "id") {
            fmt_var_id(f, *id)
        } else {
            f.write_char('?')
        }
    }

    fn fmt_unresolved(&self, f: &mut Formatter<'_>, ty: Type<'_>) -> fmt::Result {
        let Some(Attribute::Symbol(name)) = ty.get_attr(self.db, "name") else {
            return f.write_str("?unresolved");
        };

        let params = ty.params(self.db);
        if params.is_empty() {
            f.write_str(name.text(self.db))
        } else {
            f.write_str(name.text(self.db))?;
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                self.fmt(f, p)?;
            }
            f.write_char(')')
        }
    }

    fn fmt_generic(
        &self,
        f: &mut Formatter<'_>,
        dialect: &str,
        name: &str,
        ty: Type<'_>,
    ) -> fmt::Result {
        let params = ty.params(self.db);

        // Capitalize first letter for nicer display
        if dialect == "adt" || dialect == "src" {
            let mut chars = name.chars();
            if let Some(c) = chars.next() {
                for ch in c.to_uppercase() {
                    f.write_char(ch)?;
                }
                f.write_str(chars.as_str())?;
            }
        } else {
            write!(f, "{dialect}.{name}")?;
        }

        if !params.is_empty() {
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                self.fmt(f, p)?;
            }
            f.write_char(')')?;
        }

        Ok(())
    }
}

/// Convert a variable ID to a readable name (a, b, c, ..., t0, t1, ...).
fn fmt_var_id(f: &mut Formatter<'_>, id: u64) -> fmt::Result {
    if id < 26 {
        f.write_char((b'a' + id as u8) as char)
    } else {
        write!(f, "t{}", id - 26)
    }
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

    #[test]
    fn test_display_trait() {
        TributeDatabaseImpl::default().attach(|db| {
            let int_ty = *I64::new(db);
            let display = TypeDisplay::new(db, int_ty);
            assert_eq!(format!("{display}"), "Int");
        });
    }
}
