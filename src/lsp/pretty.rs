//! Type pretty-printing for display in IDE hovers.
//!
//! Converts internal IR types to user-friendly notation like:
//! - `Int`, `Float`, `String`, `Bool`
//! - `fn(Int, Int) -> Int`
//! - `fn(a) ->{Console} a`
//! - `List(a)`
//! - `{State(Int), Console | e}`

use tribute_trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func};
use tribute_trunk_ir::{Attribute, DialectType, Type};

/// Pretty-print a type to a user-friendly string.
pub fn print_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    TypePrinter::new(db).print(ty)
}

/// Type printer with database context.
struct TypePrinter<'a, 'db> {
    db: &'a dyn salsa::Database,
    _marker: std::marker::PhantomData<&'db ()>,
}

impl<'a, 'db> TypePrinter<'a, 'db>
where
    'a: 'db,
{
    fn new(db: &'a dyn salsa::Database) -> Self {
        Self {
            db,
            _marker: std::marker::PhantomData,
        }
    }

    fn print(&self, ty: Type<'db>) -> String {
        let dialect = ty.dialect(self.db).text(self.db);
        let name = ty.name(self.db).text(self.db);

        match (dialect, name) {
            // Core types with friendly names
            ("core", "nil") => "()".to_string(),
            ("core", "never") => "Never".to_string(),
            ("core", "string") => "String".to_string(),
            ("core", "bytes") => "Bytes".to_string(),
            ("core", "ptr") => "Ptr".to_string(),

            // Function type (must come before float pattern since "func" starts with 'f')
            ("core", "func") => self.print_function(ty),

            // Integer types
            ("core", name) if name.starts_with('i') => match &name[1..] {
                "1" => "Bool".to_string(),
                "64" => "Int".to_string(),
                bits => format!("I{bits}"),
            },

            // Float types
            ("core", name) if name.starts_with('f') => match &name[1..] {
                "64" => "Float".to_string(),
                bits => format!("F{bits}"),
            },

            // Tuple type
            ("core", "tuple") => self.print_tuple(ty),

            // Array type
            ("core", "array") => self.print_array(ty),

            // Effect row
            ("core", "effect_row") => self.print_effect_row(ty),

            // Ability reference
            ("core", "ability_ref") => self.print_ability_ref(ty),

            // Reference type
            ("core", "ref_") => self.print_ref(ty),

            // Type variable
            ("type", "var") => self.print_type_var(ty),

            // Type error
            ("type", "error") => "<error>".to_string(),

            // Source-level unresolved types
            ("src", "unresolved_type") => self.print_unresolved(ty),

            // Generic fallback with params
            _ => self.print_generic(dialect, name, ty),
        }
    }

    fn print_function(&self, ty: Type<'db>) -> String {
        let Some(func) = Func::from_type(self.db, ty) else {
            return "fn(?)".to_string();
        };

        let params = func.params(self.db);
        let result = func.result(self.db);
        let effect = func.effect(self.db);

        // Format parameters
        let params_str = if params.is_empty() {
            "()".to_string()
        } else {
            let param_strs: Vec<_> = params.iter().map(|&p| self.print(p)).collect();
            format!("({})", param_strs.join(", "))
        };

        // Format arrow with effect
        let arrow = match effect {
            Some(eff) => {
                if let Some(row) = EffectRowType::from_type(self.db, eff) {
                    if row.is_empty(self.db) {
                        " -> ".to_string()
                    } else {
                        format!(" ->{{{}}} ", self.print_effect_row_inner(&row))
                    }
                } else {
                    " -> ".to_string()
                }
            }
            None => " -> ".to_string(),
        };

        format!("fn{}{}{}", params_str, arrow, self.print(result))
    }

    fn print_tuple(&self, ty: Type<'db>) -> String {
        let params = ty.params(self.db);
        if params.is_empty() {
            return "()".to_string();
        }

        // Flatten cons cells into a list
        let mut elements = Vec::new();
        let mut current = ty;

        while current.is_dialect(self.db, "core", "tuple") {
            let params = current.params(self.db);
            if params.len() >= 2 {
                elements.push(self.print(params[0])); // head
                current = params[1]; // tail
            } else {
                break;
            }
        }

        // Check if tail is nil (complete tuple)
        if !current.is_dialect(self.db, "core", "nil") {
            elements.push(self.print(current));
        }

        format!("({})", elements.join(", "))
    }

    fn print_array(&self, ty: Type<'db>) -> String {
        let params = ty.params(self.db);
        if let Some(&elem) = params.first() {
            format!("Array({})", self.print(elem))
        } else {
            "Array(?)".to_string()
        }
    }

    fn print_effect_row(&self, ty: Type<'db>) -> String {
        if let Some(row) = EffectRowType::from_type(self.db, ty) {
            format!("{{{}}}", self.print_effect_row_inner(&row))
        } else {
            "{}".to_string()
        }
    }

    fn print_effect_row_inner(&self, row: &EffectRowType<'db>) -> String {
        if row.is_empty(self.db) {
            return String::new();
        }

        let abilities: Vec<_> = row
            .abilities(self.db)
            .iter()
            .map(|&a| self.print(a))
            .collect();

        match row.tail_var(self.db) {
            Some(var_id) => {
                let var_name = var_id_to_name(var_id);
                if abilities.is_empty() {
                    var_name
                } else {
                    format!("{} | {}", abilities.join(", "), var_name)
                }
            }
            None => abilities.join(", "),
        }
    }

    fn print_ability_ref(&self, ty: Type<'db>) -> String {
        if let Some(ability) = AbilityRefType::from_type(self.db, ty) {
            if let Some(name) = ability.name(self.db) {
                let params = ability.params(self.db);
                if params.is_empty() {
                    name.text(self.db).to_string()
                } else {
                    let param_strs: Vec<_> = params.iter().map(|&p| self.print(p)).collect();
                    format!("{}({})", name.text(self.db), param_strs.join(", "))
                }
            } else {
                "?ability".to_string()
            }
        } else {
            "?ability".to_string()
        }
    }

    fn print_ref(&self, ty: Type<'db>) -> String {
        let params = ty.params(self.db);
        let nullable = matches!(
            ty.get_attr(self.db, "nullable"),
            Some(Attribute::Bool(true))
        );

        if let Some(&pointee) = params.first() {
            let pointee_str = self.print(pointee);
            if nullable {
                format!("{}?", pointee_str)
            } else {
                pointee_str
            }
        } else if nullable {
            "?".to_string()
        } else {
            "Ref(?)".to_string()
        }
    }

    fn print_type_var(&self, ty: Type<'db>) -> String {
        if let Some(Attribute::IntBits(id)) = ty.get_attr(self.db, "id") {
            var_id_to_name(*id)
        } else {
            "?".to_string()
        }
    }

    fn print_unresolved(&self, ty: Type<'db>) -> String {
        if let Some(Attribute::Symbol(name)) = ty.get_attr(self.db, "name") {
            let params = ty.params(self.db);
            if params.is_empty() {
                name.text(self.db).to_string()
            } else {
                let param_strs: Vec<_> = params.iter().map(|&p| self.print(p)).collect();
                format!("{}({})", name.text(self.db), param_strs.join(", "))
            }
        } else {
            "?unresolved".to_string()
        }
    }

    fn print_generic(&self, dialect: &str, name: &str, ty: Type<'db>) -> String {
        let params = ty.params(self.db);

        // Capitalize first letter for nicer display
        let display_name = if dialect == "adt" || dialect == "src" {
            let mut chars = name.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect(),
                None => name.to_string(),
            }
        } else {
            format!("{dialect}.{name}")
        };

        if params.is_empty() {
            display_name
        } else {
            let param_strs: Vec<_> = params.iter().map(|&p| self.print(p)).collect();
            format!("{}({})", display_name, param_strs.join(", "))
        }
    }
}

/// Convert a variable ID to a readable name (a, b, c, ..., e1, e2, ...).
fn var_id_to_name(id: u64) -> String {
    if id < 26 {
        ((b'a' + id as u8) as char).to_string()
    } else {
        format!("t{}", id - 26)
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
}
