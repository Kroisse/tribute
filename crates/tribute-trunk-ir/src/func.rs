//! Function dialect operations.

use crate::{Attribute, Region, Type, dialect};
use tribute_core::Location;

dialect! {
    func {
        /// `func.func` operation: defines a function.
        pub op func[sym_name, r#type]() { body };

        /// `func.return` operation: returns values from a function.
        pub op r#return(..operands) {};
    }
}

impl<'db> Func<'db> {
    /// Build a function with a closure that constructs the entry block.
    pub fn build(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
        params: Vec<Type>,
        results: Vec<Type>,
        f: impl FnOnce(&mut crate::BlockBuilder<'db>),
    ) -> Self {
        let mut entry = crate::BlockBuilder::new(db, location).args(params.clone());
        f(&mut entry);
        let region = Region::new(db, location, vec![entry.build()]);
        Self::new(
            db,
            location,
            Attribute::String(name.to_string()),
            Attribute::Type(Type::Function { params, results }),
            region,
        )
    }

    /// Get the function name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> &str {
        let Attribute::String(name) = self.sym_name(db) else {
            panic!("func.func missing sym_name attribute")
        };
        name
    }

    /// Get the function type.
    pub fn ty(&self, db: &'db dyn salsa::Database) -> &'db Type {
        let Attribute::Type(t) = self.r#type(db) else {
            panic!("func.func missing type attribute")
        };
        t
    }
}

impl<'db> Return<'db> {
    /// Create a new return with no values.
    pub fn empty(db: &'db dyn salsa::Database, location: Location<'db>) -> Self {
        Self::new(db, location, vec![])
    }

    /// Create a new return with a single value.
    pub fn value(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        value: crate::Value<'db>,
    ) -> Self {
        Self::new(db, location, vec![value])
    }
}
