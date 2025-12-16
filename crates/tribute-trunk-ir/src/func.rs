//! Function dialect operations.

use crate::{Attribute, OpNameId, Operation, Region, Symbol, Type, define_dialect_op};
use tribute_core::Location;

define_dialect_op! {
    /// `func.func` operation: defines a function.
    pub struct Func("func", "func") {
        has_attr("sym_name"),
        has_attr("type"),
        has_region,
    }
}

impl<'db> Func<'db> {
    /// Create a new function definition.
    pub fn new(
        db: &'db dyn salsa::Database,
        name: &str,
        params: Vec<Type>,
        results: Vec<Type>,
        body: Region<'db>,
        location: Location<'db>,
    ) -> Self {
        let op_name = OpNameId::new(db, "func", "func");
        let ty = Type::Function { params, results };
        let op = Operation::of(db, op_name, location)
            .attr("sym_name", Attribute::String(name.to_string()))
            .attr("type", Attribute::Type(ty))
            .region(body)
            .build();
        Self::wrap_unchecked(op)
    }

    /// Build a function with a closure that constructs the entry block.
    pub fn build(
        db: &'db dyn salsa::Database,
        name: &str,
        params: Vec<Type>,
        results: Vec<Type>,
        location: Location<'db>,
        f: impl FnOnce(&mut crate::BlockBuilder<'db>),
    ) -> Self {
        let mut entry = crate::BlockBuilder::new(db, location).args(params.clone());
        f(&mut entry);
        let body = Region::new(db, vec![entry.build()], location);
        Self::new(db, name, params, results, body, location)
    }

    /// Get the function name.
    pub fn name(&self, db: &'db dyn salsa::Database) -> String {
        let key = Symbol::new(db, "sym_name");
        match self.op.attributes(db).get(&key) {
            Some(Attribute::String(s)) => s.clone(),
            _ => panic!("func.func missing sym_name attribute"),
        }
    }

    /// Get the function type.
    pub fn ty(&self, db: &'db dyn salsa::Database) -> Type {
        let key = Symbol::new(db, "type");
        match self.op.attributes(db).get(&key) {
            Some(Attribute::Type(t)) => t.clone(),
            _ => panic!("func.func missing type attribute"),
        }
    }

    /// Get the function body region.
    pub fn body(&self, db: &'db dyn salsa::Database) -> Region<'db> {
        self.op.regions(db)[0]
    }
}

define_dialect_op! {
    /// `func.return` operation: returns values from a function.
    pub struct Return("func", "return") {}
}

impl<'db> Return<'db> {
    /// Create a new return with no values.
    pub fn empty(db: &'db dyn salsa::Database, location: Location<'db>) -> Self {
        Self::new(db, vec![], location)
    }

    /// Create a new return with a single value.
    pub fn value(
        db: &'db dyn salsa::Database,
        value: crate::Value<'db>,
        location: Location<'db>,
    ) -> Self {
        Self::new(db, vec![value], location)
    }

    /// Create a new return with multiple values.
    pub fn new(
        db: &'db dyn salsa::Database,
        operands: Vec<crate::Value<'db>>,
        location: Location<'db>,
    ) -> Self {
        let name = OpNameId::new(db, "func", "return");
        let op = Operation::of(db, name, location).operands(operands).build();
        Self::wrap_unchecked(op)
    }

    /// Get the returned values.
    pub fn operands(&self, db: &'db dyn salsa::Database) -> &[crate::Value<'db>] {
        self.op.operands(db)
    }
}
