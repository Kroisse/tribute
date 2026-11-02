//! Metadata for source-logical declaration lowering.

use trunk_ir::refs::TypeRef;

struct PendingWellKnownType<'db> {
    definition: crate::ast::TypeDefId<'db>,
    ir_type: Option<TypeRef>,
}

pub(super) struct WellKnownTypePrescan<'db> {
    string: Option<PendingWellKnownType<'db>>,
}

impl<'db> WellKnownTypePrescan<'db> {
    pub(super) fn new(types: crate::typeck::WellKnownTypes<'db>) -> Self {
        Self {
            string: types.string.map(|ty| PendingWellKnownType {
                definition: ty.definition,
                ir_type: None,
            }),
        }
    }

    pub(super) fn is_string(&self, definition: crate::ast::TypeDefId<'db>) -> bool {
        self.string
            .as_ref()
            .is_some_and(|string| string.definition == definition)
    }

    pub(super) fn record_string(&mut self, ir_type: TypeRef) {
        if let Some(string) = &mut self.string {
            string.ir_type = Some(ir_type);
        }
    }

    pub(super) fn finish(self) -> tribute_ir::metadata::WellKnownTypes {
        tribute_ir::metadata::WellKnownTypes {
            string: self.string.and_then(|string| string.ir_type),
        }
    }
}
