//! Metadata for source-logical declaration lowering.

use trunk_ir::refs::TypeRef;

struct PendingWellKnownType {
    definition: crate::typeck::DefinitionIdentity,
    ir_type: Option<TypeRef>,
}

pub(super) struct WellKnownTypePrescan {
    string: Option<PendingWellKnownType>,
}

impl WellKnownTypePrescan {
    pub(super) fn new(types: crate::typeck::WellKnownTypes<'_>) -> Self {
        Self {
            string: types.string.map(|ty| PendingWellKnownType {
                definition: ty.definition,
                ir_type: None,
            }),
        }
    }

    pub(super) fn is_string(&self, definition: crate::typeck::DefinitionIdentity) -> bool {
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
