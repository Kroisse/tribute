//! IR lowering context.
//!
//! Manages state during AST-to-IR transformation.

use std::collections::HashMap;

use trunk_ir::dialect::core;
use trunk_ir::{DialectType, Location, PathId, Type, Value};

use crate::ast::{LocalId, NodeId, SpanMap, TypeKind};

/// Context for lowering AST to TrunkIR.
pub struct IrLoweringCtx<'db> {
    pub db: &'db dyn salsa::Database,
    pub path: PathId<'db>,
    /// Span map for looking up source locations.
    span_map: SpanMap,
    /// Stack of scopes, each mapping LocalId to SSA value.
    scopes: Vec<HashMap<LocalId, Value<'db>>>,
}

impl<'db> IrLoweringCtx<'db> {
    /// Create a new IR lowering context.
    pub fn new(db: &'db dyn salsa::Database, path: PathId<'db>, span_map: SpanMap) -> Self {
        Self {
            db,
            path,
            span_map,
            scopes: vec![HashMap::new()],
        }
    }

    /// Create a location for a node.
    pub fn location(&self, node_id: NodeId) -> Location<'db> {
        let span = self.span_map.get_or_default(node_id);
        Location::new(self.path, span)
    }

    /// Enter a new scope.
    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Exit the current scope.
    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    /// Bind a local variable to an SSA value.
    pub fn bind(&mut self, local_id: LocalId, value: Value<'db>) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(local_id, value);
        }
    }

    /// Look up a local variable.
    pub fn lookup(&self, local_id: LocalId) -> Option<Value<'db>> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(&local_id) {
                return Some(*value);
            }
        }
        None
    }

    /// Convert an AST type to a TrunkIR type.
    pub fn convert_type(&self, ty: crate::ast::Type<'db>) -> Type<'db> {
        match ty.kind(self.db) {
            TypeKind::Int => core::I64::new(self.db).as_type(),
            TypeKind::Nat => core::I64::new(self.db).as_type(),
            TypeKind::Float => core::F64::new(self.db).as_type(),
            TypeKind::Bool => core::I1::new(self.db).as_type(),
            TypeKind::String => core::String::new(self.db).as_type(),
            TypeKind::Bytes => core::Bytes::new(self.db).as_type(),
            TypeKind::Rune => core::I32::new(self.db).as_type(),
            TypeKind::Nil | TypeKind::Error => core::Nil::new(self.db).as_type(),
            TypeKind::UniVar { .. } | TypeKind::BoundVar { .. } => {
                // Unresolved type variables - use placeholder
                core::Nil::new(self.db).as_type()
            }
            TypeKind::Named { .. } => {
                // Named type - placeholder for now
                core::Nil::new(self.db).as_type()
            }
            TypeKind::Func { params, result, .. } => {
                let params: Vec<Type<'db>> = params.iter().map(|p| self.convert_type(*p)).collect();
                let result = self.convert_type(*result);
                core::Func::new(self.db, params.into(), result).as_type()
            }
            TypeKind::Tuple(_) => {
                // Tuple type - placeholder for now
                core::Nil::new(self.db).as_type()
            }
            TypeKind::App { ctor, .. } => self.convert_type(*ctor),
        }
    }

    /// Get the unit type.
    pub fn unit_type(&self) -> Type<'db> {
        core::Nil::new(self.db).as_type()
    }

    /// Get the int type.
    pub fn int_type(&self) -> Type<'db> {
        core::I64::new(self.db).as_type()
    }

    /// Get the bool type.
    pub fn bool_type(&self) -> Type<'db> {
        core::I1::new(self.db).as_type()
    }
}
