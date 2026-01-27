//! IR lowering context.
//!
//! Manages state during AST-to-IR transformation.

use std::collections::HashMap;

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core;
use trunk_ir::{DialectType, Location, PathId, Symbol, Type, Value};

use crate::ast::{LocalId, NodeId, SpanMap, TypeKind, TypeScheme};

/// Context for lowering AST to TrunkIR.
pub struct IrLoweringCtx<'db> {
    pub db: &'db dyn salsa::Database,
    pub path: PathId<'db>,
    /// Span map for looking up source locations.
    span_map: SpanMap,
    /// Stack of scopes, each mapping LocalId to SSA value.
    scopes: Vec<HashMap<LocalId, Value<'db>>>,
    /// Function type schemes from type checking, keyed by function name.
    function_types: HashMap<Symbol, TypeScheme<'db>>,
}

impl<'db> IrLoweringCtx<'db> {
    /// Create a new IR lowering context.
    pub fn new(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
        span_map: SpanMap,
        function_types: HashMap<Symbol, TypeScheme<'db>>,
    ) -> Self {
        Self {
            db,
            path,
            span_map,
            scopes: vec![HashMap::new()],
            function_types,
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

    /// Look up a function's type scheme by name.
    pub fn lookup_function_type(&self, name: Symbol) -> Option<&TypeScheme<'db>> {
        self.function_types.get(&name)
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
            TypeKind::BoundVar { .. } => {
                // Quantified type variable in TypeScheme body → type-erased any
                tribute_rt::any_type(self.db)
            }
            TypeKind::UniVar { id } => {
                // UniVar should not survive substitution — compiler internal invariant violation
                panic!(
                    "ICE: UniVar({:?}) survived substitution — should have been resolved",
                    id
                );
            }
            TypeKind::Named { .. } => {
                // Type erasure: struct/enum → tribute_rt.any
                tribute_rt::any_type(self.db)
            }
            TypeKind::Func { params, result, .. } => {
                let params: Vec<Type<'db>> = params.iter().map(|p| self.convert_type(*p)).collect();
                let result = self.convert_type(*result);
                core::Func::new(self.db, params.into(), result).as_type()
            }
            TypeKind::Tuple(_) => {
                // Type erasure: tuple → tribute_rt.any
                tribute_rt::any_type(self.db)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{EffectRow, Type as AstType, TypeKind, TypeScheme};
    use salsa_test_macros::salsa_test;

    // =========================================================================
    // convert_type tests
    // =========================================================================

    #[salsa_test]
    fn test_convert_type_bound_var_to_any(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(db, path, crate::ast::SpanMap::default(), HashMap::new());

        let ty = AstType::new(db, TypeKind::BoundVar { index: 0 });
        let ir_ty = ctx.convert_type(ty);
        assert_eq!(ir_ty, tribute_rt::any_type(db));
    }

    #[salsa_test]
    fn test_convert_type_named_to_any(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(db, path, crate::ast::SpanMap::default(), HashMap::new());

        let int_ty = AstType::new(db, TypeKind::Int);
        let ty = AstType::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        let ir_ty = ctx.convert_type(ty);
        assert_eq!(ir_ty, tribute_rt::any_type(db));
    }

    #[salsa_test]
    fn test_convert_type_tuple_to_any(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(db, path, crate::ast::SpanMap::default(), HashMap::new());

        let int_ty = AstType::new(db, TypeKind::Int);
        let bool_ty = AstType::new(db, TypeKind::Bool);
        let ty = AstType::new(db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        let ir_ty = ctx.convert_type(ty);
        assert_eq!(ir_ty, tribute_rt::any_type(db));
    }

    #[salsa_test]
    fn test_convert_type_func_with_bound_vars(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(db, path, crate::ast::SpanMap::default(), HashMap::new());

        let bound_var = AstType::new(db, TypeKind::BoundVar { index: 0 });
        let effect = EffectRow::pure(db);
        let ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );
        let ir_ty = ctx.convert_type(ty);

        // BoundVar params/result → tribute_rt.any, wrapped in core.func
        let any_ty = tribute_rt::any_type(db);
        let expected = core::Func::new(db, vec![any_ty].into(), any_ty).as_type();
        assert_eq!(ir_ty, expected);
    }

    #[salsa_test]
    fn test_convert_type_primitives_unchanged(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(db, path, crate::ast::SpanMap::default(), HashMap::new());

        // Int → I64
        let int_ty = AstType::new(db, TypeKind::Int);
        assert_eq!(ctx.convert_type(int_ty), core::I64::new(db).as_type());

        // Bool → I1
        let bool_ty = AstType::new(db, TypeKind::Bool);
        assert_eq!(ctx.convert_type(bool_ty), core::I1::new(db).as_type());

        // String → string
        let str_ty = AstType::new(db, TypeKind::String);
        assert_eq!(ctx.convert_type(str_ty), core::String::new(db).as_type());

        // Float → F64
        let float_ty = AstType::new(db, TypeKind::Float);
        assert_eq!(ctx.convert_type(float_ty), core::F64::new(db).as_type());

        // Nil → Nil
        let nil_ty = AstType::new(db, TypeKind::Nil);
        assert_eq!(ctx.convert_type(nil_ty), core::Nil::new(db).as_type());
    }

    // =========================================================================
    // lookup_function_type tests
    // =========================================================================

    #[salsa_test]
    fn test_lookup_function_type_found(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let name = Symbol::new("foo");
        let body = AstType::new(db, TypeKind::Int);
        let scheme = TypeScheme::new(db, vec![], body);

        let mut ft = HashMap::new();
        ft.insert(name, scheme);

        let ctx = IrLoweringCtx::new(db, path, crate::ast::SpanMap::default(), ft);
        assert_eq!(ctx.lookup_function_type(name), Some(&scheme));
    }

    #[salsa_test]
    fn test_lookup_function_type_not_found(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ctx = IrLoweringCtx::new(db, path, crate::ast::SpanMap::default(), HashMap::new());
        assert_eq!(ctx.lookup_function_type(Symbol::new("missing")), None);
    }
}
