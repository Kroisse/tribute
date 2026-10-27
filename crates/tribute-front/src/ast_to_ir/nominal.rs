//! Nominal types of logical lowering: the layouts of source structs and
//! enums, the row binders of the definition being lowered, and the
//! conversion of source types at the source-logical boundary.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::dialect::adt;
use tribute_ir::dialect::tribute_control::{Row, RowUnion};
use tribute_ir::dialect::tribute_rt;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{Attribute, AttributeMap, Location, TypeDataBuilder};
use trunk_ir::{Symbol, SymbolPath};

use crate::ast::{CtorId, RowVar, TypeKind, TypeScheme};

use super::row_binders::RowBinders;

/// The key component that tells nominal type origins apart.
fn nominal_origin_key(origin: crate::ast::TypeOrigin) -> String {
    match origin {
        crate::ast::TypeOrigin::Source { disambiguator: 0 } => "source".into(),
        crate::ast::TypeOrigin::Source { disambiguator } => format!("source{disambiguator}"),
        crate::ast::TypeOrigin::Builtin(crate::ast::BuiltinType::List) => "builtin_list".into(),
        crate::ast::TypeOrigin::Builtin(crate::ast::BuiltinType::Slots) => "builtin_slots".into(),
        crate::ast::TypeOrigin::Builtin(crate::ast::BuiltinType::Dispatch) => {
            "builtin_dispatch".into()
        }
        crate::ast::TypeOrigin::Synthetic => "synthetic".into(),
    }
}

/// Encode a tagged type-shape node without relying on separator characters in
/// source symbols. Each component carries its byte length, so nested keys are
/// readable in printed IR while still being uniquely decodable.
fn logical_key<I, S>(tag: &str, parts: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let parts = parts.into_iter().collect::<Vec<_>>();
    let mut key = format!("{tag}_{}", parts.len());
    for part in parts {
        let part = part.as_ref();
        key.push('_');
        key.push_str(&part.len().to_string());
        key.push('_');
        key.push_str(part);
    }
    key
}

/// The nominal layouts and row binders logical lowering reads, and the
/// conversion of source types into logical IR types.
pub(crate) struct NominalTypes<'db> {
    db: &'db dyn salsa::Database,
    /// Struct field order: CtorId → [field_names in definition order].
    /// Used for lowering Record expressions to adt.struct_new.
    struct_fields: HashMap<CtorId<'db>, Vec<Symbol>>,
    /// Field names of named-field enum variants: (enum layout, tag) →
    /// [field names in declaration order]. Used to match brace-form
    /// constructor patterns by name.
    variant_field_names: HashMap<(TypeRef, Symbol), Vec<Symbol>>,
    /// Type map: type name → arena TypeRef for adt.struct / adt.enum.
    /// Used for named structs, tuples, and (future) enum variants.
    type_map: HashMap<Symbol, TypeRef>,
    /// All source nominal identities collected before source-logical layouts
    /// are built. This lets recursive and forward fields retain `adt.typeref`
    /// while their layout is still incomplete.
    logical_nominal_declarations: HashSet<Symbol>,
    /// The row binders of each function definition of the module.
    row_binders: HashMap<SymbolPath, RowBinders>,
    /// The row binders of the definition whose types are being read.
    current_row_binders: RowBinders,
}

impl<'db> NominalTypes<'db> {
    pub(crate) fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            struct_fields: HashMap::default(),
            variant_field_names: HashMap::default(),
            type_map: HashMap::default(),
            logical_nominal_declarations: HashSet::default(),
            row_binders: HashMap::default(),
            current_row_binders: Vec::new(),
        }
    }

    /// The row binders recorded for the definition `name`.
    pub(crate) fn declared_row_binders(&self, name: &[Symbol]) -> Option<RowBinders> {
        self.row_binders.get(name).cloned()
    }

    /// Read types with `binders` as the current definition's row binders,
    /// and return the binders they replace.
    pub(crate) fn replace_current_row_binders(&mut self, binders: RowBinders) -> RowBinders {
        std::mem::replace(&mut self.current_row_binders, binders)
    }

    /// Record the row binders of the function definition `name`.
    pub(crate) fn register_row_binders(&mut self, name: SymbolPath, binders: RowBinders) {
        self.row_binders.insert(name, binders);
    }

    /// The row binders of the definition whose types are being read.
    pub(crate) fn current_row_binders(&self) -> &RowBinders {
        &self.current_row_binders
    }

    /// The IR row of `row` in the current definition.
    ///
    /// # Panics
    ///
    /// If the row's tail is not a binder of the current definition.
    pub(crate) fn ability_row(&self, row: crate::ast::AbilityRow<'db>) -> Row {
        let instances = row
            .instances(self.db)
            .iter()
            .map(|effect| self.ability_symbol(&effect.ability_id.qualified(self.db), &effect.args));
        let tail = row.rest(self.db).map(|variable| {
            let index = self
                .current_row_binders
                .iter()
                .position(|(bound, _)| *bound == variable)
                .unwrap_or_else(|| {
                    panic!(
                        "row variable {} has no binder in the current definition",
                        variable.id
                    )
                });
            u32::try_from(index).expect("row binder index fits u32")
        });
        Row::new(instances, tail)
    }

    /// The tail unions `scheme` declares, numbered by the current binders.
    /// Only a union of several open sources selects evidence by tail.
    pub(crate) fn row_unions(&self, scheme: TypeScheme<'db>) -> Vec<RowUnion> {
        let index = |variable: RowVar| {
            self.current_row_binders
                .iter()
                .position(|(bound, _)| *bound == variable)
                .and_then(|index| u32::try_from(index).ok())
        };
        scheme
            .row_unions(self.db)
            .iter()
            .filter_map(|union| {
                let result = union.result.rest(self.db)?;
                let sources: Vec<_> = union
                    .sources
                    .iter()
                    .filter_map(|source| source.rest(self.db))
                    .filter(|source| *source != result)
                    .map(index)
                    .collect::<Option<_>>()?;
                (sources.len() > 1).then_some(RowUnion {
                    result: index(result)?,
                    sources,
                })
            })
            .collect()
    }

    /// Register struct field order for lowering Record expressions.
    pub(crate) fn register_struct_fields(
        &mut self,
        ctor_id: CtorId<'db>,
        field_names: Vec<Symbol>,
    ) {
        self.struct_fields.insert(ctor_id, field_names);
    }

    /// Register the field names of a named-field enum variant.
    pub(crate) fn register_variant_field_names(
        &mut self,
        layout: TypeRef,
        tag: Symbol,
        field_names: Vec<Symbol>,
    ) {
        self.variant_field_names.insert((layout, tag), field_names);
    }

    /// Field names of a named-field enum variant in declaration order.
    pub(crate) fn variant_field_names(&self, layout: TypeRef, tag: Symbol) -> Option<Vec<Symbol>> {
        self.variant_field_names.get(&(layout, tag)).cloned()
    }

    /// Get struct field order (for lowering Record → adt.struct_new).
    pub(crate) fn get_struct_field_order(&self, ctor_id: CtorId<'db>) -> Option<&Vec<Symbol>> {
        self.struct_fields.get(&ctor_id)
    }

    /// Register a type (named struct, tuple, etc.) in the type map.
    pub(crate) fn register_type(&mut self, name: Symbol, ty: TypeRef) {
        self.type_map.insert(name, ty);
    }

    /// Get a registered type by name.
    pub(crate) fn get_type(&self, name: &Symbol) -> Option<TypeRef> {
        self.type_map.get(name).copied()
    }

    /// Mark a nominal source type as known before its layout is complete.
    pub(crate) fn declare_logical_nominal(&mut self, name: Symbol) {
        self.logical_nominal_declarations.insert(name);
    }

    /// Get the type of an AST node by NodeId.
    ///
    /// Returns the type assigned during type checking.
    /// Used to get the effect type of lambda expressions.
    pub(crate) fn resolve_adt_type(&self, ty: crate::ast::Type<'db>) -> Option<TypeRef> {
        match ty.kind(self.db) {
            TypeKind::Named { name, .. } => self.get_type(name),
            _ => None,
        }
    }

    // This conversion and the identity keys below are structural maps into IR
    // types and key strings, not folds over `Type`: their output format is the
    // contract, and the source keys sort a row's effects.

    /// Convert a source type at the source-logical `tribute_control` boundary.
    pub(crate) fn convert_logical_type(
        &self,
        ir: &mut IrContext,
        ty: crate::ast::Type<'db>,
    ) -> TypeRef {
        match ty.kind(self.db) {
            TypeKind::Int | TypeKind::Nat | TypeKind::Rune => self.i32_type(ir),
            TypeKind::Float => self.f64_type(ir),
            TypeKind::Bool => self.bool_type(ir),
            TypeKind::Bytes => self.bytes_type(ir),
            TypeKind::Nil | TypeKind::Error => self.nil_type(ir),
            TypeKind::Never => core::Never::params().intern(ir).as_type_ref(),
            TypeKind::BoundVar { .. }
            | TypeKind::LocalBoundVar { .. }
            | TypeKind::UniVar { .. } => self.anyref_type(ir),
            TypeKind::Named { id, .. } if id.is_builtin_list(self.db) => self.anyref_type(ir),
            TypeKind::Named { id, .. } if id.is_builtin_dispatch(self.db) => {
                tribute_ir::runtime_layout::closure_base(ir)
            }
            TypeKind::Named { id, args, .. } if id.is_builtin_slots(self.db) => {
                let [element] = args.as_slice() else {
                    panic!("Slots has one type argument, found {}", args.len());
                };
                let element = self.convert_logical_type(ir, *element);
                tribute_ir::dialect::adt::Array::params(element)
                    .intern(ir)
                    .as_type_ref()
            }
            TypeKind::Named { id, .. }
                if self.get_type(&id.qualified(self.db)).is_some()
                    || self
                        .logical_nominal_declarations
                        .contains(&id.qualified(self.db)) =>
            {
                self.adt_typeref(ir, &id.qualified(self.db))
            }
            TypeKind::Named { .. } => self.anyref_type(ir),
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let params = params
                    .iter()
                    .map(|param| self.convert_logical_type(ir, *param))
                    .collect::<Vec<_>>();
                let result = self.convert_logical_type(ir, *result);
                tribute_ir::dialect::tribute_control::FuncSig::params(params, result)
                    .row(self.ability_row(*effect))
                    .intern(ir)
                    .as_type_ref()
            }
            TypeKind::Tuple(elements) => {
                let elements = elements
                    .iter()
                    .map(|element| self.convert_logical_type(ir, *element))
                    .collect::<Vec<_>>();
                tribute_ir::dialect::tribute_control::logical_tuple_type(ir, &elements)
            }
            TypeKind::Continuation { arg, result, .. } => {
                let arg = self.convert_logical_type(ir, *arg);
                let result = self.convert_logical_type(ir, *result);
                tribute_ir::dialect::tribute_control::ResumeToken::params(arg, result)
                    .intern(ir)
                    .as_type_ref()
            }
        }
    }

    fn logical_type_key(&self, ty: crate::ast::Type<'db>) -> String {
        match ty.kind(self.db) {
            TypeKind::Int => "int".into(),
            TypeKind::Nat => "nat".into(),
            TypeKind::Float => "float".into(),
            TypeKind::Bool => "bool".into(),
            TypeKind::Bytes => "bytes".into(),
            TypeKind::Rune => "rune".into(),
            TypeKind::Nil => "nil".into(),
            TypeKind::Never => "never".into(),
            TypeKind::Error => "error".into(),
            TypeKind::BoundVar { index } => logical_key("bound", [index.to_string()]),
            TypeKind::LocalBoundVar { scope, index } => {
                logical_key("local_bound", [scope.to_string(), index.to_string()])
            }
            TypeKind::UniVar { id } => self.logical_univar_key(*id),
            TypeKind::Named { id, name, args } => {
                let mut parts = vec![self.logical_nominal_key(*id, name)];
                parts.extend(args.iter().map(|arg| self.logical_type_key(*arg)));
                logical_key("named", parts)
            }
            TypeKind::Func { params, result, .. } => {
                let mut parts = vec![params.len().to_string()];
                parts.extend(params.iter().map(|param| self.logical_type_key(*param)));
                parts.push(self.logical_type_key(*result));
                logical_key("callable", parts)
            }
            TypeKind::Tuple(elements) => {
                let mut parts = vec![elements.len().to_string()];
                parts.extend(
                    elements
                        .iter()
                        .map(|element| self.logical_type_key(*element)),
                );
                logical_key("tuple", parts)
            }
            TypeKind::Continuation { arg, result, .. } => logical_key(
                "resume",
                [self.logical_type_key(*arg), self.logical_type_key(*result)],
            ),
        }
    }

    /// Source-level identity of one ability instance.
    ///
    /// Unlike [`Self::logical_type_key`], this keys nominal arguments by their
    /// qualified declaration and function arguments by their effect rows, so
    /// it distinguishes every distinct source instance.
    fn ability_instance_key(
        &self,
        ability: &Symbol,
        arguments: &[crate::ast::Type<'db>],
    ) -> String {
        let mut row_vars = Vec::new();
        let mut parts = vec![ability.with_str(str::to_owned)];
        parts.extend(
            arguments
                .iter()
                .map(|arg| self.source_type_key(*arg, &mut row_vars)),
        );
        logical_key("ability", parts)
    }

    fn source_type_key(
        &self,
        ty: crate::ast::Type<'db>,
        row_vars: &mut Vec<crate::ast::RowVar>,
    ) -> String {
        match ty.kind(self.db) {
            TypeKind::Named { id, args, .. } => {
                let mut parts = vec![self.source_nominal_key(*id)];
                parts.extend(args.iter().map(|arg| self.source_type_key(*arg, row_vars)));
                logical_key("named", parts)
            }
            TypeKind::Func {
                params,
                result,
                effect,
                ..
            } => {
                let mut parts = vec![params.len().to_string()];
                parts.extend(
                    params
                        .iter()
                        .map(|param| self.source_type_key(*param, row_vars)),
                );
                parts.push(self.source_type_key(*result, row_vars));
                parts.push(self.source_row_key(*effect, row_vars));
                logical_key("callable", parts)
            }
            TypeKind::Tuple(elements) => {
                let mut parts = vec![elements.len().to_string()];
                parts.extend(
                    elements
                        .iter()
                        .map(|element| self.source_type_key(*element, row_vars)),
                );
                logical_key("tuple", parts)
            }
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => logical_key(
                "resume",
                [
                    self.source_type_key(*arg, row_vars),
                    self.source_type_key(*result, row_vars),
                    self.source_row_key(*effect, row_vars),
                ],
            ),
            _ => self.logical_type_key(ty),
        }
    }

    fn source_nominal_key(&self, id: crate::ast::TypeDefId<'db>) -> String {
        logical_key(
            "nominal",
            [
                nominal_origin_key(id.origin(self.db)),
                id.qualified(self.db).with_str(str::to_owned),
            ],
        )
    }

    /// Key an ability row with its abilities in canonical order. Row variables
    /// are numbered by first appearance within the instance key, so the key
    /// does not depend on inference numbering.
    fn source_row_key(
        &self,
        row: crate::ast::AbilityRow<'db>,
        row_vars: &mut Vec<crate::ast::RowVar>,
    ) -> String {
        let mut effects: Vec<String> = row
            .instances(self.db)
            .iter()
            .map(|effect| {
                let mut parts = vec![effect.ability_id.qualified(self.db).with_str(str::to_owned)];
                parts.extend(
                    effect
                        .args
                        .iter()
                        .map(|arg| self.source_type_key(*arg, row_vars)),
                );
                logical_key("effect", parts)
            })
            .collect();
        effects.sort();
        let rest = match row.rest(self.db) {
            Some(var) => {
                let index = row_vars
                    .iter()
                    .position(|seen| *seen == var)
                    .unwrap_or_else(|| {
                        row_vars.push(var);
                        row_vars.len() - 1
                    });
                logical_key("open", [index.to_string()])
            }
            None => "closed".to_owned(),
        };
        effects.push(rest);
        logical_key("row", effects)
    }

    fn logical_nominal_key(&self, id: crate::ast::TypeDefId<'db>, name: &Symbol) -> String {
        logical_key(
            "nominal",
            [
                nominal_origin_key(id.origin(self.db)),
                name.with_str(str::to_owned),
            ],
        )
    }

    fn logical_univar_key(&self, id: crate::ast::UniVarId<'db>) -> String {
        use crate::ast::UniVarSource;

        let source = match id.source(self.db) {
            UniVarSource::FunctionLocal { func_id, index } => logical_key(
                "function_local",
                [
                    func_id.qualified(self.db).with_str(str::to_owned),
                    index.to_string(),
                ],
            ),
            UniVarSource::Anonymous(index) => logical_key("anonymous", [index.to_string()]),
            UniVarSource::Solver { index } => logical_key("solver", [index.to_string()]),
        };
        logical_key("univar", [source, id.index(self.db).to_string()])
    }

    /// Get the `core.i32` type.
    pub(crate) fn i32_type(&self, ir: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.i32").intern(ir)
    }

    /// Get the `core.nil` type.
    pub(crate) fn nil_type(&self, ir: &mut IrContext) -> TypeRef {
        core::Nil::params().intern(ir).as_type_ref()
    }

    /// Get the `core.i1` (bool) type.
    pub(crate) fn bool_type(&self, ir: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.i1").intern(ir)
    }

    /// Get the `core.f64` type.
    pub(crate) fn f64_type(&self, ir: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.f64").intern(ir)
    }

    /// Get the `core.bytes` type.
    pub(crate) fn bytes_type(&self, ir: &mut IrContext) -> TypeRef {
        core::Bytes::params().intern(ir).as_type_ref()
    }

    /// Get the `tribute_rt.anyref` type.
    pub(crate) fn anyref_type(&self, ir: &mut IrContext) -> TypeRef {
        tribute_rt::Anyref::params().intern(ir).as_type_ref()
    }

    /// The symbol of the `ability.decl` of one ability instance.
    ///
    /// It is the instance key, derived from the checked source arguments, so
    /// arguments sharing a lowered representation stay distinct instances.
    pub(crate) fn ability_symbol(
        &self,
        ability_name: &Symbol,
        arguments: &[crate::ast::Type<'db>],
    ) -> SymbolPath {
        SymbolPath::new([Symbol::new(
            &self.ability_instance_key(ability_name, arguments),
        )])
    }

    /// Create an `adt.struct` type with name and fields.
    pub(crate) fn adt_struct_type(
        &self,
        ir: &mut IrContext,
        name: &Symbol,
        fields: &[(Symbol, TypeRef)],
    ) -> TypeRef {
        let name = ir.intern_symbol_text(name);
        let fields: Vec<_> = fields
            .iter()
            .map(|(field, ty)| (ir.intern_symbol_text(field), *ty))
            .collect();
        adt::Struct::from_fields(ir, name, fields, AttributeMap::new()).as_type_ref()
    }

    /// Create an `adt.enum` type with name and variants.
    pub(crate) fn adt_enum_type(
        &self,
        ir: &mut IrContext,
        name: &Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
    ) -> TypeRef {
        Self::enum_type(ir, name, variants, AttributeMap::new())
    }

    /// Create an `adt.enum` type that records where it was declared.
    pub(crate) fn adt_enum_type_with_definition(
        &self,
        ir: &mut IrContext,
        name: &Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
        definition: Location,
    ) -> TypeRef {
        let mut attrs = AttributeMap::new();
        attrs
            .view_mut(ir)
            .insert("tribute.definition", Attribute::Location(definition));
        Self::enum_type(ir, name, variants, attrs)
    }

    fn enum_type(
        ir: &mut IrContext,
        name: &Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
        attrs: AttributeMap,
    ) -> TypeRef {
        let name = ir.intern_symbol_text(name);
        let variants: Vec<_> = variants
            .iter()
            .map(|(variant, fields)| (ir.intern_symbol_text(variant), fields.iter().copied()))
            .collect();
        adt::Enum::from_variants(ir, name, variants, attrs).as_type_ref()
    }

    /// Create an `adt.typeref` type — a reference to a named type.
    pub(crate) fn adt_typeref(&self, ir: &mut IrContext, name: &Symbol) -> TypeRef {
        let name = ir.intern_symbol_text(name);
        adt::Typeref::params().name(name).intern(ir).as_type_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityId, Type as AstType};

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    fn func_type<'db>(
        db: &'db salsa::DatabaseImpl,
        params: Vec<AstType<'db>>,
        effect: crate::ast::AbilityRow<'db>,
    ) -> AstType<'db> {
        AstType::new(
            db,
            TypeKind::Func {
                params,
                result: AstType::new(db, TypeKind::Nat),
                effect,
            },
        )
    }

    /// Row variables are keyed by first appearance, not by inference ids, and
    /// effects are keyed in canonical order.
    #[test]
    fn ability_instance_key_normalizes_ability_rows() {
        use crate::ast::{AbilityInstance, AbilityRow, RowVar};

        let db = test_db();
        let ctx = NominalTypes::new(&db);
        let ability = Symbol::new("State");
        let effect = |name: &str| AbilityInstance {
            ability_id: AbilityId::source(&db, trunk_ir::SymbolPath::new([name])),
            args: vec![],
        };
        let open = |id, effects| AbilityRow::new(&db, effects, Some(RowVar { id }));
        let key = |row| ctx.ability_instance_key(&ability, &[func_type(&db, vec![], row)]);

        assert_eq!(key(open(3, vec![])), key(open(9, vec![])));
        assert_eq!(
            key(open(3, vec![effect("Ask"), effect("Tell")])),
            key(open(3, vec![effect("Tell"), effect("Ask")]))
        );
        assert_ne!(
            key(open(3, vec![effect("Ask")])),
            key(AbilityRow::new(&db, vec![effect("Ask")], None))
        );

        let shared = |a, b| {
            let ability = ability.clone();
            let inner = func_type(&db, vec![], open(a, vec![]));
            ctx.ability_instance_key(&ability, &[func_type(&db, vec![inner], open(b, vec![]))])
        };
        assert_eq!(shared(1, 1), shared(5, 5));
        assert_ne!(shared(1, 1), shared(1, 2));
    }

    /// Structural and nominal arguments are keyed by their source identity.
    #[test]
    fn ability_instance_key_distinguishes_source_arguments() {
        let db = test_db();
        let ctx = NominalTypes::new(&db);
        let ability = Symbol::new("State");
        let nat = AstType::new(&db, TypeKind::Nat);
        let int = AstType::new(&db, TypeKind::Int);
        let named = |id| {
            AstType::new(
                &db,
                TypeKind::Named {
                    id,
                    name: Symbol::new("Box"),
                    args: vec![nat],
                },
            )
        };
        let keys = [
            AstType::new(&db, TypeKind::Tuple(vec![nat, int])),
            AstType::new(&db, TypeKind::Tuple(vec![int, nat])),
            AstType::new(
                &db,
                TypeKind::Continuation {
                    arg: nat,
                    result: int,
                    effect: crate::ast::AbilityRow::pure(&db),
                },
            ),
            named(crate::ast::TypeDefId::builtin_list(&db)),
            named(crate::ast::TypeDefId::synthetic(
                &db,
                trunk_ir::SymbolPath::new(["a", "Box"]),
            )),
            named(crate::ast::TypeDefId::synthetic(
                &db,
                trunk_ir::SymbolPath::new(["b", "Box"]),
            )),
        ]
        .map(|arg| ctx.ability_instance_key(&ability, &[arg]));
        let distinct: HashSet<_> = keys.iter().collect();
        assert_eq!(distinct.len(), keys.len(), "{keys:#?}");
    }

    #[test]
    fn redeclared_source_types_get_distinct_logical_keys() {
        let db = test_db();
        let ctx = NominalTypes::new(&db);
        let path = trunk_ir::SymbolPath::new(["A"]);
        let named = |disambiguator| {
            AstType::new(
                &db,
                TypeKind::Named {
                    id: crate::ast::TypeDefId::declared(&db, path.clone(), disambiguator),
                    name: Symbol::new("A"),
                    args: vec![],
                },
            )
        };

        assert_ne!(
            ctx.logical_type_key(named(0)),
            ctx.logical_type_key(named(1))
        );
    }

    #[test]
    fn nominal_origin_keys_tell_every_origin_apart() {
        use crate::ast::{BuiltinType, TypeOrigin};

        let keys = [
            TypeOrigin::Source { disambiguator: 0 },
            TypeOrigin::Source { disambiguator: 1 },
            TypeOrigin::Builtin(BuiltinType::List),
            TypeOrigin::Builtin(BuiltinType::Slots),
            TypeOrigin::Builtin(BuiltinType::Dispatch),
            TypeOrigin::Synthetic,
        ]
        .map(nominal_origin_key);
        assert_eq!(
            keys,
            [
                "source",
                "source1",
                "builtin_list",
                "builtin_slots",
                "builtin_dispatch",
                "synthetic"
            ]
        );
    }
}
