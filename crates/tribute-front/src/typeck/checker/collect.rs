//! Declaration collection phase.
//!
//! Populates `ModuleTypeEnv` with function signatures, constructor types,
//! and type definitions before type checking function bodies.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::{Symbol, SymbolPath};

use crate::ast::{
    AbilityDecl, AbilityId, AbilityRow, CtorId, Decl, DeclTable, EnumDecl, FuncDecl, ResolvedRef,
    RowVar, StructDecl, Type, TypeKind, TypeParam, TypeScheme, collect_row_vars, is_type_variable,
};
use crate::typeck::context::{AbilityInfo, AbilityOpInfo};

use super::TypeChecker;

/// Declared field names in order, if every field is named. A constructor
/// without fields has an empty list.
fn field_names(fields: &[crate::ast::FieldDecl]) -> Option<Vec<Symbol>> {
    fields.iter().map(|field| field.name.clone()).collect()
}

#[derive(Default)]
struct SignatureVariables<'db> {
    types: HashMap<Symbol, u32>,
    next_type: u32,
    rows: HashMap<Symbol, RowVar>,
    next_row: u64,
    unions: Vec<crate::ast::RowUnion<'db>>,
}

impl SignatureVariables<'_> {
    /// The scheme's type parameters, each named by the variable the
    /// signature spells for it.
    fn type_params(&self) -> Vec<TypeParam> {
        let mut params = vec![TypeParam::anonymous(); self.next_type as usize];
        for (name, &index) in &self.types {
            params[index as usize] = TypeParam::named(name.clone());
        }
        params
    }

    fn fresh_row(&mut self) -> RowVar {
        // Zero is reserved for the implicit, omitted ability annotation.
        self.next_row += 1;
        RowVar { id: self.next_row }
    }
}

impl<'db> TypeChecker<'db> {
    // =========================================================================
    // Declaration collection (Phase 1)
    // =========================================================================

    /// Collect type definitions and function signatures from declarations.
    pub(crate) fn collect_declarations(&mut self, module: &DeclTable<ResolvedRef<'db>>) {
        self.predeclare_nominal_types(module);
        self.collect_type_imports(module);
        self.collect_declarations_in_order(module);
        self.report_getter_redeclarations();
    }

    /// Report a function declared under the name a struct's field already
    /// gives its getter, such as `text` in the companion module of a struct
    /// with a field `text`.
    fn report_getter_redeclarations(&mut self) {
        for (node, function) in std::mem::take(&mut self.function_decls) {
            let Some(getter) = self.env.field_function(function) else {
                continue;
            };
            Diagnostic::new(
                format!(
                    "function `{}` is already declared by field `{}` of struct `{}`",
                    function.qualified(self.db()),
                    getter.field,
                    getter.owner.name(self.db()),
                ),
                self.get_span(node),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }
    }

    /// Import a type scheme without inventing a nominal identity for its alias.
    /// Local nominal declarations retain precedence, including forward ones.
    fn collect_type_imports(&mut self, module: &DeclTable<ResolvedRef<'db>>) {
        let nominals: HashSet<(&SymbolPath, &Symbol)> = module
            .entries
            .iter()
            .filter_map(|entry| match &entry.decl {
                Decl::Struct(decl) => Some((&entry.module, &decl.name)),
                Decl::Enum(decl) => Some((&entry.module, &decl.name)),
                _ => None,
            })
            .collect();
        for entry in &module.entries {
            let Decl::Use(import) = &entry.decl else {
                continue;
            };
            // Name resolution recorded what the import names; a path
            // written relative to the enclosing module must not be
            // reread from the package root.
            let Some(target) = import.target.as_deref().filter(|target| !target.is_empty()) else {
                continue;
            };
            let Some(scheme) = self.env.lookup_type_def(target) else {
                continue;
            };
            let name = import
                .alias
                .clone()
                .unwrap_or(import.path.last().unwrap().clone());
            if nominals.contains(&(&entry.module, &name)) {
                continue;
            }
            let qualified = entry.module.child(name);
            self.env.register_type_def(qualified, scheme);
        }
    }

    fn collect_declarations_in_order(&mut self, module: &DeclTable<ResolvedRef<'db>>) {
        for entry in &module.entries {
            self.prefix = entry.module.clone();
            match &entry.decl {
                Decl::Function(func) => {
                    self.collect_function_signature(func);
                }
                Decl::Struct(s) => {
                    self.collect_struct_def(s);
                }
                Decl::Enum(e) => {
                    self.collect_enum_def(e);
                }
                Decl::ExternFunction(func) => {
                    self.collect_extern_function_signature(func);
                }
                Decl::Ability(a) => {
                    self.collect_ability_def(a);
                }
                Decl::Use(_) => {
                    // Imports don't define types directly
                }
            }
        }
        self.prefix = SymbolPath::root();
    }

    /// Register every nominal declaration before converting any annotations.
    fn predeclare_nominal_types(&mut self, module: &DeclTable<ResolvedRef<'db>>) {
        for entry in &module.entries {
            self.prefix = entry.module.clone();
            match &entry.decl {
                Decl::Struct(s) => {
                    self.predeclare_nominal_type(&s.name, s.disambiguator, &s.type_params);
                }
                Decl::Enum(e) => {
                    self.predeclare_nominal_type(&e.name, e.disambiguator, &e.type_params);
                }
                _ => {}
            }
        }
        self.prefix = SymbolPath::root();
    }

    fn predeclare_nominal_type(
        &mut self,
        name: &Symbol,
        disambiguator: u32,
        params: &[crate::ast::TypeParamDecl],
    ) {
        let qualified = self.current_prefix().child(name.clone());
        let type_params: Vec<TypeParam> = params
            .iter()
            .map(|param| TypeParam::named(param.name.clone()))
            .collect();
        let args = (0..type_params.len() as u32)
            .map(|index| Type::new(self.db(), TypeKind::BoundVar { index }))
            .collect();
        let ty = self.env.named_type_with_id(
            crate::ast::TypeDefId::declared(self.db(), qualified.clone(), disambiguator),
            qualified.to_symbol(),
            args,
        );
        self.env.register_type_def(
            qualified,
            TypeScheme::new(self.db(), type_params, Vec::new(), ty),
        );
    }

    /// Collect a function's type signature.
    ///
    /// This is called during declaration collection (Phase 1) to register
    /// function signatures in ModuleTypeEnv. Type variables in annotations
    /// are converted to BoundVars here. The declared signature is the
    /// function's final scheme, so every parameter type and the return type
    /// must be written; a missing one is reported and typed as an error.
    fn collect_function_signature(&mut self, func: &FuncDecl<ResolvedRef<'db>>) {
        // Per-function map: same lowercase name → same BoundVar index
        let mut vars = SignatureVariables::default();

        // Build parameter types from annotations
        let param_types: Vec<Type<'db>> = func
            .params
            .iter()
            .map(|p| match &p.ty {
                Some(ann) => self.annotation_to_type_for_sig(ann, &mut vars),
                None => self.missing_signature_type(
                    p.id,
                    format!(
                        "function `{}` needs a type annotation for parameter `{}`",
                        func.name, p.name
                    ),
                ),
            })
            .collect();

        // Build return type from annotation
        let return_ty = match &func.return_ty {
            Some(ann) => self.annotation_to_type_for_sig(ann, &mut vars),
            None => self.missing_signature_type(
                func.id,
                format!("function `{}` needs a return type", func.name),
            ),
        };
        // Build ability row from annotations (effects don't have BoundVars for now)
        let (effect, effect_origins) = match &func.effects {
            Some(anns) => {
                let conversion = self.effects_for_signature(anns, &mut vars);
                (conversion.row, Some(conversion.origins))
            }
            // Omitting the annotation is effect-polymorphic (`->{e}`), not
            // an assertion that the row is closed and empty.
            None => (AbilityRow::open(self.db(), RowVar { id: 0 }), None),
        };

        let func_ty = self.env.func_type(param_types, return_ty, effect);

        // Build type params from the collected BoundVars
        let type_params = vars.type_params();

        let mut effect_params = collect_row_vars(self.db(), func_ty);
        for union in &vars.unions {
            for row in union.sources.iter().chain(std::iter::once(&union.result)) {
                if let Some(var) = row.rest(self.db())
                    && !effect_params.contains(&var)
                {
                    effect_params.push(var);
                }
            }
        }
        let scheme = TypeScheme::builder(type_params, effect_params, func_ty)
            .row_unions(vars.unions)
            .build(self.db());

        // Register the function with its FuncDefId
        let func_id = self.func_def_id(&func.name);
        if let Some(origins) = effect_origins {
            self.effect_annotation_origins.insert(func_id, origins);
        }
        self.signature_row_names.insert(func_id, vars.rows);
        self.signature_type_names.insert(func_id, vars.types);
        self.env.register_function(func_id, scheme);

        self.function_decls.push((func.id, func_id));
    }

    /// Report a missing part of a function signature, typed as an error.
    fn missing_signature_type(&self, node: crate::ast::NodeId, message: String) -> Type<'db> {
        Diagnostic::new(
            message,
            self.get_span(node),
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        )
        .accumulate(self.db());
        Type::new(self.db(), TypeKind::Error)
    }

    /// Collect an extern function's type signature.
    ///
    /// Extern functions have no body, so we only need to register the type.
    /// Uses BoundVars for type parameters.
    fn collect_extern_function_signature(&mut self, func: &crate::ast::ExternFuncDecl) {
        let mut vars = SignatureVariables::default();

        let param_types: Vec<Type<'db>> = func
            .params
            .iter()
            .map(|p| match &p.ty {
                Some(ann) => self.annotation_to_type_for_sig(ann, &mut vars),
                None => {
                    let index = vars.next_type;
                    vars.next_type += 1;
                    Type::new(self.db(), TypeKind::BoundVar { index })
                }
            })
            .collect();

        let return_ty = self.annotation_to_type_for_sig(&func.return_ty, &mut vars);

        let effect = AbilityRow::pure(self.db());
        let func_ty = self.env.func_type(param_types, return_ty, effect);

        let type_params = vars.type_params();

        let mut effect_params = collect_row_vars(self.db(), func_ty);
        for union in &vars.unions {
            for row in union.sources.iter().chain(std::iter::once(&union.result)) {
                if let Some(var) = row.rest(self.db())
                    && !effect_params.contains(&var)
                {
                    effect_params.push(var);
                }
            }
        }
        let scheme = TypeScheme::builder(type_params, effect_params, func_ty)
            .row_unions(vars.unions)
            .build(self.db());

        // Register the extern function with its FuncDefId
        let func_id = self.func_def_id(&func.name);
        self.env.register_function(func_id, scheme);
        self.env.register_extern_function(func_id);

        self.function_decls.push((func.id, func_id));
    }

    /// Collect a struct definition.
    fn collect_struct_def(&mut self, s: &StructDecl) {
        let name = s.name.clone();
        let qualified_name = self.current_prefix().child(name.clone());
        let type_params: Vec<TypeParam> = s
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name.clone()))
            .collect();

        // Build name → BoundVar index lookup for field type resolution
        let type_param_indices: Vec<(Symbol, u32)> = s
            .type_params
            .iter()
            .enumerate()
            .map(|(i, tp)| (tp.name.clone(), i as u32))
            .collect();

        // The struct type itself
        let args: Vec<Type<'db>> = (0..type_params.len() as u32)
            .map(|i| Type::new(self.db(), TypeKind::BoundVar { index: i }))
            .collect();
        let struct_ty = self.env.named_type_with_id(
            crate::ast::TypeDefId::declared(self.db(), qualified_name.clone(), s.disambiguator),
            qualified_name.to_symbol(),
            args,
        );

        let scheme = TypeScheme::new(self.db(), type_params.clone(), Vec::new(), struct_ty);
        self.env.register_type_def(qualified_name.clone(), scheme);

        // Register struct constructor
        // Constructor type is: fn(field_types...) -> StructType
        let field_types: Vec<Type<'db>> = s
            .fields
            .iter()
            .map(|f| self.annotation_to_type_for_ctor(&f.ty, &type_param_indices))
            .collect();
        let fields: Vec<(Symbol, Type<'db>)> = s
            .fields
            .iter()
            .zip(&field_types)
            .filter_map(|(f, ty)| Some((f.name.clone()?, *ty)))
            .collect();

        let ctor_ty = if field_types.is_empty() {
            // Unit struct: constructor type is just the struct type
            struct_ty
        } else {
            // Field struct: constructor type is fn(fields...) -> struct_ty
            let effect = AbilityRow::pure(self.db());
            self.env.func_type(field_types, struct_ty, effect)
        };

        let effect_params = collect_row_vars(self.db(), ctor_ty);
        let ctor_scheme = TypeScheme::new(self.db(), type_params.clone(), effect_params, ctor_ty);
        let ctor_id = CtorId::new(self.db(), qualified_name.clone());
        self.env.register_constructor(ctor_id, ctor_scheme);
        if let Some(names) = field_names(&s.fields) {
            self.env.register_constructor_field_names(ctor_id, names);
        }

        // Register struct field information for accessor resolution
        self.env.register_struct_fields(
            crate::ast::TypeDefId::declared(self.db(), qualified_name, s.disambiguator),
            type_params,
            fields,
        );
    }

    /// Collect an enum definition.
    fn collect_enum_def(&mut self, e: &EnumDecl) {
        let name = e.name.clone();
        let qualified_name = self.current_prefix().child(name.clone());
        let type_params: Vec<TypeParam> = e
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name.clone()))
            .collect();

        // The enum type itself
        let args: Vec<Type<'db>> = (0..type_params.len() as u32)
            .map(|i| Type::new(self.db(), TypeKind::BoundVar { index: i }))
            .collect();
        let enum_ty = self.env.named_type_with_id(
            crate::ast::TypeDefId::declared(self.db(), qualified_name.clone(), e.disambiguator),
            qualified_name.to_symbol(),
            args,
        );

        let scheme = TypeScheme::new(self.db(), type_params.clone(), Vec::new(), enum_ty);
        self.env.register_type_def(qualified_name.clone(), scheme);

        // Register enum variant names for exhaustiveness checking
        let variant_names: Vec<Symbol> = e.variants.iter().map(|v| v.name.clone()).collect();
        self.env
            .register_enum_variants(qualified_name.clone(), variant_names);

        // Register constructors for each variant
        // Build name → BoundVar index lookup for type parameter resolution
        let type_param_indices: Vec<(Symbol, u32)> = e
            .type_params
            .iter()
            .enumerate()
            .map(|(i, tp)| (tp.name.clone(), i as u32))
            .collect();

        for variant in &e.variants {
            let ctor_ty = if variant.fields.is_empty() {
                // Unit variant: constructor type is just the enum type
                enum_ty
            } else {
                // Field variant: constructor type is fn(field_types...) -> enum_ty
                let field_types: Vec<Type<'db>> = variant
                    .fields
                    .iter()
                    .map(|f| self.annotation_to_type_for_ctor(&f.ty, &type_param_indices))
                    .collect();
                let effect = AbilityRow::pure(self.db());
                self.env.func_type(field_types, enum_ty, effect)
            };

            let effect_params = collect_row_vars(self.db(), ctor_ty);
            let ctor_scheme =
                TypeScheme::new(self.db(), type_params.clone(), effect_params, ctor_ty);
            let ctor_id = CtorId::new(self.db(), qualified_name.child(variant.name.clone()));
            self.env.register_constructor(ctor_id, ctor_scheme);
            if let Some(names) = field_names(&variant.fields) {
                self.env.register_constructor_field_names(ctor_id, names);
            }
        }
    }

    /// Collect an ability definition.
    ///
    /// Registers the ability's operations so they can be looked up during
    /// handler arm type checking.
    fn collect_ability_def(&mut self, a: &AbilityDecl) {
        // Create AbilityId for this ability
        let ability_id = AbilityId::source(self.db(), self.current_prefix().child(a.name.clone()));

        // Build type parameter info
        let type_params: Vec<TypeParam> = a
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name.clone()))
            .collect();

        // Build name → BoundVar index lookup for operation type resolution
        let type_param_indices: Vec<(Symbol, u32)> = a
            .type_params
            .iter()
            .enumerate()
            .map(|(i, tp)| (tp.name.clone(), i as u32))
            .collect();

        // Collect operation signatures
        let mut operations = HashMap::default();
        for op in &a.operations {
            let param_types: Vec<Type<'db>> = op
                .params
                .iter()
                .map(|p| {
                    p.ty.as_ref()
                        .map(|ann| self.annotation_to_type_for_ctor(ann, &type_param_indices))
                        .unwrap_or_else(|| self.env.error_type())
                })
                .collect();

            let return_type = self.annotation_to_type_for_ctor(&op.return_ty, &type_param_indices);

            operations.insert(
                op.name.clone(),
                AbilityOpInfo {
                    name: op.name.clone(),
                    kind: op.kind,
                    param_types,
                    return_type,
                },
            );
        }

        self.env.register_ability(
            ability_id,
            AbilityInfo {
                id: ability_id,
                type_params,
                operations,
            },
        );
    }

    // =========================================================================
    // Type annotation conversion for signatures (BoundVar-based)
    // =========================================================================

    /// Convert a type annotation to a Type for signature collection.
    ///
    /// Type variable names (lowercase) are mapped to BoundVar indices.
    fn annotation_to_type_for_sig(
        &self,
        ann: &crate::ast::TypeAnnotation,
        vars: &mut SignatureVariables<'db>,
    ) -> Type<'db> {
        use crate::ast::TypeAnnotationKind;

        match &ann.kind {
            TypeAnnotationKind::Named(name) if is_type_variable(name) => {
                // Lowercase name → BoundVar
                if let Some(&index) = vars.types.get(name) {
                    Type::new(self.db(), TypeKind::BoundVar { index })
                } else {
                    let index = vars.next_type;
                    vars.next_type += 1;
                    vars.types.insert(name.clone(), index);
                    Type::new(self.db(), TypeKind::BoundVar { index })
                }
            }
            TypeAnnotationKind::Named(name) => self.primitive_or_named_type(name.clone()),
            TypeAnnotationKind::Path(parts) => self.env.path_type(parts),
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_for_sig(ctor, vars);
                if let TypeKind::Named { id, name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_for_sig(a, vars))
                        .collect();
                    self.env.named_type_with_id(*id, name.clone(), arg_types)
                } else {
                    self.env.error_type()
                }
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| self.annotation_to_type_for_sig(p, vars))
                    .collect();
                let result_ty = self.annotation_to_type_for_sig(result, vars);
                let effect = self.effects_for_signature(abilities, vars).row;
                self.env.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.annotation_to_type_for_sig(e, vars))
                    .collect();
                self.env.tuple_type(elem_types)
            }
            TypeAnnotationKind::Infer => {
                // Infer: use a fresh BoundVar
                let index = vars.next_type;
                vars.next_type += 1;
                Type::new(self.db(), TypeKind::BoundVar { index })
            }
            TypeAnnotationKind::Error => self.env.error_type(),
        }
    }

    fn effects_for_signature(
        &self,
        annotations: &[crate::ast::TypeAnnotation],
        vars: &mut SignatureVariables<'db>,
    ) -> crate::ast::AbilityRowConversion<'db> {
        use crate::ast::TypeAnnotationKind;
        let conversion = crate::ast::abilities_to_ability_row_with_origins(
            self.db(),
            annotations,
            self.current_prefix(),
            &mut |ann| self.annotation_to_type_for_sig(ann, vars),
            || RowVar { id: 0 },
        );
        let mut tails = Vec::new();
        for ann in annotations {
            let var = match &ann.kind {
                TypeAnnotationKind::Named(name) if is_type_variable(name) => {
                    if let Some(var) = vars.rows.get(name) {
                        *var
                    } else {
                        let var = vars.fresh_row();
                        vars.rows.insert(name.clone(), var);
                        var
                    }
                }
                TypeAnnotationKind::Infer => vars.fresh_row(),
                _ => continue,
            };
            if !tails.contains(&var) {
                tails.push(var);
            }
        }
        let rest = match tails.as_slice() {
            [] => None,
            [var] => Some(*var),
            _ => {
                let result = vars.fresh_row();
                vars.unions.push(crate::ast::RowUnion {
                    sources: tails
                        .into_iter()
                        .map(|var| AbilityRow::open(self.db(), var))
                        .collect(),
                    result: AbilityRow::open(self.db(), result),
                });
                Some(result)
            }
        };
        crate::ast::AbilityRowConversion {
            row: AbilityRow::new(self.db(), conversion.row.instances(self.db()), rest),
            origins: conversion.origins,
        }
    }

    /// Convert a type annotation for constructor field types, using BoundVars for type parameters.
    fn annotation_to_type_for_ctor(
        &self,
        ann: &crate::ast::TypeAnnotation,
        type_param_indices: &[(Symbol, u32)],
    ) -> Type<'db> {
        use crate::ast::TypeAnnotationKind;

        match &ann.kind {
            TypeAnnotationKind::Named(name) => {
                // Check if this name is a type parameter
                if let Some(&(_, index)) = type_param_indices.iter().find(|(n, _)| n == name) {
                    return Type::new(self.db(), TypeKind::BoundVar { index });
                }
                self.primitive_or_named_type(name.clone())
            }
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_for_ctor(ctor, type_param_indices);
                if let TypeKind::Named { id, name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_for_ctor(a, type_param_indices))
                        .collect();
                    self.env.named_type_with_id(*id, name.clone(), arg_types)
                } else {
                    self.env.error_type()
                }
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| self.annotation_to_type_for_ctor(p, type_param_indices))
                    .collect();
                let result_ty = self.annotation_to_type_for_ctor(result, type_param_indices);
                // A declaration has no parameter to name a row with, so an
                // omitted row is empty and a named one is unbound.
                let abilities: Vec<_> = abilities
                    .iter()
                    .filter(|ann| match &ann.kind {
                        TypeAnnotationKind::Infer => false,
                        TypeAnnotationKind::Named(name) if is_type_variable(name) => {
                            Diagnostic::new(
                                format!("row variable `{name}` is not bound in this declaration"),
                                self.get_span(ann.id),
                                DiagnosticSeverity::Error,
                                CompilationPhase::TypeChecking,
                            )
                            .accumulate(self.db());
                            false
                        }
                        _ => true,
                    })
                    .cloned()
                    .collect();
                let effect = crate::ast::abilities_to_ability_row(
                    self.db(),
                    &abilities,
                    self.current_prefix(),
                    &mut |a| self.annotation_to_type_for_ctor(a, type_param_indices),
                    || unreachable!("a declaration's row has no tail"),
                );
                self.env.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.annotation_to_type_for_ctor(e, type_param_indices))
                    .collect();
                self.env.tuple_type(elem_types)
            }
            TypeAnnotationKind::Path(parts) => self.env.path_type(parts),
            TypeAnnotationKind::Infer | TypeAnnotationKind::Error => self.env.error_type(),
        }
    }

    /// Convert a type name to a primitive type or named type.
    fn primitive_or_named_type(&self, name: Symbol) -> Type<'db> {
        if name == "Int" {
            self.env.int_type()
        } else if name == "Nat" {
            self.env.nat_type()
        } else if name == "Float" {
            self.env.float_type()
        } else if name == "Bool" {
            self.env.bool_type()
        } else if name == "Bytes" {
            self.env.bytes_type()
        } else if name == "Rune" {
            self.env.rune_type()
        } else if name == "Nil" {
            self.env.nil_type()
        } else if name == "Never" {
            self.env.never_type()
        } else {
            self.env
                .named_type_in_scope(SymbolPath::new([name]), vec![], self.current_prefix())
        }
    }
}
