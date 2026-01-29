//! Declaration collection phase.
//!
//! Populates `ModuleTypeEnv` with function signatures, constructor types,
//! and type definitions before type checking function bodies.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    is_type_variable, Decl, EffectRow, EffectVar, EnumDecl, FuncDecl, FuncDefId, Module,
    ResolvedRef, StructDecl, Type, TypeKind, TypeParam, TypeScheme,
};

use super::TypeChecker;

impl<'db> TypeChecker<'db> {
    // =========================================================================
    // Declaration collection (Phase 1)
    // =========================================================================

    /// Collect type definitions and function signatures from declarations.
    pub(crate) fn collect_declarations(&mut self, module: &Module<ResolvedRef<'db>>) {
        for decl in &module.decls {
            match decl {
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
                Decl::Ability(_) | Decl::Use(_) => {
                    // Abilities and imports don't define types directly
                }
                Decl::Module(m) => {
                    // For inline modules, recursively collect from nested declarations
                    if let Some(body) = &m.body {
                        // Push module name to path
                        self.module_path.push(m.name);
                        // Create a temporary module to reuse collect_declarations
                        let inner_module = Module {
                            id: m.id,
                            name: Some(m.name),
                            decls: body.clone(),
                        };
                        self.collect_declarations(&inner_module);
                        // Pop module name from path
                        self.module_path.pop();
                    }
                }
            }
        }
    }

    /// Collect a function's type signature.
    ///
    /// This is called during declaration collection (Phase 1) to register
    /// function signatures in ModuleTypeEnv. Type variables in annotations
    /// are converted to BoundVars here, since actual type inference happens
    /// per-function in check_func_decl.
    fn collect_function_signature(&mut self, func: &FuncDecl<ResolvedRef<'db>>) {
        // Per-function map: same lowercase name → same BoundVar index
        let mut type_var_map: HashMap<Symbol, u32> = HashMap::new();
        let mut next_bound_var: u32 = 0;

        // Build parameter types from annotations
        let param_types: Vec<Type<'db>> = func
            .params
            .iter()
            .map(|p| match &p.ty {
                Some(ann) => self.annotation_to_type_for_sig(ann, &mut type_var_map, &mut next_bound_var),
                // No annotation: use a fresh BoundVar (will be inferred during function body check)
                None => {
                    let index = next_bound_var;
                    next_bound_var += 1;
                    Type::new(self.db(), TypeKind::BoundVar { index })
                }
            })
            .collect();

        // Build return type from annotation
        let return_ty = func
            .return_ty
            .as_ref()
            .map(|ann| self.annotation_to_type_for_sig(ann, &mut type_var_map, &mut next_bound_var))
            .unwrap_or_else(|| {
                // No annotation: use a fresh BoundVar
                let index = next_bound_var;
                next_bound_var += 1;
                Type::new(self.db(), TypeKind::BoundVar { index })
            });

        // Build effect row from annotations (effects don't have BoundVars for now)
        let effect = match &func.effects {
            Some(anns) => {
                // For effects, we use a closed row during collection
                crate::ast::abilities_to_effect_row(
                    self.db(),
                    anns,
                    &mut |ann| self.annotation_to_type_for_sig(ann, &mut type_var_map, &mut next_bound_var),
                    || EffectVar { id: 0 }, // Placeholder, will be replaced during function check
                )
            }
            None => EffectRow::pure(self.db()),
        };

        // Create function type
        let func_ty = self.env.func_type(param_types, return_ty, effect);

        // Build type params from the collected BoundVars
        let type_params: Vec<TypeParam> = (0..next_bound_var)
            .map(|_| TypeParam::anonymous())
            .collect();

        let scheme = TypeScheme::new(self.db(), type_params, func_ty);

        // Register the function with its FuncDefId (using qualified name)
        let qualified_name = self.qualified_func_name(func.name);
        let func_id = FuncDefId::new(self.db(), qualified_name);
        self.env.register_function(func_id, scheme);
    }

    /// Collect an extern function's type signature.
    ///
    /// Extern functions have no body, so we only need to register the type.
    /// Uses BoundVars for type parameters.
    fn collect_extern_function_signature(&mut self, func: &crate::ast::ExternFuncDecl) {
        let mut type_var_map: HashMap<Symbol, u32> = HashMap::new();
        let mut next_bound_var: u32 = 0;

        let param_types: Vec<Type<'db>> = func
            .params
            .iter()
            .map(|p| match &p.ty {
                Some(ann) => self.annotation_to_type_for_sig(ann, &mut type_var_map, &mut next_bound_var),
                None => {
                    let index = next_bound_var;
                    next_bound_var += 1;
                    Type::new(self.db(), TypeKind::BoundVar { index })
                }
            })
            .collect();

        let return_ty = self.annotation_to_type_for_sig(&func.return_ty, &mut type_var_map, &mut next_bound_var);

        let effect = EffectRow::pure(self.db());
        let func_ty = self.env.func_type(param_types, return_ty, effect);

        let type_params: Vec<TypeParam> = (0..next_bound_var)
            .map(|_| TypeParam::anonymous())
            .collect();

        let scheme = TypeScheme::new(self.db(), type_params, func_ty);

        let func_id = FuncDefId::new(self.db(), func.name);
        self.env.register_function(func_id, scheme);
    }

    /// Collect a struct definition.
    fn collect_struct_def(&mut self, s: &StructDecl) {
        let name = s.name;
        let type_params: Vec<TypeParam> = s
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name))
            .collect();

        // The struct type itself
        let args: Vec<Type<'db>> = (0..type_params.len() as u32)
            .map(|i| Type::new(self.db(), TypeKind::BoundVar { index: i }))
            .collect();
        let struct_ty = self.env.named_type(name, args);

        let scheme = TypeScheme::new(self.db(), type_params, struct_ty);
        self.env.register_type_def(name, scheme);
    }

    /// Collect an enum definition.
    fn collect_enum_def(&mut self, e: &EnumDecl) {
        let name = e.name;
        let type_params: Vec<TypeParam> = e
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name))
            .collect();

        // The enum type itself
        let args: Vec<Type<'db>> = (0..type_params.len() as u32)
            .map(|i| Type::new(self.db(), TypeKind::BoundVar { index: i }))
            .collect();
        let enum_ty = self.env.named_type(name, args);

        let scheme = TypeScheme::new(self.db(), type_params.clone(), enum_ty);
        self.env.register_type_def(name, scheme);

        // Register constructors for each variant
        // Build name → BoundVar index lookup for type parameter resolution
        let type_param_indices: Vec<(Symbol, u32)> = e
            .type_params
            .iter()
            .enumerate()
            .map(|(i, tp)| (tp.name, i as u32))
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
                let effect = EffectRow::pure(self.db());
                self.env.func_type(field_types, enum_ty, effect)
            };

            let ctor_scheme = TypeScheme::new(self.db(), type_params.clone(), ctor_ty);
            let ctor_id = crate::ast::CtorId::new(self.db(), variant.name);
            self.env.register_constructor(ctor_id, ctor_scheme);
        }
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
        type_var_map: &mut HashMap<Symbol, u32>,
        next_bound_var: &mut u32,
    ) -> Type<'db> {
        use crate::ast::TypeAnnotationKind;

        match &ann.kind {
            TypeAnnotationKind::Named(name) if is_type_variable(name) => {
                // Lowercase name → BoundVar
                if let Some(&index) = type_var_map.get(name) {
                    Type::new(self.db(), TypeKind::BoundVar { index })
                } else {
                    let index = *next_bound_var;
                    *next_bound_var += 1;
                    type_var_map.insert(*name, index);
                    Type::new(self.db(), TypeKind::BoundVar { index })
                }
            }
            TypeAnnotationKind::Named(name) => self.primitive_or_named_type(*name),
            TypeAnnotationKind::Path(parts) if !parts.is_empty() => {
                if let Some(&name) = parts.last() {
                    self.env.named_type(name, vec![])
                } else {
                    self.env.error_type()
                }
            }
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_for_sig(ctor, type_var_map, next_bound_var);
                if let TypeKind::Named { name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_for_sig(a, type_var_map, next_bound_var))
                        .collect();
                    self.env.named_type(*name, arg_types)
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
                    .map(|p| self.annotation_to_type_for_sig(p, type_var_map, next_bound_var))
                    .collect();
                let result_ty = self.annotation_to_type_for_sig(result, type_var_map, next_bound_var);
                let effect = crate::ast::abilities_to_effect_row(
                    self.db(),
                    abilities,
                    &mut |a| self.annotation_to_type_for_sig(a, type_var_map, next_bound_var),
                    || EffectVar { id: 0 },
                );
                self.env.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.annotation_to_type_for_sig(e, type_var_map, next_bound_var))
                    .collect();
                self.env.tuple_type(elem_types)
            }
            TypeAnnotationKind::Infer => {
                // Infer: use a fresh BoundVar
                let index = *next_bound_var;
                *next_bound_var += 1;
                Type::new(self.db(), TypeKind::BoundVar { index })
            }
            TypeAnnotationKind::Path(_) | TypeAnnotationKind::Error => self.env.error_type(),
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
                self.primitive_or_named_type(*name)
            }
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_for_ctor(ctor, type_param_indices);
                if let TypeKind::Named { name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_for_ctor(a, type_param_indices))
                        .collect();
                    self.env.named_type(*name, arg_types)
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
                let effect = crate::ast::abilities_to_effect_row(
                    self.db(),
                    abilities,
                    &mut |a| self.annotation_to_type_for_ctor(a, type_param_indices),
                    || EffectVar { id: 0 },
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
            TypeAnnotationKind::Path(parts) if !parts.is_empty() => {
                if let Some(&name) = parts.last() {
                    self.env.named_type(name, vec![])
                } else {
                    self.env.error_type()
                }
            }
            TypeAnnotationKind::Infer | TypeAnnotationKind::Path(_) | TypeAnnotationKind::Error => {
                self.env.error_type()
            }
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
        } else if name == "String" {
            self.env.string_type()
        } else if name == "Bytes" {
            self.env.bytes_type()
        } else if name == "Rune" {
            self.env.rune_type()
        } else if name == "()" {
            self.env.nil_type()
        } else {
            self.env.named_type(name, vec![])
        }
    }
}
