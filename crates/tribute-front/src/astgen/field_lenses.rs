//! Field setters and modifiers a struct declaration generates.
//!
//! `struct T { f: F }` generates `T::f::set` and `T::f::modify` as ordinary
//! functions in a generated module beside the struct, so every later phase
//! handles them as it handles source functions:
//!
//! ```text
//! mod T {
//!     mod f {
//!         fn set(it: T, value: F) ->{} T { T { ..it, f: value } }
//!         fn modify(it: T, update: fn(F) ->{e} F) ->{e} T { T { ..it, f: update(it.f) } }
//!     }
//! }
//! ```

use trunk_ir::Symbol;

use crate::ast::{
    Decl, Expr, ExprKind, FIELD_LENS_FUNCTIONS, FieldDecl, FuncDecl, ModuleDecl, NodeId, ParamDecl,
    StructDecl, TypeAnnotation, TypeAnnotationKind, UnresolvedName,
};

use super::context::AstLoweringCtx;

/// The module holding the setters and modifiers of `declaration`'s named
/// fields; `None` for a struct without one.
pub(super) fn field_lens_module(
    ctx: &mut AstLoweringCtx<'_>,
    declaration: &StructDecl,
) -> Option<ModuleDecl<UnresolvedName>> {
    let fields: Vec<Decl<UnresolvedName>> = declaration
        .fields
        .iter()
        .filter_map(|field| {
            let name = field.name.clone()?;
            let mut lens = Lens {
                ctx,
                declaration,
                field,
                name: name.clone(),
                rows: unused_row_variables(declaration)
                    .take(1 + omitted_rows(&field.ty))
                    .collect(),
            };
            let body = vec![
                Decl::Function(lens.setter()),
                Decl::Function(lens.modifier()),
            ];
            Some(Decl::Module(ModuleDecl {
                id: ctx.synthetic_id(field.id),
                name,
                is_pub: true,
                generated: true,
                body: Some(body),
            }))
        })
        .collect();
    if fields.is_empty() {
        return None;
    }
    Some(ModuleDecl {
        id: ctx.synthetic_id(declaration.id),
        name: declaration.name.clone(),
        is_pub: true,
        generated: true,
        body: Some(fields),
    })
}

/// Row variable names the struct's own annotations do not use.
fn unused_row_variables(declaration: &StructDecl) -> impl Iterator<Item = Symbol> + '_ {
    fn uses(ann: &TypeAnnotation, name: &Symbol) -> bool {
        match &ann.kind {
            TypeAnnotationKind::Named(used) => used == name,
            TypeAnnotationKind::App { ctor, args } => {
                uses(ctor, name) || args.iter().any(|arg| uses(arg, name))
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => params.iter().chain(abilities).any(|ann| uses(ann, name)) || uses(result, name),
            TypeAnnotationKind::Tuple(elements) => elements.iter().any(|ann| uses(ann, name)),
            TypeAnnotationKind::Path(_) | TypeAnnotationKind::Infer | TypeAnnotationKind::Error => {
                false
            }
        }
    }
    (0usize..)
        .map(|index| match index {
            0 => Symbol::new("e"),
            _ => Symbol::new(&format!("e{index}")),
        })
        .filter(|name| {
            declaration
                .type_params
                .iter()
                .all(|param| param.name != *name)
                && declaration
                    .fields
                    .iter()
                    .all(|field| !uses(&field.ty, name))
        })
}

/// How many function types in `ann` omit their effect row.
fn omitted_rows(ann: &TypeAnnotation) -> usize {
    match &ann.kind {
        TypeAnnotationKind::App { ctor, args } => {
            omitted_rows(ctor) + args.iter().map(omitted_rows).sum::<usize>()
        }
        TypeAnnotationKind::Func {
            params,
            result,
            abilities,
        } => {
            let rows: usize = abilities
                .iter()
                .map(|ability| match ability.kind {
                    TypeAnnotationKind::Infer => 1,
                    _ => omitted_rows(ability),
                })
                .sum();
            rows + params.iter().map(omitted_rows).sum::<usize>() + omitted_rows(result)
        }
        TypeAnnotationKind::Tuple(elements) => elements.iter().map(omitted_rows).sum(),
        TypeAnnotationKind::Named(_)
        | TypeAnnotationKind::Path(_)
        | TypeAnnotationKind::Infer
        | TypeAnnotationKind::Error => 0,
    }
}

struct Lens<'a, 'db> {
    ctx: &'a mut AstLoweringCtx<'db>,
    declaration: &'a StructDecl,
    field: &'a FieldDecl,
    name: Symbol,
    /// The modifier's callback row, then a name for each effect row the
    /// field's type omits. A signature repeats the field's type, and each
    /// occurrence of an omitted row must be the same row.
    rows: Vec<Symbol>,
}

impl Lens<'_, '_> {
    fn id(&mut self) -> NodeId {
        self.ctx.synthetic_id(self.field.id)
    }

    fn named(&mut self, name: Symbol) -> TypeAnnotation {
        TypeAnnotation {
            id: self.id(),
            kind: TypeAnnotationKind::Named(name),
        }
    }

    /// The struct applied to its own type parameters.
    fn struct_type(&mut self) -> TypeAnnotation {
        let ctor = self.named(self.declaration.name.clone());
        if self.declaration.type_params.is_empty() {
            return ctor;
        }
        let args = self
            .declaration
            .type_params
            .iter()
            .map(|param| self.named(param.name.clone()))
            .collect();
        TypeAnnotation {
            id: self.id(),
            kind: TypeAnnotationKind::App {
                ctor: Box::new(ctor),
                args,
            },
        }
    }

    /// The field's type annotation as nodes of its own, naming the effect
    /// rows it omits.
    fn field_type(&mut self) -> TypeAnnotation {
        let mut ty = self.field.ty.clone();
        self.renumber(&mut ty, &mut 1);
        ty
    }

    fn renumber(&mut self, ann: &mut TypeAnnotation, row: &mut usize) {
        ann.id = self.id();
        match &mut ann.kind {
            TypeAnnotationKind::App { ctor, args } => {
                self.renumber(ctor, row);
                args.iter_mut().for_each(|arg| self.renumber(arg, row));
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                for ability in abilities {
                    if matches!(ability.kind, TypeAnnotationKind::Infer) {
                        ability.id = self.id();
                        ability.kind = TypeAnnotationKind::Named(self.rows[*row].clone());
                        *row += 1;
                    } else {
                        self.renumber(ability, row);
                    }
                }
                params.iter_mut().for_each(|ann| self.renumber(ann, row));
                self.renumber(result, row);
            }
            TypeAnnotationKind::Tuple(elements) => {
                elements.iter_mut().for_each(|ann| self.renumber(ann, row));
            }
            TypeAnnotationKind::Named(_)
            | TypeAnnotationKind::Path(_)
            | TypeAnnotationKind::Infer
            | TypeAnnotationKind::Error => {}
        }
    }

    fn param(&mut self, name: &str, ty: TypeAnnotation) -> ParamDecl {
        ParamDecl {
            id: self.id(),
            name: Symbol::new(name),
            ty: Some(ty),
            local_id: None,
        }
    }

    fn var(&mut self, name: &str) -> Expr<UnresolvedName> {
        let id = self.id();
        Expr::new(
            id,
            ExprKind::Var(UnresolvedName::new(Symbol::new(name), id)),
        )
    }

    /// `T { ..it, f: value }` as a function body.
    fn update(&mut self, value: Expr<UnresolvedName>) -> Expr<UnresolvedName> {
        let type_name = UnresolvedName::new(self.declaration.name.clone(), self.id());
        let spread = self.var(SUBJECT);
        let record = Expr::new(
            self.id(),
            ExprKind::Record {
                type_name,
                fields: vec![(self.name.clone(), value)],
                spread: Some(spread),
            },
        );
        Expr::new(
            self.id(),
            ExprKind::Block {
                stmts: Vec::new(),
                value: record,
            },
        )
    }

    fn function(
        &mut self,
        name: &str,
        argument: ParamDecl,
        effects: Vec<TypeAnnotation>,
        body: Expr<UnresolvedName>,
    ) -> FuncDecl<UnresolvedName> {
        let subject_type = self.struct_type();
        let subject = self.param(SUBJECT, subject_type);
        FuncDecl {
            id: self.id(),
            is_pub: true,
            name: Symbol::new(name),
            type_params: Vec::new(),
            params: vec![subject, argument],
            return_ty: Some(self.struct_type()),
            effects: Some(effects),
            body,
        }
    }

    fn setter(&mut self) -> FuncDecl<UnresolvedName> {
        let ty = self.field_type();
        let argument = self.param("value", ty);
        let value = self.var("value");
        let body = self.update(value);
        self.function(FIELD_LENS_FUNCTIONS[0], argument, Vec::new(), body)
    }

    fn modifier(&mut self) -> FuncDecl<UnresolvedName> {
        let row = self.rows[0].clone();
        let callback = TypeAnnotation {
            id: self.id(),
            kind: TypeAnnotationKind::Func {
                params: vec![self.field_type()],
                result: Box::new(self.field_type()),
                abilities: vec![self.named(row.clone())],
            },
        };
        let argument = self.param("update", callback);
        let current = Expr::new(
            self.id(),
            ExprKind::MethodCall {
                receiver: self.var(SUBJECT),
                method: self.name.clone(),
                path: None,
                args: Vec::new(),
            },
        );
        let value = Expr::new(
            self.id(),
            ExprKind::Call {
                callee: self.var("update"),
                args: vec![current],
            },
        );
        let body = self.update(value);
        let effects = vec![self.named(row.clone())];
        self.function(FIELD_LENS_FUNCTIONS[1], argument, effects, body)
    }
}

/// The parameter holding the struct value.
const SUBJECT: &str = "it";
