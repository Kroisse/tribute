//! Declarative operation schemas generated from `#[dialect]` definitions.
//!
//! Every operation declared through the dialect macro exposes a static
//! [`OpSchema`] describing its operands, results, attributes, regions, and
//! successors. Schemas are registered through `inventory` so that operation
//! verification, assembly formats, and declarative rewrite tooling can look
//! them up by operation name without duplicating the definition.
//!
//! [`OpSchema::verify`] runs the operation-verifier stages described in
//! `new-plans/ir.md`: entity counts and attributes, individual type
//! constraints, type-variable bindings, projection and list relations, and
//! finally the operation's own `#[verify]` method. Each
//! stage runs only if the previous ones passed. Verification happens only at
//! explicit verifier checkpoints; parsers, raw builders, and rewrites may
//! construct operations that violate the schema in the meantime.

use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;

use crate::printer::print_type;
use crate::type_constraint::{ConstraintDesc, Projected};
use crate::types::Attribute;
use crate::{IrContext, OpRef, Symbol, TypeRef, ValueRef};

/// Static description of one dialect operation.
#[derive(Debug)]
pub struct OpSchema {
    /// Dialect name (e.g., `"arith"`).
    pub dialect: &'static str,
    /// Operation name within the dialect (e.g., `"addi"`).
    pub name: &'static str,
    /// Logical type variables declared by the typed syntax.
    pub type_vars: &'static [TypeVarSchema],
    /// Declared operands in order. At most one entry is variadic, and it is
    /// always the last one.
    pub operands: &'static [OperandSchema],
    /// Declared results.
    pub results: ResultSchema,
    /// Type constraint on the result segment.
    pub result_constraint: ValueConstraint,
    /// Declared attributes.
    pub attributes: &'static [AttributeSchema],
    /// Declared regions in order. Only the last region may be optional.
    pub regions: &'static [RegionSchema],
    /// Declared successor names in order.
    pub successors: &'static [&'static str],
    /// Calls the wrapper's `verify` method for `#[verify]` operations, after
    /// every schema check passed.
    pub verifier: Option<OpVerifier>,
}

/// An operation-local verifier generated from `#[verify]`.
pub type OpVerifier = fn(&IrContext, OpRef) -> Result<(), String>;

/// Static description of one declared operand.
#[derive(Debug)]
pub struct OperandSchema {
    pub name: &'static str,
    pub arity: Arity,
    pub constraint: ValueConstraint,
}

/// A logical type variable and the intersection of its bounds.
#[derive(Debug)]
pub struct TypeVarSchema {
    pub name: &'static str,
    pub bounds: &'static [&'static ConstraintDesc],
}

/// A resolved projection: `type_vars[var].bounds[bound].projections[index]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionRef {
    pub var: usize,
    pub bound: usize,
    pub index: usize,
}

/// Constraint on one type.
#[derive(Clone, Copy, Debug)]
pub enum TypeSpec {
    /// Unconstrained (`_`).
    Any,
    /// A named type variable; repeated uses require the same type.
    Var(usize),
    /// An anonymous variable (`impl A + B` or a direct bound path).
    Anon(&'static [&'static ConstraintDesc]),
    /// A single-type projection of a variable (`T::Input`).
    Proj(ProjectionRef),
}

/// Constraint on a type list.
#[derive(Clone, Copy, Debug)]
pub enum ListSpec {
    /// An explicit list (`(T, U)`); its length is exact.
    Types(&'static [TypeSpec]),
    /// A list projection of a variable (`S::Inputs`).
    Proj(ProjectionRef),
}

/// Type constraint on an operand or result segment.
#[derive(Clone, Copy, Debug)]
pub enum ValueConstraint {
    /// Every value satisfies the same constraint (`Value`, `Variadic`).
    Each(TypeSpec),
    /// The values match a type list exactly (`Values`).
    List(ListSpec),
}

/// Number of SSA values a declared operand or result entity binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    /// Exactly one value.
    One,
    /// Zero or more values.
    Variadic,
}

/// Static description of an operation's results.
#[derive(Debug)]
pub enum ResultSchema {
    /// Exactly the named results, in order. An empty list means no results.
    Fixed(&'static [&'static str]),
    /// Zero or more results bound to one name.
    Variadic(&'static str),
    /// Zero or one result.
    Optional(&'static str),
}

/// Static description of one declared region.
#[derive(Debug)]
pub struct RegionSchema {
    pub name: &'static str,
    /// Whether the region may be absent, e.g. the body of an external
    /// function declaration.
    pub optional: bool,
}

/// Static description of one declared attribute.
#[derive(Debug)]
pub struct AttributeSchema {
    pub name: &'static str,
    pub kind: AttributeKind,
    pub optional: bool,
    /// Type variable bound by an `Attr<S::Type>` attribute.
    pub binds: Option<usize>,
}

/// The attribute value domain accepted by a declared attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttributeKind {
    Any,
    Bool,
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
    Type,
    String,
    Symbol,
    QualifiedName,
    Bytes,
}

impl AttributeKind {
    /// Whether `attr` belongs to this kind's value domain.
    pub fn accepts(self, attr: &Attribute) -> bool {
        match (self, attr) {
            (AttributeKind::Any, _) => true,
            (AttributeKind::Bool, Attribute::Bool(_)) => true,
            (AttributeKind::I32, Attribute::Int(v)) => i32::try_from(*v).is_ok(),
            (AttributeKind::I64, Attribute::Int(v)) => i64::try_from(*v).is_ok(),
            (AttributeKind::U32, Attribute::Int(v)) => u32::try_from(*v).is_ok(),
            (AttributeKind::U64, Attribute::Int(v)) => u64::try_from(*v).is_ok(),
            (AttributeKind::F32 | AttributeKind::F64, Attribute::FloatBits(_)) => true,
            (AttributeKind::Type, Attribute::Type(_)) => true,
            (AttributeKind::String, Attribute::String(_)) => true,
            (AttributeKind::Symbol | AttributeKind::QualifiedName, Attribute::Symbol(_)) => true,
            (AttributeKind::Bytes, Attribute::Bytes(_)) => true,
            _ => false,
        }
    }
}

impl fmt::Display for AttributeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AttributeKind::Any => "any",
            AttributeKind::Bool => "bool",
            AttributeKind::I32 => "i32",
            AttributeKind::I64 => "i64",
            AttributeKind::U32 => "u32",
            AttributeKind::U64 => "u64",
            AttributeKind::F32 => "f32",
            AttributeKind::F64 => "f64",
            AttributeKind::Type => "Type",
            AttributeKind::String => "String",
            AttributeKind::Symbol => "Symbol",
            AttributeKind::QualifiedName => "QualifiedName",
            AttributeKind::Bytes => "Bytes",
        })
    }
}

/// One way an operation fails to match its schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchemaViolation {
    OperandCount {
        expected: CountRange,
        actual: usize,
    },
    ResultCount {
        expected: CountRange,
        actual: usize,
    },
    RegionCount {
        expected: CountRange,
        actual: usize,
    },
    SuccessorCount {
        expected: usize,
        actual: usize,
    },
    MissingAttribute {
        name: &'static str,
    },
    AttributeKind {
        name: &'static str,
        expected: AttributeKind,
    },
    /// A type does not satisfy a bound.
    TypeConstraint {
        site: Site,
        /// The type variable the bound belongs to, if named.
        var: Option<&'static str>,
        expected: &'static str,
        found: String,
    },
    /// A type variable was bound to a different type earlier.
    TypeMismatch {
        site: Site,
        var: &'static str,
        bound_by: Site,
        expected: String,
        found: String,
    },
    /// A type differs from a single-type projection (`T::Input`).
    ProjectionMismatch {
        site: Site,
        projection: String,
        expected: String,
        found: String,
    },
    /// A value segment differs from a list projection (`S::Inputs`).
    ListMismatch {
        segment: Segment,
        projection: String,
        expected: String,
        found: String,
    },
    /// A projection's variable has no binding in the operation.
    UnboundProjection {
        /// The operand, result, or segment the projection constrains.
        site: String,
        projection: String,
        var: &'static str,
    },
    /// The operation's own verifier rejected it.
    Verifier(String),
}

/// Where a constrained type appears in an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    Operand { index: usize, name: &'static str },
    Result { index: usize, name: &'static str },
    Attribute { name: &'static str },
}

impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Site::Operand { index, name } => write!(f, "operand #{index} `{name}`"),
            Site::Result { index, name } => write!(f, "result #{index} `{name}`"),
            Site::Attribute { name } => write!(f, "attribute `{name}`"),
        }
    }
}

/// A variadic operand or result segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    Operands(&'static str),
    Results(&'static str),
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Operands(name) => write!(f, "operands `{name}`"),
            Segment::Results(name) => write!(f, "results `{name}`"),
        }
    }
}

/// An inclusive count range; `max` is `None` when unbounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountRange {
    pub min: usize,
    pub max: Option<usize>,
}

impl CountRange {
    const fn exactly(n: usize) -> Self {
        Self {
            min: n,
            max: Some(n),
        }
    }

    fn contains(self, n: usize) -> bool {
        n >= self.min && self.max.is_none_or(|max| n <= max)
    }
}

impl fmt::Display for CountRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.max {
            Some(max) if max == self.min => write!(f, "{max}"),
            Some(max) => write!(f, "{} to {max}", self.min),
            None => write!(f, "at least {}", self.min),
        }
    }
}

impl fmt::Display for SchemaViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaViolation::OperandCount { expected, actual } => {
                write!(f, "expected {expected} operand(s), found {actual}")
            }
            SchemaViolation::ResultCount { expected, actual } => {
                write!(f, "expected {expected} result(s), found {actual}")
            }
            SchemaViolation::RegionCount { expected, actual } => {
                write!(f, "expected {expected} region(s), found {actual}")
            }
            SchemaViolation::SuccessorCount { expected, actual } => {
                write!(f, "expected {expected} successor(s), found {actual}")
            }
            SchemaViolation::MissingAttribute { name } => {
                write!(f, "missing required attribute `{name}`")
            }
            SchemaViolation::AttributeKind { name, expected } => {
                write!(f, "attribute `{name}` must be a {expected} attribute")
            }
            SchemaViolation::TypeConstraint {
                site,
                var,
                expected,
                found,
            } => match var {
                Some(var) => write!(f, "{site}: expected {var}: {expected}, found {found}"),
                None => write!(f, "{site}: expected {expected}, found {found}"),
            },
            SchemaViolation::TypeMismatch {
                site,
                var,
                bound_by,
                expected,
                found,
            } => write!(
                f,
                "{site}: expected same type as {bound_by} ({var} = {expected}), found {found}"
            ),
            SchemaViolation::ProjectionMismatch {
                site,
                projection,
                expected,
                found,
            } => write!(
                f,
                "{site}: expected {projection} = {expected}, found {found}"
            ),
            SchemaViolation::ListMismatch {
                segment,
                projection,
                expected,
                found,
            } => write!(
                f,
                "{segment}: expected {projection} = {expected}, found {found}"
            ),
            SchemaViolation::UnboundProjection {
                site,
                projection,
                var,
            } => write!(
                f,
                "{site}: cannot check {projection} because `{var}` is not bound"
            ),
            SchemaViolation::Verifier(message) => f.write_str(message),
        }
    }
}

impl OpSchema {
    /// Allowed operand count.
    pub fn operand_count(&self) -> CountRange {
        let mut count = CountRange::exactly(0);
        for operand in self.operands {
            let segment = match operand.arity {
                Arity::One => CountRange::exactly(1),
                Arity::Variadic => segment_count(&operand.constraint),
            };
            count.min += segment.min;
            count.max = count.max.zip(segment.max).map(|(a, b)| a + b);
        }
        count
    }

    /// Allowed result count.
    pub fn result_count(&self) -> CountRange {
        match self.results {
            ResultSchema::Fixed(names) => CountRange::exactly(names.len()),
            ResultSchema::Variadic(_) => segment_count(&self.result_constraint),
            ResultSchema::Optional(_) => CountRange {
                min: 0,
                max: Some(1),
            },
        }
    }

    /// Allowed region count.
    pub fn region_count(&self) -> CountRange {
        let required = self
            .regions
            .iter()
            .filter(|region| !region.optional)
            .count();
        CountRange {
            min: required,
            max: Some(self.regions.len()),
        }
    }

    /// Check `op`'s counts and attributes against this schema.
    ///
    /// Returns every violation found. Callers must not assume the typed
    /// accessors of the operation are safe to use unless this returns an
    /// empty list.
    pub fn verify_structure(&self, ctx: &IrContext, op: OpRef) -> Vec<SchemaViolation> {
        let data = ctx.op(op);
        let mut violations = Vec::new();

        let operands = ctx.op_operands(op).len();
        let expected_operands = self.operand_count();
        if !expected_operands.contains(operands) {
            violations.push(SchemaViolation::OperandCount {
                expected: expected_operands,
                actual: operands,
            });
        }

        let results = ctx.op_result_types(op).len();
        let expected_results = self.result_count();
        if !expected_results.contains(results) {
            violations.push(SchemaViolation::ResultCount {
                expected: expected_results,
                actual: results,
            });
        }

        let expected_regions = self.region_count();
        if !expected_regions.contains(data.regions.len()) {
            violations.push(SchemaViolation::RegionCount {
                expected: expected_regions,
                actual: data.regions.len(),
            });
        }

        if data.successors.len() != self.successors.len() {
            violations.push(SchemaViolation::SuccessorCount {
                expected: self.successors.len(),
                actual: data.successors.len(),
            });
        }

        for attr in self.attributes {
            match data.attributes.get(attr.name) {
                None if attr.optional => {}
                None => violations.push(SchemaViolation::MissingAttribute { name: attr.name }),
                Some(value) if attr.kind.accepts(value) => {}
                Some(_) => violations.push(SchemaViolation::AttributeKind {
                    name: attr.name,
                    expected: attr.kind,
                }),
            }
        }

        violations
    }

    /// Run every verifier stage on `op`, stopping after the first stage
    /// that reports a violation.
    pub fn verify(&self, ctx: &IrContext, op: OpRef) -> Vec<SchemaViolation> {
        let violations = self.verify_declarative(ctx, op);
        if !violations.is_empty() {
            return violations;
        }
        match self.verifier.map(|verifier| verifier(ctx, op)) {
            Some(Err(message)) => vec![SchemaViolation::Verifier(message)],
            _ => Vec::new(),
        }
    }

    /// Run the declarative stages on `op`: counts and attributes, then type
    /// constraints. Unlike [`verify`](Self::verify), this skips the
    /// `#[verify]` hook.
    pub fn verify_declarative(&self, ctx: &IrContext, op: OpRef) -> Vec<SchemaViolation> {
        let violations = self.verify_structure(ctx, op);
        if !violations.is_empty() {
            return violations;
        }
        self.verify_types(ctx, op)
    }

    /// Check type constraints, variable bindings, and projections.
    ///
    /// Requires an operation that passed [`verify_structure`](Self::verify_structure).
    pub fn verify_types(&self, ctx: &IrContext, op: OpRef) -> Vec<SchemaViolation> {
        let slots = TypeSlots::collect(self, ctx, op);
        let mut violations = Vec::new();

        // Stage 2: individual constraints.
        for slot in &slots.types {
            let (var, bounds) = match slot.spec {
                TypeSpec::Var(v) => (Some(self.type_vars[v].name), self.type_vars[v].bounds),
                TypeSpec::Anon(bounds) => (None, bounds),
                TypeSpec::Any | TypeSpec::Proj(_) => continue,
            };
            if let Some(bound) = bounds.iter().find(|bound| !(bound.matches)(ctx, slot.ty)) {
                violations.push(SchemaViolation::TypeConstraint {
                    site: slot.site,
                    var,
                    expected: bound.name,
                    found: print_type(ctx, slot.ty),
                });
            }
        }
        if !violations.is_empty() {
            return violations;
        }

        // Stage 3: the first occurrence of a variable binds it.
        let mut bindings: Vec<Option<(TypeRef, Site)>> = vec![None; self.type_vars.len()];
        for slot in &slots.types {
            let TypeSpec::Var(v) = slot.spec else {
                continue;
            };
            match bindings[v] {
                None => bindings[v] = Some((slot.ty, slot.site)),
                Some((ty, _)) if ty == slot.ty => {}
                Some((ty, bound_by)) => violations.push(SchemaViolation::TypeMismatch {
                    site: slot.site,
                    var: self.type_vars[v].name,
                    bound_by,
                    expected: print_type(ctx, ty),
                    found: print_type(ctx, slot.ty),
                }),
            }
        }
        if !violations.is_empty() {
            return violations;
        }

        // Stage 4: projections of bound variables. A projection whose
        // variable has no binding here (e.g. an absent optional attribute)
        // cannot be checked, which is itself a violation.
        let project = |p: ProjectionRef| {
            let (ty, _) = bindings[p.var]?;
            (self.type_vars[p.var].bounds[p.bound].project)(ctx, ty, p.index)
        };
        let unbound = |p: ProjectionRef, site: String| SchemaViolation::UnboundProjection {
            site,
            projection: self.projection_name(p),
            var: self.type_vars[p.var].name,
        };
        for slot in &slots.types {
            let TypeSpec::Proj(p) = slot.spec else {
                continue;
            };
            if bindings[p.var].is_none() {
                violations.push(unbound(p, slot.site.to_string()));
            } else if let Some(Projected::One(expected)) = project(p)
                && expected != slot.ty
            {
                violations.push(SchemaViolation::ProjectionMismatch {
                    site: slot.site,
                    projection: self.projection_name(p),
                    expected: print_type(ctx, expected),
                    found: print_type(ctx, slot.ty),
                });
            }
        }
        for list in &slots.lists {
            if bindings[list.projection.var].is_none() {
                violations.push(unbound(list.projection, list.segment.to_string()));
            } else if let Some(Projected::List(expected)) = project(list.projection)
                && expected != list.types.as_slice()
            {
                violations.push(SchemaViolation::ListMismatch {
                    segment: list.segment,
                    projection: self.projection_name(list.projection),
                    expected: print_type_list(ctx, expected),
                    found: print_type_list(ctx, &list.types),
                });
            }
        }
        violations
    }

    fn projection_name(&self, p: ProjectionRef) -> String {
        let var = &self.type_vars[p.var];
        format!(
            "{}::{}",
            var.name, var.bounds[p.bound].projections[p.index].name
        )
    }

    /// Infer result types for a generated builder.
    ///
    /// The `#[dialect]` macro calls this only when every result type is a
    /// fixed type, a variable bound by a single operand or a required
    /// attribute, or a projection of such a variable. Panics if a binding
    /// source is missing or does not provide the projection.
    #[doc(hidden)]
    pub fn infer_result_types(
        &self,
        ctx: &mut IrContext,
        operands: &[ValueRef],
        attrs: &[(&str, Option<&Attribute>)],
    ) -> Vec<TypeRef> {
        let binding = |ctx: &IrContext, var: usize| -> TypeRef {
            let from_attr = self
                .attributes
                .iter()
                .filter(|attr| attr.binds == Some(var) && !attr.optional)
                .find_map(
                    |attr| match attrs.iter().find(|(name, _)| *name == attr.name) {
                        Some((_, Some(Attribute::Type(ty)))) => Some(*ty),
                        _ => None,
                    },
                );
            let from_operand = || {
                self.operands
                    .iter()
                    .zip(operands)
                    .take_while(|(operand, _)| operand.arity == Arity::One)
                    .find(|(operand, _)| {
                        matches!(operand.constraint, ValueConstraint::Each(TypeSpec::Var(v)) if v == var)
                    })
                    .map(|(_, value)| ctx.value_ty(*value))
            };
            from_attr.or_else(from_operand).unwrap_or_else(|| {
                panic!(
                    "{}.{}: cannot infer type variable `{}`",
                    self.dialect, self.name, self.type_vars[var].name
                )
            })
        };
        // Projections are copied out so that fixed types can borrow `ctx`
        // mutably afterwards. Kinds were checked at compile time.
        let project = |ctx: &IrContext, p: ProjectionRef| -> Vec<TypeRef> {
            let ty = binding(ctx, p.var);
            match (self.type_vars[p.var].bounds[p.bound].project)(ctx, ty, p.index) {
                Some(Projected::One(ty)) => vec![ty],
                Some(Projected::List(types)) => types.to_vec(),
                None => panic!(
                    "{}.{}: {} = {} does not provide {}",
                    self.dialect,
                    self.name,
                    self.type_vars[p.var].name,
                    print_type(ctx, ty),
                    self.projection_name(p),
                ),
            }
        };
        let infer = |ctx: &mut IrContext, spec: &TypeSpec| match *spec {
            TypeSpec::Var(v) => binding(ctx, v),
            TypeSpec::Proj(p) => project(ctx, p)[0],
            TypeSpec::Anon(bounds) => {
                let fixed = bounds.iter().find_map(|bound| bound.fixed);
                fixed.expect("result bounds are checked at compile time")(ctx)
            }
            TypeSpec::Any => unreachable!("unconstrained results are never inferred"),
        };
        match (&self.results, &self.result_constraint) {
            (ResultSchema::Fixed([]), _) => Vec::new(),
            (ResultSchema::Fixed([_]), ValueConstraint::Each(spec)) => vec![infer(ctx, spec)],
            (ResultSchema::Variadic(_), ValueConstraint::List(ListSpec::Types(specs))) => {
                specs.iter().map(|spec| infer(ctx, spec)).collect()
            }
            (ResultSchema::Variadic(_), ValueConstraint::List(ListSpec::Proj(p))) => {
                project(ctx, *p)
            }
            _ => unreachable!("{}.{}: results are not inferable", self.dialect, self.name),
        }
    }

    /// Look up the registered schema for `dialect.name`.
    pub fn lookup(dialect: Symbol, name: Symbol) -> Option<&'static OpSchema> {
        REGISTRY.get(&(dialect, name)).copied()
    }

    /// Look up the registered schema for an operation.
    pub fn of(ctx: &IrContext, op: OpRef) -> Option<&'static OpSchema> {
        let data = ctx.op(op);
        Self::lookup(data.dialect, data.name)
    }
}

/// Allowed length of a variadic segment; explicit type lists are exact.
fn segment_count(constraint: &ValueConstraint) -> CountRange {
    match constraint {
        ValueConstraint::List(ListSpec::Types(types)) => CountRange::exactly(types.len()),
        _ => CountRange { min: 0, max: None },
    }
}

fn print_type_list(ctx: &IrContext, types: &[TypeRef]) -> String {
    use std::fmt::Write;
    let mut out = String::from("(");
    for (i, ty) in types.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write!(out, "{}", print_type(ctx, *ty)).expect("fmt::Write to String never fails");
    }
    out.push(')');
    out
}

/// One type checked against a single-type constraint.
struct TypeSlot {
    site: Site,
    ty: TypeRef,
    spec: TypeSpec,
}

/// A segment checked against a list projection.
struct ListSlot {
    segment: Segment,
    types: Vec<TypeRef>,
    projection: ProjectionRef,
}

/// The constrained types of an operation: bound attributes first, then
/// operands and results in order. Variable binding follows this order.
#[derive(Default)]
struct TypeSlots {
    types: Vec<TypeSlot>,
    lists: Vec<ListSlot>,
}

impl TypeSlots {
    fn collect(schema: &OpSchema, ctx: &IrContext, op: OpRef) -> Self {
        let mut slots = TypeSlots::default();
        let data = ctx.op(op);
        for attr in schema.attributes {
            if let (Some(var), Some(Attribute::Type(ty))) =
                (attr.binds, data.attributes.get(attr.name))
            {
                slots.types.push(TypeSlot {
                    site: Site::Attribute { name: attr.name },
                    ty: *ty,
                    spec: TypeSpec::Var(var),
                });
            }
        }

        let operand_types: Vec<TypeRef> = ctx
            .op_operands(op)
            .iter()
            .map(|value| ctx.value_ty(*value))
            .collect();
        let mut start = 0;
        for operand in schema.operands {
            let len = match operand.arity {
                Arity::One => 1,
                Arity::Variadic => operand_types.len() - start,
            };
            slots.push_segment(
                start,
                &operand_types[start..start + len],
                operand.constraint,
                |index| Site::Operand {
                    index,
                    name: operand.name,
                },
                Segment::Operands(operand.name),
            );
            start += len;
        }

        let result_types = ctx.op_result_types(op);
        let result_name = match schema.results {
            ResultSchema::Fixed(names) => names.first().copied().unwrap_or("result"),
            ResultSchema::Variadic(name) | ResultSchema::Optional(name) => name,
        };
        slots.push_segment(
            0,
            result_types,
            schema.result_constraint,
            |index| Site::Result {
                index,
                name: match schema.results {
                    ResultSchema::Fixed(names) => names[index],
                    _ => result_name,
                },
            },
            Segment::Results(result_name),
        );
        slots
    }

    fn push_segment(
        &mut self,
        start: usize,
        types: &[TypeRef],
        constraint: ValueConstraint,
        site: impl Fn(usize) -> Site,
        segment: Segment,
    ) {
        match constraint {
            ValueConstraint::Each(spec) => {
                for (i, ty) in types.iter().enumerate() {
                    self.types.push(TypeSlot {
                        site: site(start + i),
                        ty: *ty,
                        spec,
                    });
                }
            }
            ValueConstraint::List(ListSpec::Types(specs)) => {
                // Counts were checked, so the lengths agree.
                for (i, (ty, spec)) in types.iter().zip(specs).enumerate() {
                    self.types.push(TypeSlot {
                        site: site(start + i),
                        ty: *ty,
                        spec: *spec,
                    });
                }
            }
            ValueConstraint::List(ListSpec::Proj(projection)) => self.lists.push(ListSlot {
                segment,
                types: types.to_vec(),
                projection,
            }),
        }
    }
}

/// Support for `#[verify]` code generated by the dialect macro.
///
/// Generated verifiers call `Op::verify(..)` with [`VerifyFallback`] in scope.
/// An inherent `verify` method takes precedence; without one the fallback is
/// selected, and its unsatisfiable bound reports a missing method.
#[doc(hidden)]
pub mod __private {
    use crate::IrContext;

    #[diagnostic::on_unimplemented(
        message = "`{Self}` is declared with `#[verify]` but has no inherent `verify` method",
        label = "`#[verify]` requires this method",
        note = "define `impl {Self} {{ fn verify(self, ctx: &IrContext) -> Result<(), String> {{ .. }} }}`"
    )]
    pub trait VerifyMethodMissing {}

    pub trait VerifyFallback: Sized {
        fn verify(self, _ctx: &IrContext) -> Result<(), String>
        where
            Self: VerifyMethodMissing,
        {
            unreachable!("`VerifyMethodMissing` has no implementations")
        }
    }

    impl<T> VerifyFallback for T {}
}

/// Inventory entry registering an [`OpSchema`].
///
/// Emitted by the `#[dialect]` macro for every declared operation.
pub struct OpSchemaRegistration(pub &'static OpSchema);

inventory::collect!(OpSchemaRegistration);

static REGISTRY: LazyLock<HashMap<(Symbol, Symbol), &'static OpSchema>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    for OpSchemaRegistration(schema) in inventory::iter::<OpSchemaRegistration> {
        let key = (
            Symbol::from_dynamic(schema.dialect),
            Symbol::from_dynamic(schema.name),
        );
        if let Some(previous) = registry.insert(key, *schema) {
            panic!(
                "operation schema {}.{} is registered twice ({previous:p} and {schema:p})",
                schema.dialect, schema.name,
            );
        }
    }
    registry
});

#[cfg(test)]
mod typed_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, func, scf};
    use crate::ops::DialectOp;

    #[test]
    fn macro_generated_schemas_describe_declared_entities() {
        let schema = arith::Cmpi::SCHEMA;
        assert_eq!((schema.dialect, schema.name), ("arith", "cmpi"));
        assert_eq!(
            schema.operands.iter().map(|o| o.name).collect::<Vec<_>>(),
            ["lhs", "rhs"],
        );
        assert!(matches!(schema.results, ResultSchema::Fixed(["result"])));
        assert_eq!(schema.attributes.len(), 1);
        assert_eq!(schema.attributes[0].name, "predicate");
        assert_eq!(schema.attributes[0].kind, AttributeKind::Symbol);
        assert!(!schema.attributes[0].optional);

        let call = func::CallIndirect::SCHEMA;
        assert_eq!(call.operands[1].arity, Arity::Variadic);
        assert!(matches!(call.results, ResultSchema::Variadic("results")));
        assert_eq!(call.attributes[0].name, "signature");
        assert!(!call.attributes[0].optional);

        let r#if = scf::If::SCHEMA;
        assert_eq!(r#if.regions.len(), 2);
    }

    #[test]
    fn registry_finds_schemas_across_dialects() {
        let schema = OpSchema::lookup(Symbol::new("func"), Symbol::new("return"))
            .expect("func.return should be registered");
        assert_eq!((schema.dialect, schema.name), ("func", "return"));
        assert_eq!(schema.operands[0].arity, Arity::Variadic);
        assert!(OpSchema::lookup(Symbol::new("func"), Symbol::new("no_such_op")).is_none());
    }

    fn schema_errors(input: &str) -> String {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        crate::validation::validate_operation_verifiers(&ctx, module).to_string()
    }

    #[test]
    fn optional_entities_accept_absent_and_present_forms() {
        let text = schema_errors(
            r#"core.module @m {
  func.func @decl(%x: core.i32)
  func.func @def(%c: core.i1) {
    scf.if %c {
      scf.yield
    } {
      scf.yield
    }
    func.return
  }
}"#,
        );
        assert_eq!(text, "validation passed");
        assert_eq!(scf::If::SCHEMA.result_count().to_string(), "0 to 1");
        assert_eq!(func::Func::SCHEMA.region_count().to_string(), "0 to 1");
    }

    #[test]
    fn structural_violations_report_the_operation_and_entity() {
        let text = schema_errors(
            r#"core.module @m {
  func.func @f(%a: core.f64, %b: core.f64) {
    %c = arith.cmpf %a, %b {predicate = "olt"} : core.i1
    %d = arith.cmpf %a : core.i1
    func.return
  }
}"#,
        );
        assert!(
            text.contains("arith.cmpf (op0): attribute `predicate` must be a Symbol attribute"),
            "{text}"
        );
        assert!(text.contains("expected 2 operand(s), found 1"), "{text}");
        assert!(
            text.contains("missing required attribute `predicate`"),
            "{text}"
        );
        // Later operation-local checks are skipped for schema-invalid ops.
        assert!(!text.contains("unsupported predicate"), "{text}");
    }

    #[test]
    fn attribute_kinds_check_value_domains() {
        assert!(AttributeKind::U32.accepts(&Attribute::Int(7)));
        assert!(!AttributeKind::U32.accepts(&Attribute::Int(-1)));
        assert!(!AttributeKind::Symbol.accepts(&Attribute::String("x".into())));
        assert!(AttributeKind::Any.accepts(&Attribute::Unit));
    }
}
