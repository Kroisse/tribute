//! Declarative operation schemas generated from `#[dialect]` definitions.
//!
//! Every operation declared through the dialect macro exposes a static
//! [`OpSchema`] describing its operands, results, attributes, regions, and
//! successors. Schemas are registered through `inventory` so that operation
//! verification, assembly formats, and declarative rewrite tooling can look
//! them up by operation name without duplicating the definition.
//!
//! Schema verification is the first stage of the operation-verifier layer
//! described in `new-plans/ir.md`: it checks entity counts, required
//! attributes, and attribute kinds. It runs only at explicit verifier
//! checkpoints; parsers, raw builders, and rewrites may construct operations
//! that violate the schema in the meantime.

use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;

use crate::type_constraint::ConstraintDesc;
use crate::types::Attribute;
use crate::{IrContext, OpRef, Symbol};

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
}

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
    /// Unconstrained (`_`, and every legacy entity).
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
        }
    }
}

impl OpSchema {
    /// Allowed operand count.
    pub fn operand_count(&self) -> CountRange {
        let fixed = self
            .operands
            .iter()
            .filter(|operand| operand.arity == Arity::One)
            .count();
        CountRange {
            min: fixed,
            max: (fixed == self.operands.len()).then_some(fixed),
        }
    }

    /// Allowed result count.
    pub fn result_count(&self) -> CountRange {
        match self.results {
            ResultSchema::Fixed(names) => CountRange::exactly(names.len()),
            ResultSchema::Variadic(_) => CountRange { min: 0, max: None },
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
        assert!(call.attributes[0].optional);

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
