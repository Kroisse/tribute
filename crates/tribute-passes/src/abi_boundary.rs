//! Representation/ABI boundary exit verification.
//!
//! The boundary exit contract (`new-plans/ir.md`, "Representation/ABI 경계")
//! forbids upper-level control operations, types, and semantic metadata past
//! the point where target dialect lowering begins. This module reports every
//! violation of that contract; the pipeline observes the report against a list
//! of violations that later boundary work is known to remove.

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::op_interface::IndirectCallLikeOps;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_op};

/// Dialects whose operations must be consumed inside the boundary.
const FORBIDDEN_DIALECTS: &[&str] = &["tribute_control", "ability", "effect", "closure"];

/// Semantic control metadata that must not cross the boundary.
const FORBIDDEN_ATTRIBUTES: &[&str] = &[
    "tribute.calling_convention",
    "tribute.root_export_convention",
    "tribute.root_source_result",
    "tribute.cps_continuation_frame_result",
    "tribute.closure_environment_index",
];

/// Prefix of language-specific attributes, which must be classified.
const LANGUAGE_ATTRIBUTE_PREFIX: &str = "tribute.";

/// Language-specific metadata the boundary preserves: source/debug positions
/// and type identity metadata. A new key must be classified explicitly.
const PRESERVED_ATTRIBUTES: &[&str] = &[
    "tribute.definition.source",
    "tribute.definition.start",
    "tribute.definition.end",
    "tribute.type.string",
];

/// The target whose boundary exit is verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    Native,
    Wasm,
}

/// One class of boundary exit violation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::Display)]
pub enum ViolationKind {
    /// An operation of a dialect that the boundary must consume.
    #[display("forbidden op {dialect}.{name}")]
    ForbiddenOp { dialect: String, name: String },
    /// A `closure.closure` type.
    #[display("forbidden type {dialect}.{name}")]
    ForbiddenType { dialect: String, name: String },
    /// Semantic control metadata listed by the exit contract.
    #[display("forbidden attribute {_0}")]
    ForbiddenAttribute(String),
    /// A language-specific attribute that is neither forbidden nor preserved.
    #[display("unclassified attribute {_0}")]
    UnclassifiedAttribute(String),
    /// A callable signature whose result is `core.never`.
    #[display("callable result core.never")]
    NeverCallableResult,
    /// An unrealized cast whose source already has the declared type.
    #[display("identity unrealized cast")]
    IdentityCast,
    /// A `func.constant` whose type is not its target's exact signature.
    #[display("function reference differs from its target signature")]
    ReferenceSignatureMismatch,
    /// A function definition or indirect call without an exact signature.
    #[display("missing exact callable signature")]
    MissingExactSignature,
}

/// A single boundary exit violation.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display)]
#[display("{kind}: {detail}")]
pub struct BoundaryViolation {
    pub kind: ViolationKind,
    /// The operation where the violation was found, if it is not an alias.
    pub op: Option<OpRef>,
    pub detail: String,
}

/// Report every boundary exit violation in `module`, in IR order.
///
/// Operations, aliases, result types, block arguments and their attributes,
/// and nested type parameters and attributes are all inspected.
pub fn verify_boundary_exit(ctx: &IrContext, module: Module) -> Vec<BoundaryViolation> {
    let mut verifier = Verifier::new(ctx);
    for &(name, ty) in ctx.type_aliases() {
        verifier.check_type(ty, None, &format!("alias !{name}"));
    }
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    let functions = SymbolTable::collect(ctx, module);
    for op in ops {
        verifier.check_op(op, &functions);
    }
    verifier.violations
}

/// A violation class that later boundary work is known to remove.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display)]
pub enum PendingViolation {
    /// An operation `dialect.name`.
    #[display("forbidden op {_0}.{_1}")]
    Op(&'static str, &'static str),
    /// A forbidden attribute.
    #[display("forbidden attribute {_0}")]
    Attribute(&'static str),
    /// An attribute that has not been classified yet.
    #[display("unclassified attribute {_0}")]
    Unclassified(&'static str),
}

impl PendingViolation {
    /// Whether this pending entry covers `kind`.
    pub fn covers(self, kind: &ViolationKind) -> bool {
        match (self, kind) {
            (
                Self::Op(expected_dialect, expected_name),
                ViolationKind::ForbiddenOp { dialect, name },
            ) => dialect == expected_dialect && name == expected_name,
            (Self::Attribute(expected), ViolationKind::ForbiddenAttribute(name)) => {
                name == expected
            }
            (Self::Unclassified(expected), ViolationKind::UnclassifiedAttribute(name)) => {
                name == expected
            }
            _ => false,
        }
    }
}

/// Violations still present at the exit of `target`'s boundary.
pub fn pending_boundary_violations(target: TargetKind) -> &'static [PendingViolation] {
    const COMMON: [PendingViolation; 5] = [
        // Read past the exit by native entry generation, native ownership
        // planning, and Wasm `_start` generation.
        PendingViolation::Attribute("tribute.calling_convention"),
        // Recorded on physical definitions by target ABI physicalization.
        PendingViolation::Attribute("tribute.closure_environment_index"),
        // Kept on continuation frame layouts.
        PendingViolation::Attribute("tribute.cps_continuation_frame_result"),
        // Marks the root bridge's worker call.
        PendingViolation::Unclassified("tribute.root_cps_call"),
        // Left on bytes intrinsic declarations, which target lowering past
        // the exit still recognizes by name.
        PendingViolation::Unclassified("tribute.compiler_intrinsic"),
    ];
    const NATIVE: &[PendingViolation] = &COMMON;
    const WASM: &[PendingViolation] = &[
        COMMON[0],
        COMMON[1],
        COMMON[2],
        COMMON[3],
        COMMON[4],
        // Wasm evidence lowering still runs inside Wasm dialect lowering.
        PendingViolation::Op("effect", "extend"),
        PendingViolation::Op("effect", "dispatch_tail"),
        PendingViolation::Op("effect", "dispatch_cps"),
    ];
    match target {
        TargetKind::Native => NATIVE,
        TargetKind::Wasm => WASM,
    }
}

/// Violations of `target`'s boundary exit that no pending entry covers.
pub fn unexpected_boundary_violations(
    ctx: &IrContext,
    module: Module,
    target: TargetKind,
) -> Vec<BoundaryViolation> {
    let pending = pending_boundary_violations(target);
    verify_boundary_exit(ctx, module)
        .into_iter()
        .filter(|violation| !pending.iter().any(|entry| entry.covers(&violation.kind)))
        .collect()
}

/// A violation found inside a type, relative to the site that uses the type.
type TypeViolation = (ViolationKind, String);

struct Verifier<'a> {
    ctx: &'a IrContext,
    /// Violations inside each type, computed once and replayed at every use.
    type_violations: HashMap<TypeRef, Rc<[TypeViolation]>>,
    /// Types being computed, to stop recursion through cyclic references.
    computing: HashSet<TypeRef>,
    violations: Vec<BoundaryViolation>,
}

impl<'a> Verifier<'a> {
    fn new(ctx: &'a IrContext) -> Self {
        Self {
            ctx,
            type_violations: HashMap::new(),
            computing: HashSet::new(),
            violations: Vec::new(),
        }
    }

    fn report(&mut self, kind: ViolationKind, op: Option<OpRef>, detail: String) {
        self.violations.push(BoundaryViolation { kind, op, detail });
    }

    fn check_op(&mut self, op: OpRef, functions: &SymbolTable) {
        let ctx = self.ctx;
        let data = ctx.op(op);
        let op_name = format!("{}.{}", data.dialect, data.name);
        let dialect = data.dialect.to_string();
        if FORBIDDEN_DIALECTS.contains(&dialect.as_str()) {
            self.report(
                ViolationKind::ForbiddenOp {
                    dialect,
                    name: data.name.to_string(),
                },
                Some(op),
                op_name.clone(),
            );
        }
        for (name, value) in data.attributes.iter() {
            let context = format!("{op_name} attribute {name}");
            self.check_attribute_name(*name, Some(op), &context);
            self.check_attribute(value, Some(op), &context);
        }
        for &ty in ctx.op_result_types(op) {
            self.check_type(ty, Some(op), &format!("{op_name} result"));
        }
        for &region in &data.regions {
            for &block in &ctx.region(region).blocks {
                for argument in &ctx.block(block).args {
                    let context = format!("{op_name} block argument");
                    self.check_type(argument.ty, Some(op), &context);
                    for (name, value) in argument.attrs.iter() {
                        let context = format!("{context} attribute {name}");
                        self.check_attribute_name(*name, Some(op), &context);
                        self.check_attribute(value, Some(op), &context);
                    }
                }
            }
        }
        if let Ok(cast) = core::UnrealizedConversionCast::from_op(ctx, op)
            && ctx.value_ty(cast.value(ctx)) == ctx.value_ty(cast.result(ctx))
        {
            self.report(ViolationKind::IdentityCast, Some(op), op_name.clone());
        }
        if let Ok(constant) = func::Constant::from_op(ctx, op) {
            let target = constant.func_ref(ctx);
            let expected = functions
                .resolve(target)
                .and_then(|function| func::Func::from_op(ctx, function).ok())
                .map(|function| function.r#type(ctx));
            if expected.is_none() || ctx.op_result_types(op) != [expected.unwrap()] {
                self.report(
                    ViolationKind::ReferenceSignatureMismatch,
                    Some(op),
                    format!("{op_name} @{target}"),
                );
            }
        }
        let has_exact_signature = if let Ok(function) = func::Func::from_op(ctx, op) {
            Some(func::FuncSig::from_type_ref(ctx, function.r#type(ctx)).is_some())
        } else if func::CallIndirect::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op) {
            Some(IndirectCallLikeOps::exact_signature(ctx, op).is_some())
        } else {
            None
        };
        if has_exact_signature == Some(false) {
            self.report(
                ViolationKind::MissingExactSignature,
                Some(op),
                op_name.clone(),
            );
        }
    }

    fn check_attribute_name(&mut self, name: Symbol, op: Option<OpRef>, context: &str) {
        if let Some(kind) = classify_attribute_name(name) {
            self.report(kind, op, context.to_owned());
        }
    }

    fn check_attribute(&mut self, attribute: &Attribute, op: Option<OpRef>, context: &str) {
        let mut found = Vec::new();
        self.attribute_violations(attribute, "", &mut found);
        for (kind, location) in found {
            self.report(kind, op, format!("{context}{location}"));
        }
    }

    /// Violations inside an attribute value: attribute names nested in
    /// dictionaries and every nested type, with locations relative to `location`.
    fn attribute_violations(
        &mut self,
        attribute: &Attribute,
        location: &str,
        found: &mut Vec<TypeViolation>,
    ) {
        match attribute {
            Attribute::Type(ty) => {
                for (kind, inner) in self.type_violations(*ty).iter() {
                    found.push((kind.clone(), format!("{location}{inner}")));
                }
            }
            Attribute::List(values) => {
                for value in values {
                    self.attribute_violations(value, location, found);
                }
            }
            Attribute::Dict(entries) => {
                for (name, value) in entries.iter() {
                    let location = format!("{location} key {name}");
                    if let Some(kind) = classify_attribute_name(*name) {
                        found.push((kind, location.clone()));
                    }
                    self.attribute_violations(value, &location, found);
                }
            }
            _ => {}
        }
    }

    /// Report every violation inside `ty` at this use site.
    fn check_type(&mut self, ty: TypeRef, op: Option<OpRef>, context: &str) {
        for (kind, location) in self.type_violations(ty).iter() {
            self.report(kind.clone(), op, format!("{context}{location}"));
        }
    }

    /// Violations inside `ty`, with locations relative to the type.
    fn type_violations(&mut self, ty: TypeRef) -> Rc<[TypeViolation]> {
        if let Some(cached) = self.type_violations.get(&ty) {
            return cached.clone();
        }
        if !self.computing.insert(ty) {
            return Rc::from([]);
        }
        let ctx = self.ctx;
        let data = ctx.get_type(ty);
        let mut found = Vec::new();
        if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
            found.push((
                ViolationKind::ForbiddenType {
                    dialect: data.dialect.to_string(),
                    name: data.name.to_string(),
                },
                String::new(),
            ));
        }
        if let Some(signature) = func::FuncSig::from_type_ref(ctx, ty)
            && signature
                .results(ctx)
                .iter()
                .any(|&result| core::Never::matches(ctx, result))
        {
            found.push((ViolationKind::NeverCallableResult, String::new()));
        }
        for (name, value) in data.attrs.iter() {
            let location = format!(" type attribute {name}");
            if let Some(kind) = classify_attribute_name(*name) {
                found.push((kind, location.clone()));
            }
            self.attribute_violations(value, &location, &mut found);
        }
        for &parameter in data.params.iter() {
            found.extend(self.type_violations(parameter).iter().cloned());
        }
        self.computing.remove(&ty);
        let found: Rc<[TypeViolation]> = found.into();
        self.type_violations.insert(ty, found.clone());
        found
    }
}

/// The violation a language-specific attribute name represents, if any.
fn classify_attribute_name(name: Symbol) -> Option<ViolationKind> {
    let name = name.to_string();
    if FORBIDDEN_ATTRIBUTES.contains(&name.as_str()) {
        Some(ViolationKind::ForbiddenAttribute(name))
    } else if name.starts_with(LANGUAGE_ATTRIBUTE_PREFIX)
        && !PRESERVED_ATTRIBUTES.contains(&name.as_str())
    {
        Some(ViolationKind::UnclassifiedAttribute(name))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    fn kinds(input: &str) -> Vec<ViolationKind> {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        verify_boundary_exit(&ctx, module)
            .into_iter()
            .map(|violation| violation.kind)
            .collect()
    }

    fn attribute(name: &str) -> ViolationKind {
        ViolationKind::ForbiddenAttribute(name.to_owned())
    }

    #[test]
    fn physical_module_has_no_violations() {
        let violations = kinds(
            r#"core.module @test {
  func.func @target(%value: core.i32) attributes {type = func.func_sig<(core.i32) -> ()> {call_conv = @tail}, tribute.definition.source = @here} {
    func.return
  }
  func.func @caller(%value: core.i32) attributes {type = func.func_sig<(core.i32) -> ()> {call_conv = @tail}} {
    %reference = func.constant {func_ref = @target} : func.func_sig<(core.i32) -> ()> {call_conv = @tail}
    func.tail_call_indirect %reference, %value {signature = func.func_sig<(core.i32) -> ()> {call_conv = @tail}}
  }
}"#,
        );
        assert_eq!(violations, []);
    }

    #[test]
    fn forbidden_operations_and_closure_types_are_reported() {
        let violations = kinds(
            r#"core.module @test {
  !closure = closure.closure(func.func_sig<(core.i32) -> core.i32>) {}
  func.func @run(%callback: !closure) -> !closure {
    func.return %callback
  }
}"#,
        );
        assert!(violations.contains(&ViolationKind::ForbiddenType {
            dialect: "closure".to_owned(),
            name: "closure".to_owned(),
        }));

        let violations = kinds(
            r#"core.module @test {
  func.func @run(%evidence: core.ptr) {
    %environment = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %created = closure.new %environment {func_ref = @run} : core.ptr
    func.return
  }
}"#,
        );
        assert!(violations.contains(&ViolationKind::ForbiddenOp {
            dialect: "closure".to_owned(),
            name: "new".to_owned(),
        }));
    }

    #[test]
    fn semantic_metadata_is_found_in_nested_positions() {
        // Operation attribute.
        assert_eq!(
            kinds(
                r#"core.module @test {
  func.func @run() attributes {tribute.calling_convention = 2} {
    func.return
  }
}"#
            ),
            [attribute("tribute.calling_convention")]
        );
        // Type attribute inside an alias.
        assert_eq!(
            kinds(
                r#"core.module @test {
  !frame = adt.typeref() {name = @Frame, tribute.cps_continuation_frame_result = core.nil}
}"#
            ),
            [attribute("tribute.cps_continuation_frame_result")]
        );
        // Type attribute nested in a signature parameter of an attribute list.
        assert_eq!(
            kinds(
                r#"core.module @test {
  func.func @run() attributes {evidence = [func.func_sig<(adt.typeref() {name = @Frame, tribute.closure_environment_index = 0}) -> ()>]} {
    func.return
  }
}"#
            ),
            [attribute("tribute.closure_environment_index")]
        );
        // Dictionary keys and types nested in dictionary values.
        assert_eq!(
            kinds(
                r#"core.module @test {
  func.func @run() attributes {meta = [{tribute.calling_convention = 2}]} {
    func.return
  }
}"#
            ),
            [attribute("tribute.calling_convention")]
        );
        assert_eq!(
            kinds(
                r#"core.module @test {
  !frame = adt.typeref() {meta = {entry = {tribute.root_source_result = core.nil}}, name = @Frame}
}"#
            ),
            [attribute("tribute.root_source_result")]
        );
        assert_eq!(
            kinds(
                r#"core.module @test {
  func.func @run() attributes {meta = {ty = adt.typeref() {name = @Frame, tribute.closure_environment_index = 0}}} {
    func.return
  }
}"#
            ),
            [attribute("tribute.closure_environment_index")]
        );
        // Block argument attribute. Textual IR cannot spell block argument
        // attributes other than names, so attach one directly.
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%value: core.i32) {
    func.return
  }
}"#,
        );
        let function = module.ops(&ctx)[0];
        let entry = ctx.region(ctx.op(function).regions[0]).blocks[0];
        let nil = core::nil(&mut ctx).as_type_ref();
        ctx.block_mut(entry).args[0].attrs.insert(
            Symbol::new("tribute.root_source_result"),
            Attribute::Type(nil),
        );
        assert_eq!(
            verify_boundary_exit(&ctx, module)
                .into_iter()
                .map(|violation| violation.kind)
                .collect::<Vec<_>>(),
            [attribute("tribute.root_source_result")]
        );
    }

    #[test]
    fn language_attributes_must_be_classified() {
        assert_eq!(
            kinds(
                r#"core.module @test {
  func.func @run() attributes {tribute.renamed_convention = 2, tribute.definition.start = 1} {
    func.return
  }
}"#
            ),
            [ViolationKind::UnclassifiedAttribute(
                "tribute.renamed_convention".to_owned()
            )]
        );
    }

    #[test]
    fn never_results_identity_casts_and_reference_mismatches_are_reported() {
        let violations = kinds(
            r#"core.module @test {
  func.func @never() -> core.never {
    func.unreachable
  }
  func.func @target(%env: core.ptr, %value: core.i32) attributes {type = func.func_sig<(core.ptr, core.i32) -> ()> {call_conv = @tail}} {
    func.return
  }
  func.func @caller(%value: core.i32) {
    %same = core.unrealized_conversion_cast %value : core.i32
    %reference = func.constant {func_ref = @target} : func.func_sig<(core.i32) -> ()> {call_conv = @tail}
    func.return
  }
}"#,
        );
        assert!(violations.contains(&ViolationKind::NeverCallableResult));
        assert!(violations.contains(&ViolationKind::IdentityCast));
        assert!(violations.contains(&ViolationKind::ReferenceSignatureMismatch));
    }

    #[test]
    fn references_resolve_by_root_qualified_path() {
        let violations = kinds(
            r#"core.module @test {
  core.module @left {
    func.func @same(%value: core.i32) {
      func.return
    }
    func.func @take() {
      %reference = func.constant {func_ref = @"left::same"} : func.func_sig<(core.i32) -> ()>
      func.return
    }
  }
  core.module @right {
    func.func @same(%value: core.i64) -> core.i64 {
      func.return %value
    }
  }
}"#,
        );
        assert_eq!(violations, []);

        // A bare name does not resolve relative to the referencing module.
        let violations = kinds(
            r#"core.module @test {
  core.module @left {
    func.func @same(%value: core.i32) {
      func.return
    }
    func.func @take() {
      %reference = func.constant {func_ref = @same} : func.func_sig<(core.i32) -> ()>
      func.return
    }
  }
}"#,
        );
        assert_eq!(violations, [ViolationKind::ReferenceSignatureMismatch]);
    }

    #[test]
    fn untyped_references_and_indirect_calls_without_signatures_are_reported() {
        let violations = kinds(
            r#"core.module @test {
  func.func @target(%value: core.i32) {
    func.return
  }
  func.func @caller(%callee: core.ptr, %value: core.i32) {
    %reference = func.constant {func_ref = @target} : core.i32
    func.call_indirect %callee, %value
    func.return
  }
}"#,
        );
        assert!(violations.contains(&ViolationKind::ReferenceSignatureMismatch));
        assert!(violations.contains(&ViolationKind::MissingExactSignature));
    }

    #[test]
    fn unknown_keys_under_preserved_prefixes_must_be_classified() {
        assert_eq!(
            kinds(
                r#"core.module @test {
  func.func @run() attributes {tribute.definition.convention = 2} {
    func.return
  }
}"#
            ),
            [ViolationKind::UnclassifiedAttribute(
                "tribute.definition.convention".to_owned()
            )]
        );
    }

    #[test]
    fn a_shared_type_is_reported_at_every_use() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @first() -> core.never {
    func.unreachable
  }
  func.func @second() -> core.never {
    func.unreachable
  }
}"#,
        );
        let sites: Vec<_> = verify_boundary_exit(&ctx, module)
            .into_iter()
            .filter(|violation| violation.kind == ViolationKind::NeverCallableResult)
            .map(|violation| violation.op)
            .collect();
        assert_eq!(sites.len(), 2, "{sites:?}");
        assert_ne!(sites[0], sites[1]);
    }

    #[test]
    fn pending_entries_cover_only_their_own_kind() {
        let pending = PendingViolation::Op("effect", "extend");
        assert!(pending.covers(&ViolationKind::ForbiddenOp {
            dialect: "effect".to_owned(),
            name: "extend".to_owned(),
        }));
        assert!(!pending.covers(&ViolationKind::ForbiddenOp {
            dialect: "effect".to_owned(),
            name: "fresh_prompt_tag".to_owned(),
        }));
        assert!(
            !PendingViolation::Attribute("tribute.calling_convention").covers(
                &ViolationKind::UnclassifiedAttribute("tribute.calling_convention".to_owned())
            )
        );
    }
}
