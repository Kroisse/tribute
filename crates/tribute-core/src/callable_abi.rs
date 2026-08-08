//! Shared logical and lowered callable ABI layout.

use crate::CallingConvention;

/// Locate the closure environment in an already-lowered physical parameter
/// list. Evidence is recognized only by exact leading-type equality.
pub fn physical_environment_index<T: PartialEq>(params: &[T], evidence: &T) -> usize {
    usize::from(params.first() == Some(evidence))
}

/// Interpose a closure environment into an already-lowered physical ABI.
///
/// Source CPS lambdas begin with exact evidence, while generated
/// continuations and `done_k` targets do not. Convention metadata alone
/// describes semantic role and must not be used to invent hidden parameters.
pub fn interpose_physical_environment<T: Copy + PartialEq>(
    params: &[T],
    evidence: &T,
    environment: T,
) -> Vec<T> {
    let index = physical_environment_index(params, evidence);
    let mut physical = Vec::with_capacity(params.len() + 1);
    physical.extend_from_slice(&params[..index]);
    physical.push(environment);
    physical.extend_from_slice(&params[index..]);
    physical
}

/// A source callable paired with its selected calling convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallableAbi<T> {
    pub convention: CallingConvention,
    pub source_params: Vec<T>,
    pub source_result: T,
}

impl<T: Copy> CallableAbi<T> {
    pub fn new(
        convention: CallingConvention,
        source_params: impl IntoIterator<Item = T>,
        source_result: T,
    ) -> Self {
        Self {
            convention,
            source_params: source_params.into_iter().collect(),
            source_result,
        }
    }

    /// Parameter types for the current compatibility representation.
    pub fn lowered_params(&self, evidence: T, control_carrier: T) -> Vec<T> {
        let mut params = Vec::with_capacity(
            self.source_params.len()
                + usize::from(self.convention.needs_evidence())
                + usize::from(self.convention.needs_done_k()),
        );
        if self.convention.needs_evidence() {
            params.push(evidence);
        }
        if self.convention.needs_done_k() {
            params.push(control_carrier);
        }
        params.extend_from_slice(&self.source_params);
        params
    }

    /// Parameter types for the result-indexed CPS ABI.
    ///
    /// This is the production logical layout. The older two-hidden-operand
    /// helper remains solely for the explicitly isolated legacy frontend
    /// compatibility path until it is removed.
    pub fn lowered_params_with_dispatch(&self, evidence: T, done: T, dispatch: T) -> Vec<T> {
        let mut params = Vec::with_capacity(
            self.source_params.len()
                + usize::from(self.convention.needs_evidence())
                + usize::from(self.convention.needs_done_k()) * 2,
        );
        if self.convention.needs_evidence() {
            params.push(evidence);
        }
        if self.convention.needs_done_k() {
            params.push(done);
            params.push(dispatch);
        }
        params.extend_from_slice(&self.source_params);
        params
    }

    /// Result type for the current compatibility representation.
    ///
    /// Logical CPS does not directly return a source result. Until true
    /// tail-call or trampoline lowering is selected, the IR uses the supplied
    /// control carrier for the continuation chain.
    pub fn lowered_result(&self, control_carrier: T) -> T {
        if self.convention.needs_done_k() {
            control_carrier
        } else {
            self.source_result
        }
    }

    pub fn source_param_offset(&self) -> usize {
        usize::from(self.convention.needs_evidence()) + usize::from(self.convention.needs_done_k())
    }

    /// Source-parameter offset for [`Self::lowered_params_with_dispatch`].
    pub fn source_param_offset_with_dispatch(&self) -> usize {
        usize::from(self.convention.needs_evidence())
            + usize::from(self.convention.needs_done_k()) * 2
    }

    /// Interpose the physical closure environment in convention order.
    ///
    /// Direct: `env, source...`
    /// EvidenceDirect: `evidence, env, source...`
    /// Cps: `evidence, env, done_k, source...`
    pub fn interpose_environment(&self, logical_params: &[T], environment: T) -> Vec<T> {
        let legacy_len = self.lowered_params(environment, environment).len();
        let dispatch_len = legacy_len + usize::from(self.convention.needs_done_k());
        debug_assert!(
            logical_params.len() == legacy_len || logical_params.len() == dispatch_len,
            "logical parameter count must match the selected convention",
        );
        let env_index = usize::from(self.convention.needs_evidence());
        let mut physical = Vec::with_capacity(logical_params.len() + 1);
        physical.extend_from_slice(&logical_params[..env_index]);
        physical.push(environment);
        physical.extend_from_slice(&logical_params[env_index..]);
        physical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn abi(convention: CallingConvention) -> CallableAbi<&'static str> {
        CallableAbi::new(convention, ["arg"], "result")
    }

    #[test]
    fn lowered_function_layouts_are_centralized() {
        let direct = abi(CallingConvention::Direct);
        assert_eq!(direct.lowered_params("ev", "control"), ["arg"]);
        assert_eq!(direct.lowered_result("control"), "result");

        let evidence_direct = abi(CallingConvention::EvidenceDirect);
        assert_eq!(
            evidence_direct.lowered_params("ev", "control"),
            ["ev", "arg"]
        );
        assert_eq!(evidence_direct.lowered_result("control"), "result");

        let cps = abi(CallingConvention::Cps);
        assert_eq!(
            cps.lowered_params("ev", "control"),
            ["ev", "control", "arg"]
        );
        assert_eq!(cps.lowered_result("control"), "control");
    }

    #[test]
    fn physical_closure_layout_only_interposes_environment() {
        let direct = abi(CallingConvention::Direct);
        assert_eq!(
            direct.interpose_environment(&["arg"], "env"),
            ["env", "arg"]
        );

        let evidence_direct = abi(CallingConvention::EvidenceDirect);
        assert_eq!(
            evidence_direct.interpose_environment(&["ev", "arg"], "env"),
            ["ev", "env", "arg"]
        );

        let cps = abi(CallingConvention::Cps);
        assert_eq!(
            cps.interpose_environment(&["ev", "control", "arg"], "env"),
            ["ev", "env", "control", "arg"]
        );
    }

    #[test]
    fn cps_dispatch_is_a_second_hidden_operand() {
        let cps = abi(CallingConvention::Cps);
        assert_eq!(
            cps.lowered_params_with_dispatch("ev", "done", "dispatch"),
            ["ev", "done", "dispatch", "arg"]
        );
        assert_eq!(cps.source_param_offset_with_dispatch(), 3);
    }

    #[test]
    fn physical_environment_uses_only_an_exact_leading_evidence_type() {
        assert_eq!(
            interpose_physical_environment(&["evidence", "done", "arg"], &"evidence", "env"),
            ["evidence", "env", "done", "arg"]
        );
        assert_eq!(
            interpose_physical_environment(&["answer"], &"evidence", "env"),
            ["env", "answer"]
        );
        assert_eq!(
            interpose_physical_environment(&[], &"evidence", "env"),
            ["env"]
        );
    }
}
