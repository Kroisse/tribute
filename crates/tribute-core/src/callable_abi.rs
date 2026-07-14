//! Shared logical and lowered callable ABI layout.

use crate::CallingConvention;

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

    /// Interpose the physical closure environment in convention order.
    ///
    /// Direct: `env, source...`
    /// EvidenceDirect: `evidence, env, source...`
    /// Cps: `evidence, env, done_k, source...`
    pub fn interpose_environment(&self, logical_params: &[T], environment: T) -> Vec<T> {
        debug_assert_eq!(
            logical_params.len(),
            self.lowered_params(environment, environment).len(),
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
}
