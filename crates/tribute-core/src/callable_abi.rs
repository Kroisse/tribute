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

    /// Logical CPS parameter order before inserting a closure environment.
    pub fn lowered_params(&self, evidence: T, continuation_frame: T) -> Vec<T> {
        let mut params = Vec::with_capacity(
            self.source_params.len()
                + usize::from(self.convention.needs_evidence())
                + usize::from(self.convention.needs_continuation_frame()),
        );
        if self.convention.needs_evidence() {
            params.push(evidence);
        }
        if self.convention.needs_continuation_frame() {
            params.push(continuation_frame);
        }
        params.extend_from_slice(&self.source_params);
        params
    }

    pub fn source_param_offset(&self) -> usize {
        usize::from(self.convention.needs_evidence())
            + usize::from(self.convention.needs_continuation_frame())
    }

    /// Interpose the physical closure environment in convention order.
    ///
    /// Direct: `env, source...`
    /// EvidenceDirect: `evidence, env, source...`
    /// Cps: `evidence, env, continuation_frame, source...`
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
        assert_eq!(direct.lowered_params("ev", "frame"), ["arg"]);

        let evidence_direct = abi(CallingConvention::EvidenceDirect);
        assert_eq!(evidence_direct.lowered_params("ev", "frame"), ["ev", "arg"]);

        let cps = abi(CallingConvention::Cps);
        assert_eq!(cps.lowered_params("ev", "frame"), ["ev", "frame", "arg"]);
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
            cps.interpose_environment(&["ev", "frame", "arg"], "env"),
            ["ev", "env", "frame", "arg"]
        );
    }
}
