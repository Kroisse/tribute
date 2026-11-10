//! Conversion of source-logical types, callable signatures, and attributes to
//! their physical forms, and the continuation types the conversion builds.

use super::*;

impl tribute_ir::dialect::adt::convert::LayoutBlindConversion for Converter<'_> {
    type Error = std::convert::Infallible;

    fn ctx(&self) -> &IrContext {
        self.ctx
    }

    fn ctx_mut(&mut self) -> &mut IrContext {
        self.ctx
    }

    fn convert(&mut self, ty: TypeRef) -> Result<TypeRef, Self::Error> {
        Ok(self.convert_type(ty))
    }

    fn key_follows_body(&self, layout: TypeRef) -> bool {
        tribute_control::logical_tuple_elements(self.ctx, layout).is_some()
    }

    fn fix(&mut self, from: TypeRef, to: TypeRef) {
        self.converted_types.insert(from, to);
    }

    fn reset(&mut self) {
        self.converted_types.clear();
    }
}

impl Converter<'_> {
    pub(super) fn never_type(&mut self) -> TypeRef {
        core::Never::params().intern(self.ctx).as_type_ref()
    }

    /// The converted evidence type: that of the source module's marker,
    /// whose layout this conversion may give a new key.
    pub(super) fn evidence_type(&mut self) -> TypeRef {
        let source = ability::evidence_type(self.ctx, self.module);
        self.convert_type(source)
    }

    pub(super) fn resumption_type(&mut self, input: TypeRef, answer: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let frame = self.frame_type(answer);
        cps_resume_exact_type(self.ctx, evidence, input, frame)
    }

    pub(super) fn completion_type(&mut self, value: TypeRef, answer: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let frame = self.frame_type(answer);
        cps_completion_type(self.ctx, evidence, value, frame)
    }

    pub(super) fn generated_continuation_type(&mut self, function: TypeRef) -> TypeRef {
        physical_closure_type(self.ctx, function)
    }

    pub(super) fn convert_attribute(&mut self, attribute: &Attribute) -> Attribute {
        attribute.map_types(|ty| self.convert_type(ty))
    }

    pub(super) fn convert_attrs(
        &mut self,
        attrs: &AttributeMap,
    ) -> Vec<(trunk_ir::StringRef, Attribute)> {
        attrs
            .iter()
            .map(|(key, value)| (key, self.convert_attribute(value)))
            .collect()
    }

    pub(super) fn convert_type(&mut self, ty: TypeRef) -> TypeRef {
        if let Some(converted) = self.converted_types.get(&ty) {
            return *converted;
        }
        if tribute_control::FuncSig::from_type_ref(self.ctx, ty).is_some() {
            let function = self.source_callable(ty);
            let converted = physical_closure_type(self.ctx, function);
            self.converted_types.insert(ty, converted);
            return converted;
        }
        let data = self.ctx.get_type(ty).clone();
        if func::FuncSig::matches(self.ctx, ty) {
            let function = func::FuncSig::from_type_ref(self.ctx, ty)
                .expect("pre-CPS validation must reject malformed func.func_sig types");
            let source_inputs = function.inputs(self.ctx).to_vec();
            let source_results = function.results(self.ctx).to_vec();
            let inputs = source_inputs
                .into_iter()
                .map(|input| self.convert_type(input))
                .collect::<Vec<_>>();
            let results = source_results
                .into_iter()
                .map(|result| self.convert_type(result))
                .collect::<Vec<_>>();
            let mut attrs = data.attrs;
            func::FuncSig::remove_reserved_attrs(self.ctx, &mut attrs);
            for value in attrs.values_mut() {
                *value = self.convert_attribute(value);
            }
            let converted =
                func::func_sig_with_attrs(self.ctx, inputs, results, attrs).as_type_ref();
            self.converted_types.insert(ty, converted);
            return converted;
        }
        if tribute_control::ResumeToken::matches(self.ctx, ty) && data.params.len() == 2 {
            let input = self.convert_type(data.params[0]);
            let answer = self.convert_type(data.params[1]);
            let converted = self.resumption_type(input, answer);
            self.converted_types.insert(ty, converted);
            return converted;
        }
        // A tuple layout is named by its element types, so converting them
        // renames it, and tuples that differed only in rows become one
        // (`design/tribute-dialects.md` § "Nominal layout의 게시").
        if let Some(elements) = tribute_control::logical_tuple_elements(self.ctx, ty) {
            let mut elements = elements.to_vec();
            for element in &mut elements {
                *element = self.convert_type(*element);
            }
            let converted = tribute_control::logical_tuple_type(self.ctx, &elements);
            self.converted_types.insert(ty, converted);
            return converted;
        }

        // Nominal row arguments and layout row binders stay inside the
        // source-logical boundary, as callable rows do
        // (`design/tribute-dialects.md` § "Nominal row 인자"): a row
        // instance becomes its layout's key.
        if let Some(def) = tribute_ir::dialect::adt::row_instance_def(self.ctx, ty) {
            let converted = self.convert_type(def);
            self.converted_types.insert(ty, converted);
            return converted;
        }
        let erased = if tribute_ir::dialect::adt::is_nominal(self.ctx, ty) {
            Some(tribute_control::ROW_PARAMS_ATTR)
        } else {
            None
        };
        let params: Vec<_> = data
            .params
            .iter()
            .copied()
            .map(|param| self.convert_type(param))
            .collect();
        let erased = erased.and_then(|name| self.ctx.lookup_str(name));
        let attrs: Vec<_> = data
            .attrs
            .iter()
            .filter(|(key, _)| Some(*key) != erased)
            .map(|(key, value)| (key, self.convert_attribute(value)))
            .collect();
        if params == data.params.as_slice()
            && attrs.len() == data.attrs.len()
            && attrs
                .iter()
                .all(|(key, value)| data.attrs.view(self.ctx).get(*key) == Some(value))
        {
            self.converted_types.insert(ty, ty);
            return ty;
        }
        let mut builder = TypeDataBuilder::new(data.name).params(params);
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        let converted = builder.intern(self.ctx);
        self.converted_types.insert(ty, converted);
        converted
    }

    /// The callable identity of a closure of the source signature `logical`:
    /// its converted parameters and result, without hidden parameters
    /// (`design/cps-effects.md`, "Closure type의 callable").
    pub(super) fn source_callable(&mut self, logical: TypeRef) -> TypeRef {
        let callable = tribute_control::FuncSig::from_type_ref(self.ctx, logical)
            .expect("pre-CPS validation checked function type");
        let (source_result, result_attrs) = callable.result_with_attrs(self.ctx);
        let result_attrs = result_attrs.clone();
        let result = (
            self.convert_type(source_result),
            self.convert_attr_map(&result_attrs),
        );
        let params = self.convert_source_params(callable);
        let attrs = self.convert_signature_attrs(logical);
        func::func_sig_with_param_attrs(self.ctx, params, [result], attrs).as_type_ref()
    }

    /// Lower a source signature to its logical `func.func_sig` entered
    /// through `convention`, in convention order. Parameter attributes follow
    /// the parameters they describe: hidden parameters have none, and a Cps
    /// source result, replaced by `core.never`, drops its own.
    pub(super) fn lower_callable_signature(
        &mut self,
        logical: TypeRef,
        convention: CallingConvention,
    ) -> TypeRef {
        let callable = tribute_control::FuncSig::from_type_ref(self.ctx, logical)
            .expect("pre-CPS validation checked function type");
        let (source_result, result_attrs) = callable.result_with_attrs(self.ctx);
        let result_attrs = result_attrs.clone();
        let result = self.convert_type(source_result);
        let result_attrs = self.convert_attr_map(&result_attrs);
        let params = self.convert_source_params(callable);
        let evidence = (self.evidence_type(), AttributeMap::new());
        let frame = (self.frame_type(result), AttributeMap::new());
        let abi = CallableAbi::new(convention, params, (result, result_attrs));
        let params = abi.lowered_params(evidence, frame);
        let result = if convention == CallingConvention::Cps {
            (self.never_type(), AttributeMap::new())
        } else {
            abi.source_result
        };
        let attrs = self.convert_signature_attrs(logical);
        func::func_sig_with_param_attrs(self.ctx, params, [result], attrs).as_type_ref()
    }

    fn convert_source_params(
        &mut self,
        callable: tribute_control::FuncSig,
    ) -> Vec<(TypeRef, AttributeMap)> {
        let source_params: Vec<_> = callable
            .inputs_with_attrs(self.ctx)
            .map(|(param, attrs)| (param, attrs.clone()))
            .collect();
        source_params
            .into_iter()
            .map(|(param, attrs)| (self.convert_type(param), self.convert_attr_map(&attrs)))
            .collect()
    }

    fn convert_signature_attrs(&mut self, logical: TypeRef) -> AttributeMap {
        let mut attrs = self.ctx.get_type(logical).attrs.clone();
        tribute_control::FuncSig::remove_reserved_attrs(self.ctx, &mut attrs);
        for value in attrs.values_mut() {
            *value = self.convert_attribute(value);
        }
        attrs
    }

    pub(super) fn convert_attr_map(&mut self, attrs: &AttributeMap) -> AttributeMap {
        let attrs = self.convert_attrs(attrs);
        AttributeMap::from_refs(self.ctx, attrs)
    }

    pub(super) fn copy_extra_attrs(&mut self, source: OpRef, target: OpRef, excluded: &[&str]) {
        let attrs = self.convert_attrs(&self.ctx.op(source).attributes.clone());
        for (key, value) in attrs {
            if !excluded.contains(&self.ctx.str(key)) {
                self.ctx.op_attrs_mut(target).insert(key, value);
            }
        }
    }
}
