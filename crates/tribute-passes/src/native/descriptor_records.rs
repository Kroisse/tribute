//! Native runtime type descriptor records.
//!
//! Native RTTI generation emits `__tribute_rtti`, one read-only data object
//! holding a fixed-size record per RTTI index, the index's release function
//! and its descriptor, followed by the enum records, field arrays, and names
//! the records address by offset, in the format `new-plans/rc.md` fixes. Records are built while
//! nominal layouts still carry their names, so lower passes see only the
//! index.

use std::collections::HashMap;

use tribute_ir::dialect::adt;
use tribute_ir::dialect::adt::layout::get_enum_variants;
use tribute_ir::dialect::tribute_rtti::FieldKind;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::clif;
use trunk_ir::ops::DialectType;
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::Location;
use trunk_ir::{BlockRef, OpRef, StringRef, Symbol, SymbolPath, TypeRef};

use super::rtti::{RTTI_BOOL, RTTI_FLOAT, RTTI_INT, RTTI_NAT, RTTI_NIL};

/// Name of the RTTI table: one record per RTTI index.
pub const RTTI_TABLE: &str = "__tribute_rtti";

const POINTER_SIZE: usize = 8;
/// Size of one RTTI record.
pub const RECORD_SIZE: usize = 40;
/// Offset of the release function pointer in a record.
pub const RELEASE_FN_OFFSET: usize = 0;
const FIELD_SIZE: usize = 12;

/// The `kind` word of a descriptor record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    Struct = 0,
    Variant = 1,
    Enum = 2,
    Builtin = 3,
}

/// The contents of one descriptor record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorRecord {
    pub kind: RecordKind,
    pub name: String,
    /// The enum layout a variant belongs to, and the variant's position in it.
    pub owner: Option<(TypeRef, u32)>,
    pub fields: Vec<(String, FieldKind)>,
}

impl DescriptorRecord {
    /// The record of the descriptor `(ty, tag)`, read from its nominal layout.
    pub fn of_layout(
        ctx: &IrContext,
        ty: TypeRef,
        tag: Option<StringRef>,
        fields: Vec<FieldKind>,
    ) -> Self {
        match tag {
            None => {
                let layout = adt::Struct::from_type_ref(ctx, ty)
                    .expect("a struct descriptor has an adt.struct layout");
                let names = layout.fields(ctx).map(|(name, _)| name.to_owned());
                Self {
                    kind: RecordKind::Struct,
                    name: layout.name(ctx).to_owned(),
                    owner: None,
                    fields: names.zip(fields).collect(),
                }
            }
            Some(tag) => {
                let position = get_enum_variants(ctx, ty)
                    .and_then(|variants| variants.iter().position(|(name, _)| *name == tag))
                    .expect("a variant descriptor names a variant of its adt.enum layout");
                Self {
                    kind: RecordKind::Variant,
                    name: ctx.str(tag).to_owned(),
                    owner: Some((ty, position as u32)),
                    fields: positional(fields),
                }
            }
        }
    }

    fn builtin(name: &str, fields: Vec<FieldKind>) -> Self {
        Self {
            kind: RecordKind::Builtin,
            name: name.to_owned(),
            owner: None,
            fields: positional(fields),
        }
    }
}

fn positional(fields: Vec<FieldKind>) -> Vec<(String, FieldKind)> {
    fields
        .into_iter()
        .enumerate()
        .map(|(position, kind)| (position.to_string(), kind))
        .collect()
}

/// The records of the reserved RTTI indices.
fn reserved_records() -> [(u32, DescriptorRecord); 5] {
    [
        (RTTI_NIL, DescriptorRecord::builtin("Bytes", vec![])),
        (
            RTTI_BOOL,
            DescriptorRecord::builtin("Bool", vec![FieldKind::Bool]),
        ),
        (
            RTTI_NAT,
            DescriptorRecord::builtin(
                "Nat",
                vec![FieldKind::Int {
                    width: 32,
                    signed: false,
                }],
            ),
        ),
        (
            RTTI_INT,
            DescriptorRecord::builtin(
                "Int",
                vec![FieldKind::Int {
                    width: 32,
                    signed: true,
                }],
            ),
        ),
        (
            RTTI_FLOAT,
            DescriptorRecord::builtin("Float", vec![FieldKind::Float { width: 64 }]),
        ),
    ]
}

/// Emit `__tribute_rtti`: the record of every reserved index and of each of
/// `records`, with the release function `release_fns` names for its index,
/// followed by the enum records, field arrays, and names they refer to.
pub fn generate(
    ctx: &mut IrContext,
    module_block: BlockRef,
    records: Vec<(u32, DescriptorRecord)>,
    release_fns: &HashMap<u32, SymbolPath>,
    loc: Location,
) {
    let records = reserved_records()
        .into_iter()
        .chain(records)
        .collect::<Vec<_>>();
    let entries = records.iter().map(|&(index, _)| index).max().unwrap_or(0) as usize + 1;

    // Enum records follow the index records, then the field arrays, then the
    // names, so every reference is an offset known before writing.
    let mut enums = Vec::<(TypeRef, DescriptorRecord)>::new();
    for (_, record) in &records {
        if let Some((owner, _)) = record.owner
            && !enums.iter().any(|(ty, _)| *ty == owner)
        {
            let name = adt::nominal_name(ctx, owner)
                .expect("an enum layout has a nominal name")
                .to_owned();
            enums.push((
                owner,
                DescriptorRecord {
                    kind: RecordKind::Enum,
                    name,
                    owner: None,
                    fields: vec![],
                },
            ));
        }
    }
    let enum_base = entries * RECORD_SIZE;
    let fields_base = enum_base + enums.len() * RECORD_SIZE;
    let field_count = records
        .iter()
        .map(|(_, record)| record.fields.len())
        .sum::<usize>();
    let mut layout = Layout {
        bytes: vec![0; fields_base + field_count * FIELD_SIZE],
        next_fields: fields_base,
        names: HashMap::new(),
        name_bytes: Vec::new(),
        names_base: fields_base + field_count * FIELD_SIZE,
    };

    let enum_offsets = enums
        .iter()
        .enumerate()
        .map(|(position, (ty, _))| (*ty, enum_base + position * RECORD_SIZE))
        .collect::<HashMap<_, _>>();
    for (position, (_, record)) in enums.iter().enumerate() {
        layout.write_record(enum_base + position * RECORD_SIZE, record, &enum_offsets);
    }
    let mut relocs = Vec::new();
    for (index, record) in &records {
        let base = *index as usize * RECORD_SIZE;
        if let Some(release) = release_fns.get(index).cloned() {
            relocs.push((base + RELEASE_FN_OFFSET, release));
        }
        layout.write_record(base, record, &enum_offsets);
    }

    let mut bytes = layout.bytes;
    bytes.extend(layout.name_bytes);
    let relocs = reloc_region(ctx, relocs, loc);
    // Zero bytes rather than zero-initialized data, so the table lives in a
    // data section: macOS linkers reject relocations in zero-fill sections.
    let table = clif::Data::operands()
        .sym_name(Symbol::new(RTTI_TABLE))
        .bytes(bytes.into())
        .align(POINTER_SIZE as u32)
        .regions(relocs)
        .build(ctx, loc);
    ctx.push_op(module_block, table.op_ref());
}

/// The bytes of `__tribute_rtti` while it is being written.
struct Layout {
    bytes: Vec<u8>,
    /// Offset of the next unwritten field array.
    next_fields: usize,
    /// Offset of each distinct name, relative to `names_base`.
    names: HashMap<String, usize>,
    name_bytes: Vec<u8>,
    names_base: usize,
}

impl Layout {
    /// Write the descriptor part of a record at `base`, with its field array.
    fn write_record(
        &mut self,
        base: usize,
        record: &DescriptorRecord,
        enum_offsets: &HashMap<TypeRef, usize>,
    ) {
        put_u32(&mut self.bytes, base + 8, record.kind as u32);
        put_u32(&mut self.bytes, base + 12, record.fields.len() as u32);
        self.put_name(base + 16, &record.name);
        if let Some((owner, position)) = record.owner {
            put_u32(&mut self.bytes, base + 24, position);
            put_offset(&mut self.bytes, base + 28, enum_offsets[&owner]);
        }
        if !record.fields.is_empty() {
            let fields = self.next_fields;
            self.next_fields += record.fields.len() * FIELD_SIZE;
            put_offset(&mut self.bytes, base + 32, fields);
            for (position, (name, kind)) in record.fields.iter().enumerate() {
                let at = fields + position * FIELD_SIZE;
                self.put_name(at, name);
                put_u32(&mut self.bytes, at + 8, kind.record_code());
            }
        }
    }

    /// Store the offset and length of `name` at `at` and `at + 4`. An empty
    /// name has no offset.
    fn put_name(&mut self, at: usize, name: &str) {
        put_u32(&mut self.bytes, at + 4, name.len() as u32);
        if name.is_empty() {
            return;
        }
        let relative = match self.names.get(name) {
            Some(&relative) => relative,
            None => {
                let relative = self.name_bytes.len();
                self.name_bytes.extend_from_slice(name.as_bytes());
                self.names.insert(name.to_owned(), relative);
                relative
            }
        };
        put_offset(&mut self.bytes, at, self.names_base + relative);
    }
}

fn reloc_region(
    ctx: &mut IrContext,
    relocs: Vec<(usize, SymbolPath)>,
    loc: Location,
) -> Option<trunk_ir::RegionRef> {
    if relocs.is_empty() {
        return None;
    }
    let block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    for (offset, func) in relocs {
        let reloc: OpRef = clif::FuncReloc::operands()
            .offset(u32::try_from(offset).expect("record offset fits u32"))
            .func(func)
            .build(ctx, loc)
            .op_ref();
        ctx.push_op(block, reloc);
    }
    Some(ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    }))
}

fn put_offset(bytes: &mut [u8], at: usize, offset: usize) {
    put_u32(
        bytes,
        at,
        u32::try_from(offset).expect("RTTI offset fits u32"),
    );
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn variant_records_point_at_their_enum_record() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Choice = adt.enum<{name = "Choice", variants = [["None", []], ["Some", [tribute_rt.anyref]]]}>
  func.func @f() -> core.nil {
    func.return
  }
}"#,
        );
        let choice = ctx.type_alias_by_text("Choice").expect("Choice alias");
        let some = ctx.intern_str("Some");
        let record =
            DescriptorRecord::of_layout(&ctx, choice, Some(some), vec![FieldKind::Dynamic]);
        assert_eq!(record.kind, RecordKind::Variant);
        assert_eq!(record.name, "Some");
        assert_eq!(record.owner, Some((choice, 1)));
        assert_eq!(record.fields, [("0".to_owned(), FieldKind::Dynamic)]);

        let block = module.first_block(&ctx).unwrap();
        let loc = ctx.op(module.op()).location;
        let release = SymbolPath::from("__tribute_release_5");
        generate(
            &mut ctx,
            block,
            vec![(5, record)],
            &HashMap::from([(5, release.clone())]),
            loc,
        );

        let table = module
            .ops(&ctx)
            .iter()
            .find_map(|&op| clif::Data::from_op(&ctx, op).ok())
            .expect("one RTTI data object");
        assert_eq!(table.sym_name(&ctx), RTTI_TABLE);
        assert_eq!(
            table.relocations(&ctx),
            [((5 * RECORD_SIZE + RELEASE_FN_OFFSET) as u32, release)],
            "only the release function is relocated"
        );
        let bytes = table.bytes(&ctx).to_vec();
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let text = |at: usize| {
            let (offset, len) = (word(at) as usize, word(at + 4) as usize);
            std::str::from_utf8(&bytes[offset..offset + len]).unwrap()
        };

        let variant = 5 * RECORD_SIZE;
        assert_eq!(word(variant + 8), RecordKind::Variant as u32);
        assert_eq!(word(variant + 12), 1);
        assert_eq!(text(variant + 16), "Some");
        assert_eq!(word(variant + 24), 1);
        let owner = word(variant + 28) as usize;
        assert_eq!(
            owner,
            6 * RECORD_SIZE,
            "the enum record follows the index records"
        );
        assert_eq!(word(owner + 8), RecordKind::Enum as u32);
        assert_eq!(word(owner + 12), 0);
        assert_eq!(text(owner + 16), "Choice");
        let fields = word(variant + 32) as usize;
        assert_eq!(text(fields), "0");
        assert_eq!(word(fields + 8), FieldKind::Dynamic.record_code());

        // Every reserved index has a named builtin record.
        for index in 0..5 {
            let base = index * RECORD_SIZE;
            assert_eq!(word(base + 8), RecordKind::Builtin as u32);
            assert!(!text(base + 16).is_empty());
        }
    }
}
