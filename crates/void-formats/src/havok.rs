//! Havok 2016 binary tagfiles (`TAG0`) and type compendiums (`TCM0`), as
//! embedded in `physhape` `.hkx` and `physworld` `.bphysworld` resources
//! (docs/formats/havok.md).
//!
//! This is a generic, schema-driven reader: the file's own type bodies
//! (`TBOD`) define every record layout, so no Havok class layout is hard-coded
//! here. Values are decoded lazily per item; pointer, array, and string
//! fields hold item indices and are returned as [`Value::Ref`] for the caller
//! to follow. Observed encoding (O, tutorial_p door shape and physworld):
//!
//! ```text
//! section  = u32 BE (bits 30-31 flags: 0 container, 1 leaf; bits 0-29 size incl. header) + tag[4]
//! TAG0     = SDKV "20160100", DATA, TYPE{TSTR TNAM FSTR TBOD [TPAD]} | TCRF, INDX{ITEM [PTCH]}
//! TCM0     = TCID [8], TYPE{TPTR TSTR TNAM FSTR TBOD THSH TPAD}
//! TNAM     = varint n, then for types 1..n: varint name (TSTR index), varint k, k x (varint name, varint value)
//! TBOD     = repeated: varint type, varint parent, varint opt, then by opt bit:
//!            0x01 format, 0x02 subtype, 0x04 version, 0x08 size+align, 0x10 flags,
//!            0x20 members (n x {name FSTR, flags, offset, type}), 0x40 interfaces (n x {type, value}),
//!            0x80 attribute
//! ITEM     = 12-byte LE records {u32 type | flags << 24, u32 data offset, u32 count}; item 0 is null
//! PTCH     = repeated {u32 LE type, u32 LE n, n x u32 LE data offset}
//! ```

use crate::Error;

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::Invalid {
        field,
        detail: detail.into(),
    }
}

/// One section of a tagfile; `offset` is the body's offset in the parsed buffer.
#[derive(Clone, Copy, Debug)]
pub struct Section<'a> {
    pub tag: [u8; 4],
    pub container: bool,
    pub offset: usize,
    pub body: &'a [u8],
}

/// Splits `data[start..end]` into consecutive sections (one level).
pub fn sections(data: &[u8], start: usize, end: usize) -> Result<Vec<Section<'_>>, Error> {
    let mut out = Vec::new();
    let mut o = start;
    while o < end {
        if end - o < 8 {
            return Err(Error::Truncated {
                field: "havok section header",
                offset: o,
                needed: 8,
                available: end - o,
            });
        }
        let h = u32::from_be_bytes(data[o..o + 4].try_into().expect("4 bytes"));
        let (flags, size) = (h >> 30, (h & 0x3FFF_FFFF) as usize);
        let tag: [u8; 4] = data[o + 4..o + 8].try_into().expect("4 bytes");
        if size < 8 || size > end - o {
            return Err(invalid(
                "havok section",
                format!(
                    "{} at {o:#x}: size {size}, {} available",
                    String::from_utf8_lossy(&tag),
                    end - o
                ),
            ));
        }
        if flags > 1 {
            return Err(invalid("havok section flags", format!("{flags} at {o:#x}")));
        }
        out.push(Section {
            tag,
            container: flags == 0,
            offset: o + 8,
            body: &data[o + 8..o + size],
        });
        o += size;
    }
    Ok(out)
}

fn find<'a>(secs: &[Section<'a>], tag: &[u8; 4]) -> Option<Section<'a>> {
    secs.iter().find(|s| &s.tag == tag).copied()
}

fn need<'a>(secs: &[Section<'a>], tag: &'static [u8; 4]) -> Result<Section<'a>, Error> {
    find(secs, tag).ok_or_else(|| {
        invalid(
            "havok section",
            format!("missing {}", String::from_utf8_lossy(tag)),
        )
    })
}

fn children<'a>(data: &'a [u8], s: &Section<'a>) -> Result<Vec<Section<'a>>, Error> {
    if !s.container {
        return Err(invalid(
            "havok section",
            format!("{} is not a container", String::from_utf8_lossy(&s.tag)),
        ));
    }
    sections(data, s.offset, s.offset + s.body.len())
}

/// Cursor for the variable-length integers used by TNAM/TBOD.
struct Varints<'a> {
    d: &'a [u8],
    pos: usize,
    base: usize,
}

impl Varints<'_> {
    fn done(&self) -> bool {
        self.pos >= self.d.len()
    }

    fn byte(&mut self) -> Result<u8, Error> {
        let b = *self.d.get(self.pos).ok_or(Error::Truncated {
            field: "havok varint",
            offset: self.base + self.pos,
            needed: 1,
            available: 0,
        })?;
        self.pos += 1;
        Ok(b)
    }

    fn next(&mut self) -> Result<u64, Error> {
        let b = self.byte()?;
        let (extra, high) = match b {
            0x00..=0x7F => return Ok(u64::from(b)),
            0x80..=0xBF => (1, b & 0x3F),
            0xC0..=0xDF => (2, b & 0x1F),
            0xE0..=0xE7 => (3, b & 0x07),
            0xE8..=0xEF => (4, b & 0x07),
            0xF0..=0xF7 => (7, b & 0x07),
            0xF8..=0xFF => (8, 0),
        };
        let mut v = u64::from(high);
        for _ in 0..extra {
            v = (v << 8) | u64::from(self.byte()?);
        }
        Ok(v)
    }

    fn index(&mut self, len: usize, field: &'static str) -> Result<u32, Error> {
        let at = self.base + self.pos;
        let v = self.next()?;
        if v >= len as u64 {
            return Err(invalid(field, format!("{v} out of range {len} at {at:#x}")));
        }
        Ok(v as u32)
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, Error> {
        let at = self.base + self.pos;
        u32::try_from(self.next()?).map_err(|_| invalid(field, format!("too large at {at:#x}")))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub name: String,
    pub flags: u32,
    /// Byte offset within the record (absolute, not relative to the parent).
    pub offset: u32,
    pub ty: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Type {
    pub name: String,
    /// Template parameters: `t*` names hold a type index, `v*` names a value.
    pub templates: Vec<(String, u64)>,
    pub parent: u32,
    pub format: Option<u32>,
    pub subtype: Option<u32>,
    pub version: Option<u32>,
    pub size_align: Option<(u32, u32)>,
    pub flags: Option<u32>,
    pub members: Vec<Member>,
    pub interfaces: Vec<(u32, u32)>,
    pub attribute: Option<u32>,
}

/// Value category from the low format bits (D: consistent with every type in
/// the samples; unknown categories are reported, not guessed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Void,
    Opaque,
    Bool,
    String,
    Int {
        signed: bool,
    },
    Float,
    Pointer,
    Record,
    /// Variable-length array: the field holds an item index.
    Array,
    /// Inline array of `n` subtype elements.
    Fixed(u32),
}

impl Kind {
    pub fn from_format(format: u32) -> Result<Self, Error> {
        Ok(match format & 0xF {
            0 => Kind::Void,
            1 => Kind::Opaque,
            2 => Kind::Bool,
            3 => Kind::String,
            4 => Kind::Int {
                signed: format & 0x200 != 0,
            },
            5 => Kind::Float,
            6 => Kind::Pointer,
            7 => Kind::Record,
            8 if format & 0x20 != 0 => Kind::Fixed(format >> 8),
            8 => Kind::Array,
            _ => return Err(invalid("havok type format", format!("{format:#x}"))),
        })
    }
}

/// Types of a tagfile or compendium; index 0 is the null type.
#[derive(Clone, Debug, Default)]
pub struct TypeSet {
    pub types: Vec<Type>,
    /// `TCID` of a compendium.
    pub id: Option<[u8; 8]>,
}

fn strings(b: &[u8]) -> Vec<String> {
    let mut v: Vec<String> = b
        .split(|&c| c == 0)
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();
    if b.last() == Some(&0) {
        v.pop();
    }
    v
}

impl TypeSet {
    fn parse(data: &[u8], type_section: &Section) -> Result<Self, Error> {
        let secs = children(data, type_section)?;
        let tstr = strings(need(&secs, b"TSTR")?.body);
        let fstr = strings(need(&secs, b"FSTR")?.body);
        let tnam = need(&secs, b"TNAM")?;
        let mut r = Varints {
            d: tnam.body,
            pos: 0,
            base: tnam.offset,
        };
        let count = r.next()?;
        if count == 0 || count > tnam.body.len() as u64 {
            return Err(invalid("havok type count", count.to_string()));
        }
        let mut types = vec![Type::default(); count as usize];
        for t in types.iter_mut().skip(1) {
            t.name = tstr[r.index(tstr.len(), "havok type name")? as usize].clone();
            let k = r.next()?;
            if k > 16 {
                return Err(invalid("havok template count", k.to_string()));
            }
            for _ in 0..k {
                let n = tstr[r.index(tstr.len(), "havok template name")? as usize].clone();
                let v = r.next()?;
                t.templates.push((n, v));
            }
        }
        let n = types.len();
        let tbod = need(&secs, b"TBOD")?;
        let mut r = Varints {
            d: tbod.body,
            pos: 0,
            base: tbod.offset,
        };
        while !r.done() {
            let ty = r.index(n, "havok body type")? as usize;
            if ty == 0 {
                continue;
            }
            let t = &mut types[ty];
            t.parent = r.index(n, "havok parent type")?;
            let opt = r.u32("havok body options")?;
            if opt & !0xFF != 0 {
                return Err(invalid("havok body options", format!("{opt:#x}")));
            }
            if opt & 0x01 != 0 {
                t.format = Some(r.u32("havok format")?);
            }
            if opt & 0x02 != 0 {
                t.subtype = Some(r.index(n, "havok subtype")?);
            }
            if opt & 0x04 != 0 {
                t.version = Some(r.u32("havok version")?);
            }
            if opt & 0x08 != 0 {
                t.size_align = Some((r.u32("havok size")?, r.u32("havok align")?));
            }
            if opt & 0x10 != 0 {
                t.flags = Some(r.u32("havok type flags")?);
            }
            if opt & 0x20 != 0 {
                let m = r.u32("havok member count")?;
                if m as usize > tbod.body.len() {
                    return Err(invalid("havok member count", m.to_string()));
                }
                for _ in 0..m {
                    let name = fstr[r.index(fstr.len(), "havok member name")? as usize].clone();
                    t.members.push(Member {
                        name,
                        flags: r.u32("havok member flags")?,
                        offset: r.u32("havok member offset")?,
                        ty: r.index(n, "havok member type")?,
                    });
                }
            }
            if opt & 0x40 != 0 {
                let m = r.u32("havok interface count")?;
                if m as usize > tbod.body.len() {
                    return Err(invalid("havok interface count", m.to_string()));
                }
                for _ in 0..m {
                    t.interfaces.push((
                        r.index(n, "havok interface")?,
                        r.u32("havok interface value")?,
                    ));
                }
            }
            if opt & 0x80 != 0 {
                t.attribute = Some(r.u32("havok attribute")?);
            }
        }
        Ok(Self { types, id: None })
    }

    /// Parses a standalone `TCM0` type compendium.
    pub fn parse_compendium(data: &[u8]) -> Result<Self, Error> {
        let top = sections(data, 0, data.len())?;
        let [tcm] = top.as_slice() else {
            return Err(invalid("havok compendium", "expected one TCM0 section"));
        };
        if &tcm.tag != b"TCM0" {
            return Err(Error::BadMagic {
                expected: *b"TCM0",
                found: tcm.tag,
            });
        }
        let secs = children(data, tcm)?;
        let tcid = need(&secs, b"TCID")?;
        let id: [u8; 8] = tcid
            .body
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| invalid("havok TCID", "shorter than 8 bytes"))?;
        let mut set = Self::parse(data, &need(&secs, b"TYPE")?)?;
        set.id = Some(id);
        Ok(set)
    }

    pub fn get(&self, ty: u32) -> Result<&Type, Error> {
        self.types
            .get(ty as usize)
            .ok_or_else(|| invalid("havok type index", ty.to_string()))
    }

    pub fn by_name(&self, name: &str) -> Option<u32> {
        self.types
            .iter()
            .position(|t| t.name == name)
            .map(|i| i as u32)
    }

    fn chain(&self, mut ty: u32) -> Result<Vec<u32>, Error> {
        let mut out = Vec::new();
        while ty != 0 {
            if out.len() > 64 || out.contains(&ty) {
                return Err(invalid(
                    "havok parent chain",
                    format!("cycle or depth at {ty}"),
                ));
            }
            out.push(ty);
            ty = self.get(ty)?.parent;
        }
        Ok(out)
    }

    /// Format, subtype, and size of `ty`, inherited from parents when absent.
    pub fn layout(&self, ty: u32) -> Result<Layout, Error> {
        let chain = self.chain(ty)?;
        let first = |f: fn(&Type) -> Option<u32>| -> Result<Option<u32>, Error> {
            for &t in &chain {
                if let Some(v) = f(self.get(t)?) {
                    return Ok(Some(v));
                }
            }
            Ok(None)
        };
        let format = first(|t| t.format)?.ok_or_else(|| {
            invalid(
                "havok type",
                format!(
                    "{} has no format",
                    self.get(ty).map(|t| t.name.as_str()).unwrap_or("?")
                ),
            )
        })?;
        let size = first(|t| t.size_align.map(|s| s.0))?.unwrap_or(0);
        Ok(Layout {
            kind: Kind::from_format(format)?,
            format,
            subtype: first(|t| t.subtype)?,
            size,
        })
    }

    /// All members of `ty` including inherited ones, base class first.
    pub fn all_members(&self, ty: u32) -> Result<Vec<&Member>, Error> {
        let mut chain = self.chain(ty)?;
        chain.reverse();
        let mut out = Vec::new();
        for t in chain {
            out.extend(self.get(t)?.members.iter());
        }
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub kind: Kind,
    pub format: u32,
    pub subtype: Option<u32>,
    pub size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub ty: u32,
    /// High byte of the type word (O: 0x10 single object, 0x20 array).
    pub flags: u8,
    pub offset: u32,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Void,
    /// Opaque bytes are not interpreted.
    Opaque,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    /// Item index of a pointer, array, or string (0 = null/empty).
    Ref(u32),
    Fixed(Vec<Value>),
    Record(Vec<(String, Value)>),
}

impl Value {
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Record(f) => f.iter().find(|(n, _)| n == name).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn path(&self, names: &[&str]) -> Option<&Value> {
        names.iter().try_fold(self, |v, n| v.field(n))
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::Float(f) => Some(f),
            Value::Int(i) => Some(i as f64),
            Value::UInt(u) => Some(u as f64),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::UInt(u) => Some(u),
            Value::Int(i) => u64::try_from(i).ok(),
            Value::Bool(b) => Some(u64::from(b)),
            Value::Ref(r) => Some(u64::from(r)),
            _ => None,
        }
    }

    pub fn as_ref(&self) -> Option<u32> {
        match *self {
            Value::Ref(r) => Some(r),
            _ => None,
        }
    }

    /// `hkVector4f`-like values: fixed arrays of floats, or records whose
    /// single field is one.
    pub fn floats(&self) -> Option<Vec<f32>> {
        match self {
            Value::Fixed(v) => v.iter().map(|x| x.as_f64().map(|f| f as f32)).collect(),
            Value::Record(f) if f.len() == 1 => f[0].1.floats(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TagFile {
    pub sdk: String,
    pub types: TypeSet,
    pub items: Vec<Item>,
    pub data: Vec<u8>,
    /// `PTCH` groups: (type, data offsets of fields holding item indices).
    pub patches: Vec<(u32, Vec<u32>)>,
    /// `TCRF` id when the types come from a compendium.
    pub compendium_ref: Option<[u8; 8]>,
}

const MAX_DEPTH: u32 = 32;

impl TagFile {
    /// Parses a buffer that starts with the `TAG0` section header. Types come
    /// from the file's `TYPE` section or, for `TCRF` files, from `compendium`
    /// (whose `TCID` must match).
    pub fn parse(data: &[u8], compendium: Option<&TypeSet>) -> Result<Self, Error> {
        let top = sections(data, 0, data.len())?;
        let [tag] = top.as_slice() else {
            return Err(invalid(
                "havok tagfile",
                format!("{} top-level sections", top.len()),
            ));
        };
        if &tag.tag != b"TAG0" {
            return Err(Error::BadMagic {
                expected: *b"TAG0",
                found: tag.tag,
            });
        }
        let secs = children(data, tag)?;
        let sdk = String::from_utf8_lossy(need(&secs, b"SDKV")?.body).into_owned();
        if sdk != "20160100" {
            return Err(invalid(
                "havok SDK version",
                format!("{sdk} (only 20160100 is known)"),
            ));
        }
        let body = need(&secs, b"DATA")?.body.to_vec();
        let (types, compendium_ref) = match (find(&secs, b"TYPE"), find(&secs, b"TCRF")) {
            (Some(t), _) => (TypeSet::parse(data, &t)?, None),
            (None, Some(r)) => {
                let id: [u8; 8] = r
                    .body
                    .get(..8)
                    .and_then(|b| b.try_into().ok())
                    .ok_or_else(|| invalid("havok TCRF", "shorter than 8 bytes"))?;
                let c = compendium.ok_or_else(|| invalid("havok TCRF", "compendium required"))?;
                if c.id != Some(id) {
                    return Err(invalid(
                        "havok TCRF",
                        format!("id {id:02x?} does not match compendium {:02x?}", c.id),
                    ));
                }
                (c.clone(), Some(id))
            }
            (None, None) => return Err(invalid("havok tagfile", "no TYPE or TCRF section")),
        };
        let indx = children(data, &need(&secs, b"INDX")?)?;
        let item_sec = need(&indx, b"ITEM")?.body;
        if item_sec.len() % 12 != 0 {
            return Err(invalid("havok ITEM", format!("length {}", item_sec.len())));
        }
        let le = |b: &[u8; 4]| u32::from_le_bytes(*b);
        let mut items = Vec::with_capacity(item_sec.len() / 12);
        for (i, c) in item_sec.as_chunks::<12>().0.iter().enumerate() {
            let w = le(c[0..4].try_into().expect("4 bytes"));
            let item = Item {
                ty: w & 0x00FF_FFFF,
                flags: (w >> 24) as u8,
                offset: le(c[4..8].try_into().expect("4 bytes")),
                count: le(c[8..12].try_into().expect("4 bytes")),
            };
            if item.ty != 0 {
                let size = u64::from(types.layout(item.ty)?.size);
                let end = u64::from(item.offset) + size * u64::from(item.count);
                if end > body.len() as u64 {
                    return Err(invalid(
                        "havok item",
                        format!("item {i} ends at {end}, data is {} bytes", body.len()),
                    ));
                }
            }
            items.push(item);
        }
        let mut patches = Vec::new();
        if let Some(p) = find(&indx, b"PTCH") {
            let b = p.body;
            let mut o = 0;
            while o < b.len() {
                if b.len() - o < 8 {
                    return Err(invalid("havok PTCH", "truncated group"));
                }
                let word = |at: usize| le(b[at..at + 4].try_into().expect("4 bytes"));
                let (ty, n) = (word(o), word(o + 4) as usize);
                o += 8;
                if n > (b.len() - o) / 4 {
                    return Err(invalid("havok PTCH", format!("count {n}")));
                }
                patches.push((
                    ty,
                    b[o..o + 4 * n].as_chunks::<4>().0.iter().map(le).collect(),
                ));
                o += 4 * n;
            }
        }
        Ok(Self {
            sdk,
            types,
            items,
            data: body,
            patches,
            compendium_ref,
        })
    }

    pub fn type_name(&self, ty: u32) -> &str {
        self.types.get(ty).map_or("?", |t| t.name.as_str())
    }

    pub fn item(&self, i: u32) -> Result<&Item, Error> {
        self.items
            .get(i as usize)
            .ok_or_else(|| invalid("havok item index", i.to_string()))
    }

    /// Indices of items whose type is named `name`.
    pub fn items_of<'s>(&'s self, name: &'s str) -> impl Iterator<Item = u32> + 's {
        self.items
            .iter()
            .enumerate()
            .filter(move |(_, it)| it.ty != 0 && self.type_name(it.ty) == name)
            .map(|(i, _)| i as u32)
    }

    /// Element `k` of item `i`.
    pub fn element(&self, i: u32, k: u32) -> Result<Value, Error> {
        let it = *self.item(i)?;
        if k >= it.count {
            return Err(invalid(
                "havok element",
                format!("{k} >= count {}", it.count),
            ));
        }
        let size = self.types.layout(it.ty)?.size;
        self.value(it.ty, it.offset as usize + (k * size) as usize, 0)
    }

    /// All elements of item `i` (a single object yields one element).
    pub fn elements(&self, i: u32) -> Result<Vec<Value>, Error> {
        let it = *self.item(i)?;
        (0..it.count).map(|k| self.element(i, k)).collect()
    }

    /// Follows a reference value to its item's elements; null gives none.
    pub fn deref(&self, v: &Value) -> Result<Vec<Value>, Error> {
        match v {
            Value::Ref(0) => Ok(Vec::new()),
            Value::Ref(r) => self.elements(*r),
            other => Err(invalid("havok reference", format!("{other:?}"))),
        }
    }

    fn bytes(&self, at: usize, n: usize) -> Result<&[u8], Error> {
        self.data.get(at..at + n).ok_or(Error::Truncated {
            field: "havok value",
            offset: at,
            needed: n as u64,
            available: self.data.len().saturating_sub(at),
        })
    }

    fn uint(&self, at: usize, n: usize) -> Result<u64, Error> {
        let b = self.bytes(at, n)?;
        let mut v = [0u8; 8];
        v[..n].copy_from_slice(b);
        Ok(u64::from_le_bytes(v))
    }

    /// Decodes a value of type `ty` at data offset `at`.
    pub fn value(&self, ty: u32, at: usize, depth: u32) -> Result<Value, Error> {
        if depth > MAX_DEPTH {
            return Err(invalid("havok value", "nesting too deep"));
        }
        let l = self.types.layout(ty)?;
        let size = l.size as usize;
        Ok(match l.kind {
            Kind::Void => Value::Void,
            Kind::Opaque => Value::Opaque,
            Kind::Bool => Value::Bool(self.uint(at, size.clamp(1, 8))? != 0),
            Kind::Int { signed } => {
                let n = match l.format & 0x1F000 {
                    0x2000 => 1,
                    0x4000 => 2,
                    0x8000 => 4,
                    0x10000 => 8,
                    _ => size,
                };
                if !matches!(n, 1 | 2 | 4 | 8) {
                    return Err(invalid("havok int", format!("format {:#x}", l.format)));
                }
                let u = self.uint(at, n)?;
                if signed {
                    let shift = 64 - 8 * n as u32;
                    Value::Int(((u << shift) as i64) >> shift)
                } else {
                    Value::UInt(u)
                }
            }
            Kind::Float => match size {
                4 => Value::Float(f64::from(f32::from_bits(self.uint(at, 4)? as u32))),
                8 => Value::Float(f64::from_bits(self.uint(at, 8)?)),
                _ => return Err(invalid("havok float", format!("size {size}"))),
            },
            Kind::Pointer | Kind::Array | Kind::String => {
                let r = self.uint(at, 4)? as u32;
                if r as usize >= self.items.len() {
                    return Err(invalid(
                        "havok reference",
                        format!("item {r} at data {at:#x} (of {})", self.items.len()),
                    ));
                }
                Value::Ref(r)
            }
            Kind::Fixed(n) => {
                let sub = l
                    .subtype
                    .ok_or_else(|| invalid("havok fixed array", "no subtype"))?;
                let es = self.types.layout(sub)?.size as usize;
                if n as usize * es > size.max(n as usize * es) || n > 4096 {
                    return Err(invalid("havok fixed array", format!("{n} x {es}")));
                }
                Value::Fixed(
                    (0..n as usize)
                        .map(|k| self.value(sub, at + k * es, depth + 1))
                        .collect::<Result<_, _>>()?,
                )
            }
            Kind::Record => {
                let members = self.types.all_members(ty)?;
                let mut out = Vec::with_capacity(members.len());
                for m in members {
                    out.push((
                        m.name.clone(),
                        self.value(m.ty, at + m.offset as usize, depth + 1)?,
                    ));
                }
                Value::Record(out)
            }
        })
    }

    /// Data offsets at which the schema-driven decoder reads item references,
    /// across all items. Compared with `PTCH` to validate field layouts.
    pub fn reference_offsets(&self, kinds: &[Kind]) -> Result<Vec<u32>, Error> {
        let mut out = Vec::new();
        for it in self.items.iter().filter(|it| it.ty != 0) {
            let size = self.types.layout(it.ty)?.size;
            for k in 0..it.count {
                self.collect_refs(it.ty, it.offset + k * size, 0, kinds, &mut out)?;
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    fn collect_refs(
        &self,
        ty: u32,
        at: u32,
        depth: u32,
        kinds: &[Kind],
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        if depth > MAX_DEPTH {
            return Err(invalid("havok value", "nesting too deep"));
        }
        let l = self.types.layout(ty)?;
        match l.kind {
            k @ (Kind::Pointer | Kind::Array | Kind::String) => {
                if kinds.contains(&k) {
                    out.push(at);
                }
            }
            Kind::Fixed(n) => {
                let sub = l
                    .subtype
                    .ok_or_else(|| invalid("havok fixed array", "no subtype"))?;
                let es = self.types.layout(sub)?.size;
                for k in 0..n {
                    self.collect_refs(sub, at + k * es, depth + 1, kinds, out)?;
                }
            }
            Kind::Record => {
                for m in self.types.all_members(ty)? {
                    self.collect_refs(m.ty, at + m.offset, depth + 1, kinds, out)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Encodes a varint (inverse of the reader; used by tests and fixtures).
pub fn encode_varint(v: u64, out: &mut Vec<u8>) {
    match v {
        0..=0x7F => out.push(v as u8),
        0x80..=0x3FFF => out.extend_from_slice(&[0x80 | (v >> 8) as u8, v as u8]),
        0x4000..=0x1F_FFFF => {
            out.extend_from_slice(&[0xC0 | (v >> 16) as u8, (v >> 8) as u8, v as u8])
        }
        0x20_0000..=0x7FF_FFFF => {
            out.push(0xE0 | (v >> 24) as u8);
            out.extend_from_slice(&(v as u32).to_be_bytes()[1..]);
        }
        _ => {
            out.push(0xF8);
            out.extend_from_slice(&v.to_be_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(tag: &[u8; 4], leaf: bool, body: &[u8]) -> Vec<u8> {
        let size = (body.len() + 8) as u32 | if leaf { 1 << 30 } else { 0 };
        let mut o = size.to_be_bytes().to_vec();
        o.extend_from_slice(tag);
        o.extend_from_slice(body);
        o
    }

    fn vi(vals: &[u64]) -> Vec<u8> {
        let mut o = Vec::new();
        for &v in vals {
            encode_varint(v, &mut o);
        }
        o
    }

    /// Types: 1 float, 2 vec4 = float[4], 3 u8, 4 Shape { radius: float @0, verts: Array<vec4> @4, kind: u8 @8 },
    /// 5 Array<vec4>.
    fn sample() -> Vec<u8> {
        let tstr = b"float\0vec4\0hkUint8\0Shape\0hkArray\0tT\0".to_vec();
        let fstr = b"radius\0verts\0kind\0".to_vec();
        // TNAM: count 6; names with templates (Array has tT = 2).
        let tnam = vi(&[6, 0, 0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 2]);
        let mut tbod = Vec::new();
        tbod.extend(vi(&[1, 0, 0x09, 0x174605, 4, 4]));
        tbod.extend(vi(&[2, 0, 0x0B, 0x428, 1, 16, 16]));
        tbod.extend(vi(&[3, 0, 0x09, 0x2004, 1, 1]));
        tbod.extend(vi(&[
            4, 0, 0x29, 7, 12, 4, 3, 0, 0x20, 0, 1, 1, 0x20, 4, 5, 2, 0x20, 8, 3,
        ]));
        tbod.extend(vi(&[5, 0, 0x0B, 8, 2, 4, 4]));
        let mut types = section(b"TSTR", true, &tstr);
        types.extend(section(b"TNAM", true, &tnam));
        types.extend(section(b"FSTR", true, &fstr));
        types.extend(section(b"TBOD", true, &tbod));
        let mut data = Vec::new();
        data.extend(0.5f32.to_le_bytes());
        data.extend(2u32.to_le_bytes()); // verts -> item 2
        data.extend([7, 0, 0, 0]);
        data.extend([0; 4]); // pad to 16
        for k in 0..2 {
            for c in 0..4 {
                data.extend(((k * 4 + c) as f32).to_le_bytes());
            }
        }
        let mut item = vec![0u8; 12];
        for (ty, off, n) in [(0x1000_0004u32, 0u32, 1u32), (0x2000_0002, 16, 2)] {
            item.extend(ty.to_le_bytes());
            item.extend(off.to_le_bytes());
            item.extend(n.to_le_bytes());
        }
        let mut ptch = Vec::new();
        for v in [4u32, 1, 4] {
            ptch.extend(v.to_le_bytes());
        }
        let mut indx = section(b"ITEM", true, &item);
        indx.extend(section(b"PTCH", true, &ptch));
        let mut body = section(b"SDKV", true, b"20160100");
        body.extend(section(b"DATA", true, &data));
        body.extend(section(b"TYPE", false, &types));
        body.extend(section(b"INDX", false, &indx));
        section(b"TAG0", false, &body)
    }

    #[test]
    fn varints_round_trip() {
        for v in [
            0u64,
            0x7F,
            0x80,
            0x3FFF,
            0x4000,
            0x1F_FFFF,
            0x20_0000,
            0x7FF_FFFF,
            1 << 40,
        ] {
            let b = vi(&[v]);
            let mut r = Varints {
                d: &b,
                pos: 0,
                base: 0,
            };
            assert_eq!(r.next().unwrap(), v, "{v:#x}");
            assert!(r.done());
        }
    }

    #[test]
    fn decodes_schema_driven_records() {
        let t = TagFile::parse(&sample(), None).unwrap();
        assert_eq!(t.items_of("Shape").collect::<Vec<_>>(), [1]);
        let s = t.element(1, 0).unwrap();
        assert_eq!(s.field("radius"), Some(&Value::Float(0.5)));
        assert_eq!(s.field("kind"), Some(&Value::UInt(7)));
        let verts = t.deref(s.field("verts").unwrap()).unwrap();
        assert_eq!(verts.len(), 2);
        assert_eq!(verts[1].floats().unwrap(), [4.0, 5.0, 6.0, 7.0]);
        assert_eq!(t.reference_offsets(&[Kind::Array]).unwrap(), [4]);
        assert!(t.reference_offsets(&[Kind::Pointer]).unwrap().is_empty());
        assert_eq!(t.patches, [(4, vec![4])]);
    }

    #[test]
    fn rejects_truncation_and_bad_references() {
        let full = sample();
        for cut in 0..full.len() {
            assert!(TagFile::parse(&full[..cut], None).is_err(), "cut {cut}");
        }
        // Point `verts` at a nonexistent item.
        let mut bad = full.clone();
        let data_at = bad.windows(4).position(|w| w == b"DATA").unwrap() + 4;
        bad[data_at + 4] = 9;
        let t = TagFile::parse(&bad, None).unwrap();
        assert!(t.element(1, 0).is_err());
        // Item that extends past DATA.
        let mut bad = full;
        let item_at = bad.windows(4).position(|w| w == b"ITEM").unwrap() + 4;
        bad[item_at + 24 + 8] = 200;
        assert!(TagFile::parse(&bad, None).is_err());
    }

    #[test]
    fn compendium_reference_must_match() {
        let tag = sample();
        // Same file with TYPE replaced by a TCRF to id 1..8.
        let body = sections(&tag, 8, tag.len()).unwrap();
        let mut rebuilt = Vec::new();
        let mut type_body = Vec::new();
        for s in &body {
            if &s.tag == b"TYPE" {
                type_body = s.body.to_vec();
                rebuilt.extend(section(b"TCRF", true, &[1, 2, 3, 4, 5, 6, 7, 8]));
            } else {
                rebuilt.extend(section(&s.tag, !s.container, s.body));
            }
        }
        let tag2 = section(b"TAG0", false, &rebuilt);
        let mut tcm = section(b"TCID", true, &[1, 2, 3, 4, 5, 6, 7, 8]);
        tcm.extend(section(b"TYPE", false, &type_body));
        let comp = TypeSet::parse_compendium(&section(b"TCM0", false, &tcm)).unwrap();
        assert!(TagFile::parse(&tag2, None).is_err());
        let t = TagFile::parse(&tag2, Some(&comp)).unwrap();
        assert_eq!(t.compendium_ref, Some([1, 2, 3, 4, 5, 6, 7, 8]));
        let mut other = comp.clone();
        other.id = Some([0; 8]);
        assert!(TagFile::parse(&tag2, Some(&other)).is_err());
    }
}
