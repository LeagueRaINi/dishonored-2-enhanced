# Resource archives (`base/*.index`, `*.resources`, `*.sharedrsc`)

Implementation: `crates/void-res`. Inspection: `void-tool --game <ROOT> info|list|extract|verify`.

## Identity

| Artifact | SHA-256 |
| --- | --- |
| `Dishonored2.exe` | `b671f8249422862bf557d00aab2f4bb671ed4515cc10ad8a8a9decad84516180` |
| `base/master.index` | `e76d1a13ac95f5614300be65f55ebc1cde68974e792bf72d96e5e5d206aac147` |
| `base/game1_patch.index` | `b8bb68d736f45ee7a1d59e1f27bed2473c423e4b122018878743565134b5f964` |
| `base/game2_patch.index` | `25f74d3f1fcf7d4e8889c008e4ce885a13a4932564923874a6901855aff57936` |
| `base/game3_patch.index` | `9268d81fa6eb00e3dc0f0b08e50b3be0d12beb3d128941bd85f41371cfde015a` |
| `base/game4_patch.index` | `c65204710727d136e6d05e3b66d762dfdf44d91f26473e8b38b1a736be6524c8` |

Reproduce: `cargo run --release -p void-tool -- --game <ROOT> info`.

## Findings

Labels: **O** observed in bytes, **D** derived, **H** hypothesis.

### F001 Container files (O)

`.resources` and `.sharedrsc` begin with `04 53 45 52` (`\x04SER`) and are otherwise
raw data addressed only by index entries. The first game1 entry starts at offset 4.

### F002 `.index` layout (O, all 28 indexes)

Big-endian numbers, little-endian u32 string lengths, UTF-8 strings.

```
u8 0x05 | "SER" | u32 body_size (= file_len - 32) | 24 zero bytes
u32 entry_count
entry: u32 id | str kind | str source | str name | u64 offset
       | u32 size | u32 packed_size | u32 reserved | u32 flags | u16 location
```

Evidence: every index (game1-4, base through `_patch`) parses to exactly its file
length; `reserved` and the header padding are zero in every entry/file; `flags`
is 0 or 0x20 only (meaning unknown). Matches the disrev description except that
disrev groups the final 10 bytes differently; the byte layout is the same.

### F003 `location` word (O for observed values, D for meaning)

Bits 0-1 group (0 = game1), bits 2-4 slot (0 base, 1-5 `_001`-`_005`, 6 `_patch`),
bit 15 shared file. No other bits observed. In the installed patch indexes every
entry resolves to an existing file (`info`: "unresolvable locations 0").

### F004 `master.index` layout (O, single sample)

```
u8 0x04 | "SER" | u16 unknown (=2) | u16 group_count (=4)
group: str index_file | str base_resources
u32 extra_count (=24)
extra: str resources_file | u16 group          (listed in slot order)
u32 shared_count (=1)
shared: u32 n | u16 group[n] | str file         (shared_2_3 -> groups 1,2)
```

Parser requires exact consumption; a first attempt with a u16 `extra_count` failed
on the real file and was corrected. Only one sample exists, so field *widths* of
the shared section are confirmed only for this instance (H for other builds).

### F005 Precedence (D, confidence medium-high)

master.index names `gameN_patch.index` for each group. Each numbered index is a
complete superset of the previous (entry counts 48504 -> 48583 for game1, with
locations increasingly pointing at later slots). Therefore the named patch index
is authoritative and older indexes are not consulted. Not yet confirmed against
the executable's loader.

### F006 Compression (O)

When `size != packed_size` the stored bytes are a standard zlib stream (observed
header `28 15`: deflate, 1 KiB window). Decoding produces exactly `size` bytes and
consumes exactly `packed_size`. Coverage: all 219,315 entries decode (STATUS E008).

### F007 Names are not unique (O)

Duplicate names within a group: 392 (game1), 2 (game2), 4 (game3). Observed causes:
UI `.iggy` payloads listed once as `File` and again under many `iggyFile` decls
with identical sizes; a cluster of zero-size entries of mixed kinds. Lookups must
key on `(kind, name)`; whether duplicates always share bytes is not yet checked.

### F008 Content landmarks (O)

- 530 resource kinds across groups, including ~300 `cpnt*` component decls,
  `entityDef`, `model`/`baseModel`, `material`, `image`, `physhape` (Havok `.hkx`),
  `physworld`, `Navmesh`, `anim`/`animSet`/`skeleton`, `md6Def`, `renderProg`.
- Text decls (`generated/decls/**.decl`) use a brace/`key = value;` syntax.
- Per map (example `tutorial_p`): `maps/.../<map>.entities` (plain text,
  `Version 6`, top-level `component {}` and `entity {}` blocks with `inherit`/`class`),
  `<map>.bwm` (50.6 MB, presumed world geometry), `<map>.bphysworld`, `<map>.bnavmesh`,
  `<map>.tome` (presumed Umbra occlusion), `<map>.probesnetwork`, `<map>.soundpropa`,
  `generated/buildgame/.../<map>.mapresources`, `generated/maps/.../<map>.cfg`.
- `checkmap/*` and `zoo/*` dev maps ship only their `mapInfo` decls (no entities).

### F009 Entry ids name logical resources, not entries (O)

In game1_patch, 48,583 entries share 36,063 distinct ids (max 36,062); 32,318 ids
have one entry, the rest 2-38. Examples: an `image` id groups three entries for the
same texture name prefix (87,578 / 262,164 / 1,048,596 bytes, plausibly mip or
streaming tiers, H); one `iggyFile` id groups an `.imp`, its decl, `.iggy`,
font libraries, and `.raw.iggytex` atlases. Duplicate names (F007) arise inside these groups.

### F010 `.mapresources` = map dependency list by id (O, one map)

`generated/buildgame/campaign/tutorial/tutorial_p.mapresources` (135,868 bytes):

```
u64 zero | u32 count (=4852) | record[count] (28 bytes each, sizes sum exactly)
record: u32 id (BE) | 20 zero bytes | u32 0x00000080
```

4,851 ids resolve in game1_patch (first records: the map's `.bnavmesh`, `.entities`,
then Corvo materials); kinds led by image 1491, anim 702, EnvProbe 531, material 444.
Id `0x63cd` does not resolve in game1 (perhaps `common.mapresources`/another group; unknown).
1,152 records hit multi-entry ids (F009). Meaning of the constant 0x80 unknown.
Probe: `local/` notes; not yet implemented in `void-res`.

## Unknowns

- Meaning of entry `flags` 0x20 and master `unknown` = 2.
- Whether the engine applies any override beyond the patch index (e.g. loose files).
- `.mapresources`, `.bwm`, `.tome`, `.bphysworld` formats.
