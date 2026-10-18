//! Read-only probe of the values photo mode relies on (docs/raytracing.md 11.1, 11.5 step
//! 1): the Ansel session flag, the game pointer, the logic-state stack, the map state, the
//! scoped "current game" and the Ansel camera block. All reads are checked (`mem`); the
//! executable identity was verified before any hook runs (`lib.rs`).

use crate::mem;

/// `byte_1439A529C`: set by the Ansel startSession callback, cleared by stopSession.
pub const SESSION_FLAG_RVA: u64 = 0x39A_529C;
/// `idCommonLocal` (0x1425AF110) + 0x19A8: the game (`vtable +400` getter, O).
pub const GAME_PTR_RVA: u64 = 0x25B_0AB8;
/// `qword_1439A5290`: scoped "current game"; startSession uses it.
pub const CURRENT_GAME_RVA: u64 = 0x39A_5290;
/// `arkGameLogicManager` = game + 0x1F6EC8; +8 points back to the game; state stack
/// `int*` at +1320, count in the low 24 bits of +1328 (`sub_140787900`).
pub const MANAGER: u64 = 0x1F_6EC8;
const MANAGER_GAME: u64 = 8;
const STACK_PTR: u64 = 1320;
const STACK_COUNT: u64 = 1328;
/// `game+3412056 == 3`: a map is running (`idGameLocal` vtable +64, `sub_1408A9670`).
pub const MAP_STATE: u64 = 3_412_056;
pub const MAP_RUNNING: u32 = 3;
/// `ansel::Camera` block: position[3], quaternion (x, y, z, w) at +12, fov (degrees,
/// vertical) at +28, projection offset[2] at +32.
pub const CAMERA_RVA: u64 = 0x397_BC60;

/// Logic-state IDs (d2enhanced, verified there in game).
pub const STATE_IN_GAME: u32 = 1;
pub const STATE_PAUSE: u32 = 2;
pub const STATE_CINEMATIC: u32 = 5;
pub const STATE_MAIN_MENU: u32 = 6;
pub const STATE_MAP_TRANSITION: u32 = 15;

pub fn state_name(id: u32) -> &'static str {
    match id {
        0 => "none",
        STATE_IN_GAME => "in game",
        STATE_PAUSE => "pause",
        STATE_CINEMATIC => "cinematic",
        STATE_MAIN_MENU => "main menu",
        STATE_MAP_TRANSITION => "map transition",
        _ => "other",
    }
}

/// One reading.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Probe {
    /// The executable image was found (all other fields are None/0 without it).
    pub base: bool,
    pub session_flag: Option<u8>,
    /// `*(u64*)0x1425B0AB8`.
    pub game: u64,
    /// `qword_1439A5290`.
    pub current_game: u64,
    /// `manager+8 == game`.
    pub manager_ok: bool,
    /// Top of the logic-state stack (0 = empty) and its depth.
    pub top: u32,
    pub depth: u32,
    /// `game+3412056`.
    pub map_state: Option<u32>,
    /// Ansel camera block: position, quaternion, fov (degrees).
    pub camera: Option<[f32; 8]>,
}

impl Probe {
    pub fn map_running(&self) -> bool {
        self.map_state == Some(MAP_RUNNING)
    }

    /// The reading with the camera block rounded to centimetres / 0.01 (change logging).
    fn rounded(&self) -> Probe {
        Probe {
            camera: self.camera.map(|c| c.map(|v| (v * 100.0).round() / 100.0)),
            ..self.clone()
        }
    }

    pub fn same_as(&self, other: &Probe) -> bool {
        self.rounded() == other.rounded()
    }

    /// Equal in every field but the camera block (change logging: state changes are logged
    /// at once, camera moves at most once per second).
    pub fn same_except_camera(&self, other: &Probe) -> bool {
        Probe {
            camera: None,
            ..self.clone()
        } == Probe {
            camera: None,
            ..other.clone()
        }
    }

    pub fn describe(&self) -> String {
        if !self.base {
            return "no executable image".into();
        }
        let cam = self.camera.map_or("unreadable".into(), |c| {
            format!(
                "pos ({:.2} {:.2} {:.2}) quat ({:.3} {:.3} {:.3} {:.3}) fov {:.2}",
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            )
        });
        format!(
            "session flag {}, game {:#x} (current {:#x}{}), manager {}, state {} ({}) depth {}, map state {}, camera {cam}",
            self.session_flag
                .map_or("unreadable".into(), |f| f.to_string()),
            self.game,
            self.current_game,
            if self.current_game == self.game {
                ", same"
            } else {
                ", differs"
            },
            if self.manager_ok {
                "ok"
            } else {
                "no back-pointer"
            },
            self.top,
            state_name(self.top),
            self.depth,
            self.map_state
                .map_or("unreadable".into(), |s| s.to_string())
        )
    }
}

/// Reads all values (about ten checked reads).
pub fn read() -> Probe {
    let Some(base) = mem::image_base() else {
        return Probe::default();
    };
    let mut flag = [0u8; 1];
    let session_flag = mem::read_into(base + SESSION_FLAG_RVA, &mut flag).then_some(flag[0]);
    let game = mem::u64_at(base + GAME_PTR_RVA).unwrap_or(0);
    let current_game = mem::u64_at(base + CURRENT_GAME_RVA).unwrap_or(0);
    let camera = mem::f32s::<8>(base + CAMERA_RVA);
    let mut p = Probe {
        base: true,
        session_flag,
        game,
        current_game,
        camera,
        ..Probe::default()
    };
    if game == 0 {
        return p;
    }
    let manager = game + MANAGER;
    p.manager_ok = mem::u64_at(manager + MANAGER_GAME) == Some(game);
    p.map_state = mem::u32_at(game + MAP_STATE);
    if p.manager_ok {
        let depth = mem::u32_at(manager + STACK_COUNT).map_or(0, |c| c & 0x00FF_FFFF);
        p.depth = depth;
        if (1..=4096).contains(&depth) {
            p.top = mem::u64_at(manager + STACK_PTR)
                .filter(|&s| s != 0)
                .and_then(|s| mem::u32_at(s + 4 * u64::from(depth - 1)))
                .unwrap_or(0);
        }
    }
    p
}
