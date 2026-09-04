//! The pause menu: a small model of the user-facing render settings, the
//! navigation state, and how it lays itself out into a character grid the
//! overlay kernel blits. Reached with Escape; arrow keys move and adjust.
//!
//! The menu owns only *choices*. Turning a choice into GPU state (rebuilding a
//! pipeline for a new scene or resolution, re-grading for a colour mode) is
//! main.rs's job -- the menu just reports what is selected.

use crate::font::glyph_index;

/// Top-level game mode. Flythrough (the free-flight explorer) is the only one
/// built; the planned modes -- photo, time-trial, the deep-dive -- slot in
/// here, and main.rs branches on `Menu::mode`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode { Flythrough }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scene { Schwarzschild, Kerr, Binary }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Color { Cinematic, TrueColor, Arcade }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Upscale { Nearest, Smooth, MetalFX }

/// Named lenses: focal length in mm (0 marks the fisheye, handled specially).
pub const LENSES: &[(&str, f64)] = &[
    ("10MM", 10.0), ("18MM", 18.0), ("24MM", 24.0),
    ("35MM", 35.0), ("50MM", 50.0), ("FISHEYE", 0.0),
];

/// Internal render heights; width follows the window's 16:9. Higher is sharper
/// and slower. The label is what the row shows.
pub const RES_LEVELS: &[(&str, usize)] = &[
    ("240P", 240), ("360P", 360), ("480P", 480), ("720P", 720), ("1080P", 1080),
];

/// Accretion-disc fidelity, cheapest first. The value is the volumetric gas
/// march stride in units of M -- SMALLER is finer and slower. The 0.0 sentinel
/// is not a stride at all: it selects the thin-plane disc (volumetric gas off,
/// a single equatorial crossing), the cheapest possible disc. main.rs maps this
/// to `volume: None`, which flips the shader's VOL constant and turns the
/// thin-plane path on.
pub const GAS_LEVELS: &[(&str, f32)] = &[
    ("THIN DISC", 0.0), ("LOW", 0.64), ("MEDIUM", 0.32), ("HIGH", 0.16),
];

/// Which row is highlighted. Order is the on-screen order.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Row { Mode, Scene, Color, Resolution, Gas, Upscale, Lens }
const ROWS: [Row; 7] = [
    Row::Mode, Row::Scene, Row::Color, Row::Resolution, Row::Gas, Row::Upscale, Row::Lens,
];

pub struct Menu {
    pub open: bool,
    pub cursor: usize,
    pub mode: Mode,
    pub scene: Scene,
    pub color: Color,
    pub res: usize,     // index into RES_LEVELS
    pub gas: usize,     // index into GAS_LEVELS
    pub upscale: Upscale,
    pub lens: usize,    // index into LENSES
    /// Bumped whenever a choice changes, so main.rs knows to re-apply.
    pub revision: u64,
}

impl Default for Menu {
    fn default() -> Self {
        Self {
            open: false, cursor: 0,
            mode: Mode::Flythrough,
            scene: Scene::Kerr,
            color: Color::TrueColor,
            res: 1,               // 360p
            gas: 2,               // medium (stride 0.32)
            upscale: Upscale::MetalFX,
            lens: 2,              // 24mm
            revision: 0,
        }
    }
}

impl Menu {
    pub fn toggle(&mut self) { self.open = !self.open; }

    pub fn move_cursor(&mut self, delta: i32) {
        let n = ROWS.len() as i32;
        self.cursor = (((self.cursor as i32 + delta) % n + n) % n) as usize;
    }

    /// Adjust the highlighted row by `dir` (-1 / +1), wrapping. Returns true if
    /// something changed (main.rs re-applies on true).
    pub fn adjust(&mut self, dir: i32) -> bool {
        let before = self.revision;
        match ROWS[self.cursor] {
            Row::Mode => {
                // Only Flythrough exists today; cycling is a no-op until the
                // other modes land. No revision bump -- nothing to re-apply.
            }
            Row::Scene => {
                self.scene = cycle3([Scene::Schwarzschild, Scene::Kerr, Scene::Binary],
                                    self.scene, dir);
                self.revision += 1;
            }
            Row::Color => {
                self.color = cycle3([Color::Cinematic, Color::TrueColor, Color::Arcade],
                                    self.color, dir);
                self.revision += 1;
            }
            Row::Resolution => {
                self.res = wrap(self.res, RES_LEVELS.len(), dir);
                self.revision += 1;
            }
            Row::Gas => {
                self.gas = wrap(self.gas, GAS_LEVELS.len(), dir);
                self.revision += 1;
            }
            Row::Upscale => {
                self.upscale = cycle3([Upscale::Nearest, Upscale::Smooth, Upscale::MetalFX],
                                      self.upscale, dir);
                self.revision += 1;
            }
            Row::Lens => {
                self.lens = wrap(self.lens, LENSES.len(), dir);
                self.revision += 1;
            }
        }
        self.revision != before
    }

    pub fn mode_label(&self) -> &'static str {
        match self.mode { Mode::Flythrough => "FLYTHROUGH" }
    }
    pub fn scene_label(&self) -> &'static str {
        match self.scene {
            Scene::Schwarzschild => "SCHWARZSCHILD",
            Scene::Kerr => "KERR",
            Scene::Binary => "BINARY",
        }
    }
    pub fn color_label(&self) -> &'static str {
        match self.color {
            Color::Cinematic => "CINEMATIC", Color::TrueColor => "TRUE COLOR",
            Color::Arcade => "ARCADE",
        }
    }
    pub fn upscale_label(&self) -> &'static str {
        match self.upscale {
            Upscale::Nearest => "NEAREST", Upscale::Smooth => "SMOOTH",
            Upscale::MetalFX => "METALFX",
        }
    }

    /// The menu as a `COLS x ROWS_N` grid of glyph slots plus a per-cell
    /// attribute (0 normal, 1 dim label, 2 highlighted value, 3 title). The
    /// overlay kernel reads exactly this.
    pub fn render_grid(&self) -> TextGrid {
        let mut g = TextGrid::new();
        g.put(0, "SPACETIME", 3);
        g.put_kv(2, "MODE", self.mode_label(), self.cursor == 0);
        g.put_kv(3, "SCENE", self.scene_label(), self.cursor == 1);
        g.put_kv(4, "COLOR", self.color_label(), self.cursor == 2);
        g.put_kv(5, "RESOLUTION", RES_LEVELS[self.res].0, self.cursor == 3);
        g.put_kv(6, "GAS", GAS_LEVELS[self.gas].0, self.cursor == 4);
        g.put_kv(7, "UPSCALE", self.upscale_label(), self.cursor == 5);
        g.put_kv(8, "LENS", LENSES[self.lens].0, self.cursor == 6);
        g.put(9, "ARROWS MOVE/ADJUST   ESC CLOSE", 1);
        g
    }
}

fn wrap(i: usize, n: usize, dir: i32) -> usize {
    (((i as i32 + dir) % n as i32 + n as i32) % n as i32) as usize
}
fn cycle3<T: Copy + PartialEq>(opts: [T; 3], cur: T, dir: i32) -> T {
    let i = opts.iter().position(|&o| o == cur).unwrap_or(0);
    opts[wrap(i, 3, dir)]
}

pub const GRID_COLS: usize = 34;
pub const GRID_ROWS: usize = 10;
/// Column where the value half of a "LABEL   VALUE" row begins.
const VALUE_COL: usize = 15;

pub struct TextGrid {
    /// Glyph slot per cell, row-major.
    pub cells: Vec<u8>,
    /// Attribute per cell (see `render_grid`).
    pub attr: Vec<u8>,
    pub cols: usize,
    pub rows: usize,
}

impl TextGrid {
    fn new() -> Self {
        Self {
            cells: vec![0u8; GRID_COLS * GRID_ROWS],
            attr: vec![0u8; GRID_COLS * GRID_ROWS],
            cols: GRID_COLS, rows: GRID_ROWS,
        }
    }
    fn write_at(&mut self, row: usize, col: usize, text: &str, attr: u8) {
        for (k, ch) in text.chars().enumerate() {
            let c = col + k;
            if row >= self.rows || c >= self.cols { break; }
            self.cells[row * self.cols + c] = glyph_index(ch) as u8;
            self.attr[row * self.cols + c] = attr;
        }
    }
    fn put(&mut self, row: usize, text: &str, attr: u8) { self.write_at(row, 1, text, attr); }
    /// A "LABEL      VALUE" row; the whole row lights when selected, with a
    /// leading cursor mark and the value emphasised.
    fn put_kv(&mut self, row: usize, label: &str, value: &str, selected: bool) {
        if selected { self.write_at(row, 0, ">", 2); }
        self.write_at(row, 1, label, if selected { 2 } else { 1 });
        self.write_at(row, VALUE_COL, value, if selected { 2 } else { 0 });
    }
}
