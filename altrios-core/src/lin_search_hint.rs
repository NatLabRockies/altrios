use crate::si;

#[derive(PartialEq)]
pub enum Dir {
    Unk,
    Fwd,
    Bwd,
}

/// Has method that returns offset from start of PathTpc
pub trait GetOffset {
    /// Returns offset from start of PathTpc
    fn get_offset(&self) -> si::Length;
}

/// Contains method to calculate the index immediately before `offset` given the previous calculated
/// index, `idx`, and a direction `DirT`.
pub trait LinSearchHint {
    /// Calculate the index immediately before `offset` given the previous calculated index, `idx`,
    /// and a direction `DirT`.
    fn calc_idx(&self, offset: si::Length, idx: usize, dir: &Dir) -> anyhow::Result<usize>;
}
