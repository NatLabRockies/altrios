use crate::imports::*;
use crate::track::PathResCoeff;
use crate::train::TrainState;

/// Calculates and returns resistance force
///
/// # Arguments
/// - `res_coeff`: resistance force per train weight
/// - `state`: current [TrainState]
fn calc_res_val(res_coeff: si::Ratio, state: &TrainState) -> anyhow::Result<si::Force> {
    Ok(res_coeff * *state.weight_static.get_fresh(|| format_dbg!())?)
}

#[serde_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Point {
    /// index within corresponding [PathResCoeff]
    idx: usize,
}

#[pyo3_api]
impl Point {}

impl Init for Point {}
impl SerdeAPI for Point {}

impl Point {
    pub fn new(path_res_coeffs: &[PathResCoeff], state: &TrainState) -> anyhow::Result<Self> {
        Ok(Self {
            idx: path_res_coeffs.calc_idx(
                *state.offset.get_fresh(|| format_dbg!())?
                    - *state.length.get_fresh(|| format_dbg!())? * 0.5,
                0,
                &Dir::Fwd,
            )?,
        })
    }

    pub fn calc_res(
        &mut self,
        path_res_coeffs: &[PathResCoeff],
        state: &TrainState,
        dir: &Dir,
    ) -> anyhow::Result<si::Force> {
        self.idx = path_res_coeffs.calc_idx(
            *state.offset.get_fresh(|| format_dbg!())?
                - *state.length.get_fresh(|| format_dbg!())? * 0.5,
            self.idx,
            dir,
        )?;
        calc_res_val(path_res_coeffs[self.idx].res_coeff, state)
    }

    pub fn res_coeff_front(&self, path_res_coeffs: &[PathResCoeff]) -> si::Ratio {
        path_res_coeffs[self.idx].res_coeff
    }

    pub fn res_net_front(
        &self,
        path_res_coeffs: &[PathResCoeff],
        state: &TrainState,
    ) -> anyhow::Result<si::Length> {
        Ok(path_res_coeffs[self.idx].calc_res_val(*state.offset.get_fresh(|| format_dbg!())?))
    }

    /// Returns index of current element containing front of train within `PathTPC`
    pub fn path_tpc_idx_front(&self) -> usize {
        self.idx
    }

    pub fn fix_cache(&mut self, idx_sub: usize) {
        self.idx -= idx_sub;
    }
}

#[ext(CalcResStrap)]
impl [PathResCoeff] {
    fn calc_res_strap(
        &self,
        idx_front: usize,
        idx_back: usize,
        state: &TrainState,
    ) -> anyhow::Result<si::Ratio> {
        debug_assert!(*state.length.get_unchecked(|| format_dbg!())? > si::Length::ZERO);
        Ok(
            (self[idx_front].calc_res_val(*state.offset.get_unchecked(|| format_dbg!())?)
                - self[idx_back].calc_res_val(*state.offset_back.get_unchecked(|| format_dbg!())?))
                / *state.length.get_unchecked(|| format_dbg!())?,
        )
    }
}

#[serde_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Strap {
    /// index of front of train within corresponding [PathResCoeff]
    idx_front: usize,
    /// index of back of train within corresponding [PathResCoeff]
    idx_back: usize,
}

#[pyo3_api]
impl Strap {}

impl Init for Strap {}
impl SerdeAPI for Strap {}

impl Strap {
    pub fn new(vals: &[PathResCoeff], state: &TrainState) -> anyhow::Result<Self> {
        if vals.len() <= 1 {
            Ok(Self {
                idx_front: 0,
                idx_back: 0,
            })
        } else {
            let idx_back = vals.calc_idx(
                *state.offset.get_fresh(|| format_dbg!())?
                    - *state.length.get_fresh(|| format_dbg!())?,
                0,
                &Dir::Fwd,
            )?;
            Ok(Self {
                idx_back,
                idx_front: vals.calc_idx(
                    *state.offset.get_fresh(|| format_dbg!())?,
                    idx_back,
                    &Dir::Fwd,
                )?,
            })
        }
    }

    pub fn calc_res(
        &mut self,
        path_res_coeffs: &[PathResCoeff],
        state: &TrainState,
        dir: &Dir,
    ) -> anyhow::Result<si::Force> {
        match dir {
            Dir::Fwd => {
                self.idx_front = path_res_coeffs.calc_idx(
                    *state.offset.get_unchecked(|| format_dbg!())?,
                    self.idx_front,
                    dir,
                )?;
            }
            Dir::Bwd => {
                self.idx_back = path_res_coeffs.calc_idx(
                    *state.offset_back.get_unchecked(|| format_dbg!())?,
                    self.idx_back,
                    dir,
                )?;
            }
            Dir::Unk => {
                self.idx_front = path_res_coeffs.calc_idx(
                    *state.offset.get_unchecked(|| format_dbg!())?,
                    self.idx_front,
                    dir,
                )?;
                self.idx_back = path_res_coeffs.calc_idx(
                    *state.offset_back.get_unchecked(|| format_dbg!())?,
                    self.idx_back,
                    dir,
                )?;
            }
        }

        let res_coeff: si::Ratio = if self.idx_front == self.idx_back {
            path_res_coeffs[self.idx_front].res_coeff
        } else {
            match dir {
                Dir::Fwd => {
                    self.idx_back = path_res_coeffs.calc_idx(
                        *state.offset_back.get_unchecked(|| format_dbg!())?,
                        self.idx_back,
                        dir,
                    )?;
                }
                Dir::Bwd => {
                    self.idx_front = path_res_coeffs.calc_idx(
                        *state.offset.get_unchecked(|| format_dbg!())?,
                        self.idx_front,
                        dir,
                    )?;
                }
                _ => {}
            }
            path_res_coeffs.calc_res_strap(self.idx_front, self.idx_back, state)?
        };

        let res_val: si::Force = calc_res_val(res_coeff, state)?;
        Ok(res_val)
    }

    pub fn res_coeff_front(&self, path_res_coeffs: &[PathResCoeff]) -> si::Ratio {
        path_res_coeffs[self.idx_front].res_coeff
    }

    pub fn res_coeff_back(&self, path_res_coeffs: &[PathResCoeff]) -> si::Ratio {
        path_res_coeffs[self.idx_back].res_coeff
    }

    pub fn res_net_front(
        &self,
        path_res_coeffs: &[PathResCoeff],
        state: &TrainState,
    ) -> anyhow::Result<si::Length> {
        Ok(path_res_coeffs[self.idx_front]
            .calc_res_val(*state.offset.get_unchecked(|| format_dbg!())?))
    }

    pub fn res_net_back(
        &self,
        path_res_coeffs: &[PathResCoeff],
        state: &TrainState,
    ) -> anyhow::Result<si::Length> {
        Ok(path_res_coeffs[self.idx_back]
            .calc_res_val(*state.offset_back.get_fresh(|| format_dbg!())?))
    }

    /// Returns index of current element containing front of train within `PathTPC`
    pub fn path_tpc_idx_front(&self) -> usize {
        self.idx_front
    }

    pub fn fix_cache(&mut self, idx_sub: usize) {
        self.idx_back -= idx_sub;
        self.idx_front -= idx_sub;
    }
}
