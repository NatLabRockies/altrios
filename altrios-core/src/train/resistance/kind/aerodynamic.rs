use crate::imports::*;
use crate::train::TrainState;

// TODO implement method for elevation-dependent air density
#[serde_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Basic {
    cd_area: si::Area,
}

#[pyo3_api]
impl Basic {}

impl Init for Basic {}
impl SerdeAPI for Basic {}

impl Basic {
    pub fn new(cd_area: si::Area) -> Self {
        Self { cd_area }
    }

    /// Note that the factor of 0.5 typically used in
    /// [the drag equation](https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation)
    /// is traditionally lumped into a coefficient in the Davis equation and is treated
    /// the same here.
    pub fn calc_res(&mut self, state: &TrainState) -> anyhow::Result<si::Force> {
        Ok(self.cd_area
            * uc::rho_air()
            * *state.speed.get_unchecked(|| format_dbg!())?
            * *state.speed.get_unchecked(|| format_dbg!())?)
    }
}
