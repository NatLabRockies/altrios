use crate::imports::*;
use crate::train::TrainState;

#[serde_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Basic {
    ratio: si::Ratio,
}

#[pyo3_api]
impl Basic {}

impl Init for Basic {}
impl SerdeAPI for Basic {}

impl Basic {
    pub fn new(ratio: si::Ratio) -> Self {
        Self { ratio }
    }
    pub fn calc_res(&mut self, state: &TrainState) -> anyhow::Result<si::Force> {
        Ok(self.ratio * *state.weight_static.get_unchecked(|| format_dbg!())?)
    }
}
