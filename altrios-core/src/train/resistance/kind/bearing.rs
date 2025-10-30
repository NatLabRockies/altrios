use crate::imports::*;

#[serde_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Basic {
    force: si::Force,
}

impl Basic {}

impl Basic {
    pub fn new(force: si::Force) -> Self {
        Self { force }
    }
    pub fn calc_res(&mut self) -> si::Force {
        self.force
    }
}
