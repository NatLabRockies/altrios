use crate::imports::*;

#[serde_api]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct LinkIdxTime {
    time: si::Time,
    link_idx: LinkIdx,
}

#[pyo3_api]
impl LinkIdxTime {}
