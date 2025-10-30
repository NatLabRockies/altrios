use super::cat_power::*;
use super::elev::*;
use super::heading::*;
use super::link_idx::*;
use super::speed::*;

use crate::imports::*;

#[serde_api]
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
/// An arbitrary unit of single track that does not include turnouts
///
/// # Note:
/// This struct is to be deprecated and superseded by [super::link_impl::Link],
/// which includes either a train-type-independent `speed_set` or a
/// train-type-dependent `speed_sets` HashMap, superseding [LinkOld]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct LinkOld {
    pub elevs: Vec<Elev>,
    #[serde(default)]
    pub headings: Vec<Heading>,

    pub speed_sets: Vec<OldSpeedSet>,
    #[serde(default)]
    pub cat_power_limits: Vec<CatPowerLimit>,
    pub length: si::Length,

    /// see [EstTime::idx_next]
    pub idx_next: LinkIdx,
    /// see [EstTime::idx_next_alt]  
    /// if it does not exist, it should be `LinkIdx{idx: 0}`
    pub idx_next_alt: LinkIdx,
    /// see [EstTime::idx_prev]
    pub idx_prev: LinkIdx,
    /// see [EstTime::idx_prev_alt]  
    /// if it does not exist, it should be `LinkIdx{idx: 0}`
    pub idx_prev_alt: LinkIdx,
    /// Index of current link
    pub idx_curr: LinkIdx,
    /// Index of adjacent link in reverse direction
    pub idx_flip: LinkIdx,
    /// Optional OpenStreetMap ID -- not used in simulation
    pub osm_id: Option<String>,
    #[serde(default)]
    pub link_idxs_lockout: Vec<LinkIdx>,
}

#[pyo3_api]
impl LinkOld {}

impl Init for LinkOld {}
impl SerdeAPI for LinkOld {}
