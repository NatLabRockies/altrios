use super::*;

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods, SetCumulative)]
/// Legacy veryion of [ReversibleEnergyStorage]
///
/// Struct for modeling technology-naive Reversible Energy Storage (e.g. battery, flywheel).
pub struct ReversibleEnergyStorageLegacy {
    /// struct for tracking current state
    #[serde(default)]
    pub state: ReversibleEnergyStorageState,
    /// ReversibleEnergyStorage mass
    #[serde(default)]
    pub(super) mass: Option<si::Mass>,
    /// ReversibleEnergyStorage volume, used as a sanity check
    #[serde(default)]
    pub(super) volume: Option<si::Volume>,
    /// ReversibleEnergyStorage specific energy
    pub(super) specific_energy: Option<si::SpecificEnergy>,
    /// ReversibleEnergyStorage energy density (note that pressure has the same units as energy density)
    pub energy_density: Option<si::Pressure>,
    /// efficiency map grid values - indexed temp; soc; c_rate;
    pub eta_interp_grid: [Vec<f64>; 3],

    /// Values of efficiencies at grid points:
    /// - temperature
    /// - soc
    /// - c_rate
    pub eta_interp_values: Vec<Vec<Vec<f64>>>,
    /// Max output (and input) power battery can produce (accept)
    pub pwr_out_max: si::Power,

    /// Total energy capacity of battery of full discharge SOC of 0.0 and 1.0
    pub energy_capacity: si::Energy,

    /// Hard limit on minimum SOC, e.g. 0.05
    pub min_soc: si::Ratio,
    /// Hard limit on maximum SOC, e.g. 0.95
    pub max_soc: si::Ratio,
    /// SOC at which negative/charge power begins to ramp down.
    /// Should always be slightly below [Self::max_soc].
    pub soc_hi_ramp_start: Option<si::Ratio>,
    /// SOC at which positive/discharge power begins to ramp down.
    /// Should always be slightly above [Self::min_soc].
    pub soc_lo_ramp_start: Option<si::Ratio>,
    /// Time step interval at which history is saved
    pub save_interval: Option<usize>,
    #[serde(default)]
    /// Custom vector of [Self::state]
    pub history: ReversibleEnergyStorageStateHistoryVec,
}
impl Init for ReversibleEnergyStorageLegacy {}
impl SerdeAPI for ReversibleEnergyStorageLegacy {}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq, HistoryVec)]
// component limits
/// ReversibleEnergyStorage state variables
pub struct ReversibleEnergyStorageStateLegacy {
    // limits
    // TODO: create separate binning for cat power and maximum catenary power capability
    pub pwr_cat_max: TrackedState<si::Power>,
    /// max output power for propulsion during positive traction
    pub pwr_prop_out_max: TrackedState<si::Power>,
    /// max regen power for propulsion during negative traction
    pub pwr_regen_out_max: TrackedState<si::Power>,
    /// max discharge power total
    pub pwr_disch_max: TrackedState<si::Power>,
    /// max charge power on the output side
    pub pwr_charge_max: TrackedState<si::Power>,

    /// simulation step
    pub i: TrackedState<usize>,

    /// state of charge (SOC)
    pub soc: TrackedState<si::Ratio>,
    /// Chemical <-> Electrical conversion efficiency based on current power demand
    pub eta: TrackedState<si::Ratio>,
    /// State of Health (SOH)
    pub soh: TrackedState<f64>,

    // TODO: add `pwr_out_neg_electrical` and `pwr_out_pos_electrical` and corresponding energies

    // powers
    /// total electrical power; positive is discharging
    pub pwr_out_electrical: TrackedState<si::Power>,
    /// electrical power going to propulsion
    pub pwr_out_propulsion: TrackedState<si::Power>,
    /// electrical power going to aux loads
    pub pwr_aux: TrackedState<si::Power>,
    /// power dissipated as loss
    pub pwr_loss: TrackedState<si::Power>,
    /// chemical power; positive is discharging
    pub pwr_out_chemical: TrackedState<si::Power>,

    // cumulative energies
    /// cumulative total electrical energy; positive is discharging
    pub energy_out_electrical: TrackedState<si::Energy>,
    /// cumulative electrical energy going to propulsion
    pub energy_out_propulsion: TrackedState<si::Energy>,
    /// cumulative electrical energy going to aux loads
    pub energy_aux: TrackedState<si::Energy>,
    /// cumulative energy dissipated as loss
    pub energy_loss: TrackedState<si::Energy>,
    /// cumulative chemical energy; positive is discharging
    pub energy_out_chemical: TrackedState<si::Energy>,

    /// dynamically updated max SOC limit
    pub max_soc: TrackedState<si::Ratio>,
    /// dynamically updated SOC at which negative/charge power begins to ramp down.
    pub soc_hi_ramp_start: TrackedState<si::Ratio>,
    /// dynamically updated min SOC limit
    pub min_soc: TrackedState<si::Ratio>,
    /// dynamically updated SOC at which positive/discharge power begins to ramp down.
    pub soc_lo_ramp_start: TrackedState<si::Ratio>,

    /// component temperature
    pub temperature_celsius: TrackedState<f64>,
}

impl Init for ReversibleEnergyStorageStateLegacy {}
impl SerdeAPI for ReversibleEnergyStorageStateLegacy {}

impl Default for ReversibleEnergyStorageStateLegacy {
    fn default() -> Self {
        Self {
            i: Default::default(),
            soc: TrackedState::new(uc::R * 0.95),
            soh: TrackedState::new(1.0),
            eta: Default::default(),
            pwr_prop_out_max: Default::default(),
            pwr_regen_out_max: Default::default(),
            pwr_disch_max: Default::default(),
            pwr_charge_max: Default::default(),
            pwr_cat_max: Default::default(),
            pwr_out_electrical: Default::default(),
            pwr_out_propulsion: Default::default(),
            pwr_aux: Default::default(),
            pwr_out_chemical: Default::default(),
            pwr_loss: Default::default(),
            energy_out_electrical: Default::default(),
            energy_out_propulsion: Default::default(),
            energy_aux: Default::default(),
            energy_out_chemical: Default::default(),
            energy_loss: Default::default(),
            max_soc: TrackedState::new(uc::R * 1.0),
            soc_hi_ramp_start: TrackedState::new(uc::R * 1.0),
            min_soc: TrackedState::new(si::Ratio::ZERO),
            soc_lo_ramp_start: TrackedState::new(si::Ratio::ZERO),
            temperature_celsius: TrackedState::new(45.0),
        }
    }
}
