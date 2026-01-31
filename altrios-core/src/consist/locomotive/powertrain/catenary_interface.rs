use super::*;
use crate::imports::*;
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct for modeling onboard components between (and including) catenary to the electric drivetrain of a locomotive. 
pub struct CatenaryInterface {
    #[serde(default)]
    /// struct for tracking current state
    pub state: CatenaryInterfaceState,    
    /// Maximum output power assuming that Cat2Interface and Interface2Cat have same max values
    pub pwr_out_max: si::Power,
    /// Time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state] haha
    #[serde(default)]
    pub history: CatenaryInterfaceStateHistoryVec,
}

#[pyo3_api]
impl CatenaryInterface {
    #[new]
    #[pyo3(signature = (pwr_out_max_watts, save_interval=None))]
    fn __new__(
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(
            pwr_out_max_watts,
            save_interval,
        )
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl CatenaryInterface {
    /// Catenary interface efficiency constant
    const ETA: f64 = 0.99;

    pub fn new(
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        let history = CatenaryInterfaceStateHistoryVec::new();
        let pwr_out_max_watts = uc::W * pwr_out_max_watts;
        let state = CatenaryInterfaceState::default();

        let ci = CatenaryInterface {
            state,
            pwr_out_max: pwr_out_max_watts,
            save_interval,
            history,
        };
        Ok(ci)
    }
    pub fn set_cat_power_limit(
        &mut self,
        new_limit: Option<si::Power>,
    ) -> anyhow::Result<()> {
        self.state.pwr_elec_in_max.update(
            new_limit.unwrap_or(si::Power::ZERO),
            || format_dbg!(),
        )?;
        Ok(())
    }
    /// TODO mbruchon not really a required amount, more like a max amount that could be needed
    /// Set `pwr_in_req` required to achieve desired `pwr_out_req` with time step size `dt`.
    pub fn set_pwr_in_req(&mut self, pwr_out_req_max: si::Power, _dt: si::Time) -> anyhow::Result<()> {
        self.state
            .pwr_out_req_max
            .update(pwr_out_req_max, || format_dbg!())?;

        let eta = uc::R * Self::ETA;

        self.state.pwr_elec_out.update(
            (*self.state.pwr_elec_in_max.get_fresh(|| format_dbg!())? 
                * eta)
                .min(pwr_out_req_max),
            || format_dbg!(),
        )?;

        self.state.pwr_elec_in.update(
            *self.state.pwr_elec_out.get_fresh(|| format_dbg!())? 
                / eta,
            || format_dbg!(),
        )?;

        self.state.pwr_loss.update(
            (*self.state.pwr_elec_out.get_fresh(|| format_dbg!())?
                - *self.state.pwr_elec_in.get_fresh(|| format_dbg!())?)
            .abs(),
            || format_dbg!(),
        )?;

        Ok(())
    }

}

impl Init for CatenaryInterface {
    fn init(&mut self) -> Result<(), Error> {
        self.state.init()?;
        Ok(())
    }
}
impl SerdeAPI for CatenaryInterface {}

impl Default for CatenaryInterface {
    fn default() -> Self {
        let file_contents = include_str!("catenary_interface.default.yaml");
        let mut cat_int = Self::from_yaml(file_contents, false).unwrap();
        cat_int.init().unwrap();
        cat_int
    }
}

impl ElectricMachine for CatenaryInterface {
    /// Set current max possible output power, `pwr_mech_out_max`,
    /// given `pwr_in_max` from upstream component.
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()> {
        ensure!(pwr_aux.is_none(), format_dbg!(pwr_aux.is_none()));
        let eta = uc::R * Self::ETA;
        self.state.pwr_elec_out_max.update(
            self.pwr_out_max.min(pwr_in_max * eta).max(si::Power::ZERO),
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Set current power out max ramp rate, `pwr_rate_out_max` given `pwr_rate_in_max`
    /// from upstream component.  
    fn set_pwr_rate_out_max(&mut self, _: si::PowerRate) -> anyhow::Result<()> {
        Ok(())
    }
}

#[serde_api]
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Serialize,
    PartialEq,
    HistoryVec,
    StateMethods,
    SetCumulative,
)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct CatenaryInterfaceState {
    /// index
    pub i: TrackedState<usize>,

    // Current values
    /// Maximum potential necessary power requirement from boundary conditions
    pub pwr_out_req_max: TrackedState<si::Power>,
    /// Max electrical power out of interface.
    pub pwr_elec_out_max: TrackedState<si::Power>,
    /// Max electrical power into interface from catenary.
    pub pwr_elec_in_max: TrackedState<si::Power>,
    /// Electrical power into interface from catenary.
    /// negative value indicates V2Catenary.
    pub pwr_elec_in: TrackedState<si::Power>,
    /// Electrical power out of interface into next module (e-drive).
    pub pwr_elec_out: TrackedState<si::Power>,
    /// Power lost in conversion steps, etc.
    pub pwr_loss: TrackedState<si::Power>,

    // Cumulative energy values
    /// cumulative electrical energy in from catenary
    /// negative value indicates V2Catenary.
    pub energy_elec_in: TrackedState<si::Energy>,
    /// cumulative electrical energy in from catenary
    /// negative value indicates V2Catenary.
    pub energy_elec_out: TrackedState<si::Energy>,
    /// Cumulative energy lost in power conversion.
    pub energy_loss: TrackedState<si::Energy>,
}

#[pyo3_api]
impl CatenaryInterfaceState {}

impl Init for CatenaryInterfaceState {}
impl SerdeAPI for CatenaryInterfaceState {}