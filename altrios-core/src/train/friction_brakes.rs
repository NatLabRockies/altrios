use crate::imports::*;

#[serde_api]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
// brake propagation rate is ~800 ft/s (about speed of sound)
// ramp up duration is ~30 s
pub struct FricBrake {
    /// max static force achievable
    pub force_max: si::Force,
    /// time to go from zero to max braking force
    pub ramp_up_time: si::Time,
    /// ramp-up correction factor
    pub ramp_up_coeff: si::Ratio,
    // commented out.  This stuff needs refinement but
    // added complexity is probably worthwhile
    // /// time to go from max braking force to zero braking force
    // pub ramp_down_time: si::Time,
    // /// rate at which brakes can be recovered after full release
    // pub recharge_rate_pa_per_sec: f64,
    // TODO: add in whatever is needed to estimate aux load impact
    #[serde(default)]
    pub state: FricBrakeState,
    #[serde(default)]
    /// Custom vector of [Self::state]
    pub history: FricBrakeStateHistoryVec,
    pub save_interval: Option<usize>,
}

#[pyo3_api]
impl FricBrake {
    #[new]
    #[pyo3(signature = (
        force_max_newtons,
        ramp_up_time_seconds=None,
        ramp_up_coeff=None,
        state=None,
        save_interval=None,
    ))]
    fn __new__(
        force_max_newtons: f64,
        ramp_up_time_seconds: Option<f64>,
        ramp_up_coeff: Option<f64>,
        state: Option<FricBrakeState>,
        save_interval: Option<usize>,
    ) -> Self {
        Self::new(
            force_max_newtons * uc::N,
            ramp_up_time_seconds.map(|ruts| ruts * uc::S),
            ramp_up_coeff.map(|ruc| ruc * uc::R),
            state,
            save_interval,
        )
    }
}

impl Init for FricBrake {}
impl SerdeAPI for FricBrake {}

impl Default for FricBrake {
    fn default() -> Self {
        Self {
            force_max: 600_000.0 * uc::LBF,
            ramp_up_time: 0.0 * uc::S,
            ramp_up_coeff: 0.6 * uc::R,
            state: Default::default(),
            history: Default::default(),
            save_interval: Default::default(),
        }
    }
}

impl FricBrake {
    pub fn new(
        force_max: si::Force,
        ramp_up_time: Option<si::Time>,
        ramp_up_coeff: Option<si::Ratio>,
        // recharge_rate_pa_per_sec: f64,
        state: Option<FricBrakeState>,
        save_interval: Option<usize>,
    ) -> Self {
        let mut state = state.unwrap_or_default();
        state
            .force_max_curr
            .update_unchecked(force_max, || format_dbg!())
            .unwrap();
        let fric_brake_def: Self = Default::default();
        let ramp_up_time = ramp_up_time.unwrap_or(fric_brake_def.ramp_up_time);
        let ramp_up_coeff = ramp_up_coeff.unwrap_or(fric_brake_def.ramp_up_coeff);
        Self {
            force_max,
            ramp_up_time,
            ramp_up_coeff,
            // recharge_rate_pa_per_sec,
            state,
            history: Default::default(),
            save_interval,
        }
    }

    pub fn set_cur_force_max_out(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // maybe check parameter values here and propagate any errors
        self.state.force_max_curr.update(
            (*self.state.force.get_stale(|| format_dbg!())?
                + self.force_max / self.ramp_up_time * dt)
                .min(self.force_max),
            || format_dbg!(),
        )
    }
}

// TODO: figure out a way to make the braking reasonably polymorphic (e.g. for autonomous rail
// vehicles)
#[serde_api]
#[derive(
    Debug,
    Default,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    HistoryVec,
    StateMethods,
    SetCumulative,
)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct FricBrakeState {
    /// index counter
    pub i: TrackedState<usize>,
    // actual applied force of brakes
    pub force: TrackedState<si::Force>,
    // time-varying max force of brakes in current time step
    pub force_max_curr: TrackedState<si::Force>,
    // pressure: si::Pressure,
}

#[pyo3_api]
impl FricBrakeState {
    #[new]
    fn __new__() -> Self {
        Self::new()
    }
}

impl SerdeAPI for FricBrakeState {}
impl Init for FricBrakeState {}

impl FricBrakeState {
    /// TODO: this method needs to accept arguments
    pub fn new() -> Self {
        Self::default()
    }
}
