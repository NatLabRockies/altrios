use super::*;
use crate::consist::locomotive::powertrain::ElectricMachine;
use crate::imports::*;
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct for modeling electric drivetrain.  This includes power electronics, motor, axle ...
/// everything involved in converting high voltage electrical power to force exerted by the wheel on the track.  
pub struct ElectricDrivetrain {
    #[serde(default)]
    /// struct for tracking current state
    pub state: ElectricDrivetrainState,
    /// Shaft output power fraction array at which efficiencies are evaluated.
    pub pwr_out_frac_interp: Vec<f64>,

    /// Efficiency array corresponding to [Self::pwr_out_frac_interp] and [Self::pwr_in_frac_interp]
    pub eta_interp: Vec<f64>,
    /// Electrical input power fraction array at which efficiencies are evaluated.
    /// Calculated during runtime if not provided.
    #[serde(skip)]
    pub pwr_in_frac_interp: Vec<f64>,
    /// ElectricDrivetrain maximum output power assuming that positive and negative tractive powers have same magnitude
    pub pwr_out_max: si::Power,
    // TODO: add `mass` here
    /// Time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state] haha
    #[serde(default)]
    pub history: ElectricDrivetrainStateHistoryVec,
}

#[pyo3_api]
impl ElectricDrivetrain {
    #[new]
    #[pyo3(signature = (pwr_out_frac_interp, eta_interp, pwr_out_max_watts, save_interval=None))]
    fn __new__(
        pwr_out_frac_interp: Vec<f64>,
        eta_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(
            pwr_out_frac_interp,
            eta_interp,
            pwr_out_max_watts,
            save_interval,
        )
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }

    #[getter("eta_max")]
    fn get_eta_max_py(&self) -> f64 {
        self.get_eta_max()
    }

    #[setter("__eta_max")]
    fn set_eta_max_py(&mut self, eta_max: f64) -> anyhow::Result<()> {
        Ok(self.set_eta_max(eta_max).map_err(PyValueError::new_err)?)
    }

    #[getter("eta_min")]
    fn get_eta_min_py(&self) -> f64 {
        self.get_eta_min()
    }

    #[getter("eta_range")]
    fn get_eta_range_py(&self) -> f64 {
        self.get_eta_range()
    }

    #[setter("__eta_range")]
    fn set_eta_range_py(&mut self, eta_range: f64) -> anyhow::Result<()> {
        Ok(self
            .set_eta_range(eta_range)
            .map_err(PyValueError::new_err)?)
    }
}

impl ElectricDrivetrain {
    pub fn new(
        pwr_out_frac_interp: Vec<f64>,
        eta_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            eta_interp.len() == pwr_out_frac_interp.len(),
            format!(
                "{}\nedrv eta_interp and pwr_out_frac_interp must be the same length",
                eta_interp.len() == pwr_out_frac_interp.len()
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x >= 0.0),
            format!(
                "{}\nedrv pwr_out_frac_interp must be non-negative",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x >= 0.0))
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x <= 1.0),
            format!(
                "{}\nedrv pwr_out_frac_interp must be less than or equal to 1.0",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x <= 1.0))
            )
        );

        let history = ElectricDrivetrainStateHistoryVec::new();
        let pwr_out_max_watts = uc::W * pwr_out_max_watts;
        let state = ElectricDrivetrainState::default();

        let mut edrv = ElectricDrivetrain {
            state,
            pwr_out_frac_interp,
            eta_interp,
            pwr_in_frac_interp: Vec::new(),
            pwr_out_max: pwr_out_max_watts,
            save_interval,
            history,
        };
        edrv.set_pwr_in_frac_interp()?;
        Ok(edrv)
    }

    pub fn set_pwr_in_frac_interp(&mut self) -> anyhow::Result<()> {
        // make sure vector has been created
        self.pwr_in_frac_interp = self
            .pwr_out_frac_interp
            .iter()
            .zip(self.eta_interp.iter())
            .map(|(x, y)| x / y)
            .collect();
        // verify monotonicity
        ensure!(
            self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1]),
            format!(
                "{}\nedrv pwr_in_frac_interp ({:?}) must be monotonically increasing",
                format_dbg!(self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1])),
                self.pwr_in_frac_interp
            )
        );
        Ok(())
    }

    pub fn set_cur_pwr_regen_max(&mut self, pwr_max_regen_in: si::Power) -> anyhow::Result<()> {
        if self.pwr_in_frac_interp.is_empty() {
            self.set_pwr_in_frac_interp()?;
        }
        let eta = uc::R
            * interp1d(
                &(pwr_max_regen_in / self.pwr_out_max)
                    .get::<si::ratio>()
                    .abs(),
                &self.pwr_out_frac_interp,
                &self.eta_interp,
                false,
            )?;
        self.state.pwr_mech_regen_max.update(
            (pwr_max_regen_in * eta).min(self.pwr_out_max),
            || format_dbg!(),
        )?;
        ensure!(*self.state.pwr_mech_regen_max.get_fresh(|| format_dbg!())? >= si::Power::ZERO);
        Ok(())
    }

    /// Set `pwr_in_req` required to achieve desired `pwr_out_req` with time step size `dt`.
    pub fn set_pwr_in_req(&mut self, pwr_out_req: si::Power, _dt: si::Time) -> anyhow::Result<()> {
        ensure!(
            almost_le_uom(&pwr_out_req.abs(), &self.pwr_out_max, None),
            format!(
                "{}\nedrv required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                format_dbg!(pwr_out_req.abs() <= self.pwr_out_max),
                pwr_out_req.abs().get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()
            ),
        );

        ensure!(
            almost_le_uom(
                &pwr_out_req,
                self.state.pwr_mech_out_max.get_fresh(|| format_dbg!())?,
                Some(1e-5)
            ),
            format!(
                "{}\nedrv required power ({:.6} MW) exceeds dynamic max power ({:.6} MW)",
                format_dbg!(
                    pwr_out_req.abs()
                        <= *self.state.pwr_mech_out_max.get_fresh(|| format_dbg!())?
                ),
                pwr_out_req.get::<si::megawatt>(),
                self.state
                    .pwr_mech_out_max
                    .get_fresh(|| format_dbg!())?
                    .get::<si::megawatt>()
            ),
        );

        self.state
            .pwr_out_req
            .update(pwr_out_req, || format_dbg!())?;

        self.state.eta.update(
            uc::R
                * interp1d(
                    &(pwr_out_req / self.pwr_out_max).get::<si::ratio>().abs(),
                    &self.pwr_out_frac_interp,
                    &self.eta_interp,
                    false,
                )
                .with_context(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        ensure!(
            *self.state.eta.get_fresh(|| format_dbg!())? >= 0.0 * uc::R
                || *self.state.eta.get_fresh(|| format_dbg!())? <= 1.0 * uc::R,
            format!(
                "{}\nedrv eta ({}) must be between 0 and 1",
                format_dbg!(
                    *self.state.eta.get_fresh(|| format_dbg!())? >= 0.0 * uc::R
                        || *self.state.eta.get_fresh(|| format_dbg!())? <= 1.0 * uc::R
                ),
                self.state
                    .eta
                    .get_fresh(|| format_dbg!())?
                    .get::<si::ratio>()
            )
        );

        // `pwr_mech_prop_out` is `pwr_out_req` unless `pwr_out_req` is more negative than `pwr_mech_regen_max`,
        // in which case, excess is handled by `pwr_mech_dyn_brake`
        self.state.pwr_mech_prop_out.update(
            pwr_out_req.max(-*self.state.pwr_mech_regen_max.get_fresh(|| format_dbg!())?),
            || format_dbg!(),
        )?;

        self.state.pwr_mech_dyn_brake.update(
            -(pwr_out_req - *self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?),
            || format_dbg!(),
        )?;
        ensure!(
            *self.state.pwr_mech_dyn_brake.get_fresh(|| format_dbg!())? >= si::Power::ZERO,
            "Mech Dynamic Brake Power cannot be below 0.0"
        );

        // if pwr_out_req is negative, need to multiply by eta
        self.state.pwr_elec_prop_in.update(
            if pwr_out_req > si::Power::ZERO {
                *self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?
                    / *self.state.eta.get_fresh(|| format_dbg!())?
            } else {
                *self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?
                    * *self.state.eta.get_fresh(|| format_dbg!())?
            },
            || format_dbg!(),
        )?;

        self.state.pwr_elec_dyn_brake.update(
            *self.state.pwr_mech_dyn_brake.get_fresh(|| format_dbg!())?
                * *self.state.eta.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;

        // loss does not account for dynamic braking
        self.state.pwr_loss.update(
            (*self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?
                - *self.state.pwr_elec_prop_in.get_fresh(|| format_dbg!())?)
            .abs(),
            || format_dbg!(),
        )?;

        Ok(())
    }

    impl_get_set_eta_max_min!();
    impl_get_set_eta_range!();
}

// failed attempt at making path to default platform independent
// const EDRV_DEFAULT_PATH_STR: &'static str = include_str!(concat!(
//     env!("CARGO_MANIFEST_DIR"),
//     "/src/consist/locomotive/powertrain/electric_drivetrain.default.yaml"
// ));

impl Init for ElectricDrivetrain {
    fn init(&mut self) -> Result<(), Error> {
        self.state.init()?;
        Ok(())
    }
}
impl SerdeAPI for ElectricDrivetrain {}

impl Default for ElectricDrivetrain {
    fn default() -> Self {
        // let file_contents = include_str!(EDRV_DEFAULT_PATH_STR);
        let file_contents = include_str!("electric_drivetrain.default.yaml");
        let mut edrv = Self::from_yaml(file_contents, false).unwrap();
        edrv.init().unwrap();
        edrv
    }
}

impl ElectricMachine for ElectricDrivetrain {
    /// Set current max possible output power, `pwr_mech_out_max`,
    /// given `pwr_in_max` from upstream component.
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()> {
        ensure!(pwr_aux.is_none(), format_dbg!(pwr_aux.is_none()));
        if self.pwr_in_frac_interp.is_empty() {
            self.set_pwr_in_frac_interp()?;
        }
        let eta = uc::R
            * interp1d(
                &(pwr_in_max / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_in_frac_interp,
                &self.eta_interp,
                false,
            )?;

        self.state.pwr_mech_out_max.update(
            self.pwr_out_max.min(pwr_in_max * eta).max(si::Power::ZERO),
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Set current power out max ramp rate, `pwr_rate_out_max` given `pwr_rate_in_max`
    /// from upstream component.  
    fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate) -> anyhow::Result<()> {
        self.state.pwr_rate_out_max.update(
            if *self.state.eta.get_stale(|| format_dbg!())? > si::Ratio::ZERO {
                pwr_rate_in_max * *self.state.eta.get_stale(|| format_dbg!())?
            } else {
                pwr_rate_in_max * uc::R * 1.0
            },
            || format_dbg!(),
        )?;
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
pub struct ElectricDrivetrainState {
    /// index
    pub i: TrackedState<usize>,
    /// Component efficiency based on current power demand.
    pub eta: TrackedState<si::Ratio>,
    // Component limits
    /// Maximum possible positive traction power.
    pub pwr_mech_out_max: TrackedState<si::Power>,
    /// Maximum possible regeneration power going to ReversibleEnergyStorage.
    pub pwr_mech_regen_max: TrackedState<si::Power>,
    /// max ramp-up rate
    pub pwr_rate_out_max: TrackedState<si::PowerRate>,

    // Current values
    /// Raw power requirement from boundary conditions
    pub pwr_out_req: TrackedState<si::Power>,
    /// Electrical power to propulsion from ReversibleEnergyStorage and Generator.
    /// negative value indicates regenerative braking
    pub pwr_elec_prop_in: TrackedState<si::Power>,
    /// Mechanical power to propulsion, corrected by efficiency, from ReversibleEnergyStorage and Generator.
    /// Negative value indicates regenerative braking.
    pub pwr_mech_prop_out: TrackedState<si::Power>,
    /// Mechanical power from dynamic braking.  Positive value indicates braking; this should be zero otherwise.
    pub pwr_mech_dyn_brake: TrackedState<si::Power>,
    /// Electrical power from dynamic braking, dissipated as heat.
    pub pwr_elec_dyn_brake: TrackedState<si::Power>,
    /// Power lost in regeneratively converting mechanical power to power that can be absorbed by the battery.
    pub pwr_loss: TrackedState<si::Power>,

    // Cumulative energy values
    /// cumulative mech energy in from fc
    pub energy_elec_prop_in: TrackedState<si::Energy>,
    /// cumulative elec energy out
    pub energy_mech_prop_out: TrackedState<si::Energy>,
    /// cumulative energy has lost due to imperfect efficiency
    /// Mechanical energy from dynamic braking.
    pub energy_mech_dyn_brake: TrackedState<si::Energy>,
    /// Electrical energy from dynamic braking, dissipated as heat.
    pub energy_elec_dyn_brake: TrackedState<si::Energy>,
    /// Cumulative energy lost in regeneratively converting mechanical power to power that can be absorbed by the battery.
    pub energy_loss: TrackedState<si::Energy>,
}

#[pyo3_api]
impl ElectricDrivetrainState {}

impl Init for ElectricDrivetrainState {}
impl SerdeAPI for ElectricDrivetrainState {}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_edrv() -> ElectricDrivetrain {
        ElectricDrivetrain::new(vec![0.0, 1.0], vec![0.9, 0.8], 8e6, None).unwrap()
    }

    #[test]
    fn test_that_i_increments() {
        let mut edrv = test_edrv();
        edrv.check_and_reset(|| format_dbg!()).unwrap();
        edrv.step(|| format_dbg!()).unwrap();
        assert_eq!(1, *edrv.state.i.get_fresh(|| format_dbg!()).unwrap());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_that_history_has_len_1() {
        let mut edrv: ElectricDrivetrain = ElectricDrivetrain::default();
        edrv.save_interval = Some(1);
        assert!(edrv.history.is_empty());
        edrv.save_state(|| format_dbg!()).unwrap();
        assert_eq!(1, edrv.history.len());
    }

    #[test]
    fn test_that_history_has_len_0() {
        let mut edrv: ElectricDrivetrain = ElectricDrivetrain::default();
        assert!(edrv.history.is_empty());
        edrv.save_state(|| format_dbg!()).unwrap();
        assert!(edrv.history.is_empty());
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut res = test_edrv();
        let eta_max = 0.9;
        let eta_min = 0.8;
        let eta_range = 0.1;

        eta_test_body!(res, eta_max, eta_min, eta_range);
    }
}
