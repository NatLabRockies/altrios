use super::*;

// FUTURE: think about how to incorporate life modeling for Fuel Cells and other tech

const TOL: f64 = 1e-3;

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct for modeling Fuel Converter (e.g. engine, fuel cell.)
pub struct FuelConverter {
    #[serde(default)]
    /// struct for tracking current state
    pub state: FuelConverterState,
    /// FuelConverter mass
    #[serde(default)]
    mass: Option<si::Mass>,
    /// FuelConverter specific power
    specific_pwr: Option<si::SpecificPower>,
    /// max rated brake output power
    pub pwr_out_max: si::Power,
    /// starting/baseline transient power limit
    #[serde(default)]
    pub pwr_out_max_init: si::Power,
    // TODO: consider a ramp down rate, which may be needed for fuel cells
    /// lag time for ramp up
    pub pwr_ramp_lag: si::Time,
    /// Fuel converter brake power fraction array at which efficiencies are evaluated.
    /// This fuel converter efficiency model assumes that speed and load (or voltage and current) will
    /// always be controlled for operating at max possible efficiency for the power demand
    pub pwr_out_frac_interp: Vec<f64>,
    /// fuel converter efficiency array
    pub eta_interp: Vec<f64>,
    /// pwr at which peak efficiency occurs
    #[serde(skip)]
    pub(crate) pwr_for_peak_eff: si::Power,
    /// idle fuel power to overcome internal friction (not including aux load)
    pub pwr_idle_fuel: si::Power,
    /// Interpolator for derating dynamic engine peak power based on altitude
    /// and temperature. When interpolating, this returns fraction of normal
    /// peak power, e.g. a value of 1 means no derating and a value of 0 means
    /// the engine is completely disabled.
    pub elev_and_temp_derate: Option<Interp2DOwned<f64, strategy::Linear>>,
    /// time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: FuelConverterStateHistoryVec, // TODO: spec out fuel tank size and track kg of fuel
}

#[pyo3_api]
impl FuelConverter {
    // optional, custom, struct-specific pymethods
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

    #[getter("mass_kg")]
    fn get_mass_py(&self) -> anyhow::Result<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_pwr_kw_per_kg(&self) -> Option<f64> {
        self.specific_pwr
            .map(|sp| sp.get::<si::kilowatt_per_kilogram>())
    }

    #[pyo3(name = "set_default_elev_and_temp_derate")]
    fn set_default_elev_and_temp_derate_py(&mut self) {
        self.set_default_elev_and_temp_derate()
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl Default for FuelConverter {
    fn default() -> Self {
        let file_contents = include_str!("fuel_converter.default.yaml");
        let mut fc = Self::from_yaml(file_contents, false).unwrap();
        fc.init().unwrap();
        fc
    }
}

impl Init for FuelConverter {
    fn init(&mut self) -> Result<(), Error> {
        self.state.init()?;
        Ok(())
    }
}
impl SerdeAPI for FuelConverter {}

impl Mass for FuelConverter {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self.derived_mass().with_context(|| format_dbg!())?;
        if let (Some(derived_mass), Some(set_mass)) = (derived_mass, self.mass) {
            ensure!(
                utils::almost_eq_uom(&set_mass, &derived_mass, None),
                format!(
                    "{}",
                    format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                )
            );
        }
        Ok(self.mass)
    }

    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        let derived_mass = self.derived_mass().with_context(|| format_dbg!())?;
        if let (Some(derived_mass), Some(new_mass)) = (derived_mass, new_mass) {
            if derived_mass != new_mass {
                match side_effect {
                    MassSideEffect::Extensive => {
                        self.pwr_out_max = self.specific_pwr.ok_or_else(|| {
                            anyhow!(
                                "{}\nExpected `self.specific_pwr` to be `Some`.",
                                format_dbg!()
                            )
                        })? * new_mass;
                    }
                    MassSideEffect::Intensive => {
                        self.specific_pwr = Some(self.pwr_out_max / new_mass);
                    }
                    MassSideEffect::None => {
                        self.specific_pwr = None;
                    }
                }
            }
        } else if new_mass.is_none() {
            self.specific_pwr = None;
        }
        self.mass = new_mass;
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(self
            .specific_pwr
            .map(|specific_pwr| self.pwr_out_max / specific_pwr))
    }

    fn expunge_mass_fields(&mut self) {
        self.mass = None;
        self.specific_pwr = None;
    }
}

// non-py methods
impl FuelConverter {
    /// Get fuel converter max power output given time step, dt
    pub fn set_cur_pwr_out_max(
        &mut self,
        elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        ensure!(
            dt > si::Time::ZERO,
            format!(
                "{}\n dt must always be greater than 0.0",
                format_dbg!(dt > si::Time::ZERO)
            )
        );

        let pwr_max_derated = match (&mut self.elev_and_temp_derate, elev_and_temp) {
            (Some(elev_and_temp_derate), Some(elev_and_temp)) => {
                let derate_factor = elev_and_temp_derate.interpolate(&[
                    elev_and_temp.0.get::<si::meter>(),
                    elev_and_temp.1.get::<si::degree_celsius>(),
                ])?;
                ensure!((0.0..=1.0).contains(&derate_factor), format_dbg!());
                derate_factor * self.pwr_out_max
            }
            (None, Some(_)) => bail!(
                "{}\nExpected (self.elev_and_temp_derate, elev_and_temp) to both be Some or None",
                format_dbg!()
            ),
            (Some(_), None) => bail!(
                "{}\nExpected (self.elev_and_temp_derate, elev_and_temp) to both be Some or None",
                format_dbg!()
            ),
            (None, None) => self.pwr_out_max,
        };

        self.pwr_out_max_init = self.pwr_out_max_init.max(self.pwr_out_max / 10.);

        self.state.pwr_out_max.update(
            (*self.state.pwr_shaft.get_stale(|| format_dbg!())?
                + self.pwr_out_max / self.pwr_ramp_lag * dt)
                .min(self.pwr_out_max)
                .min(pwr_max_derated)
                .max(self.pwr_out_max_init),
            || format_dbg!(),
        )?;
        #[cfg(test)]
        {
            ensure!(self.state.pwr_out_max.get_fresh(|| format_dbg!())? <= &self.pwr_out_max)
        }
        Ok(())
    }

    /// Solve for fuel usage for a given required fuel converter power output
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: bool,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        if engine_on {
            self.state.time_on.increment(dt, || format_dbg!())?;
        } else {
            self.state
                .time_on
                .update(si::Time::ZERO, || format_dbg!())?;
        }

        if assert_limits {
            ensure!(
                utils::almost_le_uom(&pwr_out_req, &self.pwr_out_max, Some(TOL)),
                format!(
                "{}\nfc pwr_out_req ({:.6} MW) must be less than or equal to static pwr_out_max ({:.6} MW)",
                format_dbg!(utils::almost_le_uom(&pwr_out_req, &self.pwr_out_max, Some(TOL))),
                pwr_out_req.get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()),
            );
            ensure!(
                utils::almost_le_uom(&pwr_out_req, self.state.pwr_out_max.get_fresh(|| format_dbg!())?, Some(TOL)),
                format!("{}\nfc pwr_out_req ({:.6} kW) must be less than or equal to current transient pwr_out_max ({:.6} kW)",
                format_dbg!(utils::almost_le_uom(&pwr_out_req, self.state.pwr_out_max.get_fresh(|| format_dbg!())?, Some(TOL))),
                pwr_out_req.get::<si::kilowatt>(),
                self.state.pwr_out_max.get_fresh(|| format_dbg!())?.get::<si::kilowatt>()),
            );
        }
        ensure!(
            pwr_out_req >= si::Power::ZERO,
            format!(
                "{}\nfc pwr_out_req ({:.6} MW) must be greater than or equal to zero",
                format_dbg!(pwr_out_req >= si::Power::ZERO),
                pwr_out_req.get::<si::megawatt>()
            )
        );

        // if the engine is not on, `pwr_out_req` should be 0.0
        ensure!(
            engine_on || (pwr_out_req == si::Power::ZERO),
            format!(
                "{}\nEngine is off but `pwr_out_req` is non-zero\n`pwr_out_req`: {} kW",
                format_dbg!(engine_on || (pwr_out_req == si::Power::ZERO)),
                pwr_out_req.get::<si::kilowatt>(),
            )
        );

        self.state.pwr_shaft.update(pwr_out_req, || format_dbg!())?;
        self.state.eta.update(
            uc::R
                * interp1d(
                    &(pwr_out_req / self.pwr_out_max).get::<si::ratio>(),
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
                "{}\nfc eta ({}) must be between 0 and 1",
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

        self.state.engine_on.update(engine_on, || format_dbg!())?;
        self.state.pwr_idle_fuel.update(
            if *self.state.engine_on.get_fresh(|| format_dbg!())? {
                self.pwr_idle_fuel
            } else {
                si::Power::ZERO
            },
            || format_dbg!(),
        )?;
        // if the engine is not on, `pwr_out_req` should be 0.0
        ensure!(
            *self.state.engine_on.get_fresh(|| format_dbg!())? || (pwr_out_req == si::Power::ZERO),
            format!(
                "{}\nEngine is off but pwr_out_req is non-zero",
                format_dbg!(
                    *self.state.engine_on.get_fresh(|| format_dbg!())?
                        || pwr_out_req == si::Power::ZERO
                )
            )
        );
        self.state.pwr_fuel.update(
            pwr_out_req / *self.state.eta.get_fresh(|| format_dbg!())? + self.pwr_idle_fuel,
            || format_dbg!(),
        )?;
        self.state.pwr_loss.update(
            *self.state.pwr_fuel.get_fresh(|| format_dbg!())?
                - *self.state.pwr_shaft.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;

        ensure!(
            self.state
                .pwr_loss
                .get_fresh(|| format_dbg!())?
                .get::<si::watt>()
                >= 0.0,
            format!(
                "{}\n`pwr_loss` must be non-negative",
                format_dbg!(
                    self.state
                        .pwr_loss
                        .get_fresh(|| format_dbg!())?
                        .get::<si::watt>()
                        >= 0.0
                )
            )
        );
        Ok(())
    }

    impl_get_set_eta_max_min!();
    impl_get_set_eta_range!();

    fn set_default_elev_and_temp_derate(&mut self) {
        self.elev_and_temp_derate = Some(
            Interp2D::new(
                array![0.0, 1_000.0, 2_000.0], // elevation in meters
                array![0.0, 35.0, 45.0, 50.0], // temperature in degrees Celsius
                // Fraction of static peak power
                array![
                    [1.0, 1.0, 0.95, 0.8],
                    [0.95, 0.95, 0.9025, 0.76],
                    [0.8, 0.8, 0.76, 0.64],
                ],
                strategy::Linear,
                Extrapolate::Clamp,
            )
            .unwrap(),
        );
    }
}

#[serde_api]
#[derive(
    Clone, Debug, Deserialize, Serialize, PartialEq, HistoryVec, StateMethods, SetCumulative,
)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct FuelConverterState {
    /// iteration counter
    pub i: TrackedState<usize>,
    /// max power fc can produce at current time
    pub pwr_out_max: TrackedState<si::Power>,
    /// efficiency evaluated at current demand
    pub eta: TrackedState<si::Ratio>,
    /// instantaneous shaft power going to generator
    pub pwr_shaft: TrackedState<si::Power>,
    /// instantaneous fuel power flow
    pub pwr_fuel: TrackedState<si::Power>,
    /// loss power, including idle
    pub pwr_loss: TrackedState<si::Power>,
    /// idle fuel flow rate power
    pub pwr_idle_fuel: TrackedState<si::Power>,
    /// cumulative shaft energy fc has provided to generator
    pub energy_shaft: TrackedState<si::Energy>,
    /// cumulative fuel energy fc has consumed
    pub energy_fuel: TrackedState<si::Energy>,
    /// cumulative energy fc has lost due to imperfect efficiency
    pub energy_loss: TrackedState<si::Energy>,
    /// cumulative fuel energy fc has lost due to idle
    pub energy_idle_fuel: TrackedState<si::Energy>,
    /// If true, engine is on, and if false, off (no idle)
    pub engine_on: TrackedState<bool>,
    /// elapsed time since engine was turned on
    pub time_on: TrackedState<si::Time>,
}

#[pyo3_api]
impl FuelConverterState {}

impl Init for FuelConverterState {}
impl SerdeAPI for FuelConverterState {}
impl Default for FuelConverterState {
    fn default() -> Self {
        Self {
            i: Default::default(),
            pwr_out_max: Default::default(),
            eta: Default::default(),
            pwr_fuel: Default::default(),
            pwr_shaft: Default::default(),
            pwr_loss: Default::default(),
            pwr_idle_fuel: Default::default(),
            energy_fuel: Default::default(),
            energy_shaft: Default::default(),
            energy_loss: Default::default(),
            energy_idle_fuel: Default::default(),
            engine_on: TrackedState::new(true),
            time_on: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_fc() -> FuelConverter {
        FuelConverter {
            pwr_out_max: 8_000e3 * uc::W,
            pwr_out_max_init: 800e3 * uc::W,
            pwr_ramp_lag: 25.0 * uc::S,
            pwr_out_frac_interp: Vec::linspace(0.01, 1.0, 5),
            eta_interp: vec![0.2, 0.32, 0.35, 0.4, 0.38],
            pwr_idle_fuel: 500e3 * uc::W,
            save_interval: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_that_fuel_grtr_than_shaft_energy() {
        let mut fc = test_fc();
        //performing check and reset on entire state for the new engine we created

        fc.state.check_and_reset(|| format_dbg!()).unwrap();
        fc.state
            .pwr_out_max
            .update(uc::MW * 2., || format_dbg!())
            .unwrap();

        fc.solve_energy_consumption(uc::W * 2_000e3, uc::S * 1.0, true, true)
            .unwrap();
        assert!(
            fc.state.pwr_fuel.get_fresh(|| format_dbg!()).unwrap()
                > fc.state.pwr_shaft.get_fresh(|| format_dbg!()).unwrap()
        );
    }

    #[test]
    fn test_default() {
        let _fc = FuelConverter::default();
    }

    #[test]
    fn test_that_max_power_includes_rate() {
        let mut fc = test_fc();
        fc.check_and_reset(|| format_dbg!()).unwrap();
        fc.set_cur_pwr_out_max(None, uc::S * 1.0).unwrap();
        let pwr_out_max = *fc.state.pwr_out_max.get_fresh(|| format_dbg!()).unwrap();
        assert!(pwr_out_max < fc.pwr_out_max);
    }

    #[test]
    fn test_that_i_increments() {
        let mut fc = test_fc();
        fc.check_and_reset(|| format_dbg!()).unwrap();
        fc.step(|| format_dbg!()).unwrap();
        assert_eq!(1, *fc.state.i.get_fresh(|| format_dbg!()).unwrap());
    }

    #[test]
    fn test_that_fuel_is_monotonic() {
        let mut fc = test_fc();
        fc.check_and_reset(|| format_dbg!()).unwrap();
        fc.step(|| format_dbg!()).unwrap();

        fc.state
            .pwr_out_max
            .update(uc::MW * 2.0, || format_dbg!())
            .unwrap();
        fc.save_interval = Some(1);
        fc.solve_energy_consumption(uc::W * 2_000e3, uc::S * 1.0, true, true)
            .unwrap();
        fc.set_cumulative(uc::S * 1.0, || format_dbg!()).unwrap();
        fc.save_state(|| format_dbg!()).unwrap();
        fc.check_and_reset(|| format_dbg!()).unwrap();
        fc.step(|| format_dbg!()).unwrap();
        fc.state
            .pwr_out_max
            .update(uc::MW * 2.0, || format_dbg!())
            .unwrap();
        fc.solve_energy_consumption(uc::W * 2_000e3, uc::S * 1.0, true, true)
            .unwrap();
        fc.set_cumulative(uc::S * 1.0, || format_dbg!()).unwrap();
        fc.save_state(|| format_dbg!()).unwrap();

        assert!(
            fc.history.energy_fuel[1]
                .get_fresh(|| format_dbg!())
                .unwrap()
                > fc.history.energy_fuel[0]
                    .get_fresh(|| format_dbg!())
                    .unwrap()
        );
        assert!(
            fc.history.energy_loss[1]
                .get_fresh(|| format_dbg!())
                .unwrap()
                > fc.history.energy_loss[0]
                    .get_fresh(|| format_dbg!())
                    .unwrap()
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_that_history_has_len_1() {
        let mut fc: FuelConverter = FuelConverter::default();
        fc.save_interval = Some(1);
        assert!(fc.history.is_empty());
        fc.save_state(|| format_dbg!()).unwrap();
        assert_eq!(1, fc.history.len());
    }

    #[test]
    fn test_that_history_has_len_0() {
        let mut fc: FuelConverter = FuelConverter::default();
        assert!(fc.history.is_empty());
        fc.save_state(|| format_dbg!()).unwrap();
        assert!(fc.history.is_empty());
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut fc = test_fc();
        let eta_max = 0.4;
        let eta_min = 0.2;
        let eta_range = 0.2;

        eta_test_body!(fc, eta_max, eta_min, eta_range);
    }
}
