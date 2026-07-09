use super::*;
use crate::consist::locomotive::powertrain::ElectricMachine;

#[cfg(feature = "pyo3")]
use crate::pyo3::*;

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct for modeling generator/alternator.
pub struct Generator {
    /// Generator mass
    #[serde(default)]
    mass: Option<si::Mass>,
    /// Generator specific power
    specific_pwr: Option<si::SpecificPower>,
    // no macro-generated setter because derived parameters would get messed up
    /// Generator brake power fraction array at which efficiencies are evaluated.
    pub pwr_out_frac_interp: Vec<f64>,
    // no macro-generated setter because derived parameters would get messed up
    /// Generator efficiency array correpsonding to [Self::pwr_out_frac_interp]
    /// and [Self::pwr_in_frac_interp].
    pub eta_interp: Vec<f64>,
    /// Mechanical input power fraction array at which efficiencies are
    /// evaluated.  This vec is calculated during initialization. Each element
    /// represents the current input power divided by peak output power.
    #[serde(skip)]
    pub pwr_in_frac_interp: Vec<f64>,
    /// Generator max power out
    pub pwr_out_max: si::Power,
    /// Time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    #[serde(default)]
    /// struct for tracking current state
    pub state: GeneratorState,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: GeneratorStateHistoryVec,
}

#[pyo3_api]
impl Generator {
    /// Initialize a fuel converter object
    #[new]
    #[pyo3(signature = (
        pwr_out_frac_interp,
        eta_interp,
        pwr_out_max_watts,
        save_interval=None,
    ))]
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
    fn get_mass_py(&mut self) -> anyhow::Result<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_pwr_kw_per_kg(&self) -> Option<f64> {
        self.specific_pwr
            .map(|sp| sp.get::<si::kilowatt_per_kilogram>())
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl Init for Generator {
    fn init(&mut self) -> Result<(), Error> {
        let _ = self
            .mass()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.state.init()?;
        Ok(())
    }
}
impl SerdeAPI for Generator {}

impl Mass for Generator {
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
                        self.pwr_out_max = self.specific_pwr.with_context(|| {
                            format!(
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
        self.specific_pwr = None;
        self.mass = None;
    }
}

impl Generator {
    pub fn new(
        pwr_out_frac_interp: Vec<f64>,
        eta_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            eta_interp.len() == pwr_out_frac_interp.len(),
            format!(
                "{}\ngen eta_interp and pwr_out_frac_interp must be the same length",
                format_dbg!(eta_interp.len() == pwr_out_frac_interp.len())
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x >= 0.0),
            format!(
                "{}\ngen pwr_out_frac_interp must be non-negative",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x >= 0.0))
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x <= 1.0),
            format!(
                "{}\ngen pwr_out_frac_interp must be less than or equal to 1.0",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x <= 1.0))
            )
        );

        let history = GeneratorStateHistoryVec::new();
        let pwr_out_max = uc::W * pwr_out_max_watts;
        let state = GeneratorState::default();

        let mut gen = Generator {
            state,
            pwr_out_frac_interp,
            eta_interp,
            pwr_in_frac_interp: Vec::new(),
            pwr_out_max,
            save_interval,
            history,
            ..Default::default()
        };
        gen.set_pwr_in_frac_interp()?;
        Ok(gen)
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
                "{}\ngen pwr_in_frac_interp ({:?}) must be monotonically increasing",
                format_dbg!(self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1])),
                self.pwr_in_frac_interp
            )
        );
        Ok(())
    }

    pub fn set_pwr_in_req(
        &mut self,
        pwr_prop_req: si::Power,
        pwr_aux: si::Power,
        engine_on: bool,
        _dt: si::Time,
    ) -> anyhow::Result<()> {
        // generator cannot regen
        ensure!(
            pwr_prop_req >= si::Power::ZERO,
            format!(
                "{}\ngen propulsion power is negative",
                format_dbg!(pwr_prop_req >= si::Power::ZERO)
            )
        );
        ensure!(
            pwr_prop_req + pwr_aux <= self.pwr_out_max,
            format!(
                "{}\ngen required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                format_dbg!(pwr_prop_req + pwr_aux <= self.pwr_out_max),
                (pwr_prop_req + pwr_aux).get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()
            ),
        );

        ensure!(
            utils::almost_le_uom(
                &(pwr_prop_req + pwr_aux),
                self.state.pwr_elec_out_max.get_fresh(|| format_dbg!())?,
                None
            ),
            format!(
                "{}\ngen required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                format_dbg!(pwr_prop_req + pwr_aux <= self.pwr_out_max),
                (pwr_prop_req + pwr_aux).get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()
            ),
        );

        // if the engine is not on, `pwr_out_req` should be 0.0
        ensure!(
            engine_on || (pwr_prop_req + pwr_aux == si::Power::ZERO),
            format!(
                "{}\nEngine is off but `pwr_prop_req + pwr_aux` is non-zero\n`pwr_out_req`: {} kW
{} kW
{} kW",
                format_dbg!(engine_on || (pwr_prop_req + pwr_aux == si::Power::ZERO))
                    .replace("\"", ""),
                format_dbg!((pwr_prop_req + pwr_aux).get::<si::kilowatt>()).replace("\"", ""),
                format_dbg!(pwr_prop_req.get::<si::kilowatt>()).replace("\"", ""),
                format_dbg!(pwr_aux.get::<si::kilowatt>()).replace("\"", ""),
            )
        );

        self.state.eta.update(
            uc::R
                * interp1d(
                    &((pwr_prop_req + pwr_aux) / self.pwr_out_max)
                        .get::<si::ratio>()
                        .abs(),
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
                "{}\ngen eta ({}) must be between 0 and 1",
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

        self.state
            .pwr_elec_prop_out
            .update(pwr_prop_req, || format_dbg!())?;

        self.state.pwr_elec_aux.update(pwr_aux, || format_dbg!())?;

        self.state.pwr_mech_in.update(
            (*self.state.pwr_elec_prop_out.get_fresh(|| format_dbg!())?
                + *self.state.pwr_elec_aux.get_fresh(|| format_dbg!())?)
                / *self.state.eta.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        ensure!(
            *self.state.pwr_mech_in.get_fresh(|| format_dbg!())? >= si::Power::ZERO,
            format!(
                "{}\nfc can only produce positive power",
                format_dbg!(
                    *self.state.pwr_mech_in.get_fresh(|| format_dbg!())? >= si::Power::ZERO
                )
            ),
        );

        self.state.pwr_loss.update(
            *self.state.pwr_mech_in.get_fresh(|| format_dbg!())?
                - (*self.state.pwr_elec_prop_out.get_fresh(|| format_dbg!())?
                    + *self.state.pwr_elec_aux.get_fresh(|| format_dbg!())?),
            || format_dbg!(),
        )?;

        Ok(())
    }

    impl_get_set_eta_max_min!();
    impl_get_set_eta_range!();
}

impl Default for Generator {
    fn default() -> Self {
        let file_contents = include_str!("generator.default.yaml");
        let mut gen = Self::from_yaml(file_contents, false).unwrap();
        gen.init().unwrap();
        gen
    }
}

impl ElectricMachine for Generator {
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()> {
        if self.pwr_in_frac_interp.is_empty() {
            // make sure vector has been populated
            self.set_pwr_in_frac_interp()?;
        }
        let eta = uc::R
            * interp1d(
                &(pwr_in_max / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_in_frac_interp,
                &self.eta_interp,
                false,
            )?;
        ensure!(
            eta <= uc::R && eta >= uc::R * 0.0,
            format!("Invalid `eta`: {}", eta.get::<si::ratio>())
        );
        self.state
            .pwr_elec_out_max
            .update((pwr_in_max * eta).min(self.pwr_out_max), || format_dbg!())?;
        ensure!(
            *self.state.pwr_elec_out_max.get_fresh(|| format_dbg!())? >= si::Power::ZERO,
            format_dbg!(self
                .state
                .pwr_elec_out_max
                .get_fresh(|| format_dbg!())?
                .get::<si::kilowatt>())
        );
        if let Some(pwr_aux) = pwr_aux {
            ensure!(
                pwr_aux >= si::Power::ZERO,
                format_dbg!(pwr_aux.get::<si::kilowatt>())
            )
        };
        self.state.pwr_elec_prop_out_max.update(
            *self.state.pwr_elec_out_max.get_fresh(|| format_dbg!())? - pwr_aux.unwrap(),
            || format_dbg!(),
        )?;

        Ok(())
    }

    fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate) -> anyhow::Result<()> {
        self.state.pwr_rate_out_max.update(
            pwr_rate_in_max
                * if self
                    .state
                    .eta
                    .get_stale(|| format_dbg!())?
                    .get::<si::ratio>()
                    > 0.0
                {
                    *self.state.eta.get_stale(|| format_dbg!())?
                } else {
                    uc::R * 1.0
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
pub struct GeneratorState {
    /// iteration counter
    pub i: TrackedState<usize>,
    /// efficiency evaluated at current power demand
    pub eta: TrackedState<si::Ratio>,
    /// max possible power output for propulsion
    pub pwr_elec_prop_out_max: TrackedState<si::Power>,
    /// max possible power output total
    pub pwr_elec_out_max: TrackedState<si::Power>,
    /// max possible power output rate
    pub pwr_rate_out_max: TrackedState<si::PowerRate>,
    /// mechanical power input
    pub pwr_mech_in: TrackedState<si::Power>,
    /// electrical power output to propulsion
    pub pwr_elec_prop_out: TrackedState<si::Power>,
    /// electrical power output to aux loads
    pub pwr_elec_aux: TrackedState<si::Power>,
    /// power lost due to conversion inefficiency
    pub pwr_loss: TrackedState<si::Power>,
    /// cumulative mech energy in from fc
    pub energy_mech_in: TrackedState<si::Energy>,
    /// cumulative elec energy out to propulsion
    pub energy_elec_prop_out: TrackedState<si::Energy>,
    /// cumulative elec energy to aux loads
    pub energy_elec_aux: TrackedState<si::Energy>,
    /// cumulative energy has lost due to imperfect efficiency
    pub energy_loss: TrackedState<si::Energy>,
}

#[pyo3_api]
impl GeneratorState {}

impl Init for GeneratorState {}
impl SerdeAPI for GeneratorState {}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_gen() -> Generator {
        Generator::new(vec![0.0, 1.0], vec![0.9, 0.8], 8e6, None).unwrap()
    }

    #[test]
    fn test_that_i_increments() {
        let mut gen = test_gen();
        gen.check_and_reset(|| format_dbg!()).unwrap();
        gen.step(|| format_dbg!()).unwrap();
        assert!(1 == *gen.state.i.get_fresh(|| format_dbg!()).unwrap());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_that_history_has_len_1() {
        let mut gen: Generator = Generator::default();
        gen.save_interval = Some(1);
        assert!(gen.history.is_empty());
        gen.save_state(|| format_dbg!()).unwrap();
        assert!(1 == gen.history.len());
    }

    #[test]
    fn test_that_history_has_len_0() {
        let mut gen: Generator = Generator::default();
        assert!(gen.history.is_empty());
        gen.save_state(|| format_dbg!()).unwrap();
        assert!(gen.history.is_empty());
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut res = test_gen();
        let eta_max = 0.9;
        let eta_min = 0.8;
        let eta_range = 0.1;

        eta_test_body!(res, eta_max, eta_min, eta_range);
    }
}
