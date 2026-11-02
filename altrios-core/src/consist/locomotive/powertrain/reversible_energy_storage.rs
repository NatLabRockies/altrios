use super::*;

#[cfg(feature = "pyo3")]
use crate::pyo3::*;

pub(crate) mod res_legacy;

const TOL: f64 = 1e-3;

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods, SetCumulative)]
/// Struct for modeling technology-naive Reversible Energy Storage (e.g. battery, flywheel).
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct ReversibleEnergyStorage {
    /// struct for tracking current state
    #[serde(default)]
    pub state: ReversibleEnergyStorageState,
    /// ReversibleEnergyStorage mass
    #[serde(default)]
    mass: Option<si::Mass>,
    /// ReversibleEnergyStorage volume, used as a sanity check
    #[serde(default)]
    volume: Option<si::Volume>,
    /// ReversibleEnergyStorage specific energy
    specific_energy: Option<si::SpecificEnergy>,
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
    /// Time step interval at which history is saved
    pub save_interval: Option<usize>,
    #[serde(default)]
    /// Custom vector of [Self::state]
    pub history: ReversibleEnergyStorageStateHistoryVec,
}

#[pyo3_api]
impl ReversibleEnergyStorage {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        temperature_interp_grid,
        soc_interp_grid,
        c_rate_interp_grid,
        eta_interp_values,
        pwr_out_max_watts,
        energy_capacity_joules,
        min_soc,
        max_soc,
        initial_soc,
        initial_temperature_celcius,
        soc_hi_ramp_start=None,
        soc_lo_ramp_start=None,
        save_interval=None,
    ))]
    fn __new__(
        temperature_interp_grid: Vec<f64>,
        soc_interp_grid: Vec<f64>,
        c_rate_interp_grid: Vec<f64>,
        eta_interp_values: Vec<Vec<Vec<f64>>>,
        pwr_out_max_watts: f64,
        energy_capacity_joules: f64,
        min_soc: f64,
        max_soc: f64,
        initial_soc: f64,
        initial_temperature_celcius: f64,
        soc_hi_ramp_start: Option<f64>,
        soc_lo_ramp_start: Option<f64>,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(
            temperature_interp_grid,
            soc_interp_grid,
            c_rate_interp_grid,
            eta_interp_values,
            pwr_out_max_watts,
            energy_capacity_joules,
            min_soc,
            max_soc,
            initial_soc,
            initial_temperature_celcius,
            soc_hi_ramp_start,
            soc_lo_ramp_start,
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
        self.set_eta_range(eta_range)
    }

    // TODO: uncomment and fix
    // #[setter("__mass_kg")]
    // fn set_mass_py(&mut self, mass_kg: Option<f64>) -> anyhow::Result<()> {
    //     self.set_mass(mass_kg.map(|m| m * uc::KG))?;
    //     Ok(())
    // }

    #[getter("mass_kg")]
    fn get_mass_kg_py(&mut self) -> anyhow::Result<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_energy_kjoules_per_kg(&self) -> Option<f64> {
        self.specific_energy
            .map(|se| se.get::<si::kilojoule_per_kilogram>())
    }

    #[setter("__volume_m3")]
    fn update_volume_py(&mut self, volume_m3: Option<f64>) -> anyhow::Result<()> {
        let volume = volume_m3.map(|v| v * uc::M3);
        self.update_volume(volume)?;
        Ok(())
    }

    #[getter("volume_m3")]
    fn get_volume_py(&mut self) -> anyhow::Result<Option<f64>> {
        Ok(self.volume()?.map(|v| v.get::<si::cubic_meter>()))
    }

    #[getter]
    fn get_energy_density_kjoules_per_m3(&self) -> Option<f64> {
        self.specific_energy
            .map(|se| se.get::<si::kilojoule_per_kilogram>())
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl Default for ReversibleEnergyStorage {
    fn default() -> Self {
        let file_contents = include_str!("reversible_energy_storage.default.yaml");
        let mut res = Self::from_yaml(file_contents, false).unwrap();
        res.state
            .soc
            .update_unchecked(res.max_soc, || format_dbg!())
            .unwrap();
        res.init().unwrap();
        res
    }
}

impl Mass for ReversibleEnergyStorage {
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
                        self.energy_capacity = self.specific_energy.ok_or_else(|| {
                            anyhow!(
                                "{}\nExpected `self.specific_energy` to be `Some`.",
                                format_dbg!()
                            )
                        })? * new_mass;
                    }
                    MassSideEffect::Intensive => {
                        self.specific_energy = Some(self.energy_capacity / new_mass);
                    }
                    MassSideEffect::None => {
                        self.specific_energy = None;
                    }
                }
            }
        } else if new_mass.is_none() {
            self.specific_energy = None;
        }
        self.mass = new_mass;

        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(self
            .specific_energy
            .map(|specific_energy| self.energy_capacity / specific_energy))
    }

    fn expunge_mass_fields(&mut self) {
        self.mass = None;
        self.specific_energy = None;
    }
}

impl Init for ReversibleEnergyStorage {
    fn init(&mut self) -> Result<(), Error> {
        let _ = self
            .mass()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.state
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        Ok(())
    }
}
impl SerdeAPI for ReversibleEnergyStorage {
    fn from_file<P: AsRef<Path>>(filepath: P, skip_init: bool) -> Result<Self, Error> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| {
                Error::SerdeError(format!("File extension could not be parsed: {filepath:?}"))
            })?;
        let mut file = File::open(filepath)
            .with_context(|| {
                if !filepath.exists() {
                    format!("File not found: {filepath:?}")
                } else {
                    format!("Could not open file: {filepath:?}")
                }
            })
            .map_err(|err| Error::SerdeError(format!("{err}")))?;
        let mut network = match Self::from_reader(&mut file, extension, skip_init) {
            Ok(network) => network,
            Err(err) => res_legacy::ReversibleEnergyStorageLegacy::from_file(filepath, false)
                .map_err(|old_err| {
                    Error::SerdeError(
                        format!(
                            "\nattempting to load as `ReversibleEnergyStorage`:\n{err}\nattempting to load as `ReversibleEnergyStorageLegacy`:\n{old_err}"
                        )
                    )
                })?
                .into(),
        };
        network.init()?;

        Ok(network)
    }
}

impl From<res_legacy::ReversibleEnergyStorageLegacy> for ReversibleEnergyStorage {
    fn from(value: res_legacy::ReversibleEnergyStorageLegacy) -> Self {
        Self {
            state: value.state,
            mass: value.mass,
            volume: value.volume,
            specific_energy: value.specific_energy,
            energy_density: value.energy_density,
            eta_interp_grid: value.eta_interp_grid,
            eta_interp_values: value.eta_interp_values,
            pwr_out_max: value.pwr_out_max,
            energy_capacity: value.energy_capacity,
            min_soc: value.min_soc,
            max_soc: value.max_soc,
            save_interval: value.save_interval,
            history: value.history,
        }
    }
}

#[allow(unused)]
impl ReversibleEnergyStorage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        temperature_interp_grid: Vec<f64>,
        soc_interp_grid: Vec<f64>,
        c_rate_interp_grid: Vec<f64>,
        eta_interp_values: Vec<Vec<Vec<f64>>>,
        pwr_out_max_watts: f64,
        energy_capacity_joules: f64,
        min_soc: f64,
        max_soc: f64,
        initial_soc: f64,
        initial_temperature_celcius: f64,
        soc_hi_ramp_start: Option<f64>,
        soc_lo_ramp_start: Option<f64>,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            temperature_interp_grid.len() == eta_interp_values.len(),
            format!(
                "{}\nres temperature grid size must match eta_interp_values dimension 0",
                format_dbg!(temperature_interp_grid.len() == eta_interp_values.len())
            )
        );
        ensure!(
            is_sorted(&temperature_interp_grid),
            format!(
                "{}\nres temperature grid must be sorted",
                format_dbg!(is_sorted(&temperature_interp_grid))
            )
        );
        ensure!(
            soc_interp_grid.len() == eta_interp_values[0].len(),
            format!(
                "{}\nsoc grid size must match eta_interp_values dimension 1",
                format_dbg!(soc_interp_grid.len() == eta_interp_values[0].len())
            )
        );
        ensure!(
            is_sorted(&soc_interp_grid),
            format!(
                "{}\nsoc grid must be sorted",
                format_dbg!(is_sorted(&soc_interp_grid))
            )
        );
        ensure!(
            c_rate_interp_grid.len() == eta_interp_values[0][0].len(),
            format!(
                "{}\nc rate grid size must match eta_interp_values dimension 2",
                format_dbg!(c_rate_interp_grid.len() == eta_interp_values[0][0].len())
            )
        );
        ensure!(
            is_sorted(&soc_interp_grid),
            format!(
                "{}\ncrate grid must be sorted",
                format_dbg!(is_sorted(&soc_interp_grid))
            )
        );
        ensure!(
            min_soc <= initial_soc || initial_soc <= max_soc,
            format!(
                "{}\ninitial soc must be between min and max soc, inclusive",
                format_dbg!(min_soc <= initial_soc || initial_soc <= max_soc)
            )
        );

        let initial_state = ReversibleEnergyStorageState {
            soc: TrackedState::new(uc::R * initial_soc),
            temperature_celsius: TrackedState::new(initial_temperature_celcius),
            ..Default::default()
        };
        let interp_grid = [temperature_interp_grid, soc_interp_grid, c_rate_interp_grid];
        Ok(ReversibleEnergyStorage {
            eta_interp_grid: interp_grid,
            eta_interp_values,
            pwr_out_max: uc::W * pwr_out_max_watts,
            energy_capacity: uc::J * energy_capacity_joules,
            min_soc: uc::R * min_soc,
            max_soc: uc::R * max_soc,
            state: initial_state,
            save_interval,
            history: ReversibleEnergyStorageStateHistoryVec::new(),
            ..Default::default()
        })
    }

    fn volume(&self) -> anyhow::Result<Option<si::Volume>> {
        self.check_vol_consistent()?;
        Ok(self.volume)
    }

    fn update_volume(&mut self, volume: Option<si::Volume>) -> anyhow::Result<()> {
        match volume {
            Some(volume) => {
                self.energy_density = Some(self.energy_capacity / volume);
                self.volume = Some(volume);
            }
            None => match self.energy_density {
                Some(e) => self.volume = Some(self.energy_capacity / e),
                None => {
                    bail!(format!(
                        "{}\n{}",
                        format_dbg!(),
                        "Volume must be provided or `self.energy_density` must be set"
                    ));
                }
            },
        };

        Ok(())
    }

    fn check_vol_consistent(&self) -> anyhow::Result<()> {
        if let Some(vol) = &self.volume {
            if let Some(e) = &self.energy_density {
                ensure!(
                    self.energy_capacity / *e == *vol,
                    format!(
                        "{}\n{}",
                        format_dbg!(),
                        "ReversibleEnergyStorage `energy_capacity`, `energy_density` and `volume` are not consistent")
                )
            }
        }
        Ok(())
    }

    /// Sets and returns max output and max regen power based on current state
    /// # Arguments
    /// - `dt`: time step size
    /// - `disch_buffer`: buffer above static minimum SOC below which discharging is linearly derated
    /// - `chrg_buffer`: buffer below static maximum SOC above which charging is linearly derated
    pub fn set_curr_pwr_out_max(
        &mut self,
        dt: si::Time,
        pwr_aux: si::Power,
        disch_buffer: si::Energy,
        chrg_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        self.set_pwr_charge_max(pwr_aux, dt, chrg_buffer)
            .with_context(|| format_dbg!())?;
        self.set_pwr_disch_max(pwr_aux, dt, disch_buffer)
            .with_context(|| format_dbg!())?;

        Ok(())
    }

    /// # Arguments
    /// - `dt`: simulation time step size
    /// - `buffer`: buffer below static maximum SOC above which charging is disabled
    pub fn set_pwr_charge_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
        chrg_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        // to protect against excessive topping off of the battery
        let soc_buffer_delta = (chrg_buffer
            / (self.energy_capacity * (self.max_soc - self.min_soc)))
            .max(si::Ratio::ZERO);
        ensure!(soc_buffer_delta >= si::Ratio::ZERO, "{}", format_dbg!());
        self.state
            .soc_chrg_buffer
            .update(self.max_soc - soc_buffer_delta, || format_dbg!())?;
        let pwr_max_for_dt = ((self.max_soc - *self.state.soc.get_stale(|| format_dbg!())?)
            * self.energy_capacity
            / dt)
            .max(si::Power::ZERO);
        self.state.pwr_charge_max.update(
            if *self.state.soc.get_stale(|| format_dbg!())?
                <= *self.state.soc_chrg_buffer.get_fresh(|| format_dbg!())?
            {
                self.pwr_out_max
            } else if *self.state.soc.get_stale(|| format_dbg!())? < self.max_soc
                && soc_buffer_delta > si::Ratio::ZERO
            {
                self.pwr_out_max * (self.max_soc - *self.state.soc.get_stale(|| format_dbg!())?)
                    / soc_buffer_delta
            } else {
                // current SOC is less than both
                si::Power::ZERO
            }
            .min(pwr_max_for_dt),
            || format_dbg!(),
        )?;

        ensure!(
            *self.state.pwr_charge_max.get_fresh(|| format_dbg!())? >= si::Power::ZERO,
            "{}\n`{}` ({} W) must be greater than or equal to zero\n{}",
            format_dbg!(),
            stringify!(*self.state.pwr_charge_max.get_fresh(|| format_dbg!())?),
            self.state
                .pwr_charge_max
                .get_fresh(|| format_dbg!())?
                .get::<si::watt>()
                .format_eng(None),
            format_dbg!(soc_buffer_delta)
        );

        self.state.pwr_regen_max.update(
            *self.state.pwr_charge_max.get_fresh(|| format_dbg!())? + pwr_aux,
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// # Arguments
    /// - `dt`: simulation time step size
    /// - `buffer`: buffer above static minimum SOC below which discharging is linearly derated
    pub fn set_pwr_disch_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
        disch_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        // to protect against excessive bottoming out of the battery
        let soc_buffer_delta = (disch_buffer / self.energy_capacity_usable()).max(si::Ratio::ZERO);
        ensure!(soc_buffer_delta >= si::Ratio::ZERO, "{}", format_dbg!());
        self.state
            .soc_disch_buffer
            .update(self.min_soc + soc_buffer_delta, || format_dbg!())?;
        let pwr_max_for_dt = ((*self.state.soc.get_stale(|| format_dbg!())? - self.min_soc)
            * self.energy_capacity
            / dt)
            .max(si::Power::ZERO);
        self.state.pwr_disch_max.update(
            if *self.state.soc.get_stale(|| format_dbg!())?
                > *self.state.soc_disch_buffer.get_fresh(|| format_dbg!())?
            {
                self.pwr_out_max
            } else if *self.state.soc.get_stale(|| format_dbg!())? > self.min_soc
                && soc_buffer_delta > si::Ratio::ZERO
            {
                self.pwr_out_max * (*self.state.soc.get_stale(|| format_dbg!())? - self.min_soc)
                    / soc_buffer_delta
            } else {
                // current SOC is less than both
                si::Power::ZERO
            }
            .min(pwr_max_for_dt),
            || format_dbg!(),
        )?;

        ensure!(
            *self.state.pwr_disch_max.get_fresh(|| format_dbg!())? >= si::Power::ZERO,
            "{}\n`{}` ({} W) must be greater than or equal to zero\n{}",
            format_dbg!(),
            stringify!(*self.state.pwr_disch_max.get_fresh(|| format_dbg!())?),
            self.state
                .pwr_disch_max
                .get_fresh(|| format_dbg!())?
                .get::<si::watt>()
                .format_eng(None),
            format_dbg!(soc_buffer_delta)
        );

        self.state.pwr_prop_max.update(
            *self.state.pwr_disch_max.get_fresh(|| format_dbg!())? - pwr_aux,
            || format_dbg!(),
        )?;

        // TODO: remove this when implementing catenary
        self.state.pwr_cat_max.mark_fresh(|| format_dbg!());

        Ok(())
    }

    pub fn solve_energy_consumption(
        &mut self,
        pwr_prop_req: si::Power,
        pwr_aux_req: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let state = &mut self.state;

        ensure!(
            *state.soc.get_stale(|| format_dbg!())? <= self.max_soc
                || pwr_prop_req >= si::Power::ZERO,
            "{}\npwr_prop_req must be greater than 0 if SOC is over max SOC\nstate.soc = {}",
            format_dbg!(
                *state.soc.get_stale(|| format_dbg!())? <= self.max_soc
                    || pwr_prop_req >= si::Power::ZERO
            ),
            state.soc.get_stale(|| format_dbg!())?.get::<si::ratio>()
        );
        ensure!(
            *state.soc.get_stale(|| format_dbg!())? >= self.min_soc
                || pwr_prop_req <= si::Power::ZERO,
            "{}\npwr_prop_req must be less than 0 if SOC is below min SOC\nstate.soc = {}",
            format_dbg!(
                *state.soc.get_stale(|| format_dbg!())? >= self.min_soc
                    || pwr_prop_req <= si::Power::ZERO
            ),
            state.soc.get_stale(|| format_dbg!())?.get::<si::ratio>()
        );

        if pwr_prop_req + pwr_aux_req >= si::Power::ZERO {
            ensure!(
                utils::almost_le_uom(&(pwr_prop_req), state.pwr_disch_max.get_fresh(||format_dbg!())?, Some(TOL)),
                "{}\nres required power for propulsion ({:.6} MW) exceeds transient max propulsion power ({:.6} MW)\nstate.soc = {}
{}
{}
",
                format_dbg!(utils::almost_le_uom(&(pwr_prop_req), state.pwr_prop_max.get_fresh(||format_dbg!())?, Some(TOL))),
                pwr_prop_req.get::<si::megawatt>(),
                state.pwr_prop_max.get_fresh(||format_dbg!())?.get::<si::megawatt>(),
                state.soc.get_stale(||format_dbg!())?.get::<si::ratio>(),
                format_dbg!(pwr_aux_req.get::<si::kilowatt>()),
                format_dbg!(state.pwr_disch_max.get_fresh(||format_dbg!())?.get::<si::kilowatt>())
            );
            ensure!(
                utils::almost_le_uom(&(pwr_prop_req + pwr_aux_req), state.pwr_disch_max.get_fresh(|| format_dbg!())?, Some(TOL)),
                "{}\nres required power ({:.6} MW) exceeds static max discharge power ({:.6} MW)\nstate.soc = {}",

                format_dbg!(utils::almost_le_uom(
                    &(pwr_prop_req + pwr_aux_req),
                    &self.pwr_out_max,
                    Some(TOL)
                )),
                (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                state.pwr_disch_max.get_fresh(||format_dbg!())?.get::<si::megawatt>(),
                state.soc.get_fresh(||format_dbg!())?.get::<si::ratio>()
            );
            ensure!(
                utils::almost_le_uom(&(pwr_prop_req + pwr_aux_req), state.pwr_disch_max.get_fresh(||format_dbg!())?, Some(TOL)),
                "{}\nres required power ({:.6} MW) exceeds transient max discharge power ({:.6} MW)\nstate.soc = {}",
                format_dbg!(utils::almost_le_uom(&(pwr_prop_req + pwr_aux_req), state.pwr_disch_max.get_fresh(||format_dbg!())?, Some(TOL))),
                (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                state.pwr_disch_max.get_fresh(||format_dbg!())?.get::<si::megawatt>(),
                state.soc.get_fresh(||format_dbg!())?.get::<si::ratio>()
            );
        } else {
            ensure!(
                utils::almost_ge_uom(&(pwr_prop_req + pwr_aux_req), &-self.pwr_out_max, Some(TOL)),
                format!(
                    "{}\nres required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_prop_req + pwr_aux_req),
                        &-self.pwr_out_max,
                        Some(TOL)
                    )),
                    (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                    state
                        .pwr_charge_max
                        .get_fresh(|| format_dbg!())?
                        .get::<si::megawatt>()
                )
            );
            ensure!(
                utils::almost_ge_uom(
                    &(pwr_prop_req + pwr_aux_req),
                    &-*state.pwr_charge_max.get_fresh(|| format_dbg!())?,
                    Some(TOL)
                ),
                format!(
                    "{}\nres required power ({:.6} MW) exceeds transient max power ({:.6} MW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_prop_req + pwr_aux_req),
                        &-*state.pwr_charge_max.get_fresh(|| format_dbg!())?,
                        Some(TOL)
                    )),
                    (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                    state
                        .pwr_charge_max
                        .get_fresh(|| format_dbg!())?
                        .get::<si::megawatt>()
                )
            );
        }

        state
            .pwr_out_propulsion
            .update(pwr_prop_req, || format_dbg!())?;
        state.pwr_aux.update(pwr_aux_req, || format_dbg!())?;

        state.pwr_out_electrical.update(
            *state.pwr_out_propulsion.get_fresh(|| format_dbg!())?
                + *state.pwr_aux.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;

        let c_rate = state
            .pwr_out_electrical
            .get_fresh(|| format_dbg!())?
            .get::<si::watt>()
            / (self.energy_capacity.get::<si::watt_hour>());
        // evaluate the battery efficiency at the current state
        state.temperature_celsius.mark_fresh(|| format_dbg!())?;
        let eta_point = [
            *state.temperature_celsius.get_fresh(|| format_dbg!())?,
            state.soc.get_stale(|| format_dbg!())?.get::<si::ratio>(),
            c_rate,
        ];
        let eta = interp3d(&eta_point, &self.eta_interp_grid, &self.eta_interp_values).unwrap();

        state.eta.update(uc::R * eta, || format_dbg!())?;
        ensure!(
            *state.eta.get_fresh(|| format_dbg!())? >= 0.0 * uc::R
                || *state.eta.get_fresh(|| format_dbg!())? <= 1.0 * uc::R,
            format!(
                "{}\nres eta ({}) must be between 0 and 1",
                format_dbg!(
                    *state.eta.get_fresh(|| format_dbg!())? >= 0.0 * uc::R
                        || *state.eta.get_fresh(|| format_dbg!())? <= 1.0 * uc::R
                ),
                state.eta.get_fresh(|| format_dbg!())?.get::<si::ratio>()
            )
        );

        if *state.pwr_out_electrical.get_fresh(|| format_dbg!())? > si::Power::ZERO {
            // if positive, chemical power must be greater than electrical power
            // i.e. not all chemical power can be converted to electrical power
            state.pwr_out_chemical.update(
                *state.pwr_out_electrical.get_fresh(|| format_dbg!())? / eta,
                || format_dbg!(),
            )?;
        } else {
            // if negative, chemical power, must be less than electrical power
            // i.e. not all electrical power can be converted back to chemical power
            state.pwr_out_chemical.update(
                *state.pwr_out_electrical.get_fresh(|| format_dbg!())? * eta,
                || format_dbg!(),
            )?;
        }

        state.pwr_loss.update(
            (*state.pwr_out_chemical.get_fresh(|| format_dbg!())?
                - *state.pwr_out_electrical.get_fresh(|| format_dbg!())?)
            .abs(),
            || format_dbg!(),
        )?;

        let new_soc = *state.soc.get_stale(|| format_dbg!())?
            - *state.pwr_out_chemical.get_fresh(|| format_dbg!())? * dt / self.energy_capacity;
        state.soc.update(new_soc, || format_dbg!())?;

        // TODO: change this when implementing soh
        state.soh.mark_fresh(|| format_dbg!())?;

        Ok(())
    }

    pub fn get_eta_max(&self) -> f64 {
        // since eta is all f64 between 0 and 1, NEG_INFINITY is safe
        self.eta_interp_values
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr2| {
                curr2
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc, curr1| {
                        curr1
                            .iter()
                            .fold(f64::NEG_INFINITY, |acc, &curr| acc.max(curr))
                            .max(acc)
                    })
                    .max(acc)
            })
    }

    /// Scales eta_interp by ratio of new `eta_max` per current calculated
    /// max linearly, such that `eta_min` is untouched
    pub fn set_eta_max(&mut self, eta_max: f64) -> Result<(), String> {
        if (self.get_eta_min()..=1.0).contains(&eta_max) {
            // this appears to be efficient way to get max of Vec<f64>
            let old_max = self.get_eta_max();
            self.eta_interp_values = self
                .eta_interp_values
                .iter()
                .map(|v2| {
                    v2.iter()
                        .map(|v1| v1.iter().map(|val| val * eta_max / old_max).collect())
                        .collect()
                })
                .collect();
            Ok(())
        } else {
            Err(format!(
                "`eta_max` ({:.3}) must be between `eta_min` ({:.3}) and 1.0",
                eta_max,
                self.get_eta_min()
            ))
        }
    }

    pub fn get_eta_min(&self) -> f64 {
        // since eta is all f64 between 0 and 1, INFINITY is safe
        self.eta_interp_values
            .iter()
            .fold(f64::INFINITY, |acc, curr2| {
                curr2
                    .iter()
                    .fold(f64::INFINITY, |acc, curr1| {
                        curr1
                            .iter()
                            .fold(f64::INFINITY, |acc, &curr| acc.min(curr))
                            .min(acc)
                    })
                    .min(acc)
            })
    }

    /// Max value of `eta_interp` minus min value of `eta_interp`.
    pub fn get_eta_range(&self) -> f64 {
        self.get_eta_max() - self.get_eta_min()
    }

    /// Scales values of `eta_interp` without changing max such that max - min
    /// is equal to new range
    pub fn set_eta_range(&mut self, eta_range: f64) -> anyhow::Result<()> {
        let eta_max = self.get_eta_max();
        if eta_range == 0.0 {
            self.eta_interp_values = self
                .eta_interp_values
                .iter()
                .map(|v2| {
                    v2.iter()
                        // this is sloppy but should work
                        .map(|v1| v1.iter().map(|_val| eta_max).collect())
                        .collect()
                })
                .collect();
            Ok(())
        } else if (0.0..=1.0).contains(&eta_range) {
            let old_min = self.get_eta_min();
            let old_range = self.get_eta_max() - old_min;

            self.eta_interp_values = self
                .eta_interp_values
                .iter()
                .map(|v2| {
                    v2.iter()
                        .map(|v1| {
                            v1.iter()
                                .map(|val| eta_max + (val - eta_max) * eta_range / old_range)
                                .collect()
                        })
                        .collect()
                })
                .collect();
            if self.get_eta_min() < 0.0 {
                let val_neg = self.get_eta_min();
                self.eta_interp_values = self
                    .eta_interp_values
                    .iter()
                    .map(|v2| {
                        v2.iter()
                            .map(|v1| v1.iter().map(|val| val - val_neg).collect())
                            .collect()
                    })
                    .collect();
            }
            ensure!(
                self.get_eta_max() <= 1.0,
                format!(
                    "{}\n`eta_max` ({:.3}) must be no greater than 1.0",
                    format_dbg!(self.get_eta_max() <= 1.0),
                    self.get_eta_max()
                )
            );
            Ok(())
        } else {
            bail!("`eta_range` ({:.3}) must be between 0.0 and 1.0", eta_range)
        }
    }

    /// Usable energy capacity, accounting for SOC limits
    pub fn energy_capacity_usable(&self) -> si::Energy {
        self.energy_capacity * (self.max_soc - self.min_soc)
    }

    /// Mean efficiency in charge direction
    pub fn mean_chrg_eff(&self) -> anyhow::Result<si::Ratio> {
        let mut eta_sum = si::Ratio::ZERO;
        for (eta, pwr_out) in self
            .history
            .eta
            .iter()
            .zip(self.history.pwr_out_electrical.clone())
        {
            eta_sum += if *pwr_out.get_fresh(|| format_dbg!())? < si::Power::ZERO {
                *eta.get_fresh(|| format_dbg!())?
            } else {
                si::Ratio::ZERO
            };
        }

        Ok(eta_sum / (self.history.len() as f64))
    }

    /// Mean efficiency in discharge direction
    pub fn mean_dschrg_eff(&self) -> anyhow::Result<si::Ratio> {
        let mut eta_sum = si::Ratio::ZERO;
        for (eta, pwr_out) in self
            .history
            .eta
            .iter()
            .zip(self.history.pwr_out_electrical.clone())
        {
            eta_sum += if *pwr_out.get_fresh(|| format_dbg!())? >= si::Power::ZERO {
                *eta.get_fresh(|| format_dbg!())?
            } else {
                si::Ratio::ZERO
            };
        }

        Ok(eta_sum / (self.history.len() as f64))
    }
}

#[serde_api]
#[derive(
    Clone, Deserialize, Serialize, Debug, PartialEq, HistoryVec, StateMethods, SetCumulative,
)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
// component limits
/// ReversibleEnergyStorage state variables
pub struct ReversibleEnergyStorageState {
    // limits
    // TODO: create separate binning for cat power and maximum catenary power capability
    pub pwr_cat_max: TrackedState<si::Power>,
    /// max output power for propulsion during positive traction
    pub pwr_prop_max: TrackedState<si::Power>,
    /// max regen power for propulsion during negative traction
    pub pwr_regen_max: TrackedState<si::Power>,
    /// max discharge power at the battery terminals
    pub pwr_disch_max: TrackedState<si::Power>,
    /// max charge power at the battery terminals
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

    /// buffer above minimum SOC at which battery max discharge rate is linearly
    /// reduced as soc approaches `min_soc`
    pub soc_disch_buffer: TrackedState<si::Ratio>,
    /// buffer below maximum SOC at which battery max charge rate is linearly
    /// reduced as soc approaches `min_soc`
    pub soc_chrg_buffer: TrackedState<si::Ratio>,

    /// component temperature
    pub temperature_celsius: TrackedState<f64>,
}

#[pyo3_api]
impl ReversibleEnergyStorageState {}

impl SerdeAPI for ReversibleEnergyStorageState {}
impl Init for ReversibleEnergyStorageState {}

impl Default for ReversibleEnergyStorageState {
    fn default() -> Self {
        Self {
            i: Default::default(),
            soc: TrackedState::new(uc::R * 0.95),
            soh: TrackedState::new(1.0),
            eta: Default::default(),
            pwr_prop_max: Default::default(),
            pwr_regen_max: Default::default(),
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
            soc_chrg_buffer: TrackedState::new(uc::R * 1.0),
            soc_disch_buffer: TrackedState::new(uc::R * 0.0),
            temperature_celsius: TrackedState::new(45.0),
        }
    }
}

mod tests {
    use super::*;

    fn _mock_res() -> ReversibleEnergyStorage {
        ReversibleEnergyStorage::default()
    }

    #[test]
    fn test_res_constructor() {
        let _res = _mock_res();
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut res = _mock_res();
        let eta_max = 0.998958;
        let eta_min = 0.662822531196789;
        let eta_range = 0.336135468803211;

        eta_test_body!(res, eta_max, eta_min, eta_range);
    }
}
