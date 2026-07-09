use super::environment::TemperatureTrace;
use super::train_imports::*;

#[serde_api]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct SpeedTrace {
    /// simulation time
    pub time: Vec<si::Time>,
    /// simulation speed
    pub speed: Vec<si::Velocity>,
    /// Whether engine is on
    pub engine_on: Option<Vec<bool>>,
}

#[pyo3_api]
impl SpeedTrace {
    #[new]
    #[pyo3(signature = (
        time_seconds,
        speed_meters_per_second,
        engine_on=None
    ))]
    fn __new__(
        time_seconds: Vec<f64>,
        speed_meters_per_second: Vec<f64>,
        engine_on: Option<Vec<bool>>,
    ) -> anyhow::Result<Self> {
        Ok(Self::new(time_seconds, speed_meters_per_second, engine_on))
    }

    #[staticmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(filepath: &Bound<PyAny>) -> anyhow::Result<Self> {
        Self::from_csv_file(PathBuf::extract_bound(filepath)?)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    #[pyo3(name = "to_csv_file")]
    fn to_csv_file_py(&self, filepath: &Bound<PyAny>) -> anyhow::Result<()> {
        self.to_csv_file(PathBuf::extract_bound(filepath)?)
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl SpeedTrace {
    pub fn new(time_s: Vec<f64>, speed_mps: Vec<f64>, engine_on: Option<Vec<bool>>) -> Self {
        SpeedTrace {
            time: time_s.iter().map(|x| uc::S * (*x)).collect(),
            speed: speed_mps.iter().map(|x| uc::MPS * (*x)).collect(),
            engine_on,
        }
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or_else(|| self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.speed = self.speed[start_idx..end_idx].to_vec();
        self.engine_on = self
            .engine_on
            .as_ref()
            .map(|eo| eo[start_idx..end_idx].to_vec());
        Ok(())
    }

    pub fn dt(&self, i: usize) -> si::Time {
        self.time[i] - self.time[i - 1]
    }

    pub fn mean(&self, i: usize) -> si::Velocity {
        0.5 * (self.speed[i] + self.speed[i - 1])
    }

    pub fn acc(&self, i: usize) -> si::Acceleration {
        (self.speed[i] - self.speed[i - 1]) / self.dt(i)
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// method to prevent rust-analyzer from complaining
    pub fn is_empty(&self) -> bool {
        self.time.is_empty() && self.speed.is_empty() && self.engine_on.is_none()
    }

    pub fn push(&mut self, speed_element: SpeedTraceElement) -> anyhow::Result<()> {
        self.time.push(speed_element.time);
        self.speed.push(speed_element.speed);
        self.engine_on
            .as_mut()
            .map(|eo| match speed_element.engine_on {
                Some(seeeo) => {
                    eo.push(seeeo);
                    Ok(())
                }
                None => bail!(
                    "`engine_one` in `SpeedTraceElement` and `SpeedTrace` must both have same option variant."),
            });
        Ok(())
    }

    pub fn empty() -> Self {
        Self {
            time: Vec::new(),
            speed: Vec::new(),
            engine_on: None,
        }
    }

    /// Load speed trace from csv file
    pub fn from_csv_file<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();

        // create empty SpeedTrace to be populated
        let mut st = Self::empty();

        let file = File::open(filepath)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let st_elem: SpeedTraceElement = result?;
            st.push(st_elem)?;
        }
        ensure!(
            !st.is_empty(),
            "Invalid SpeedTrace file {:?}; SpeedTrace is empty",
            filepath
        );
        Ok(st)
    }

    /// Save speed trace to csv file
    pub fn to_csv_file<P: AsRef<Path>>(&self, filepath: P) -> anyhow::Result<()> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filepath)?;
        let mut wrtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(file);
        let engine_on: Vec<Option<bool>> = match &self.engine_on {
            Some(eo_vec) => eo_vec
                .iter()
                .map(|eo| Some(*eo))
                .collect::<Vec<Option<bool>>>(),
            None => vec![None; self.len()],
        };
        for ((time, speed), engine_on) in self.time.iter().zip(&self.speed).zip(engine_on) {
            wrtr.serialize(SpeedTraceElement {
                time: *time,
                speed: *speed,
                engine_on,
            })?;
        }
        wrtr.flush()?;
        Ok(())
    }
}

impl Init for SpeedTrace {}
impl SerdeAPI for SpeedTrace {}

impl Default for SpeedTrace {
    fn default() -> Self {
        let mut speed_mps: Vec<f64> = Vec::linspace(0.0, 20.0, 800);
        speed_mps.append(&mut [20.0; 100].to_vec());
        speed_mps.append(&mut Vec::linspace(20.0, 0.0, 200));
        speed_mps.push(0.0);
        let time_s: Vec<f64> = (0..speed_mps.len()).map(|x| x as f64).collect();
        Self::new(time_s, speed_mps, None)
    }
}

/// Element of [SpeedTrace].  Used for vec-like operations.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeedTraceElement {
    /// simulation time
    #[serde(alias = "time_seconds")]
    time: si::Time,
    /// prescribed speed
    #[serde(alias = "speed_meters_per_second")]
    speed: si::Velocity,
    /// whether engine is on
    engine_on: Option<bool>,
}

#[serde_api]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Train simulation in which speed is prescribed.  Note that this is not guaranteed to
/// produce identical results to [super::SpeedLimitTrainSim] because of differences in braking
/// controls but should generally be very close (i.e. error in cumulative fuel/battery energy
/// should be less than 0.1%)
pub struct SetSpeedTrainSim {
    pub loco_con: Consist,
    pub n_cars_by_type: HashMap<String, u32>,
    #[serde(default)]
    pub state: TrainState,
    pub speed_trace: SpeedTrace,

    /// train resistance calculation
    pub train_res: TrainRes,

    path_tpc: PathTpc,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: TrainStateHistoryVec,

    save_interval: Option<usize>,
    /// Time-dependent temperature at sea level that can be corrected for
    /// altitude using a standard model
    temp_trace: Option<TemperatureTrace>,
}

#[pyo3_api]
impl SetSpeedTrainSim {
    #[getter]
    pub fn get_res_strap(&self) -> anyhow::Result<Option<method::Strap>> {
        match &self.train_res {
            TrainRes::Strap(strap) => Ok(Some(strap.clone())),
            _ => Ok(None),
        }
    }

    #[getter]
    pub fn get_res_point(&self) -> anyhow::Result<Option<method::Point>> {
        match &self.train_res {
            TrainRes::Point(point) => Ok(Some(point.clone())),
            _ => Ok(None),
        }
    }

    #[pyo3(name = "walk")]
    /// Exposes `walk` to Python.
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[pyo3(name = "step")]
    fn step_py(&mut self) -> anyhow::Result<()> {
        self.step(|| format_dbg!())
    }

    #[pyo3(name = "set_save_interval")]
    #[pyo3(signature = (save_interval=None))]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }

    #[pyo3(name = "get_save_interval")]
    fn get_save_interval_py(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.get_save_interval())
    }

    #[pyo3(name = "trim_failed_steps")]
    fn trim_failed_steps_py(&mut self) -> anyhow::Result<()> {
        self.trim_failed_steps()?;
        Ok(())
    }
}

pub struct SetSpeedTrainSimBuilder {
    pub loco_con: Consist,
    /// Number of railcars by type on the train
    pub n_cars_by_type: HashMap<String, u32>,
    pub state: TrainState,
    pub speed_trace: SpeedTrace,
    pub train_res: TrainRes,
    pub path_tpc: PathTpc,
    pub save_interval: Option<usize>,
    /// Time-dependent temperature at sea level that can be corrected for altitude using a standard model
    pub temp_trace: Option<TemperatureTrace>,
}

impl From<SetSpeedTrainSimBuilder> for SetSpeedTrainSim {
    fn from(value: SetSpeedTrainSimBuilder) -> Self {
        SetSpeedTrainSim {
            loco_con: value.loco_con,
            n_cars_by_type: value.n_cars_by_type,
            state: value.state,
            speed_trace: value.speed_trace,
            train_res: value.train_res,
            path_tpc: value.path_tpc,
            history: Default::default(),
            save_interval: value.save_interval,
            temp_trace: value.temp_trace,
        }
    }
}

impl SetSpeedTrainSim {
    /// Trims off any portion of the trip that failed to run
    pub fn trim_failed_steps(&mut self) -> anyhow::Result<()> {
        if *self.state.i.get_fresh(|| format_dbg!())? <= 1 {
            bail!("`walk` method has not proceeded past first time step.")
        }
        self.speed_trace
            .trim(None, Some(*self.state.i.get_fresh(|| format_dbg!())?))?;

        Ok(())
    }

    /// Sets `save_interval` for self and nested `loco_con`.
    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        self.loco_con.set_save_interval(save_interval);
    }

    /// Returns `self.save_interval` and asserts that this is equal
    /// to `self.loco_con.get_save_interval()`.
    pub fn get_save_interval(&self) -> Option<usize> {
        // this ensures that save interval has been propagated
        assert_eq!(self.save_interval, self.loco_con.get_save_interval());
        self.save_interval
    }

    /// Solves time step.
    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        // checking on speed trace to ensure it is at least stopped or moving forward (no backwards)
        let dt = self.speed_trace.time[*self.state.i.get_fresh(|| format_dbg!())?]
            - *self.state.time.get_stale(|| format_dbg!())?;
        self.state.dt.update(dt, || format_dbg!())?;

        ensure!(
            self.speed_trace.speed[*self.state.i.get_fresh(|| format_dbg!())?]
                >= si::Velocity::ZERO,
            format_dbg!(
                self.speed_trace.speed[*self.state.i.get_fresh(|| format_dbg!())?]
                    >= si::Velocity::ZERO
            )
        );
        self.loco_con
            .state
            .pwr_cat_lim
            .mark_fresh(|| format_dbg!())?;
        // not used in set_speed_train_sim
        self.state.speed_limit.mark_fresh(|| format_dbg!())?;
        // not used in set_speed_train_sim
        self.state.speed_target.mark_fresh(|| format_dbg!())?;
        // not used in set_speed_train_sim
        self.state.mass_static.mark_fresh(|| format_dbg!())?;
        // not used in set_speed_train_sim
        self.state.mass_rot.mark_fresh(|| format_dbg!())?;
        // not used in set_speed_train_sim
        self.state.mass_freight.mark_fresh(|| format_dbg!())?;
        // TODO: update this if length ever becomes dynamic
        self.state.length.mark_fresh(|| format_dbg!())?;
        // set the catenary power limit.  I'm assuming it is 0 at this point.
        // self.loco_con.set_cat_power_limit(
        //     &self.path_tpc,
        //     *self.state.offset.get_fresh(|| format_dbg!())?,
        // )?;
        // set aux power loads.  this will be calculated in the locomotive model and be loco type dependent.
        self.loco_con.set_pwr_aux(Some(true))?;
        let train_mass = self.state.mass_compound().with_context(|| format_dbg!())?;

        let elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)> =
            if let Some(tt) = &self.temp_trace {
                Some((
                    *self.state.elev_front.get_fresh(|| format_dbg!())?,
                    tt.get_temp_at_time_and_elev(
                        *self.state.time.get_fresh(|| format_dbg!())?,
                        *self.state.elev_front.get_fresh(|| format_dbg!())?,
                    )
                    .with_context(|| format_dbg!())?,
                ))
            } else {
                None
            };

        // set the max power out for the consist based on calculation of each loco state
        self.loco_con.set_curr_pwr_max_out(
            si::Power::ZERO,
            elev_and_temp,
            train_mass,
            *self.state.speed.get_stale(|| format_dbg!())?,
            *self
                .state
                .lookahead_speed_limit
                .get_fresh(|| format_dbg!())?,
            *self.state.lookahead_elev.get_fresh(|| format_dbg!())?,
            self.speed_trace
                .dt(*self.state.i.get_fresh(|| format_dbg!())?),
        )?;
        // calculate the train resistance for current time steps.  Based on train config and calculated in train model.
        self.train_res
            .update_res(&mut self.state, &self.path_tpc, &Dir::Fwd)?;
        // figure out how much power is needed to pull train with current speed trace.
        self.solve_required_pwr(
            self.speed_trace
                .dt(*self.state.i.get_fresh(|| format_dbg!())?),
        )?;
        self.loco_con.solve_energy_consumption(
            *self.state.pwr_whl_out.get_fresh(|| format_dbg!())?,
            train_mass,
            self.speed_trace.speed[*self.state.i.get_fresh(|| format_dbg!())?],
            self.speed_trace
                .dt(*self.state.i.get_fresh(|| format_dbg!())?),
            Some(true),
        )?;
        // advance time
        self.state.time.increment(dt, || format_dbg!())?;
        // update speed
        self.state.speed.update(
            self.speed_trace.speed[*self.state.i.get_fresh(|| format_dbg!())?],
            || format_dbg!(),
        )?;
        set_link_and_offset(&mut self.state, &self.path_tpc)?;
        // update offset
        self.state.offset.increment(
            self.speed_trace
                .mean(*self.state.i.get_fresh(|| format_dbg!())?)
                * *self.state.dt.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        // update total distance
        self.state.total_dist.increment(
            (self
                .speed_trace
                .mean(*self.state.i.get_fresh(|| format_dbg!())?)
                * *self.state.dt.get_fresh(|| format_dbg!())?)
            .abs(),
            || format_dbg!(),
        )?;
        self.set_cumulative(
            *self.state.dt.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Iterates `save_state` and `step` through all time steps.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state(|| format_dbg!())?;
        loop {
            if *self.state.i.get_fresh(|| format_dbg!())? > self.speed_trace.len() - 2 {
                break;
            }
            self.step(|| format_dbg!()).with_context(|| format_dbg!())?;
        }
        Ok(())
    }

    /// Sets power requirements based on:
    /// - rolling resistance
    /// - drag
    /// - inertia
    /// - acceleration
    pub fn solve_required_pwr(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // This calculates the maximum power from loco based on current power, ramp rate, and dt of model.  will return 0 if this is negative.
        let pwr_pos_max = self
            .loco_con
            .state
            .pwr_out_max
            .get_fresh(|| format_dbg!())?
            .min(
                si::Power::ZERO.max(
                    *self.state.pwr_whl_out.get_stale(|| format_dbg!())?
                        + *self
                            .loco_con
                            .state
                            .pwr_rate_out_max
                            .get_fresh(|| format_dbg!())?
                            * *self.state.dt.get_fresh(|| format_dbg!())?,
                ),
            );

        // find max dynamic braking power as positive value
        let pwr_neg_max = self
            .loco_con
            .state
            .pwr_dyn_brake_max
            .get_fresh(|| format_dbg!())?
            .max(si::Power::ZERO);

        // not sure why we have these checks if the max function worked earlier.
        ensure!(
            pwr_pos_max >= si::Power::ZERO,
            format_dbg!(pwr_pos_max >= si::Power::ZERO)
        );

        // res for resistance is a horrible name.  It collides with reversible energy storage.  This like is calculating train resistance for the time step.
        self.state.pwr_res.update(
            self.state.res_net().with_context(|| format_dbg!())?
                * self
                    .speed_trace
                    .mean(*self.state.i.get_fresh(|| format_dbg!())?),
            || format_dbg!(),
        )?;
        // find power to accelerate the train mass from an energy perspective.
        self.state.pwr_accel.update(
            self.state.mass_compound().with_context(|| format_dbg!())?
                / (2.0
                    * self
                        .speed_trace
                        .dt(*self.state.i.get_fresh(|| format_dbg!())?))
                * (self.speed_trace.speed[*self.state.i.get_fresh(|| format_dbg!())?]
                    .powi(typenum::P2::new())
                    - self.speed_trace.speed[*self.state.i.get_fresh(|| format_dbg!())? - 1]
                        .powi(typenum::P2::new())),
            || format_dbg!(),
        )?;

        // total power exerted by the consist to move the train, without limits applied
        let pwr_whl_out_unclipped = *self.state.pwr_accel.get_fresh(|| format_dbg!())?
            + *self.state.pwr_res.get_fresh(|| format_dbg!())?;

        // limit power to within the consist capability
        self.state.pwr_whl_out.update(
            pwr_whl_out_unclipped.max(-pwr_neg_max).min(pwr_pos_max),
            || format_dbg!(),
        )?;

        // add to positive or negative wheel energy tracking.
        if *self.state.pwr_whl_out.get_fresh(|| format_dbg!())? >= 0. * uc::W {
            self.state.energy_whl_out_pos.increment(
                *self.state.pwr_whl_out.get_fresh(|| format_dbg!())? * dt,
                || format_dbg!(),
            )?;
            self.state
                .energy_whl_out_neg
                .increment(si::Energy::ZERO, || format_dbg!())?;
        } else {
            self.state.energy_whl_out_neg.increment(
                -*self.state.pwr_whl_out.get_fresh(|| format_dbg!())? * dt,
                || format_dbg!(),
            )?;
            self.state
                .energy_whl_out_pos
                .increment(si::Energy::ZERO, || format_dbg!())?;
        }
        Ok(())
    }
}

impl StateMethods for SetSpeedTrainSim {}
impl CheckAndResetState for SetSpeedTrainSim {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        // self.state.speed_limit.mark_fresh(|| format_dbg!())?;
        self.state
            .check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.loco_con
            .check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}
impl SetCumulative for SetSpeedTrainSim {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        self.state
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.loco_con
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}

impl Step for SetSpeedTrainSim {
    /// Solves step, saves state, steps nested `loco_con`, and increments `self.i`.
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        let i = *self.state.i.get_fresh(|| format_dbg!())?;
        self.check_and_reset(|| format_dbg!())?;
        self.state
            .i
            .increment(1, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.loco_con.step(|| format_dbg!())?;
        self.solve_step()
            .with_context(|| format!("{}\ntime step: {}", loc(), i))?;

        self.save_state(|| format_dbg!())?;
        Ok(())
    }
}
impl SaveState for SetSpeedTrainSim {
    /// Saves current time step for self and nested `loco_con`.
    fn save_state<F: Fn() -> String>(&mut self, _loc: F) -> anyhow::Result<()> {
        if let Some(interval) = self.save_interval {
            if self.state.i.get_fresh(|| format_dbg!())? % interval == 0 {
                self.history.push(self.state.clone());
                self.loco_con.save_state(|| format_dbg!())?;
            }
        }
        Ok(())
    }
}
impl Init for SetSpeedTrainSim {
    fn init(&mut self) -> Result<(), Error> {
        self.loco_con.init()?;
        self.speed_trace.init()?;
        self.train_res.init()?;
        self.path_tpc.init()?;
        self.state.init()?;
        self.history.init()?;
        Ok(())
    }
}
impl SerdeAPI for SetSpeedTrainSim {}

impl Default for SetSpeedTrainSim {
    fn default() -> Self {
        Self {
            loco_con: Consist::default(),
            n_cars_by_type: Default::default(),
            state: TrainState::valid(),
            train_res: TrainRes::valid(),
            path_tpc: PathTpc::valid(),
            speed_trace: SpeedTrace::default(),
            history: TrainStateHistoryVec::default(),
            save_interval: None,
            temp_trace: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SetSpeedTrainSim;

    #[test]
    fn test_set_speed_train_sim() {
        let mut train_sim = SetSpeedTrainSim::default();
        train_sim.walk().unwrap();
        assert!(
            *train_sim
                .loco_con
                .state
                .i
                .get_fresh(|| format_dbg!())
                .unwrap()
                > 1
        );
    }
}
