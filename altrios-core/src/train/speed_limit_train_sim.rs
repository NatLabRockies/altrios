use super::environment::TemperatureTrace;
use super::{braking_point::BrakingPoints, friction_brakes::*, train_imports::*};
use crate::imports::*;
use crate::track::link::network::Network;
use crate::track::{LinkPoint, Location};

#[serde_api]
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct LinkIdxTime {
    pub link_idx: LinkIdx,
    pub time: si::Time,
}

#[pyo3_api]
impl LinkIdxTime {
    #[new]
    fn __new__(link_idx: LinkIdx, time_seconds: f64) -> Self {
        Self::new(link_idx, time_seconds * uc::S)
    }
}

impl Init for LinkIdxTime {}
impl SerdeAPI for LinkIdxTime {}

impl LinkIdxTime {
    pub fn new(link_idx: LinkIdx, time: si::Time) -> Self {
        Self { link_idx, time }
    }
}

#[serde_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct that contains a `Vec<LinkIdxTime>` for the purpose of providing `SerdeAPI` for
/// `Vec<LinkIdxTime>` in Python
pub struct TimedLinkPath(pub Vec<LinkIdxTime>);

#[pyo3_api]
impl TimedLinkPath {}

impl Init for TimedLinkPath {}
impl SerdeAPI for TimedLinkPath {}

impl TimedLinkPath {
    /// Implement the non-Python `new` method.
    pub fn new(value: Vec<LinkIdxTime>) -> Self {
        Self(value)
    }
}

impl AsRef<[LinkIdxTime]> for TimedLinkPath {
    fn as_ref(&self) -> &[LinkIdxTime] {
        &self.0
    }
}

impl From<&Vec<LinkIdxTime>> for TimedLinkPath {
    fn from(value: &Vec<LinkIdxTime>) -> Self {
        Self(value.to_vec())
    }
}

#[serde_api]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Train simulation in which speed is allowed to vary according to train capabilities and speed
/// limit.  Note that this is not guaranteed to produce identical results to [super::SetSpeedTrainSim]
/// because of differences in braking controls but should generally be very close (i.e. error in cumulative
/// fuel/battery energy should be less than 0.1%)
pub struct SpeedLimitTrainSim {
    pub train_id: String,
    pub origs: Vec<Location>,
    pub dests: Vec<Location>,
    // #[has_state]
    pub loco_con: Consist,
    /// Number of railcars by type on the train
    pub n_cars_by_type: HashMap<String, u32>,
    #[serde(default)]
    pub state: TrainState,

    pub train_res: TrainRes,

    pub path_tpc: PathTpc,

    pub braking_points: BrakingPoints,
    // #[has_state]
    pub fric_brake: FricBrake,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: TrainStateHistoryVec,

    save_interval: Option<usize>,
    simulation_days: Option<i32>,
    scenario_year: Option<i32>,
    /// Time-dependent temperature at sea level that can be corrected for
    /// altitude using a standard model
    temp_trace: Option<TemperatureTrace>,
}

#[pyo3_api]
impl SpeedLimitTrainSim {
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

    #[pyo3(name = "get_kilometers")]
    pub fn get_kilometers_py(&self, annualize: bool) -> anyhow::Result<f64> {
        self.get_kilometers(annualize)
    }

    #[pyo3(name = "get_megagram_kilometers")]
    pub fn get_megagram_kilometers_py(&self, annualize: bool) -> anyhow::Result<f64> {
        self.get_megagram_kilometers(annualize)
    }

    #[pyo3(name = "get_car_kilometers")]
    pub fn get_car_kilometers_py(&self, annualize: bool) -> anyhow::Result<f64> {
        self.get_car_kilometers(annualize)
    }

    #[pyo3(name = "get_cars_moved")]
    pub fn get_cars_moved_py(&self, annualize: bool) -> f64 {
        self.get_cars_moved(annualize)
    }

    #[pyo3(name = "get_res_kilometers")]
    pub fn get_res_kilometers_py(&mut self, annualize: bool) -> anyhow::Result<f64> {
        self.get_res_kilometers(annualize)
    }

    #[pyo3(name = "get_non_res_kilometers")]
    pub fn get_non_res_kilometers_py(&mut self, annualize: bool) -> anyhow::Result<f64> {
        self.get_non_res_kilometers(annualize)
    }

    #[pyo3(name = "get_net_energy_res_joules")]
    pub fn get_net_energy_res_py(&self, annualize: bool) -> anyhow::Result<f64> {
        Ok(self.get_net_energy_res(annualize)?.get::<si::joule>())
    }

    #[pyo3(name = "get_energy_fuel_joules")]
    pub fn get_energy_fuel_py(&self, annualize: bool) -> anyhow::Result<f64> {
        Ok(self.get_energy_fuel(annualize)?.get::<si::joule>())
    }

    #[pyo3(name = "get_energy_fuel_soc_corrected_joules")]
    pub fn get_energy_fuel_soc_corrected_py(&self) -> anyhow::Result<f64> {
        Ok(self
            .get_energy_fuel_soc_corrected()
            .map_err(|err| anyhow!("{:?}", err))?
            .get::<si::joule>())
    }

    #[pyo3(name = "walk")]
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[staticmethod]
    #[pyo3(name = "valid")]
    fn valid_py() -> Self {
        Self::valid()
    }

    #[pyo3(name = "extend_path")]
    pub fn extend_path_py(
        &mut self,
        network_file_path: String,
        link_path: Vec<LinkIdx>,
    ) -> anyhow::Result<()> {
        let network = Vec::<Link>::from_file(network_file_path, false).unwrap();

        self.extend_path_tpc(&network, &link_path)?;
        Ok(())
    }

    #[pyo3(name = "walk_timed_path")]
    pub fn walk_timed_path_py(
        &mut self,
        network: &Bound<PyAny>,
        timed_path: &Bound<PyAny>,
    ) -> anyhow::Result<()> {
        let network = match network.extract::<Network>() {
            Ok(n) => n,
            Err(_) => {
                let n = network
                    .extract::<Vec<Link>>()
                    .map_err(|_| anyhow!("{}", format_dbg!()))?;
                Network(Default::default(), n)
            }
        };

        let timed_path = match timed_path.extract::<TimedLinkPath>() {
            Ok(tp) => tp,
            Err(_) => {
                let tp = timed_path
                    .extract::<Vec<LinkIdxTime>>()
                    .map_err(|_| anyhow!("{}", format_dbg!()))?;
                TimedLinkPath(tp)
            }
        };
        self.walk_timed_path(&network, timed_path)
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

pub struct SpeedLimitTrainSimBuilder {
    pub train_id: String,
    pub origs: Vec<Location>,
    pub dests: Vec<Location>,
    pub loco_con: Consist,
    /// Number of railcars by type on the train
    pub n_cars_by_type: HashMap<String, u32>,
    pub state: TrainState,
    pub train_res: TrainRes,
    pub path_tpc: PathTpc,
    pub fric_brake: FricBrake,
    pub save_interval: Option<usize>,
    pub simulation_days: Option<i32>,
    pub scenario_year: Option<i32>,
    /// Time-dependent temperature at sea level that can be corrected for altitude using a standard model
    pub temp_trace: Option<TemperatureTrace>,
}

impl From<SpeedLimitTrainSimBuilder> for SpeedLimitTrainSim {
    fn from(value: SpeedLimitTrainSimBuilder) -> Self {
        SpeedLimitTrainSim {
            train_id: value.train_id,
            origs: value.origs,
            dests: value.dests,
            loco_con: value.loco_con,
            n_cars_by_type: value.n_cars_by_type,
            state: value.state,
            train_res: value.train_res,
            path_tpc: value.path_tpc,
            braking_points: Default::default(),
            fric_brake: value.fric_brake,
            history: Default::default(),
            save_interval: value.save_interval,
            simulation_days: value.simulation_days,
            scenario_year: value.scenario_year,
            temp_trace: value.temp_trace,
        }
    }
}

impl SpeedLimitTrainSim {
    /// Returns the scaling factor to be used when converting partial-year
    /// simulations to a full year of output metrics.
    pub fn get_scaling_factor(&self, annualize: bool) -> f64 {
        if annualize {
            match self.simulation_days {
                Some(val) => 365.25 / val as f64,
                None => 365.25,
            }
        } else {
            1.0
        }
    }

    pub fn get_kilometers(&self, annualize: bool) -> anyhow::Result<f64> {
        Ok(self
            .state
            .total_dist
            .get_fresh(|| format_dbg!())?
            .get::<si::kilometer>()
            * self.get_scaling_factor(annualize))
    }

    pub fn get_megagram_kilometers(&self, annualize: bool) -> anyhow::Result<f64> {
        Ok(self
            .state
            .mass_freight
            .get_fresh(|| format_dbg!())?
            .get::<si::megagram>()
            * self
                .state
                .total_dist
                .get_fresh(|| format_dbg!())?
                .get::<si::kilometer>()
            * self.get_scaling_factor(annualize))
    }

    pub fn get_car_kilometers(&self, annualize: bool) -> anyhow::Result<f64> {
        let n_cars = self.get_cars_moved(annualize);
        // Note: n_cars already includes an annualization scaling factor; no need to multiply twice.
        Ok(self
            .state
            .total_dist
            .get_fresh(|| format_dbg!())?
            .get::<si::kilometer>()
            * n_cars)
    }

    pub fn get_cars_moved(&self, annualize: bool) -> f64 {
        let n_cars: f64 = self.n_cars_by_type.values().fold(0, |acc, n| *n + acc) as f64;
        n_cars * self.get_scaling_factor(annualize)
    }

    pub fn get_res_kilometers(&mut self, annualize: bool) -> anyhow::Result<f64> {
        let n_res = self.loco_con.n_res_equipped() as f64;
        Ok(self
            .state
            .total_dist
            .get_fresh(|| format_dbg!())?
            .get::<si::kilometer>()
            * n_res
            * self.get_scaling_factor(annualize))
    }

    pub fn get_non_res_kilometers(&mut self, annualize: bool) -> anyhow::Result<f64> {
        let n_res = self.loco_con.n_res_equipped() as usize;
        Ok(self
            .state
            .total_dist
            .get_fresh(|| format_dbg!())?
            .get::<si::kilometer>()
            * ((self.loco_con.loco_vec.len() - n_res) as f64)
            * self.get_scaling_factor(annualize))
    }

    pub fn get_energy_fuel(&self, annualize: bool) -> anyhow::Result<si::Energy> {
        Ok(self.loco_con.get_energy_fuel()? * self.get_scaling_factor(annualize))
    }

    /// Returns total fuel and fuel-equivalent battery energy used for consist
    pub fn get_energy_fuel_soc_corrected(&self) -> anyhow::Result<si::Energy> {
        if self.save_interval != Some(1) && self.history.is_empty() {
            bail!("Expected `save_interval = Some(1)` and non-empty history");
        }

        let eta_eng_mean_consist = {
            let mut energy_fuel_consist = si::Energy::ZERO;
            let mut energy_fc_shaft_consist = si::Energy::ZERO;
            for loco in self.loco_con.loco_vec.clone() {
                if let Some(fc) = loco.fuel_converter() {
                    energy_fuel_consist += *fc.state.energy_fuel.get_fresh(|| format_dbg!())?;
                    energy_fc_shaft_consist +=
                        *fc.state.energy_shaft.get_fresh(|| format_dbg!())?;
                }
            }
            energy_fc_shaft_consist / energy_fuel_consist
        };

        let mut loco_con_res_fuel_equiv = si::Energy::ZERO;
        for loco in self.loco_con.loco_vec.clone() {
            if let Some(res) = loco.reversible_energy_storage() {
                let delta_soc: si::Ratio = *res
                    .history
                    .soc
                    .last()
                    .unwrap()
                    .get_fresh(|| format_dbg!())?
                    - *res
                        .history
                        .soc
                        .first()
                        .unwrap()
                        .get_fresh(|| format_dbg!())?;

                // net energy at each time step
                let mut d_energy_elec: Vec<si::Energy> = vec![];
                let mut d_energy_chem: Vec<si::Energy> = vec![];
                for x in res.history.energy_out_electrical.windows(2) {
                    d_energy_elec.push(
                        *x[1].get_fresh(|| format_dbg!())? - *x[0].get_fresh(|| format_dbg!())?,
                    )
                }
                for x in res.history.energy_out_chemical.windows(2) {
                    d_energy_chem.push(
                        *x[1].get_fresh(|| format_dbg!())? - *x[0].get_fresh(|| format_dbg!())?,
                    )
                }

                let loco_res_fuel_equiv: si::Energy = if delta_soc < si::Ratio::ZERO {
                    let energy_elec_pos: si::Energy =
                        d_energy_elec.iter().fold(si::Energy::ZERO, |acc, curr| {
                            if *curr >= si::Energy::ZERO {
                                acc + *curr
                            } else {
                                acc
                            }
                        });
                    let energy_chem_pos: si::Energy =
                        d_energy_chem.iter().fold(si::Energy::ZERO, |acc, curr| {
                            if *curr >= si::Energy::ZERO {
                                acc + *curr
                            } else {
                                acc
                            }
                        });
                    -delta_soc * res.energy_capacity_usable() * energy_elec_pos
                        / energy_chem_pos
                        / eta_eng_mean_consist
                } else {
                    let energy_elec_neg: si::Energy =
                        d_energy_elec.iter().fold(si::Energy::ZERO, |acc, curr| {
                            if *curr <= si::Energy::ZERO {
                                acc + *curr
                            } else {
                                acc
                            }
                        });
                    let energy_chem_neg: si::Energy =
                        d_energy_chem.iter().fold(si::Energy::ZERO, |acc, curr| {
                            if *curr <= si::Energy::ZERO {
                                acc + *curr
                            } else {
                                acc
                            }
                        });
                    -delta_soc * res.energy_capacity_usable() * energy_elec_neg
                        / energy_chem_neg
                        / eta_eng_mean_consist
                };
                loco_con_res_fuel_equiv += loco_res_fuel_equiv;
            }
        }

        Ok(loco_con_res_fuel_equiv
            + self
                .loco_con
                .get_energy_fuel()
                .with_context(|| format_dbg!())?)
    }

    pub fn get_net_energy_res(&self, annualize: bool) -> anyhow::Result<si::Energy> {
        Ok(self.loco_con.get_net_energy_res()? * self.get_scaling_factor(annualize))
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        self.loco_con.set_save_interval(save_interval);
        self.fric_brake.save_interval = save_interval;
    }
    pub fn get_save_interval(&self) -> Option<usize> {
        self.save_interval
    }

    pub fn extend_path_tpc(
        &mut self,
        network: &[Link],
        link_path: &[LinkIdx],
    ) -> anyhow::Result<()> {
        self.path_tpc
            .extend(network, link_path)
            .with_context(|| format_dbg!())?;
        self.recalc_braking_points()
            .with_context(|| format_dbg!())?;
        Ok(())
    }
    pub fn clear_path(&mut self) {
        // let link_point_del = self.path_tpc.clear(self.state.offset_back);
        // self.train_res.fix_cache(&link_point_del);
    }

    pub fn finish(&mut self) {
        self.path_tpc.finish()
    }
    pub fn is_finished(&self) -> bool {
        self.path_tpc.is_finished()
    }
    pub fn offset_begin(&self) -> si::Length {
        self.path_tpc.offset_begin()
    }
    pub fn offset_end(&self) -> si::Length {
        self.path_tpc.offset_end()
    }
    pub fn link_idx_last(&self) -> Option<&LinkIdx> {
        self.path_tpc.link_idx_last()
    }
    pub fn link_points(&self) -> &[LinkPoint] {
        self.path_tpc.link_points()
    }

    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        self.loco_con
            .state
            .pwr_cat_lim
            .mark_fresh(|| format_dbg!())?;
        // set catenary power limit
        // self.loco_con.set_cat_power_limit(
        //     &self.path_tpc,
        //     *self.state.offset.get_fresh(|| format_dbg!())?,
        // )?;
        // set aux power for the consist
        timer!(self
            .loco_con
            .set_pwr_aux(Some(true))
            .with_context(|| format_dbg!())?);

        let elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)> =
            if let Some(tt) = &self.temp_trace {
                Some((
                    *self.state.elev_front.get_stale(|| format_dbg!())?,
                    tt.get_temp_at_time_and_elev(
                        *self.state.time.get_stale(|| format_dbg!())?,
                        *self.state.elev_front.get_stale(|| format_dbg!())?,
                    )
                    .with_context(|| format_dbg!())?,
                ))
            } else {
                None
            };

        self.state.dt.mark_fresh(|| format_dbg!())?;

        // set the maximum power out based on dt.
        timer!(self
            .loco_con
            .set_curr_pwr_max_out(
                None,
                elev_and_temp,
                Some(self.state.mass_compound().with_context(|| format_dbg!())?),
                Some(*self.state.speed.get_stale(|| format_dbg!())?),
                *self.state.dt.get_fresh(|| format_dbg!())?,
            )
            .with_context(|| format_dbg!())?);
        // calculate new resistance
        timer!(self
            .train_res
            .update_res(&mut self.state, &self.path_tpc, &Dir::Fwd)
            .with_context(|| format_dbg!())?);
        timer!(set_link_and_offset(&mut self.state, &self.path_tpc)?);
        // solve the required power
        timer!(self.solve_required_pwr().with_context(|| format_dbg!())?);

        timer!(self
            .loco_con
            .solve_energy_consumption(
                *self.state.pwr_whl_out.get_fresh(|| format_dbg!())?,
                Some(self.state.mass_compound().with_context(|| format_dbg!())?),
                Some(*self.state.speed.get_fresh(|| format_dbg!())?),
                *self.state.dt.get_fresh(|| format_dbg!())?,
                Some(true),
            )
            .with_context(|| format_dbg!())?);

        timer!(self.set_cumulative(
            *self.state.dt.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?);

        Ok(())
    }

    /// Walks until getting to the end of the path
    fn walk_internal(&mut self) -> anyhow::Result<()> {
        while *self.state.offset.get_fresh(|| format_dbg!())?
            < self.path_tpc.offset_end() - 1000.0 * uc::FT
            || (*self.state.offset.get_fresh(|| format_dbg!())? < self.path_tpc.offset_end()
                && *self.state.speed.get_fresh(|| format_dbg!())? != si::Velocity::ZERO)
        {
            self.step(|| format_dbg!())?;
        }
        Ok(())
    }

    /// Iterates `save_state` and `step` until offset >= final offset --
    /// i.e. moves train forward until it reaches destination.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state(|| format_dbg!())?;
        self.walk_internal()?;
        Ok(())
    }

    /// Iterates `save_state` and `step` until offset >= final offset --
    /// i.e. moves train forward and extends path TPC until it reaches destination.
    pub fn walk_timed_path<P: AsRef<[LinkIdxTime]>, Q: AsRef<[Link]>>(
        &mut self,
        network: Q,
        timed_path: P,
    ) -> anyhow::Result<()> {
        let network = network.as_ref();
        let timed_path = timed_path.as_ref();
        if timed_path.is_empty() {
            bail!("Timed path cannot be empty!");
        }

        // Get train length to check if we need additional previous links
        let train_length = *self.state.length.get_fresh(|| format_dbg!())?;
        let first_link_idx = timed_path[0].link_idx;
        let first_link = &network[first_link_idx.idx()];

        // Collect previous links if the train is longer than the first link
        let mut initial_link_path = Vec::with_capacity(16);
        if first_link.length < train_length {
            let mut total_length = first_link.length;
            let mut current_link_idx = first_link_idx;

            // Keep adding previous links until we have enough length for the train
            while total_length < train_length {
                let current_link = &network[current_link_idx.idx()];

                // Get the previous link (prefer idx_prev over idx_prev_alt)
                let prev_link_idx = if current_link.idx_prev.is_real() {
                    current_link.idx_prev
                } else if current_link.idx_prev_alt.is_real() {
                    current_link.idx_prev_alt
                } else {
                    // No more previous links available
                    bail!(
                        "Train too long for initial route in walk_timed_path: train length ({:.2} m) exceeds available \
                        track length ({:.2} m). The first link and its previous links do not \
                        provide enough length to place the train. Consider reducing train length.",
                        train_length.get::<si::meter>(),
                        total_length.get::<si::meter>(),
                    );
                };

                let prev_link = &network[prev_link_idx.idx()];
                total_length += prev_link.length;
                initial_link_path.push(prev_link_idx);
                current_link_idx = prev_link_idx;
            }

            // Reverse the path so it goes from previous links to origin
            initial_link_path.reverse();
        }

        self.save_state(|| format_dbg!())?;

        // If we have previous links to add, extend the path with them first
        if !initial_link_path.is_empty() {
            self.extend_path_tpc(network, &initial_link_path)
                .with_context(|| format!("{}\nExtending with previous links for train length", format_dbg!()))?;
        }

        let mut idx_prev = 0;
        while idx_prev != timed_path.len() - 1 {
            let mut idx_next = idx_prev + 1;
            while idx_next + 1 < timed_path.len() - 1
                && timed_path[idx_next].time < *self.state.time.get_fresh(|| format_dbg!())?
            {
                idx_next += 1;
            }
            let time_extend = timed_path[idx_next - 1].time;
            self.extend_path_tpc(
                network,
                &timed_path[idx_prev..idx_next]
                    .iter()
                    .map(|x| x.link_idx)
                    .collect::<Vec<LinkIdx>>(),
            )
            .with_context(|| format!("{}\nidx_prev: {idx_prev}", format_dbg!()))?;
            idx_prev = idx_next;
            while *self.state.time.get_fresh(|| format_dbg!())? < time_extend {
                self.step(|| format_dbg!()).with_context(|| {
                    format!(
                        "{}\n`self.state.time`: {:?}\n`time_extend`: {:?}",
                        format_dbg!(),
                        self.state.time,
                        time_extend
                    )
                })?;
            }
        }

        // Chad thinks this finishes up the last stretch of the timed path but
        // does not rerun the whole thing
        self.walk_internal().with_context(|| format_dbg!())?;
        Ok(())
    }

    /// Sets power requirements based on:
    /// - rolling resistance
    /// - drag
    /// - inertia
    /// - target acceleration
    pub fn solve_required_pwr(&mut self) -> anyhow::Result<()> {
        let res_net = self.state.res_net().with_context(|| format_dbg!())?;

        // Verify that train can slow down -- if `self.state.res_net()`, which
        // includes grade, is negative (negative `res_net` means the downgrade
        // is steep enough that the train is overcoming bearing, rolling,
        // drag, ... all resistive forces to accelerate downhill in the absence
        // of any braking), and if `self.state.res_net()` is negative and has
        // a higher magnitude than `self.fric_brake.force_max`, then the train
        // cannot slow down.
        // TODO: dial this back to just show `self.state` via debug print
        ensure!(
            self.fric_brake.force_max + self.state.res_net()? > si::Force::ZERO,
            format!(
                "Insufficient braking force.\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                format_dbg!(self.fric_brake.force_max + self.state.res_net()? > si::Force::ZERO),
                format_dbg!(self.fric_brake.force_max),
                format_dbg!(self.state.res_net()?),
                format_dbg!(self.state.res_grade),
                format_dbg!(self.state.grade_front),
                format_dbg!(self.state.grade_back),
                format_dbg!(self.state.elev_front),
                format_dbg!(self.state.offset),
                format_dbg!(self.state.offset_back),
                format_dbg!(self.state.speed),
                format_dbg!(self.state.speed_limit),
                format_dbg!(self.state.speed_target),
                format_dbg!(self.state.time),
                format_dbg!(self.state.dt),
                format_dbg!(self.state.i),
                format_dbg!(self.state.total_dist),
                format_dbg!(self.state.link_idx_front),
                format_dbg!(self.state.offset_in_link)
            )
        );

        // TODO: Validate that this makes sense considering friction brakes
        // this figures out when to start braking in advance of a speed limit
        // drop.  Takes into account air brake dynamics. I have not reviewed
        // this code, but that is my understanding.
        let (speed_limit, speed_target) = self
            .braking_points
            .calc_speeds(
                *self.state.offset.get_stale(|| format_dbg!())?,
                *self.state.speed.get_stale(|| format_dbg!())?,
                self.fric_brake.ramp_up_time * self.fric_brake.ramp_up_coeff,
            )
            .with_context(|| format_dbg!())?;
        self.state
            .speed_limit
            .update(speed_limit, || format_dbg!())?;
        self.state
            .speed_target
            .update(speed_target, || format_dbg!())?;

        let f_applied_target = res_net
            + self.state.mass_compound().with_context(|| format_dbg!())?
                * (speed_target - *self.state.speed.get_stale(|| format_dbg!())?)
                / *self.state.dt.get_fresh(|| format_dbg!())?;

        // calculate the max positive tractive effort.  this is the same as set_speed_train_sim
        let pwr_pos_max = self
            .loco_con
            .state
            .pwr_out_max
            .get_fresh(|| format_dbg!())?
            .min(
                si::Power::ZERO.max(
                    // NOTE: the effect of rate may already be accounted for in this snippet
                    // from fuel_converter.rs:

                    // ```
                    // self.state.pwr_out_max = (self.state.pwr_brake
                    //     + (self.pwr_out_max / self.pwr_ramp_lag) * dt)
                    //     .min(self.pwr_out_max)
                    //     .max(self.pwr_out_max_init);
                    // ```
                    *self.state.pwr_whl_out.get_stale(|| format_dbg!())?
                        + *self
                            .loco_con
                            .state
                            .pwr_rate_out_max
                            .get_fresh(|| format_dbg!())?
                            // TODO check if this ought to be updated earlier so we can call `get_fresh` here
                            * *self.state.dt.get_fresh(|| format_dbg!())?,
                ),
            );

        // calculate the max braking that a consist can apply
        let pwr_neg_max = self
            .loco_con
            .state
            .pwr_dyn_brake_max
            .get_fresh(|| format_dbg!())?
            .max(si::Power::ZERO);
        ensure!(
            pwr_pos_max >= si::Power::ZERO,
            format_dbg!(pwr_pos_max >= si::Power::ZERO)
        );
        let time_per_mass = *self.state.dt.get_fresh(|| format_dbg!())?
            / self.state.mass_compound().with_context(|| format_dbg!())?;

        // Concept: calculate the final speed such that the worst case
        // (i.e. maximum) acceleration force does not exceed `power_max`
        // Base equation: m * (v_max - v_curr) / dt = p_max / v_max – f_res
        // figuring out how fast we can be be going by next timestep.
        let v_max = 0.5
            * (*self.state.speed.get_stale(|| format_dbg!())? - res_net * time_per_mass
                + ((*self.state.speed.get_stale(|| format_dbg!())? - res_net * time_per_mass)
                    * (*self.state.speed.get_stale(|| format_dbg!())? - res_net * time_per_mass)
                    + 4.0 * time_per_mass * pwr_pos_max)
                    .sqrt());

        // Final v_max value should also be bounded by speed_target
        // maximum achievable positive tractive force
        let f_pos_max = self
            .loco_con
            .force_max()?
            .min(pwr_pos_max / speed_target.min(v_max));
        // Verify that train has sufficient power to move
        if *self.state.speed.get_stale(|| format_dbg!())? < uc::MPH * 0.1 && f_pos_max <= res_net {
            let mut soc_vec: Vec<String> = vec![];
            self.loco_con
                .loco_vec
                .iter()
                .try_for_each(|loco| -> anyhow::Result<()> {
                    if let Some(res) = loco.reversible_energy_storage() {
                        soc_vec.push(
                            res.state
                                .soc
                                .get_fresh(|| format_dbg!())?
                                .get::<si::ratio>()
                                .format_eng(Some(5)),
                        );
                    }
                    Ok(())
                })?;
            bail!(
                "{}\nTrain does not have sufficient power to move!
\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}", // ,\nlink={:?}
                format_dbg!(),
                // force_max
                format!(
                    "force_max: {} N",
                    self.loco_con
                        .force_max()?
                        .get::<si::newton>()
                        .format_eng(Some(5))
                ),
                // force based on speed target
                format!(
                    "pwr_pos_max / speed_target.min(v_max): {} N",
                    (pwr_pos_max / speed_target.min(v_max))
                        .get::<si::newton>()
                        .format_eng(Some(5))
                ),
                // pwr_pos_max
                format!(
                    "pwr_pos_max: {} W",
                    pwr_pos_max.get::<si::watt>().format_eng(Some(5)),
                ),
                // SOC across all RES-equipped locomotives
                format!("SOCs: {:?}", soc_vec),
                // minimum allowable SOC across all RES-equipped locomotives
                format!(
                    "Minimum allowed SOCs: {:?}",
                    self.loco_con
                        .loco_vec
                        .iter()
                        .map(|loco| {
                            loco.reversible_energy_storage()
                                .map(|res| res.min_soc.get::<si::ratio>().format_eng(Some(5)))
                                .unwrap_or_else(|| "N/A".into())
                        })
                        .collect::<Vec<String>>()
                ),
                // grade at front of train
                format!(
                    "grade_front: {}",
                    self.state
                        .grade_front
                        .get_fresh(|| format_dbg!())?
                        .get::<si::ratio>()
                        .format_eng(Some(5))
                ),
                // grade at rear of train
                format!(
                    "grade_back: {}",
                    self.state
                        .grade_back
                        .get_fresh(|| format_dbg!())?
                        .get::<si::ratio>()
                        .format_eng(Some(5))
                ),
                format!(
                    "f_pos_max: {} N",
                    f_pos_max.get::<si::newton>().format_eng(Some(5))
                ),
                format!(
                    "res_net: {} N",
                    res_net.get::<si::newton>().format_eng(Some(5))
                ),
                format_dbg!(self.fric_brake.force_max),
                format_dbg!(self.state.res_grade),
                format_dbg!(self.state.elev_front),
                format_dbg!(self.state.elev_back),
                format_dbg!(self.state.offset),
                format_dbg!(self.state.offset_back),
                format_dbg!(self.state.speed),
                format_dbg!(self.state.speed_limit),
                format_dbg!(self.state.speed_target),
                format_dbg!(self.state.time),
                format_dbg!(self.state.dt),
                format_dbg!(self.state.i),
                format_dbg!(self.state.total_dist),
                format_dbg!(self.state.link_idx_front),
                format_dbg!(self.state.offset_in_link)
            )
        }

        // set the maximum friction braking force that is possible.
        self.fric_brake
            .set_cur_force_max_out(*self.state.dt.get_fresh(|| format_dbg!())?)
            .with_context(|| format_dbg!())?;

        // Transition speed between force and power limited negative traction
        // figure out the velocity where power and force limits coincide
        let v_neg_trac_lim: si::Velocity = *self
            .loco_con
            .state
            .pwr_dyn_brake_max
            .get_fresh(|| format_dbg!())?
            / self.loco_con.force_max()?;

        // TODO: Make sure that train handling rules for consist dynamic braking force limit is respected!
        // figure out how much dynamic braking can be used as regenerative
        // braking to recharge the [ReversibleEnergyStorage]
        let f_max_consist_regen_dyn: si::Force =
            if *self.state.speed.get_stale(|| format_dbg!())? > v_neg_trac_lim {
                // If there is enough braking to slow down at v_max
                let f_max_dyn_fast = *self
                    .loco_con
                    .state
                    .pwr_dyn_brake_max
                    .get_fresh(|| format_dbg!())?
                    / v_max;
                if res_net
                    + *self
                        .fric_brake
                        .state
                        .force_max_curr
                        .get_fresh(|| format_dbg!())?
                    + f_max_dyn_fast
                    >= si::Force::ZERO
                {
                    *self
                        .loco_con
                        .state
                        .pwr_dyn_brake_max
                        .get_fresh(|| format_dbg!())?
                        / v_max // self.state.speed
                } else {
                    f_max_dyn_fast
                }
            } else {
                self.loco_con.force_max()?
            };

        // total impetus force applied to control train speed
        // calculating the applied drawbar force based on targets and enforcing limits.
        let f_applied = f_pos_max.min(
            f_applied_target.max(
                -*self
                    .fric_brake
                    .state
                    .force_max_curr
                    .get_fresh(|| format_dbg!())?
                    - f_max_consist_regen_dyn,
            ),
        );

        // physics......
        let vel_change = time_per_mass * (f_applied - res_net);
        let vel_avg = *self.state.speed.get_stale(|| format_dbg!())? + 0.5 * vel_change;

        // updating states of the train.
        self.state
            .pwr_res
            .update(res_net * vel_avg, || format_dbg!())?;
        self.state.pwr_accel.update(
            self.state.mass_compound().with_context(|| format_dbg!())?
                / (2.0 * *self.state.dt.get_fresh(|| format_dbg!())?)
                * ((*self.state.speed.get_stale(|| format_dbg!())? + vel_change)
                    * (*self.state.speed.get_stale(|| format_dbg!())? + vel_change)
                    - *self.state.speed.get_stale(|| format_dbg!())?
                        * *self.state.speed.get_stale(|| format_dbg!())?),
            || format_dbg!(),
        )?;

        self.state.time.increment(
            *self.state.dt.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        self.state.offset.increment(
            *self.state.dt.get_fresh(|| format_dbg!())? * vel_avg,
            || format_dbg!(),
        )?;
        self.state.total_dist.increment(
            *self.state.dt.get_fresh(|| format_dbg!())? * vel_avg.abs(),
            || format_dbg!(),
        )?;

        let new_speed = *self.state.speed.get_stale(|| format_dbg!())? + vel_change;
        self.state.speed.update(
            if utils::almost_eq_uom(&new_speed, &speed_target, None) {
                speed_target
            } else {
                new_speed
            },
            || format_dbg!(),
        )?;

        let (f_consist, fric_brake_force): (si::Force, si::Force) = if f_applied >= si::Force::ZERO
        {
            // net positive traction is being exerted on train
            (f_applied, si::Force::ZERO)
        } else {
            // net negative traction is being exerted on train
            let f_consist = f_applied + *self.fric_brake.state.force.get_stale(|| format_dbg!())?;
            // If the friction brakes should be released, don't add power -- as
            // of 2025-06-10, trying to figure out what this means
            if f_consist >= si::Force::ZERO {
                (
                    si::Force::ZERO,
                    *self.fric_brake.state.force.get_stale(|| format_dbg!())?,
                )
            }
            // If the current friction brakes and consist regen + dyn can handle
            // things, don't add friction braking
            else if f_consist + f_max_consist_regen_dyn >= si::Force::ZERO {
                // TODO: this needs careful scrutiny
                (
                    f_consist,
                    *self.fric_brake.state.force.get_stale(|| format_dbg!())?,
                )
            }
            // If the friction braking must increase, max out the regen dyn first
            else {
                (
                    -f_max_consist_regen_dyn,
                    -(f_applied + f_max_consist_regen_dyn),
                )
            }
        };

        self.fric_brake
            .state
            .force
            .update(fric_brake_force, || format_dbg!())?;

        ensure!(
            utils::almost_le_uom(
                self.fric_brake.state.force.get_fresh(|| format_dbg!())?,
                self.fric_brake
                    .state
                    .force_max_curr
                    .get_fresh(|| format_dbg!())?,
                None
            ),
            "Too much force requested from friction brake! Req={:?}, max={:?}",
            self.fric_brake.state.force,
            self.fric_brake.state.force_max_curr,
        );

        let pwr_whl_out_unclipped = f_consist * *self.state.speed.get_fresh(|| format_dbg!())?;

        // this allows for float rounding error overshoot
        ensure!(
            utils::almost_le_uom(&pwr_whl_out_unclipped, &pwr_pos_max, Some(1.0e-7)),
            format!("{}\nPower wheel out is larger than max positive power! pwr_whl_out={:?}, pwr_pos_max={:?}",
            format_dbg!(utils::almost_le_uom(self.state.pwr_whl_out.get_fresh(|| format_dbg!())?, &pwr_pos_max, Some(1.0e-7))),
            self.state.pwr_whl_out.get_fresh(|| format_dbg!())?,
            pwr_pos_max)
        );
        ensure!(
            utils::almost_le_uom(&-pwr_whl_out_unclipped, &pwr_neg_max, Some(1.0e-7)),
            format!("{}\nPower wheel out is larger than max negative power! pwr_whl_out={:?}, pwr_neg_max={:?}
            {:?}\n{:?}\n{:?}\n{:?}",
            format_dbg!(utils::almost_le_uom(&-*self.state.pwr_whl_out.get_fresh(|| format_dbg!())?, &pwr_neg_max, Some(1.0e-7))),
            -*self.state.pwr_whl_out.get_fresh(|| format_dbg!())?,
            pwr_neg_max,
            self.state.speed.get_fresh(|| format_dbg!())?,
            *self.fric_brake.state.force .get_fresh(|| format_dbg!())?* *self.state.speed.get_fresh(|| format_dbg!())?,
            vel_change,
        res_net)
        );
        self.state.pwr_whl_out.update(
            pwr_whl_out_unclipped.max(-pwr_neg_max).min(pwr_pos_max),
            || format_dbg!(),
        )?;

        if *self.state.pwr_whl_out.get_fresh(|| format_dbg!())? >= 0. * uc::W {
            self.state.energy_whl_out_pos.increment(
                *self.state.pwr_whl_out.get_fresh(|| format_dbg!())?
                    * *self.state.dt.get_fresh(|| format_dbg!())?,
                || format_dbg!(),
            )?;
            self.state
                .energy_whl_out_neg
                .increment(si::Energy::ZERO, || format_dbg!())?;
        } else {
            self.state.energy_whl_out_neg.increment(
                -*self.state.pwr_whl_out.get_fresh(|| format_dbg!())?
                    * *self.state.dt.get_fresh(|| format_dbg!())?,
                || format_dbg!(),
            )?;
            self.state
                .energy_whl_out_pos
                .increment(si::Energy::ZERO, || format_dbg!())?;
        }

        Ok(())
    }

    fn recalc_braking_points(&mut self) -> anyhow::Result<()> {
        self.braking_points
            .recalc(
                &self.state,
                &self.fric_brake,
                &self.train_res,
                &self.path_tpc,
            )
            .with_context(|| format_dbg!())
    }
}

impl StateMethods for SpeedLimitTrainSim {}
impl CheckAndResetState for SpeedLimitTrainSim {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.state
            .check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.loco_con
            .check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.fric_brake
            .check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}
impl SetCumulative for SpeedLimitTrainSim {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        self.state
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.loco_con
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.fric_brake
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}
impl SaveState for SpeedLimitTrainSim {
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        if let Some(interval) = self.save_interval {
            if self
                .state
                .i
                .get_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?
                % interval
                == 0
            {
                self.history.push(self.state.clone());
                self.loco_con.save_state(|| format_dbg!())?;
                self.fric_brake.save_state(|| format_dbg!())?;
            }
        }
        Ok(())
    }
}

impl Step for SpeedLimitTrainSim {
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        let i = *self.state.i.get_fresh(|| format_dbg!())?;
        // NOTE: change this if length becomes dynamic
        self.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.state
            .i
            .increment(1, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.loco_con
            .step(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.fric_brake
            .step(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.state.length.mark_fresh(|| format_dbg!())?;
        self.state.mass_static.mark_fresh(|| format_dbg!())?;
        self.state.mass_rot.mark_fresh(|| format_dbg!())?;
        self.state.mass_freight.mark_fresh(|| format_dbg!())?;
        #[cfg(feature = "timer")]
        println!("\n");
        timer!(self
            .solve_step()
            .with_context(|| format!("{}\ntime step: {}", loc(), i))?);
        self.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}

impl Init for SpeedLimitTrainSim {
    fn init(&mut self) -> Result<(), Error> {
        self.origs.init()?;
        self.dests.init()?;
        self.loco_con.init()?;
        self.state.init()?;
        self.train_res.init()?;
        self.path_tpc.init()?;
        self.braking_points.init()?;
        self.fric_brake.init()?;
        self.history.init()?;
        Ok(())
    }
}
impl SerdeAPI for SpeedLimitTrainSim {}
impl Default for SpeedLimitTrainSim {
    fn default() -> Self {
        let mut slts = Self {
            train_id: Default::default(),
            origs: Default::default(),
            dests: Default::default(),
            loco_con: Default::default(),
            n_cars_by_type: Default::default(),
            state: TrainState::valid(),
            train_res: TrainRes::valid(),
            path_tpc: PathTpc::default(),
            braking_points: Default::default(),
            fric_brake: Default::default(),
            history: Default::default(),
            temp_trace: Default::default(),
            save_interval: None,
            simulation_days: None,
            scenario_year: None,
        };
        slts.set_save_interval(None);
        slts.init().unwrap();
        slts
    }
}

impl Valid for SpeedLimitTrainSim {
    fn valid() -> Self {
        let mut train_sim = Self {
            path_tpc: PathTpc::valid(),
            ..Default::default()
        };
        train_sim.recalc_braking_points().unwrap();
        train_sim
    }
}

pub fn speed_limit_train_sim_fwd() -> SpeedLimitTrainSim {
    let mut speed_limit_train_sim = SpeedLimitTrainSim::valid();
    speed_limit_train_sim.path_tpc = PathTpc::new(TrainParams::valid());
    speed_limit_train_sim.origs = vec![
        Location {
            location_id: "Barstow".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(96),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
        Location {
            location_id: "Barstow".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(634),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
    ];
    speed_limit_train_sim.dests = vec![
        Location {
            location_id: "Stockton".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(288),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
        Location {
            location_id: "Stockton".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(826),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
    ];
    speed_limit_train_sim
}

pub fn speed_limit_train_sim_rev() -> SpeedLimitTrainSim {
    let mut sltsr = speed_limit_train_sim_fwd();
    std::mem::swap(&mut sltsr.origs, &mut sltsr.dests);
    sltsr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    // TODO: Add more SpeedLimitTrainSim cases
    impl Cases for SpeedLimitTrainSim {}

    #[test]
    fn test_to_from_file_for_train_sim() {
        let ts0 = SOLVED_SPEED_LIM_TRAIN_SIM.clone();
        let tempdir = tempfile::tempdir().unwrap();

        // // test to and from file for yaml
        let temp_yaml_path = tempdir.path().join("ts.yaml");
        ts0.to_file(temp_yaml_path.clone()).unwrap();
        let ts_yaml = crate::prelude::SpeedLimitTrainSim::from_file(temp_yaml_path, false).unwrap();

        // `to_yaml` is probably needed to get around problems with NAN
        assert_eq!(ts_yaml.to_yaml().unwrap(), ts0.to_yaml().unwrap());

        // test to and from file for msgpack
        let temp_msgpack_path = tempdir.path().join("ts.msgpack");
        ts0.to_file(temp_msgpack_path.clone()).unwrap();
        let ts_msgpack = SpeedLimitTrainSim::from_file(temp_msgpack_path, false).unwrap();

        // `to_yaml` is probably needed to get around problems with NAN
        assert_eq!(ts_msgpack.to_yaml().unwrap(), ts0.to_yaml().unwrap());
    }

    lazy_static! {
        static ref SOLVED_SPEED_LIM_TRAIN_SIM: crate::prelude::SpeedLimitTrainSim = {
            let mut ts = crate::prelude::SpeedLimitTrainSim::valid();
            ts.init().unwrap();
            ts.walk().unwrap();
            ts
        };
    }
}
