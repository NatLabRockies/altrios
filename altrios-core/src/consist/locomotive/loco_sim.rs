//! Module for standalone simulation of locomotive powertrains

use rayon::prelude::*;

use super::locomotive::Locomotive;
use crate::consist::LocoTrait;
use crate::imports::*;

#[cfg(doc)]
use super::locomotive::HybridLoco;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Container
pub struct PowerTrace {
    /// simulation time
    pub time: Vec<si::Time>,
    /// simulation power
    pub pwr: Vec<si::Power>,
    /// Whether engine is on
    pub engine_on: Vec<Option<bool>>,
    #[serde(default)]
    /// Speed, needed only if simulating [HybridLoco]  
    pub train_speed: Vec<si::Velocity>,
    /// Train mass, needed only if simulating [HybridLoco]
    pub train_mass: Option<si::Mass>,
}

#[pyo3_api]
impl PowerTrace {
    #[staticmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(pathstr: String) -> anyhow::Result<Self> {
        Self::from_csv_file(&pathstr)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl Init for PowerTrace {}
impl SerdeAPI for PowerTrace {}

impl PowerTrace {
    pub fn empty() -> Self {
        Self {
            time: Vec::new(),
            pwr: Vec::new(),
            engine_on: Vec::new(),
            train_speed: Vec::new(),
            train_mass: None,
        }
    }

    pub fn dt_at_i(&self, i: usize) -> anyhow::Result<si::Time> {
        ensure!(i > 0);
        Ok(*self.time.get(i).with_context(|| format_dbg!(i))?
            - *self.time.get(i - 1).with_context(|| format_dbg!(i - 1))?)
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, pt_element: PowerTraceElement) {
        self.time.push(pt_element.time);
        self.pwr.push(pt_element.pwr);
        self.engine_on.push(pt_element.engine_on);
        if let Some(train_speed) = pt_element.train_speed {
            self.train_speed.push(train_speed);
        }
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or_else(|| self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.pwr = self.pwr[start_idx..end_idx].to_vec();
        self.engine_on = self.engine_on[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Load cycle from csv file
    pub fn from_csv_file(pathstr: &str) -> Result<Self, anyhow::Error> {
        let pathbuf = PathBuf::from(&pathstr);

        // create empty cycle to be populated
        let mut pt = Self::empty();

        let file = File::open(pathbuf)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let pt_elem: PowerTraceElement = result?;
            pt.push(pt_elem);
        }
        if pt.is_empty() {
            bail!("Invalid PowerTrace file; PowerTrace is empty")
        } else {
            Ok(pt)
        }
    }
}

impl Default for PowerTrace {
    fn default() -> Self {
        let pwr_max_watts = 1.5e6;
        let pwr_watts_ramp: Vec<f64> = Vec::linspace(0., pwr_max_watts, 300);
        let mut pwr_watts = pwr_watts_ramp.clone();
        pwr_watts.append(&mut vec![pwr_max_watts; 100]);
        pwr_watts.append(&mut pwr_watts_ramp.iter().rev().copied().collect());
        let time_s: Vec<f64> = (0..pwr_watts.len()).map(|x| x as f64).collect();
        let time_len = time_s.len();
        let mut pt = Self {
            time: time_s.iter().map(|t| *t * uc::S).collect(),
            pwr: pwr_watts.iter().map(|p| *p * uc::W).collect(),
            engine_on: vec![Some(true); time_len],
            train_speed: vec![10.0 * uc::MPH; time_len],
            train_mass: Some(1e6 * uc::LB),
        };
        pt.init().unwrap();
        pt
    }
}

/// Element of [PowerTrace].  Used for vec-like operations.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct PowerTraceElement {
    /// simulation time
    time: si::Time,
    /// simulation power
    pwr: si::Power,
    /// Whether engine is on
    engine_on: Option<bool>,
    /// speed at time step
    train_speed: Option<si::Velocity>,
}

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct for simulating operation of a standalone locomotive.
pub struct LocomotiveSimulation {
    pub loco_unit: Locomotive,
    pub power_trace: PowerTrace,
}

#[pyo3_api]
impl LocomotiveSimulation {
    #[new]
    #[pyo3(signature = (loco_unit, power_trace, save_interval=None))]
    fn __new__(
        loco_unit: Locomotive,
        power_trace: PowerTrace,
        save_interval: Option<usize>,
    ) -> Self {
        Self::new(loco_unit, power_trace, save_interval)
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
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.set_save_interval(save_interval);
        Ok(())
    }

    #[pyo3(name = "get_save_interval")]
    fn get_save_interval_py(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.loco_unit.get_save_interval())
    }

    #[pyo3(name = "trim_failed_steps")]
    fn trim_failed_steps_py(&mut self) -> anyhow::Result<()> {
        self.trim_failed_steps()?;
        Ok(())
    }
}

impl LocomotiveSimulation {
    pub fn new(
        loco_unit: Locomotive,
        power_trace: PowerTrace,
        save_interval: Option<usize>,
    ) -> Self {
        let mut loco_sim = Self {
            loco_unit,
            power_trace,
        };
        loco_sim.loco_unit.set_save_interval(save_interval);
        loco_sim
    }

    /// Trims off any portion of the trip that failed to run
    pub fn trim_failed_steps(&mut self) -> anyhow::Result<()> {
        if *self.loco_unit.state.i.get_stale(|| format_dbg!())? <= 1 {
            bail!("`walk` method has not proceeded past first time step.")
        }
        self.power_trace.trim(
            None,
            Some(*self.loco_unit.state.i.get_stale(|| format_dbg!())?),
        )?;

        Ok(())
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.loco_unit.set_save_interval(save_interval);
    }

    pub fn get_save_interval(&self) -> Option<usize> {
        self.loco_unit.get_save_interval()
    }

    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        // linear aux model
        let i = *self.loco_unit.state.i.get_fresh(|| format_dbg!())?;
        let engine_on: Option<bool> = self
            .power_trace
            .engine_on
            .get(i)
            .cloned()
            .with_context(|| format_dbg!())?;
        self.loco_unit.set_pwr_aux(engine_on)?;
        let train_mass = self.power_trace.train_mass;
        let train_speed = if !self.power_trace.train_speed.is_empty() {
            Some(self.power_trace.train_speed[i])
        } else {
            None
        };
        let dt = self.power_trace.dt_at_i(i).with_context(|| format_dbg!())?;
        self.loco_unit
            .set_curr_pwr_max_out(None, None, train_mass, train_speed, dt)?;
        let pwr_out_req = self
            .power_trace
            .pwr
            .get(i)
            .cloned()
            .with_context(|| format_dbg!())?;
        self.solve_energy_consumption(pwr_out_req, dt, engine_on, train_mass, train_speed)?;
        ensure!(
            utils::almost_eq_uom(
                &pwr_out_req,
                self.loco_unit.state.pwr_out.get_fresh(|| format_dbg!())?,
                None
            ),
            format_dbg!(
                (utils::almost_eq_uom(
                    &pwr_out_req,
                    self.loco_unit.state.pwr_out.get_fresh(|| format_dbg!())?,
                    None
                ))
            )
        );
        self.set_cumulative(dt, || format_dbg!())?;
        Ok(())
    }

    /// Iterates `save_state` and `step` through all time steps.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state(|| format_dbg!())?;
        loop {
            if *self.loco_unit.state.i.get_fresh(|| format_dbg!())? > self.power_trace.len() - 2 {
                break;
            }
            self.step(|| format_dbg!())?;
        }
        ensure!(*self.loco_unit.state.i.get_fresh(|| format_dbg!())? == self.power_trace.len() - 1);
        Ok(())
    }

    /// Solves for fuel and RES consumption
    /// Arguments:
    /// ----------
    /// pwr_out_req: float, output brake power required from fuel converter.
    /// dt: current time step size
    /// engine_on: whether or not locomotive is active
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: Option<bool>,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<()> {
        self.loco_unit.solve_energy_consumption(
            pwr_out_req,
            dt,
            engine_on,
            train_mass,
            train_speed,
        )?;
        Ok(())
    }
}

impl Step for LocomotiveSimulation {
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.check_and_reset(|| format_dbg!())?;
        self.loco_unit.step(|| format_dbg!())?;
        let i = *self.loco_unit.state.i.get_fresh(|| format_dbg!())?;

        self.solve_step()
            .with_context(|| format!("{}\ntime step: {}", loc(), i))?;
        self.save_state(|| format_dbg!())?;
        Ok(())
    }
}

impl SaveState for LocomotiveSimulation {
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.loco_unit
            .save_state(|| format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}

impl CheckAndResetState for LocomotiveSimulation {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.loco_unit
            .check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))
    }
}

impl SetCumulative for LocomotiveSimulation {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        self.loco_unit
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))
    }
}

impl StateMethods for LocomotiveSimulation {}

impl Init for LocomotiveSimulation {
    fn init(&mut self) -> Result<(), Error> {
        self.loco_unit.init()?;
        self.power_trace.init()?;
        Ok(())
    }
}
impl SerdeAPI for LocomotiveSimulation {}

impl Default for LocomotiveSimulation {
    fn default() -> Self {
        let power_trace = PowerTrace::default();
        let loco_unit = Locomotive::default();
        let mut ls = Self::new(loco_unit, power_trace, None);
        ls.init().unwrap();
        ls
    }
}

#[serde_api]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct LocomotiveSimulationVec(pub Vec<LocomotiveSimulation>);
impl LocomotiveSimulationVec {
    pub fn new(value: Vec<LocomotiveSimulation>) -> Self {
        Self(value)
    }
}

#[pyo3_api]
impl LocomotiveSimulationVec {
    #[new]
    /// Rust-defined `__new__` magic method for Python used exposed via PyO3.
    fn __new__(v: Vec<LocomotiveSimulation>) -> Self {
        Self(v)
    }

    #[pyo3(name = "walk")]
    #[pyo3(signature = (b_parallelize=None))]
    /// Exposes `walk` to Python.
    fn walk_py(&mut self, b_parallelize: Option<bool>) -> anyhow::Result<()> {
        let b_par = b_parallelize.unwrap_or(false);
        self.walk(b_par)
    }
}

impl Init for LocomotiveSimulationVec {
    fn init(&mut self) -> Result<(), Error> {
        self.0.iter_mut().try_for_each(|l| l.init())?;
        Ok(())
    }
}
impl SerdeAPI for LocomotiveSimulationVec {}
impl Default for LocomotiveSimulationVec {
    fn default() -> Self {
        Self(vec![LocomotiveSimulation::default(); 3])
    }
}

impl LocomotiveSimulationVec {
    /// Calls `walk` for each locomotive in vec.
    pub fn walk(&mut self, parallelize: bool) -> anyhow::Result<()> {
        if parallelize {
            self.0
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(i, loco_sim)| {
                    loco_sim
                        .walk()
                        .map_err(|err| err.context(format!("loco_sim idx:{i}")))
                })?;
        } else {
            self.0
                .iter_mut()
                .enumerate()
                .try_for_each(|(i, loco_sim)| {
                    loco_sim
                        .walk()
                        .map_err(|err| err.context(format!("loco_sim idx:{i}")))
                })?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::{Locomotive, LocomotiveSimulation, LocomotiveSimulationVec, PowerTrace};
    use crate::consist::locomotive::PowertrainType;

    #[test]
    fn test_loco_sim_vec_par() {
        let mut loco_sim_vec = LocomotiveSimulationVec::default();
        loco_sim_vec.walk(true).unwrap();
    }

    #[test]
    fn test_loco_sim_vec_ser() {
        let mut loco_sim_vec = LocomotiveSimulationVec::default();
        loco_sim_vec.walk(false).unwrap();
    }

    #[test]
    fn test_power_trace_serialize() {
        let pt = PowerTrace::default();
        let json = serde_json::to_string(&pt).unwrap();
        println!("{json}");
        let new_pt: PowerTrace = serde_json::from_str(json.as_str()).unwrap();
        println!("{new_pt:?}");
    }

    #[test]
    fn test_power_trace_serialize_yaml() {
        let pt = PowerTrace::default();
        let yaml = serde_yaml::to_string(&pt).unwrap();
        println!("{yaml}");
        let new_pt: PowerTrace = serde_yaml::from_str(yaml.as_str()).unwrap();
        println!("{new_pt:?}");
    }

    #[test]
    fn test_conventional_locomotive_sim() {
        let cl = Locomotive::default();
        let pt = PowerTrace::default();
        let mut loco_sim = LocomotiveSimulation::new(cl, pt, None);
        loco_sim.walk().unwrap();
    }

    #[test]
    fn test_hybrid_locomotive_sim() {
        let hel = Locomotive::default_hybrid_electric_loco();
        let pt = PowerTrace::default();
        let mut loco_sim = LocomotiveSimulation::new(hel, pt, None);
        loco_sim.walk().unwrap();
    }

    #[test]
    fn test_battery_locomotive_sim() {
        let bel = Locomotive::default_battery_electric_loco();
        let pt = PowerTrace::default();
        let mut loco_sim = LocomotiveSimulation::new(bel, pt, None);
        loco_sim.walk().unwrap();
    }

    #[test]
    fn test_set_save_interval() {
        let mut ls = LocomotiveSimulation::default();

        assert!(ls.get_save_interval().is_none());
        assert!(ls.loco_unit.get_save_interval().is_none());
        assert!(match &ls.loco_unit.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                loco.fc.save_interval
            }
            _ => None,
        }
        .is_none());

        ls.set_save_interval(Some(1));

        assert_eq!(ls.get_save_interval(), Some(1));
        assert_eq!(ls.loco_unit.get_save_interval(), Some(1));
        assert_eq!(
            match &ls.loco_unit.loco_type {
                PowertrainType::ConventionalLoco(loco) => {
                    loco.fc.save_interval
                }
                _ => None,
            },
            Some(1)
        );

        ls.walk().unwrap();

        assert_eq!(ls.get_save_interval(), Some(1));
        assert_eq!(ls.loco_unit.get_save_interval(), Some(1));
        assert_eq!(
            match &ls.loco_unit.loco_type {
                PowertrainType::ConventionalLoco(loco) => {
                    loco.fc.save_interval
                }
                _ => None,
            },
            Some(1)
        );
    }

    #[test]
    fn test_power_trace_trim() {
        let pt = PowerTrace::default();
        let mut pt_test = pt.clone();

        assert!(pt == pt_test);
        pt_test.trim(None, None).unwrap();
        assert!(pt == pt_test);

        pt_test.trim(None, Some(10)).unwrap();
        assert!(pt_test != pt);
        assert!(pt_test.len() == 10);
    }
}
