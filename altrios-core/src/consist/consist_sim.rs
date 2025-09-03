use serde::{Deserialize, Serialize};

use crate::consist::locomotive::loco_sim::PowerTrace;
use crate::consist::Consist;
use crate::consist::LocoTrait;
use crate::imports::*;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct ConsistSimulation {
    pub loco_con: Consist,
    pub power_trace: PowerTrace,
}

#[pyo3_api]
impl ConsistSimulation {
    #[new]
    #[pyo3(signature = (consist, power_trace, save_interval=None))]
    fn __new__(consist: Consist, power_trace: PowerTrace, save_interval: Option<usize>) -> Self {
        Self::new(consist, power_trace, save_interval)
    }

    #[pyo3(name = "walk")]
    /// Exposes `walk` to python
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
        Ok(self.loco_con.get_save_interval())
    }

    #[pyo3(name = "trim_failed_steps")]
    fn trim_failed_steps_py(&mut self) -> anyhow::Result<()> {
        self.trim_failed_steps()?;
        Ok(())
    }
}

impl ConsistSimulation {
    pub fn new(consist: Consist, power_trace: PowerTrace, save_interval: Option<usize>) -> Self {
        let mut consist_sim = Self {
            loco_con: consist,
            power_trace,
        };
        consist_sim.loco_con.set_save_interval(save_interval);
        consist_sim
    }

    /// Trims off any portion of the trip that failed to run
    pub fn trim_failed_steps(&mut self) -> anyhow::Result<()> {
        if *self.loco_con.state.i.get_fresh(|| format_dbg!())? <= 1 {
            bail!("`walk` method has not proceeded past first time step.")
        }
        self.power_trace.trim(
            None,
            Some(*self.loco_con.state.i.get_fresh(|| format_dbg!())?),
        )?;

        Ok(())
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.loco_con.set_save_interval(save_interval);
    }

    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        self.loco_con
            .state
            .pwr_cat_lim
            .mark_fresh(|| format_dbg!())?;
        self.loco_con
            .set_pwr_aux(Some(true))
            .with_context(|| format_dbg!())?;
        let train_mass = self.power_trace.train_mass;
        let i = *self.loco_con.state.i.get_fresh(|| format_dbg!())?;
        let train_speed = if !self.power_trace.train_speed.is_empty() {
            Some(self.power_trace.train_speed[i])
        } else {
            None
        };
        let dt = self.power_trace.dt_at_i(i).with_context(|| format_dbg!())?;
        self.loco_con
            .set_curr_pwr_max_out(None, None, train_mass, train_speed, dt)
            .with_context(|| format_dbg!())?;
        self.solve_energy_consumption(
            self.power_trace.pwr[*self.loco_con.state.i.get_fresh(|| format_dbg!())?],
            train_mass,
            train_speed,
            dt,
        )
        .with_context(|| format_dbg!())?;
        self.set_cumulative(dt, || format_dbg!())?;
        Ok(())
    }

    /// Iterates step to solve all time steps.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state(|| format_dbg!())?;
        loop {
            if *self.loco_con.state.i.get_fresh(|| format_dbg!())? > self.power_trace.len() - 2 {
                break;
            }
            self.step(|| format_dbg!())?;
        }
        Ok(())
    }

    /// Solves for fuel and RES consumption
    /// Arguments:
    /// ----------
    /// pwr_out_req: float, output brake power required from fuel converter.
    /// dt: time step size
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // TODO:
        // - this function needs to be modified to accept catenary power limit(s) and/or the PathTPC instance
        // - figure out an algorith for apportioning catenary power among the locomotives in the consist
        // - make sure catenary power gets applied to charging
        self.loco_con.solve_energy_consumption(
            pwr_out_req,
            train_mass,
            train_speed,
            dt,
            Some(true),
        )?;
        Ok(())
    }
}

impl StateMethods for ConsistSimulation {}

impl SetCumulative for ConsistSimulation {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        self.loco_con
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}

impl CheckAndResetState for ConsistSimulation {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.loco_con
            .check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}

impl Step for ConsistSimulation {
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.loco_con
            .step(|| format!("{}\n{}", loc(), format_dbg!()))?;
        let i = *self
            .loco_con
            .state
            .i
            .get_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.solve_step()
            .with_context(|| format!("{}\ntime step: {}", loc(), i))?;
        self.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}

impl SaveState for ConsistSimulation {
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.loco_con
            .save_state(|| format!("{}\n{}", loc(), format_dbg!()))
    }
}

impl Init for ConsistSimulation {
    fn init(&mut self) -> Result<(), Error> {
        self.loco_con.init()?;
        self.power_trace.init()?;
        Ok(())
    }
}
impl SerdeAPI for ConsistSimulation {}

impl Default for ConsistSimulation {
    fn default() -> Self {
        let mut consist_sim = Self::new(Consist::default(), PowerTrace::default(), Some(1));
        consist_sim.init().unwrap();
        consist_sim
    }
}

#[cfg(test)]
mod tests {
    use super::{Consist, ConsistSimulation};
    use crate::consist::locomotive::loco_sim::PowerTrace;

    #[test]
    fn test_consist_sim() {
        let consist = Consist::default();
        let pt = PowerTrace::default();
        let mut consist_sim = ConsistSimulation::new(consist, pt, None);
        consist_sim.walk().unwrap();
    }
}
