use super::*;

// FUTURE: think about how to incorporate life modeling for Fuel Cells and other tech

const TOL: f64 = 1e-3;

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct for modeling Fuel Converter (e.g. engine, fuel cell.)
pub struct AESSController {
    #[serde(default)]
    /// struct for tracking current state
    pub state: AESSStateState,
    /// Threshold for turning engine on to run compressor for air brakes
    brake_press_thresh: si::pressure,
    /// brake system volume
    pub brake_system_volume: si::Power,
    /// air leak rate from brake system
    pub brake_leak_rate: si::flow,
    /// start battery capacity
    pub batt_capacity: si::electric_charge,
    /// pub battery volatage start thresh
    pub batt_volt_thresh: si::voltage,
    /// aux load for ECU, HVAC, .....
    pub aux_load_draw: si::power,
    /// soc as a function of voltage for starter battery
    pub soc_voltage_curve: Vec<f64>,
    /// voltage cutpoints for soc curve
    pub voltage_cutpoints_for_soc: Vec<f64>,
    /// save interval for AESS controller
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: AESSHistoryVec, // TODO: spec out fuel tank size and track kg of fuel
}

#[pyo3_api]
impl AESSController {
    // optional, custom, struct-specific pymethods

    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl Default for AESSController {
    fn default() -> Self {
        let file_contents = include_str!("fuel_converter.default.yaml");
        let mut aess_cntrl = Self::from_yaml(file_contents, false).unwrap();
        aess_cntrl.init().unwrap();
        aess_cntrl
    }
}

impl Init for AESSController {
    fn init(&mut self) -> Result<(), Error> {
        self.state.init()?;
        Ok(())
    }
}
impl SerdeAPI for AESSController {}

// non-py methods
impl FuelConverter {
    pub fn determine_states(
        &mut self,
        loco_pwr_demand: si::power,
        elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // solve for updated states
        if self.engine_on {
            //if self.batt
            //calculate soc based on curve of lead acid battery
            //model battery charging
            //calculate new voltage

            //model air compressor and leaks if pressure warrants it turning on
            //model HVAC loads based on temp
            //calculate air flow required for heat rejection and what fan power might be?

            //calculate engine warming up based upon current power
        } else {
            //model aux draw on battery
            //model air leaks with compressor off
            //model engine temp cooling off
        }

        //calculate wether engine should be on for next dt
        //calculate aux draw
        Ok(())
    }
}

#[serde_api]
#[derive(
    Clone, Debug, Deserialize, Serialize, PartialEq, HistoryVec, StateMethods, SetCumulative,
)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct AESSState {
    /// iteration counter
    pub i: TrackedState<usize>,
    /// max power fc can produce at current time
    pub time_in_state: TrackedState<si::time>,
    /// efficiency evaluated at current demand
    pub brake_pressure: TrackedState<si::pressure>,
    /// instantaneous shaft power going to generator
    pub starter_batt_voltage: TrackedState<si::voltage>,
    /// instantaneous fuel power flow
    pub last_state_change_cause: TrackedState<usize>,
    /// loss power, including idle
    pub brake_press_start_count: TrackedState<usize>,
    /// idle fuel flow rate power
    pub batt_volt_start_count: TrackedState<usize>,
    /// cumulative shaft energy fc has provided to generator
    pub engine_temp_start_count: TrackedState<usize>,
    /// cumulative fuel energy fc has consumed
    pub ambient_temp_start_count: TrackedState<si::time>,
    /// cumulative energy fc has lost due to imperfect efficiency
    pub time_idling: TrackedState<si::time>,
    /// cumulative fuel energy fc has lost due to idle
    pub time_off: TrackedState<si::time>,
    /// If true, engine is on, and if false, off (no idle)
    pub time_in_idle_notch: TrackedState<si::time>,
}

#[pyo3_api]
impl FuelConverterState {}

impl Init for AESSState {}
impl SerdeAPI for AESSState {}
impl Default for AESSState {
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
