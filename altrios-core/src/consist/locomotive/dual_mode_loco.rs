use super::powertrain::electric_drivetrain::ElectricDrivetrain;
use super::powertrain::fuel_converter::FuelConverter;
use super::powertrain::catenary_interface::CatenaryInterface;
use super::powertrain::generator::Generator;
use super::powertrain::ElectricMachine;
use super::LocoTrait;
use super::*;
use crate::imports::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, StateMethods, SetCumulative)]
#[serde_api]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq, from_py_object))]
/// Dual-mode locomotive
pub struct DualModeLoco {
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub gen: Generator,
    #[has_state]
    pub edrv: ElectricDrivetrain,
    /// Optional catenary/pantograph interface. When `Some`, the locomotive can
    /// draw traction power from the catenary in place of the engine. When
    /// `None`, it has no pantograph and runs as a diesel-electric.
    #[has_state]
    #[serde(default)]
    pub cat_int: Option<CatenaryInterface>,

}

#[pyo3_api]
impl DualModeLoco {
    #[new]
    pub fn __new__(
        fuel_converter: FuelConverter,
        generator: Generator,
        electric_drivetrain: ElectricDrivetrain,
        cat_int: Option<CatenaryInterface>,
    ) -> Self {
        Self {
            fc: fuel_converter,
            gen: generator,
            edrv: electric_drivetrain,
            cat_int,
        }
    }
}

impl DualModeLoco {
    pub fn new(
        fuel_converter: FuelConverter,
        generator: Generator,
        electric_drivetrain: ElectricDrivetrain,
        cat_int: Option<CatenaryInterface>,
    ) -> Self {
        Self {
            fc: fuel_converter,
            gen: generator,
            edrv: electric_drivetrain,
            cat_int,
        }
    }

    /// # Arguments
    /// - `pwr_out_req`: power required at the wheel/rail interface
    /// - `dt`: time step size
    /// - `loco_on`: whether engine is on (i.e. rotating and consuming at least idle fuel)
    /// - `pwr_aux`: power demand for auxilliary systems
    /// - `assert_limits`: whether to fail when powertrain capabilities are exceeded
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        loco_on: bool,
        pwr_aux: si::Power,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;

        let pwr_elec_prop_in = *self
            .edrv
            .state
            .pwr_elec_prop_in
            .get_fresh(|| format_dbg!())?;

        // Tell the catenary the full electrical demand (propulsion + aux) so it
        // can cover aux load with any surplus capacity, enabling the engine to
        // shut off when catenary power is sufficient. During regeneration
        // (`pwr_elec_prop_in < 0`, only possible under a receptive wire since
        // that is the only regen sink this locomotive has) the sum nets the
        // recovered power against aux first and returns the remainder.
        let cat_pwr_out = if let Some(cat_int) = self.cat_int.as_mut() {
            cat_int.set_pwr_in_req(pwr_elec_prop_in + pwr_aux, dt)?;
            *cat_int.state.pwr_elec_out.get_fresh(|| format_dbg!())?
        } else {
            si::Power::ZERO
        };

        let (gen_prop_req, gen_aux_req) = if pwr_elec_prop_in >= si::Power::ZERO {
            // Catenary covers propulsion first, then aux with any remaining capacity.
            let cat_to_prop = cat_pwr_out.min(pwr_elec_prop_in);
            let cat_to_aux = cat_pwr_out - cat_to_prop;
            (
                (pwr_elec_prop_in - cat_to_prop).max(si::Power::ZERO),
                (pwr_aux - cat_to_aux).max(si::Power::ZERO),
            )
        } else {
            // Regenerating: recovered power serves aux, the wire takes or gives
            // the net, and the engine covers only aux the two cannot.
            let net_req = pwr_elec_prop_in + pwr_aux;
            (
                si::Power::ZERO,
                (net_req - cat_pwr_out).max(si::Power::ZERO),
            )
        };

        // Engine only needs to be on if the generator must supply non-zero power.
        let engine_on = loco_on && (gen_prop_req + gen_aux_req > si::Power::ZERO);

        self.gen
            .set_pwr_in_req(gen_prop_req, gen_aux_req, engine_on, dt)?;

        self.fc.solve_energy_consumption(
            *self.gen.state.pwr_mech_in.get_fresh(|| format_dbg!())?,
            dt,
            engine_on,
            assert_limits,
        )?;
        Ok(())
    }
}

impl Mass for DualModeLoco {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.derived_mass().with_context(|| format_dbg!())
    }

    fn set_mass(
        &mut self,
        _new_mass: Option<si::Mass>,
        _side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        Err(anyhow!(
            "`set_mass` not enabled for {}",
            stringify!(DualModeLoco)
        ))
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.fc.mass().with_context(|| format_dbg!())
    }

    fn expunge_mass_fields(&mut self) {
        self.fc.expunge_mass_fields();
        self.gen.expunge_mass_fields();
    }
}

impl Default for DualModeLoco {
    fn default() -> Self {
        Self {
            fc: Default::default(),
            gen: Default::default(),
            edrv: Default::default(),
            cat_int: Some(Default::default()),
        }
    }
}

impl Init for DualModeLoco {
    fn init(&mut self) -> Result<(), Error> {
        self.fc.init()?;
        self.gen.init()?;
        self.edrv.init()?;
        if let Some(cat_int) = self.cat_int.as_mut() {
            cat_int.init()?;
        }
        Ok(())
    }
}
impl SerdeAPI for DualModeLoco {}

impl LocoTrait for DualModeLoco {
    fn set_cat_power_limit(
        &mut self,
        pwr_cat_lim: Option<si::Power>,
        loss_factor: Option<f64>,
    ) -> anyhow::Result<()> {
        if let Some(cat_int) = self.cat_int.as_mut() {
            cat_int.set_cat_power_limit(pwr_cat_lim, loss_factor)?;
        }
        Ok(())
    }
    /// returns current max power
    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)>,
        _train_mass: Option<si::Mass>,
        _train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let cat_pwr_out_max = if let Some(cat_int) = self.cat_int.as_mut() {
            let pwr_elec_in_max = *cat_int.state.pwr_elec_in_max.get_fresh(|| format_dbg!())?;
            cat_int.set_cur_pwr_max_out(pwr_elec_in_max, None)?;
            *cat_int.state.pwr_elec_out_max.get_fresh(|| format_dbg!())?
        } else {
            si::Power::ZERO
        };
        self.fc.set_cur_pwr_out_max(elev_and_temp, dt)?;
        self.gen.set_cur_pwr_max_out(
            *self.fc.state.pwr_out_max.get_fresh(|| format_dbg!())?,
            Some(pwr_aux.with_context(|| format_dbg!("`pwr_aux` not provided"))?),
        )?;
        self.edrv.set_cur_pwr_max_out(
            *self
                .gen
                .state
                .pwr_elec_prop_out_max
                .get_fresh(|| format_dbg!())?
                + cat_pwr_out_max,
            None,
        )?;
        // A dual-mode unit has no battery: its only regen sink is a receptive
        // wire, and off the wire (or under a non-receptive one) braking energy
        // goes to the dynamic-brake grids.
        let receptive = self.cat_int.as_ref().map_or(false, |c| c.receptive);
        self.edrv.set_cur_pwr_regen_max(if receptive {
            cat_pwr_out_max
        } else {
            si::Power::ZERO
        })?;
        self.gen
            .set_pwr_rate_out_max(self.fc.pwr_out_max / self.fc.pwr_ramp_lag)?;
        self.edrv.set_pwr_rate_out_max(
            *self
                .gen
                .state
                .pwr_rate_out_max
                .get_fresh(|| format_dbg!())?,
        )?;
        Ok(())
    }

    fn get_energy_loss(&self) -> anyhow::Result<si::Energy> {
        let cat_energy_loss = match &self.cat_int {
            Some(cat_int) => *cat_int.state.energy_loss.get_stale(|| format_dbg!())?,
            None => si::Energy::ZERO,
        };
        Ok(*self.fc.state.energy_loss.get_stale(|| format_dbg!())?
            + *self.gen.state.energy_loss.get_stale(|| format_dbg!())?
            + *self.edrv.state.energy_loss.get_stale(|| format_dbg!())?
            + cat_energy_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_off_when_catenary_sufficient() {
        let mut dml = DualModeLoco::default();
        let dt = 1.0 * uc::S;
        let pwr_aux = 200e3 * uc::W;
        let pwr_out_req = 1_000e3 * uc::W; // 1 MW propulsion

        // check_and_reset transitions all TrackedState fields from their initial
        // Fresh state to Stale; step() increments the iteration counters.
        dml.check_and_reset(|| format_dbg!()).unwrap();
        dml.step(|| format_dbg!()).unwrap();

        // Set catenary power limit well above total demand (10 MW >> 1.2 MW).
        dml.cat_int
            .as_mut()
            .unwrap()
            .set_cat_power_limit(Some(10_000e3 * uc::W), None)
            .unwrap();
        dml.set_curr_pwr_max_out(Some(pwr_aux), None, None, None, dt)
            .unwrap();
        dml.solve_energy_consumption(pwr_out_req, dt, true, pwr_aux, false)
            .unwrap();

        assert!(
            !*dml.fc.state.engine_on.get_fresh(|| format_dbg!()).unwrap(),
            "Engine should be off when catenary covers full propulsion and aux demand"
        );
        assert_eq!(
            dml.fc
                .state
                .pwr_fuel
                .get_fresh(|| format_dbg!())
                .unwrap()
                .get::<si::watt>(),
            0.0,
            "Fuel power should be zero when engine is off"
        );
    }
}
