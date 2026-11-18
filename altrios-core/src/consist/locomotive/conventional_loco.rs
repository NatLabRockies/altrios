use super::powertrain::electric_drivetrain::ElectricDrivetrain;
use super::powertrain::fuel_converter::FuelConverter;
use super::powertrain::generator::Generator;
use super::powertrain::ElectricMachine;
use super::LocoTrait;
use super::*;
use crate::imports::*;

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize, StateMethods, SetCumulative)]
#[serde_api]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Conventional locomotive
pub struct ConventionalLoco {
    // The fields in this struct are all locally defined structs and are therefore not documented in
    // this context
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub gen: Generator,
    #[has_state]
    pub edrv: ElectricDrivetrain,
}

#[pyo3_api]
impl ConventionalLoco {
    #[new]
    pub fn __new__(
        fuel_converter: FuelConverter,
        generator: Generator,
        electric_drivetrain: ElectricDrivetrain,
    ) -> Self {
        Self {
            fc: fuel_converter,
            gen: generator,
            edrv: electric_drivetrain,
        }
    }
}

impl ConventionalLoco {
    pub fn new(
        fuel_converter: FuelConverter,
        generator: Generator,
        electric_drivetrain: ElectricDrivetrain,
    ) -> Self {
        Self {
            fc: fuel_converter,
            gen: generator,
            edrv: electric_drivetrain,
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

        self.gen.set_pwr_in_req(
            // TODO: maybe this should be either zero or greater than or equal to zero if not loco_on
            *self
                .edrv
                .state
                .pwr_elec_prop_in
                .get_fresh(|| format_dbg!())?,
            pwr_aux,
            loco_on,
            dt,
        )?;

        self.fc.solve_energy_consumption(
            *self.gen.state.pwr_mech_in.get_fresh(|| format_dbg!())?,
            dt,
            loco_on,
            assert_limits,
        )?;
        Ok(())
    }
}

impl Mass for ConventionalLoco {
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
            stringify!(ConventionalLoco)
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

impl Init for ConventionalLoco {
    fn init(&mut self) -> Result<(), Error> {
        self.fc.init()?;
        self.gen.init()?;
        self.edrv.init()?;
        Ok(())
    }
}
impl SerdeAPI for ConventionalLoco {}

impl LocoTrait for ConventionalLoco {
    /// returns current max power, current max power rate, and current max regen
    /// power that can be absorbed by the RES/battery
    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: si::Power,
        elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)>,
        _train_mass: si::Mass,
        _train_speed: si::Velocity,
        _speed_limit_lookahead: (si::Velocity, si::Velocity),
        _elev_lookahead: (si::Length, si::Length),
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.fc.set_cur_pwr_out_max(elev_and_temp, dt)?;
        self.gen.set_cur_pwr_max_out(
            *self.fc.state.pwr_out_max.get_fresh(|| format_dbg!())?,
            Some(pwr_aux),
        )?;
        self.edrv.set_cur_pwr_max_out(
            *self
                .gen
                .state
                .pwr_elec_prop_out_max
                .get_fresh(|| format_dbg!())?,
            None,
        )?;
        self.edrv.set_cur_pwr_regen_max(si::Power::ZERO)?;
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
        Ok(*self.fc.state.energy_loss.get_stale(|| format_dbg!())?
            + *self.gen.state.energy_loss.get_stale(|| format_dbg!())?
            + *self.edrv.state.energy_loss.get_stale(|| format_dbg!())?)
    }
}
