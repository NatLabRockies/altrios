use super::powertrain::electric_drivetrain::{ElectricDrivetrain, ElectricDrivetrainState};
use super::powertrain::fuel_converter::FuelConverter;
use super::powertrain::generator::{Generator, GeneratorState};
use super::powertrain::reversible_energy_storage::{
    ReversibleEnergyStorage, ReversibleEnergyStorageState,
};
use super::powertrain::ElectricMachine;
use super::*;
use super::{LocoTrait, Mass, MassSideEffect};
use crate::imports::*;

#[serde_api]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq, from_py_object))]
/// Hybrid locomotive with both engine and reversible energy storage (aka battery)  
/// This type of locomotive is not likely to be widely prevalent due to modularity of consists.
pub struct HybridLoco {
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub gen: Generator,
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub edrv: ElectricDrivetrain,
    /// control strategy for distributing power demand between `fc` and `res`
    #[serde(default)]
    #[has_state]
    pub pt_cntrl: HybridPowertrainControls,
    /// which component carries the aux (hotel) load when both could. Defaults to
    /// [HEVAuxControls::AuxOnResPriority], the behavior before this was wired up.
    #[serde(default)]
    pub aux_cntrl: HEVAuxControls,
}

#[pyo3_api]
impl HybridLoco {}

impl Default for HybridLoco {
    fn default() -> Self {
        Self {
            fc: {
                let mut fc: FuelConverter = Default::default();
                // dial it back for Hybrid
                fc.pwr_out_max /= 2.0;
                fc
            },
            gen: Default::default(),
            res: {
                let mut res: ReversibleEnergyStorage = Default::default();
                // dial it back for Hybrid
                res.pwr_out_max /= 3.0;
                res.energy_capacity /= 8.0;
                res
            },
            edrv: Default::default(),
            aux_cntrl: Default::default(),
            pt_cntrl: Default::default(),
        }
    }
}

impl Init for HybridLoco {
    fn init(&mut self) -> Result<(), Error> {
        self.fc.init()?;
        self.gen.init()?;
        self.res.init()?;
        self.edrv.init()?;
        self.pt_cntrl.init()?;
        Ok(())
    }
}
impl SerdeAPI for HybridLoco {}

impl Mass for HybridLoco {
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
            stringify!(HybridLoco)
        ))
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.fc.mass().with_context(|| format_dbg!())
    }

    fn expunge_mass_fields(&mut self) {
        self.fc.expunge_mass_fields();
        self.gen.expunge_mass_fields();
        self.res.expunge_mass_fields();
    }
}

impl LocoTrait for Box<HybridLoco> {
    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)>,
        // amount of assigned train mass for this locomotive
        train_mass_for_loco: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let mass_for_loco: si::Mass = train_mass_for_loco.with_context(|| {
            format!(
                "{}\n`train_mass_for_loco` must be provided for `HybridLoco` ",
                format_dbg!()
            )
        })?;
        let train_speed: si::Velocity = train_speed.with_context(|| {
            format!(
                "{}\n`train_speed` must be provided for `HybridLoco` ",
                format_dbg!()
            )
        })?;
        match &mut self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => {
                rgwb.state.on_time_too_short.update(*self.fc.state.engine_on.get_stale(|| format_dbg!())? && *self.fc.state.time_on.get_stale(|| format_dbg!())?
                    < rgwb.fc_min_time_on.with_context(|| {
                    anyhow!(
                        "{}\n Expected `ResGreedyWithBuffers::init` to have been called beforehand.",
                        format_dbg!()
                    )
                })?, || format_dbg!())?;
                // Shutdown delay. Both timers are advanced here, from the
                // PREVIOUS step's resolved causes, because a delay is by
                // definition a hold-over from the past -- this step's causes
                // are not known until `get_pwr_gen_and_res` runs.
                // `time_since_fc_cause` resets to zero on any step a physical
                // cause fired and otherwise accumulates; the engine is held on
                // while it is still inside the delay. The timer can only ADD a
                // reason to run, so a cause arriving in this step starts the
                // engine regardless of either timer.
                let engine_was_on = *self.fc.state.engine_on.get_stale(|| format_dbg!())?;
                let had_cause = *rgwb.state.fc_cause_latch.get_stale(|| format_dbg!())?;
                let since = if had_cause {
                    si::Time::ZERO
                } else {
                    *rgwb.state.time_since_fc_cause.get_stale(|| format_dbg!())? + dt
                };
                rgwb.state
                    .time_since_fc_cause
                    .update(since, || format_dbg!())?;
                let delay = rgwb.fc_shutdown_delay.with_context(|| {
                    anyhow!(
                        "{}  Expected `ResGreedyWithBuffers::init` to have been called beforehand.",
                        format_dbg!()
                    )
                })?;
                rgwb.state
                    .shutdown_delay_pending
                    .update(engine_was_on && since < delay, || format_dbg!())?;
            }
        };

        self.fc.set_cur_pwr_out_max(elev_and_temp, dt)?;
        let disch_buffer: si::Energy = match &self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => {
                (0.5 * mass_for_loco
                    * (rgwb
                        .speed_soc_disch_buffer
                        .with_context(|| format_dbg!())?
                        .powi(typenum::P2::new())
                        - train_speed.powi(typenum::P2::new())))
                .max(si::Energy::ZERO)
                    * rgwb
                        .speed_soc_disch_buffer_coeff
                        .with_context(|| format_dbg!())?
            }
        };
        let chrg_buffer: si::Energy = match &self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => {
                (0.5 * mass_for_loco
                    * (train_speed.powi(typenum::P2::new())
                        - rgwb
                            .speed_soc_regen_buffer
                            .with_context(|| format_dbg!())?
                            .powi(typenum::P2::new())))
                .max(si::Energy::ZERO)
                    * rgwb
                        .speed_soc_regen_buffer_coeff
                        .with_context(|| format_dbg!())?
            }
        };

        let regen_bypasses_buffer = match &self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => rgwb
                .regen_bypasses_buffer
                .with_context(|| format_dbg!())?,
        };
        let pwr_charge_max_buffered = self.res.set_curr_pwr_out_max_split(
            dt,
            pwr_aux.with_context(|| format!("{}\nExpected `pwr_aux` to be Some", format_dbg!()))?,
            disch_buffer,
            chrg_buffer,
            !regen_bypasses_buffer,
        )?;
        match &mut self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => {
                rgwb.state
                    .pwr_charge_max_for_fc
                    .update(pwr_charge_max_buffered, || format_dbg!())?;
            }
        };

        self.gen.set_cur_pwr_max_out(
            *self.fc.state.pwr_out_max.get_fresh(|| format_dbg!())?,
            Some(si::Power::ZERO),
        )?;

        self.edrv.set_cur_pwr_max_out(
            *self
                .gen
                .state
                .pwr_elec_prop_out_max
                .get_fresh(|| format_dbg!())?
                + *self.res.state.pwr_prop_max.get_fresh(|| format_dbg!())?,
            None,
        )?;

        self.edrv
            .set_cur_pwr_regen_max(*self.res.state.pwr_charge_max.get_fresh(|| format_dbg!())?)?;

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
            + *self.res.state.energy_loss.get_stale(|| format_dbg!())?
            + *self.edrv.state.energy_loss.get_stale(|| format_dbg!())?)
    }
}

impl HybridLoco {
    /// Solve fc and res energy consumption
    /// # Arguments:
    /// - `pwr_out_req`: tractive power require
    /// - `train_speed`: current train speed
    /// - `dt`: time step size
    ///
    /// Controls decide the split between the alternator and the battery
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        train_mass: si::Mass,
        train_speed: si::Velocity,
        dt: si::Time,
        pwr_aux: si::Power,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;
        let (gen_pwr_out_req, res_pwr_out_req) = self
            .pt_cntrl
            .get_pwr_gen_and_res(
                *self
                    .edrv
                    .state
                    .pwr_elec_prop_in
                    .get_fresh(|| format_dbg!())?,
                train_mass,
                train_speed,
                &self.fc,
                &self.gen,
                &self.edrv.state,
                &self.res,
                pwr_aux,
            )
            .with_context(|| format_dbg!())?;

        let fc_on: bool = self.pt_cntrl.fc_on().with_context(|| format_dbg!())?;
        // Aux (hotel) load goes to the engine when the controls prefer it AND the
        // engine is running; otherwise the battery carries it. Previously the
        // battery carried aux unconditionally, so the pack drained for hotel load
        // even while the engine was turning -- biasing SOC down and overstating
        // how hard the pack works.
        let aux_on_fc: bool = fc_on && self.aux_cntrl.is_aux_on_fc_priority();
        let pwr_aux_fc = if aux_on_fc { pwr_aux } else { si::Power::ZERO };
        let pwr_aux_res = if aux_on_fc { si::Power::ZERO } else { pwr_aux };
        self.gen
            .set_pwr_in_req(
                // TODO: maybe this should be zero if not loco_on
                gen_pwr_out_req,
                pwr_aux_fc,
                fc_on,
                dt,
            )
            .with_context(|| format_dbg!(fc_on))?;
        let fc_pwr_mech_out = *self.gen.state.pwr_mech_in.get_fresh(|| format_dbg!())?;

        self.fc
            .solve_energy_consumption(fc_pwr_mech_out, dt, fc_on, assert_limits)
            .with_context(|| {
                format!(
                    "{}

{} kW

{} kW

{} kW
{} kW
{} kW
{} kW

{} kW

{} kW
{} kW
{} kW
{} kW",
                    format_dbg!(fc_on).replace("\"", ""),
                    format_dbg!(pwr_out_req.get::<si::kilowatt>()).replace("\"", ""),
                    format_dbg!(self.edrv.state.pwr_elec_prop_in).replace("\"", ""),
                    format_dbg!(gen_pwr_out_req.get::<si::kilowatt>()).replace("\"", ""),
                    format_dbg!(self.gen.state.pwr_elec_aux).replace("\"", ""),
                    format_dbg!(self.gen.state.pwr_elec_out_max).replace("\"", ""),
                    format_dbg!(self.gen.state.pwr_mech_in).replace("\"", ""),
                    format_dbg!(res_pwr_out_req.get::<si::kilowatt>()).replace("\"", ""),
                    format_dbg!(self.res.state.pwr_prop_max).replace("\"", ""),
                    format_dbg!(self.res.state.pwr_disch_max).replace("\"", ""),
                    format_dbg!(self.res.state.pwr_regen_max).replace("\"", ""),
                    format_dbg!(self.res.state.pwr_charge_max).replace("\"", ""),
                )
            })?;

        self.res
            .solve_energy_consumption(
                res_pwr_out_req,
                pwr_aux_res,
                dt,
            )
            .with_context(|| format_dbg!())?;

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default, IsVariant, From, TryInto)]
pub enum HEVAuxControls {
    /// If feasible, use [ReversibleEnergyStorage] to handle aux power demand
    #[default]
    AuxOnResPriority,
    /// If feasible, use [FuelConverter] to handle aux power demand
    AuxOnFcPriority,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, IsVariant, From, TryInto)]
pub enum HybridPowertrainControls {
    /// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
    /// and discharge power inside of static min and max SOC range.  Also, includes
    /// buffer for forcing [FuelConverter] to be active/on.
    RGWDB(Box<RESGreedyWithDynamicBuffers>),
}

impl StateMethods for HybridPowertrainControls {}
impl SaveState for HybridPowertrainControls {
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => rgwdb.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?,
        }
        Ok(())
    }
}

impl Step for HybridPowertrainControls {
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => rgwdb.step(|| format!("{}\n{}", loc(), format_dbg!()))?,
        }
        Ok(())
    }
}

impl CheckAndResetState for HybridPowertrainControls {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => {
                rgwdb.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
        }
        Ok(())
    }
}

impl SetCumulative for HybridPowertrainControls {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => {
                rgwdb.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?
            }
        }
        Ok(())
    }
}

impl Default for HybridPowertrainControls {
    fn default() -> Self {
        Self::RGWDB(Default::default())
    }
}

impl Init for HybridPowertrainControls {
    fn init(&mut self) -> Result<(), Error> {
        match self {
            Self::RGWDB(rgwb) => rgwb.init()?,
        }
        Ok(())
    }
}

impl SerdeAPI for HybridPowertrainControls {}

impl FuelConverterOn for HybridPowertrainControls {
    fn fc_on(&self) -> anyhow::Result<bool> {
        match self {
            Self::RGWDB(rgwdb) => rgwdb.fc_on(),
        }
    }
}

impl HybridPowertrainControls {
    /// Determines power split between engine/generator and battery
    ///
    /// # Arguments
    /// - `pwr_res_and_gen_to_edrv`: tractive power required as input to [ElectricDrivetrain]
    /// - `veh_state`: vehicle state
    /// - `hev_state`: HEV powertrain state
    /// - `fc`: fuel converter
    /// - `em_state`: electric machine state
    /// - `res`: reversible energy storage (e.g. high voltage battery)
    #[allow(clippy::too_many_arguments)]
    fn get_pwr_gen_and_res(
        &mut self,
        pwr_res_and_gen_to_edrv: si::Power,
        // TODO: make a `TrainMomentum` object to pass both mass and speed
        train_mass: si::Mass,
        train_speed: si::Velocity,
        fc: &FuelConverter,
        gen: &Generator,
        edrv_state: &ElectricDrivetrainState,
        res: &ReversibleEnergyStorage,
        pwr_aux: si::Power,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        let fc_state = &fc.state;
        ensure!(
            // `almost` is in case of negligible numerical precision discrepancies
            almost_le_uom(
                &pwr_res_and_gen_to_edrv,
                &(*gen
                    .state
                    .pwr_elec_prop_out_max
                    .get_fresh(|| format_dbg!())?
                    + *res.state.pwr_prop_max.get_fresh(|| format_dbg!())?),
                None
            ),
            "{}
`pwr_out_req`: {:?}
`em_state.pwr_mech_fwd_out_max`: {:?}
`fc_state.pwr_prop_max`: {:?}
`res.state.soc`: {:?}",
            format_dbg!(),
            pwr_res_and_gen_to_edrv,
            edrv_state.pwr_mech_out_max,
            fc_state.pwr_out_max,
            res.state.soc
        );

        let (gen_prop_pwr, res_prop_pwr) = match self {
            Self::RGWDB(ref mut rgwdb) => {
                // handle_fc_on_causes_for_temp(fc, rgwdb, hev_state)?;
                rgwdb.handle_fc_on_causes_for_speed(train_speed)?;
                rgwdb.handle_fc_on_causes_for_aux_demand(pwr_aux, &res.state)?;
                rgwdb.handle_fc_on_causes_for_low_soc(res, train_mass, train_speed)?;
                // `handle_fc_*` below here are asymmetrical for positive tractive power only
                rgwdb.handle_fc_on_causes_for_pwr_demand(
                    pwr_res_and_gen_to_edrv,
                    &gen.state,
                    &res.state,
                )?;

                let res_prop_pwr = pwr_res_and_gen_to_edrv
                    .min(*res.state.pwr_prop_max.get_fresh(|| format_dbg!())?)
                    .max(-*res.state.pwr_regen_max.get_fresh(|| format_dbg!())?);
                // What the battery may take from the ENGINE: the regen-buffer-
                // derated charge limit plus the aux share that
                // `pwr_regen_max` carries over `pwr_charge_max`. Regen itself is
                // bounded by the full `pwr_regen_max` above and below.
                let pwr_regen_max_for_fc = (*rgwdb
                    .state
                    .pwr_charge_max_for_fc
                    .get_fresh(|| format_dbg!())?
                    + *res.state.pwr_regen_max.get_fresh(|| format_dbg!())?
                    - *res.state.pwr_charge_max.get_fresh(|| format_dbg!())?)
                .min(*res.state.pwr_regen_max.get_fresh(|| format_dbg!())?);

                rgwdb
                    .state
                    .fc_temperature_too_low
                    .mark_fresh(|| format_dbg!())?;
                // Every physical cause for this step is now set. Latch whether
                // any of them fired, so the next step's shutdown delay knows how
                // long it has been since the engine last had a real reason to run.
                let physical = rgwdb
                    .state
                    .fc_on_from_physical_causes()
                    .with_context(|| format_dbg!())?;
                rgwdb
                    .state
                    .fc_cause_latch
                    .update(physical, || format_dbg!())?;
                if !rgwdb.state.fc_on().with_context(|| format_dbg!())? {
                    // engine is off, and `em_pwr` has already been limited within bounds
                    ensure!(
                        res_prop_pwr == pwr_res_and_gen_to_edrv,
                        format!(
                            "{}\n{}\n`res_prop_pwr` must be able to handle everything when engine is off",
                            format_dbg!(res_prop_pwr.get::<si::kilowatt>()),
                            format_dbg!(pwr_res_and_gen_to_edrv.get::<si::kilowatt>())
                        )
                    );
                    (si::Power::ZERO, pwr_res_and_gen_to_edrv)
                } else {
                    // engine has been forced on
                    let pwr_gen_elec_out_for_eff_fc =
                        get_pwr_gen_elec_out_for_eff_fc(fc, gen, rgwdb)?;
                    let gen_pwr = if pwr_res_and_gen_to_edrv < si::Power::ZERO {
                        // negative tractive power
                        // max power system can receive from engine during negative traction
                        (pwr_regen_max_for_fc + pwr_res_and_gen_to_edrv)
                            // or peak efficiency power if it's lower than above
                            .min(pwr_gen_elec_out_for_eff_fc)
                            // but not negative
                            .max(si::Power::ZERO)
                    } else {
                        // positive tractive power
                        if pwr_res_and_gen_to_edrv - res_prop_pwr > pwr_gen_elec_out_for_eff_fc {
                            // engine needs to run higher than peak efficiency point
                            pwr_res_and_gen_to_edrv - res_prop_pwr
                        } else {
                            // engine does not need to run higher than peak
                            // efficiency point to make tractive demand

                            // gen handles all power not covered by em
                            (pwr_res_and_gen_to_edrv - res_prop_pwr)
                                // and if that's less than the
                                // efficiency-focused value, then operate at
                                // that value
                                .max(pwr_gen_elec_out_for_eff_fc)
                                // but don't exceed what what the battery can
                                // absorb + tractive demand
                                .min(pwr_res_and_gen_to_edrv + pwr_regen_max_for_fc)
                        }
                    }
                    // and don't exceed what the fc -> gen can do
                    .min(
                        *gen.state
                            .pwr_elec_prop_out_max
                            .get_fresh(|| format_dbg!())?,
                    );

                    // recalculate `em_pwr` based on `fc_pwr`
                    let res_pwr_corrected = (pwr_res_and_gen_to_edrv - gen_pwr)
                        .max(-*res.state.pwr_regen_max.get_fresh(|| format_dbg!())?);
                    (gen_pwr, res_pwr_corrected)
                }
            }
        };

        ensure!(
            almost_le_uom(
                &res_prop_pwr,
                res.state.pwr_prop_max.get_fresh(|| format_dbg!())?,
                None
            ),
            format!(
                "{}\n{}",
                format_dbg!(res_prop_pwr),
                format_dbg!(res.state.pwr_prop_max)
            )
        );

        Ok((gen_prop_pwr, res_prop_pwr))
    }
}

fn get_pwr_gen_elec_out_for_eff_fc(
    fc: &FuelConverter,
    gen: &Generator,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
) -> anyhow::Result<si::Power> {
    let pwr_gen_elec_out_for_eff_fc: si::Power =
        if let Some(pwr_gen_elec_out_for_eff_fc) = rgwdb.pwr_gen_elec_out_for_eff_fc {
            pwr_gen_elec_out_for_eff_fc
        } else {
            let frac_of_pwr_for_peak_eff: si::Ratio = rgwdb
                .frac_of_max_pwr_to_run_fc
                .with_context(|| format_dbg!())?;
            let mut gen = gen.clone();
            // this assumes that the generator has a fairly flat efficiency curve
            gen.state.pwr_elec_out_max.mark_stale();
            gen.state.pwr_elec_prop_out_max.mark_stale();
            gen.set_cur_pwr_max_out(
                frac_of_pwr_for_peak_eff * fc.pwr_out_max,
                Some(si::Power::ZERO),
            )
            .with_context(|| format_dbg!())?;
            rgwdb.pwr_gen_elec_out_for_eff_fc =
                Some(*gen.state.pwr_elec_out_max.get_fresh(|| format_dbg!())?);
            *gen.state.pwr_elec_out_max.get_fresh(|| format_dbg!())?
        };

    Ok(pwr_gen_elec_out_for_eff_fc)
}

/// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
/// and discharge power inside of static min and max SOC range.  Also, includes
/// buffer for forcing [FuelConverter] to be active/on. See [Self::init] for
/// default values.
#[serde_api]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq, from_py_object))]
#[non_exhaustive]
pub struct RESGreedyWithDynamicBuffers {
    /// RES energy delta from minimum SOC corresponding to kinetic energy of
    /// vehicle at this speed that triggers ramp down in RES discharge.
    pub speed_soc_disch_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of accel buffer
    pub speed_soc_disch_buffer_coeff: Option<si::Ratio>,
    /// RES energy delta from minimum SOC corresponding to kinetic energy of
    /// vehicle at this speed that triggers FC to be forced on.
    pub speed_soc_fc_on_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of [Self::speed_soc_fc_on_buffer]
    pub speed_soc_fc_on_buffer_coeff: Option<si::Ratio>,
    /// RES energy delta from maximum SOC corresponding to kinetic energy of
    /// vehicle at current speed minus kinetic energy of vehicle at this speed
    /// triggers ramp down in RES discharge
    pub speed_soc_regen_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of regen buffer
    pub speed_soc_regen_buffer_coeff: Option<si::Ratio>,
    /// When true (the default), the regen buffer limits only how far the ENGINE
    /// may charge the battery, so headroom is reserved for braking without
    /// throttling regen itself, which charges at the full rating up to `max_soc`.
    /// False restores the original behaviour, where the buffer derates all
    /// charging, regen included.
    pub regen_bypasses_buffer: Option<bool>,
    // TODO: make it so that the engine never goes off at all
    /// Minimum time engine must remain on if it was on during the previous
    /// simulation time step.
    pub fc_min_time_on: Option<si::Time>,
    /// How long the engine keeps running after the last reason to run it went
    /// away, before it is allowed to shut down. This is a SHUTDOWN DELAY, not a
    /// minimum-off lockout: it only ever keeps a running engine running, so a
    /// mandatory restart (cold engine, low SOC, propulsion demand the pack
    /// cannot serve) is never blocked by a timer. A lockout would have been the
    /// other way round -- it would hold the engine off through a demand spike --
    /// which is both unsafe and not what a real unit does.
    pub fc_shutdown_delay: Option<si::Time>,
    /// Fraction of the battery's current discharge limit that may be spent on
    /// aux (hotel) load before the engine is forced on. Keeps discharge headroom
    /// in reserve for traction.
    pub frac_res_disch_for_aux: Option<si::Ratio>,
    /// Speed at which [FuelConverter] is forced on.
    pub speed_fc_forced_on: Option<si::Velocity>,
    /// Fraction of total aux and powertrain rated power at which
    /// [FuelConverter] is forced on.
    pub frac_pwr_demand_fc_forced_on: Option<si::Ratio>,
    /// Force engine, if on, to run at this fraction of peak power or the
    /// required power, whichever is greater. If SOC is below min buffer or
    /// engine is otherwise forced on and battery has room to receive charge,
    /// engine will run at this level and charge.
    pub frac_of_max_pwr_to_run_fc: Option<si::Ratio>,
    /// Force generator, if engine is on, to run at this power to help run engine efficiently
    pub pwr_gen_elec_out_for_eff_fc: Option<si::Power>,
    // /// temperature at which engine is forced on to warm up
    // #[serde(default)]
    // pub temp_fc_forced_on: Option<si::Temperature>,
    // /// temperature at which engine is allowed to turn off due to being sufficiently warm
    // #[serde(default)]
    // pub temp_fc_allowed_off: Option<si::Temperature>,
    /// current state of control variables
    #[serde(default)]
    pub state: RGWDBState,
    #[serde(default)]
    /// history of current state
    pub history: RGWDBStateHistoryVec,
}

#[pyo3_api]
impl RESGreedyWithDynamicBuffers {}

impl RESGreedyWithDynamicBuffers {
    fn set_soc_fc_on_buffer(
        &mut self,
        res: &ReversibleEnergyStorage,
        mass: si::Mass,
        train_speed: si::Velocity,
    ) -> anyhow::Result<()> {
        self.state.soc_fc_on_buffer.update(
            {
                let energy_delta_to_buffer_speed: si::Energy = 0.5
                    * mass
                    * (self
                        .speed_soc_fc_on_buffer
                        .with_context(|| format_dbg!())?
                        .powi(typenum::P2::new())
                        - train_speed.powi(typenum::P2::new()));
                energy_delta_to_buffer_speed.max(si::Energy::ZERO)
                    * self
                        .speed_soc_fc_on_buffer_coeff
                        .with_context(|| format_dbg!())?
            } / res.energy_capacity_usable()
                + res.min_soc,
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Determines whether power demand requires engine to be on.  Not needed during
    /// negative traction.
    fn handle_fc_on_causes_for_pwr_demand(
        &mut self,
        pwr_out_req: si::Power,
        gen_state: &GeneratorState,
        res_state: &ReversibleEnergyStorageState,
    ) -> Result<(), anyhow::Error> {
        let frac_pwr_demand_fc_forced_on: si::Ratio = self
            .frac_pwr_demand_fc_forced_on
            .with_context(|| format_dbg!())?;
        self.state.propulsion_power_demand_soft.update(
            pwr_out_req
                > frac_pwr_demand_fc_forced_on
                    * *res_state.pwr_disch_max.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        self.state.propulsion_power_demand.update(
            pwr_out_req - *gen_state.pwr_elec_out_max.get_fresh(|| format_dbg!())?
                >= si::Power::ZERO,
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Determines whether the engine must be on to carry the aux (hotel) load.
    ///
    /// True when aux demand exceeds `frac_res_disch_for_aux` of what the battery
    /// can currently discharge. The fraction keeps a reserve for traction: a pack
    /// sized for power, not for hotel load, should not spend its discharge
    /// headroom on auxiliaries and then be unable to assist a grade.
    fn handle_fc_on_causes_for_aux_demand(
        &mut self,
        pwr_aux: si::Power,
        res_state: &ReversibleEnergyStorageState,
    ) -> anyhow::Result<()> {
        let frac: si::Ratio = self
            .frac_res_disch_for_aux
            .with_context(|| format_dbg!())?;
        self.state.aux_power_demand.update(
            pwr_aux > frac * *res_state.pwr_disch_max.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Determines whether enigne must be on for high speed
    fn handle_fc_on_causes_for_speed(&mut self, train_speed: si::Velocity) -> anyhow::Result<()> {
        self.state.train_speed_above_threshold.update(
            train_speed > self.speed_fc_forced_on.with_context(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Detemrines whether engine must be on to charge battery
    fn handle_fc_on_causes_for_low_soc(
        &mut self,
        res: &ReversibleEnergyStorage,
        mass: si::Mass,
        train_speed: si::Velocity,
    ) -> anyhow::Result<()> {
        self.set_soc_fc_on_buffer(res, mass, train_speed)?;
        self.state.charging_for_low_soc.update(
            *res.state.soc.get_stale(|| format_dbg!())?
                < *self.state.soc_fc_on_buffer.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        Ok(())
    }
}

impl FuelConverterOn for RESGreedyWithDynamicBuffers {
    fn fc_on(&self) -> anyhow::Result<bool> {
        self.state.fc_on().with_context(|| format_dbg!())
    }
}

impl Init for RESGreedyWithDynamicBuffers {
    fn init(&mut self) -> Result<(), Error> {
        // TODO: tunnel buffer!
        // TODO: make sure these values propagate to the documented defaults above
        init_opt_default!(self, speed_soc_disch_buffer, 40.0 * uc::MPH);
        init_opt_default!(self, speed_soc_disch_buffer_coeff, 1.0 * uc::R);
        init_opt_default!(self, speed_soc_fc_on_buffer, 100.0 * uc::MPH);
        init_opt_default!(self, speed_soc_fc_on_buffer_coeff, 1.0 * uc::R);
        init_opt_default!(self, frac_res_disch_for_aux, 0.5 * uc::R);
        init_opt_default!(self, speed_soc_regen_buffer, 10. * uc::MPH);
        init_opt_default!(self, speed_soc_regen_buffer_coeff, 1.0 * uc::R);
        init_opt_default!(self, regen_bypasses_buffer, true);
        // 60 s, not the 5 s this defaulted to. Five seconds is shorter than a
        // single dispatch transient, so it did nothing: the engine could restart
        // within one step of shutting down. Sixty seconds is the light-load
        // shutdown spec and roughly what a road unit's start logic allows.
        init_opt_default!(self, fc_min_time_on, uc::S * 60.0);
        // Hold a running engine for half a minute after the last reason to run
        // it clears. Together with the 60 s minimum run time this bounds the
        // cycling rate; neither timer can keep the engine OFF.
        init_opt_default!(self, fc_shutdown_delay, uc::S * 30.0);
        // Force FC to be on all the time by default
        // Was 0.0 mph, which made `train_speed > speed_fc_forced_on` true for ANY
        // movement and kept the engine running whenever the train was not stopped.
        // At 5 mph the engine may shut down while creeping and switching, where
        // the pack can carry both traction and aux; the aux cause above guards
        // against dumping hotel load on a battery that cannot hold it. This is a
        // starting value, not a calibrated one -- it is the threshold the
        // light-load shutdown study varies.
        init_opt_default!(self, speed_fc_forced_on, uc::MPH * 5.0);
        init_opt_default!(self, frac_pwr_demand_fc_forced_on, uc::R * 0.75);
        // 20% of peak power gets most of peak efficiency
        init_opt_default!(self, frac_of_max_pwr_to_run_fc, 0.2 * uc::R);
        Ok(())
    }
}
impl SerdeAPI for RESGreedyWithDynamicBuffers {}

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
#[serde(default)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq, from_py_object))]
/// State for [RESGreedyWithDynamicBuffers ]
pub struct RGWDBState {
    /// time step index
    pub i: TrackedState<usize>,

    /// Engine must be on to self heat if thermal model is enabled
    fc_temperature_too_low: TrackedState<bool>,
    /// Engine must be on for high train speed to ensure powertrain can meet
    /// any spikes in power demand
    train_speed_above_threshold: TrackedState<bool>,
    /// Engine has not been on long enough (usually 30 s)
    on_time_too_short: TrackedState<bool>,
    /// Powertrain power demand exceeds motor and/or battery capabilities
    propulsion_power_demand: TrackedState<bool>,
    /// Powertrain power demand exceeds optimal motor and/or battery output
    propulsion_power_demand_soft: TrackedState<bool>,
    /// Aux (hotel) power demand exceeds what the battery should serve, so the
    /// engine must run to carry it. Without this cause the engine never starts
    /// for aux alone, and shutting it down at low speed silently dumps the whole
    /// hotel load on the pack.
    aux_power_demand: TrackedState<bool>,
    /// SOC is below min buffer so FC is charging RES
    charging_for_low_soc: TrackedState<bool>,
    /// A running engine is inside its shutdown delay: every physical reason to
    /// run has gone away, but not for long enough yet. Keeps the engine from
    /// chattering on and off across a sequence of short light-load dips.
    shutdown_delay_pending: TrackedState<bool>,
    /// True on any step where a PHYSICAL cause demanded the engine, i.e. the
    /// `fc_on` causes excluding the two timers. Latched so the next step can
    /// measure how long it has been since one last fired.
    fc_cause_latch: TrackedState<bool>,
    /// Time since [Self::fc_cause_latch] was last true. Reset to zero on every
    /// step a physical cause fires, and compared against
    /// [RESGreedyWithDynamicBuffers::fc_shutdown_delay].
    pub time_since_fc_cause: TrackedState<si::Time>,

    /// buffer at which FC is forced on
    pub soc_fc_on_buffer: TrackedState<si::Ratio>,

    /// Battery charge power available to ENGINE charging: the regen-buffer-derated
    /// charge limit when [RESGreedyWithDynamicBuffers::regen_bypasses_buffer] is
    /// true, else the battery's own (already derated) charge limit
    pub pwr_charge_max_for_fc: TrackedState<si::Power>,
}

#[pyo3_api]
impl RGWDBState {}

impl Init for RGWDBState {}
impl SerdeAPI for RGWDBState {}

impl RGWDBState {
    /// OR of the PHYSICAL reasons to run the engine -- everything except the
    /// minimum-run-time and shutdown-delay timers. This is what the timers are
    /// measured against, and it is also the set a mandatory restart comes from,
    /// so no timer may ever suppress it.
    fn fc_on_from_physical_causes(&self) -> anyhow::Result<bool> {
        Ok(*self.fc_temperature_too_low.get_fresh(|| format_dbg!())?
            || *self
                .train_speed_above_threshold
                .get_fresh(|| format_dbg!())?
            || *self.propulsion_power_demand.get_fresh(|| format_dbg!())?
            || *self
                .propulsion_power_demand_soft
                .get_fresh(|| format_dbg!())?
            || *self.aux_power_demand.get_fresh(|| format_dbg!())?
            || *self.charging_for_low_soc.get_fresh(|| format_dbg!())?)
    }
}

impl FuelConverterOn for RGWDBState {
    /// If any of the causes are true, engine must be on. The two timers
    /// (`on_time_too_short`, `shutdown_delay_pending`) can only ever ADD to the
    /// physical causes, never mask one.
    fn fc_on(&self) -> anyhow::Result<bool> {
        Ok(self.fc_on_from_physical_causes()?
            || *self.on_time_too_short.get_fresh(|| format_dbg!())?
            || *self.shutdown_delay_pending.get_fresh(|| format_dbg!())?)
    }
}
