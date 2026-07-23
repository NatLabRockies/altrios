use super::powertrain::catenary_interface::CatenaryInterface;
use super::powertrain::electric_drivetrain::ElectricDrivetrain;
use super::powertrain::reversible_energy_storage::ReversibleEnergyStorage;
use super::powertrain::ElectricMachine;
use super::*;
use super::{LocoTrait, Mass, MassSideEffect};
use crate::imports::*;

#[serde_api]
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Battery electric locomotive
pub struct BatteryElectricLoco {
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub edrv: ElectricDrivetrain,
    /// control strategy for distributing power demand between `fc` and `res`
    #[has_state]
    #[serde(default)]
    pub pt_cntrl: BatteryPowertrainControls,
    /// Optional catenary/pantograph interface. When `Some`, the locomotive can
    /// draw catenary power -- for traction first, then to charge the battery
    /// with any leftover capacity. When `None`, this is a conventional
    /// battery-only locomotive with no pantograph.
    #[has_state]
    #[serde(default)]
    pub cat_int: Option<CatenaryInterface>,
    // /// field for tracking current state
    // #[serde(default)]
    // pub state: BELState,
    // /// vector of [Self::state]
    // #[serde(default)]
    // pub history: BELStateHistoryVec,
}

#[pyo3_api]
impl BatteryElectricLoco {}

impl BatteryElectricLoco {
    /// Solve energy consumption for the current power output required
    /// Arguments:
    /// - pwr_out_req: tractive power required
    /// - dt: time step size
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        pwr_aux: si::Power,
    ) -> anyhow::Result<()> {
        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;
        let pwr_elec_prop_in = *self
            .edrv
            .state
            .pwr_elec_prop_in
            .get_fresh(|| format_dbg!())?;

        if let Some(cat_int) = self.cat_int.as_mut() {
            if pwr_elec_prop_in > si::Power::ZERO {
                // Positive traction with catenary available: the catenary covers
                // propulsion first, then aux, and any leftover capacity charges
                // the battery (bounded by the battery's max charge power, which
                // already accounts for max SOC via the charge buffer).
                let pwr_charge_max =
                    *self.res.state.pwr_charge_max.get_fresh(|| format_dbg!())?;
                // Ask the catenary for propulsion + aux + whatever the battery can
                // absorb, so surplus capacity is used to charge the battery.
                cat_int.set_pwr_in_req(pwr_elec_prop_in + pwr_aux + pwr_charge_max, dt)?;
                let cat_pwr_out = *cat_int.state.pwr_elec_out.get_fresh(|| format_dbg!())?;

                // Allocate delivered catenary power: propulsion, then aux, then
                // battery charging.
                let cat_to_prop = cat_pwr_out.min(pwr_elec_prop_in);
                let cat_to_aux = (cat_pwr_out - cat_to_prop).min(pwr_aux);
                let cat_to_charge = (cat_pwr_out - cat_to_prop - cat_to_aux)
                    .min(pwr_charge_max)
                    .max(si::Power::ZERO);

                // Battery supplies the propulsion the catenary could not cover,
                // minus any surplus catenary power charging it (net negative =>
                // charging).
                let res_prop_req = pwr_elec_prop_in - cat_to_prop - cat_to_charge;
                let res_aux_req = (pwr_aux - cat_to_aux).max(si::Power::ZERO);
                self.res
                    .solve_energy_consumption(res_prop_req, res_aux_req, dt)?;
            } else {
                // Braking/coasting: keep the catenary idle (grid draw zero) so
                // behavior matches the no-catenary configuration -- the battery
                // absorbs regen up to its charge limit and the rest is dissipated
                // by the dynamic brake. (Regen-to-grid is intentionally not
                // implemented here.)
                cat_int.set_pwr_in_req(si::Power::ZERO, dt)?;
                self.res.solve_energy_consumption(
                    pwr_elec_prop_in,
                    pwr_aux
                        .min(
                            *self.res.state.pwr_prop_max.get_fresh(|| format_dbg!())?
                                - pwr_elec_prop_in,
                        )
                        .max(si::Power::ZERO),
                    dt,
                )?;
            }
        } else if pwr_elec_prop_in > si::Power::ZERO {
            // No catenary, positive traction
            self.res
                .solve_energy_consumption(pwr_elec_prop_in, pwr_aux, dt)?;
        } else {
            // No catenary, negative traction
            self.res.solve_energy_consumption(
                pwr_elec_prop_in,
                // limit aux power to whatever is actually available
                pwr_aux
                    // whatever power is available from regen plus normal
                    .min(
                        *self.res.state.pwr_prop_max.get_fresh(|| format_dbg!())?
                            - pwr_elec_prop_in,
                    )
                    .max(si::Power::ZERO),
                dt,
            )?;
        }
        Ok(())
    }
}

impl Mass for BatteryElectricLoco {
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
            stringify!(BatteryElectricLoco)
        ))
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.res.mass().with_context(|| format_dbg!())
    }

    fn expunge_mass_fields(&mut self) {
        self.res.expunge_mass_fields();
    }
}

impl Init for BatteryElectricLoco {
    fn init(&mut self) -> Result<(), Error> {
        self.res.init()?;
        self.edrv.init()?;
        self.pt_cntrl.init()?;
        if let Some(cat_int) = self.cat_int.as_mut() {
            cat_int.init()?;
        }
        Ok(())
    }
}
impl SerdeAPI for BatteryElectricLoco {}

impl LocoTrait for BatteryElectricLoco {
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

    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        _elev_and_temp: Option<(si::Length, si::ThermodynamicTemperature)>,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let mass_for_loco: si::Mass = train_mass.with_context(|| {
            format!(
                "{}\n`train_mass_for_loco` must be provided for `BatteryElectricLoco` ",
                format_dbg!()
            )
        })?;
        let train_speed: si::Velocity = train_speed.with_context(|| {
            format!(
                "{}\n`train_speed` must be provided for `BatteryElectricLoco` ",
                format_dbg!()
            )
        })?;

        let disch_buffer: si::Energy = match &self.pt_cntrl {
            BatteryPowertrainControls::RGWDB(rgwb) => {
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
            BatteryPowertrainControls::RGWDB(rgwb) => {
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

        self.res.set_curr_pwr_out_max(
            dt,
            pwr_aux.with_context(|| anyhow!(format_dbg!("`pwr_aux` not provided")))?,
            disch_buffer,
            chrg_buffer,
        )?;
        // If a catenary interface is present, update its max deliverable power and
        // add it to the drivetrain's max propulsion input, so the locomotive can
        // draw traction power from the catenary in addition to the battery.
        let cat_pwr_out_max = if let Some(cat_int) = self.cat_int.as_mut() {
            let pwr_elec_in_max =
                *cat_int.state.pwr_elec_in_max.get_fresh(|| format_dbg!())?;
            cat_int.set_cur_pwr_max_out(pwr_elec_in_max, None)?;
            *cat_int.state.pwr_elec_out_max.get_fresh(|| format_dbg!())?
        } else {
            si::Power::ZERO
        };
        self.edrv.set_cur_pwr_max_out(
            *self.res.state.pwr_prop_max.get_fresh(|| format_dbg!())? + cat_pwr_out_max,
            None,
        )?;
        self.edrv
            .set_cur_pwr_regen_max(*self.res.state.pwr_charge_max.get_fresh(|| format_dbg!())?)?;

        // power rate is never limiting in BEL, but assuming dt will be same
        // in next time step, we can synthesize a rate
        self.edrv.set_pwr_rate_out_max(
            (*self
                .edrv
                .state
                .pwr_mech_out_max
                .get_fresh(|| format_dbg!())?
                - *self
                    .edrv
                    .state
                    .pwr_mech_prop_out
                    .get_stale(|| format_dbg!())?)
                / dt,
        )?;
        Ok(())
    }

    fn get_energy_loss(&self) -> anyhow::Result<si::Energy> {
        let cat_energy_loss = match self.cat_int.as_ref() {
            Some(cat_int) => *cat_int.state.energy_loss.get_fresh(|| format_dbg!())?,
            None => si::Energy::ZERO,
        };
        Ok(*self.res.state.energy_loss.get_fresh(|| format_dbg!())?
            + *self.edrv.state.energy_loss.get_fresh(|| format_dbg!())?
            + cat_energy_loss)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, IsVariant, From, TryInto)]
pub enum BatteryPowertrainControls {
    /// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
    /// and discharge power inside of static min and max SOC range.  Also, includes
    /// buffer for forcing [FuelConverter] to be active/on.
    RGWDB(Box<RESGreedyWithDynamicBuffersBEL>),
}

impl Default for BatteryPowertrainControls {
    fn default() -> Self {
        Self::RGWDB(Default::default())
    }
}

impl Init for BatteryPowertrainControls {
    fn init(&mut self) -> Result<(), Error> {
        match self {
            Self::RGWDB(rgwb) => rgwb.init()?,
        }
        Ok(())
    }
}

impl SetCumulative for BatteryPowertrainControls {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => {
                rgwdb.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?
            }
        }
        Ok(())
    }
}

impl Step for BatteryPowertrainControls {
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => rgwdb.step(|| format!("{}\n{}", loc(), format_dbg!()))?,
        }
        Ok(())
    }
}

impl SaveState for BatteryPowertrainControls {
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => rgwdb.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?,
        }
        Ok(())
    }
}

impl CheckAndResetState for BatteryPowertrainControls {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwdb) => {
                rgwdb.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
        }
        Ok(())
    }
}

impl StateMethods for BatteryPowertrainControls {}

/// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
/// and discharge power inside of static min and max SOC range.  Also, includes
/// buffer for forcing [FuelConverter] to be active/on. See [Self::init] for
/// default values.
#[serde_api]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default, StateMethods, SetCumulative)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
#[non_exhaustive]
pub struct RESGreedyWithDynamicBuffersBEL {
    /// RES energy delta from minimum SOC corresponding to kinetic energy of
    /// vehicle at this speed that triggers ramp down in RES discharge.
    pub speed_soc_disch_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of accel buffer
    pub speed_soc_disch_buffer_coeff: Option<si::Ratio>,
    /// RES energy delta from maximum SOC corresponding to kinetic energy of
    /// vehicle at current speed minus kinetic energy of vehicle at this speed
    /// triggers ramp down in RES discharge
    pub speed_soc_regen_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of regen buffer
    pub speed_soc_regen_buffer_coeff: Option<si::Ratio>,
    #[serde(default)]
    pub state: RGWDBStateBEL,
    #[serde(default)]
    /// history of current state
    pub history: RGWDBStateBELHistoryVec,
}

#[pyo3_api]
impl RESGreedyWithDynamicBuffersBEL {}

impl Init for RESGreedyWithDynamicBuffersBEL {
    fn init(&mut self) -> Result<(), Error> {
        init_opt_default!(self, speed_soc_disch_buffer, 40.0 * uc::MPH);
        init_opt_default!(self, speed_soc_disch_buffer_coeff, 1.0 * uc::R);
        init_opt_default!(self, speed_soc_regen_buffer, 10. * uc::MPH);
        init_opt_default!(self, speed_soc_regen_buffer_coeff, 1.0 * uc::R);
        Ok(())
    }
}
impl SerdeAPI for RESGreedyWithDynamicBuffersBEL {}

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
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// State for [RESGreedyWithDynamicBuffers ]
pub struct RGWDBStateBEL {
    /// time step index
    pub i: TrackedState<usize>,
}

#[pyo3_api]
impl RGWDBStateBEL {}

impl Init for RGWDBStateBEL {}
impl SerdeAPI for RGWDBStateBEL {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive one time step of a BEL through the standard per-step call sequence.
    fn step_bel(
        bel: &mut BatteryElectricLoco,
        cat_power_limit: Option<si::Power>,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) {
        let train_mass = Some(1_000e3 * uc::KG);
        let train_speed = Some(20.0 * uc::MPS);
        bel.check_and_reset(|| format_dbg!()).unwrap();
        bel.step(|| format_dbg!()).unwrap();
        bel.set_cat_power_limit(cat_power_limit, None).unwrap();
        bel.set_curr_pwr_max_out(Some(pwr_aux), None, train_mass, train_speed, dt)
            .unwrap();
        bel.solve_energy_consumption(pwr_out_req, dt, pwr_aux).unwrap();
    }

    #[test]
    fn test_bel_catenary_covers_traction_and_charges_battery() {
        let mut bel = BatteryElectricLoco::default();
        bel.cat_int = Some(CatenaryInterface::new(10_000e3, Some(1)).unwrap());
        bel.init().unwrap();
        let dt = 1.0 * uc::S;
        let pwr_aux = 100e3 * uc::W;
        let pwr_out_req = 1_000e3 * uc::W;

        // Catenary limit well above total demand: catenary should cover all
        // propulsion (battery not discharging for traction).
        step_bel(&mut bel, Some(10_000e3 * uc::W), pwr_out_req, pwr_aux, dt);

        let cat = bel.cat_int.as_ref().unwrap();
        let cat_out = *cat.state.pwr_elec_out.get_fresh(|| format_dbg!()).unwrap();
        assert!(
            cat_out > si::Power::ZERO,
            "catenary should deliver power when a limit is available"
        );
        let res_prop = *bel
            .res
            .state
            .pwr_out_propulsion
            .get_fresh(|| format_dbg!())
            .unwrap();
        assert!(
            res_prop <= 1e-3 * uc::W,
            "battery should not discharge for propulsion when catenary covers it (got {} W)",
            res_prop.get::<si::watt>()
        );
    }

    #[test]
    fn test_bel_no_catenary_power_uses_battery() {
        let mut bel = BatteryElectricLoco::default();
        bel.cat_int = Some(CatenaryInterface::new(10_000e3, Some(1)).unwrap());
        bel.init().unwrap();
        let dt = 1.0 * uc::S;
        let pwr_aux = 100e3 * uc::W;
        let pwr_out_req = 1_000e3 * uc::W;

        // Catenary present but power limit is zero (e.g. unelectrified section):
        // the battery must supply the full propulsion, matching the no-catenary
        // configuration.
        step_bel(&mut bel, Some(si::Power::ZERO), pwr_out_req, pwr_aux, dt);

        let cat = bel.cat_int.as_ref().unwrap();
        let cat_out = *cat.state.pwr_elec_out.get_fresh(|| format_dbg!()).unwrap();
        assert_eq!(
            cat_out.get::<si::watt>(),
            0.0,
            "catenary should deliver no power when the limit is zero"
        );
        let res_prop = *bel
            .res
            .state
            .pwr_out_propulsion
            .get_fresh(|| format_dbg!())
            .unwrap();
        assert!(
            res_prop > si::Power::ZERO,
            "battery must supply propulsion when catenary power is unavailable"
        );
    }
}
