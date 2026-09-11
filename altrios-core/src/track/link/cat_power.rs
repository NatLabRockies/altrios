use crate::imports::*;

#[serde_api]
#[derive(Clone, Default, Debug, PartialEq, PartialOrd, Deserialize, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq, from_py_object))]
/// Struct representing local train-level power limits for catenary charging
pub struct CatPowerLimit {
    /// start of current power limit
    pub offset_start: si::Length,
    /// end of current power limit
    pub offset_end: si::Length,
    /// maximum possible catenary charging rate  
    /// assumed to be identical for charging and discharging
    pub power_limit: si::Power,
    /// Effective series resistance (ohms) of the catenary loop between this
    /// energized zone and its feeding substation network, used to compute
    /// resistive line loss as `resistance_to_source * (pwr / system_voltage)^2`.
    #[serde(default)]
    pub resistance_to_source: si::ElectricalResistance,
    /// Effective series reactance (ohms) of the catenary loop between this
    /// energized zone and its feeding substation network. Affects voltage drop
    /// (hence voltage-limited capacity) but not resistive energy loss.
    #[serde(default)]
    pub reactance_to_source: si::ElectricalResistance,
    /// Nominal catenary voltage applied to this energized zone, used with
    /// [Self::resistance_to_source] to compute resistive line loss as
    /// `resistance_to_source * (pwr / voltage)^2`. Normally stamped from the
    /// network-level `system_voltage`. A non-positive value disables loss.
    #[serde(default)]
    pub voltage: si::ElectricPotential,
    /// Optional user-defined catenary district (the primary feeding substation;
    /// the one supplying the larger share when fed from two sides).
    pub district_id: Option<String>,
    /// Optional second feeding substation/district when this zone is fed from
    /// both sides (between two substations). Used only in postprocessing to
    /// apportion load and emissions between the two grids.
    #[serde(default)]
    pub district_id_secondary: Option<String>,
    /// Fraction (0..1) of the substation load supplied by the primary
    /// [Self::district_id] when fed from two sides (the remainder comes from
    /// [Self::district_id_secondary]). ``None`` means single-sided (all primary).
    #[serde(default)]
    pub feed_fraction: Option<f64>,
}

#[pyo3_api]
impl CatPowerLimit {}

impl Init for CatPowerLimit {}
impl SerdeAPI for CatPowerLimit {}

impl Valid for CatPowerLimit {
    fn valid() -> Self {
        Self {
            offset_start: si::Length::ZERO,
            offset_end: uc::M * 10000.0,
            power_limit: uc::W * 5.0e6,
            resistance_to_source: si::ElectricalResistance::ZERO,
            reactance_to_source: si::ElectricalResistance::ZERO,
            voltage: uc::KV * 25.0,
            district_id: None,
            district_id_secondary: None,
            feed_fraction: None,
        }
    }
}

impl ObjState for CatPowerLimit {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset_start, "Offset start");
        si_chk_num_gez(&mut errors, &self.offset_end, "Offset end");
        si_chk_num_gez(&mut errors, &self.power_limit, "Power limit");
        si_chk_num_gez(&mut errors, &self.resistance_to_source, "Resistance to source");
        si_chk_num_gez(&mut errors, &self.reactance_to_source, "Reactance to source");
        si_chk_num_gez(&mut errors, &self.voltage, "Voltage");
        if self.offset_start > self.offset_end {
            errors.push(anyhow!(
                "Offset end = {:?} must be at least equal to offset start = {:?}!",
                self.offset_end,
                self.offset_start
            ));
        }
        errors.make_err()
    }
}

impl ObjState for Vec<CatPowerLimit> {
    fn is_fake(&self) -> bool {
        (**self).is_fake()
    }
    fn validate(&self) -> ValidationResults {
        (**self).validate()
    }
}

impl Valid for Vec<CatPowerLimit> {
    fn valid() -> Self {
        vec![CatPowerLimit::valid()]
    }
}

impl ObjState for [CatPowerLimit] {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Catenary power limit");
        early_err!(errors, "Catenary power limits");

        // Offsets are expected sorted by `offset_start`. Consecutive limits
        // overlap when the next one starts before the previous one ends;
        // properly separated (or touching) limits are valid, since a gap between
        // limits simply means no catenary power there.
        if self
            .windows(2)
            .any(|w| w[1].offset_start < w[0].offset_end)
        {
            errors.push(anyhow!(
                "Catenary power limit offset pairs must be non-overlapping!"
            ));
        }
        errors.make_err()
    }
}
