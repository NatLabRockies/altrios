use crate::imports::*;

#[serde_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct RailVehicle {
    /// Unique user-defined identifier for the car type.  This should include
    /// meta information about the car type's weight, e.g. `loaded`, `empty`, `partial`.
    #[serde(alias = "Car Type")]
    pub car_type: String,

    /// Identifier for the freight type carried by this car type (e.g., Intermodal).
    #[serde(alias = "Freight Type")]
    pub freight_type: String,

    /// Railcar length (between pulling-faces)
    #[serde(alias = "Length (m)")]
    pub length: si::Length,
    /// Railcar axle count (typically 4)
    #[serde(alias = "Axle Count")]
    pub axle_count: u8,
    /// Brake valve count (typically 1)
    #[serde(alias = "Brake Count")]
    pub brake_count: u8,

    /// Railcar mass, not including freight
    #[serde(alias = "Mass Static Base (kg)")]
    pub mass_static_base: si::Mass,
    /// Freight component of total static railcar mass
    #[serde(alias = "Mass Freight (kg)")]
    pub mass_freight: si::Mass,
    /// Railcar speed limit
    #[serde(alias = "Speed Max (m/s)")]
    pub speed_max: si::Velocity,
    /// Braking ratio -- braking force per rail vehicle weight
    #[serde(alias = "Braking Ratio")]
    pub braking_ratio: si::Ratio,

    /// Additional mass value to adjust for rotating mass in wheels and axles (typically 1,500 lbs or 680 kg)
    #[serde(alias = "Mass Extra per Axle (kg)")]
    pub mass_rot_per_axle: si::Mass,
    /// Bearing resistance as force
    #[serde(alias = "Bearing Res per Axle (N)")]
    pub bearing_res_per_axle: si::Force,
    /// Rolling resistance ratio (lb/ton is customary, lb/lb internal to code).
    /// This is the rolling resistance force per weight for each axle.
    #[serde(alias = "Rolling Ratio")]
    pub rolling_ratio: si::Ratio,
    /// Davis B coefficient (typically very close to zero)
    #[serde(alias = "Davis B (s/m)")]
    pub davis_b: si::InverseVelocity,
    /// Drag area (Cd*A), where Cd is drag coefficient and A is front cross-sectional area
    #[serde(alias = "Cd*A (m^2)")]
    pub cd_area: si::Area,
    /// Curve coefficient 0
    #[serde(alias = "Curve Coefficient 0")]
    pub curve_coeff_0: si::Ratio,
    /// Curve coefficient 1
    #[serde(alias = "Curve Coefficient 1")]
    pub curve_coeff_1: si::Ratio,
    /// Curve coefficient 2
    #[serde(alias = "Curve Coefficient 2")]
    pub curve_coeff_2: si::Ratio,
}

#[pyo3_api]
impl RailVehicle {}

impl Init for RailVehicle {}
impl SerdeAPI for RailVehicle {}

impl Mass for RailVehicle {
    /// Static mass of rail vehicle, not including effective rotational mass
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.derived_mass()
    }

    fn set_mass(
        &mut self,
        _new_mass: Option<si::Mass>,
        _side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        bail!("`set_mass` is not enabled for `RailVehicle`")
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(Some(self.mass_static_base + self.mass_freight))
    }

    fn expunge_mass_fields(&mut self) {}
}
