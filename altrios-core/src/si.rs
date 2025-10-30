//! Import uom si system and add unit constants
//! Zero values should be created using standard uom syntax ($Quantity::ZERO) after adding "use crate::imports::*"
//! Non-zero values should be created using standard uom syntax ($Quantity::new::<$unit>($value)) or multiplication syntax ($value * $UNIT_CONSTANT)

use uom::si;

pub use si::angle::degree;
pub use si::area::square_meter;
pub use si::available_energy::{joule_per_kilogram, kilojoule_per_kilogram};
pub use si::curvature::{degree_per_meter, radian_per_meter};
pub use si::energy::{joule, watt_hour};
pub use si::f64::{
    Acceleration, Angle, Area, AvailableEnergy as SpecificEnergy, Curvature, Energy, Force,
    Frequency, InverseVelocity, Length, Mass, MassDensity, Power, PowerRate, Pressure, Ratio,
    SpecificHeatCapacity, SpecificPower, TemperatureInterval, ThermodynamicTemperature, Time,
    Velocity, Volume,
};
pub use si::force::{newton, pound_force};
pub use si::length::{foot, kilometer, meter};
pub use si::mass::{kilogram, megagram};
pub use si::power::{kilowatt, megawatt, watt};
pub use si::power_rate::watt_per_second;
pub use si::ratio::{percent, ratio};
pub use si::specific_power::kilowatt_per_kilogram;
pub use si::thermodynamic_temperature::{degree_celsius, kelvin};
pub use si::time::{hour, second};
pub use si::volume::cubic_meter;
