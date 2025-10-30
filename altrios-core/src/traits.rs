use crate::imports::*;
pub mod serde_api;
pub use serde_api::*;

///Standardizes conversion from smaller than usize types for indexing.
pub trait Idx {
    fn idx(self) -> usize;
}

#[duplicate_item(Self; [u8]; [u16])]
impl Idx for Self {
    fn idx(self) -> usize {
        self.into()
    }
}

impl Idx for u32 {
    fn idx(self) -> usize {
        self.try_into().unwrap()
    }
}

impl Idx for Option<NonZeroU16> {
    fn idx(self) -> usize {
        self.map(u16::from).unwrap_or(0) as usize
    }
}

/// Trait implemented for indexing types, specifically `usize`, to assist in
/// converting them into an `Option<NonZeroUxxxx>`.
///
/// This is necessary because both the common type conversion trait ([From])
/// and the types involved (e.g. `Option<NonZeroU16>` ) are from the
/// standard library.  Rust does not allow for the combination of traits and
/// types if both are from external libraries.
///
/// This trait is default implemented for [usize] to convert into any type that
/// already implements [TryFrom]<[NonZeroUsize]>, which includes most of the
/// NonZeroUxxxx types.
///
/// This approach will error on a loss of precision.  So, if the [usize] value
/// does not fit into the `NonZero` type, then an [Error] variant is returned.
///
/// If the value is `0`, then an [Ok]\([None]) variant is returned.
///
/// Note that the base `NonZero` types already have a similar feature if
/// converting from the same basic type (e.g. [u16] to [Option]<[NonZeroU16]>),
/// where the type is guaranteed to fit, but just might be `0`.  If that is the
/// use case, then just use the `new()` method, which returns the [None] variant
/// if the value is `0`.
///
/// The intended usage is as follows:
/// ```
/// # use std::num::NonZeroU8;
/// # use altrios_core::traits::TryFromIdx;
///
/// let non_zero : usize = 42;
/// let good_val : Option<NonZeroU8> = non_zero.try_from_idx().unwrap();
/// assert!(good_val == NonZeroU8::new(42));
///
/// let zero : usize = 0;
/// let none_val : Option<NonZeroU8> = zero.try_from_idx().unwrap();
/// assert!(none_val == None);
///
/// let too_big : usize = 256;
/// let bad_val : Result<Option<NonZeroU8>, _> = too_big.try_from_idx();
/// assert!(bad_val.is_err());
/// ```
pub trait TryFromIdx<T> {
    type Error;

    fn try_from_idx(&self) -> Result<Option<T>, Self::Error>;
}

impl<T> TryFromIdx<T> for usize
where
    T: TryFrom<NonZeroUsize>,
{
    type Error = <T as TryFrom<NonZeroUsize>>::Error;

    fn try_from_idx(&self) -> Result<Option<T>, Self::Error> {
        NonZeroUsize::new(*self).map_or(
            // If value is a 0-valued usize, then we immediately return an
            // Ok(None).
            Ok(None),
            // Otherwise we attempt to convert it from a NonZeroUsize into a
            // different type with potentially smaller accuracy.
            |val| {
                T::try_from(val)
                    // We wrap a valid result in Some
                    .map(Some)
            },
        )
    }
}

pub trait Linspace {
    fn linspace(start: f64, stop: f64, n_elements: usize) -> Vec<f64> {
        let n_steps = n_elements - 1;
        let step_size = (stop - start) / n_steps as f64;
        let v_norm: Vec<f64> = (0..=n_steps)
            .collect::<Vec<usize>>()
            .iter()
            .map(|x| *x as f64)
            .collect();
        let v = v_norm.iter().map(|x| (x * step_size) + start).collect();
        v
    }
}

impl Linspace for Vec<f64> {}

/// Provides method for checking if an instance of `Self` is equal to `Self::default`
pub trait EqDefault: Default + PartialEq {
    /// Checks if an instance of `Self` is equal to `Self::default`
    fn eq_default(&self) -> bool {
        *self == Self::default()
    }
}
impl<T: Default + PartialEq> EqDefault for T {}

#[derive(Default, Deserialize, Serialize, Debug, Clone, PartialEq)]
/// Governs which side effect to trigger when setting mass
pub enum MassSideEffect {
    /// To be used when [MassSideEffect] is not applicable
    #[default]
    None,
    /// Set the extensive parameter -- e.g. energy, power -- as a side effect
    Extensive,
    /// Set the intensive parameter -- e.g. specific power, specific energy -- as a side effect
    Intensive,
}

impl TryFrom<String> for MassSideEffect {
    type Error = anyhow::Error;
    fn try_from(value: String) -> anyhow::Result<MassSideEffect> {
        let mass_side_effect = match value.as_str() {
            "None" => Self::None,
            "Extensive" => Self::Extensive,
            "Intensive" => Self::Intensive,
            _ => {
                bail!(format!(
                    "`MassSideEffect` must be 'Intensive', 'Extensive', or 'None'. "
                ))
            }
        };
        Ok(mass_side_effect)
    }
}

pub trait Mass {
    /// Returns mass of Self, either from `self.mass` or
    /// the derived from fields that store mass data. `Mass::mass` also checks that
    /// derived mass, if Some, is same as `self.mass`.
    fn mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets component mass to `mass`, or if `None` is provided for `mass`,
    /// sets mass based on other component parameters (e.g. power and power
    /// density, sum of fields containing mass)
    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()>;

    /// Returns derived mass (e.g. sum of mass fields, or
    /// calculation involving mass specific properties).  If
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets all fields that are used in calculating derived mass to `None`.
    /// Does not touch `self.mass`.
    fn expunge_mass_fields(&mut self);

    /// Sets any mass-specific property with appropriate side effects
    fn set_mass_specific_property(&mut self) -> anyhow::Result<()> {
        // TODO: remove this default implementation when this method has been universally implemented.
        // For structs without specific properties, return an error
        Ok(())
    }
}

/// Super trait to ensure that related traits are implemented together
pub trait StateMethods: SetCumulative + SaveState + Step + CheckAndResetState {}

/// Trait for setting cumulative values based on rate values
pub trait SetCumulative {
    /// Sets cumulative values based on rate values
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()>;
}

/// Provides method that saves `self.state` to `self.history` and propagates to any fields with
/// `state`
pub trait SaveState {
    /// Saves `self.state` to `self.history` and propagates to any fields with `state`
    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()>;
}

/// Trait that provides method for incrementing `i` field of this and all contained structs,
/// recursively
pub trait Step {
    /// Increments `i` field of this and all contained structs, recursively
    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()>;
}

/// Provides methods for getting and setting the save interval
pub trait HistoryMethods: SaveState {
    /// Recursively sets save interval
    /// # Arguments
    /// - `save_interval`: time step interval at which to save `self.state` to `self.history`
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()>;
    /// Returns save interval for `self` but does not guarantee recursive consistency in nested
    /// objects
    fn save_interval(&self) -> anyhow::Result<Option<usize>>;
    /// Remove all history
    fn clear(&mut self);
}
