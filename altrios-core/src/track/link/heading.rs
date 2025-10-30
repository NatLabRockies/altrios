use crate::imports::*;

/// Struct containing heading for a particular offset w.r.t. `Link`
#[serde_api]
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Heading {
    #[serde(alias = "offset")]
    pub offset: si::Length,

    #[serde(alias = "heading")]
    pub heading: si::Angle,
    /// Optional latitude at `self.offset`.  No checks are currently performed to ensure consistency
    /// between headind and lat/lon, and this is not actually used in the code.  
    #[serde(alias = "Lat")]
    pub lat: Option<f64>,
    /// Optional longitude at `self.offset`.  No checks are currently performed to ensure
    /// consistency between headind and lat/lon, and this is not actually used in the code.
    #[serde(alias = "Lon")]
    pub lon: Option<f64>,
}

#[pyo3_api]
impl Heading {}

impl Init for Heading {}
impl SerdeAPI for Heading {}

impl Valid for Heading {}

impl ObjState for Heading {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        si_chk_num_gez(&mut errors, &self.heading, "Heading");
        if self.heading >= uc::REV {
            errors.push(anyhow!(
                "Heading = {:?} must be less than one revolution (2*pi radians)!",
                self.heading,
            ));
        }
        errors.make_err()
    }
}

impl ObjState for Vec<Heading> {
    fn is_fake(&self) -> bool {
        (**self).is_fake()
    }
    fn validate(&self) -> ValidationResults {
        (**self).validate()
    }
}

impl Valid for Vec<Heading> {
    fn valid() -> Self {
        let offset_end = uc::M * 10000.0;
        vec![
            Heading::valid(),
            Heading {
                offset: offset_end * 0.5,
                heading: si::Angle::ZERO,
                lat: Default::default(),
                lon: Default::default(),
            },
            Heading {
                offset: offset_end,
                heading: uc::RAD,
                lat: Default::default(),
                lon: Default::default(),
            },
        ]
    }
}

impl ObjState for [Heading] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }

    fn validate(&self) -> ValidationResults {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Heading");
        if self.len() < 2 {
            errors.push(anyhow!("There must be at least two headings!"));
        }
        if !self.windows(2).all(|w| w[0].offset < w[1].offset) {
            let err_pairs: Vec<Vec<si::Length>> = self
                .windows(2)
                .filter(|w| w[0].offset >= w[1].offset)
                .map(|w| vec![w[0].offset, w[1].offset])
                .collect();
            errors.push(anyhow!(
                "Offsets must be sorted and unique! Invalid offsets: {:?}",
                err_pairs
            ));
        }
        errors.make_err()
    }
}

#[cfg(test)]
mod test_heading {
    use super::*;
    use crate::testing::*;

    impl Cases for Heading {
        fn real_cases() -> Vec<Self> {
            vec![
                Self::valid(),
                Self {
                    offset: uc::M,
                    ..Self::valid()
                },
                Self {
                    offset: uc::M * f64::INFINITY,
                    ..Self::valid()
                },
                Self {
                    heading: si::Angle::ZERO,
                    ..Self::valid()
                },
            ]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                Self {
                    offset: uc::M * f64::NEG_INFINITY,
                    ..Self::valid()
                },
                Self {
                    offset: -uc::M,
                    ..Self::valid()
                },
                Self {
                    offset: uc::M * f64::NAN,
                    ..Self::valid()
                },
                Self {
                    heading: uc::REV,
                    ..Self::valid()
                },
                Self {
                    heading: uc::RAD * -0.00000001,
                    ..Self::valid()
                },
                Self {
                    heading: uc::RAD * f64::NAN,
                    ..Self::valid()
                },
            ]
        }
    }
    check_cases!(Heading);
}

#[cfg(test)]
mod test_headings {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<Heading> {
        fn fake_cases() -> Vec<Self> {
            vec![vec![]]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![vec![Heading::valid()]]
        }
    }
    check_cases!(Vec<Heading>);
    check_vec_elems!(Heading);
    check_vec_sorted!(Heading);
    check_vec_duplicates!(Heading);

    #[test]
    fn check_duplicates() {
        for mut case in Vec::<Heading>::real_cases() {
            case.push(*case.last().unwrap());
            case.validate().unwrap_err();
            case.last_mut().unwrap().heading += uc::RAD;
            case.validate().unwrap_err();
            case.last_mut().unwrap().offset += uc::M;
            case.validate().unwrap();
            case.last_mut().unwrap().heading -= uc::RAD;
            case.validate().unwrap();
        }
    }
}
