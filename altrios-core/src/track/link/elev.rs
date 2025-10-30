use crate::imports::*;

/// Struct containing elevation for a particular offset w.r.t. `Link`
#[serde_api]
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Elev {
    #[serde(alias = "offset")]
    pub offset: si::Length,
    #[serde(alias = "elev")]
    pub elev: si::Length,
}

#[pyo3_api]
impl Elev {
    #[new]
    fn __new__(offset_meters: f64, elev_meters: f64) -> PyResult<Self> {
        Ok(Self::new(offset_meters * uc::M, elev_meters * uc::M))
    }
}

impl Init for Elev {}
impl SerdeAPI for Elev {}

impl Elev {
    pub fn new(offset: si::Length, elev: si::Length) -> Self {
        Self { offset, elev }
    }
}

impl Valid for Elev {}

impl ObjState for Elev {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        si_chk_num_fin(&mut errors, &self.elev, "Elevation value");
        errors.make_err()
    }
}

impl ObjState for Vec<Elev> {
    fn is_fake(&self) -> bool {
        (**self).is_fake()
    }
    fn validate(&self) -> ValidationResults {
        (**self).validate()
    }
}

impl Valid for Vec<Elev> {
    fn valid() -> Self {
        let offset_end = uc::M * 10000.0;
        let elev = uc::M * 20.0;
        vec![
            Elev::valid(),
            Elev {
                offset: offset_end * 0.5,
                elev,
            },
            Elev {
                offset: offset_end,
                ..Elev::valid()
            },
        ]
    }
}

impl ObjState for [Elev] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }

    fn validate(&self) -> ValidationResults {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Elevation");
        if self.len() < 2 {
            errors.push(anyhow!("There must be at least two elevations!"));
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
mod test_elev {
    use super::*;
    use crate::testing::*;

    impl Cases for Elev {
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
                    elev: uc::M * f64::NEG_INFINITY,
                    ..Self::valid()
                },
                Self {
                    elev: uc::M * f64::INFINITY,
                    ..Self::valid()
                },
                Self {
                    elev: uc::M * f64::NAN,
                    ..Self::valid()
                },
            ]
        }
    }
    check_cases!(Elev);
}

#[cfg(test)]
mod test_elevs {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<Elev> {
        fn fake_cases() -> Vec<Self> {
            vec![vec![]]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![vec![Elev::valid()]]
        }
    }
    check_cases!(Vec<Elev>);
    check_vec_elems!(Elev);
    check_vec_sorted!(Elev);
    check_vec_duplicates!(Elev);

    #[test]
    fn check_duplicates() {
        for mut case in Vec::<Elev>::real_cases() {
            case.push(*case.last().unwrap());
            case.validate().unwrap_err();
            case.last_mut().unwrap().elev += uc::M;
            case.validate().unwrap_err();
            case.last_mut().unwrap().offset += uc::M;
            case.validate().unwrap();
            case.last_mut().unwrap().elev -= uc::M;
            case.validate().unwrap();
        }
    }
}
