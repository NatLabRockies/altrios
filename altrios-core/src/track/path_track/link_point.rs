use super::super::LinkIdx;
use crate::imports::*;

/// Point along PathTpc representing the start of a link and the number of grade, curve, and cat
/// power limit points contained within the same link,`link_idx`, in the PathTpc.
///
/// Note that for the `*_count` fields, these represent points contained in the link for which grade,
/// curve, ... information is known, not including the final point in the link.
#[serde_api]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct LinkPoint {
    /// Distance relative to the start of the PathTpc where `link_idx` starts
    pub offset: si::Length,

    /// Number of grade points in the current link
    pub grade_count: usize,

    /// Number of curve points in the current link
    pub curve_count: usize,

    /// Number of catenary power limit points in the current link
    pub cat_power_count: usize,

    /// [LinkIdx] of current link
    pub link_idx: LinkIdx,
}

#[pyo3_api]
impl LinkPoint {}

impl Init for LinkPoint {}
impl SerdeAPI for LinkPoint {}

impl LinkPoint {
    pub fn add_counts(&mut self, other: &Self) {
        self.grade_count += other.grade_count;
        self.curve_count += other.curve_count;
        self.cat_power_count += other.cat_power_count;
    }
}

impl Valid for LinkPoint {
    fn valid() -> Self {
        Self {
            offset: uc::M * 10000.0,
            grade_count: 2,
            curve_count: 2,
            cat_power_count: 0,
            link_idx: LinkIdx::default(),
        }
    }
}

impl ObjState for LinkPoint {
    fn validate(&self) -> Result<(), crate::validate::ValidationErrors> {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        errors.make_err()
    }
}

impl ObjState for Vec<LinkPoint> {
    fn is_fake(&self) -> bool {
        (**self).is_fake()
    }
    fn validate(&self) -> ValidationResults {
        (**self).validate()
    }
}

impl GetOffset for LinkPoint {
    fn get_offset(&self) -> si::Length {
        self.offset
    }
}

impl Valid for Vec<LinkPoint> {
    fn valid() -> Self {
        vec![
            LinkPoint {
                link_idx: LinkIdx::valid(),
                ..LinkPoint::default()
            },
            LinkPoint::valid(),
        ]
    }
}

impl ObjState for [LinkPoint] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }
    fn validate(&self) -> Result<(), crate::validate::ValidationErrors> {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Link point");
        if self.len() < 2 {
            errors.push(anyhow!("There must be at least two link points!"));
        }
        early_err!(errors, "Link points");

        for link_point in &self[..(self.len() - 1)] {
            if link_point.link_idx.is_fake() {
                errors.push(anyhow!(
                    "All link point link indices (except for the last one) must be real!"
                ));
            }
        }
        if self.last().unwrap().link_idx.is_real() {
            errors.push(anyhow!("Last link point link index must be fake!"));
        }

        if !self.windows(2).all(|w| w[0].offset < w[1].offset) {
            let err_pairs: Vec<Vec<si::Length>> = self
                .windows(2)
                .filter(|w| w[0].offset >= w[1].offset)
                .map(|w| vec![w[0].offset, w[1].offset])
                .collect();
            errors.push(anyhow!(
                "Link point offsets must be sorted and unique! Invalid offsets: {:?}",
                err_pairs
            ));
        }

        errors.make_err()
    }
}

#[cfg(test)]
mod test_link_point {
    use super::*;
    use crate::testing::*;

    impl Cases for LinkPoint {
        fn real_cases() -> Vec<Self> {
            vec![Self::default(), Self::valid()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                Self {
                    offset: -uc::M,
                    ..Self::default()
                },
                Self {
                    offset: uc::M * f64::NAN,
                    ..Self::default()
                },
                Self {
                    offset: -uc::M,
                    ..Self::valid()
                },
                Self {
                    offset: uc::M * f64::NAN,
                    ..Self::valid()
                },
            ]
        }
    }

    check_cases!(LinkPoint);
}

#[cfg(test)]
mod test_link_points {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<LinkPoint> {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid(), {
                let mut base = Self::valid();
                base.push(*base.last().unwrap());
                base.last_mut().unwrap().offset += uc::M;
                let base_len = base.len();
                base[base_len - 2].link_idx = LinkIdx::valid();
                base
            }]
        }
        fn fake_cases() -> Vec<Self> {
            vec![vec![]]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                vec![LinkPoint::default()],
                vec![LinkPoint::valid()],
                vec![LinkPoint::valid(), LinkPoint::valid()],
                vec![LinkPoint::valid(), LinkPoint::default()],
                vec![LinkPoint::default(), LinkPoint::valid()],
                vec![LinkPoint::default(), LinkPoint::default()],
                Self::valid().into_iter().rev().collect::<Self>(),
                {
                    let mut base = Self::valid();
                    base.push(*base.last().unwrap());
                    base.last_mut().unwrap().offset += uc::M;
                    base
                },
                {
                    let mut base = Self::valid();
                    base.last_mut().unwrap().offset = base.first().unwrap().offset;
                    base
                },
                {
                    let mut base = Self::valid();
                    base.first_mut().unwrap().offset = base.last().unwrap().offset;
                    base
                },
                {
                    let mut base = Self::valid();
                    base.first_mut().unwrap().offset = base.last().unwrap().offset + uc::M;
                    base
                },
            ]
        }
    }
    check_cases!(Vec<LinkPoint>);
    check_vec_elems!(LinkPoint);
    check_vec_sorted!(LinkPoint);
    check_vec_duplicates!(LinkPoint);
}
