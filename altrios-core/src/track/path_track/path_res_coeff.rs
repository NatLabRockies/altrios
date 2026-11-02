use crate::imports::*;

#[serde_api]
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct containing linear resistance coefficients for a particular offset with respect to start
/// of `PathTpc`
pub struct PathResCoeff {
    /// Distance from start of `PathTpc`
    pub offset: si::Length,
    /// Represents non-dimensional grade resistance (aka grade) or curvature resistance.
    pub res_coeff: si::Ratio,
    /// Cumulative sum of `res_coeff` times length up to this `PathResCoeff` along `PathTpc`
    pub res_net: si::Length,
}

#[pyo3_api]
impl PathResCoeff {}

impl Init for PathResCoeff {}
impl SerdeAPI for PathResCoeff {}

impl PathResCoeff {
    /// Cumulative sum of `res_coeff` times length up to this `offset` along `PathTpc`
    pub fn calc_res_val(&self, offset: si::Length) -> si::Length {
        self.res_net + self.res_coeff * (offset - self.offset)
    }
}

impl GetOffset for PathResCoeff {
    fn get_offset(&self) -> si::Length {
        self.offset
    }
}

impl LinSearchHint for &[PathResCoeff] {
    /// # Arguments
    /// - `offset`: linear displacement of front of train from initial starting
    ///   position of back of train along entire PathTPC
    /// - `idx`: index of front (TODO: clarify this) of train within corresponding [PathResCoeff]
    /// - `dir`: direction of train along PathTPC
    fn calc_idx(&self, offset: si::Length, mut idx: usize, dir: &Dir) -> anyhow::Result<usize> {
        if dir != &Dir::Bwd {
            ensure!(
                offset <= self.last().unwrap().get_offset(),
                "{}\nOffset in forward direction larger than last slice offset at idx: {}!",
                format_dbg!(),
                idx
            );
            while self[idx + 1].get_offset() < offset {
                idx += 1;
            }
        } else if dir != &Dir::Fwd {
            ensure!(
                self.first().unwrap().get_offset() <= offset,
                "{}\nOffset in reverse direction ({:.2} m) is smaller than first slice offset ({:.2} m) at idx: {}! \
                This usually means the train's back end extends beyond the beginning of the path. \
                Ensure the initial link(s) provide enough length for the train.",
                format_dbg!(),
                offset.get::<si::meter>(),
                self.first().unwrap().get_offset().get::<si::meter>(),
                idx
            );
            while offset < self[idx].get_offset() {
                idx -= 1;
            }
        }
        Ok(idx)
    }
}

impl Valid for PathResCoeff {}

impl ObjState for PathResCoeff {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        si_chk_num_fin(&mut errors, &self.res_net, "Res net");
        si_chk_num_fin(&mut errors, &self.res_coeff, "Res coeff");
        errors.make_err()
    }
}

impl ObjState for Vec<PathResCoeff> {
    fn is_fake(&self) -> bool {
        (**self).is_fake()
    }
    fn validate(&self) -> ValidationResults {
        (**self).validate()
    }
}

impl Valid for Vec<PathResCoeff> {
    fn valid() -> Self {
        let offset_end = uc::M * 10000.0;
        let coeff_max = uc::M * 50.0;
        vec![
            PathResCoeff {
                res_coeff: coeff_max / offset_end,
                ..PathResCoeff::valid()
            },
            PathResCoeff {
                offset: offset_end,
                res_net: coeff_max,
                ..PathResCoeff::valid()
            },
        ]
    }
}

impl ObjState for [PathResCoeff] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }

    fn validate(&self) -> ValidationResults {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Path res coeff");
        if self.len() < 2 {
            errors.push(anyhow!("There must be at least two path res coeffs!"));
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
        if !self
            .windows(2)
            .all(|w| w[0].res_coeff == (w[1].res_net - w[0].res_net) / (w[1].offset - w[0].offset))
        {
            errors.push(anyhow!(
                "Res coeff must equal change in res net over change in offset!"
            ));
        }
        si_chk_num_eqz(
            &mut errors,
            &self.last().unwrap().res_coeff,
            "Last res coeff",
        );
        errors.make_err()
    }
}
