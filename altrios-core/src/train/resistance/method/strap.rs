use super::super::kind::*;
use super::super::ResMethod;
use crate::imports::*;
use crate::track::{LinkPoint, PathResCoeff, PathTpc};
use crate::train::TrainState;

#[serde_api]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Strap {
    bearing: bearing::Basic,
    rolling: rolling::Basic,
    davis_b: davis_b::Basic,
    aerodynamic: aerodynamic::Basic,
    grade: path_res::Strap,
    curve: path_res::Strap,
}

#[pyo3_api]
impl Strap {}

impl Init for Strap {}
impl SerdeAPI for Strap {}

impl Strap {
    pub fn new(
        bearing: bearing::Basic,
        rolling: rolling::Basic,
        davis_b: davis_b::Basic,
        aerodynamic: aerodynamic::Basic,
        grade: path_res::Strap,
        curve: path_res::Strap,
    ) -> Self {
        Self {
            bearing,
            rolling,
            davis_b,
            aerodynamic,
            grade,
            curve,
        }
    }
}
impl ResMethod for Strap {
    fn update_res(
        &mut self,
        state: &mut TrainState,
        path_tpc: &PathTpc,
        dir: &Dir,
    ) -> anyhow::Result<()> {
        // TODO: figure out why these need `update_unchecked` and try to remove that need
        state.offset_back.update_unchecked(
            *state.offset.get_unchecked(|| format_dbg!())?
                - *state.length.get_unchecked(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        state.weight_static.update_unchecked(
            state
                .mass()
                .with_context(|| format_dbg!())? // extract result
                .with_context(|| "{}\nExpected `Some`.")? // extract option
                * uc::ACC_GRAV,
            || format_dbg!(),
        )?;
        state
            .res_bearing
            .update_unchecked(self.bearing.calc_res(), || format_dbg!())?;
        state
            .res_rolling
            .update_unchecked(self.rolling.calc_res(state)?, || format_dbg!())?;
        state
            .res_davis_b
            .update_unchecked(self.davis_b.calc_res(state)?, || format_dbg!())?;
        state
            .res_aero
            .update_unchecked(self.aerodynamic.calc_res(state)?, || format_dbg!())?;
        state.res_grade.update_unchecked(
            self.grade.calc_res(path_tpc.grades(), state, dir)?,
            || format_dbg!(),
        )?;
        state.res_curve.update_unchecked(
            self.curve.calc_res(path_tpc.curves(), state, dir)?,
            || format_dbg!(),
        )?;
        state.grade_front.update_unchecked(
            self.grade.res_coeff_front(path_tpc.grades()),
            || format_dbg!(),
        )?;
        state.grade_back.update_unchecked(
            self.grade.res_coeff_back(path_tpc.grades()),
            || format_dbg!(),
        )?;
        state.elev_front.update_unchecked(
            self.grade.res_net_front(path_tpc.grades(), state)?,
            || format_dbg!(),
        )?;
        state.elev_back.update_unchecked(
            self.grade.res_net_back(path_tpc.grades(), state)?,
            || format_dbg!(),
        )?;
        Ok(())
    }

    fn fix_cache(&mut self, link_point_del: &LinkPoint) {
        self.grade.fix_cache(link_point_del.grade_count);
        self.curve.fix_cache(link_point_del.curve_count);
    }
}
impl Valid for Strap {
    fn valid() -> Self {
        Self {
            bearing: bearing::Basic::new(40.0 * 100.0 * uc::LBF),
            rolling: rolling::Basic::new(1.5 * uc::LB / uc::TON),
            davis_b: davis_b::Basic::new(0.03 / uc::MPH * uc::LB / uc::TON),
            aerodynamic: aerodynamic::Basic::new(
                5.0 * 100.0 * uc::FT2 / 10000.0 / 1.225 * uc::MPS / uc::MPH * uc::MPS / uc::MPH
                    * uc::LBF
                    / uc::N
                    * 100.0,
            ),
            grade: path_res::Strap::new(&Vec::<PathResCoeff>::valid(), &TrainState::valid())
                .unwrap(),
            curve: path_res::Strap::new(&Vec::<PathResCoeff>::valid(), &TrainState::valid())
                .unwrap(),
        }
    }
}
