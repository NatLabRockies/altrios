use super::super::kind::*;
use super::super::ResMethod;
use super::*;
use crate::imports::*;
use crate::track::{LinkPoint, PathResCoeff};

#[serde_api]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Point {
    bearing: bearing::Basic,
    rolling: rolling::Basic,
    davis_b: davis_b::Basic,
    aerodynamic: aerodynamic::Basic,
    grade: path_res::Point,
    curve: path_res::Point,
}

#[pyo3_api]
impl Point {}

impl Init for Point {}
impl SerdeAPI for Point {}

impl ResMethod for Point {
    fn update_res(
        &mut self,
        state: &mut TrainState,
        path_tpc: &PathTpc,
        dir: &Dir,
    ) -> anyhow::Result<()> {
        state.offset_back.update(
            *state.offset.get_fresh(|| format_dbg!())?
                - *state.length.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        state.weight_static.update(
            state
                .mass()
                .with_context(|| format_dbg!())?
                .with_context(|| "{}\nExpected `Some`.")?
                * uc::ACC_GRAV,
            || format_dbg!(),
        )?;
        state
            .res_bearing
            .update(self.bearing.calc_res(), || format_dbg!())?;
        state
            .res_rolling
            .update(self.rolling.calc_res(state)?, || format_dbg!())?;
        state
            .res_davis_b
            .update(self.davis_b.calc_res(state)?, || format_dbg!())?;
        state
            .res_aero
            .update(self.aerodynamic.calc_res(state)?, || format_dbg!())?;
        state.res_grade.update(
            self.grade.calc_res(path_tpc.grades(), state, dir)?,
            || format_dbg!(),
        )?;
        state.res_curve.update(
            self.curve.calc_res(path_tpc.curves(), state, dir)?,
            || format_dbg!(),
        )?;
        state.grade_front.update(
            self.grade.res_coeff_front(path_tpc.grades()),
            || format_dbg!(),
        )?;
        state.elev_front.update(
            self.grade.res_net_front(path_tpc.grades(), state)?,
            || format_dbg!(),
        )?;

        Ok(())
    }

    fn fix_cache(&mut self, link_point_del: &LinkPoint) {
        self.grade.fix_cache(link_point_del.grade_count);
        self.curve.fix_cache(link_point_del.curve_count);
    }
}

impl Valid for Point {
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
            grade: path_res::Point::new(&Vec::<PathResCoeff>::valid(), &TrainState::valid())
                .unwrap(),
            curve: path_res::Point::new(&Vec::<PathResCoeff>::valid(), &TrainState::valid())
                .unwrap(),
        }
    }
}
