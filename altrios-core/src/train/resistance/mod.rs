pub mod kind;
pub mod method;

use crate::imports::*;
use crate::track::LinkPoint;
use crate::track::PathTpc;
use crate::train::TrainState;

pub trait ResMethod {
    fn update_res(
        &mut self,
        state: &mut TrainState,
        path_tpc: &PathTpc,
        dir: &Dir,
    ) -> anyhow::Result<()>;
    fn fix_cache(&mut self, link_point_del: &LinkPoint);
}

#[serde_api]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Wrapper for `TrainRes` to enable exposing contents of enum variants in python
pub struct TrainResWrapper(pub TrainRes);

#[pyo3_api]
impl TrainResWrapper {
    #[getter("point")]
    fn get_point_py(&self) -> Option<method::Point> {
        self.get_point()
    }

    #[getter("strap")]
    fn get_strap_py(&self) -> Option<method::Strap> {
        self.get_strap()
    }
}

impl Init for TrainResWrapper {}
impl SerdeAPI for TrainResWrapper {}

#[cfg(feature = "pyo3")]
impl TrainResWrapper {
    fn get_point(&self) -> Option<method::Point> {
        match &self.0 {
            TrainRes::Point(p) => Some(p.clone()),
            _ => None,
        }
    }

    fn get_strap(&self) -> Option<method::Strap> {
        match &self.0 {
            TrainRes::Strap(s) => Some(s.clone()),
            _ => None,
        }
    }
}

/// Train resistance calculator that calculates resistive powers due to rolling, curvature, flange,
/// grade, and bearing resistances.
// TODO: May also include inertial -- figure this out and modify doc string above
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrainRes {
    Point(method::Point),
    Strap(method::Strap),
}

impl Init for TrainRes {}
impl SerdeAPI for TrainRes {}

impl ResMethod for TrainRes {
    fn update_res(
        &mut self,
        state: &mut TrainState,
        path_tpc: &PathTpc,
        dir: &Dir,
    ) -> anyhow::Result<()> {
        match self {
            TrainRes::Point(p) => p.update_res(state, path_tpc, dir),
            TrainRes::Strap(s) => s.update_res(state, path_tpc, dir),
        }
    }
    fn fix_cache(&mut self, link_point_del: &LinkPoint) {
        match self {
            TrainRes::Point(p) => p.fix_cache(link_point_del),
            TrainRes::Strap(s) => s.fix_cache(link_point_del),
        }
    }
}

impl Default for TrainRes {
    fn default() -> Self {
        Self::Strap(method::Strap::default())
    }
}

impl Valid for TrainRes {
    fn valid() -> Self {
        Self::Strap(method::Strap::valid())
    }
}

// #[cfg(test)]
// mod test_train_res {
//     use super::*;

//     #[test]
//     fn check_output() {
//         let file = File::create("train_res_test.yaml").unwrap();
//         serde_yaml::to_writer(file, &TrainRes::valid()).unwrap();
//     }
// }
