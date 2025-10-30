use super::environment::TemperatureTrace;
use super::resistance::kind as res_kind;
use super::resistance::method as res_method;
use super::speed_limit_train_sim::TimedLinkPath;
#[cfg(feature = "pyo3")]
use super::TrainResWrapper;
use crate::consist::locomotive::locomotive_model::PowertrainType;

use super::{
    friction_brakes::*, rail_vehicle::RailVehicle, train_imports::*, InitTrainState,
    SetSpeedTrainSim, SetSpeedTrainSimBuilder, SpeedLimitTrainSim, SpeedLimitTrainSimBuilder,
    SpeedTrace, TrainState,
};
use crate::track::link::link_idx::LinkPath;
use crate::track::link::network::Network;
use crate::track::LocationMap;

use polars::prelude::*;
use polars_lazy::dsl::max_horizontal;
#[allow(unused_imports)]
use polars_lazy::prelude::*;
use pyo3_polars::PyDataFrame;

#[serde_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// User-defined train configuration used to generate
/// [crate::prelude::TrainParams]. Any optional fields will be populated later
/// in [TrainSimBuilder::make_train_sim_parts]
pub struct TrainConfig {
    /// Types of rail vehicle composing the train
    pub rail_vehicles: Vec<RailVehicle>,
    /// Number of railcars by type on the train
    pub n_cars_by_type: HashMap<String, u32>,
    /// Train type matching one of the PTC types
    pub train_type: TrainType,
    /// Train length that overrides the railcar specific value, if provided
    pub train_length: Option<si::Length>,
    /// Total train mass that overrides the railcar specific values, if provided
    pub train_mass: Option<si::Mass>,

    #[serde(default)]
    /// Optional vector of drag areas (i.e. drag coeff. times frontal area)
    /// for each car.  If provided, the total drag area (drag coefficient
    /// times frontal area) calculated from this vector is the sum of these
    /// coefficients. Otherwise, each rail car's drag contribution based on its
    /// drag coefficient and frontal area will be summed across the train.
    pub cd_area_vec: Option<Vec<si::Area>>,
}

#[pyo3_api]
impl TrainConfig {
    #[new]
    #[pyo3(signature = (
        rail_vehicles,
        n_cars_by_type,
        train_type=None,
        train_length_meters=None,
        train_mass_kilograms=None,
        cd_area_vec=None,
    ))]
    fn __new__(
        rail_vehicles: Vec<RailVehicle>,
        n_cars_by_type: HashMap<String, u32>,
        train_type: Option<TrainType>,
        train_length_meters: Option<f64>,
        train_mass_kilograms: Option<f64>,
        cd_area_vec: Option<Vec<f64>>,
    ) -> anyhow::Result<Self> {
        Self::new(
            rail_vehicles,
            n_cars_by_type,
            train_type.unwrap_or_default(),
            train_length_meters.map(|v| v * uc::M),
            train_mass_kilograms.map(|v| v * uc::KG),
            cd_area_vec.map(|dcv| dcv.iter().map(|dc| *dc * uc::M2).collect()),
        )
    }

    #[pyo3(name = "make_train_params")]
    /// - `rail_vehicles` - list of `RailVehicle` objects with 1 element for each _type_ of rail vehicle
    fn make_train_params_py(&self) -> anyhow::Result<TrainParams> {
        self.make_train_params()
    }

    #[getter]
    fn get_train_length_meters(&self) -> Option<f64> {
        self.train_length.map(|l| l.get::<si::meter>())
    }

    #[getter]
    fn get_train_mass_kilograms(&self) -> Option<f64> {
        self.train_mass.map(|l| l.get::<si::kilogram>())
    }

    #[getter]
    fn get_cd_area_vec_meters_squared(&self) -> Option<Vec<f64>> {
        self.cd_area_vec.as_ref().map(|dcv| {
            dcv.iter()
                .cloned()
                .map(|x| x.get::<si::square_meter>())
                .collect()
        })
    }

    fn set_cd_area_vec(&mut self, new_val: Vec<f64>) -> anyhow::Result<()> {
        self.cd_area_vec = Some(new_val.iter().map(|x| *x * uc::M2).collect());
        Ok(())
    }
}

impl Init for TrainConfig {
    fn init(&mut self) -> Result<(), Error> {
        if let Some(dcv) = &self.cd_area_vec {
            // TODO: account for locomotive drag here, too
            if dcv.len() as u32 != self.cars_total() {
                return Err(Error::InitError(
                    "`cd_area_vec` len and `cars_total()` do not match".into(),
                ));
            }
        };
        Ok(())
    }
}
impl SerdeAPI for TrainConfig {}

impl TrainConfig {
    pub fn new(
        rail_vehicles: Vec<RailVehicle>,
        n_cars_by_type: HashMap<String, u32>,
        train_type: TrainType,
        train_length: Option<si::Length>,
        train_mass: Option<si::Mass>,
        cd_area_vec: Option<Vec<si::Area>>,
    ) -> anyhow::Result<Self> {
        let mut train_config = Self {
            rail_vehicles,
            n_cars_by_type,
            train_type,
            train_length,
            train_mass,
            cd_area_vec,
        };
        train_config.init()?;
        Ok(train_config)
    }

    pub fn cars_total(&self) -> u32 {
        self.n_cars_by_type.values().fold(0, |acc, n| *n + acc)
    }

    /// # Arguments
    /// - `rail_vehicles` - slice of `RailVehicle` objects with 1 element for each _type_ of rail vehicle
    /// # Important
    /// This method assumes that any calling method has already checked that
    /// all the `car_type` fields in `rail_vehicles` have matching keys in
    /// `self.n_cars_by_type`.
    pub fn make_train_params(&self) -> anyhow::Result<TrainParams> {
        // total towed mass of rail vehicles
        let towed_mass_static = self.train_mass.unwrap_or({
            self.rail_vehicles.iter().try_fold(
                0. * uc::KG,
                |acc, rv| -> anyhow::Result<si::Mass> {
                    Ok(acc
                        + rv.mass()
                            .with_context(|| format_dbg!())?
                            .with_context(|| "`make_train_params` failed")?
                            * *self
                                .n_cars_by_type
                                .get(&rv.car_type)
                                .with_context(|| format_dbg!())?
                                as f64
                            * uc::R)
                },
            )?
        });

        let length: si::Length = match self.train_length {
            Some(tl) => tl,
            None => self
                .rail_vehicles
                .iter()
                .fold(0. * uc::M, |acc, rv| -> si::Length {
                    acc + rv.length * *self.n_cars_by_type.get(&rv.car_type).unwrap() as f64
                }),
        };

        let train_params = TrainParams {
            length,
            speed_max: self.rail_vehicles.iter().fold(
                f64::INFINITY * uc::MPS,
                |acc, rv| -> si::Velocity {
                    if *self.n_cars_by_type.get(&rv.car_type).unwrap() > 0 {
                        acc.min(rv.speed_max)
                    } else {
                        acc
                    }
                },
            ),
            towed_mass_static,
            mass_per_brake: (towed_mass_static + {
                let mass_rot = self
                    .rail_vehicles
                    .iter()
                    .fold(0. * uc::KG, |acc, rv| -> si::Mass {
                        acc + rv.mass_rot_per_axle
                            * *self.n_cars_by_type.get(&rv.car_type).unwrap() as f64
                            * rv.axle_count as f64
                    });
                mass_rot
            }) / self.rail_vehicles.iter().fold(0, |acc, rv| -> u32 {
                acc + rv.brake_count as u32 * *self.n_cars_by_type.get(&rv.car_type).unwrap()
            }) as f64,
            axle_count: self.rail_vehicles.iter().fold(0, |acc, rv| -> u32 {
                acc + rv.axle_count as u32 * *self.n_cars_by_type.get(&rv.car_type).unwrap()
            }),
            train_type: self.train_type,
            // TODO: change it so that curve coefficient is specified at the train level, and replace `unwrap` function calls
            // with proper result handling, and relpace `first().unwrap()` with real code.
            curve_coeff_0: self.rail_vehicles.first().unwrap().curve_coeff_0,
            curve_coeff_1: self.rail_vehicles.first().unwrap().curve_coeff_1,
            curve_coeff_2: self.rail_vehicles.first().unwrap().curve_coeff_2,
        };
        Ok(train_params)
    }
}

impl Valid for TrainConfig {
    fn valid() -> Self {
        Self {
            rail_vehicles: vec![RailVehicle::default()],
            n_cars_by_type: HashMap::from([("Bulk".into(), 100_u32)]),
            train_type: TrainType::Freight,
            train_length: None,
            train_mass: None,
            cd_area_vec: None,
        }
    }
}

#[serde_api]
#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct TrainSimBuilder {
    /// Unique, user-defined identifier for the train
    pub train_id: String,
    pub train_config: TrainConfig,
    pub loco_con: Consist,
    /// Origin_ID from train planner to map to track network locations.  Only needed if
    /// [Self::make_speed_limit_train_sim] will be called.
    pub origin_id: Option<String>,
    /// Destination_ID from train planner to map to track network locations.  Only needed if
    /// [Self::make_speed_limit_train_sim] will be called.
    pub destination_id: Option<String>,

    init_train_state: Option<InitTrainState>,
}

#[pyo3_api]
impl TrainSimBuilder {
    #[new]
    #[pyo3(signature = (
        train_id,
        train_config,
        loco_con,
        origin_id=None,
        destination_id=None,
        init_train_state=None,
    ))]
    fn __new__(
        train_id: String,
        train_config: TrainConfig,
        loco_con: Consist,
        origin_id: Option<String>,
        destination_id: Option<String>,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self::new(
            train_id,
            train_config,
            loco_con,
            origin_id,
            destination_id,
            init_train_state,
        )
    }

    #[pyo3(
        name = "make_set_speed_train_sim",
        signature = (
            network,
            link_path,
            speed_trace,
            save_interval=None,
            temp_trace=None,
        )
    )]
    fn make_set_speed_train_sim_py(
        &self,
        network: &Bound<PyAny>,
        link_path: &Bound<PyAny>,
        speed_trace: SpeedTrace,
        save_interval: Option<usize>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<SetSpeedTrainSim> {
        let network = match network.extract::<Network>() {
            Ok(n) => n,
            Err(_) => {
                let n = network
                    .extract::<Vec<Link>>()
                    .map_err(|_| anyhow!("{}", format_dbg!()))?;
                Network(Default::default(), n)
            }
        };

        let link_path = match link_path.extract::<LinkPath>() {
            Ok(lp) => lp,
            Err(_) => {
                let lp = link_path
                    .extract::<Vec<LinkIdx>>()
                    .map_err(|_| anyhow!("{}", format_dbg!()))?;
                LinkPath(lp)
            }
        };

        self.make_set_speed_train_sim(network, link_path, speed_trace, save_interval, temp_trace)
    }

    #[pyo3(
        name = "make_set_speed_train_sim_and_parts",
        signature = (
            network,
            link_path,
            speed_trace,
            save_interval=None,
            temp_trace=None,
        )
    )]
    fn make_set_speed_train_sim_and_parts_py(
        &self,
        network: &Bound<PyAny>,
        link_path: &Bound<PyAny>,
        speed_trace: SpeedTrace,
        save_interval: Option<usize>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<(
        SetSpeedTrainSim,
        TrainParams,
        PathTpc,
        TrainResWrapper,
        FricBrake,
    )> {
        let network = match network.extract::<Network>() {
            Ok(n) => n,
            Err(_) => {
                let n = network
                    .extract::<Vec<Link>>()
                    .map_err(|_| anyhow!("{}", format_dbg!()))?;
                Network(Default::default(), n)
            }
        };

        let link_path = match link_path.extract::<LinkPath>() {
            Ok(lp) => lp,
            Err(_) => {
                let lp = link_path
                    .extract::<Vec<LinkIdx>>()
                    .map_err(|_| anyhow!("{}", format_dbg!()))?;
                LinkPath(lp)
            }
        };

        let (train_sim, train_params, path_tpc, tr, fb) = self
            .make_set_speed_train_sim_and_parts(
                network,
                link_path,
                speed_trace,
                save_interval,
                temp_trace,
            )
            .with_context(|| format_dbg!())?;

        let trw = TrainResWrapper(tr);
        Ok((train_sim, train_params, path_tpc, trw, fb))
    }

    #[pyo3(
        name = "make_speed_limit_train_sim",
        signature = (
            location_map,
            save_interval=None,
            simulation_days=None,
            scenario_year=None,
            temp_trace=None,
        )
    )]
    fn make_speed_limit_train_sim_py(
        &self,
        location_map: LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<SpeedLimitTrainSim> {
        self.make_speed_limit_train_sim(
            &location_map,
            save_interval,
            simulation_days,
            scenario_year,
            temp_trace,
        )
    }

    #[pyo3(
        name = "make_speed_limit_train_sim_and_parts",
        signature = (
            location_map,
            save_interval=None,
            simulation_days=None,
            scenario_year=None,
            temp_trace=None,
        )
    )]
    fn make_speed_limit_train_sim_and_parts_py(
        &self,
        location_map: LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<(SpeedLimitTrainSim, PathTpc, TrainResWrapper, FricBrake)> {
        let (ts, path_tpc, tr, fb) = self.make_speed_limit_train_sim_and_parts(
            &location_map,
            save_interval,
            simulation_days,
            scenario_year,
            temp_trace,
        )?;

        let trw = TrainResWrapper(tr);
        Ok((ts, path_tpc, trw, fb))
    }
}

impl Init for TrainSimBuilder {}
impl SerdeAPI for TrainSimBuilder {}

impl TrainSimBuilder {
    pub fn new(
        train_id: String,
        train_config: TrainConfig,
        loco_con: Consist,
        origin_id: Option<String>,
        destination_id: Option<String>,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self {
            train_id,
            train_config,
            loco_con,
            origin_id,
            destination_id,
            init_train_state,
        }
    }

    fn make_train_sim_parts(
        &self,
        save_interval: Option<usize>,
    ) -> anyhow::Result<(TrainParams, TrainState, PathTpc, TrainRes, FricBrake)> {
        let rvs = &self.train_config.rail_vehicles;
        // check that `self.train_config.n_cars_by_type` has keys matching `rail_vehicles`
        self.check_rv_keys()?;
        let train_params = self
            .train_config
            .make_train_params()
            .with_context(|| format_dbg!())?;

        let length = train_params.length;
        // total train weight including locomotives, baseline rail vehicle masses, and freight mass
        let train_mass_static = train_params.towed_mass_static
            + self
                .loco_con
                .mass()
                .with_context(|| format_dbg!())?
                .unwrap_or_else(|| 0. * uc::KG);

        let mass_rot = rvs.iter().fold(0. * uc::KG, |acc, rv| -> si::Mass {
            acc + rv.mass_rot_per_axle
                * *self.train_config.n_cars_by_type.get(&rv.car_type).unwrap() as f64
                * rv.axle_count as f64
        });
        let mass_freight = rvs.iter().fold(0. * uc::KG, |acc, rv| -> si::Mass {
            acc + rv.mass_freight
                * *self.train_config.n_cars_by_type.get(&rv.car_type).unwrap() as f64
        });
        let max_fric_braking = uc::ACC_GRAV
            * train_params.towed_mass_static
            * rvs.iter().fold(0. * uc::R, |acc, rv| -> si::Ratio {
                acc + rv.braking_ratio
                    * *self.train_config.n_cars_by_type.get(&rv.car_type).unwrap() as f64
            })
            / self.train_config.cars_total() as f64;

        let state = TrainState::new(
            length,
            train_mass_static,
            mass_rot,
            mass_freight,
            self.init_train_state.clone(),
        );

        let path_tpc = PathTpc::new(train_params);

        let train_res = {
            let res_bearing = res_kind::bearing::Basic::new(rvs.iter().fold(
                0. * uc::N,
                |acc, rv| -> si::Force {
                    acc + rv.bearing_res_per_axle
                        * rv.axle_count as f64
                        * *self.train_config.n_cars_by_type.get(&rv.car_type).unwrap() as f64
                },
            ));
            // Sum of mass-averaged rolling resistances across all railcars
            let res_rolling = res_kind::rolling::Basic::new(rvs.iter().try_fold(
                0.0 * uc::R,
                |acc, rv| -> anyhow::Result<si::Ratio> {
                    Ok(acc
                        + rv.rolling_ratio
                            * rv.mass()
                                .with_context(|| format_dbg!())?
                                .with_context(|| format!("{}\nExpected `Some`", format_dbg!()))?
                            / train_params.towed_mass_static // does not include locomotive consist mass -- TODO: fix this, carefully                            
                            * *self.train_config.n_cars_by_type.get(&rv.car_type).unwrap() as f64
                            * uc::R)
                },
            )?);
            let davis_b = res_kind::davis_b::Basic::new(rvs.iter().try_fold(
                0.0 * uc::S / uc::M,
                |acc, rv| -> anyhow::Result<si::InverseVelocity> {
                    Ok(acc
                        + rv.davis_b
                            * rv.mass()
                                .with_context(|| format_dbg!())?
                                .with_context(|| format!("{}\nExpected `Some`", format_dbg!()))?
                            / train_params.towed_mass_static // does not include locomotive consist mass -- TODO: fix this, carefully
                            * *self.train_config.n_cars_by_type.get(&rv.car_type).unwrap() as f64
                            * uc::R)
                },
            )?);
            let res_aero =
                res_kind::aerodynamic::Basic::new(match &self.train_config.cd_area_vec {
                    Some(dave) => dave.iter().fold(0. * uc::M2, |acc, dc| *dc + acc),
                    None => rvs.iter().fold(0.0 * uc::M2, |acc, rv| -> si::Area {
                        acc + rv.cd_area
                            * *self.train_config.n_cars_by_type.get(&rv.car_type).unwrap() as f64
                    }),
                });
            let res_grade = res_kind::path_res::Strap::new(path_tpc.grades(), &state)?;
            let res_curve = res_kind::path_res::Strap::new(path_tpc.curves(), &state)?;
            TrainRes::Strap(res_method::Strap::new(
                res_bearing,
                res_rolling,
                davis_b,
                res_aero,
                res_grade,
                res_curve,
            ))
        };

        let fric_brake = FricBrake::new(max_fric_braking, None, None, None, save_interval);

        Ok((train_params, state, path_tpc, train_res, fric_brake))
    }

    fn check_rv_keys(&self) -> anyhow::Result<()> {
        let rv_car_type_set = HashSet::<String>::from_iter(
            self.train_config
                .rail_vehicles
                .iter()
                .map(|rv| rv.car_type.clone()),
        );
        let n_cars_type_set =
            HashSet::<String>::from_iter(self.train_config.n_cars_by_type.keys().cloned());
        let extra_keys_in_rv = rv_car_type_set
            .difference(&n_cars_type_set)
            .collect::<Vec<&String>>();
        let extra_keys_in_n_cars = n_cars_type_set
            .difference(&rv_car_type_set)
            .collect::<Vec<&String>>();
        if !extra_keys_in_rv.is_empty() {
            bail!(
                "Extra values in `car_type` for `rail_vehicles` that are not in `n_cars_by_type`: {:?}",
                extra_keys_in_rv
            );
        }
        if !extra_keys_in_n_cars.is_empty() {
            bail!(
                "Extra values in `n_cars_by_type` that are not in `car_type` for `rail_vehicles`: {:?}",
                extra_keys_in_n_cars
            );
        }
        Ok(())
    }

    pub fn make_set_speed_train_sim<Q: AsRef<[Link]>, R: AsRef<[LinkIdx]>>(
        &self,
        network: Q,
        link_path: R,
        speed_trace: SpeedTrace,
        save_interval: Option<usize>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<SetSpeedTrainSim> {
        ensure!(
            self.origin_id.is_none() & self.destination_id.is_none(),
            "{}\n`origin_id` and `destination_id` must both be `None` when calling `make_set_speed_train_sim`.",
            format_dbg!()
        );

        let (_, state, mut path_tpc, train_res, _fric_brake) = self
            .make_train_sim_parts(save_interval)
            .with_context(|| format_dbg!())?;

        path_tpc.extend(network, link_path)?;
        Ok(SetSpeedTrainSimBuilder {
            loco_con: self.loco_con.clone(),
            n_cars_by_type: self.train_config.n_cars_by_type.clone(),
            state,
            speed_trace,
            train_res,
            path_tpc,
            save_interval,
            temp_trace,
        }
        .into())
    }

    pub fn make_set_speed_train_sim_and_parts<Q: AsRef<[Link]>, R: AsRef<[LinkIdx]>>(
        &self,
        network: Q,
        link_path: R,
        speed_trace: SpeedTrace,
        save_interval: Option<usize>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<(SetSpeedTrainSim, TrainParams, PathTpc, TrainRes, FricBrake)> {
        ensure!(
            self.origin_id.is_none() & self.destination_id.is_none(),
            "{}\n`origin_id` and `destination_id` must both be `None` when calling `make_set_speed_train_sim`.",
            format_dbg!()
        );

        let (train_params, state, mut path_tpc, train_res, fric_brake) = self
            .make_train_sim_parts(save_interval)
            .with_context(|| format_dbg!())?;

        path_tpc.extend(network, link_path)?;
        Ok((
            SetSpeedTrainSimBuilder {
                loco_con: self.loco_con.clone(),
                n_cars_by_type: self.train_config.n_cars_by_type.clone(),
                state,
                speed_trace,
                train_res: train_res.clone(),
                path_tpc: path_tpc.clone(),
                save_interval,
                temp_trace,
            }
            .into(),
            train_params,
            path_tpc,
            train_res,
            fric_brake,
        ))
    }

    pub fn make_speed_limit_train_sim(
        &self,
        location_map: &LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<SpeedLimitTrainSim> {
        let (_, state, path_tpc, train_res, fric_brake) = self
            .make_train_sim_parts(save_interval)
            .with_context(|| format_dbg!())?;

        ensure!(
            self.origin_id.is_some() & self.destination_id.is_some(),
            "{}\nBoth `origin_id` and `destination_id` must be provided when initializing{} ",
            format_dbg!(),
            "`TrainSimBuilder` for `make_speed_limit_train_sim` to work."
        );

        Ok(SpeedLimitTrainSimBuilder {
            train_id: self.train_id.clone(),
            // `self.origin_id` verified to be `Some` earlier
            origs: location_map
                .get(self.origin_id.as_ref().unwrap())
                .with_context(|| {
                    anyhow!(format!(
                        "{}\n`origin_id`: \"{}\" not found in `location_map` keys: {:?}",
                        format_dbg!(),
                        self.origin_id.as_ref().unwrap(),
                        location_map.keys(),
                    ))
                })?
                .to_vec(),
            // `self.destination_id` verified to be `Some` earlier
            dests: location_map
                .get(self.destination_id.as_ref().unwrap())
                .with_context(|| {
                    anyhow!(format!(
                        "{}\n`destination_id`: \"{}\" not found in `location_map` keys: {:?}",
                        format_dbg!(),
                        self.destination_id.as_ref().unwrap(),
                        location_map.keys(),
                    ))
                })?
                .to_vec(),
            loco_con: self.loco_con.clone(),
            n_cars_by_type: self.train_config.n_cars_by_type.clone(),
            state,
            train_res,
            path_tpc,
            fric_brake,
            save_interval,
            simulation_days,
            scenario_year,
            temp_trace,
        }
        .into())
    }

    pub fn make_speed_limit_train_sim_and_parts(
        &self,
        location_map: &LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
        temp_trace: Option<TemperatureTrace>,
    ) -> anyhow::Result<(SpeedLimitTrainSim, PathTpc, TrainRes, FricBrake)> {
        let (_, state, path_tpc, train_res, fric_brake) = self
            .make_train_sim_parts(save_interval)
            .with_context(|| format_dbg!())?;

        ensure!(
            self.origin_id.is_some() & self.destination_id.is_some(),
            "{}\nBoth `origin_id` and `destination_id` must be provided when initializing{} ",
            format_dbg!(),
            "`TrainSimBuilder` for `make_speed_limit_train_sim` to work."
        );

        let ts = SpeedLimitTrainSimBuilder {
            train_id: self.train_id.clone(),
            // `self.origin_id` verified to be `Some` earlier
            origs: location_map
                .get(self.origin_id.as_ref().unwrap())
                .with_context(|| {
                    anyhow!(format!(
                        "{}\n`origin_id`: \"{}\" not found in `location_map` keys: {:?}",
                        format_dbg!(),
                        self.origin_id.as_ref().unwrap(),
                        location_map.keys(),
                    ))
                })?
                .to_vec(),
            // `self.destination_id` verified to be `Some` earlier
            dests: location_map
                .get(self.destination_id.as_ref().unwrap())
                .with_context(|| {
                    anyhow!(format!(
                        "{}\n`destination_id`: \"{}\" not found in `location_map` keys: {:?}",
                        format_dbg!(),
                        self.destination_id.as_ref().unwrap(),
                        location_map.keys(),
                    ))
                })?
                .to_vec(),
            loco_con: self.loco_con.clone(),
            n_cars_by_type: self.train_config.n_cars_by_type.clone(),
            state,
            train_res: train_res.clone(),
            path_tpc: path_tpc.clone(),
            fric_brake: fric_brake.clone(),
            save_interval,
            simulation_days,
            scenario_year,
            temp_trace,
        };
        Ok((ts.into(), path_tpc, train_res, fric_brake))
    }
}

/// Converts either `Column::Series` or `Column::Scalar` to `Series`
fn to_series(col: Column) -> anyhow::Result<Series> {
    match col.clone() {
        Column::Series(s) => Ok(s.take()),
        Column::Scalar(s) => Ok(s.to_series()),
        Column::Partitioned(_) => bail!("{}\nPartitioned column!", format_dbg!()),
    }
}

#[allow(unused_variables)]
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn run_speed_limit_train_sims(
    mut speed_limit_train_sims: SpeedLimitTrainSimVec,
    network: &Bound<PyAny>,
    train_consist_plan_py: PyDataFrame,
    loco_pool_py: PyDataFrame,
    refuel_facilities_py: PyDataFrame,
    timed_paths: Vec<TimedLinkPath>,
) -> anyhow::Result<(SpeedLimitTrainSimVec, PyDataFrame)> {
    let network = match network.extract::<Network>() {
        Ok(n) => n,
        Err(_) => {
            let n = network
                .extract::<Vec<Link>>()
                .map_err(|_| anyhow!("{}", format_dbg!()))?;
            Network(Default::default(), n)
        }
    };

    let train_consist_plan: DataFrame = train_consist_plan_py.clone().into();
    let mut loco_pool: DataFrame = loco_pool_py.clone().into();
    let refuel_facilities: DataFrame = refuel_facilities_py.clone().into();

    loco_pool = loco_pool
        .lazy()
        .with_columns(vec![
            lit(f64::ZERO).alias("Trip_Energy_J").to_owned(),
            lit(f64::INFINITY).alias("Ready_Time_Min").to_owned(),
            lit(f64::INFINITY).alias("Ready_Time_Est").to_owned(),
            lit("Ready").alias("Status").to_owned(),
            col("SOC_Max_J").alias("SOC_J").to_owned(),
        ])
        .collect()
        .with_context(|| format_dbg!())?;

    let mut arrival_times = train_consist_plan
        .clone()
        .lazy()
        .select(vec![
            col("Arrival_Time_Actual_Hr"),
            col("Locomotive_ID"),
            col("Destination_ID"),
            col("TrainSimVec_Index"),
        ])
        .sort_by_exprs(
            vec![col("Arrival_Time_Actual_Hr"), col("Locomotive_ID")],
            SortMultipleOptions::default(),
        )
        .collect()
        .with_context(|| format_dbg!())?;

    let departure_times = train_consist_plan
        .clone()
        .lazy()
        .select(vec![col("Departure_Time_Actual_Hr"), col("Locomotive_ID")])
        .sort_by_exprs(
            vec![col("Locomotive_ID"), col("Departure_Time_Actual_Hr")],
            SortMultipleOptions::default(),
            // vec![false, false],
            // false,
            // false,
        )
        .collect()
        .with_context(|| format_dbg!())?;

    let mut refuel_sessions = DataFrame::default();

    let active_loco_statuses =
        Series::from_iter(vec!["Refueling".to_string(), "Dispatched".to_string()]);
    let mut current_time: f64 = arrival_times
        .column("Arrival_Time_Actual_Hr")
        .with_context(|| format_dbg!())?
        .f64()
        .with_context(|| format_dbg!())?
        .min()
        .unwrap();

    let mut done = false;
    while !done {
        let arrivals_mask = arrival_times
            .column("Arrival_Time_Actual_Hr")?
            .equal(&Column::new(
                "current_time_const".into(),
                vec![current_time; arrival_times.height()],
            ))
            .with_context(|| format_dbg!())?;
        let arrivals = arrival_times
            .clone()
            .filter(&arrivals_mask)
            .with_context(|| format_dbg!())?;
        let arrivals_merged = loco_pool
            .clone()
            .left_join(&arrivals, ["Locomotive_ID"], ["Locomotive_ID"])
            .with_context(|| format_dbg!())?;
        let arrival_locations = arrivals_merged.column("Destination_ID")?;
        if arrivals.height() > 0 {
            let arrival_ids = arrivals
                .column("Locomotive_ID")
                .with_context(|| format_dbg!())?;
            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![
                    when(col("Locomotive_ID").is_in(lit(
                        to_series(arrival_ids.clone()).with_context(|| format_dbg!())?,
                    )))
                    .then(lit("Queued"))
                    .otherwise(col("Status"))
                    .alias("Status"),
                    when(col("Locomotive_ID").is_in(lit(
                        to_series(arrival_ids.clone()).with_context(|| format_dbg!())?,
                    )))
                    .then(lit(current_time))
                    .otherwise(col("Ready_Time_Est"))
                    .alias("Ready_Time_Est"),
                    when(col("Locomotive_ID").is_in(lit(
                        to_series(arrival_ids.clone()).with_context(|| format_dbg!())?,
                    )))
                    .then(lit(arrival_locations
                        .clone()
                        .as_series()
                        .with_context(|| format_dbg!())?
                        .clone()))
                    .otherwise(col("Node"))
                    .alias("Node"),
                ])
                .drop(vec![
                    "Refueler_J_Per_Hr",
                    "Refueler_Efficiency",
                    "Port_Count",
                    "Battery_Headroom_J",
                ])
                .join(
                    refuel_facilities.clone().lazy().select(&[
                        col("Node"),
                        col("Locomotive_Type"),
                        col("Fuel_Type"),
                        col("Refueler_J_Per_Hr"),
                        col("Refueler_Efficiency"),
                        col("Port_Count"),
                        col("Battery_Headroom_J"),
                    ]),
                    [col("Node"), col("Locomotive_Type"), col("Fuel_Type")],
                    [col("Node"), col("Locomotive_Type"), col("Fuel_Type")],
                    JoinArgs::new(JoinType::Left),
                )
                .with_columns(vec![col("Battery_Headroom_J").fill_null(0)])
                .with_columns(vec![max_horizontal([
                    col("SOC_Max_J") - col("Battery_Headroom_J"),
                    col("SOC_Min_J"),
                ])
                .with_context(|| format_dbg!())?
                .alias("SOC_Target_J")])
                .sort(["Locomotive_ID"], SortMultipleOptions::default())
                .collect()
                .with_context(|| format_dbg!())?;

            let indices = arrivals
                .column("TrainSimVec_Index")
                .with_context(|| format_dbg!())?
                .u32()
                .with_context(|| format_dbg!())?
                .unique()
                .with_context(|| format_dbg!())?;
            for index in indices.into_iter() {
                let idx = index.unwrap() as usize;
                let departing_soc_pct_iter = train_consist_plan
                    .clone()
                    .lazy()
                    // retain rows in which "TrainSimVec_Index" equals current `index`
                    .filter(col("TrainSimVec_Index").eq(index.unwrap()))
                    // Select "Locomotive_ID" column
                    .select(vec![col("Locomotive_ID")])
                    // find unique locomotive IDs
                    .unique(None, UniqueKeepStrategy::First)
                    .join(
                        loco_pool.clone().lazy(),
                        [col("Locomotive_ID")],
                        [col("Locomotive_ID")],
                        JoinArgs::new(JoinType::Left),
                    )
                    .sort(["Locomotive_ID"], SortMultipleOptions::default())
                    .with_columns(vec![(col("SOC_J") / col("Capacity_J")).alias("SOC_Pct")])
                    .collect()
                    .with_context(|| format_dbg!())?;

                let departing_soc_pct = to_series(
                    departing_soc_pct_iter
                        .column("SOC_Pct")
                        .with_context(|| format_dbg!())?
                        .clone(),
                )
                .with_context(|| format_dbg!())?;

                let departing_soc_pct_vec: Vec<f64> = departing_soc_pct
                    .f64()
                    .with_context(|| format_dbg!())?
                    .into_no_null_iter()
                    .collect();
                let sim = &mut speed_limit_train_sims.0[idx];
                sim.loco_con
                    .loco_vec
                    .iter_mut()
                    .zip(departing_soc_pct_vec)
                    .try_for_each(|(loco, soc)| {
                        if let Some(res) = &mut loco.reversible_energy_storage_mut() {
                            res.state
                                .soc
                                .update_unchecked(soc * uc::R, || format_dbg!())
                        } else {
                            Ok(())
                        }
                    })?;
                let _ = sim
                    .walk_timed_path(&network, &timed_paths[idx])
                    .map_err(|err| err.context(format!("train sim idx: {}", idx)));

                let mut new_soc_vec: Vec<f64> = vec![];
                sim.loco_con
                    .loco_vec
                    .iter()
                    .try_for_each(|loco| -> anyhow::Result<()> {
                        match loco.loco_type {
                            PowertrainType::BatteryElectricLoco(_) => {
                                new_soc_vec.push(
                                    (*loco
                                        .reversible_energy_storage()
                                        .unwrap()
                                        .state
                                        .soc
                                        .get_fresh(|| format_dbg!())?
                                        * loco
                                            .reversible_energy_storage()
                                            .unwrap()
                                            .energy_capacity)
                                        .get::<si::joule>(),
                                );
                            }
                            _ => new_soc_vec.push(f64::ZERO),
                        }
                        Ok(())
                    })
                    .with_context(|| format_dbg!())?;
                let mut new_energy_j_vec: Vec<f64> = vec![];
                sim.loco_con
                    .loco_vec
                    .iter()
                    .try_for_each(|loco| -> anyhow::Result<()> {
                        new_energy_j_vec.push(match loco.loco_type {
                            PowertrainType::BatteryElectricLoco(_) => loco
                                .reversible_energy_storage()
                                .unwrap()
                                .state
                                .energy_out_chemical
                                .get_fresh(|| format_dbg!())?
                                .get::<si::joule>(),
                            _ => f64::ZERO,
                        });
                        Ok(())
                    })
                    .with_context(|| format_dbg!())?;
                let mut all_current_socs: Vec<f64> = loco_pool
                    .column("SOC_J")
                    .with_context(|| format_dbg!())?
                    .f64()
                    .with_context(|| format_dbg!())?
                    .into_no_null_iter()
                    .collect();
                let mut all_energy_j: Vec<f64> = (loco_pool
                    .column("SOC_J")
                    .with_context(|| format_dbg!())?
                    .f64()?
                    * 0.0)
                    .into_no_null_iter()
                    .collect();
                let idx_mask = arrival_times
                    .column("TrainSimVec_Index")
                    .with_context(|| format_dbg!())?
                    .equal(&Column::new(
                        "idx_const".into(),
                        vec![idx as u32; arrival_times.height()],
                    ))
                    .with_context(|| format_dbg!())?;
                let arrival_locos = arrival_times
                    .filter(&idx_mask)
                    .with_context(|| format_dbg!())?;
                let arrival_loco_ids = arrival_locos
                    .column("Locomotive_ID")
                    .with_context(|| format_dbg!())?
                    .u32()
                    .with_context(|| format_dbg!())?;
                let arrival_loco_mask: ChunkedArray<BooleanType> = is_in(
                    loco_pool
                        .column("Locomotive_ID")
                        .with_context(|| format_dbg!())?
                        .as_series()
                        .with_context(|| format_dbg!())?,
                    &Series::from(arrival_loco_ids.clone()),
                )
                .with_context(|| format_dbg!())?;

                // Get the indices of true values in the boolean ChunkedArray
                let arrival_loco_indices: Vec<usize> = arrival_loco_mask
                    .into_iter()
                    .enumerate()
                    .filter(|(_, val)| val.unwrap_or_default())
                    .map(|(i, _)| i)
                    .collect();

                // TODO: rewrite this a little so it doesn't depend on the previous sort
                for (index, value) in arrival_loco_indices.iter().zip(new_soc_vec) {
                    all_current_socs[*index] = value;
                }
                for (index, value) in arrival_loco_indices.iter().zip(new_energy_j_vec) {
                    all_energy_j[*index] = value;
                }
                loco_pool = loco_pool
                    .lazy()
                    .with_columns(vec![
                        when(lit(arrival_loco_mask.clone().into_series()))
                            .then(lit(Series::new("SOC_J".into(), all_current_socs)))
                            .otherwise(col("SOC_J"))
                            .alias("SOC_J"),
                        when(lit(arrival_loco_mask.into_series()))
                            .then(lit(Series::new("Trip_Energy_J".into(), all_energy_j)))
                            .otherwise(col("Trip_Energy_J"))
                            .alias("Trip_Energy_J"),
                    ])
                    .collect()
                    .with_context(|| format_dbg!())?;
            }
            loco_pool = loco_pool
                .lazy()
                .sort(["Ready_Time_Est"], SortMultipleOptions::default())
                .collect()
                .with_context(|| format_dbg!())?;
        }

        let refueling_mask = (loco_pool)
            .column("Status")
            .with_context(|| format_dbg!())?
            .equal(&Column::new(
                "refueling_const".into(),
                vec!["Refueling"; loco_pool.height()],
            ))
            .with_context(|| format_dbg!())?;
        let refueling_finished_mask = refueling_mask.clone()
            & (loco_pool)
                .column("Ready_Time_Est")
                .with_context(|| format_dbg!())?
                .equal(&Column::new(
                    "current_time_const".into(),
                    vec![current_time; refueling_mask.len()],
                ))
                .with_context(|| format_dbg!())?;
        let refueling_finished = loco_pool
            .clone()
            .filter(&refueling_finished_mask)
            .with_context(|| format_dbg!())?;
        if refueling_finished_mask.sum().unwrap_or_default() > 0 {
            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![when(lit(refueling_finished_mask.into_series()))
                    .then(lit("Ready"))
                    .otherwise(col("Status"))
                    .alias("Status")])
                .collect()
                .with_context(|| format_dbg!())?;
        }

        if (arrivals.height() > 0) || (refueling_finished.height() > 0) {
            // update queue
            let place_in_queue_iter = loco_pool
                .clone()
                .lazy()
                .select(&[((col("Status").eq(lit("Refueling")).sum().over([
                    "Node",
                    "Locomotive_Type",
                    "Fuel_Type",
                ])) + (col("Status").eq(lit("Queued")).over([
                    "Node",
                    "Locomotive_Type",
                    "Fuel_Type",
                ])))
                .alias("place_in_queue")])
                .collect()?;
            let place_in_queue = place_in_queue_iter
                .column("place_in_queue")?
                .as_series()
                .with_context(|| format_dbg!())?;
            let future_times_mask = departure_times
                .column("Departure_Time_Actual_Hr")?
                .f64()?
                .gt(current_time);

            let next_departure_time = departure_times
                .clone()
                .lazy()
                .filter(col("Departure_Time_Actual_Hr").gt(lit(current_time)))
                .group_by(["Locomotive_ID"])
                .agg([col("Departure_Time_Actual_Hr").min()])
                .collect()
                .with_context(|| format_dbg!())?;

            let departures_merged = loco_pool.clone().left_join(
                &next_departure_time,
                ["Locomotive_ID"],
                ["Locomotive_ID"],
            )?;
            let departure_times = departures_merged
                .column("Departure_Time_Actual_Hr")?
                .f64()?;

            let target_j = loco_pool
                .clone()
                .lazy()
                .select(&[(col("SOC_Max_J") - col("Battery_Headroom_J")).alias("Target_J")])
                .collect()?
                .column("Target_J")?
                .clone();
            let target_j_f64 = target_j.f64()?;
            let current_j = loco_pool.column("SOC_J")?.f64()?;

            let soc_target: Vec<f64> = target_j_f64
                .into_iter()
                .zip(current_j.into_iter())
                .map(|(b, v)| b.unwrap_or(f64::ZERO).max(v.unwrap_or(f64::ZERO)))
                .collect::<Vec<_>>();
            let soc_target_series = Series::new("soc_target".into(), soc_target);

            let refuel_end_time_ideal_iter = loco_pool
                .clone()
                .lazy()
                .select(&[(lit(current_time)
                    + (max_horizontal([col("SOC_J"), col("SOC_Target_J")])? - col("SOC_J"))
                        / col("Refueler_J_Per_Hr"))
                .alias("Refuel_End_Time")])
                .collect()?;
            let refuel_end_time_ideal = refuel_end_time_ideal_iter
                .column("Refuel_End_Time")?
                .as_series()
                .with_context(|| format_dbg!())?;

            let refuel_end_time: Vec<f64> = departure_times
                .into_iter()
                .zip(refuel_end_time_ideal.f64()?.into_iter())
                .map(|(b, v)| b.unwrap_or(f64::INFINITY).min(v.unwrap_or(f64::INFINITY)))
                .collect::<Vec<_>>();

            let mut refuel_duration: Vec<f64> = refuel_end_time.clone();
            for element in refuel_duration.iter_mut() {
                *element -= current_time;
            }

            let refuel_duration_series = Series::new("refuel_duration".into(), refuel_duration);
            let refuel_end_series = Series::new("refuel_end_time".into(), refuel_end_time);

            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![
                    lit(place_in_queue.clone()),
                    lit(refuel_duration_series.clone()),
                    lit(refuel_end_series.clone()),
                ])
                .collect()
                .with_context(|| format_dbg!())?;

            // store the filter as an Expr
            let refuel_starting = loco_pool
                .clone()
                .lazy()
                .filter(
                    col("Status")
                        .eq(lit("Queued"))
                        .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                )
                .collect()
                .with_context(|| format_dbg!())?;

            let these_refuel_sessions = refuel_starting
                .clone()
                .lazy()
                .with_columns(vec![
                    (col("Refueler_J_Per_Hr") * col("refuel_duration")
                        / col("Refueler_Efficiency"))
                    .alias("Refuel_Energy_J"),
                    (col("refuel_end_time") - col("refuel_duration")).alias("Refuel_Start_Time_Hr"),
                ])
                .rename(
                    ["refuel_end_time", "refuel_duration"],
                    ["Refuel_End_Time_Hr", "Refuel_Duration_Hr"],
                    true,
                )
                .select(vec![
                    col("Node"),
                    col("Locomotive_Type"),
                    col("Fuel_Type"),
                    col("Locomotive_ID"),
                    col("Refueler_J_Per_Hr"),
                    col("Refueler_Efficiency"),
                    col("Trip_Energy_J"),
                    col("SOC_J"),
                    col("Refuel_Energy_J"),
                    col("Refuel_Duration_Hr"),
                    col("Refuel_Start_Time_Hr"),
                    col("Refuel_End_Time_Hr"),
                ])
                .collect()
                .with_context(|| format_dbg!())?;
            refuel_sessions.vstack_mut(&these_refuel_sessions)?;
            // set finishedCharging times to min(max soc OR departure time)
            loco_pool = loco_pool
                .clone()
                .lazy()
                .with_columns(vec![
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                    )
                    .then(col("SOC_J") + col("refuel_duration") * col("Refueler_J_Per_Hr"))
                    .otherwise(col("SOC_J"))
                    .alias("SOC_J"),
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                    )
                    .then(col("refuel_end_time"))
                    .otherwise(col("Ready_Time_Est"))
                    .alias("Ready_Time_Est"),
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                    )
                    .then(lit("Refueling"))
                    .otherwise(col("Status"))
                    .alias("Status"),
                ])
                .collect()
                .with_context(|| format_dbg!())?;

            loco_pool = loco_pool.drop("place_in_queue")?;
            loco_pool = loco_pool.drop("refuel_duration")?;
            loco_pool = loco_pool.drop("refuel_end_time")?;
        }

        let active_loco_ready_times_iter = loco_pool
            .clone()
            .lazy()
            .filter(col("Status").is_in(lit(active_loco_statuses.clone())))
            .select(vec![col("Ready_Time_Est")])
            .collect()?;
        let active_loco_ready_times = active_loco_ready_times_iter
            .column("Ready_Time_Est")
            .with_context(|| format_dbg!(active_loco_ready_times_iter))?;
        arrival_times = arrival_times
            .lazy()
            .filter(col("Arrival_Time_Actual_Hr").gt(current_time))
            .collect()?;
        let arrival_times_remaining_iter = arrival_times
            .clone()
            .lazy()
            .select(vec![
                col("Arrival_Time_Actual_Hr").alias("Arrival_Time_Actual_Hr")
            ])
            .collect()?;
        let arrival_times_remaining = arrival_times_remaining_iter
            .column("Arrival_Time_Actual_Hr")
            .with_context(|| format_dbg!())?;

        if (arrival_times_remaining.is_empty()) & (active_loco_ready_times.is_empty()) {
            done = true;
        } else {
            let min1 = active_loco_ready_times
                .f64()?
                .min()
                .unwrap_or(f64::INFINITY);
            let min2 = arrival_times_remaining
                .f64()?
                .min()
                .unwrap_or(f64::INFINITY);
            current_time = f64::min(min1, min2);
        }
    }

    Ok((speed_limit_train_sims, PyDataFrame(refuel_sessions)))
}

// This MUST remain a unit struct to trigger correct tolist() behavior
#[serde_api]
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct SpeedLimitTrainSimVec(pub Vec<SpeedLimitTrainSim>);

#[pyo3_api]
impl SpeedLimitTrainSimVec {
    #![allow(non_snake_case)]
    #[pyo3(name = "get_energy_fuel_joules")]
    pub fn get_energy_fuel_py(&self, annualize: bool) -> anyhow::Result<f64> {
        Ok(self.get_energy_fuel(annualize)?.get::<si::joule>())
    }

    #[pyo3(name = "get_net_energy_res_joules")]
    pub fn get_net_energy_res_py(&self, annualize: bool) -> anyhow::Result<f64> {
        Ok(self.get_net_energy_res(annualize)?.get::<si::joule>())
    }

    #[pyo3(name = "get_kilometers")]
    pub fn get_kilometers_py(&self, annualize: bool) -> anyhow::Result<f64> {
        self.get_kilometers(annualize)
    }

    #[pyo3(name = "get_megagram_kilometers")]
    pub fn get_megagram_kilometers_py(&self, annualize: bool) -> anyhow::Result<f64> {
        self.get_megagram_kilometers(annualize)
    }

    #[pyo3(name = "get_car_kilometers")]
    pub fn get_car_kilometers_py(&self, annualize: bool) -> anyhow::Result<f64> {
        self.get_car_kilometers(annualize)
    }

    #[pyo3(name = "get_cars_moved")]
    pub fn get_cars_moved_py(&self, annualize: bool) -> f64 {
        self.get_cars_moved(annualize)
    }

    #[pyo3(name = "get_res_kilometers")]
    pub fn get_res_kilometers_py(&mut self, annualize: bool) -> anyhow::Result<f64> {
        self.get_res_kilometers(annualize)
    }

    #[pyo3(name = "get_non_res_kilometers")]
    pub fn get_non_res_kilometers_py(&mut self, annualize: bool) -> anyhow::Result<f64> {
        self.get_non_res_kilometers(annualize)
    }

    #[pyo3(name = "set_save_interval")]
    #[pyo3(signature = (save_interval=None))]
    pub fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }

    #[new]
    /// Rust-defined `__new__` magic method for Python used exposed via PyO3.
    fn __new__(v: Vec<SpeedLimitTrainSim>) -> Self {
        Self(v)
    }
}

impl SpeedLimitTrainSimVec {
    pub fn new(value: Vec<SpeedLimitTrainSim>) -> Self {
        Self(value)
    }

    pub fn get_energy_fuel(&self, annualize: bool) -> anyhow::Result<si::Energy> {
        self.0.iter().try_fold(si::Energy::ZERO, |acc, sim| {
            Ok(acc + sim.get_energy_fuel(annualize)?)
        })
    }

    pub fn get_net_energy_res(&self, annualize: bool) -> anyhow::Result<si::Energy> {
        self.0.iter().try_fold(si::Energy::ZERO, |acc, sim| {
            Ok(acc + sim.get_net_energy_res(annualize)?)
        })
    }

    pub fn get_kilometers(&self, annualize: bool) -> anyhow::Result<f64> {
        self.0
            .iter()
            .try_fold(0.0, |acc, sim| Ok(acc + sim.get_kilometers(annualize)?))
    }

    pub fn get_megagram_kilometers(&self, annualize: bool) -> anyhow::Result<f64> {
        self.0.iter().try_fold(0.0, |acc, sim| {
            Ok(acc + sim.get_megagram_kilometers(annualize)?)
        })
    }

    pub fn get_car_kilometers(&self, annualize: bool) -> anyhow::Result<f64> {
        self.0
            .iter()
            .try_fold(0.0, |acc, sim| Ok(acc + sim.get_car_kilometers(annualize)?))
    }

    pub fn get_cars_moved(&self, annualize: bool) -> f64 {
        self.0.iter().map(|sim| sim.get_cars_moved(annualize)).sum()
    }

    pub fn get_res_kilometers(&mut self, annualize: bool) -> anyhow::Result<f64> {
        self.0
            .iter_mut()
            .try_fold(0.0, |acc, sim| Ok(acc + sim.get_res_kilometers(annualize)?))
    }

    pub fn get_non_res_kilometers(&mut self, annualize: bool) -> anyhow::Result<f64> {
        self.0.iter_mut().try_fold(0.0, |acc, sim| {
            Ok(acc + sim.get_non_res_kilometers(annualize)?)
        })
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.0
            .iter_mut()
            .for_each(|slts| slts.set_save_interval(save_interval));
    }
}

impl Init for SpeedLimitTrainSimVec {
    fn init(&mut self) -> Result<(), Error> {
        self.0.iter_mut().try_for_each(|ts| ts.init())?;
        Ok(())
    }
}
impl SerdeAPI for SpeedLimitTrainSimVec {}
