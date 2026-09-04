use super::environment::TemperatureTrace;
use super::resistance::kind as res_kind;
use super::resistance::method as res_method;
#[cfg(feature = "pyo3")]
use super::TrainResWrapper;

use super::{
    friction_brakes::*, rail_vehicle::RailVehicle, train_imports::*, InitTrainState,
    SetSpeedTrainSim, SetSpeedTrainSimBuilder, SpeedLimitTrainSim, SpeedLimitTrainSimBuilder,
    SpeedTrace, TimedLinkPath, TrainState,
};
use crate::track::link::link_idx::LinkPath;
use crate::track::link::network::Network;
use crate::track::LocationMap;

#[allow(unused_imports)]
use rayon::prelude::*;

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

    #[pyo3(name = "walk_timed_paths")]
    #[pyo3(signature = (network, timed_paths, parallelize=None))]
    /// Walk every contained sim over its corresponding timed path, optionally in
    /// parallel (rayon). `parallelize` defaults to True.
    fn walk_timed_paths_py(
        &mut self,
        network: Network,
        timed_paths: Vec<TimedLinkPath>,
        parallelize: Option<bool>,
    ) -> anyhow::Result<()> {
        self.walk_timed_paths(&network, &timed_paths, parallelize.unwrap_or(true))
    }

    #[pyo3(name = "tolist")]
    /// Return the contained (walked) sims as a Python list of clones.
    fn tolist_py(&self) -> Vec<SpeedLimitTrainSim> {
        self.0.clone()
    }
}

impl SpeedLimitTrainSimVec {
    pub fn new(value: Vec<SpeedLimitTrainSim>) -> Self {
        Self(value)
    }

    /// Walk each contained sim over its corresponding timed path. When
    /// `parallelize`, sims are walked concurrently with rayon; each walk is pure
    /// Rust, so the worker threads need no GIL.
    pub fn walk_timed_paths(
        &mut self,
        network: &Network,
        timed_paths: &[TimedLinkPath],
        parallelize: bool,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.0.len() == timed_paths.len(),
            "walk_timed_paths: {} sims but {} timed paths",
            self.0.len(),
            timed_paths.len()
        );
        if parallelize {
            self.0
                .par_iter_mut()
                .zip(timed_paths.par_iter())
                .enumerate()
                .try_for_each(|(i, (sim, tp))| {
                    sim.walk_timed_path(network, tp)
                        .map_err(|err| err.context(format!("sim idx:{}", i)))
                })
        } else {
            self.0
                .iter_mut()
                .zip(timed_paths.iter())
                .enumerate()
                .try_for_each(|(i, (sim, tp))| {
                    sim.walk_timed_path(network, tp)
                        .map_err(|err| err.context(format!("sim idx:{}", i)))
                })
        }
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
