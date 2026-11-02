use crate::imports::*;
use crate::track::PathTpc;

#[serde_api]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, HistoryVec)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// For `SetSpeedTrainSim`, it is typically best to use the default for this.
pub struct InitTrainState {
    pub time: TrackedState<si::Time>,
    pub offset: TrackedState<si::Length>,
    pub speed: TrackedState<si::Velocity>,
}

#[pyo3_api]
impl InitTrainState {
    #[new]
    #[pyo3(signature = (
        time_seconds=None,
        offset_meters=None,
        speed_meters_per_second=None,
    ))]
    fn __new__(
        time_seconds: Option<f64>,
        offset_meters: Option<f64>,
        speed_meters_per_second: Option<f64>,
    ) -> Self {
        Self::new(
            time_seconds.map(|x| x * uc::S),
            offset_meters.map(|x| x * uc::M),
            speed_meters_per_second.map(|x| x * uc::MPS),
        )
    }
}

impl Init for InitTrainState {}
impl SerdeAPI for InitTrainState {}

impl Default for InitTrainState {
    fn default() -> Self {
        Self {
            time: TrackedState::new(si::Time::ZERO),
            offset: TrackedState::new(f64::NAN * uc::M),
            speed: TrackedState::new(si::Velocity::ZERO),
        }
    }
}

impl InitTrainState {
    pub fn new(
        time: Option<si::Time>,
        offset: Option<si::Length>,
        speed: Option<si::Velocity>,
    ) -> Self {
        let base = InitTrainState::default();
        Self {
            time: TrackedState::new(
                time.unwrap_or(*base.time.get_fresh(|| format_dbg!()).unwrap()),
            ),
            offset: TrackedState::new(
                offset.unwrap_or(*base.offset.get_fresh(|| format_dbg!()).unwrap()),
            ),
            speed: TrackedState::new(
                speed.unwrap_or(*base.speed.get_fresh(|| format_dbg!()).unwrap()),
            ),
        }
    }
}

#[serde_api]
#[derive(
    Debug, Clone, Serialize, Deserialize, HistoryVec, PartialEq, StateMethods, SetCumulative,
)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct TrainState {
    /// time since user-defined datum
    pub time: TrackedState<si::Time>,
    /// index for time steps
    pub i: TrackedState<usize>,
    /// Linear-along-track, directional distance of front of train from original
    /// starting position of back of train.
    ///
    /// If this is provided in [InitTrainState::new], it gets set as the train length or the value,
    /// whichever is larger, and if it is not provided, then it defaults to the train length.
    pub offset: TrackedState<si::Length>,
    /// Linear-along-track, directional distance of back of train from original
    /// starting position of back of train.
    pub offset_back: TrackedState<si::Length>,
    /// Linear-along-track, cumulative, absolute distance from initial starting position.
    pub total_dist: TrackedState<si::Length>,
    /// Current link containing head end (i.e. pulling locomotives) of train
    pub link_idx_front: TrackedState<u32>,
    /// Current link containing tail/back end of train
    pub link_idx_back: TrackedState<u32>,
    /// Offset from start of current link
    pub offset_in_link: TrackedState<si::Length>,
    /// Achieved speed based on consist capabilities and train resistance
    pub speed: TrackedState<si::Velocity>,
    /// Speed limit
    pub speed_limit: TrackedState<si::Velocity>,
    /// Speed target from meet-pass planner
    pub speed_target: TrackedState<si::Velocity>,
    /// Time step size
    pub dt: TrackedState<si::Time>,
    /// Train length
    pub length: TrackedState<si::Length>,
    /// Static mass of train, including freight
    pub mass_static: TrackedState<si::Mass>,
    /// Effective additional mass of train due to rotational inertia
    pub mass_rot: TrackedState<si::Mass>,
    /// Mass of freight being hauled by the train (not including railcar empty weight)
    pub mass_freight: TrackedState<si::Mass>,
    /// Static weight of train
    pub weight_static: TrackedState<si::Force>,
    /// Rolling resistance force
    pub res_rolling: TrackedState<si::Force>,
    /// Bearing resistance force
    pub res_bearing: TrackedState<si::Force>,
    /// Davis B term resistance force
    pub res_davis_b: TrackedState<si::Force>,
    /// Aerodynamic resistance force
    pub res_aero: TrackedState<si::Force>,
    /// Grade resistance force
    pub res_grade: TrackedState<si::Force>,
    /// Curvature resistance force
    pub res_curve: TrackedState<si::Force>,

    /// Grade at front of train
    pub grade_front: TrackedState<si::Ratio>,
    /// Grade at back of train of train if strap method is used
    pub grade_back: TrackedState<si::Ratio>,
    /// Elevation at front of train
    pub elev_front: TrackedState<si::Length>,
    /// Elevation at back of train
    pub elev_back: TrackedState<si::Length>,

    /// Power to overcome train resistance forces
    pub pwr_res: TrackedState<si::Power>,
    /// Power to overcome inertial forces
    pub pwr_accel: TrackedState<si::Power>,
    /// Total tractive power exerted by locomotive consist
    pub pwr_whl_out: TrackedState<si::Power>,
    /// Integral of [Self::pwr_whl_out]
    pub energy_whl_out: TrackedState<si::Energy>,
    /// Energy out during positive or zero traction
    pub energy_whl_out_pos: TrackedState<si::Energy>,
    /// Energy out during negative traction (positive value means negative traction)
    pub energy_whl_out_neg: TrackedState<si::Energy>,
}

impl Init for TrainState {}
impl SerdeAPI for TrainState {}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            time: Default::default(),
            i: Default::default(),
            offset: Default::default(),
            offset_back: Default::default(),
            total_dist: Default::default(),
            link_idx_front: Default::default(),
            link_idx_back: Default::default(),
            offset_in_link: Default::default(),
            speed: Default::default(),
            speed_limit: Default::default(),
            dt: TrackedState::new(uc::S),
            length: Default::default(),
            mass_static: Default::default(),
            mass_rot: Default::default(),
            mass_freight: Default::default(),
            elev_front: Default::default(),
            elev_back: Default::default(),
            energy_whl_out: Default::default(),
            grade_front: Default::default(),
            grade_back: Default::default(),
            speed_target: Default::default(),
            weight_static: Default::default(),
            res_rolling: Default::default(),
            res_bearing: Default::default(),
            res_davis_b: Default::default(),
            res_aero: Default::default(),
            res_grade: Default::default(),
            res_curve: Default::default(),
            pwr_res: Default::default(),
            pwr_accel: Default::default(),
            pwr_whl_out: Default::default(),
            energy_whl_out_pos: Default::default(),
            energy_whl_out_neg: Default::default(),
        }
    }
}

impl Mass for TrainState {
    /// Static mass of train, not including effective rotational mass
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.derived_mass()
    }

    fn set_mass(
        &mut self,
        _new_mass: Option<si::Mass>,
        _side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        bail!("`set_mass` is not enabled for `TrainState`")
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        // NOTE: if we ever dynamically change mass, this needs attention!
        Ok(Some(*self.mass_static.get_unchecked(|| format_dbg!())?))
    }

    fn expunge_mass_fields(&mut self) {}
}

impl TrainState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        length: si::Length,
        mass_static: si::Mass,
        mass_rot: si::Mass,
        mass_freight: si::Mass,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        let init_train_state = init_train_state.unwrap_or_default();
        let offset = init_train_state
            .offset
            .get_fresh(|| format_dbg!())
            .unwrap()
            .max(length);
        Self {
            time: init_train_state.time,
            i: Default::default(),
            offset: TrackedState::new(offset),
            offset_back: TrackedState::new(offset - length),
            total_dist: TrackedState::new(si::Length::ZERO),
            speed: init_train_state.speed.clone(),
            // this needs to be set to something greater than or equal to actual speed and will be
            // updated after the first time step anyway
            speed_limit: init_train_state.speed,
            length: TrackedState::new(length),
            mass_static: TrackedState::new(mass_static),
            mass_rot: TrackedState::new(mass_rot),
            mass_freight: TrackedState::new(mass_freight),
            ..Self::default()
        }
    }

    pub fn res_net(&self) -> anyhow::Result<si::Force> {
        Ok(*self.res_rolling.get_fresh(|| format_dbg!())?
            + *self.res_bearing.get_fresh(|| format_dbg!())?
            + *self.res_davis_b.get_fresh(|| format_dbg!())?
            + *self.res_aero.get_fresh(|| format_dbg!())?
            + *self.res_grade.get_fresh(|| format_dbg!())?
            + *self.res_curve.get_fresh(|| format_dbg!())?)
    }

    /// All base, freight, and rotational mass
    pub fn mass_compound(&self) -> anyhow::Result<si::Mass> {
        Ok(self
            .mass()
            .with_context(|| format_dbg!())? // extract result
            .with_context(|| format!("{}\nExpected `Some`", format_dbg!()))? // extract option
            + *self.mass_rot.get_unchecked(|| format_dbg!())?)
    }
}

impl Valid for TrainState {
    fn valid() -> Self {
        Self {
            length: TrackedState::new(2000.0 * uc::M),
            offset: TrackedState::new(2000.0 * uc::M),
            offset_back: TrackedState::new(si::Length::ZERO),
            mass_static: TrackedState::new(6000.0 * uc::TON),
            mass_rot: TrackedState::new(200.0 * uc::TON),

            dt: TrackedState::new(uc::S),
            ..Self::default()
        }
    }
}

// TODO: Add new values!
impl ObjState for TrainState {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if let Err(err) = self.mass_static.get_fresh(|| format_dbg!()) {
            errors.push(err);
            return errors.make_err();
        }
        if let Err(err) = self.length.get_fresh(|| format_dbg!()) {
            errors.push(err);
            return errors.make_err();
        }
        si_chk_num_gtz_fin(
            &mut errors,
            self.mass_static.get_fresh(|| format_dbg!()).unwrap(),
            "Mass static",
        );
        si_chk_num_gtz_fin(
            &mut errors,
            self.length.get_fresh(|| format_dbg!()).unwrap(),
            "Length",
        );
        // si_chk_num_gtz_fin(&mut errors, &self.res_bearing, "Resistance bearing");
        // si_chk_num_fin(&mut errors, &self.res_davis_b, "Resistance Davis B");
        // si_chk_num_gtz_fin(&mut errors, &self.cd_area, "cd area");
        errors.make_err()
    }
}

/// Sets `link_idx_front` and `offset_in_link` based on `state` and `path_tpc`
///
/// Assumes that `offset` in `link_points()` is monotically increasing, which may not always be true.
pub fn set_link_and_offset(state: &mut TrainState, path_tpc: &PathTpc) -> anyhow::Result<()> {
    // index of current link within `path_tpc`
    // if the link_point.offset is greater than the train `state` offset, then
    // the train is in the previous link
    let offset = *state.offset.get_stale(|| format_dbg!())?;
    let idx_curr_link = path_tpc
        .link_points()
        .iter()
        .position(|&lp| lp.offset > offset)
        // if None, assume that it's the last element
        .unwrap_or_else(|| path_tpc.link_points().len())
        - 1;
    let link_point = path_tpc
        .link_points()
        .get(idx_curr_link)
        .with_context(|| format_dbg!())?;
    state
        .link_idx_front
        .update(link_point.link_idx.idx() as u32, || format_dbg!())?;
    state.offset_in_link.update(
        *state.offset.get_stale(|| format_dbg!())? - link_point.offset,
        || format_dbg!(),
    )?;

    // link index of back of train at current time step
    let offset_back = *state.offset_back.get_fresh(|| format_dbg!())?;
    let idx_back_link = path_tpc
        .link_points()
        .iter()
        .position(|&lp| lp.offset > offset_back)
        // if None, assume that it's the last element
        .unwrap_or_else(|| path_tpc.link_points().len())
        - 1;
    state.link_idx_back.update(
        path_tpc
            .link_points()
            .get(idx_back_link)
            .with_context(|| format_dbg!())?
            .link_idx
            .idx() as u32,
        || format_dbg!(),
    )?;

    Ok(())
}
