use super::{friction_brakes::FricBrake, train_imports::*};

#[serde_api]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct BrakingPoint {
    pub offset: si::Length,
    pub speed_limit: si::Velocity,
    pub speed_target: si::Velocity,
}

#[pyo3_api]
impl BrakingPoint {}

impl Init for BrakingPoint {}
impl SerdeAPI for BrakingPoint {}

impl ObjState for BrakingPoint {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        si_chk_num_fin(&mut errors, &self.speed_limit, "Speed limit");
        si_chk_num_fin(&mut errors, &self.speed_target, "Speed target");
        errors.make_err()
    }
}

#[serde_api]
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct BrakingPoints {
    points: Vec<BrakingPoint>,
    /// index within [Self::points]
    idx_curr: usize,
    /// tolerance above speed limit for speed limit violations -- i.e. if the
    /// speed limit is 30 mph and `speed_limit_tol` is 10 mph (the default),
    /// then the allowed speed limit when checking is 40 mph
    #[serde(default = "def_speed_lim_tol")]
    speed_limit_tol: si::Velocity,
}

fn def_speed_lim_tol() -> si::Velocity {
    10.0 * uc::MPH
}

impl Init for BrakingPoints {}
impl SerdeAPI for BrakingPoints {}

#[pyo3_api]
impl BrakingPoints {}

impl BrakingPoints {
    /// Arguments:
    /// - offset: location along the current TPC path since train started moving
    /// - speed: current train speed
    /// - adj_ramp_up_time: corrected ramp up time to account for approximately
    ///   linear brake build up
    pub fn calc_speeds(
        &mut self,
        offset: si::Length,
        speed: si::Velocity,
        adj_ramp_up_time: si::Time,
    ) -> anyhow::Result<(si::Velocity, si::Velocity)> {
        if self.points.first().unwrap().offset <= offset {
            self.idx_curr = 0;
        } else {
            while self.points[self.idx_curr - 1].offset <= offset {
                self.idx_curr -= 1;
            }
        }
        ensure!(
            speed <= self.points[self.idx_curr].speed_limit + self.speed_limit_tol,
            "Speed limit violated! idx_curr={:?}, offset={:?}, speed={speed:?}, speed_limit={:?}, speed_target={:?}",
            self.idx_curr,
            self.points[self.idx_curr].offset,
            self.points[self.idx_curr].speed_limit,
            self.points[self.idx_curr].speed_target
        );

        // need to make a way for this to never decrease until a stop happens or maybe never at all
        // need to maybe save `offset_far`
        let offset_far = offset + speed * adj_ramp_up_time;
        let mut speed_target = self.points[self.idx_curr].speed_target;
        let mut idx = self.idx_curr;
        while idx >= 1 && self.points[idx - 1].offset <= offset_far {
            speed_target = speed_target.min(self.points[idx - 1].speed_target);
            idx -= 1;
        }

        Ok((self.points[self.idx_curr].speed_limit, speed_target))
    }

    /// Any time [PathTpc] is updated, everything is recalculated
    pub fn recalc(
        &mut self,
        train_state: &TrainState,
        fric_brake: &FricBrake,
        train_res: &TrainRes,
        path_tpc: &PathTpc,
    ) -> anyhow::Result<()> {
        self.points.clear();

        self.points.push(BrakingPoint {
            offset: path_tpc.offset_end(),

            ..Default::default()
        });

        let mut train_state = train_state.clone();

        let mut train_res = train_res.clone();

        // `update_unchecked` is needed here because `solve_required_pwr` also calls this

        train_state
            .offset
            .update_unchecked(path_tpc.offset_end(), || format_dbg!())?;

        train_state
            .speed
            .update_unchecked(si::Velocity::ZERO, || format_dbg!())?;

        train_res.update_res(&mut train_state, path_tpc, &Dir::Unk)?;

        let speed_points = path_tpc.speed_points();

        let mut idx = path_tpc.speed_points().len();

        // Iterate backwards through all the speed points

        while 0 < idx {
            idx -= 1;

            if speed_points[idx].speed_limit.abs() > self.points.last().unwrap().speed_limit {
                // Iterate until breaking through the speed limit curve
                loop {
                    let bp_curr = *self.points.last().unwrap();
                    // Update speed limit
                    while bp_curr.offset <= speed_points[idx].offset {
                        idx -= 1;
                    }
                    let speed_limit = speed_points[idx].speed_limit.abs();
                    train_state
                        .offset
                        .update_unchecked(bp_curr.offset, || format_dbg!())?;
                    train_state
                        .speed
                        .update_unchecked(bp_curr.speed_limit, || format_dbg!())?;
                    train_res.update_res(&mut train_state, path_tpc, &Dir::Bwd)?;
                    ensure!(
                        fric_brake.force_max + train_state.res_net()? > si::Force::ZERO,
                        format!(
                            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                            format_dbg!(
                                fric_brake.force_max + train_state.res_net()? > si::Force::ZERO
                            ),
                            format_dbg!(fric_brake.force_max),
                            format_dbg!(train_state.res_net()?),
                            format_dbg!(train_state.res_grade),
                            format_dbg!(train_state.grade_front),
                            format_dbg!(train_state.grade_back),
                            format_dbg!(train_state.elev_front),
                            format_dbg!(train_state.elev_back),
                            format_dbg!(train_state.offset),
                            format_dbg!(train_state.offset_back),
                            format_dbg!(train_state.speed),
                            format_dbg!(train_state.speed_limit),
                            format_dbg!(train_state.speed_target),
                            format_dbg!(train_state.time),
                            format_dbg!(train_state.dt),
                            format_dbg!(train_state.i),
                            format_dbg!(train_state.total_dist),
                            format_dbg!(train_state.link_idx_front),
                            format_dbg!(train_state.offset_in_link)
                        )
                    );
                    let vel_change = *train_state.dt.get_fresh(|| format_dbg!())?
                        * (fric_brake.force_max + train_state.res_net()?)
                        / train_state.mass_compound().with_context(|| format_dbg!())?;
                    // exit after adding a couple of points if the next braking curve point will exceed the speed limit
                    if speed_limit < bp_curr.speed_limit + vel_change {
                        self.points.push(BrakingPoint {
                            offset: bp_curr.offset
                                - *train_state.dt.get_fresh(|| format_dbg!())? * speed_limit,
                            speed_limit,
                            speed_target: bp_curr.speed_target,
                        });
                        if bp_curr.speed_limit == speed_points[idx].speed_limit.abs() {
                            break;
                        }
                    } else {
                        // Add normal point to braking curve
                        self.points.push(BrakingPoint {
                            offset: bp_curr.offset
                                - *train_state.dt.get_fresh(|| format_dbg!())?
                                    * (bp_curr.speed_limit + 0.5 * vel_change),
                            speed_limit: bp_curr.speed_limit + vel_change,
                            speed_target: bp_curr.speed_target,
                        });
                    }
                    // Exit if the braking point passed the beginning of the path
                    if self.points.last().unwrap().offset < path_tpc.offset_begin() {
                        break;
                    }
                }
            }

            self.points.push(BrakingPoint {
                offset: speed_points[idx].offset,
                speed_limit: speed_points[idx].speed_limit.abs(),
                speed_target: speed_points[idx].speed_limit.abs(),
            });
        }

        self.idx_curr = self.points.len() - 1;

        Ok(())
    }
}
