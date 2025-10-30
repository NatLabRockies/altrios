use super::cat_power::*;
use super::elev::*;
use super::heading::*;
use super::link_idx::*;
use super::link_old::LinkOld;
use super::speed::*;
use crate::imports::*;

#[cfg(doc)]
use crate::meet_pass::est_times::*;

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
struct OldSpeedSets(Vec<OldSpeedSet>);

#[serde_api]
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// An arbitrary unit of single track that does not include turnouts
pub struct Link {
    /// Index of current link
    pub idx_curr: LinkIdx,
    /// Index of current link in reverse direction
    pub idx_flip: LinkIdx,
    /// see [EstTime::idx_next]
    pub idx_next: LinkIdx,
    /// see [EstTime::idx_next_alt]  
    /// if it does not exist, it should be `LinkIdx{idx: 0}`
    pub idx_next_alt: LinkIdx,
    /// see [EstTime::idx_prev]
    pub idx_prev: LinkIdx,
    /// see [EstTime::idx_prev_alt]  
    /// if it does not exist, it should be `LinkIdx{idx: 0}`
    pub idx_prev_alt: LinkIdx,
    /// Optional OpenStreetMap ID -- not used in simulation
    pub osm_id: Option<String>,
    /// Total length of [Self]
    #[serde(alias = "length")]
    pub length: si::Length,

    /// Spatial vector of elevation values and corresponding positions along track
    pub elevs: Vec<Elev>,
    #[serde(default)]
    /// Spatial vector of compass heading values and corresponding positions along track
    pub headings: Vec<Heading>,
    /// Map of train types and corresponding speed sets
    #[serde(default)]
    pub speed_sets: HashMap<TrainType, SpeedSet>,
    /// Optional train-type-neutral [SpeedSet].  If provided, overrides [Link::speed_sets].
    pub speed_set: Option<SpeedSet>,
    #[serde(default)]
    /// Spatial vector of catenary power limit values and corresponding positions along track
    pub cat_power_limits: Vec<CatPowerLimit>,

    #[serde(default)]
    /// Prevents provided links from being occupied when the current link has a train on it. An
    /// example would be at a switch, where there would be foul links running from the switch points
    /// to the clearance point. Due to the geometric overlap of the foul links, only one may be
    /// occupied at a given time. For further explanation, see the [graphical
    /// example](https://nrel.github.io/altrios/api-doc/rail-network.html?highlight=network#link-lockout).
    pub link_idxs_lockout: Vec<LinkIdx>,

    #[serde(skip)]
    pub err_tol: Option<NetworkErrTol>,
}

#[pyo3_api]
impl Link {}

impl Link {
    fn is_linked_prev(&self, idx: LinkIdx) -> bool {
        self.idx_curr.is_fake() || self.idx_prev == idx || self.idx_prev_alt == idx
    }
    fn is_linked_next(&self, idx: LinkIdx) -> bool {
        self.idx_curr.is_fake() || self.idx_next == idx || self.idx_next_alt == idx
    }

    /// Sets `self.speed_set` based on `self.speed_sets` value corresponding to `train_type` key
    pub fn set_speed_set_for_train_type(&mut self, train_type: TrainType) -> anyhow::Result<()> {
        self.speed_set = Some(
            self.speed_sets
                .get(&train_type)
                .with_context(|| {
                    anyhow!(
                        "No value found for train_type: {:?} in `speed_sets`.",
                        train_type
                    )
                })?
                .clone(),
        );
        self.speed_sets = HashMap::new();
        Ok(())
    }
}

impl Init for Link {}
impl SerdeAPI for Link {}

impl From<LinkOld> for Link {
    fn from(l: LinkOld) -> Self {
        let mut speed_sets: HashMap<TrainType, SpeedSet> = HashMap::new();
        for oss in l.speed_sets {
            speed_sets.insert(
                oss.train_type,
                SpeedSet {
                    speed_limits: oss.speed_limits,
                    speed_params: oss.speed_params,
                    is_head_end: oss.is_head_end,
                },
            );
        }

        Self {
            elevs: l.elevs,
            headings: l.headings,
            speed_sets,
            speed_set: Default::default(),
            cat_power_limits: l.cat_power_limits,
            length: l.length,
            idx_next: l.idx_next,
            idx_next_alt: l.idx_next_alt,
            idx_prev: l.idx_prev,
            idx_prev_alt: l.idx_prev_alt,
            idx_curr: l.idx_curr,
            idx_flip: l.idx_flip,
            osm_id: l.osm_id,
            link_idxs_lockout: l.link_idxs_lockout,
            err_tol: Some(Default::default()),
        }
    }
}

impl Valid for Link {
    fn valid() -> Self {
        Self {
            elevs: Vec::<Elev>::valid(),
            headings: Vec::<Heading>::valid(),
            speed_sets: HashMap::<TrainType, SpeedSet>::valid(),
            length: uc::M * 10000.0,
            idx_curr: LinkIdx::valid(),
            ..Self::default()
        }
    }
}

impl ObjState for Link {
    fn is_fake(&self) -> bool {
        self.idx_curr.is_fake()
    }

    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.is_fake() {
            validate_field_fake(&mut errors, &self.idx_next, "Link index next");
            validate_field_fake(&mut errors, &self.idx_next_alt, "Link index next alt");
            validate_field_fake(&mut errors, &self.idx_prev, "Link index prev");
            validate_field_fake(&mut errors, &self.idx_prev_alt, "Link index prev alt");
            validate_field_fake(&mut errors, &self.idx_curr, "Link index curr");
            validate_field_fake(&mut errors, &self.idx_flip, "Link index flip");
            si_chk_num_eqz(&mut errors, &self.length, "Link length");
            validate_field_fake(&mut errors, &self.elevs, "Elevations");
            validate_field_fake(&mut errors, &self.headings, "Headings");
            validate_field_fake(&mut errors, &self.speed_sets, "Speed sets");
            validate_field_fake(&mut errors, &self.speed_sets, "Speed sets");
            if let Some(speed_set) = &self.speed_set {
                validate_field_fake(&mut errors, speed_set, "Speed set");
            }
            // validate cat_power_limits
            if !self.cat_power_limits.is_empty() {
                errors.push(anyhow!(
                    "Catenary power limits = {:?} must be empty!",
                    self.cat_power_limits
                ));
            }
        } else {
            // Link is real
            si_chk_num_gtz(&mut errors, &self.length, "Link length");
            validate_field_real(&mut errors, &self.elevs, "Elevations");
            if !self.headings.is_empty() {
                validate_field_real(&mut errors, &self.headings, "Headings");
            }
            if !self.speed_sets.is_empty() {
                validate_field_real(&mut errors, &self.speed_sets, "Speed sets");
                if self.speed_set.is_some() {
                    errors.push(anyhow!(
                        "`speed_sets` is not empty and `speed_set` is `Some(speed_set). {}",
                        "Change one of these."
                    ));
                }
            } else if let Some(speed_set) = &self.speed_set {
                validate_field_real(&mut errors, speed_set, "Speed set");
                if !self.speed_sets.is_empty() {
                    errors.push(anyhow!(
                        "`speed_sets` is not empty and `speed_set` is `Some(speed_set)`. {}",
                        "Change one of these."
                    ));
                }
            } else {
                errors.push(anyhow!(
                    "{}\n`SpeedSets` is empty and `SpeedSet` is `None`. {}",
                    format_dbg!(),
                    "One of these fields must be provided"
                ));
            }
            validate_field_real(&mut errors, &self.cat_power_limits, "Catenary power limits");

            early_err!(errors, "Link");

            if self.idx_flip.is_real() {
                for (var, name) in [
                    (self.idx_curr, "curr"),
                    (self.idx_next, "next"),
                    (self.idx_next_alt, "next alt"),
                    (self.idx_prev, "prev"),
                    (self.idx_prev_alt, "prev alt"),
                ] {
                    if var == self.idx_flip {
                        errors.push(anyhow!(
                            "Link index flip = {:?} and link index {} = {:?} must be different!",
                            self.idx_flip,
                            name,
                            var
                        ));
                    }
                }
            }
            if self.idx_next_alt.is_real() && self.idx_next.is_fake() {
                errors.push(anyhow!(
                    "Link index next = {:?} must be real when link index next alt = {:?} is real!",
                    self.idx_next,
                    self.idx_next_alt
                ));
            }
            if self.idx_prev_alt.is_real() && self.idx_prev.is_fake() {
                errors.push(anyhow!(
                    "Link index prev = {:?} must be real when link index prev alt = {:?} is real!",
                    self.idx_prev,
                    self.idx_prev_alt
                ));
            }

            // verify that first offset is zero
            if self.elevs.first().unwrap().offset != si::Length::ZERO {
                errors.push(anyhow!(
                    "First elevation offset = {:?} is invalid, must equal zero!",
                    self.elevs.first().unwrap().offset
                ));
            }
            // verify that last offset is equal to length
            if self.elevs.last().unwrap().offset != self.length {
                errors.push(anyhow!(
                    "Last elevation offset = {:?} is invalid, must equal length = {:?}!",
                    self.elevs.last().unwrap().offset,
                    self.length
                ));
            }
            if !self.headings.is_empty() {
                // verify that first offset is zero
                if self.headings.first().unwrap().offset != si::Length::ZERO {
                    errors.push(anyhow!(
                        "First heading offset = {:?} is invalid, must equal zero!",
                        self.headings.first().unwrap().offset
                    ));
                }
                // verify that last offset is equal to length
                if self.headings.last().unwrap().offset != self.length {
                    errors.push(anyhow!(
                        "Last heading offset = {:?} is invalid, must equal length = {:?}!",
                        self.headings.last().unwrap().offset,
                        self.length
                    ));
                }
            }
            // if cat power limits are not specified for entire length of link, assume that no cat
            // power is available
            if !self.cat_power_limits.is_empty() {
                if self.cat_power_limits.first().unwrap().offset_start < si::Length::ZERO {
                    errors.push(anyhow!(
                        "First cat power limit offset start = {:?} is invalid, must be greater than or equal to zero!",
                        self.cat_power_limits.first().unwrap().offset_start
                    ));
                }
                if self.cat_power_limits.last().unwrap().offset_end > self.length {
                    errors.push(anyhow!(
                        "Last cat power limit offset end = {:?} is invalid, must be less than or equal to length = {:?}!",
                        self.cat_power_limits.last().unwrap().offset_end,
                        self.length
                    ));
                }
            }

            if let Some(err_tol) = &self.err_tol {
                let grades: Vec<si::Ratio> = self
                    .elevs
                    .windows(2)
                    .map(|w| (w[1].elev - w[0].elev) / (w[1].offset - w[0].offset))
                    .collect();
                if let Some(max_allowed_abs_grade) = err_tol.max_grade {
                    match grades.iter().map(|g| g.abs()).reduce(si::Ratio::max) {
                        Some(max_abs_grade) => {
                            if max_abs_grade > max_allowed_abs_grade {
                                let idx_max_grade = grades
                                    .iter()
                                    .position(|&g| g.abs() == max_abs_grade)
                                    .with_context(|| format_dbg!())
                                    .unwrap(); // pretty sure this unwrap is safe
                                errors.push(anyhow!(
                        "{} -- Max absolute grade ({}%) exceeds max allowed grade ({}%) at offset: {} m",
                        format_dbg!(),
                        max_abs_grade.get::<si::percent>(),
                        max_allowed_abs_grade.get::<si::percent>(),
                        // Add 1 to the index because grades is 1 element longer
                        self.elevs[idx_max_grade+ 1].offset.get::<si::meter>()
                    ));
                            }
                        }
                        None => errors.push(anyhow!(
                            "{} -- Failed to calculate max absolute grade.",
                            format_dbg!()
                        )),
                    };
                }
                if let Some(max_allowed_abs_curv) = err_tol.max_curv {
                    // TODO: make sure that all headings are between 0 and 2 pi
                    let curves: Vec<si::Curvature> = self
                        .headings
                        .windows(2)
                        .map(|w| {
                            // the 3.0 is to make sure heading changes that cross
                            // through zero still result in positive numbers
                            let dh: si::Angle = (w[1].heading - w[0].heading + 3.0 * uc::REV / 2.0)
                                % uc::REV
                                - uc::REV / 2.0;
                            let dx: si::Length = w[1].offset - w[0].offset;
                            (dh / dx).into()
                        })
                        .collect();
                    match curves.iter().map(|y| y.abs()).reduce(si::Curvature::max) {
                        Some(max_abs_curv) => {
                            if max_abs_curv > max_allowed_abs_curv {
                                let idx_max_curv = curves
                                    .iter()
                                    .position(|&c| c.abs() == max_abs_curv)
                                    .with_context(|| format_dbg!())
                                    .unwrap(); // pretty sure this unwrap is safe
                                errors.push(anyhow!(
                    "{} -- Max curvature ({} degrees per 100 feet) exceeds max allowed curvature ({} degrees per 100 feet) at offset: {} m",
                    format_dbg!(),
                    max_abs_curv.get::<si::degree_per_meter>() / 3.28084 * 100.0,
                    max_allowed_abs_curv.get::<si::degree_per_meter>() / 3.28084 * 100.0,
                    // Add 1 to the index because headings is 1 element longer
                    self.headings[idx_max_curv + 1].offset.get::<si::meter>()
                ));
                            }
                        }
                        None => errors.push(anyhow!(
                            "{} -- Failed to calculate max absolute curvature.",
                            format_dbg!()
                        )),
                    };
                }
            }
        }

        errors.make_err()
    }
}

#[serde_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct that contains a `Vec<Link>`, and optional parameters for setting
/// error tolerances in checks performed by [Init::init]
pub struct Network(pub NetworkErrTol, pub Vec<Link>);

#[pyo3_api]
impl Network {
    fn __len__(&self) -> usize {
        self.1.len()
    }

    #[pyo3(name = "set_speed_set_for_train_type")]
    fn set_speed_set_for_train_type_py(&mut self, train_type: TrainType) -> anyhow::Result<()> {
        self.set_speed_set_for_train_type(train_type)
    }
}

#[serde_api]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Network error tolerances used in [Network::init]
pub struct NetworkErrTol {
    /// Maximum absolute grade allowed at any point in the network
    pub max_grade: Option<si::Ratio>,

    /// Maximum absolute curvature allowed anywhere in the network
    pub max_curv: Option<si::Curvature>,

    /// Maximum allowed step change in heading for coincident nodes in adjacent links
    pub max_heading_step: Option<si::Angle>,

    /// Maximum allowed step change in elevation for coincident nodes in
    /// adjacent links, should be very small
    pub max_elev_step: Option<si::Length>,
}

#[pyo3_api]
impl NetworkErrTol {}

impl Init for NetworkErrTol {
    fn init(&mut self) -> Result<(), Error> {
        let def: Self = Default::default();
        self.max_grade = self.max_grade.or(def.max_grade);
        self.max_curv = self.max_curv.or(def.max_curv);
        self.max_heading_step = self.max_heading_step.or(def.max_heading_step);
        self.max_elev_step = self.max_elev_step.or(def.max_elev_step);
        Ok(())
    }
}
impl SerdeAPI for NetworkErrTol {}
impl Default for NetworkErrTol {
    fn default() -> Self {
        Self {
            max_grade: Some(0.06 * uc::R),
            max_curv: Some((15.0 * uc::DEG / (100.0 * uc::FT)).into()),
            max_heading_step: Some(7.0 * uc::DEG),
            max_elev_step: Some(0.0 * uc::M),
        }
    }
}

impl Network {
    /// Sets `self.speed_set` based on `self.speed_sets` value corresponding to `train_type` key for
    /// all links
    pub fn set_speed_set_for_train_type(&mut self, train_type: TrainType) -> anyhow::Result<()> {
        for l in self.1.iter_mut().skip(1) {
            l.set_speed_set_for_train_type(train_type)
                .with_context(|| format!("`idx_curr`: {}", l.idx_curr))?;
        }
        Ok(())
    }
}

impl ObjState for Network {
    fn is_fake(&self) -> bool {
        self.1.is_fake()
    }
    fn validate(&self) -> ValidationResults {
        let err_tol = self.0.clone();
        // NOTE: it may be better to change the trait to `fn validate(&mut self) -> ValidationResults {`
        // to avoid the need for this clone()
        let mut links = self.1.clone();
        // propagate error tolerance to links
        links
            .iter_mut()
            .for_each(|link| link.err_tol = Some(err_tol.clone()));
        links.validate()
    }
}

impl SerdeAPI for Network {
    fn from_file<P: AsRef<Path>>(filepath: P, skip_init: bool) -> Result<Self, Error> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| {
                Error::SerdeError(format!("File extension could not be parsed: {filepath:?}"))
            })?;
        let mut file = File::open(filepath).map_err(|err| {
            Error::SerdeError(format!(
                "{err}\n{}",
                if !filepath.exists() {
                    format!("File not found: {filepath:?}")
                } else {
                    format!("Could not open file: {filepath:?}")
                }
            ))
        })?;
        match Self::from_reader(&mut file, extension, skip_init) {
            Ok(network) => {
                // init already happened in `from_reader`
                Ok(network)
            }
            Err(err) => {
                match err {
                    // if the outer error `err` is a SerdeError, try the old network format
                    Error::SerdeError(_) => {
                        let filepath_canon = filepath.canonicalize().map_err(|err_canon| {
                            Error::SerdeError(format!("{err_canon}\n`canonicalize` failed"))
                        })?;
                        let mut filepath_copy: String = filepath_canon
                            .as_os_str()
                            .to_str()
                            .ok_or_else(|| {
                                Error::SerdeError("Unable to extract `filepath` as str".into())
                            })?
                            .into();
                        filepath_copy = filepath_copy.replace(
                            &format!(".{extension}"),
                            &format!("_updated_format.{extension}"),
                        );
                        match NetworkOld::from_file(filepath, skip_init) {
                            Err(err_old) => match err_old {
                                Error::SerdeError(_) => {
                                    // if the outer error `err_old` is a SerdeError, try the unchecked network format
                                    match NetworkUnchecked::from_file(
                                        filepath, skip_init,
                                    ) {
                                        Ok(network_unchecked) => {
                                            let network: Network = network_unchecked.into();
                                            network.to_file(&filepath_copy).map_err(|tf_err| {
                                                Error::SerdeError(format!(
                                                    "{}\n{tf_err}",
                                                    format_dbg!()
                                                ))
                                            })?;
                                            Err(Error::SerdeError(format!(
                                                    "{}\n    `Network`: {err}\n    `NetworkOld`: {err_old}\nMissing error tolerance.  Wrote copy of network file in new format with error tolerances to {}",
                                                    format_dbg!(),
                                                    filepath_copy
                                                )))
                                        }
                                        Err(err_unchecked) => Err(err_unchecked).map_err(|err_unchecked | {
                                                Error::SerdeError(
                                                    format!("\n    `Network`: {err}\n    `NetworkOld`: {err_old}\n    `NetworkUnchecked`: {err_unchecked}"))
                                        }),
                                    }
                                }.map_err(|err_unchecked | Error::SerdeError(format!("{}\n{err_unchecked}", format_dbg!()))),
                                _ => Err(err_old).map_err(|err_old| {
                                    Error::SerdeError(format!(
                                        "{}\n{err}\n{err_old}",
                                        format_dbg!()
                                    ))
                                }),
                            },
                            Ok(network_old) => {
                                let network: Network = network_old.into();
                                network
                                    .to_file(&filepath_copy)
                                    .map_err(|tf_err| Error::SerdeError(format!("{tf_err}")))?;
                                Err(Error::SerdeError(format!(
                                    "{}\n`Network`: {err}\nDeprecated network format.  Wrote copy of network file in new format to {}",
                                    format_dbg!(),
                                    filepath_copy
                                )))
                            }
                        }
                    }
                    _ => Err(err)
                        .map_err(|err| Error::SerdeError(format!("{}\n{err}", format_dbg!()))),
                }
            }
        }
    }
}
impl Init for Network {
    fn init(&mut self) -> Result<(), Error> {
        // if self.1.is_none() {
        //     return Err(Error::InitError(format!(
        //         "`network.1` (error tolerances must be provided, e.g.\n{})",
        //         NetworkErrTol::default().to_str("json")
        //     )));
        // }
        self.validate()
            .map_err(|err| Error::InitError(format!("\n{}\n{err}", format_dbg!())))
    }
}

impl From<NetworkOld> for Network {
    fn from(old: NetworkOld) -> Self {
        Network(
            Default::default(),
            old.0.iter().map(|l| Link::from(l.clone())).collect(),
        )
    }
}
impl From<NetworkUnchecked> for Network {
    fn from(unchecked: NetworkUnchecked) -> Self {
        Network(Default::default(), unchecked.0)
    }
}

#[serde_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Deprecated Struct that contains a `Vec<Link>` for the purpose of providing
/// `SerdeAPI` for `Vec<Link>` in Python.  This is used solely to enable
/// backwards compatibility.
struct NetworkUnchecked(pub Vec<Link>);

#[pyo3_api]
impl NetworkUnchecked {}

impl Init for NetworkUnchecked {}
impl SerdeAPI for NetworkUnchecked {}

#[serde_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct that contains a `Vec<LinkOld>` for the purpose of providing `SerdeAPI` for `Vec<Link>` in
/// Python
///
/// # Note:
/// This struct will be deprecated and superseded by [Network], which has the
/// option for either a train-type-independent `speed_set` or a train-type-dependent
/// `speed_sets` HashMap
struct NetworkOld(pub Vec<LinkOld>);

#[pyo3_api]
impl NetworkOld {}

impl Init for NetworkOld {}
impl SerdeAPI for NetworkOld {}

impl AsRef<[Link]> for Network {
    fn as_ref(&self) -> &[Link] {
        &self.1
    }
}

impl From<&Vec<Link>> for Network {
    fn from(value: &Vec<Link>) -> Self {
        Self(Default::default(), value.to_vec())
    }
}

impl Valid for Vec<Link> {
    fn valid() -> Self {
        vec![Link::default(), Link::valid()]
    }
}

impl ObjState for [Link] {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.len() < 2 {
            errors.push(anyhow!(
                "There must be at least two links (one physical and one dummy)!"
            ));
            early_err!(errors, "Links");
        }
        validate_slice_fake(&mut errors, &self[..1], "Link");
        validate_slice_real_shift(&mut errors, &self[1..], "Link", 1);
        early_err!(errors, "Links");

        for (idx, link) in self.iter().enumerate().skip(1) {
            // Validate flip and curr
            if link.idx_curr.idx() != idx {
                errors.push(anyhow!(
                    "Link idx {} is not equal to index in vector {}!",
                    link.idx_curr,
                    idx
                ))
            }
            if link.idx_flip == link.idx_curr {
                errors.push(anyhow!(
                    "Normal {} and flipped {} links must be different!",
                    link.idx_curr,
                    link.idx_flip
                ));
            }
            if link.idx_flip.is_real() && self[link.idx_flip.idx()].idx_flip != link.idx_curr {
                errors.push(anyhow!(
                    "Flipped link {} does not properly reference current link {}!",
                    link.idx_flip,
                    link.idx_curr
                ));
            }

            // Validate next
            if link.idx_next.is_real() {
                for (link_next, name) in [
                    (&self[link.idx_next.idx()], "next link"),
                    (&self[link.idx_next_alt.idx()], "next link alt"),
                ] {
                    if !link_next.is_linked_prev(link.idx_curr) {
                        errors.push(anyhow!(
                            "Current link {} with {} {} prev idx {} and prev idx alt {} do not point back!",
                            link.idx_curr,
                            name,
                            link_next.idx_curr,
                            link_next.idx_prev,
                            link_next.idx_prev_alt,
                        ));
                    }
                    if link.idx_next_alt.is_real() && link_next.idx_prev_alt.is_real() {
                        errors.push(anyhow!(
                            "Current link {} and {} {} have coincident switch points!",
                            link.idx_curr,
                            name,
                            link_next.idx_curr,
                        ));
                    }

                    // Verify that coincident link points have matching elevations within a tolerance parameter
                    // since `err_tol` propagates from `Network`, only one needs to be checked
                    // these unwraps should be guaranteed to be checked before this
                    if !link.elevs.is_empty() && !link_next.elevs.is_empty() {
                        let elev_delta = link.elevs.last().unwrap().elev - link_next.elevs[0].elev;
                        if elev_delta.abs()
                            > link_next.err_tol.as_ref().unwrap().max_elev_step.unwrap()
                        {
                            errors.push(anyhow!(
                            "Difference between last elevation in current link {} and first elevation in {} {} ({} m) exceeds elevation error tolerance: {} m",
                            link.idx_curr,
                            link_next.idx_curr,
                            name,
                            elev_delta.get::<si::meter>(),
                            link_next.err_tol.as_ref().unwrap().max_elev_step.unwrap().get::<si::meter>()
                        ))
                        }
                    }
                    // Verify that coincident link points have matching heading within a tolerance parameter
                    // since `err_tol` propagates from `Network`, only one needs to be checked
                    // these unwraps should be guaranteed to be checked before this
                    if !link.headings.is_empty() && !link_next.headings.is_empty() {
                        let heading_delta = {
                            let w = [
                                link.headings.last().unwrap().heading,
                                link_next.headings[0].heading,
                            ];
                            (w[1] - w[0] + 3.0 * uc::REV / 2.0) % uc::REV - uc::REV / 2.0
                        };
                        if heading_delta
                            > link_next
                                .err_tol
                                .as_ref()
                                .unwrap()
                                .max_heading_step
                                .unwrap()
                        {
                            errors.push(anyhow!(
                            "Difference between last heading in current link {} and first heading in {} {} ({} deg) exceed heading error tolerance: {} deg",
                            link.idx_curr,
                            link_next.idx_curr,
                            name,
                            heading_delta.get::<si::degree>(),
                            link_next.err_tol.as_ref().unwrap().max_heading_step.unwrap().get::<si::degree>()
                        ))
                        }
                    }
                }
            } else if link.idx_next_alt.is_real() {
                errors.push(anyhow!(
                    "Next idx alt {} is real when next idx {} is fake!",
                    link.idx_next_alt,
                    link.idx_next,
                ));
            }

            // Validate prev
            if link.idx_prev.is_real() {
                for (link_prev, name) in [
                    (&self[link.idx_prev.idx()], "prev link"),
                    (&self[link.idx_prev_alt.idx()], "prev link alt"),
                ] {
                    if !link_prev.is_linked_next(link.idx_curr) {
                        errors.push(anyhow!(
                            "Current link {} with {} {} next idx {} and next idx alt {} do not point back!",
                            link.idx_curr,
                            name,
                            link_prev.idx_curr,
                            link_prev.idx_next,
                            link_prev.idx_next_alt,
                        ));
                    }
                    if link.idx_prev_alt.is_real() && link_prev.idx_next_alt.is_real() {
                        errors.push(anyhow!(
                            "Current link {} and {} {} have coincident switch points!",
                            link.idx_curr,
                            name,
                            link_prev.idx_curr,
                        ));
                    }
                }
            } else if link.idx_prev_alt.is_real() {
                errors.push(anyhow!(
                    "Prev idx alt {} is real when prev idx {} is fake!",
                    link.idx_prev_alt,
                    link.idx_prev
                ));
            }
        }

        // TODO: validate coincident data for adjoining links (e.g. elevation, heading, and maybe lat/lon)
        // either in the above for loop or an equivalent loop below
        // for (idx, link) in self.iter().enumerate().skip(1) {

        errors.make_err()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    impl Cases for Link {
        fn real_cases() -> Vec<Self> {
            vec![
                Self::valid(),
                Self {
                    idx_flip: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_next: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_next: LinkIdx::new(2),
                    idx_next_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev: LinkIdx::new(2),
                    idx_prev_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
            ]
        }
        fn fake_cases() -> Vec<Self> {
            vec![Self::default()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                Self {
                    elevs: Vec::<Elev>::invalid_cases().first().unwrap().to_vec(),
                    ..Self::valid()
                },
                Self {
                    elevs: Vec::<Elev>::new(),
                    ..Self::valid()
                },
                Self {
                    length: si::Length::ZERO,
                    ..Self::valid()
                },
                Self {
                    length: -uc::M,
                    ..Self::valid()
                },
                Self {
                    length: uc::M * f64::NAN,
                    ..Self::valid()
                },
                Self {
                    idx_curr: LinkIdx::default(),
                    ..Self::valid()
                },
                Self {
                    idx_flip: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_next_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_next: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_next: LinkIdx::new(3),
                    idx_next_alt: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_prev: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_prev: LinkIdx::new(3),
                    idx_prev_alt: LinkIdx::new(2),
                    ..Self::valid()
                },
            ]
        }
    }

    check_cases!(Link);

    #[test]
    fn check_elevs_start() {
        for mut link in Link::real_cases() {
            link.elevs.first_mut().unwrap().offset -= uc::M;
            link.validate().unwrap_err();
        }
    }

    #[test]
    fn check_elevs_end() {
        for mut link in Link::real_cases() {
            link.elevs.last_mut().unwrap().offset += uc::M;
            link.validate().unwrap_err();
        }
    }

    impl Cases for Vec<Link> {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![vec![], Self::valid()[..1].to_vec()]
        }
    }
    // check_cases!(Vec<Link>);
    // check_vec_elems!(Link);

    #[test]
    fn test_to_and_from_file_for_links() {
        let links = Vec::<Link>::valid();
        let tempdir = tempfile::tempdir().unwrap();
        let temp_file_path = tempdir.path().join("links_test2.yaml");
        links.to_file(temp_file_path.clone()).unwrap();
        assert_eq!(Vec::<Link>::from_file(temp_file_path, true).unwrap(), links);
        tempdir.close().unwrap();
    }

    #[test]
    fn test_set_speed_set_from_train_type() {
        let network_file_path = project_root::get_project_root()
            .unwrap()
            .join("python/altrios/resources/networks/Taconite.yaml");
        let network_speed_sets = {
            let network = Network::from_file(network_file_path, false);
            if let Err(err) = &network {
                panic!("{err}");
            }
            network
        }
        .unwrap();
        let mut network_speed_set = network_speed_sets.clone();
        network_speed_set
            .set_speed_set_for_train_type(TrainType::Freight)
            .unwrap();
        assert!(
            network_speed_sets.1[1].speed_sets[&TrainType::Freight]
                == *network_speed_set.1[1].speed_set.as_ref().unwrap()
        );
        assert!(network_speed_set.1[0].speed_sets.is_empty());
        assert!(network_speed_sets.1[0].speed_set.is_none());
    }
}
