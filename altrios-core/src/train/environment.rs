use super::train_imports::*;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Container
pub struct TemperatureTraceBuilder {
    /// simulation elapsed time
    pub time: Vec<si::Time>,
    /// ambient temperature at sea level
    pub temp_at_sea_level: Vec<si::ThermodynamicTemperature>,
}

#[pyo3_api]
impl TemperatureTraceBuilder {
    #[staticmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(pathstr: String) -> anyhow::Result<Self> {
        Self::from_csv_file(&pathstr)
    }

    fn __len__(&self) -> usize {
        self.len()
    }
}

impl Init for TemperatureTraceBuilder {}
impl SerdeAPI for TemperatureTraceBuilder {}

impl TemperatureTraceBuilder {
    fn empty() -> Self {
        Self {
            time: Vec::new(),
            temp_at_sea_level: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, tt_element: TemperatureTraceElement) {
        self.time.push(tt_element.time);
        self.temp_at_sea_level.push(tt_element.temp_at_sea_level);
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or_else(|| self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Load cycle from csv file
    pub fn from_csv_file(pathstr: &str) -> Result<Self, anyhow::Error> {
        let pathbuf = PathBuf::from(&pathstr);

        // create empty temperature trace to be populated
        let mut tt = Self::empty();

        let file = File::open(pathbuf)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let tt_elem: TemperatureTraceElement = result?;
            tt.push(tt_elem);
        }
        if tt.is_empty() {
            bail!("Invalid TemperatureTrace file; TemperatureTrace is empty")
        } else {
            Ok(tt)
        }
    }
}

impl Default for TemperatureTraceBuilder {
    fn default() -> Self {
        let time_s: Vec<f64> = (0..1).map(|x| x as f64).collect();
        let mut tt = Self {
            time: time_s.iter().map(|t| *t * uc::S).collect(),
            temp_at_sea_level: vec![(22.0 + 273.15) * uc::KELVIN],
        };
        tt.init().unwrap();
        tt
    }
}

/// Element of [TemperatureTrace].  Used for vec-like operations.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct TemperatureTraceElement {
    /// simulation time
    time: si::Time,
    /// ambient temperature at sea level
    pub temp_at_sea_level: si::ThermodynamicTemperature,
}

#[serde_api]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Container for an interpolator of temperature at sea level (to be corrected for altitude)
pub struct TemperatureTrace(pub(crate) Interp1DOwned<f64, strategy::Linear>);

#[pyo3_api]
impl TemperatureTrace {}

impl Init for TemperatureTrace {}
impl SerdeAPI for TemperatureTrace {}

impl TemperatureTrace {
    pub fn get_temp_at_time_and_elev(
        &self,
        time: si::Time,
        elev: si::Length,
    ) -> anyhow::Result<si::ThermodynamicTemperature> {
        Ok(self.get_temp_at_elev(self.get_temp_at_time_and_sea_level(time)?, elev))
    }

    fn get_temp_at_time_and_sea_level(
        &self,
        time: si::Time,
    ) -> anyhow::Result<si::ThermodynamicTemperature> {
        Ok(self
            .0
            .interpolate(&[time.get::<si::second>()])
            .map(|te| (te + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?)
    }

    /// Source: <https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html>  
    ///
    /// # Equations used
    /// T = 15.04 - .00649 * h  
    fn get_temp_at_elev(
        &self,
        temp_at_sea_level: si::ThermodynamicTemperature,
        elev: si::Length,
    ) -> si::ThermodynamicTemperature {
        (((15.04 - 0.00649 * elev.get::<si::meter>())
            + (temp_at_sea_level.get::<si::degree_celsius>() - 15.04))
            + uc::CELSIUS_TO_KELVIN)
            * uc::KELVIN
    }
}

impl Serialize for TemperatureTrace {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let builder: TemperatureTraceBuilder = TemperatureTraceBuilder::try_from(self.clone())
            .map_err(|e| serde::ser::Error::custom(format!("{e:?}")))?;
        builder.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TemperatureTrace {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value: TemperatureTraceBuilder = TemperatureTraceBuilder::deserialize(deserializer)?;
        let tt: Self =
            Self::try_from(value).map_err(|e| serde::de::Error::custom(format!("{e:?}")))?;
        Ok(tt)
    }
}

impl TryFrom<TemperatureTraceBuilder> for TemperatureTrace {
    type Error = anyhow::Error;
    fn try_from(value: TemperatureTraceBuilder) -> anyhow::Result<Self> {
        Ok(Self(Interp1D::new(
            value.time.iter().map(|t| t.get::<si::second>()).collect(),
            value
                .temp_at_sea_level
                .iter()
                .map(|te| te.get::<si::degree_celsius>())
                .collect(),
            strategy::Linear,
            Extrapolate::Clamp,
        )?))
    }
}

impl TryFrom<TemperatureTrace> for TemperatureTraceBuilder {
    type Error = anyhow::Error;
    fn try_from(value: TemperatureTrace) -> anyhow::Result<Self> {
        Ok(Self {
            time: value.0.data.grid[0].iter().map(|x| *x * uc::S).collect(),
            temp_at_sea_level: value
                .0
                .data
                .values
                .iter()
                .map(|y| (*y + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
                .collect(),
        })
    }
}

impl Default for TemperatureTrace {
    fn default() -> Self {
        Self::try_from(TemperatureTraceBuilder::default()).unwrap()
    }
}

impl TemperatureTrace {}
