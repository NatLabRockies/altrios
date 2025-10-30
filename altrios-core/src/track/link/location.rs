use crate::imports::*;
use serde_this_or_that::as_bool;

use super::link_idx::*;

#[serde_api]
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Location {
    /// User-defined name of the location/terminal
    #[serde(rename = "Location ID")]
    pub location_id: String,
    #[serde(rename = "Offset (m)")]
    pub offset: si::Length,
    #[serde(rename = "Link Index")]
    pub link_idx: LinkIdx,
    #[serde(rename = "Is Front End")]
    #[serde(deserialize_with = "as_bool")]
    pub is_front_end: bool,
    #[serde(rename = "Grid Emissions Region")]
    pub grid_emissions_region: String,
    #[serde(rename = "Electricity Price Region")]
    pub electricity_price_region: String,
    #[serde(rename = "Liquid Fuel Price Region")]
    pub liquid_fuel_price_region: String,
}

#[pyo3_api]
impl Location {}

impl Init for Location {}
impl SerdeAPI for Location {}

pub type LocationMap = HashMap<String, Vec<Location>>;

#[cfg(feature = "pyo3")]
#[cfg_attr(feature = "pyo3", pyfunction(name = "import_locations"))]
pub fn import_locations_py(filepath: &Bound<PyAny>) -> anyhow::Result<LocationMap> {
    import_locations(PathBuf::extract_bound(filepath)?)
}

pub fn import_locations<P: AsRef<Path>>(filepath: P) -> anyhow::Result<LocationMap> {
    let file_read = File::open(filepath.as_ref())?;
    let mut reader = csv::Reader::from_reader(file_read);
    let mut location_map = LocationMap::default();
    for result in reader.deserialize() {
        let location: Location = result?;
        if !location_map.contains_key(&location.location_id) {
            location_map.insert(location.location_id.clone(), vec![]);
        }
        location_map
            .get_mut(&location.location_id)
            .unwrap()
            .push(location);
    }
    Ok(location_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locations_import() {
        import_locations(Path::new("./src/track/link/locations.csv")).unwrap();
    }
}
