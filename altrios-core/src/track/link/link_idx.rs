use crate::imports::*;
use serde::{de::Visitor, Deserializer, Serializer};
use std::fmt;
use std::io::prelude::*;

#[serde_api]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// See [supplementary documentation.](https://nrel.github.io/altrios/doc/rail-network.html)
pub struct LinkIdx {
    /// index of link within network
    idx: u32,
}

#[pyo3_api]
impl LinkIdx {
    #[new]
    fn __new__(idx: u32) -> Self {
        Self::new(idx)
    }
}

impl Init for LinkIdx {}
impl SerdeAPI for LinkIdx {}

pub const LINK_IDX_NA: LinkIdx = LinkIdx { idx: 0 };

impl LinkIdx {
    pub fn new(idx: u32) -> Self {
        Self { idx }
    }
    pub fn idx(&self) -> usize {
        self.idx.idx()
    }
}

impl std::hash::Hash for LinkIdx {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        hasher.write_u32(self.idx);
    }
}
impl nohash_hasher::IsEnabled for LinkIdx {}

impl fmt::Display for LinkIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.idx)
    }
}

impl Serialize for LinkIdx {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.idx)
    }
}

impl<'de> Deserialize<'de> for LinkIdx {
    fn deserialize<D>(deserializer: D) -> Result<LinkIdx, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LinkIdxVisitor;
        impl Visitor<'_> for LinkIdxVisitor {
            type Value = LinkIdx;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("integer")
            }

            fn visit_u32<E>(self, v: u32) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(LinkIdx::new(v))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if v >= u64::from(u32::MIN) && v <= u64::from(u32::MAX) {
                    Ok(LinkIdx::new(v as u32))
                } else {
                    Err(E::custom(format!("u32 out of range: {v}")))
                }
            }
        }

        deserializer.deserialize_u32(LinkIdxVisitor)
    }
}

impl Valid for LinkIdx {
    fn valid() -> Self {
        Self { idx: 1 }
    }
}

impl ObjState for LinkIdx {
    fn is_fake(&self) -> bool {
        self.idx == 0
    }
}

#[serde_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
/// Struct that contains a `Vec<LinkIdx>` for the purpose of providing `SerdeAPI` for
/// `Vec<LinkIdx>` in Python
pub struct LinkPath(pub Vec<LinkIdx>);

#[pyo3_api]
impl LinkPath {
    #[staticmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(filepath: &Bound<PyAny>) -> anyhow::Result<Self> {
        Self::from_csv_file(PathBuf::extract_bound(filepath)?)
    }

    #[pyo3(name = "to_csv_file")]
    fn to_csv_file_py(&self, filepath: &Bound<PyAny>) -> anyhow::Result<()> {
        self.to_csv_file(PathBuf::extract_bound(filepath)?)
    }
}

impl Init for LinkPath {}
impl SerdeAPI for LinkPath {}

impl AsRef<[LinkIdx]> for LinkPath {
    fn as_ref(&self) -> &[LinkIdx] {
        &self.0
    }
}

impl From<&Vec<LinkIdx>> for LinkPath {
    fn from(value: &Vec<LinkIdx>) -> Self {
        Self(value.to_vec())
    }
}

impl LinkPath {
    /// Load from csv file
    pub fn from_csv_file<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let mut lp = vec![];

        let file = File::open(filepath)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let pt_elem: LinkIdx = result?;
            lp.push(pt_elem);
        }
        if lp.is_empty() {
            bail!("Invalid PowerTrace file; PowerTrace is empty")
        } else {
            Ok(Self(lp))
        }
    }

    /// Save to csv file
    pub fn to_csv_file<P: AsRef<Path>>(&self, filepath: P) -> anyhow::Result<()> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filepath)?;
        let mut file = std::io::LineWriter::new(file);
        // write header
        file.write_all(b"link_idx\n")?;
        let mut wrtr = csv::WriterBuilder::new()
            // .has_headers(true) -- this line has no effect because of the custom `impl Deserialize`
            .from_writer(file);
        for link_idx in &self.0 {
            wrtr.serialize(link_idx)?;
        }
        wrtr.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod test_link_idx {
    use super::*;
    use crate::testing::*;

    impl Cases for LinkIdx {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid()]
        }
        fn fake_cases() -> Vec<Self> {
            vec![Self::new(0)]
        }
    }
    check_cases!(LinkIdx);

    #[test]
    fn check_new() {
        assert!(LinkIdx::new(0) == LinkIdx { idx: 0 });
        assert!(LinkIdx::new(1) == LinkIdx { idx: 1 });
        assert!(LinkIdx::new(4294967295) == LinkIdx { idx: 4294967295 });
    }
}
