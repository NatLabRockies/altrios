/// Provides method for determining whether [FuelConverter] is currently active/on
pub trait FuelConverterOn {
    fn fc_on(&self) -> anyhow::Result<bool>;
}
