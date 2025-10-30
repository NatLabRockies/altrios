use super::*;

/// Traits defining power flow interfaces for electric machines
pub trait ElectricMachine {
    /// Sets current max power output given `pwr_in_max` from upstream component
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()>;
    /// Sets current max power output rate given `pwr_rate_in_max` from upstream component
    fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate) -> anyhow::Result<()>;
}
