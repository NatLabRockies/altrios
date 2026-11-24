use super::*;
use crate::si;

#[test]
/// Unit test for standalone consist.
fn test_consist() {
    let mut consist = Consist::default();

    assert_eq!(
        *consist
            .state
            .pwr_out_max
            .get_fresh(|| format_dbg!())
            .unwrap(),
        si::Power::ZERO
    );
    assert_eq!(
        *consist
            .state
            .pwr_rate_out_max
            .get_fresh(|| format_dbg!())
            .unwrap(),
        si::PowerRate::ZERO
    );
    assert_eq!(
        *consist
            .state
            .pwr_regen_max
            .get_fresh(|| format_dbg!())
            .unwrap(),
        si::Power::ZERO
    );
    consist.check_and_reset(|| format_dbg!()).unwrap();
    consist.set_pwr_aux(Some(true)).unwrap();
    consist
        .set_curr_pwr_max_out(
            None,
            None,
            Some(5e6 * uc::LB),
            Some(10.0 * uc::MPH),
            1.0 * uc::S,
        )
        .unwrap();
    assert!(
        *consist
            .state
            .pwr_out_max
            .get_fresh(|| format_dbg!())
            .unwrap()
            > si::Power::ZERO
    );
    assert!(
        *consist
            .state
            .pwr_rate_out_max
            .get_fresh(|| format_dbg!())
            .unwrap()
            > si::PowerRate::ZERO
    );
    assert!(
        *consist
            .state
            .pwr_regen_max
            .get_fresh(|| format_dbg!())
            .unwrap()
            == si::Power::ZERO
    );

    assert_eq!(
        *consist
            .state
            .energy_out
            .get_stale(|| format_dbg!())
            .unwrap(),
        si::Energy::ZERO
    );
    assert_eq!(
        *consist
            .state
            .energy_fuel
            .get_stale(|| format_dbg!())
            .unwrap(),
        si::Energy::ZERO
    );
    assert_eq!(
        *consist
            .state
            .energy_reves
            .get_stale(|| format_dbg!())
            .unwrap(),
        si::Energy::ZERO
    );
    consist
        .solve_energy_consumption(
            uc::W * 1e6,
            Some(5e6 * uc::LB),
            Some(10.0 * uc::MPH),
            uc::S * 1.0,
            Some(true),
        )
        .unwrap();
    consist.set_cumulative(uc::S, || format_dbg!()).unwrap();
    assert!(
        *consist
            .state
            .energy_out
            .get_fresh(|| format_dbg!())
            .unwrap()
            > si::Energy::ZERO
    );
    assert!(
        *consist
            .state
            .energy_fuel
            .get_fresh(|| format_dbg!())
            .unwrap()
            > si::Energy::ZERO
    );
    assert!(
        *consist
            .state
            .energy_reves
            .get_fresh(|| format_dbg!())
            .unwrap()
            > si::Energy::ZERO
    );
}

#[test]
/// The energy solve gives each locomotive the same share of the train mass as the power limits
/// (weighted by usable battery energy), so the hybrid's engine-on SOC buffer uses its share.
fn test_solve_uses_loco_share_of_train_mass() {
    use crate::consist::locomotive::HybridPowertrainControls;
    // the default consist has a battery-electric and a hybrid locomotive
    let mut consist = Consist::default();
    let train_mass = 5e6 * uc::LB;
    let speed = 10.0 * uc::MPH;
    consist.check_and_reset(|| format_dbg!()).unwrap();
    consist.set_pwr_aux(Some(true)).unwrap();
    consist
        .set_curr_pwr_max_out(None, None, Some(train_mass), Some(speed), 1.0 * uc::S)
        .unwrap();
    consist
        .solve_energy_consumption(
            uc::W * 1e6,
            Some(train_mass),
            Some(speed),
            uc::S * 1.0,
            Some(true),
        )
        .unwrap();

    let res_total_usable_energy = consist.loco_vec.iter().fold(si::Energy::ZERO, |acc, l| {
        acc + l
            .reversible_energy_storage()
            .map(|res| res.energy_capacity_usable())
            .unwrap_or(si::Energy::ZERO)
    });
    let hel = consist
        .loco_vec
        .iter()
        .find_map(|l| match &l.loco_type {
            PowertrainType::HybridLoco(hel) => Some(hel),
            _ => None,
        })
        .unwrap();
    let HybridPowertrainControls::RGWDB(rgwdb) = &hel.pt_cntrl;
    let mass_for_loco = hel.res.energy_capacity_usable() / res_total_usable_energy * train_mass;
    assert!(mass_for_loco < train_mass);
    let expected: si::Ratio = 0.5
        * mass_for_loco
        * (rgwdb
            .speed_soc_fc_on_buffer
            .unwrap()
            .powi(typenum::P2::new())
            - speed.powi(typenum::P2::new()))
        * rgwdb.speed_soc_fc_on_buffer_coeff.unwrap()
        / hel.res.energy_capacity_usable()
        + hel.res.min_soc;
    let actual = *rgwdb
        .state
        .soc_fc_on_buffer
        .get_fresh(|| format_dbg!())
        .unwrap();
    assert!(
        almost_eq_uom(&actual, &expected, None),
        "{actual:?} vs {expected:?}"
    );
}
