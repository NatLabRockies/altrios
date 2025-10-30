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
