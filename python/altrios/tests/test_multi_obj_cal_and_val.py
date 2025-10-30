import unittest
from .mock_resources import *


class TestLocomotive(unittest.TestCase):
    def test_pymoo_mod_err_build(self):
        return None
        # TODO: remove line above this and fix this test at some point
        from altrios.optimization import cal_and_val as cval
        from altrios import LocomotiveSimulation, Locomotive, PowerTrace
        mock_df = mock_pymoo_conv_cal_df()
        pt = PowerTrace(
            mock_df['time [s]'].to_numpy(),
            mock_df['Tractive Power [W]'].to_numpy(),
            engine_on=mock_df['engine_on'],
        )
        loco_sim = LocomotiveSimulation(
            loco_unit=Locomotive.default(),
            power_trace=pt,
            save_interval=1)
        loco_sim_ser = loco_sim.to_json()
        mod_err = cval.ModelError(
            ser_model_dict={0: loco_sim_ser},
            dfs={0: mock_df},
            objectives=[(
                "Fuel Power [W]",
                "loco_unit.fc.history.pwr_fuel_watts"
            )
            ],
            params=(
                "loco_unit.fc.eta_max",
                "loco_unit.fc.eta_range",
                # "loco_unit.fc.pwr_idle_fuel_watts",
            ),
            model_type='LocomotiveSimulation',
            verbose=False,
        )

        updated_mod0 = mod_err.update_params(xs=[0.433, 0.233])
        self.assertAlmostEqual(
            updated_mod0[0].loco_unit.fc.eta_max, 0.433)
        self.assertAlmostEqual(
            updated_mod0[0].loco_unit.fc.eta_range, 0.233)
        error1 = mod_err.get_errors(updated_mod0)[0]['Fuel Power [W]']
        updated_mod1 = mod_err.update_params(xs=[0.433, 0.243])
        self.assertAlmostEqual(
            updated_mod1[0].loco_unit.fc.eta_max, 0.433)
        self.assertAlmostEqual(
            updated_mod1[0].loco_unit.fc.eta_range, 0.243)
        error2 = mod_err.get_errors(updated_mod1)[0]['Fuel Power [W]']
        self.assertTrue(error1 != error2)
        updated_mod2 = mod_err.update_params(xs=[0.463, 0.243])
        self.assertAlmostEqual(
            updated_mod2[0].loco_unit.fc.eta_max, 0.463)
        self.assertAlmostEqual(
            updated_mod2[0].loco_unit.fc.eta_range, 0.243)
        error3 = mod_err.get_errors(updated_mod2)[0]['Fuel Power [W]']
        self.assertTrue(error2 != error3)
        self.assertTrue(error1 != error3)


if __name__ == "__main__":
    a = TestLocomotive()
    a.test_pymoo_mod_err_build()
