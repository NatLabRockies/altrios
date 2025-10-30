import unittest

from altrios import package_root

import altrios as alt


class TestRES(unittest.TestCase):
    def test_load_from_excel(self):
        test_file = (
            package_root()
            / "resources"
            / "powertrains"
            / "reversible_energy_storages"
            / "Kokam_NMC_75Ah.xlsx"
        )
        res = alt.ReversibleEnergyStorage.from_excel(test_file)
        res_dict = res.to_pydict()

        self.assertEqual(res_dict["min_soc"], 0.1)
        self.assertEqual(res_dict["max_soc"], 0.9)

    def test_load_from_yaml_absolute_path(self):
        test_file = (
            package_root()
            / "resources"
            / "powertrains"
            / "reversible_energy_storages"
            / "Kokam_NMC_75Ah_flx_drive.yaml"
        )
        res = alt.ReversibleEnergyStorage.from_file(str(test_file))
        res_dict = res.to_pydict()

        self.assertEqual(res_dict["min_soc"], 0.05)
        self.assertEqual(res_dict["max_soc"], 0.95)
