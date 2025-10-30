import unittest

from altrios import package_root

import altrios as alt


class TestFuelConverter(unittest.TestCase):
    def test_load_from_yaml_absolute_path(self):
        test_file = (
            package_root()
            / "resources"
            / "powertrains"
            / "fuel_converters"
            / "wabtec_tier4.yaml"
        )
        fc = alt.FuelConverter.from_file(str(test_file))

        self.assertEqual(fc.to_pydict()["pwr_out_max_watts"], 3.356e6)
        self.assertEqual(fc.to_pydict()["pwr_idle_fuel_watts"], 1.97032784e04)
