import unittest
import polars as pl
from .mock_resources import *
import altrios as alt
from altrios import metric_calculator

class TestMetricCalculator(unittest.TestCase):
    def test_dummy_sim(self):        
        scenario_infos = []
        years = [2020, 2021]
        for year in years:
            scenario_infos.append(metric_calculator.ScenarioInfo(
                mock_speed_limit_train_simulation_vector(scenario_year=year), 
                21,
                False,
                year, 
                pl.DataFrame(),
                pl.DataFrame(),
                pl.DataFrame(),
                pl.DataFrame(),
                pl.DataFrame(),
                pl.DataFrame(),
                False))
            
        for info in scenario_infos:
            tkm = metric_calculator.calculate_freight_moved(info,'million tonne-mi')
            self.assertEqual(tkm.filter(pl.col("Metric") == pl.lit("Freight_Moved")).get_column("Units").len(), 1)

