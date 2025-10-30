import unittest

from .mock_resources import mock_consist_simulation


class TestConsistSimulation(unittest.TestCase):
    def test_walk_no_save(self):
        mock_sim = mock_consist_simulation(save_interval=None)

        mock_sim.walk()

    def test_walk_w_save(self):
        mock_sim = mock_consist_simulation(save_interval=1)

        mock_sim.walk()
