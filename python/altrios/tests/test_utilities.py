import tempfile
import unittest
from pathlib import Path

from .mock_resources import *

import altrios as alt


class TestUtilities(unittest.TestCase):
    def test_copy_demo_files(self):
        v = f"v{alt.__version__}"
        prepend_str = f"# %% Copied from ALTRIOS version '{v}'. Guaranteed compatibility with this version only.\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            tf_path = Path(tmpdir)
            alt.copy_demo_files(tf_path)
            with open(next(tf_path.glob("*demo*.py")), "r") as file:
                lines = file.readlines()
                assert prepend_str in lines[0]
                assert len(lines) > 3
