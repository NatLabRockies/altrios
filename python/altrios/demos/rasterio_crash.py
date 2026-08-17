# %%
"""This file is a demo snippet to reproduce the crash that we are seeing in rasterio
with the seemless-3dep package when we are trying to download the elevation data
from USGS.  I wanted something small to reproduce and troubleshoot the problem that
could be shared on stackoverflow, other people's computers, and github."""

import sys
import os

env_folder_path = os.path.dirname(sys.executable)
os.environ["GDAL_DATA"] = env_folder_path + "/Library/share/gdal"
os.environ["GDAL_DRIVER_PATH"] = env_folder_path + "/Library/lib/gdalplugins"
os.environ["GEOTIFF_CSV"] = env_folder_path + "/Library/share/epsg_csv"
os.environ["PROJ_LIB"] = env_folder_path + "/Library/share/proj"


import seamless_3dep as s3dep
from pathlib import Path

# %%

bounds = (
    -118.0932754546651,
    33.50715209143235,
    -116.39542435390693,
    35.502944655119265,
)
LayerTiffDir = Path("3DEP Download Test")
LayerTiffDir.mkdir(parents=True, exist_ok=True)
tiff_files = s3dep.get_dem(bounds, LayerTiffDir)
