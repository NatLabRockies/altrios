# -*- coding: utf-8 -*-
"""
NetworkBuilder (AWS 3DEP COG window-read version)

Key changes vs original:
- No seamless_3dep download / no GDAL BuildVRT.
- Draping uses AWS public USGS 3DEP COGs via rasterio HTTP range requests.
- Reads only small windows per link (NOT full tile into memory).
- Fixes savgol warning logic (no spam when apply_savgol=False).
- Fixes drape_geometry_aws signature and main calling order.

Author: Qianqian Tong (adapted from Garrett Anderson's builder)
"""

import os
import sys
import ast
import math
import uuid
import pickle
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import overpy
import shapely
import shapely.ops
from shapely.geometry import LineString, Point
from geographiclib.geodesic import Geodesic
import momepy

import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from rasterio.transform import rowcol
from pyproj import Transformer

from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from tenacity import retry, stop_after_attempt, wait_fixed
from diskcache import Cache


# GDAL/PROJ env (keep as current had)
env_folder_path = os.path.dirname(sys.executable)
if Path(env_folder_path + "/Library/share/gdal").exists():
    os.environ["GDAL_DATA"] = env_folder_path + "/Library/share/gdal"
    os.environ["GDAL_DRIVER_PATH"] = env_folder_path + "/Library/lib/gdalplugins"
    os.environ["GEOTIFF_CSV"] = env_folder_path + "/Library/share/epsg_csv"
    os.environ["PROJ_LIB"] = env_folder_path + "/Library/share/proj"
    os.environ["PROJ_NETWORK"] = "OFF"
else:
    import pyproj
    pyproj.datadir.set_data_dir(os.path.dirname(env_folder_path) + "/share/proj")
    os.environ["PROJ_LIB"] = os.path.dirname(env_folder_path) + "/share/proj"

cache = Cache("./network_builder_cache")


# AWS 3DEP COG settings
AWS_3DEP_BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
AWS_TILE_FMT = "{tile}/USGS_13_{tile}.tif"


def aws_tile_name(lat_deg: int, lon_deg: int) -> str:
    """
    USGS 1x1 degree tile name: nXXwYYY, e.g. n30w097
    lat_deg, lon_deg are integer tile origins:
      lat_deg = floor(lat), lon_deg = floor(lon)
    """
    lat_prefix = "s" if lat_deg < 0 else "n"
    lon_prefix = "w" if lon_deg < 0 else "e"
    return f"{lat_prefix}{abs(lat_deg):02d}{lon_prefix}{abs(lon_deg):03d}"


def tiles_for_bounds(bounds4326: Tuple[float, float, float, float]) -> List[str]:
    """
    bounds4326 = (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    Returns all 1x1 tiles intersecting bounds.
    """
    min_lon, min_lat, max_lon, max_lat = bounds4326

    lon_start = math.floor(min_lon)
    lon_end = math.ceil(max_lon) - 1
    lat_start = math.floor(min_lat)
    lat_end = math.ceil(max_lat) - 1

    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append(aws_tile_name(lat, lon))
    return sorted(set(tiles))


def aws_cog_url(tile: str) -> str:
    return f"{AWS_3DEP_BASE}/{AWS_TILE_FMT.format(tile=tile)}"


@retry(stop=stop_after_attempt(6), wait=wait_fixed(10))
def call_osm(layer_query: str):
    print("calling osm server....")
    api = overpy.Overpass()
    api.default_max_retry_count = 5
    return api.query(layer_query)


def point_from_coord(coord):
    try:
        return shapely.to_wkt(Point(coord[0], coord[1]))
    except Exception:
        return np.nan


def heading_difference(heading_a: np.float64, heading_b: np.float64):
    heading_a = np.mod(heading_a, 360.0)
    heading_b = np.mod(heading_b, 360.0)
    absolute_difference = np.abs(heading_a - heading_b)
    return np.min([absolute_difference, 360.0 - absolute_difference])


def smooth_link_data(
    offsets,
    values,
    window_length=100,
    order=3,
    segment_length=2,
    unwrap_heading=False,
    interp_offsets=None,
    interp_values=None,
    apply_savgol=True,
):
    """
    - If apply_savgol=False: do NOT print warning. Just return linear-interp result.
    - Only warn when smoothing fails due to NaNs.
    """
    offsets = np.asarray(offsets, dtype=float)
    values = np.asarray(values, dtype=float)

    if unwrap_heading:
        values = np.unwrap(values, period=360)

    # Build uniform samples for filter
    if interp_offsets is None or interp_values is None:
        f = interpolate.interp1d(offsets, values, bounds_error=False, fill_value="extrapolate")
        interp_offsets = np.arange(0, np.max(offsets) + 1e-9, segment_length)
        interp_values = f(interp_offsets)

    # If user disables SavGol, just return interpolation back to offsets.
    if not apply_savgol:
        f2 = interpolate.interp1d(
            interp_offsets, interp_values, bounds_error=False, fill_value="extrapolate"
        )
        out = f2(offsets)
        out[0] = values[0]
        out[-1] = values[-1]
        return list(map(float, out))

    # Apply SavGol safely
    wl = int(min(len(interp_values), window_length))
    if wl < 5:
        # too short to filter
        f2 = interpolate.interp1d(interp_offsets, interp_values, bounds_error=False, fill_value="extrapolate")
        out = f2(offsets)
        out[0] = values[0]
        out[-1] = values[-1]
        return list(map(float, out))

    # Savgol requires odd window length
    if wl % 2 == 0:
        wl -= 1
    poly = int(min(order, wl - 1))
    if poly < 1:
        poly = 1

    savgol_vals = savgol_filter(interp_values, wl, poly, mode="nearest")

    f2 = interpolate.interp1d(
        interp_offsets, savgol_vals, bounds_error=False, fill_value="extrapolate"
    )
    out = f2(offsets)
    out[0] = values[0]
    out[-1] = values[-1]

    if np.isnan(out).any():
        print("WARNING: smoothing produced NaN, falling back to linear interpolation.")
        f2 = interpolate.interp1d(interp_offsets, interp_values, bounds_error=False, fill_value="extrapolate")
        out = f2(offsets)
        out[0] = values[0]
        out[-1] = values[-1]

    return list(map(float, out))


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


# AWS tile dataset cache (open once; read windows on demand)
class AwsCogPool:
    """
    Opens AWS COG datasets lazily and keeps them open.
    Samples elevations by reading one window per link per tile (fast, low HTTP calls).
    """

    def __init__(self, tile_urls: Dict[str, str]):
        self.tile_urls = tile_urls
        self.ds_map: Dict[str, rasterio.io.DatasetReader] = {}
        self.transformer_map: Dict[str, Transformer] = {}

    def open(self, tile: str) -> rasterio.io.DatasetReader:
        if tile in self.ds_map:
            return self.ds_map[tile]
        url = self.tile_urls[tile]
        try:
            ds = rasterio.open(url)
        except RasterioIOError as e:
            raise RuntimeError(f"Failed to open AWS COG for tile={tile}, url={url}") from e
        self.ds_map[tile] = ds
        self.transformer_map[tile] = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return ds

    def _lonlat_to_rc(self, tile: str, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ds = self.open(tile)
        tf = self.transformer_map[tile]
        x, y = tf.transform(lon, lat)

        b = ds.bounds
        inside = (x >= b.left) & (x <= b.right) & (y >= b.bottom) & (y <= b.top)
        if not np.any(inside):
            rr = np.zeros_like(lon, dtype=int)
            cc = np.zeros_like(lon, dtype=int)
            return rr, cc, inside

        rr, cc = rowcol(ds.transform, x[inside], y[inside], op=np.floor)
        rr = np.asarray(rr, dtype=int)
        cc = np.asarray(cc, dtype=int)

        # clip
        rr = np.clip(rr, 0, ds.height - 1)
        cc = np.clip(cc, 0, ds.width - 1)

        # scatter back
        rr_full = np.zeros_like(lon, dtype=int)
        cc_full = np.zeros_like(lon, dtype=int)
        rr_full[inside] = rr
        cc_full[inside] = cc
        return rr_full, cc_full, inside

    def sample_link_points(self, lon: np.ndarray, lat: np.ndarray, tiles_in_use: List[str]) -> np.ndarray:
        """
        For a set of lon/lat points (1 link), try tiles one-by-one.
        For each tile, read ONE window covering all points inside that tile.
        """
        elev = np.full(lon.shape, np.nan, dtype=np.float32)

        for tile in tiles_in_use:
            ds = self.open(tile)
            rr, cc, inside = self._lonlat_to_rc(tile, lon, lat)
            need = np.isnan(elev) & inside
            if not np.any(need):
                continue

            r = rr[need]
            c = cc[need]
            r0, r1 = int(r.min()), int(r.max())
            c0, c1 = int(c.min()), int(c.max())

            # Build window (inclusive)
            win = Window.from_slices((r0, r1 + 1), (c0, c1 + 1))

            # Read one window
            arr = ds.read(1, window=win, masked=True).filled(np.nan).astype(np.float32)

            # Index within window
            elev_vals = arr[r - r0, c - c0]
            elev[need] = elev_vals

        return elev


class NetworkBuilder:
    def __init__(
        self,
        input_geopackage_path: Path | str,
        data_folder: Path | str,
        builder_name: str,
        milepost_layer_name: str,
        speed_restriction_path: dict,
        input_regions_layer_name="network_regions",
        input_locations_layer_name="network_locations",
    ):
        self.input_geopackage = Path(input_geopackage_path)
        self.input_regions_layer_name = input_regions_layer_name
        self.input_locations_layer_name = input_locations_layer_name
        self.milepost_layer_name = milepost_layer_name
        self.restriction_table_paths = speed_restriction_path

        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=False, exist_ok=True)

        self.builder_name = builder_name
        self.geopackage_path = Path(self.data_folder, builder_name + ".gpkg")

        self.switchresult = []
        self.geod = Geodesic.WGS84

    # GPKG helpers
    def delete_and_create_layer(self, layername: str, gdf: gpd.GeoDataFrame):
        if self.geopackage_path.exists():
            try:
                if layername in fiona.listlayers(self.geopackage_path.resolve()):
                    fiona.remove(self.geopackage_path, layer=layername)
            except Exception:
                pass
        gdf.to_file(self.geopackage_path, driver="GPKG", layer=layername, mode="a")

    # -----------------------
    # Step 1: parse regions
    # -----------------------
    def input_geopackage_parsing(self):
        if self.geopackage_path.exists():
            try:
                for layername in fiona.listlayers(self.geopackage_path.resolve()):
                    fiona.remove(self.geopackage_path, layer=layername)
                    print(f"deleted: {layername}")
            except Exception:
                pass

        regions_gdf = gpd.read_file(self.input_geopackage, layer=self.input_regions_layer_name)

        for _, row in regions_gdf.iterrows():
            single_region_gdf = gpd.GeoDataFrame(
                [{"region_name": row.region_name}],
                geometry=[row.geometry],
                crs="EPSG:4326",
            )
            self.delete_and_create_layer(row.region_name, single_region_gdf)

    # Step 2: OSM download
    def download_osm_data(self):
        base_query = """[out:xml] [timeout:999];
                        (
                            nwr[railway]({},{},{},{});
                        );
                        (._;>;);
                        out body;"""

        # purge previous derived layers
        for layername in list(fiona.listlayers(self.geopackage_path)):
            if any(s in layername for s in [
                "_osm", "_switches", "_split", "_offhead", "_bothdir",
                "_draped", "_linked", "_grouped", "_TOjoined", "_TOPoint"
            ]):
                fiona.remove(self.geopackage_path, layer=layername)
                print(f"deleted: {layername}")

        for layername in fiona.listlayers(self.geopackage_path):
            print(f"download osm data for {layername}")
            geolayer = gpd.read_file(self.geopackage_path, layer=layername)

            bounds = tuple(geolayer.total_bounds)  # (minx, miny, maxx, maxy)
            layer_query = base_query.format(bounds[1], bounds[0], bounds[3], bounds[2])

            result = call_osm(layer_query)
            track_data = result.ways

            track_gdfs = []
            all_switch_gdfs = []

            for way in track_data:
                coords = []
                node_tags = []
                for node in way.nodes:
                    coords.append([node.lon, node.lat])
                    node_tags.append(node.tags)
                    if "railway" in node.tags and node.tags["railway"] == "switch":
                        switch_gdf = gpd.GeoDataFrame(
                            data=[node.tags],
                            geometry=[Point([node.lon, node.lat])],
                            crs="EPSG:4326",
                        )
                        switch_gdf["osm_link_id"] = way.id
                        all_switch_gdfs.append(switch_gdf)

                way_gdf = gpd.GeoDataFrame(
                    data=[way.tags],
                    geometry=[LineString(coords)],
                    crs="EPSG:4326",
                )
                way_gdf["Node Tags"] = str(node_tags)
                way_gdf["osm_id"] = way.id
                track_gdfs.append(way_gdf)

            for node_ in result.nodes:
                if ("railway" in node_.tags) or ("railway:switch" in node_.tags):
                    if (node_.tags.get("railway") == "junction") or ("railway:switch" in node_.tags):
                        switch_gdf = gpd.GeoDataFrame(
                            data=[{"railway": "junction"}],
                            geometry=[Point([node_.lon, node_.lat])],
                            crs="EPSG:4326",
                        )
                        switch_gdf["osm_link_id"] = node_.id
                        all_switch_gdfs.append(switch_gdf)

            if len(track_gdfs) == 0:
                raise RuntimeError(f"No OSM ways returned for layer {layername}.")

            track_gdf = pd.concat(track_gdfs, ignore_index=True)
            switch_gdf = pd.concat(all_switch_gdfs, ignore_index=True) if len(all_switch_gdfs) else gpd.GeoDataFrame(
                {"railway": []}, geometry=[], crs="EPSG:4326"
            )

            # filters (keep your intent)
            if "railway" in track_gdf.columns:
                bad_rail = {
                    "abandoned","construction","defect_detector","dismantled","disused",
                    "light_rail","miniature","monorail","narrow_gauge","platform","proposed",
                    "razed","signal_box","station","tram","traverser","turntable","yard"
                }
                track_gdf = track_gdf[~track_gdf["railway"].isin(bad_rail)]

            if "usage" in track_gdf.columns:
                track_gdf = track_gdf[~track_gdf["usage"].isin({"military","industrial","tourism"})]

            if "service" in track_gdf.columns:
                track_gdf = track_gdf[~track_gdf["service"].isin({"construction","spur","yard"})]

            if "highway" in track_gdf.columns:
                track_gdf = track_gdf[track_gdf["highway"].isna()]

            track_gdf = gpd.GeoDataFrame(track_gdf, geometry=track_gdf.geometry, crs="EPSG:4326")
            track_gdf = track_gdf.clip(geolayer)

            track_gdf.to_file(self.geopackage_path, driver="GPKG", layer=layername + "_osm", mode="a")
            switch_gdf.to_file(self.geopackage_path, driver="GPKG", layer=layername + "_switches", mode="a")

    # Step 3: clean geometry
    def clean_geometry(self):
        for layername in fiona.listlayers(self.geopackage_path):
            if "_osm" not in layername:
                continue

            trackdata = gpd.read_file(self.geopackage_path, layer=layername)
            switch_data = gpd.read_file(self.geopackage_path, layer=layername.replace("_osm", "_switches"))

            print("beginning removal of false nodes")
            trackdata = momepy.remove_false_nodes(trackdata)
            print("removed false nodes")

            split_trackdata = []
            for _, row in trackdata.iterrows():
                intersecting_switches = switch_data.loc[switch_data.geometry.intersects(row.geometry), :]
                geometry_to_split = [row.geometry]

                for sw in intersecting_switches.geometry:
                    temp = []
                    for link in geometry_to_split:
                        split_result = shapely.ops.split(link, sw)
                        for seg in split_result.geoms:
                            temp.append(seg)
                    geometry_to_split = temp

                for link in geometry_to_split:
                    row2 = row.drop("geometry")
                    split_trackdata.append(
                        gpd.GeoDataFrame(data=[row2], geometry=[link], crs="EPSG:4326")
                    )

            split_trackdata = pd.concat(split_trackdata, ignore_index=True)
            split_trackdata["uid"] = split_trackdata.geometry.apply(lambda _: uuid.uuid1())

            out_layer = layername.replace("_osm", "_split")
            try:
                fiona.remove(self.geopackage_path, layer=out_layer)
            except Exception:
                pass

            split_trackdata.to_file(self.geopackage_path, driver="GPKG", layer=out_layer, mode="a")

    # Step 4: reverse links
    def create_reverse_links(self):
        for layername in fiona.listlayers(self.geopackage_path):
            if "_split" not in layername:
                continue

            trackdata = gpd.read_file(self.geopackage_path, layer=layername)
            reverse_trackdata = trackdata.copy()
            reverse_trackdata.geometry = reverse_trackdata.geometry.reverse()

            reverse_trackdata["direction"] = "reverse"
            trackdata["direction"] = "forward"
            trackdata = pd.concat([trackdata, reverse_trackdata], ignore_index=True)

            trackdata["start coord"] = trackdata.apply(lambda x: point_from_coord(x["geometry"].coords[0]), axis=1)
            trackdata["end coord"] = trackdata.apply(lambda x: point_from_coord(x["geometry"].coords[-1]), axis=1)

            out_layer = layername.replace("_split", "_bothdir")
            try:
                fiona.remove(self.geopackage_path, layer=out_layer)
            except Exception:
                pass

            trackdata.to_file(self.geopackage_path, driver="GPKG", layer=out_layer, mode="a")

    # Step 5: offsets/headings
    def distance_heading_calc(self, linkdata: LineString):
        headings = []
        distances = [0.0]
        g = None

        coords = list(linkdata.coords)
        if len(coords) < 2:
            headings = [0.0]
            offsets = [0.0]
        else:
            for i in range(len(coords) - 1):
                base = coords[i]
                nxt = coords[i + 1]
                g = self.geod.Inverse(base[1], base[0], nxt[1], nxt[0])
                headings.append(g["azi1"])
                distances.append(g["s12"])
            headings.append(headings[-1] if headings else 0.0)
            offsets = list(map(float, np.cumsum(distances)))

        smooth_headings = smooth_link_data(offsets, headings, unwrap_heading=True, apply_savgol=True)

        s = pd.Series()
        s["distances"] = list(map(float, distances))
        s["offsets"] = offsets
        s["headings"] = list(map(float, headings))
        s["smooth headings"] = smooth_headings
        return s

    def calc_offsets_headings(self):
        for layername in fiona.listlayers(self.geopackage_path):
            if "_bothdir" not in layername:
                continue

            print(f"calculating offsets and headings for {layername}")
            trackdata = gpd.read_file(self.geopackage_path, layer=layername)
            temp = trackdata.geometry.apply(lambda x: self.distance_heading_calc(x))
            trackdata = pd.concat([trackdata, temp], axis=1)
            self.delete_and_create_layer(layername.replace("_bothdir", "_offhead"), trackdata)


    # Step 6: DRAPE (AWS)  <-- window-read per link
    def _get_region_bounds4326(self) -> Tuple[float, float, float, float]:
        """
        Find the base region polygon layer (no suffix) and return bounds in EPSG:4326.
        Assumes region layers were created by input_geopackage_parsing().
        """
        base_layers = []
        for layername in fiona.listlayers(self.geopackage_path):
            if any(s in layername for s in ["_osm", "_switches", "_split", "_bothdir", "_offhead", "_draped", "_linked"]):
                continue
            base_layers.append(layername)

        if len(base_layers) == 0:
            raise RuntimeError("No base region layer found in geopackage.")

        # Usually only one region per build; if multiple, union bounds.
        b = None
        for ln in base_layers:
            g = gpd.read_file(self.geopackage_path, layer=ln)
            if g.crs is None:
                g = g.set_crs("EPSG:4326")
            if g.crs.to_epsg() != 4326:
                g = g.to_crs("EPSG:4326")
            tb = tuple(g.total_bounds)  # (minx, miny, maxx, maxy)
            if b is None:
                b = tb
            else:
                b = (min(b[0], tb[0]), min(b[1], tb[1]), max(b[2], tb[2]), max(b[3], tb[3]))
        return b

    def drape_geometry_aws(
        self,
        tile_urls: Optional[Dict[str, str]] = None,
        resample_length: float = 10.0,
        apply_savgol: bool = False,
    ):
        """
        - If tile_urls is None: auto-select needed tiles from region bounds.
        - Samples elevation by reading ONE window per link per tile (no full-tile loads).
        """
        bounds4326 = self._get_region_bounds4326()
        need_tiles = tiles_for_bounds(bounds4326)

        if tile_urls is None:
            tile_urls = {t: aws_cog_url(t) for t in need_tiles}
        else:
            # keep only tiles we actually need
            tile_urls = {t: tile_urls[t] for t in need_tiles if t in tile_urls}
            missing = [t for t in need_tiles if t not in tile_urls]
            if missing:
                # fallback: fill missing from default AWS URL pattern
                for t in missing:
                    tile_urls[t] = aws_cog_url(t)

        # Rasterio Env: encourage multirange requests for COG
        env_opts = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
            "GDAL_HTTP_MULTIRANGE": "YES",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        }

        with rasterio.Env(**env_opts):
            pool = AwsCogPool(tile_urls)

            for layername in fiona.listlayers(self.geopackage_path):
                if "_offhead" not in layername:
                    continue

                print(f"draping layer (AWS): {layername}")
                trackdata = gpd.read_file(self.geopackage_path, layer=layername)

                if trackdata.crs is None:
                    trackdata = trackdata.set_crs("EPSG:4326")
                if trackdata.crs.to_epsg() != 4326:
                    trackdata = trackdata.to_crs("EPSG:4326")

                elevations_all = []
                elevations_raw_all = []
                num_elev_segs = []

                for _, row in trackdata.iterrows():
                    offsets = ast.literal_eval(row.offsets) if isinstance(row.offsets, str) else row.offsets
                    offsets = np.asarray(offsets, dtype=float)
                    length = float(offsets[-1]) if len(offsets) else 0.0

                    if length <= 0.0:
                        elevations_all.append([0.0] * len(offsets))
                        elevations_raw_all.append([0.0])
                        num_elev_segs.append(1)
                        continue

                    nseg = max(int(np.ceil(length / resample_length)) + 1, 2)
                    num_elev_segs.append(int(nseg))
                    fracs = np.linspace(0.0, 1.0, nseg)

                    pts = shapely.line_interpolate_point(row.geometry, fracs, normalized=True)
                    lon = np.array([p.x for p in pts], dtype=float)
                    lat = np.array([p.y for p in pts], dtype=float)

                    elev = pool.sample_link_points(lon, lat, list(tile_urls.keys()))

                    # fill NaNs along the link
                    good = np.isfinite(elev)
                    if good.sum() >= 2:
                        elev = np.interp(np.arange(len(elev)), np.where(good)[0], elev[good]).astype(np.float32)
                    else:
                        elev[:] = 0.0

                    elevations_raw_all.append(list(map(float, elev)))

                    # interpolate back to original offsets
                    elev_on_offsets = np.interp(offsets, fracs * length, elev).astype(np.float32)

                    if apply_savgol:
                        elev_on_offsets = np.asarray(
                            smooth_link_data(
                                offsets,
                                elev_on_offsets,
                                window_length=200,
                                order=3,
                                segment_length=2,
                                unwrap_heading=False,
                                apply_savgol=True,
                            ),
                            dtype=float,
                        )

                    elevations_all.append(list(map(float, elev_on_offsets)))

                trackdata["elevations"] = elevations_all
                trackdata["elevations raw"] = elevations_raw_all
                trackdata["number of elevation segments"] = num_elev_segs

                self.delete_and_create_layer(layername.replace("_offhead", "_draped"), trackdata)

        print("AWS draping complete.")


    # Step 7: build links
    def build_links(self):
        for layername in fiona.listlayers(self.geopackage_path):
            if "_draped" not in layername:
                continue

            buffer_diameter = 1  # meters
            trackdata = gpd.read_file(self.geopackage_path, layer=layername)

            trackdata["yaml_idx"] = trackdata.index.values + 1

            trackdata["start coord"] = trackdata["start coord"].apply(lambda x: shapely.from_wkt(x))
            trackdata["end coord"] = trackdata["end coord"].apply(lambda x: shapely.from_wkt(x))

            trackdata["next_idx"] = 0
            trackdata["next_idx_alt"] = 0
            trackdata["prev_idx"] = 0
            trackdata["prev_idx_alt"] = 0

            def _get_head(x, first=True):
                if isinstance(x, str):
                    lst = ast.literal_eval(x)
                else:
                    lst = x
                return lst[0] if first else lst[-1]

            trackdata["start heading"] = trackdata["smooth headings"].apply(lambda x: _get_head(x, True))
            trackdata["end heading"] = trackdata["smooth headings"].apply(lambda x: _get_head(x, False))

            # endpoint buffers in meters, then back to 4326
            start_trackdata = gpd.GeoDataFrame(
                trackdata[["yaml_idx"]],
                geometry=trackdata["start coord"],
                crs="EPSG:4326",
            ).to_crs("ESRI:102009")
            start_trackdata.geometry = start_trackdata.buffer(buffer_diameter)
            start_trackdata = start_trackdata.to_crs("EPSG:4326")

            end_trackdata = gpd.GeoDataFrame(
                trackdata[["yaml_idx"]],
                geometry=trackdata["end coord"],
                crs="EPSG:4326",
            ).to_crs("ESRI:102009")
            end_trackdata.geometry = end_trackdata.buffer(buffer_diameter)
            end_trackdata = end_trackdata.to_crs("EPSG:4326")

            switches = (
                gpd.read_file(self.geopackage_path, layer=layername.replace("_draped", "_switches"))
                .to_crs("ESRI:102009")
                .buffer(2.5)
            ).to_crs("EPSG:4326")

            for _, row in trackdata.iterrows():
                # next links
                rowgdf = gpd.GeoDataFrame([row["yaml_idx"]], geometry=[row["end coord"]], crs="EPSG:4326")
                potential_next = gpd.sjoin(
                    start_trackdata[trackdata.covers(row.geometry) == False],
                    rowgdf,
                    how="inner",
                    rsuffix="_row",
                ).copy()
                potential_next = trackdata[trackdata.yaml_idx.isin(potential_next.yaml_idx)]
                switch_at_end = switches[switches.intersects(row["end coord"])]

                potential_next["heading difference"] = potential_next["start heading"].apply(
                    lambda x: heading_difference(x, row["end heading"])
                )
                potential_next = potential_next[potential_next["heading difference"] < 25.0].sort_values("heading difference")

                if potential_next.shape[0] == 1:
                    trackdata.loc[row.name, "next_idx"] = int(potential_next.yaml_idx.values[0])
                elif potential_next.shape[0] >= 2:
                    trackdata.loc[row.name, "next_idx"] = int(potential_next.yaml_idx.values[0])
                    if switch_at_end.shape[0] > 0:
                        trackdata.loc[row.name, "next_idx_alt"] = int(potential_next.yaml_idx.values[1])
                    else:
                        print(f"WARNING: multiple next links but no switch for uid {row.uid}, {row.direction}")

                # prev links
                rowgdf = gpd.GeoDataFrame([row["yaml_idx"]], geometry=[row["start coord"]], crs="EPSG:4326")
                potential_prev = gpd.sjoin(
                    end_trackdata[trackdata.covers(row.geometry) == False],
                    rowgdf,
                    how="inner",
                    rsuffix="_row",
                ).copy()
                potential_prev = trackdata[trackdata.yaml_idx.isin(potential_prev.yaml_idx)]
                switch_at_start = switches[switches.intersects(row["start coord"])]

                potential_prev["heading difference"] = potential_prev["end heading"].apply(
                    lambda x: heading_difference(x, row["start heading"])
                )
                potential_prev = potential_prev[potential_prev["heading difference"] < 25.0].sort_values("heading difference")

                if potential_prev.shape[0] == 1:
                    trackdata.loc[row.name, "prev_idx"] = int(potential_prev.yaml_idx.values[0])
                elif potential_prev.shape[0] >= 2:
                    trackdata.loc[row.name, "prev_idx"] = int(potential_prev.yaml_idx.values[0])
                    if switch_at_start.shape[0] > 0:
                        trackdata.loc[row.name, "prev_idx_alt"] = int(potential_prev.yaml_idx.values[1])
                    else:
                        print(f"WARNING: multiple prev links but no switch for uid {row.uid}, {row.direction}")

            self.delete_and_create_layer(layername.replace("_draped", "_linked"), trackdata)


    # Step 8: identify links
    def indentify_links(self):
        min_link_length = 1000
        max_link_dist = 1500

        locations = gpd.read_file(self.input_geopackage, layer=self.input_locations_layer_name).to_crs("ESRI:102009")

        for layername in fiona.listlayers(self.geopackage_path):
            if "_linked" not in layername:
                continue

            trackdata = gpd.read_file(self.geopackage_path, layer=layername).to_crs("ESRI:102009")
            long_links = trackdata[(trackdata.length >= min_link_length) & (trackdata.direction == "forward")]

            long_links = gpd.sjoin_nearest(
                long_links,
                locations,
                how="inner",
                max_distance=max_link_dist,
                distance_col="match dist [m]",
            )

            osm_id_mapping = {}
            self.delete_and_create_layer(
                layername.replace("_linked", "_locations"),
                locations[locations.Location.isin(long_links.Location.unique())].to_crs(epsg=4326),
            )

            for loc in long_links.Location.unique():
                loc_links = long_links[long_links.Location == loc].sort_values("match dist [m]")
                best = loc_links.iloc[0, :].copy()
                osm_id_mapping[best.uid] = best.Location

            final_location_mapping = []
            for key in osm_id_mapping.keys():
                fwd = trackdata[(trackdata.uid == key) & (trackdata.direction == "forward")]
                rev = trackdata[(trackdata.uid == key) & (trackdata.direction == "reverse")]

                final_location_mapping.append({
                    "Location ID": osm_id_mapping[key],
                    "Link Index": int(fwd.yaml_idx.values[0]),
                    "Offset (m)": 0,
                    "Is Front End": False,
                    "Grid Emissions Region": "MROWc",
                    "Electricity Price Region": "MN",
                    "Liquid Fuel Price Region": "MN",
                })
                final_location_mapping.append({
                    "Location ID": osm_id_mapping[key],
                    "Link Index": int(rev.yaml_idx.values[0]),
                    "Offset (m)": 0,
                    "Is Front End": False,
                    "Grid Emissions Region": "MROWc",
                    "Electricity Price Region": "MN",
                    "Liquid Fuel Price Region": "MN",
                })

            final_location_mapping = pd.DataFrame(final_location_mapping)

            out_dir = Path(self.data_folder / "Generated Networks" / layername.replace("_linked", ""))
            out_dir.mkdir(parents=True, exist_ok=True)
            final_location_mapping.to_csv(out_dir / "Network Locations.csv", index=False)

    def convert_to_yaml(self, scale_loc_links=True, desired_length_meters=3500):
        """
        Convert _linked layers directly to ALTRIOS YAML format.
        No dependency on _mileposts layer.
        """

        for layername in fiona.listlayers(self.geopackage_path):

            # 🔥 现在读取 _linked 层
            if "_linked" not in layername:
                continue

            print(f"Converting {layername} to YAML...")

            trackdata = gpd.read_file(self.geopackage_path, layer=layername)

            # Output folder
            network_output_dir = Path(
                self.data_folder
                / "Generated Networks"
                / layername.replace("_linked", "")
            )
            network_output_dir.mkdir(parents=True, exist_ok=True)

            # Load location links if exists
            location_csv = network_output_dir / "Network Locations.csv"
            if location_csv.exists():
                location_data = pd.read_csv(location_csv)
                links_to_scale = location_data["Link Index"].to_list()
            else:
                links_to_scale = []

            track_list = []

            # ALTRIOS dummy link 0
            track_list.append({
                "idx_curr": 0,
                "idx_flip": 0,
                "idx_next": 0,
                "idx_next_alt": 0,
                "idx_prev": 0,
                "idx_prev_alt": 0,
                "osm_id": 0,
                "length_meters": 0,
                "elevs": [],
                "headings": [],
                "speed_set": None,
                "cat_power_limits": [],
                "link_idxs_lockout": [],
            })

            for _, row in trackdata.iterrows():

                # ---- Reverse link ----
                reverse_link = trackdata[
                    (trackdata.covers(row.geometry))
                    & (trackdata.yaml_idx != row.yaml_idx)
                    ]

                if reverse_link.shape[0] != 1:
                    raise ValueError(
                        f"reverse link count was {reverse_link.shape[0]} for yaml_idx {row.yaml_idx}"
                    )

                # ---- Parse headings ----
                if isinstance(row["smooth headings"], str):
                    headings = ast.literal_eval(row["smooth headings"])
                else:
                    headings = row["smooth headings"]

                # ---- Parse offsets ----
                if isinstance(row.offsets, str):
                    offsets = ast.literal_eval(row.offsets)
                else:
                    offsets = row.offsets

                # ---- Parse elevations ----
                if isinstance(row.elevations, str):
                    elevations = ast.literal_eval(row.elevations.replace("nan", "-12345.0"))
                else:
                    elevations = row.elevations

                # ---- Optional scaling ----
                if (row.yaml_idx in links_to_scale) and (offsets[-1] < desired_length_meters):
                    multiplier = desired_length_meters / offsets[-1]
                    offsets = [x * multiplier for x in offsets]

                # ---- Geometry coords ----
                lats = []
                lons = []
                for coord in row.geometry.coords:
                    lons.append(coord[0])
                    lats.append(coord[1])

                link_elevs = []
                link_headings = []

                for i in range(len(offsets)):

                    # remove edge spikes (<500m rule)
                    if (i == 0) or (i == len(offsets) - 1) or (
                            (offsets[i] > 500) and ((offsets[-1] - offsets[i]) > 500)
                    ):

                        if elevations[i] != -12345.0:
                            link_elevs.append({
                                "offset_meters": float(offsets[i]),
                                "elev": float(elevations[i]),
                            })

                        link_headings.append({
                            "offset_meters": float(offsets[i]),
                            "lat": float(lats[i]),
                            "lon": float(lons[i]),
                            "heading_radians": float(
                                np.mod(headings[i] * np.pi / 180, 2 * np.pi)
                            ),
                        })

                link_dict = {
                    "idx_curr": int(row.yaml_idx),
                    "idx_flip": int(reverse_link.yaml_idx.values[0]),
                    "idx_next": int(row.next_idx) if pd.notna(row.next_idx) else 0,
                    "idx_next_alt": int(row.next_idx_alt) if pd.notna(row.next_idx_alt) else 0,
                    "idx_prev": int(row.prev_idx) if pd.notna(row.prev_idx) else 0,
                    "idx_prev_alt": int(row.prev_idx_alt) if pd.notna(row.prev_idx_alt) else 0,
                    "osm_id": int(row.osm_id) if ("osm_id" in row and pd.notna(row.osm_id)) else 0,
                    "uid": str(row.uid) if "uid" in row else "",
                    "length_meters": float(offsets[-1]),
                    "elevs": link_elevs,
                    "headings": link_headings,
                    "speed_set": None,
                    "cat_power_limits": [],
                    "link_idxs_lockout": [],
                }

                track_list.append(link_dict)

            network_dict = [
                {
                    "max_grade": 20.25,
                    "max_curv_radians_per_meter": 20.020,
                    "max_heading_step_radians": 20.24,
                    "max_elev_step_meters": 10.0,
                },
                track_list,
            ]

            with open(network_output_dir / "Network.pickle", "wb") as handle:
                pickle.dump(network_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(network_output_dir / "Network.yaml", "w") as f:
                f.write(
                    """---
    # Generated with AWS-based ALTRIOS NetworkBuilder
    # All coordinates are WGS84
    """
                )
                f.write(
                    yaml.dump(
                        network_dict,
                        sort_keys=False,
                        default_flow_style=False,
                        Dumper=NoAliasDumper,
                    )
                )

            print(f"YAML generated at: {network_output_dir}")

    def build_network(self, apply_savgol_elev: bool = False):
        self.input_geopackage_parsing()
        self.download_osm_data()
        self.clean_geometry()
        self.create_reverse_links()
        self.calc_offsets_headings()

        # AWS drape
        self.drape_geometry_aws(
            tile_urls=None,              # auto
            resample_length=10.0,
            apply_savgol=apply_savgol_elev,
        )

        self.build_links()
        self.indentify_links()
        self.convert_to_yaml()
        return True


if __name__ == "__main__":
    try:
        os.chdir(Path(__file__).parent)
    except Exception:
        pass

    builds = [
        {
            "input": Path(__file__).parents[2] / "Demo_Network/Demo_Network.gpkg",
            "output folder": Path(__file__).parents[2] / "SouthCentralTX",
            "name": "SouthCentralTX_NoSavgol50",
            "apply_savgol_elev": False,
        },
    ]

    for build in builds:
        print(f"Now processing {build['name']}.....")

        builder = NetworkBuilder(
            build["input"],
            build["output folder"],
            build["name"],
            "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_Rail_Mileposts/FeatureServer/0/query",
            {},
        )

        builder.build_network(apply_savgol_elev=build["apply_savgol_elev"])
        print("Done.")
