"""
Created on Wed Apr  2 10:19:07 2025

@author: ganderson
"""

# %%
# set gdal variables before the import
import ast
import concurrent.futures
import contextlib
import os
import pickle
import sys
import uuid
from pathlib import Path

# from typing import list
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import interpolate
from scipy.signal import savgol_filter
from tenacity import retry, stop_after_attempt, wait_fixed

env_folder_path = os.path.dirname(sys.executable)
if Path(env_folder_path + "/Library/share/gdal").exists():
    os.environ["GDAL_DATA"] = env_folder_path + "/Library/share/gdal"
    os.environ["GDAL_DRIVER_PATH"] = env_folder_path + "/Library/lib/gdalplugins"
    os.environ["GEOTIFF_CSV"] = env_folder_path + "/Library/share/epsg_csv"
    os.environ["PROJ_LIB"] = env_folder_path + "/Library/share/proj"
    os.environ["PROJ_NETWORK"] = "OFF"
else:
    os.environ["PROJ_LIB"] = os.path.dirname(env_folder_path) + "/share/proj"
    os.environ["PROJ_DATA"] = os.path.dirname(env_folder_path) + "/share/proj"
    import pyproj

    pyproj.datadir.set_data_dir(os.path.dirname(env_folder_path) + "/share/proj")

# disabled E402 on geospatial libraries because some paths need to be setup prior to loading them.


import fiona  # noqa: E402
import geopandas as gpd  # noqa: E402
import momepy  # noqa: E402
import overpy  # noqa: E402
import rasterio  # noqa: E402
import rasterio.mask  # noqa: E402
import seamless_3dep as s3dep  # noqa: E402
import shapely  # noqa: E402
from geographiclib.geodesic import Geodesic  # noqa: E402
from osgeo import gdal  # noqa: E402
from rasterio.plot import show  # noqa: E402


# retry 10 times and wait 3 minutes between attempts if it errors out
@retry(stop=stop_after_attempt(10), wait=wait_fixed(180_000))
def call_osm(layer_query):
    """
    Query the OpenStreetMap (OSM) Overpass API with the provided query string,
    using automatic retries if the server is busy.

    This function wraps the Overpass API call to allow for retry logic (e.g., with tenacity)
    when the OSM server is overloaded or temporarily unavailable.

    Args:
        LayerQuery (str): The Overpass QL query string to send to the OSM server.

    Returns
    -------
        overpy.Result: The result of the Overpass API query.

    Raises
    ------
        overpy.exception.OverpassException: If the query fails after the maximum number of retries.
    """
    # this is just a wrapper around overpass so that I can use tenacity to retry things with
    # the server is too busy
    print("calling osm server....")
    api = overpy.Overpass()
    api.default_max_retry_count = 5
    return api.query(layer_query)


def point_from_coord(coord):
    """
    Take the coordinate from a point in a linestring and returns
    a shapeply.Point.  This was needed for error handling where there were a
    couple nans.

    Parameters
    ----------
    coord : shapely coordinate.  Either a list or tuple that specifies x,y point

    Returns
    -------
    point : the coordinate returned as a shapely.Point

    """
    try:
        point = shapely.to_wkt(shapely.Point(coord[0], coord[1]))
    except:
        point = np.nan

    return point


def heading_difference(heading_a: np.float64, heading_b: np.float64):
    """
    Calculate the absolute difference in angle between two
    headings.  This function assumes units of degrees.

    Parameters
    ----------
    heading_a : np.float64
        heading of first line segment in degrees.
    heading_b : np.float64
        heading of second line segment in degrees

    Returns
    -------
    min_angle : TYPE
        absolute difference between two headings in degrees.

    """
    heading_a = np.mod(heading_a, 360.0)
    heading_b = np.mod(heading_b, 360.0)
    absolute_difference = np.abs(heading_a - heading_b)
    min_angle = np.min([absolute_difference, 360.0 - absolute_difference])
    return min_angle


def located_point_value_on_line(
    point_gdf: gpd.GeoDataFrame,
    link_line: shapely.LineString,
    cols: list[str],
    buffer_width: np.float64 = 25.0,
    local_crs: str = "ESRI:102009",
):
    """
    Locate points in a GeoDataFrame along a linestring that are within a
    specified distance of the line.  returns a list that contains the fractional
    offset location of the point along the line with the values in specified columns.

    Parameters
    ----------
    point_gdf : gpd.GeoDataFrame
        GeoDataFrome containing the points and the attribute data that needs to
        be located on the supplied line.

    link_line : shapely.LineString
        The line that is to be used to identify and locate the points.

    cols : List[str]
        Columns containing the data that is to be associated with point on the
        line.

    buffer_width : np.float64, optional
        DESCRIPTION. The default is 25.0.
     : TYPE
        This is the widith in meters used to create buffer along line.  Buffer
        is used to identify appropriate points.

    Returns
    -------
    offset_value_list : List[Dict]
        List of dictionarys contains the fractional offset and supplied column
        values that were lcoated on the line.

    """
    line_gdf = gpd.GeoDataFrame(
        [],
        geometry=[link_line],
        crs="EPSG:4326",
    )  # doing this to ensure a crs is associated to line
    line_gdf_buffer = line_gdf.to_crs(local_crs)  # reproject to get meters
    line_gdf_buffer.geometry = line_gdf_buffer.buffer(
        buffer_width,
        # create 25 m buffer around track.  Line ends flat so only points next to line not past end.
    )
    line_gdf_buffer = line_gdf_buffer.to_crs(
        "EPSG:4326",
    )  # got back to wgs 84 to get back to a common crs

    # get all points that fall within the buffer
    points_of_interest = point_gdf.sjoin(
        line_gdf_buffer,
        how="inner",
        predicate="intersects",
    )

    # calculate fraction of the distance along the line the point is.  Using fract here so that
    # the length can be found by using the ellipsoidal length calculated in a prior step.
    points_of_interest["line offset"] = points_of_interest.geometry.apply(
        lambda x: shapely.line_locate_point(line_gdf.geometry, x, normalized=True),
    )
    # selct all to points within 25 m of track, we will then project them onto
    # line using line_locate_point, these values will then go into speed_sets and
    # grade/cruve verification

    offset_value_list = []
    for idx, row in points_of_interest.iterrows():
        offset_value_dict = {"offset": row["line offset"]}
        for col in cols:
            offset_value_dict[col] = row[col]
        offset_value_list.append(offset_value_dict)
    return offset_value_list


def smooth_link_data(
    offsets,
    values,
    window_length=100,
    order=3,
    segment_length=2,
    unwrap_heading=False,
    interp_offsets=[],
    interp_values=[],
):
    """
    Smooth the input values along specified offsets using a Savitzky-Golay filter,
    optionally unwrapping headings and interpolating to evenly spaced segments.

    Parameters
    ----------
    offsets : array-like
        The positions along the line (e.g., distances) corresponding to the values.
    values : array-like
        The data values to be smoothed (e.g., elevations or headings).
    window_length : int, optional
        The length of the filter window (number of coefficients). Default is 100.
    order : int, optional
        The order of the polynomial used to fit the samples. Default is 3.
    segment_length : int, optional
        The length of each segment for interpolation. Default is 2.
    unwrap_heading : bool, optional
        Whether to unwrap heading values before smoothing. Default is False.
    interp_offsets : array-like, optional
        Offsets to use for interpolation. If empty, will be generated.
    interp_values : array-like, optional
        Values to use for interpolation. If empty, will be generated.

    Returns
    -------
    list
        The smoothed values corresponding to the input offsets.
    """
    # this is going to fit a spline to the elations with the ends constrained to
    # smooth everything out.

    if unwrap_heading:
        values = np.unwrap(values, period=360)

    if len(interp_offsets) == 0:
        # break track into even segments to let filter think we have evenly spaced data
        f = interpolate.interp1d(offsets, values)
        interp_offsets = np.arange(0, np.max(offsets), segment_length)
        interp_values = f(interp_offsets)

    # apply filter
    savgol_heading = savgol_filter(
        interp_values,
        np.min([len(interp_values), window_length]),
        np.min([len(interp_values) - 1, order]),
        mode="nearest",
    )

    # interpolate back to the original points and replace end points
    f2 = interpolate.interp1d(
        interp_offsets,
        savgol_heading,
        bounds_error=False,
        fill_value="extrapolate",
    )

    smooth_values = f2(offsets)
    smooth_values[0] = values[0]
    smooth_values[-1] = values[-1]

    # this catches a few short links that were returning nan values
    if np.isnan(smooth_values).any():
        print("WARNING: smoothing did not work for a link.")
        smooth_values = values

    return list(smooth_values)


def _drape_row(args):
    (
        idx,
        row,
        offsets,
        resample_fractions,
        length,
        buffer,
        circle_buffer_m,
        vrt_path_wstr,
        track_buffer_wkb,
        local_crs,
        dem_crs,  # dem crs = EPSG:4269
        apply_median,
    ) = args
    elevs = []
    try:
        with rasterio.open(vrt_path_wstr) as src:
            resampled_line = shapely.LineString(
                shapely.line_interpolate_point(
                    row.geometry,
                    resample_fractions,
                    normalized=True,
                ),
            )
            # reconstruct track_buffer from WKB
            track_buffer = shapely.from_wkb(track_buffer_wkb)
            for coord in resampled_line.coords:
                if apply_median:
                    circle_buffer = (
                        gpd.GeoSeries([shapely.Point(coord)], crs=dem_crs)
                        .to_crs(local_crs)
                        .buffer(circle_buffer_m)
                        .to_crs(dem_crs)
                    )
                    track_circle_intersect = circle_buffer.intersection(track_buffer)[0]
                    if type(track_circle_intersect) is shapely.Polygon:
                        track_circle_intersect = [track_circle_intersect]
                    else:
                        track_circle_intersect = track_circle_intersect.geoms
                    masked_elevation, transform = rasterio.mask.mask(
                        src,
                        gpd.GeoSeries(track_circle_intersect, crs=dem_crs),
                        crop=True,
                        all_touched=True,
                        nodata=np.nan,
                    )
                    elevs.append(np.nanmedian(masked_elevation))
                else:
                    elevs.append(src.sample(shapely.Point(coord)))

            if np.nanmedian(masked_elevation) == 0.0:
                fig, ax = plt.subplots(figsize=(10, 10))
                # show(src, ax=ax, transform=src.transform)
                show(masked_elevation, ax=ax, transform=transform)
                gpd.GeoSeries(track_circle_intersect, crs=dem_crs).plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="red",
                    linewidth=1.5,
                )
                plt.savefig(f"draping debug {idx}.png")
                plt.close()
                with open(f"draping debug {idx}.txt", "w") as file:
                    file.write(str(masked_elevation) + "\n")
                    file.write("-------------------------\n")
                    file.write(str(np.nanmedian(masked_elevation)) + "\n")

    except Exception:
        elevs = [np.nan] * len(resample_fractions)

    return idx, elevs


def parse_to_stats(link_series):
    """
    Parse the 'TO stats' field from a link series and return a DataFrame of statistics.

    Parameters
    ----------
    link_series : pd.Series
        A pandas Series containing a 'TO stats' field and an 'offsets' field.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed statistics with offsets scaled appropriately.
    """
    to_stat_list = ast.literal_eval(link_series["TO stats"].replace(": nan", ": ''"))

    if len(to_stat_list) > 0:
        max_offset = np.max(ast.literal_eval(link_series.offsets))
        to_stat_df = pd.DataFrame(to_stat_list)
        to_stat_df.offset = (
            to_stat_df.offset * max_offset
        )  # need to multiply by the length here because the function that places the TO point on
        # the line gives a fraction of the distance along the line.
        to_stat_df = to_stat_df.sort_values(by=["offset"])
    else:
        to_stat_df = pd.DataFrame()

    return to_stat_df


class NoAliasDumper(yaml.SafeDumper):
    """
    A custom YAML dumper that disables YAML anchors and aliases for cleaner output.

    This class is used for formatting YAML output without anchors and aliases,
    which can be useful for configuration files or outputs where readability is preferred.
    """

    # used for formatting yaml output.
    # https://ttl255.com/yaml-anchors-and-aliases-and-how-to-disable-them/
    def ignore_aliases(self, data):  # noqa: D102
        return True


class NetworkBuilder:
    """
    NetworkBuilder is responsible for constructing railway network data from geospatial sources,
    including parsing input geopackages, downloading and processing OSM and elevation data,
    cleaning and linking geometries, and exporting the resulting network in various formats.

    Attributes
    ----------
    input_geopackage : Path
        Path to the input geopackage defining regions and locations.
    data_folder : Path
        Directory for output data, intermediate files, and downloads.
    builder_name : str
        Name used for the output geopackage and network files.
    geopackage_path : Path
        Path to the output geopackage.
    local_crs : str
        Local coordinate reference system used for spatial operations.
    geod : Geodesic
        Geodesic object for distance and heading calculations.

    Methods
    -------
    delete_and_create_layer(layername, gdf)
        Deletes and recreates a layer in the geopackage.
    input_geopackage_parsing()
        Parses the input geopackage and creates region layers.
    download_elevation()
        Downloads elevation data for each region.
    download_osm_data()
        Downloads OSM railway data for each region.
    create_virtual_raster()
        Combines elevation data into virtual rasters.
    drape_geometry(...)
        Applies elevation data to track geometries.
    clean_geometry()
        Cleans and splits OSM track data at switches.
    create_reverse_links()
        Generates reverse-direction links for each track.
    distance_heading_calc(linkdata)
        Calculates distances and headings for a linestring.
    calc_offsets_headings()
        Calculates offsets and headings for all links.
    build_links()
        Links track segments based on spatial and heading criteria.
    convert_to_yaml(...)
        Exports the network to YAML and pickle formats.
    build_network(...)
        Runs the full network build process.
    indentify_links()
        Associates network links with specified locations.
    apply_speed_restrictions(link_yaml_idx, trackdata, speed_limit_mph)
        Applies speed restrictions to a link.
    """

    def __init__(
        self,
        input_geopackage_path: Path | str,
        data_folder: Path | str,
        builder_name,
        input_regions_layer_name="network_regions",
        input_locations_layer_name="network_locations",
        local_crs="ESRI:102009",
    ):
        """
        Initialize network builder object.

        Parameters
        ----------
        input_geopackage_path : TYPE
            This geopackage will define the regions and locations.
        data_folder : TYPE
            This the folder that will contain the output data, intermediate
            files, and other downloaded data.
        builder_name : TYPE
            This is the name that will be used for the geopackage that contains
            all the networks.
        input_regions_layer_name : String
            This is the name of the layer in the input geopackage that contain polygons that
            define the regions that need to be generated.  This layer must contain a field called
            "region_name" that will be used to name region.
        input_locations_layer_name : String
            This is the name of a point layer in the input geopackage that contains the network
            locations for all possible networks in the input geopackage.  This layer must contain
            a field called "Location".

        Returns
        -------
        None.

        """
        self.input_geopackage = Path(input_geopackage_path)
        self.input_regions_layer_name = input_regions_layer_name
        self.input_locations_layer_name = input_locations_layer_name

        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=False, exist_ok=True)

        self.builder_name = builder_name
        self.geopackage_path = Path(data_folder, builder_name + ".gpkg")
        self.milepost_csv_path = ""
        self.track_files = []
        self.switchresult = []
        self.geod = Geodesic.WGS84
        self.local_crs = local_crs

    def delete_and_create_layer(self, layername: str, gdf: gpd.GeoDataFrame):
        """
        Delete layer from geopackage before saving.  GeoPandas only
        allows replacing the entire geopackage or appending to the existing
        layer if it exists.  This uses the geopackage_path that is a property
        of the NetworkBuilder.

        Parameters
        ----------
        layername : str
            Desired name for GeoDataFrame being written to geopackage.
        gdf : gpd.GeoDataFrame
            GeoDataFrame that is to be written to geopackage.

        Returns
        -------
        None.

        """
        if self.geopackage_path.exists():
            try:  # this will fail if there are no layers in the geopackage
                if layername in fiona.listlayers(self.geopackage_path.resolve()):
                    fiona.remove(self.geopackage_path, layer=layername)
            except:
                pass

        gdf.to_file(self.geopackage_path, driver="GPKG", layer=layername, mode="a")

    def input_geopackage_parsing(self):
        """
        Read input geopackage that defines regions and locations.  A
        layer for each region will be created that will then be used for the later steps
        in the network generation process.

        Returns
        -------
        None.

        """
        # check if output geopackage exists.  If it does, we will delete existing
        # layers that need to be replaced
        if self.geopackage_path.exists():
            try:  # this fails with an empyt geopackage for some reason
                for layername in fiona.listlayers(self.geopackage_path.resolve()):
                    # purge previous all layers
                    fiona.remove(self.geopackage_path, layer=layername)
                    print(f"deleted: {layername}")
            except:
                pass

        # read the input geopackage region layer
        regions_gdf = gpd.read_file(
            self.input_geopackage,
            layer=self.input_regions_layer_name,
        )

        # iterate through region layer to make individual region layers in the output geopackage.
        # This may be a bit unnecessary, but this structure let's us use several different
        # file input types that may be proprietary.
        for idx, row in regions_gdf.iterrows():
            # if row.region_name == "Amarillo_FortWorth":
            single_region_gdf = gpd.GeoDataFrame(
                [{"region_name": row.region_name}],
                geometry=[row.geometry],
                crs="EPSG:4326",
            )
            self.delete_and_create_layer(row.region_name, single_region_gdf)

    def download_elevation(self):
        """
        Download all relevant GeoTiffs from the USGS that are
        part of the 3DEP dataset.  They are saved to the folder specified by the
        data_folder property.


        Returns
        -------
        None.

        """
        # TODO this needs to happen before the osm data is downloaded or filter out
        # all the other layers
        for layername in fiona.listlayers(self.geopackage_path):
            if not (
                "_osm" in layername
                or "_switches" in layername
                or "_split" in layername
                or "_offhead" in layername
                or "_bothdir" in layername
                or "_draped" in layername
                or "_linked" in layername
                or "_grouped" in layername
                or "_TOjoined" in layername
                or "_TOPoint" in layername
            ):
                print(f"downloading layer data for layer: {layername}")
                geolayer = gpd.read_file(self.geopackage_path, layer=layername)
                # get the bounding box for the layer and add a bit to it.
                bounds = tuple(geolayer.buffer(0.5).total_bounds)
                layer_tiff_dir = Path(self.data_folder / "Elevation Data" / layername)
                layer_tiff_dir.mkdir(parents=True, exist_ok=True)
                # Download DEM
                print(f"***************************{bounds}**************")
                s3dep.get_dem(bounds, layer_tiff_dir)

        print("Elevation download complete")

    def download_osm_data(self):
        """
        Download all rail dail from OpenStreetMap based on the
        extent of the TO track file.  It will delete previously created layers
        if they exist.  All track and switch data is parsed and converted to
        GeoDataFrames that are saved to the geopackage with _osm and _switches
        appended to the TO layer that was created by the parse_to_data function.
        The layers may be useful for debugging problemattic networks.

        Returns
        -------
        None.

        """
        base_query = """[out:xml] [timeout:999];
                        (
                            nwr[railway]({},{},{},{});
                        );
                        (._;>;);
                        out body;"""

        # api = overpy.Overpass()
        # api.default_max_retry_count = 5

        for layername in fiona.listlayers(self.geopackage_path):
            # purge previous osm layers
            if (
                "_osm" in layername
                or "_switches" in layername
                or "_split" in layername
                or "_offhead" in layername
                or "_bothdir" in layername
                or "_draped" in layername
                or "_linked" in layername
                or "_grouped" in layername
                or "_TOjoined" in layername
                or "_TOPoint" in layername
            ):
                fiona.remove(self.geopackage_path, layer=layername)
                print(f"deleted: {layername}")

        for layername in fiona.listlayers(self.geopackage_path):
            print(f"download osm data for {layername}")
            geolayer = gpd.read_file(self.geopackage_path, layer=layername)
            # geopandas and overpass api use different order for bounding boxes
            bounds = tuple(geolayer.total_bounds)
            layer_query = base_query.format(bounds[1], bounds[0], bounds[3], bounds[2])
            result = call_osm(layer_query)
            track_data = result.ways

            track_gdf = []
            all_switch_gdf = []
            for way in track_data:
                # TODO figure out how to grab osm_id
                coords = []
                node_tags = []
                for node in way.nodes:
                    coords.append([node.lon, node.lat])
                    node_tags.append(node.tags)
                    if "railway" in node.tags.keys():
                        if node.tags["railway"] == "switch":
                            switch_gdf = gpd.GeoDataFrame(
                                data=[node.tags],
                                geometry=[shapely.Point([node.lon, node.lat])],
                                crs="EPSG:4326",
                            )
                            switch_gdf["osm_link_id"] = way.id
                            all_switch_gdf.append(switch_gdf)

                way_gdf = gpd.GeoDataFrame(
                    data=[way.tags],
                    geometry=[shapely.LineString(coords)],
                    crs="EPSG:4326",
                )
                way_gdf["Node Tags"] = str(node_tags)
                way_gdf["osm_id"] = way.id
                track_gdf.append(way_gdf)
            for (
                result_node
            ) in (
                result.nodes
            ):  # Only look for junctions and add them to the list of switches
                if (
                    "railway" in result_node.tags.keys()
                    or "railway:switch" in result_node.tags.keys()
                ):
                    if "railway" not in result_node.tags.keys():
                        # this is going to fix the problems of railway not being in the tabs
                        # for the keys.  This could be better.....
                        result_node.tags["railway"] = ""

                    if (
                        result_node.tags["railway"] == "junction"
                        or "railway:switch" in result_node.tags.keys()
                    ):
                        switch_gdf = gpd.GeoDataFrame(
                            data=[{"railway": "junction"}],
                            geometry=[
                                shapely.Point([result_node.lon, result_node.lat]),
                            ],
                            crs="EPSG:4326",
                        )
                        switch_gdf["osm_link_id"] = result_node.id
                        all_switch_gdf.append(switch_gdf)
            track_gdf = pd.concat(track_gdf)

            if "fixme" in track_gdf.columns:
                track_gdf = track_gdf.drop("fixme", axis=1)

            all_switch_gdf = pd.concat(all_switch_gdf)

            # TODO add logic here to filter down to mainline only plus a buffer. use usage column
            # will also need to reproject to get buffer.

            track_gdf = track_gdf[track_gdf.railway != "abandoned"]
            track_gdf = track_gdf[track_gdf.railway != "construction"]
            track_gdf = track_gdf[track_gdf.railway != "defect_detector"]
            track_gdf = track_gdf[track_gdf.railway != "dismantled"]
            track_gdf = track_gdf[track_gdf.railway != "disused"]
            track_gdf = track_gdf[track_gdf.railway != "light_rail"]
            track_gdf = track_gdf[track_gdf.railway != "miniature"]
            track_gdf = track_gdf[track_gdf.railway != "monorail"]
            track_gdf = track_gdf[track_gdf.railway != "narrow_gauge"]
            track_gdf = track_gdf[track_gdf.railway != "platform"]
            track_gdf = track_gdf[track_gdf.railway != "proposed"]
            track_gdf = track_gdf[track_gdf.railway != "razed"]
            track_gdf = track_gdf[track_gdf.railway != "signal_box"]
            track_gdf = track_gdf[track_gdf.railway != "station"]
            track_gdf = track_gdf[track_gdf.railway != "tram"]
            track_gdf = track_gdf[track_gdf.railway != "traverser"]
            track_gdf = track_gdf[track_gdf.railway != "turntable"]
            track_gdf = track_gdf[track_gdf.railway != "yard"]

            track_gdf = track_gdf[track_gdf.usage != "military"]
            track_gdf = track_gdf[track_gdf.usage != "industrial"]
            track_gdf = track_gdf[track_gdf.usage != "tourism"]

            track_gdf = track_gdf[track_gdf.service != "construction"]
            track_gdf = track_gdf[track_gdf.service != "spur"]
            track_gdf = track_gdf[track_gdf.service != "yard"]

            if "building" in track_gdf.columns.values:
                track_gdf = track_gdf[track_gdf.building != "train_station"]

            if "Note" in track_gdf.columns.values:
                track_gdf = track_gdf.drop("Note", axis=1)

            track_gdf = track_gdf.clip(geolayer)

            track_gdf.to_file(
                self.geopackage_path,
                driver="GPKG",
                layer=layername + "_osm",
                mode="a",
            )

            all_switch_gdf.to_file(
                self.geopackage_path,
                driver="GPKG",
                layer=layername + "_switches",
                mode="a",
            )

            self.switchresult = track_data

    def create_virtual_raster(self):
        """
        Combine the previously downloaded elevation data by folder.
        The folder name is based upon the TO Track File name.  All geotiffs in
        each folder will be used to create a virtual raster with the same name.
        This file is save in the specified data_folder.

        Returns
        -------
        None.

        """
        for layername in fiona.listlayers(self.geopackage_path):
            print(f"creating virtual raster for {layername}")
            # geolayer = gpd.read_file(self.geopackage_path, layer=layername)
            vrt_name = self.data_folder / "Elevation Data" / (layername + ".vrt")
            layer_tiff_dir = self.data_folder / "Elevation Data" / layername

            tiff_files = layer_tiff_dir.glob("*.tiff")

            tiff_strs = [str(tiff_file) for tiff_file in tiff_files]

            gdal.BuildVRT(str(vrt_name), tiff_strs)
            # have not added filtering yet.

    def drape_geometry(
        self,
        resample_length=10,
        circle_buffer_m=250,
        buffer=2,
        dem_crs="EPSG:4269",
        apply_median=True,
        apply_savgol=True,
    ):
        """Parallelized draping operation."""
        for layername in fiona.listlayers(self.geopackage_path):
            if "_offhead" in layername:
                print(f"draping layer: {layername}")
                vrt_path = (
                    self.data_folder
                    / "Elevation Data"
                    / (layername.replace("_offhead", "") + ".vrt")
                )
                trackdata = gpd.read_file(self.geopackage_path, layer=layername)
                track_data = trackdata.to_crs(dem_crs)

                # create 2m track buffer
                track_buffer = track_data.to_crs(self.local_crs).buffer(buffer)
                track_buffer = (
                    gpd.GeoDataFrame([], geometry=track_buffer, crs=self.local_crs)
                    .dissolve()
                    .to_crs(dem_crs)
                ).geometry.iloc[0]
                track_buffer_wkb = track_buffer.wkb

                args_list = []
                for idx, row in track_data.iterrows():
                    offsets = ast.literal_eval(row.offsets)
                    length = offsets[-1]
                    number_of_segments = np.ceil(length / resample_length)
                    segment_fraction = 1 / number_of_segments
                    resample_fractions = np.arange(0, 1.00001, segment_fraction)
                    args_list.append(
                        (
                            idx,
                            row,
                            offsets,
                            resample_fractions,
                            length,
                            buffer,
                            circle_buffer_m,
                            str(vrt_path),
                            track_buffer_wkb,
                            self.local_crs,
                            dem_crs,
                            apply_median,
                        ),
                    )

                results = {}
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for idx, elevs in executor.map(_drape_row, args_list):
                        results[idx] = elevs

                line_elevations = []
                line_elevations_raw = []
                num_of_elev_segs = []
                for idx, row in track_data.iterrows():
                    offsets = ast.literal_eval(row.offsets)

                    # execute this block if savitsky golay filtering is desired.  This
                    # occurs by default.added this option to support writing paper with
                    # UT Austin to evaluate impact of different
                    # filtering operation in the network generation process.
                    if apply_savgol:
                        length = offsets[-1]
                        number_of_segments = np.ceil(length / resample_length)
                        num_of_elev_segs.append(int(number_of_segments))
                        segment_fraction = 1 / number_of_segments
                        resample_fractions = np.arange(0, 1.00001, segment_fraction)
                        elevs = results[idx]
                        line_elevations.append(
                            list(
                                map(
                                    float,
                                    smooth_link_data(
                                        offsets,
                                        elevs,
                                        window_length=200,
                                        interp_offsets=resample_fractions * length,
                                        interp_values=elevs,
                                    ),
                                ),
                            ),
                        )
                    else:
                        # throw in the raw values if we're not filtering.
                        line_elevations.append(list(map(float, elevs)))
                    line_elevations_raw.append(list(map(float, elevs)))

                trackdata["elevations"] = line_elevations
                trackdata["elevations raw"] = line_elevations_raw
                track_data["number of elevation segments"] = num_of_elev_segs

                self.delete_and_create_layer(
                    layername.replace("_offhead", "_draped"),
                    trackdata,
                )

    def clean_geometry(self):
        """
        Clean the OpenStreetMap data.  The main cleaning step is
        splitting rail links that are not split at switches.  It uses the
        switches found int he '_switches' layer to split all lines that are in
        the '_osm' layer.  If there are links that are not split in the final
        network, it is most likely due to the node at the link intersection
        not containing the proper data.  These nodes should have a 'railway:switch'
        property assigned to them.  This can easily be done on the OpenStreetMap
        website after registering.

        Returns
        -------
        None.

        """
        # this only splits tracks at the switches.  Not planning to do the equivalent of what
        # we were doing in QGIS becaues splitting at all line intersections creates a lof of other
        # problems.  Closing gaps shouldn't be much of a problem with the OSM data.  The joining
        # process uses a buffer anyways.
        for layername in fiona.listlayers(self.geopackage_path):
            if "_osm" in layername:
                trackdata = gpd.read_file(self.geopackage_path, layer=layername)

                switch_data = gpd.read_file(
                    self.geopackage_path,
                    layer=layername.replace("_osm", "_switches"),
                )
                print("beginning removal of false nodes")
                trackdata = momepy.remove_false_nodes(trackdata)

                print("removed false nodes")
                split_trackdata = []
                for idx, row in trackdata.iterrows():
                    # loop through each link of network
                    # https://geopandas.org/en/v0.14.1/docs/reference/api/geopandas.GeoSeries.intersects.html
                    intersecting_switches = switch_data.loc[
                        switch_data.geometry.intersects(row.geometry),
                        :,
                    ]

                    # looping through the line and switches so that we keep splitting segments if
                    # multiple switches on a single line segment
                    # initialize to one element list
                    geometry_to_split = [row.geometry]
                    for switch in intersecting_switches.geometry:
                        # loop through each switch that interects current link.

                        # temp = []
                        for link in geometry_to_split:
                            # look the geometry to split.  this will grow with each switch that
                            # splits it.
                            # Might have a link with many switches that need to be split.
                            # https://shapely.readthedocs.io/en/stable/manual.html#shapely.ops.split
                            split_result = shapely.ops.split(link, switch)

                            temp = list(split_result.geoms)  # double check this works.
                            # The three lines below got replaced by this.
                            # for line in split_result.geoms:
                            #     # have to loop through split result because it return a geometry collection.
                            #     temp.append(line)

                        geometry_to_split = temp

                    for link in geometry_to_split:
                        # build up the geodataframe again with all the other columns
                        # TODO figure otu how to get all the node metadata parsed out here.  The
                        # list of node metadata will be the length of the original line but the
                        # geometry will be split up and shorter.
                        row2 = row.drop("geometry")
                        split_trackdata.append(
                            gpd.GeoDataFrame(
                                data=[row2],
                                geometry=[link],
                                crs="EPSG:4326",
                            ),
                        )
                split_trackdata = pd.concat(split_trackdata)

                split_trackdata["uid"] = split_trackdata.geometry.apply(
                    lambda x: uuid.uuid1(),
                )

                try:
                    # try to delete layer but it may not exist.
                    fiona.remove(
                        self.geopackage_path,
                        layer=layername.replace("_osm", "_split"),
                    )
                except Exception:
                    pass

                split_trackdata.to_file(
                    self.geopackage_path,
                    driver="GPKG",
                    layer=layername.replace("_osm", "_split"),
                    mode="a",
                )

    def create_reverse_links(self):
        """
        Create the reverse links from the data downloaded from
        OpenStreetMap.  The osm data is assumed to be in the forward direction
        even though it may not actually be forward (or asscending) per the AAR
        standard.  This step needs to be conducted prior to calculating headings.
        This function simply uses the reverse method to change the order of the
        points in the LineSTring.  It also adds and end and start heading for
        each line that is used later in the linking process.

        Returns
        -------
        None.

        """
        # this will take all links and reverse the order of the points in the linestring
        # and call it a reverse link.
        for layername in fiona.listlayers(self.geopackage_path):
            if "_split" in layername:
                trackdata = gpd.read_file(self.geopackage_path, layer=layername)

                reverse_trackdata = trackdata.copy()
                reverse_trackdata.geometry = reverse_trackdata.geometry.reverse()

                reverse_trackdata["direction"] = "reverse"
                trackdata["direction"] = "forward"

                trackdata = pd.concat([trackdata, reverse_trackdata])
                trackdata["start coord"] = trackdata.apply(
                    lambda x: point_from_coord(x["geometry"].coords[0]),
                    axis=1,
                )
                trackdata["end coord"] = trackdata.apply(
                    lambda x: point_from_coord(x["geometry"].coords[-1]),
                    axis=1,
                )
                # trackdata['start point'] = trackdata.geometry.

                # TODO replace with new function
                try:
                    # try to delete layer but it may not exist.
                    fiona.remove(self.geopackage_path, layer=layername + "_bothdir")
                except Exception:
                    pass

                trackdata.to_file(
                    self.geopackage_path,
                    driver="GPKG",
                    layer=layername.replace("_split", "_bothdir"),
                    mode="a",
                )

    def distance_heading_calc(self, linkdata: shapely.LineString):
        """
        Calculate headings for each row.  This is called by
        calc_offsets_heading with the apply method in pandas.  The offset is
        calculated with the Karney equations in the geogeographiclib that
        assumes a spheroid for the lengths.  It returns a series of headings and
        offsets that will be placed in a cell in a GeoDataFrame.

        Parameters
        ----------
        linkdata : shapely.LineString
            The line to be broken into points to calculate headings and offsets.

        Returns
        -------
        result : pd.Series
            This is a series that contains the distances, offsets, and headings.
            These are each contained in a list.  The list is the length of the
            points in the line except for distnaces.  The distances are the
            distances between each point. It is one element shorter.  The offsets
            are the cumulatvie sum of the distances.  The headings are the same
            length of the number of points.  The heading is identical for the
            last two segments of the LineString because it required a reverse
            rather than a forward approach.
        """
        # https://geographiclib.sourceforge.io/Python/2.0/examples.html#basic-geodesic-calculations
        headings = []
        distances = [
            0,
        ]  # prepending 0 because 0 is the first offset.  Iterating n-1 times
        for point_idx in range(len(linkdata.coords) - 1):
            basepoint = linkdata.coords[point_idx]
            nextpoint = linkdata.coords[point_idx + 1]

            g = self.geod.Inverse(
                basepoint[1],
                basepoint[0],
                nextpoint[1],
                nextpoint[0],
            )
            headings.append(g["azi1"])  # degrees
            distances.append(g["s12"])  # meters

        headings.append(
            g["azi1"],
            # appending hear a last time because we need a heading going into the last point.
        )

        result = pd.Series()
        result["distances"] = distances
        result["offsets"] = list(
            map(float, list(np.cumsum(distances))),
        )  # convert back to list to keep format of data in cell consistent
        result["headings"] = headings
        result["smooth headings"] = list(
            map(float, smooth_link_data(result.offsets, headings, unwrap_heading=True)),
        )
        return result

    def calc_offsets_headings(self):
        """
        Calculate the headings and offsets for each linestring.  Call the
        distance_heading_calc method to do the actual calculation.  saves the
        layer to the geopackage_path with an _offhead appended to name.

        Returns
        -------
        None.

        """
        for layername in fiona.listlayers(self.geopackage_path):
            if "_bothdir" in layername:
                print(f"calculating offsets and headings for {layername}")
                trackdata = gpd.read_file(self.geopackage_path, layer=layername)
                temp = trackdata.geometry.apply(lambda x: self.distance_heading_calc(x))
                trackdata = pd.concat([trackdata, temp], axis=1)
                self.delete_and_create_layer(
                    layername.replace("_bothdir", "_offhead"),
                    trackdata,
                )

    def build_links(self):
        """
        Link the network.  The output is saved with
        _linked appended to the name.  The output of this will be used by the
        yaml conversion function.

        Returns
        -------
        None.

        """
        for layername in fiona.listlayers(self.geopackage_path):
            buffer_diameter = 1

            if "_draped" in layername:

                trackdata = gpd.read_file(self.geopackage_path, layer=layername)
                # caluclate the index that this link will be in the yaml file.
                # have to account for the dummy 0 link.
                trackdata["yaml_idx"] = trackdata.index.values + 1

                # convert stand and end coordinates to shapely objects from
                # WKT strings.
                trackdata["start coord"] = trackdata["start coord"].apply(
                    lambda x: shapely.from_wkt(x),
                )
                trackdata["end coord"] = trackdata["end coord"].apply(
                    lambda x: shapely.from_wkt(x),
                )

                # initialize all the linking columns for the yaml file to 0
                trackdata["next_idx"] = 0
                trackdata["next_idx_alt"] = 0
                trackdata["prev_idx"] = 0
                trackdata["prev_idx_alt"] = 0

                # convert heading list that is a string to a list and grab first
                # or last element.  The could be more efficent with one conversion
                # but it works.
                trackdata["start heading"] = trackdata["smooth headings"].apply(
                    lambda x: ast.literal_eval(x)[0],
                )
                trackdata["end heading"] = trackdata["smooth headings"].apply(
                    lambda x: ast.literal_eval(x)[-1],
                )

                # create buffers for the start and end points of each link.
                # buffers get created in units of meters.
                start_trackdata = gpd.GeoDataFrame(
                    trackdata[["yaml_idx"]],
                    geometry=trackdata["start coord"],
                    crs="EPSG:4326",
                ).to_crs(self.local_crs)
                start_buffer = start_trackdata.buffer(buffer_diameter)
                start_trackdata.geometry = start_buffer
                start_trackdata = start_trackdata.to_crs("EPSG:4326")

                end_trackdata = gpd.GeoDataFrame(
                    trackdata[["yaml_idx"]],
                    geometry=trackdata["end coord"],
                    crs="EPSG:4326",
                ).to_crs(self.local_crs)
                end_buffer = end_trackdata.buffer(buffer_diameter)
                end_trackdata.geometry = end_buffer
                end_trackdata = end_trackdata.to_crs("EPSG:4326")

                # read switch layer for track file, reprojec to CRS with meters for units, and
                # create a 2.5 m buffer to give some robustness to switch linking algorithm
                switches = (
                    gpd.read_file(
                        self.geopackage_path,
                        layer=layername.replace("_draped", "_switches"),
                    )
                    .to_crs(self.local_crs)
                    .buffer(2.5)
                ).to_crs("EPSG:4326")
                track_map = start_trackdata.explore()
                track_map.save("test.html")
                for idx, row in trackdata.iterrows():
                    # print('--------------{}, {}------------------'.format(idx, row.osm_id))
                    # ------------------------- next link business-----------------------
                    # grab links that has a starting point coincident with the end of the
                    # current link and don't have the same osm_id (eliminate the reverse
                    # direction for the current link)'

                    # THIS BIT IS THE OLD WAY I DID IT. IT HAD PROBLEMS WITH SPLIT LINKS
                    # potential_prev_links = trackdata[end_trackdata.geometry.intersects(
                    #     row["start coord"]) & (row.osm_id != trackdata.osm_id)].copy()

                    # create the rowgdf with the start coordinate to perform a splatial join
                    # with the end of the the other links.  The covers grabs all links that
                    # are not the opposite direction link of the current link.  Cannot use
                    # osm_id because the splitting at switches retains that same osm_id
                    rowgdf = gpd.GeoDataFrame(
                        [row["yaml_idx"]],
                        geometry=[row["end coord"]],
                        crs="EPSG:4326",
                    )
                    potential_next_links = gpd.sjoin(
                        start_trackdata[not trackdata.covers(row.geometry)],
                        rowgdf,
                        how="inner",
                        rsuffix="_row",
                    ).copy()

                    potential_next_links = trackdata[
                        trackdata.yaml_idx.isin(potential_next_links.yaml_idx)
                    ]

                    # grab switches at end of link.  May be multiple points that represent
                    # a single switch because it was part of several links
                    switch_at_end_of_link = switches[
                        switches.intersects(row["end coord"])
                    ]

                    # calculate the difference in headings between the end of the current
                    # link and start of the next links.  Take absolute value because I don't
                    # care about the direction.  Just the magnitude.
                    potential_next_links["heading difference"] = potential_next_links[
                        "start heading"
                    ].apply(lambda x: heading_difference(x, row["end heading"]))

                    # get rid of links that are point in direction that are much different.
                    # This is to get rid of links that are on switches that you can't access
                    # for the current direction.
                    potential_next_links = potential_next_links[
                        potential_next_links["heading difference"] < 25.0
                    ]

                    # sort the links by closest heading to figure out next vs next alt.
                    potential_next_links = potential_next_links.sort_values(
                        "heading difference",
                    )

                    # if only one link, there is no next alt
                    if potential_next_links.shape[0] == 1:
                        trackdata.loc[row.name, "next_idx"] = (
                            potential_next_links.yaml_idx.values[0]
                        )

                    # if there are 2 or more links, we need to check for a switch
                    elif potential_next_links.shape[0] >= 2:

                        # if switch exists, we can link for an alt leg too
                        if switch_at_end_of_link.shape[0] > 0:

                            # the 0 element has the closest heading so we link it as next
                            trackdata.loc[row.name, "next_idx"] = (
                                potential_next_links.yaml_idx.values[0]
                            )

                            # the next closest heading gets to be the next alt link
                            trackdata.loc[row.name, "next_idx_alt"] = (
                                potential_next_links.yaml_idx.values[1]
                            )

                            # it there are lot of extra links we throw a warning to indicate
                            # geometry problems that should probably be checked
                            if potential_next_links.shape[0] > 2:
                                print(
                                    f"""WARNING: too many possible links for osm_id {row.uid},
                                    {row.direction} at end of link""",
                                )

                        # if there is no switch, we only link next and generate a warning.
                        # there could be places where a switch does exist, but the tracks
                        # don't connect.  Make the user aware so they can investigate.
                        else:

                            trackdata.loc[row.name, "next_idx"] = (
                                potential_next_links.yaml_idx.values[0]
                            )

                            print(
                                f""""WARNING: multiple links but no switch for osm_id {row.uid},
                                {row.direction} at end of link.  Only linking link with
                                closest heading""",
                            )

                    # ----------------prev link business-------------------------------

                    # grab links that has an end point coincident with the start of the
                    # current link and don't have the same osm_id (eliminate the reverse
                    # direction for the current link)'

                    # THIS BIT IS THE OLD WAY I DID IT. IT HAD PROBLEMS WITH SPLIT LINKS
                    # potential_prev_links = trackdata[end_trackdata.geometry.intersects(
                    #     row["start coord"]) & (row.osm_id != trackdata.osm_id)].copy()

                    # create the rowgdf with the start coordinate to perform a splatial join
                    # with the end of the the other links.  The covers grabs all links that
                    # are not the opposite direction link of the current link.  Cannot use
                    # osm_id because the splitting at switches retains that same osm_id
                    rowgdf = gpd.GeoDataFrame(
                        [row["yaml_idx"]],
                        geometry=[row["start coord"]],
                        crs="EPSG:4326",
                    )
                    potential_prev_links = gpd.sjoin(
                        end_trackdata[not trackdata.covers(row.geometry)],
                        rowgdf,
                        how="inner",
                        rsuffix="_row",
                    ).copy()

                    potential_prev_links = trackdata[
                        trackdata.yaml_idx.isin(potential_prev_links.yaml_idx)
                    ]
                    # grab switches at start of link.  May be multiple points that represent
                    # a single switch because it was part of several links
                    switch_at_start_of_link = switches[
                        switches.intersects(row["start coord"])
                    ]

                    # calculate the difference in headings between the start of the current
                    # link and end of the previous links.  Take absolute value because I don't
                    # care about the direction.  Just the magnitude.
                    potential_prev_links["heading difference"] = potential_prev_links[
                        "end heading"
                    ].apply(lambda x: heading_difference(x, row["start heading"]))

                    # get rid of links that are pointed in direction that are much different.
                    # This is to get rid of links that are on switches that you can't access
                    # for the current direction.
                    potential_prev_links = potential_prev_links[
                        potential_prev_links["heading difference"] < 25.0
                    ]

                    # sort the links by closest heading to figure out prev vs prev alt.
                    potential_prev_links = potential_prev_links.sort_values(
                        "heading difference",
                    )

                    # if only one link, there is no prev alt
                    if potential_prev_links.shape[0] == 1:
                        trackdata.loc[row.name, "prev_idx"] = (
                            potential_prev_links.yaml_idx.values[0]
                        )

                    # if there are 2 or more links, we need to check for a switch
                    elif potential_prev_links.shape[0] >= 2:

                        # if switch exists, we can link for an alt leg too
                        if switch_at_start_of_link.shape[0] > 0:

                            # the 0 element has the closest heading so we link it as next
                            trackdata.loc[row.name, "prev_idx"] = (
                                potential_prev_links.yaml_idx.values[0]
                            )

                            # the next closest heading gets to be the next alt link
                            trackdata.loc[row.name, "prev_idx_alt"] = (
                                potential_prev_links.yaml_idx.values[1]
                            )

                            # it there are lot of extra links we throw a warning to indicate
                            # geometry problems that should probably be checked
                            if potential_prev_links.shape[0] > 2:
                                print(
                                    f"""WARNING: too many possible links for osm_id {row.uid},
                                    {row.direction} at start of link""",
                                )

                        # if there is no switch, we only link next and generate a warning.
                        # there could be places where a switch does exist, but the tracks
                        # don't connect.  Make the user aware so they can investigate.
                        else:

                            trackdata.loc[row.name, "prev_idx"] = (
                                potential_prev_links.yaml_idx.values[0]
                            )

                            print(
                                f"""WARNING: multiple links but no switch for osm_id {row.uid},
                                {row.direction} at start of link.  Only linking link with
                                closest heading""",
                            )

                self.delete_and_create_layer(
                    layername.replace("_draped", "_linked"),
                    trackdata,
                )

    def convert_to_yaml(
        self,
        scale_loc_links=True,
        desired_length_meters=3500,
        speed_limit_mph=60.0,
    ):
        """
        Convert "_linked" layers to a yaml file.  The output is saved to
        the 'Generated Networks folder' contained in the 'data_folder'.

        scale_loc_links: This will scale the location links if set to True

        desired_length_meters:  This is the desired length in meters so that the offsets
            can be scaled to provide a link that is long enough for long trains.


        Raises
        ------
        ValueError
            Raises error is the link in the opposite direction is not found in
            dataset.

        Returns
        -------
        None.

        """
        for layername in fiona.listlayers(self.geopackage_path):
            buffer_diameter = 1

            if "_linked" in layername:
                trackdata = gpd.read_file(self.geopackage_path, layer=layername)

                location_data = pd.read_csv(
                    str(self.data_folder)
                    + "/Generated Networks/"
                    + layername.replace("_linked", "/Network Locations.csv"),
                )
                links_to_scale = location_data["Link Index"].to_list()
                track_list = []

                # populate link 0 that ALTRIOS requires
                link_dict = {
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
                }
                track_list.append(link_dict)
                for idx, row in trackdata.iterrows():

                    # grab the reverse link and check to make sure there is
                    # only a single reverse link.
                    reverse_link = trackdata[
                        (trackdata.covers(row.geometry))
                        & (trackdata.yaml_idx != row.yaml_idx)
                    ]
                    if reverse_link.shape[0] != 1:
                        raise ValueError(
                            f"""reverse link count was {reverse_link.shape[0]} for yaml_idx
                            {row.yaml_idx}.  This count should always be 1""",
                        )

                    # parse out all the headings and offsets that were lists
                    # that got stored as stings.
                    headings = ast.literal_eval(row["smooth headings"])
                    offsets = ast.literal_eval(row.offsets)

                    if (row.yaml_idx in links_to_scale) & (
                        offsets[-1] < desired_length_meters
                    ):
                        multiplier = desired_length_meters / offsets[-1]
                        offsets = [x * multiplier for x in offsets]

                    try:
                        elevations = ast.literal_eval(row.elevations)
                    except Exception:
                        # placed this try statement here because sometime elevation data is not
                        # available.nans get placed in the draping operation.  This is to make the
                        # code work.  The Example I came across was track in mexico that we don't
                        # care about yet. The currentwork is focused within the US.  Long term it
                        # may be worth falling back to SRTM data maybe.
                        # This needs to be solved, but I don't have an immediate solution.
                        elevations = [0.0] * len(offsets)

                    lats = []
                    lons = []
                    for coord in row.geometry.coords:
                        lons.append(coord[0])
                        lats.append(coord[1])

                    link_elevs = []
                    link_headings = []
                    link_speed_sets = self.apply_speed_restrictions(
                        row.yaml_idx,
                        trackdata,
                        speed_limit_mph=speed_limit_mph,
                    )

                    for idx in range(len(offsets)):

                        if (idx == 0) or (idx == len(offsets) - 1):
                            link_elevs.append(
                                {
                                    "offset_meters": offsets[idx],
                                    "elev": elevations[idx],
                                },
                            )

                            link_headings.append(
                                {
                                    "offset_meters": offsets[idx],
                                    "lat": lats[idx],
                                    "lon": lons[idx],
                                    "heading_radians": float(
                                        np.mod(headings[idx] * np.pi / 180, 2 * np.pi),
                                    ),
                                },
                            )
                        elif (offsets[idx] > 500) and (
                            (offsets[-1] - offsets[idx]) > 500
                        ):
                            link_elevs.append(
                                {
                                    "offset_meters": offsets[idx],
                                    "elev": elevations[idx],
                                },
                            )

                            link_headings.append(
                                {
                                    "offset_meters": offsets[idx],
                                    "lat": lats[idx],
                                    "lon": lons[idx],
                                    "heading_radians": float(
                                        np.mod(headings[idx] * np.pi / 180, 2 * np.pi),
                                    ),
                                },
                            )

                    link_dict = {
                        "idx_curr": row.yaml_idx,
                        "idx_flip": int(reverse_link.yaml_idx.values[0]),
                        "idx_next": row.next_idx,
                        "idx_next_alt": row.next_idx_alt,
                        "idx_prev": row.prev_idx,
                        "idx_prev_alt": row.prev_idx_alt,
                        "osm_id": row.osm_id,
                        "uid": row.uid,
                        "length_meters": offsets[-1],
                        "elevs": link_elevs,
                        "headings": link_headings,
                        "speed_set": link_speed_sets,
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

                network_output_dir = Path(
                    self.data_folder
                    / "Generated Networks"
                    / layername.replace("_linked", ""),
                )
                network_output_dir.mkdir(parents=True, exist_ok=True)
                print(network_output_dir)
                with open(network_output_dir / "Network.pickle", "wb") as handle:
                    pickle.dump(network_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(network_output_dir / "Network.yaml", "w") as f:

                    f.write(
                        """---
  # Generated with Wabtec ALTRIOS network builder.
  # All Lat, Lon coordinates are WGS84
  # TODO: find and fix grade spikes then dial this back to 0.06
  # TODO: fix heading change spikes and then lower this to 0.009 (~15 deg per 100 ft)
  # TODO: fix heading change spikes and then lower this to 0.12 (~7 degrees)\n""",
                    )
                    f.write(
                        yaml.dump(
                            network_dict,
                            sort_keys=False,
                            default_flow_style=False,
                            Dumper=NoAliasDumper,
                        ),
                    )

    def build_network(self, force_dem_download=False, speed_limit_mph=60.0):
        """
        Run the full build process to create networks from the Trip
        Optimizer track files.

        Parameters
        ----------
        force_dem_download : TYPE, optional
            Set to true to force DEMs to be redownloaded. May be necessary if
            the download fails.  This will take a long time to complete.
            The default is False.

        Returns
        -------
        build_complete : TYPE
            returns True is it complete.  This is not very useful and should be
            fleshed out in the future.

        """
        build_complete = False
        # TODO replace MyBuilder with self
        self.input_geopackage_parsing()
        self.download_osm_data()
        self.clean_geometry()
        self.create_reverse_links()
        self.calc_offsets_headings()

        if force_dem_download:
            self.download_elevation()
            self.create_virtual_raster()

        try:
            self.drape_geometry()
        except Exception:
            print("downloading elevation for networks. This may take a while")
            self.download_elevation()
            self.create_virtual_raster()
            self.drape_geometry()

        self.build_links()
        self.indentify_links()
        self.convert_to_yaml(speed_limit_mph=speed_limit_mph)

        build_complete = True

        # TODO provide more diagnostics for what files were created so user can
        # understand what was done

        return build_complete

    def indentify_links(self):
        """
        Line up individual links within the network to locations
        that are needed to specify train routes.  The locations are specified
        in a csv file that has the columns "Location", "Lat", and "Lon".  The
        file location is specified when the Building is initialized.  The
        resulting file is named "Network Locations.csv" and is placed in the
        folder with the network.

        Returns
        -------
        None.

        """
        min_link_length = 1000  # this is the minimum link length for the location
        # maximum distance in meters link can be from coordinate specified for location
        max_link_distance_from_coord = 1500

        locations = gpd.read_file(
            self.input_geopackage,
            layer=self.input_locations_layer_name,
        ).to_crs(self.local_crs)

        for layername in fiona.listlayers(self.geopackage_path):
            if "_linked" in layername:
                trackdata = gpd.read_file(self.geopackage_path, layer=layername).to_crs(
                    self.local_crs,
                )

                # only grabbing forward link.  Will grab reverse based upon osm_id.
                # convert crs to get units of meters
                long_links = trackdata[
                    (trackdata.length >= min_link_length)
                    & (trackdata.direction == "forward")
                ]

                long_links = gpd.sjoin_nearest(
                    long_links,
                    locations,
                    how="inner",
                    max_distance=max_link_distance_from_coord,
                    distance_col="match dist [m]",
                )

                long_links["length"] = long_links.length
                osm_id_mapping = {}

                self.delete_and_create_layer(
                    layername.replace("_linked", "_locations"),
                    locations[
                        locations.Location.isin(long_links.Location.unique())
                    ].to_crs(epsg=4326),
                )

                for long_link in long_links.Location.unique():
                    loc_links = long_links[
                        long_links.Location == long_link
                    ].sort_values(
                        "match dist [m]",  # "length"
                    )
                    longest_loc_link = loc_links.iloc[0, :].copy()
                    osm_id_mapping[longest_loc_link.uid] = longest_loc_link.Location

                final_location_mapping = []
                for key in osm_id_mapping.keys():
                    fwd = trackdata[
                        (trackdata.uid == key) & (trackdata.direction == "forward")
                    ]
                    rev = trackdata[
                        (trackdata.uid == key) & (trackdata.direction == "reverse")
                    ]

                    final_location_mapping.append(
                        {
                            "Location ID": osm_id_mapping[key],
                            "Link Index": fwd.yaml_idx.values[0],
                            "Offset (m)": 0,
                            "Is Front End": False,
                            "Grid Emissions Region": "MROWc",
                            "Electricity Price Region": "MN",
                            "Liquid Fuel Price Region": "MN",
                        },
                    )
                    final_location_mapping.append(
                        {
                            "Location ID": osm_id_mapping[key],
                            "Link Index": rev.yaml_idx.values[0],
                            "Offset (m)": 0,
                            "Is Front End": False,
                            "Grid Emissions Region": "MROWc",
                            "Electricity Price Region": "MN",
                            "Liquid Fuel Price Region": "MN",
                        },
                    )

                final_location_mapping = pd.DataFrame(final_location_mapping)

                network_output_dir = Path(
                    self.data_folder
                    / "Generated Networks"
                    / layername.replace("_linked", "")
                )
                network_output_dir.mkdir(parents=True, exist_ok=True)

                final_location_mapping.to_csv(
                    network_output_dir / "Network Locations.csv",
                    index=False,
                )

    def apply_speed_restrictions(self, link_yaml_idx, trackdata, speed_limit_mph=60.0):
        """
        Apply speed restrictions to a specific link in the network.

        Parameters
        ----------
        link_yaml_idx : int
            The YAML index of the link to which speed restrictions will be applied.
        trackdata : pd.DataFrame or gpd.GeoDataFrame
            The DataFrame containing track information.
        speed_limit_mph : float, optional
            The speed limit in miles per hour to apply to the link (default is 60.0).

        Returns
        -------
        dict
            A dictionary containing speed restriction information for the link.
        """
        # extract the link we are interested
        link_data = trackdata[trackdata.yaml_idx == link_yaml_idx]
        # calc current link length here because it could be used in a couple different places below.
        link_length = np.max(ast.literal_eval(link_data.offsets.values[0]))
        # TODO figure out way to apply better speed restrictions from OSM data
        speed_restict_dict = {
            "speed_limits": [],
            "speed_params": [],
            "is_head_end": False,
        }

        speed_restict_dict["mp_dir"] = "unknown"
        speed_restict_dict["speed_limits"].append(
            {
                "offset_start_meters": 0,
                "offset_end_meters": float(link_length),
                "speed_meters_per_second": speed_limit_mph / 2.23693629,
            }
        )

        return speed_restict_dict


if __name__ == "__main__":

    # set current working directory to path of this script.
    # have try statement because spyder throws an error.  VS code needs it.
    with contextlib.suppress(Exception):
        os.chdir(Path(__file__).parent)

    MyBuilder = NetworkBuilder(
        # alt.resources_root() / "networks/NetworkInput_small.gpkg",
        # "resources/networks/NetworkInput_small.gpkg",
        "/home/garrett/Documents/ALTRIOS_Extras/Tomas Networks/NetworkInput_small.gpkg",
        "/home/garrett/Documents/ALTRIOS_Extras/Tomas Networks/Network Builder Test Small Ouput",
        "SmallNetworkBuild",
    )

    MyBuilder.build_network()

    MyBuilder.input_geopackage_parsing()
    MyBuilder.download_osm_data()
    MyBuilder.clean_geometry()
    MyBuilder.create_reverse_links()
    MyBuilder.calc_offsets_headings()
    MyBuilder.drape_geometry()
    MyBuilder.build_links()
    MyBuilder.indentify_links()
    MyBuilder.convert_to_yaml(speed_limit_mph=70)

    # MyBuilder = NetworkBuilder(
    #     "/home/garrett/Documents/ALTRIOS_Extras/Tomas Networks/NetworkInput_small (clovis only).gpkg",  # noqa: E501
    #     "Clovis Only Build",
    #     "ClovisBuilder",
    # )

    # MyBuilder.input_geopackage_parsing()

    # # MyBuilder.download_elevation()
    # MyBuilder.create_virtual_raster()
    # MyBuilder.build_network(
    #     speed_limit_mph=70.0,
    # )
