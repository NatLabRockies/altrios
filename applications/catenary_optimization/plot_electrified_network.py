"""Publication-quality maps of electrified (catenary) track over a lat/lon network.

Given an ALTRIOS network whose link ``headings`` carry ``lat``/``lon`` (as the
Spokane-Lethbridge corridor does) plus an energization-placement result (the
per-link energized sub-intervals + built substations), this renders a static map
with the electrified track drawn over the full corridor and the feeding
substations marked.

The intended use is NOT interactivity: it produces manuscript-ready figures. The
stack is **GeoPandas + Cartopy + Matplotlib**:

- The corridor can span ~800 miles across several degrees of latitude, where a
  Web Mercator basemap would inflate north-south scale by ``1/cos(lat)`` (~1.5x
  near 49 deg N). Instead the map is drawn in a **Lambert Conformal Conic**
  projection (conformal, standard parallels fit to the data) -- the conventional
  choice for mid-latitude regional maps -- so shapes and local scale are true.
- Track geometry and substations are carried as GeoPandas ``GeoDataFrame``\\s in
  EPSG:4326 and reprojected (via pyproj) into the map CRS.
- Cartopy supplies the projected axes, a lat/lon graticule, and the basemap: a
  **topographic tile layer** (Esri World Topo / OpenTopoMap, reprojected into the
  map CRS) or, offline, Natural Earth vector features.
- Matplotlib writes true vector output (``.pdf`` / ``.svg``) or high-DPI raster
  (``.png``) with no headless browser (unlike Plotly/Folium static export).

**Category coloring.** The network geometry itself is drawn as a single line
whose color varies by segment: gray where track is out of feeding range, blue
where it is within track-distance range of a built/existing substation but not
electrified, and orange where it is electrified. "Within range" uses the same
along-corridor TRACK distance as the optimizer and line-loss model (not crow-
flies). The scale bar and legend report distances in **miles**.

Requires the optional ``mapping`` extra: ``pip install altrios[mapping]``
(cartopy, geopandas, pyproj).

Primary entry point -- :func:`plot_electrified_track` -- accepts the objects the
``energization_optimization_corridor`` workflow already has in hand:

    from applications.catenary_optimization.plot_electrified_network import plot_electrified_track

    plan = rule_results["baseline"]              # EnergizationPlacementResult
    plot_electrified_track(
        network,                                 # alt.Network (WGS84 headings)
        result=plan,                             # -> energized intervals + built subs
        substations=substations,                 # list[cp.Substation] for markers
        out_path="figures/baseline_catenary.pdf",
        title="Optimized catenary -- Spokane to Lethbridge (baseline)",
    )

Run standalone for a smoke test (draws the bare corridor):

    python applications/catenary_optimization/plot_electrified_network.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless, file-only backend (no GUI needed)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_M_PER_MILE = 1609.344

# Topographic tile sources (no API key). ``{z}/{x}/{y}`` order noted per source.
_TILE_SOURCES: Dict[str, Tuple[str, str, str]] = {
    # name: (url template, xyz order, attribution)
    "topo": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "zyx", "Esri World Topographic Map",
    ),
    "relief": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        "zyx", "Esri World Shaded Relief",
    ),
    "terrain": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
        "zyx", "Esri World Terrain Base",
    ),
    "opentopo": (
        "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "zxy", "OpenTopoMap (CC-BY-SA)",
    ),
}


# ---------------------------------------------------------------------------
# Geometry helpers (projection-independent; operate on raw lat/lon)
# ---------------------------------------------------------------------------
def _links_from_network(network) -> List[dict]:
    """Return the list of link dicts (index 0 = dummy) from various inputs.

    Accepts an ``alt.Network`` (anything exposing ``to_pydict``), the tuple that
    ``Network.to_pydict()`` returns, or an already-extracted list of link dicts.
    """
    if hasattr(network, "to_pydict"):
        return list(network.to_pydict()[1])
    if isinstance(network, tuple):
        return list(network[1])
    return list(network)


def _link_polyline(link: dict) -> List[Tuple[float, float, float]]:
    """Return a link's centerline as ``[(offset_m, lat, lon), ...]``, sorted.

    Points missing lat/lon are dropped; the result may be empty (e.g. dummy or
    buffer links without geometry).
    """
    pts: List[Tuple[float, float, float]] = []
    for h in link.get("headings") or []:
        lat, lon = h.get("lat"), h.get("lon")
        off = h.get("offset_meters")
        if lat is None or lon is None or off is None:
            continue
        pts.append((float(off), float(lat), float(lon)))
    pts.sort(key=lambda t: t[0])
    return pts


def _interp(polyline: Sequence[Tuple[float, float, float]], offset: float
            ) -> Tuple[float, float]:
    """Linearly interpolate ``(lat, lon)`` at ``offset`` along ``polyline``."""
    if offset <= polyline[0][0]:
        return polyline[0][1], polyline[0][2]
    if offset >= polyline[-1][0]:
        return polyline[-1][1], polyline[-1][2]
    for (o0, la0, lo0), (o1, la1, lo1) in zip(polyline[:-1], polyline[1:]):
        if o0 <= offset <= o1:
            span = o1 - o0
            f = 0.0 if span == 0 else (offset - o0) / span
            return la0 + f * (la1 - la0), lo0 + f * (lo1 - lo0)
    return polyline[-1][1], polyline[-1][2]


def _interval_coords(polyline: Sequence[Tuple[float, float, float]],
                     start: float, end: float
                     ) -> List[Tuple[float, float]]:
    """Return the ``(lat, lon)`` polyline of the sub-interval ``[start, end]``.

    Includes the interpolated endpoints plus every original vertex strictly
    between them, so curvature within the interval is preserved.
    """
    if start > end:
        start, end = end, start
    coords = [_interp(polyline, start)]
    for off, lat, lon in polyline:
        if start < off < end:
            coords.append((lat, lon))
    coords.append(_interp(polyline, end))
    return coords


# ---------------------------------------------------------------------------
# 1-D interval algebra (for classifying both- vs single-direction coverage)
# ---------------------------------------------------------------------------
def _norm(intervals, tol: float = 1.0):
    """Sort + merge overlapping/adjacent ``(start, end)`` intervals (meters)."""
    ivs = sorted((min(s, e), max(s, e)) for s, e in intervals)
    out: List[List[float]] = []
    for s, e in ivs:
        if out and s <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _intersect(a, b):
    """Intersection of two interval lists."""
    a, b = _norm(a), _norm(b)
    res, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            res.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return res


def _subtract(a, b):
    """``a`` minus ``b`` (interval lists)."""
    a, b = _norm(a), _norm(b)
    res = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                res.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
        if cur < e:
            res.append((cur, e))
    return res


# ---------------------------------------------------------------------------
# Substation extraction
# ---------------------------------------------------------------------------
def _sub_attr(sub, name, default=None):
    """Read ``name`` from a substation given as an object or a dict."""
    if isinstance(sub, dict):
        return sub.get(name, default)
    return getattr(sub, name, default)


def _categorize_substations(
    substations: Iterable,
    built_ids: Optional[set],
) -> Dict[str, List[Tuple[float, float, str]]]:
    """Split substations into existing / built / candidate ``(lat, lon, id)`` lists."""
    groups: Dict[str, List[Tuple[float, float, str]]] = {
        "existing": [], "built": [], "candidate": []
    }
    for sub in substations or []:
        lat, lon = _sub_attr(sub, "lat"), _sub_attr(sub, "lon")
        if lat is None or lon is None:
            continue
        sid = _sub_attr(sub, "id", "")
        if _sub_attr(sub, "existing", False):
            key = "existing"
        elif built_ids is not None and sid in built_ids:
            key = "built"
        else:
            key = "candidate"
        groups[key].append((float(lat), float(lon), str(sid)))
    return groups


# ---------------------------------------------------------------------------
# Projection, basemap, scale bar
# ---------------------------------------------------------------------------
def _lambert_conformal(lat_min, lat_max, lon_min, lon_max):
    """A Lambert Conformal Conic CRS fit to the data's lat/lon extent."""
    import cartopy.crs as ccrs

    dlat = max(lat_max - lat_min, 1e-6)
    return ccrs.LambertConformal(
        central_longitude=(lon_min + lon_max) / 2.0,
        central_latitude=(lat_min + lat_max) / 2.0,
        standard_parallels=(lat_min + dlat / 6.0, lat_max - dlat / 6.0),
    )


def _make_tiler(url: str, order: str):
    """Return a Cartopy tile source that fetches ``url`` (a ``{z}/{x}/{y}`` template)."""
    from cartopy.io.img_tiles import GoogleTiles

    class _Tiler(GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            return url.format(x=x, y=y, z=z)

    _ = order  # url template already encodes tile axis order
    return _Tiler()


def _tile_zoom(lon_span: float, override: Optional[int]) -> int:
    """Pick a tile zoom level from the longitude span (or use ``override``)."""
    if override:
        return int(override)
    z = int(math.log2(360.0 / max(lon_span, 1e-6))) + 2
    return max(3, min(11, z))


def _nice_step(span: float) -> float:
    """Return a 'nice' scale-bar length (1/2/5 x 10^k) near ~30% of ``span``."""
    if span <= 0:
        return 1.0
    raw = span * 0.3
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def _add_scale_bar(ax):
    """Draw a scale bar in MILES (projected LCC units are meters, near-true)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_mi = (x1 - x0) / _M_PER_MILE
    bar_mi = _nice_step(span_mi)
    bar_m = bar_mi * _M_PER_MILE
    bx0 = x0 + 0.06 * (x1 - x0)
    by = y0 + 0.07 * (y1 - y0)
    ax.plot([bx0, bx0 + bar_m], [by, by], color="black", linewidth=2.5,
            solid_capstyle="butt", zorder=9)
    ax.text(bx0 + bar_m / 2.0, by + 0.015 * (y1 - y0), f"{bar_mi:g} mi",
            ha="center", va="bottom", fontsize=8, zorder=9)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def plot_electrified_track(
    network,
    *,
    result=None,
    energized_link_intervals: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    substations: Optional[Iterable] = None,
    built_substation_ids: Optional[Iterable] = None,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    show_base_network: bool = True,
    show_candidate_substations: bool = False,
    basemap="topo",
    tiles_zoom: Optional[int] = None,
    gridlines: bool = True,
    scale_bar: bool = True,
    projection=None,
    substation_feed_distance_miles: Optional[float] = None,
    electrified_color: str = "#e8710a",
    covered_color: str = "#1f6feb",
    uncovered_color: str = "#9aa0a6",
    electrified_width: float = 2.6,
    track_width: float = 1.6,
    padding_deg: float = 0.4,
    figsize: Tuple[float, float] = (11.0, 7.0),
    dpi: int = 300,
    ax=None,
):
    """Render a projected static map of electrified track; return ``(fig, ax)``.

    Parameters
    ----------
    network
        An ``alt.Network`` (or its ``to_pydict()`` tuple, or a list of link
        dicts) whose link ``headings`` carry ``lat``/``lon``.
    result
        An energization-placement result (e.g. from ``en.energize_optimized``).
        When given, ``energized_link_intervals`` and ``built_substation_ids`` are
        taken from it unless passed explicitly.
    energized_link_intervals
        ``{link_idx: [(start_m, end_m), ...]}`` of energized sub-intervals.
        Overrides ``result.energized_link_intervals`` when provided.
    substations
        Iterable of substations (``cp.Substation`` objects or dicts with
        ``lat``/``lon``/``id``/``existing``) to mark on the map.
    built_substation_ids
        Ids of substations that were built. Defaults to
        ``result.built_substation_ids``. Existing substations are always shown as
        feeding regardless of this set.
    out_path
        Output file. The extension selects the format: ``.pdf``/``.svg`` (vector,
        best for manuscripts) or ``.png``/``.tif`` (raster at ``dpi``). ``None``
        writes nothing and just returns the figure.
    basemap
        Basemap style. A topographic tile key -- ``"topo"`` (Esri World Topo,
        default), ``"relief"``, ``"terrain"``, ``"opentopo"`` -- fetches tiles and
        reprojects them into the map CRS. ``"features"`` uses offline Natural Earth
        vector features. ``None``/``False`` draws no basemap. Tiles need internet;
        on failure it falls back to features with a warning.
    tiles_zoom
        Override the tile zoom level (higher = more detail, more tiles).
    gridlines, scale_bar
        Draw a lat/lon graticule with degree labels / a scale bar (in miles).
    projection
        A ``cartopy.crs.Projection`` for the map. Defaults to a Lambert Conformal
        Conic fit to the corridor's extent (recommended for long corridors).
    both_direction_color, single_direction_color
        (Deprecated names kept out of the signature.) Colors are now
        ``electrified_color`` / ``covered_color`` / ``uncovered_color`` below.
    substation_feed_distance_miles
        Feed distance (track miles) defining the blue "within range" band; should
        match the value used in the optimization run. Defaults to 20.0.
    electrified_color, covered_color, uncovered_color
        Segment colors of the single network line: electrified (orange), not
        electrified but within track-distance range of a built/existing
        substation (blue), and out of range (gray). Coverage uses the same
        corridor-arc TRACK distance as the optimizer and line-loss model.
    dpi
        Raster resolution for ``.png``/``.tif`` (ignored by vector formats).
    ax
        Draw into an existing Cartopy ``GeoAxes`` (for multi-panel figures)
        instead of creating a new projected figure.

    Returns
    -------
    (matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxes)
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import geopandas as gpd
        from shapely.geometry import LineString, MultiLineString
        from shapely.ops import linemerge
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise ImportError(
            "plot_electrified_track needs the 'mapping' extra "
            "(cartopy, geopandas, pyproj). Install with: pip install altrios[mapping]"
        ) from exc

    if result is not None:
        if energized_link_intervals is None:
            energized_link_intervals = getattr(result, "energized_link_intervals", None)
        if built_substation_ids is None:
            built_substation_ids = getattr(result, "built_substation_ids", None)
    energized_link_intervals = energized_link_intervals or {}
    built_ids = set(built_substation_ids) if built_substation_ids is not None else None

    links = _links_from_network(network)
    polylines = {
        L["idx_curr"]: _link_polyline(L)
        for L in links
        if L.get("idx_curr") and L.get("headings")
    }
    flip_of = {L["idx_curr"]: L.get("idx_flip", 0) for L in links if L.get("idx_curr")}

    # --- per-link feeding coverage (same corridor-arc TRACK distance the
    #     optimizer + line-loss model use), for the built plan ---------------
    covered: Dict[int, bool] = {}
    if substations:
        try:
            from altrios.optimization.energization_placement import link_coverage

            feed = substation_feed_distance_miles
            covered = link_coverage(
                network, substations, built_substation_ids=built_ids,
                substation_feed_distance_miles=20.0 if feed is None else float(feed),
            )
        except Exception as exc:  # noqa: BLE001 - coverage shading is best-effort
            import warnings

            warnings.warn(f"coverage shading unavailable ({exc}); all track out of range")

    # --- classify every physical link and color its FULL polyline (the exact
    #     geometry the base network draws cleanly). Coloring whole links -- not
    #     interpolated sub-segments -- avoids the "fan" artifacts that arose from
    #     reconstructing/clamping partial intervals. Links are short (~0.05 mi),
    #     so per-link coloring is visually continuous and plenty granular. -----
    lats_all: List[float] = []
    lons_all: List[float] = []

    def _track(coords) -> LineString:
        for lat, lon in coords:
            lats_all.append(lat)
            lons_all.append(lon)
        return LineString([(lon, lat) for lat, lon in coords])

    elec_geoms: List[LineString] = []
    covered_geoms: List[LineString] = []
    uncovered_geoms: List[LineString] = []
    elec_miles = covered_miles = uncovered_miles = 0.0
    seen: set = set()
    for idx, poly in polylines.items():
        if not poly or idx in seen:
            continue
        flip = flip_of.get(idx, 0)
        seen.add(idx)
        seen.add(flip)
        length_m = poly[-1][0]
        a_int = energized_link_intervals.get(idx, [])
        b_int = energized_link_intervals.get(flip, [])
        b_mapped = [(length_m - e, length_m - s) for s, e in b_int]
        elec = _intersect(_norm(a_int + b_mapped), [(0.0, length_m)])
        elec_len = sum(e - s for s, e in elec)
        miles = length_m / _M_PER_MILE
        geom = _track([(lat, lon) for _o, lat, lon in poly])
        if length_m > 0.0 and elec_len >= 0.5 * length_m:
            elec_geoms.append(geom)
            elec_miles += miles
        elif covered.get(idx) or covered.get(flip):
            covered_geoms.append(geom)
            covered_miles += miles
        else:
            uncovered_geoms.append(geom)
            uncovered_miles += miles

    sub_groups = _categorize_substations(substations, built_ids) if substations else {}
    for pts in sub_groups.values():
        for lat, lon, _sid in pts:
            lats_all.append(lat)
            lons_all.append(lon)

    if not lats_all:
        raise ValueError("network has no lat/lon geometry to plot")
    lat_min, lat_max = min(lats_all), max(lats_all)
    lon_min, lon_max = min(lons_all), max(lons_all)

    # --- projected axes -----------------------------------------------------
    proj = projection or _lambert_conformal(lat_min, lat_max, lon_min, lon_max)
    pc = ccrs.PlateCarree()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure
    ax.set_extent(
        [lon_min - padding_deg, lon_max + padding_deg,
         lat_min - padding_deg, lat_max + padding_deg],
        crs=pc,
    )

    # --- basemap ------------------------------------------------------------
    mode = "topo" if basemap is True else (None if not basemap else str(basemap))

    def _add_features():
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f1ec", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbe7f0", zorder=0)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#dbe7f0",
                       edgecolor="none", zorder=1)
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#b8b8b8",
                       linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#7a7a7a",
                       linewidth=0.8, zorder=1)

    attribution = None
    if mode in _TILE_SOURCES:
        url, order, attribution = _TILE_SOURCES[mode]
        zoom = _tile_zoom(lon_max - lon_min, tiles_zoom)
        try:
            ax.add_image(_make_tiler(url, order), zoom, interpolation="bilinear",
                         zorder=0)
        except Exception as exc:  # noqa: BLE001 - tiles are best-effort (need net)
            import warnings

            warnings.warn(f"topographic tiles unavailable ({exc}); using Natural Earth")
            attribution = None
            try:
                _add_features()
            except Exception:  # noqa: BLE001
                pass
    elif mode == "features":
        try:
            _add_features()
        except Exception as exc:  # noqa: BLE001 - basemap is best-effort
            import warnings

            warnings.warn(f"basemap features unavailable ({exc}); drawing without them")

    # --- track layers (reproject via GeoPandas, plot in map CRS) -----------
    def _merge_lines(geoms):
        """Stitch adjacent links that share endpoints into continuous lines.

        Drawing each ~0.05 mi link as its own segment left tiny gaps / "fans" at
        curves (each segment had its own line caps). Merging same-category links
        into maximal continuous ``LineString``\\s makes matplotlib draw one joined
        path per run, so curves render smoothly.
        """
        if len(geoms) <= 1:
            return geoms
        merged = linemerge(MultiLineString(geoms))
        if merged.is_empty:
            return geoms
        if merged.geom_type == "LineString":
            return [merged]
        return list(merged.geoms)

    def _plot_lines(geoms, color, width, z):
        if not geoms:
            return
        gdf = gpd.GeoDataFrame(geometry=_merge_lines(geoms), crs="EPSG:4326").to_crs(proj)
        gdf.plot(ax=ax, color=color, linewidth=width, zorder=z)
        # Round caps/joins so any remaining segment ends blend at curves.
        try:
            coll = ax.collections[-1]
            coll.set_capstyle("round")
            coll.set_joinstyle("round")
        except Exception:  # noqa: BLE001 - cosmetic only
            pass

    # network coloured by category: out-of-range gray, within-range blue
    # (bottom), then substations, then electrified orange on top.
    if show_base_network:
        _plot_lines(uncovered_geoms, uncovered_color, track_width, 2)
        _plot_lines(covered_geoms, covered_color, track_width, 3)

    # --- substation markers (under the electrified track) -------------------
    legend_handles: List[Line2D] = []
    marker_spec = [
        ("existing", "Existing substation", "#1a7f37", "s", 70, 5),
        ("built", "Built substation", "#0b3d91", "o", 90, 5),
    ]
    if show_candidate_substations:
        marker_spec.append(("candidate", "Candidate substation", "#b0b3b8", "o", 28, 4))
    for key, label, color, marker, size, z in marker_spec:
        pts = sub_groups.get(key) or []
        if not pts:
            continue
        lons = [p[1] for p in pts]
        lats = [p[0] for p in pts]
        ax.scatter(lons, lats, s=size, c=color, marker=marker, edgecolors="white",
                   linewidths=0.7, zorder=z, transform=pc)
        legend_handles.append(Line2D(
            [], [], color=color, marker=marker, linestyle="none",
            markersize=math.sqrt(size), markeredgecolor="white", label=label,
        ))

    # electrified track ON TOP of substations
    _plot_lines(elec_geoms, electrified_color, electrified_width, 7)

    # --- graticule ----------------------------------------------------------
    if gridlines:
        gl = ax.gridlines(draw_labels=True, color="0.6", linestyle=":",
                          linewidth=0.5, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
    if scale_bar:
        _add_scale_bar(ax)

    # --- legend -------------------------------------------------------------
    line_handles: List[Line2D] = []
    if elec_geoms:
        line_handles.append(Line2D(
            [], [], color=electrified_color, linewidth=electrified_width,
            label=f"Electrified ({elec_miles:.0f} mi)"))
    if show_base_network and covered_geoms:
        line_handles.append(Line2D(
            [], [], color=covered_color, linewidth=track_width,
            label=f"Within range, not electrified ({covered_miles:.0f} mi)"))
    if show_base_network and uncovered_geoms:
        line_handles.append(Line2D(
            [], [], color=uncovered_color, linewidth=track_width,
            label="Out of range"))
    handles = line_handles + legend_handles
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=8,
                  framealpha=0.9, edgecolor="#444")

    if attribution:
        ax.text(0.995, 0.008, attribution, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6, color="#333",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none",
                      "pad": 1.0}, zorder=10)

    if title:
        ax.set_title(title, fontsize=12)

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")

    return fig, ax


def corridor_projection(network):
    """Return the default Lambert Conformal Conic projection fit to ``network``.

    Handy for building multi-panel figures whose ``GeoAxes`` must share one CRS
    before being passed to :func:`plot_electrified_track` via ``ax=``.
    """
    lats: List[float] = []
    lons: List[float] = []
    for L in _links_from_network(network):
        for h in L.get("headings") or []:
            la, lo = h.get("lat"), h.get("lon")
            if la is not None and lo is not None:
                lats.append(float(la))
                lons.append(float(lo))
    if not lats:
        raise ValueError("network has no lat/lon geometry")
    return _lambert_conformal(min(lats), max(lats), min(lons), max(lons))


def plot_comparison(
    network,
    panels: Sequence[dict],
    *,
    substations: Optional[Iterable] = None,
    out_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    **kwargs,
):
    """Render several plans side-by-side in one figure sharing a projection.

    ``panels`` is a sequence of dicts, each with a ``label`` (subplot title) and
    a plan given as either ``result`` or ``energized_link_intervals`` (plus
    optional ``built_substation_ids``). Any extra ``**kwargs`` (e.g. ``basemap``,
    ``show_candidate_substations``) are forwarded to every panel's
    :func:`plot_electrified_track`.

    Returns ``(fig, axes)``.
    """
    proj = corridor_projection(network)
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=figsize or (11.0 * n, 7.0),
        subplot_kw={"projection": proj},
    )
    if n == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        plot_electrified_track(
            network, ax=ax, substations=substations,
            result=panel.get("result"),
            energized_link_intervals=panel.get("energized_link_intervals"),
            built_substation_ids=panel.get("built_substation_ids"),
            title=panel.get("label"),
            **kwargs,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    # Smoke test: draw the bare Spokane-Lethbridge corridor (no electrification).
    import altrios as alt

    net_path = (alt.resources_root() / "networks" / "Spokane_Lethbridge"
                / "spokane_lethbridge.yaml")
    net = alt.Network.from_file(str(net_path))
    out = Path(__file__).resolve().parent / "electrified_network_smoketest.png"
    plot_electrified_track(
        net, out_path=str(out),
        title="Spokane-Lethbridge corridor (smoke test)",
    )
    print(f"wrote {out}")
