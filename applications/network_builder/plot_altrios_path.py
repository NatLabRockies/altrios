import json
import plotly.graph_objects as go
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_json(json_path: str, html_path: str) -> None:
    """Plot forward-link geometry from JSON as an HTML map.

    Args:
        json_path: Geometry JSON with per-link `link_idx` and `coordinates`.
        html_path: Destination HTML file.
    """
    logger.info(f"Loading {json_path}...")
    with open(json_path, "r") as f:
        geometry_collection = json.load(f)

    logger.info(f"Plotting {len(geometry_collection) // 2} links...")

    lats = []
    lons = []

    for item in geometry_collection:
        # Only plot forward links to avoid double drawing
        if item["link_idx"] % 2 != 0:
            coords = item["coordinates"]
            lons.extend([coords[0][0], coords[1][0], None])
            lats.extend([coords[0][1], coords[1][1], None])

    fig = go.Figure(
        go.Scattermapbox(
            mode="lines",
            lon=lons,
            lat=lats,
            marker={"size": 5},
            line={"width": 3, "color": "red"},
            name="ALTRIOS Path",
        )
    )

    # Calculate map center
    valid_lats = [l for l in lats if l is not None]
    valid_lons = [l for l in lons if l is not None]

    if valid_lats:
        center_lat = sum(valid_lats) / len(valid_lats)
        center_lon = sum(valid_lons) / len(valid_lons)
    else:
        center_lat = 0
        center_lon = 0

    fig.update_layout(
        margin={"l": 0, "t": 0, "b": 0, "r": 0},
        mapbox={
            "style": "open-street-map",
            "center": {"lon": center_lon, "lat": center_lat},
            "zoom": 7,
        },
        title="Spokane to Lethbridge (ALTRIOS Path)",
    )

    fig.write_html(html_path)
    logger.info(f"Plot saved to {html_path}")


if __name__ == "__main__":
    plot_json(
        "networks/spokane_lethbridge_path_geometry.json",
        "networks/spokane_lethbridge_path_plot.html",
    )
