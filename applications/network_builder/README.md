# Network Builder

Builds a routable ALTRIOS network (mainline + passing sidings) for a rail
corridor from OpenStreetMap data, with USGS elevations, per-point lat/lons,
and an interactive route map.

```
OSM PBFs ──build_corridor──▶ OSM XML ──build_network──▶ ALTRIOS YAML + geometry JSON + locations CSV
 (Geofabrik)  (osmium clip)              │
                                         └──plot_json──▶ interactive HTML map
```

## Quickstart

```bash
uv sync                      # one-time setup (or: pip install altrios[networkbuilder])

# Build + plot the bundled Spokane -> Lethbridge demo corridor (offline, fast):
uv run python applications/network_builder/network_builder_demo.py

# Real elevations from USGS/OpenTopoData (adds ~10 min of API calls):
uv run python applications/network_builder/network_builder_demo.py --with-elevation

# A different corridor, defined purely as data:
uv run python applications/network_builder/network_builder_demo.py \
    --config applications/network_builder/corridors/winnipeg_chicago.yaml \
    --download --with-elevation
```

Outputs land in `applications/network_builder/results/` (gitignored):

| File | Contents |
|---|---|
| `<name>.yaml` | The ALTRIOS network (`altrios.Network.from_file` loads it) |
| `<name>_geometry.json` | `[lon, lat]` endpoints per link, for plotting/GIS |
| `<name>_locations.csv` | Origin/destination rows for the train planner (both directions) |
| `<name>_plot.html` | Interactive route map |

## Defining a corridor

Corridors are pure data — a YAML file passed via `--config` (see
`corridors/winnipeg_chicago.yaml`), matching the .yaml format of the other
ALTRIOS inputs, or a `BuilderConfig` instance in code.
Any subset of fields may be given; the rest keep their defaults
(see `builder_config.py` for the full reference):

- `output_name` — basename for all outputs and the default extract
  `networks/<output_name>_local.osm.xml`.
- `waypoints` — `[name, lat, lon]` list, ordered start → end. Waypoints do
  double duty: they define the corridor clip polygon **and** the mainline
  routing. Coordinates only need to be near the track (each is snapped to the
  nearest rail node), but add a waypoint at every route-decision point —
  between waypoints the router takes the *shortest* rail path, which will
  happily hop onto a parallel railroad. Prefer points on the through mainline
  over yards/stations. The first/last waypoint names label the locations CSV.
- `regions` — Geofabrik PBF URLs covering the corridor (only used with
  `--download`, which needs the `osmium-tool` CLI).
- `buffer_deg` — clip buffer around the waypoint line (0.4° ≈ 45 km). Too
  small risks cutting the rail line between waypoints.
- `max_grade` — cap for the elevation grade-spike filter (0.04 = 4%).
- `endpoint_buffer_meters` — length of the buffer links at both corridor ends
  (default 3000 m). ALTRIOS starts a train with its **tail** at offset 0 of
  the origin link, so this must exceed the longest train by its
  friction-braking distance from line speed (~200 m for a typical freight
  consist); the front-of-train travel surplus over the corridor length is
  `buffer − train_length`.
- `origin_regions` / `destination_regions` — (grid emissions, electricity
  price, liquid fuel price) region codes for the locations CSV.
- `include_lat_lons` (default on) — embed each node's lat/lon in the link
  `headings` entries. These are optional `Heading.lat`/`Heading.lon` fields in
  the ALTRIOS schema: ignored by the simulation, round-tripped by serde, and
  give downstream tools `(offset_meters, lat, lon)` per link straight from the
  YAML.
- `mapzen_fallback` (default on) — see elevation sources below.

## How the build works

1. **Extract** (`--download`): download regional PBFs, `osmium merge`, clip to
   the buffered waypoint polygon, filter `railway=rail`, convert to XML.
2. **Route**: snap waypoints to rail nodes; chain shortest paths into the
   mainline; prune out-and-back excursions (a waypoint that snaps to a
   dead-end spur would otherwise make the path retrace itself — untraversable
   for a train and poison for the elevation lookup).
3. **Sidings**: edges tagged `service=siding/yard/spur/crossover` (or
   industrial/military/test usage) form components; those sharing nodes with
   the mainline become passing sidings (two junctions) or stubs (one).
4. **Elevations** (`--with-elevation`): US points (lat < 49.001°) in bulk from
   OpenTopoData `ned10m` (USGS NED 10 m); Canadian points directly from the
   USGS 3DEP EPQS service, whose coverage extends past the border and shares
   the US vertical datum — so there is no dataset seam (artificial grade step)
   at border crossings. Points outside 3DEP coverage fall back to Mapzen
   (`mapzen_fallback=False` forces USGS-only); anything still missing is
   interpolated along the path. Savitzky-Golay smoothing plus a `max_grade`
   spike cap are applied to the mainline. Behind a TLS-intercepting proxy,
   point `REQUESTS_CA_BUNDLE` at your CA bundle.
5. **Links**: every OSM edge becomes a forward/reverse link pair (`idx_flip`),
   with mirrored elevations and headings offset by π, so grades are correct in
   both travel directions. Sidings attach to the mainline through
   `idx_next_alt`/`idx_prev_alt` in both directions, which is what lets the
   ALTRIOS dispatcher route meets/passes. The reciprocity invariants ALTRIOS
   requires are validated before export; a violation fails the build.
6. **Locations**: two rows per endpoint (forward + reverse direction links,
   the `default_locations.csv` convention), so trains can be dispatched either
   way out of the box.

## Using the result

```python
import altrios as alt

network = alt.Network.from_file("results/spokane_lethbridge.yaml")
location_map = alt.import_locations("results/spokane_lethbridge_locations.csv")
# then the standard flow: TrainSimBuilder -> make_speed_limit_train_sim
# -> make_est_times -> run_dispatch -> walk_timed_path
# (see python/altrios/demos/speed_limit_train_sim_demo.py)
```

Practical checks after building a new corridor: open the HTML plot and verify
the route follows the intended railroad (add waypoints where it shortcuts);
confirm the log line `Siding wiring OK`; and confirm the mainline length and
elevation profile look plausible before running simulations.
