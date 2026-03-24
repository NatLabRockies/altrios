import json
import os

filepath = "applications/network_builder/results/orm_data_route/orm_route_Eastport_Kingsgate_Border.json"

with open(filepath, "r") as f:
    data = json.load(f)

print(f"Loaded data from {filepath}")

crossing_ways = []

for element in data["elements"]:
    if element["type"] == "way" and "geometry" in element:
        geometry = element["geometry"]
        lats = [pt["lat"] for pt in geometry]
        min_lat = min(lats)
        max_lat = max(lats)

        # Check if way crosses the 49th parallel (approximate border)
        if min_lat < 49.0 and max_lat > 49.0:
            print(f"Found crossing way ID: {element['id']}")
            # Find the segment crossing 49.0
            for i in range(len(geometry) - 1):
                p1 = geometry[i]
                p2 = geometry[i + 1]

                if (p1["lat"] <= 49.0 <= p2["lat"]) or (p2["lat"] <= 49.0 <= p1["lat"]):
                    print(f"Crossing Segment:")
                    print(f"  Point 1: {p1['lat']}, {p1['lon']}")
                    print(f"  Point 2: {p2['lat']}, {p2['lon']}")
                    crossing_ways.append((p1, p2))

if not crossing_ways:
    print("No direct crossing found in this file (exact 49.0 check).")
