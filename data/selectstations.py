#!/usr/bin/env python3
"""
Select 25 FMI weather stations as FL nodes for the wind-power forecasting
project, following the regional layout in FML_Project_Architecture.md
(Phase 1, Step 1.1).

Workflow:
    1. Run GetFMIData.py over a short window (e.g. 1 hour) to produce
       a "discovery" CSV containing every FMI station that reported
       observations in the Finland bbox.
    2. Run this script on that CSV. It:
       - loads the unique (fmisid, name, lat, lon) per station
       - for each of 5 wind-farm regions, picks the k=5 stations
         geographically nearest to a hand-picked anchor point
         (anchors taken from the architecture doc's list of large
         Finnish wind farms)
       - de-duplicates so the same station can't be assigned to two
         regions (regions are processed south-to-north, which matches
         the geographic ordering of the wind farms in the architecture)
       - writes data/selected_stations.json — the canonical, reproducible
         list of 25 FL nodes that every later phase reads from.

Reproducibility note (rubric criterion 15):
    The output JSON is the single source of truth for "which 25 stations
    is this project using?". Commit it to git. If you change the anchors
    or k, re-run and commit the new JSON — never edit it by hand.

Usage:
    # Step 1: discovery fetch (one hour is enough — we just need station
    # metadata, not their measurements)
    python data/GetFMIData.py \
        --start 2024-06-01T12:00:00Z \
        --end   2024-06-01T13:00:00Z \
        --timeseries \
        --out data/_discovery.csv

    # Step 2: pick the 25 stations
    python data/SelectStations.py \
        --discovery-csv data/_discovery.csv \
        --out data/selected_stations.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Region anchors
# ---------------------------------------------------------------------------
# Approximate (lat, lon) of the major wind farm each region is built around,
# taken from the architecture doc (Phase 1, Step 1.1). Public locations of
# operational Finnish wind farms; small offsets don't matter — we only need
# the anchor to rank stations by distance.
#
# Order is south -> north, which determines tie-breaking when the same
# station is closest to two anchors (the southerly region wins).

WIND_FARM_REGIONS: dict[str, dict] = {
    "satakunta_sw_coast": {
        "anchor": (61.60, 21.40),  # Tahkoluoto (Pori)
        "description": "Tahkoluoto, Oosinselkä",
    },
    "ostrobothnia_coast": {
        "anchor": (62.30, 21.40),  # Pjelax-Böle (Närpes)
        "description": "Pjelax-Böle 380MW",
    },
    "central_ostrobothnia": {
        "anchor": (63.70, 23.70),  # Mutkalampi area (Kalajoki / Kannus)
        "description": "Mutkalampi 404MW, Juurakko, Lestijärvi 455MW",
    },
    "northern_ostrobothnia": {
        "anchor": (64.70, 24.70),  # Siikajoki
        "description": "Siikajoki 236MW",
    },
    "kemi_lapland": {
        "anchor": (65.70, 24.50),  # Kemin Ajos
        "description": "Kemin Ajos",
    },
}

STATIONS_PER_REGION = 5  # 5 regions × 5 = 25 nodes total


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    earth_r_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * earth_r_km * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_station_inventory(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a discovery CSV produced by GetFMIData.py and collapse it to
    one row per station with stable metadata.

    The discovery CSV is long-format (many parameter rows per station per
    time); we only need (station, fmisid, lat, lon).
    """
    df = pd.read_csv(csv_path)

    required = {"station", "fmisid", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Discovery CSV {csv_path} is missing columns: {missing}. "
            f"Did you run GetFMIData.py with the expected schema?"
        )

    stations = (
        df[["station", "fmisid", "lat", "lon"]]
        .dropna()
        .drop_duplicates(subset=["fmisid"])
        .reset_index(drop=True)
    )

    # fmisid can come in as float64 when there are NaNs in the column.
    stations["fmisid"] = stations["fmisid"].astype(int)
    return stations


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_stations(
    stations_df: pd.DataFrame,
    k_per_region: int = STATIONS_PER_REGION,
) -> dict:
    """
    For each region, pick the k geographically nearest stations to the
    region's anchor point.

    De-duplication: once a station is assigned to a region, it cannot be
    re-assigned to a later region. Regions are processed in the dict
    insertion order (south -> north), so southerly regions get first pick
    of any contested coastal stations.
    """
    used: set[int] = set()
    selection: dict = {}

    for region, info in WIND_FARM_REGIONS.items():
        anchor_lat, anchor_lon = info["anchor"]

        # Filter to stations not yet claimed by an earlier region
        pool = stations_df[~stations_df["fmisid"].isin(used)].copy()

        # Vectorised Haversine
        pool["dist_km"] = pool.apply(
            lambda r: haversine_km(anchor_lat, anchor_lon, r["lat"], r["lon"]),
            axis=1,
        )

        picked = pool.nsmallest(k_per_region, "dist_km")
        if len(picked) < k_per_region:
            raise RuntimeError(
                f"Region {region} got only {len(picked)} candidates "
                f"(needed {k_per_region}). Check the discovery CSV "
                f"covers all of Finland."
            )
        used.update(int(f) for f in picked["fmisid"])

        selection[region] = {
            "anchor_lat": anchor_lat,
            "anchor_lon": anchor_lon,
            "description": info["description"],
            "stations": [
                {
                    "fmisid": int(r["fmisid"]),
                    "name": str(r["station"]),
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "dist_km_to_anchor": round(float(r["dist_km"]), 1),
                }
                for _, r in picked.iterrows()
            ],
        }

    return selection


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_output(selection: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


def print_summary(selection: dict) -> None:
    print()
    total = 0
    for region, info in selection.items():
        print(f"{region}  — {info['description']}")
        print(f"  anchor: ({info['anchor_lat']:.2f}, {info['anchor_lon']:.2f})")
        for s in info["stations"]:
            print(
                f"    [{s['fmisid']:>6}]  {s['name']:<30s}  "
                f"({s['lat']:.2f}, {s['lon']:.2f})  "
                f"{s['dist_km_to_anchor']:>5.1f} km"
            )
            total += 1
        print()
    print(f"Total stations selected: {total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--discovery-csv",
        required=True,
        help="CSV from GetFMIData.py listing all candidate FMI stations.",
    )
    ap.add_argument(
        "--out",
        default="data/selected_stations.json",
        help="Where to write the canonical 25-station JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=STATIONS_PER_REGION,
        help="Stations per region (default: %(default)s).",
    )
    ap.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=[],
        help="FMISIDs to exclude from selection (e.g. stations known to lack required data).",
    )
    args = ap.parse_args()

    stations = load_station_inventory(args.discovery_csv)
    if args.exclude:
        before = len(stations)
        stations = stations[~stations["fmisid"].isin(args.exclude)].reset_index(drop=True)
        print(f"Excluded {before - len(stations)} FMISIDs: {sorted(args.exclude)}")
    print(f"Loaded {len(stations)} candidate stations from {args.discovery_csv}")

    selection = select_stations(stations, k_per_region=args.k)
    print_summary(selection)
    write_output(selection, args.out)


if __name__ == "__main__":
    main()
    