#!/usr/bin/env python3
"""
Download FMI weather observations and build a NetworkX spatial graph.

Graph structure
---------------
Each node represents one FMI weather station.
Node attributes:
  lat        float   – WGS84 latitude
  lon        float   – WGS84 longitude
  fmisid     int     – FMI station ID
  times      np.ndarray[datetime64[s]]  – observation timestamps
  <param>    np.ndarray[float64]        – measurements (NaN where missing)
  <param>_unit str   – unit string for that parameter

Edges connect each station to its k nearest neighbours (Haversine distance).
Edge attributes:
  distance_km  float

Outputs
-------
  fmi_graph.pkl         – pickled nx.Graph (preserves numpy arrays)
  fmi_stations_map.png  – map of Finland with station locations

Usage
-----
python BuildFMIGraph.py --start 2025-01-01T00:00:00Z --end 2025-01-02T00:00:00Z

Add --chunk-minutes 360 (default) to avoid large single responses from FMI.
Add --k-neighbors 0 to produce a node-only graph (no edges).
"""

import argparse
import datetime as dt
import math
import os
import pickle
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from fmiopendata.wfs import download_stored_query


# ── optional: geopandas for Finland boundary ──────────────────────────────────
try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_utc(t: dt.datetime) -> dt.datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def _iso_z(t: dt.datetime) -> str:
    return _to_utc(t).isoformat(timespec="seconds").replace("+00:00", "Z")


def _iter_chunks(
    start: dt.datetime, end: dt.datetime, minutes: int
) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    delta = dt.timedelta(minutes=minutes)
    cur = start
    while cur < end:
        yield cur, min(cur + delta, end)
        cur = min(cur + delta, end)


# ─────────────────────────────────────────────────────────────────────────────
# Geography
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# FMI download
# ─────────────────────────────────────────────────────────────────────────────

def _query_chunk(bbox: str, start: str, end: str) -> Any:
    return download_stored_query(
        "fmi::observations::weather::multipointcoverage",
        args=[f"bbox={bbox}", f"starttime={start}", f"endtime={end}", "timeseries=True"],
    )


def download_fmi(
    bbox: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    chunk_minutes: int,
) -> Dict[str, Any]:
    """
    Download FMI timeseries data for all stations in bbox.

    Returns a dict keyed by station name:
        {
          'fmisid': int,
          'lat': float,
          'lon': float,
          'times': [datetime, ...],
          '<param>': {'values': [float|None, ...], 'unit': str},
          ...
        }
    """
    merged: Dict[str, Any] = {}

    if chunk_minutes > 0:
        chunks = list(_iter_chunks(start_dt, end_dt, chunk_minutes))
    else:
        chunks = [(start_dt, end_dt)]

    for i, (cs, ce) in enumerate(chunks, 1):
        cs_s, ce_s = _iso_z(cs), _iso_z(ce)
        print(f"  [{i}/{len(chunks)}] {cs_s} → {ce_s}")

        obs = _query_chunk(bbox, cs_s, ce_s)
        meta = obs.location_metadata  # name -> {fmisid, latitude, longitude}

        for station, sdata in obs.data.items():
            times: list = sdata.get("times", [])
            if not times:
                continue

            m = meta.get(station, {})

            if station not in merged:
                merged[station] = {
                    "fmisid": m.get("fmisid"),
                    "lat": m.get("latitude"),
                    "lon": m.get("longitude"),
                    "times": list(times),
                }
                for param, payload in sdata.items():
                    if param == "times":
                        continue
                    merged[station][param] = {
                        "values": list(payload.get("values", [])),
                        "unit": payload.get("unit", ""),
                    }
            else:
                merged[station]["times"].extend(times)
                for param, payload in sdata.items():
                    if param == "times":
                        continue
                    if param not in merged[station]:
                        merged[station][param] = {
                            "values": list(payload.get("values", [])),
                            "unit": payload.get("unit", ""),
                        }
                    else:
                        merged[station][param]["values"].extend(payload.get("values", []))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Build NetworkX graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(station_data: Dict[str, Any], k_neighbors: int = 5) -> nx.Graph:
    """
    Build a NetworkX Graph from FMI station data.

    Parameters
    ----------
    station_data:
        Output of download_fmi().
    k_neighbors:
        Each station is connected to its k nearest neighbours.
        Pass 0 to skip edge creation.
    """
    G = nx.Graph()
    _SKIP = {"fmisid", "lat", "lon", "times"}

    # ── add nodes ──────────────────────────────────────────────────────────
    for name, data in station_data.items():
        attrs: Dict[str, Any] = {
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "fmisid": data.get("fmisid"),
            "times": np.array(
                [_to_utc(t) if isinstance(t, dt.datetime) else t for t in data.get("times", [])],
                dtype="datetime64[s]",
            ),
        }

        for key, val in data.items():
            if key in _SKIP:
                continue
            raw = val.get("values", [])
            attrs[key] = np.array(
                [v if v is not None else np.nan for v in raw], dtype=np.float64
            )
            attrs[f"{key}_unit"] = val.get("unit", "")

        G.add_node(name, **attrs)

    # ── add k-NN spatial edges ─────────────────────────────────────────────
    if k_neighbors > 0:
        node_list = [
            (n, d) for n, d in G.nodes(data=True)
            if d.get("lat") is not None and d.get("lon") is not None
        ]
        for i, (ni, di) in enumerate(node_list):
            dists: List[Tuple[float, str]] = []
            for j, (nj, dj) in enumerate(node_list):
                if i == j:
                    continue
                d_km = _haversine_km(di["lat"], di["lon"], dj["lat"], dj["lon"])
                dists.append((d_km, nj))
            dists.sort()
            for d_km, nj in dists[:k_neighbors]:
                if not G.has_edge(ni, nj):
                    G.add_edge(ni, nj, distance_km=round(d_km, 2))

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _load_finland_boundary():
    """Return a GeoDataFrame with Finland's polygon, or None."""
    if not _HAS_GPD:
        return None

    # Download (and cache) the Natural Earth 110m admin-0 countries shapefile.
    # pooch is already a transitive dependency of geodatasets.
    try:
        import pooch
        path = pooch.retrieve(
            url="https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
            known_hash=None,
            fname="ne_110m_admin_0_countries.zip",
            path=pooch.os_cache("fmi_graph"),
        )
        world = gpd.read_file(f"zip://{path}")
        # Column name differs by version: NAME_EN, NAME, SOVEREIGNT
        for col in ("NAME_EN", "NAME", "SOVEREIGNT", "name"):
            if col in world.columns:
                fi = world[world[col] == "Finland"]
                if not fi.empty:
                    return fi
    except Exception as e:
        print(f"  [warn] Could not load Finland boundary: {e}")

    return None


def plot_stations(G: nx.Graph, out_path: str, show_edges: bool = True) -> None:
    """
    Save a PNG map of Finland with FMI station locations.

    Parameters
    ----------
    G:
        Graph built by build_graph().
    out_path:
        Destination PNG path.
    show_edges:
        Whether to draw k-NN edges as light grey lines.
    """
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_facecolor("#eaf4fb")

    # ── Finland boundary ───────────────────────────────────────────────────
    fi = _load_finland_boundary()
    if fi is not None:
        fi.boundary.plot(ax=ax, color="#2c6fad", linewidth=1.4, zorder=2)
        fi.plot(ax=ax, color="#d6e9f5", zorder=1)
    else:
        # Minimal fallback: just set reasonable axis limits
        ax.set_xlim(18.5, 32.5)
        ax.set_ylim(59.0, 70.5)
        ax.text(
            25.5, 64.5, "Finland\n(boundary unavailable;\ninstall geopandas)",
            ha="center", va="center", fontsize=9, color="gray",
        )

    # ── k-NN edges ─────────────────────────────────────────────────────────
    if show_edges:
        for u, v in G.edges():
            u_d, v_d = G.nodes[u], G.nodes[v]
            if None not in (u_d.get("lat"), v_d.get("lat")):
                ax.plot(
                    [u_d["lon"], v_d["lon"]],
                    [u_d["lat"], v_d["lat"]],
                    color="#aaaaaa", linewidth=0.4, alpha=0.6, zorder=3,
                )

    # ── station scatter ────────────────────────────────────────────────────
    lats, lons = [], []
    for _, attrs in G.nodes(data=True):
        if attrs.get("lat") is not None and attrs.get("lon") is not None:
            lats.append(attrs["lat"])
            lons.append(attrs["lon"])

    ax.scatter(
        lons, lats, s=22, c="#d62728", edgecolors="white", linewidths=0.4,
        zorder=5, label=f"FMI stations  (n={len(lats)})",
    )

    # ── legend / labels ────────────────────────────────────────────────────
    station_handle = mlines.Line2D(
        [], [], marker="o", color="w", markerfacecolor="#d62728",
        markeredgecolor="white", markersize=7,
        label=f"FMI stations  (n={len(lats)})",
    )
    boundary_handle = mlines.Line2D(
        [], [], color="#2c6fad", linewidth=1.4, label="Finland boundary"
    )
    handles = [boundary_handle, station_handle] if fi is not None else [station_handle]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.set_title("FMI Observation Stations – Finland", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Map saved  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download FMI weather observations and build a NetworkX spatial graph."
    )
    ap.add_argument(
        "--bbox", default="19,59,32,72",
        help="Bounding box 'minLon,minLat,maxLon,maxLat' (WGS84). Default: Finland",
    )
    ap.add_argument("--start", required=True, help="Start time, e.g. 2025-01-01T00:00:00Z")
    ap.add_argument("--end",   required=True, help="End time,   e.g. 2025-01-02T00:00:00Z")
    ap.add_argument(
        "--chunk-minutes", type=int, default=360,
        help="Split request into chunks of this many minutes (default: 360). Use 0 to disable.",
    )
    ap.add_argument(
        "--k-neighbors", type=int, default=5,
        help="k nearest-neighbour edges per station (default: 5). Use 0 for node-only graph.",
    )
    ap.add_argument(
        "--no-edges-on-map", action="store_true",
        help="Do not draw k-NN edges on the station map.",
    )
    ap.add_argument(
        "--out-graph", default="fmi_graph.pkl",
        help="Output path for the pickled NetworkX graph (default: fmi_graph.pkl)",
    )
    ap.add_argument(
        "--out-map", default="fmi_stations_map.png",
        help="Output path for the station map PNG (default: fmi_stations_map.png)",
    )
    args = ap.parse_args()

    start_dt = _to_utc(dt.datetime.fromisoformat(args.start.replace("Z", "+00:00")))
    end_dt   = _to_utc(dt.datetime.fromisoformat(args.end.replace("Z",   "+00:00")))
    if end_dt <= start_dt:
        raise ValueError("--end must be strictly after --start")

    # ── download ───────────────────────────────────────────────────────────
    print(f"\nDownloading FMI observations  {_iso_z(start_dt)} → {_iso_z(end_dt)}")
    print(f"  bbox={args.bbox}  chunk={args.chunk_minutes} min")
    station_data = download_fmi(args.bbox, start_dt, end_dt, args.chunk_minutes)
    print(f"  Stations received: {len(station_data)}")

    # ── build graph ────────────────────────────────────────────────────────
    print(f"\nBuilding NetworkX graph  (k_neighbors={args.k_neighbors})")
    G = build_graph(station_data, k_neighbors=args.k_neighbors)
    print(f"  Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}")

    # ── save graph ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_graph) or ".", exist_ok=True)
    with open(args.out_graph, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Graph saved → {args.out_graph}")

    # ── plot ───────────────────────────────────────────────────────────────
    print("\nPlotting station map …")
    plot_stations(G, args.out_map, show_edges=not args.no_edges_on_map)

    # ── summary ────────────────────────────────────────────────────────────
    sample_name, sample_attrs = next(iter(G.nodes(data=True)))
    params = [
        k for k in sample_attrs
        if k not in {"lat", "lon", "fmisid", "times"} and not k.endswith("_unit")
    ]
    n_obs = len(sample_attrs.get("times", []))
    print(f"\nSample node  '{sample_name}'")
    print(f"  lat={sample_attrs.get('lat'):.4f}  lon={sample_attrs.get('lon'):.4f}")
    print(f"  Observations: {n_obs}")
    print(f"  Parameters ({len(params)}): {params[:8]}")
    if len(params) > 8:
        print(f"    … and {len(params) - 8} more")

    print("\nTo reload the graph:")
    print("  import pickle, networkx as nx")
    print(f"  G = pickle.load(open('{args.out_graph}', 'rb'))")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
# 1) One day of data, 6-hour chunks, 5 nearest neighbours per station
#    python BuildFMIGraph.py \
#      --start 2025-01-01T00:00:00Z \
#      --end   2025-01-02T00:00:00Z
#
# 2) Node-only graph (no edges), no edge lines on map
#    python BuildFMIGraph.py \
#      --start 2025-01-01T00:00:00Z \
#      --end   2025-01-01T06:00:00Z \
#      --k-neighbors 0 --no-edges-on-map
#
# 3) Custom output paths
#    python BuildFMIGraph.py \
#      --start 2025-03-01T00:00:00Z \
#      --end   2025-03-08T00:00:00Z \
#      --chunk-minutes 1440 \
#      --out-graph data/week_graph.pkl \
#      --out-map   data/week_stations.png
#
# HOW TO INSPECT THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
# import pickle
# import numpy as np
# G = pickle.load(open("fmi_graph.pkl", "rb"))
#
# # List all nodes (stations)
# print(list(G.nodes)[:5])
#
# # Access node attributes
# node = G.nodes["Helsinki Kaisaniemi"]
# print(node["lat"], node["lon"])
# print(node["times"][:3])                    # numpy datetime64 array
# print(node["Air temperature"][:10])         # numpy float64 array
# print(node["Air temperature_unit"])
#
# # Iterate over edges with distance
# for u, v, d in G.edges(data=True):
#     print(u, "↔", v, f"  {d['distance_km']} km")
#     break
