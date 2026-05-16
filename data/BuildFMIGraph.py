#!/usr/bin/env python3
"""
Phase 3: graph construction for the two FL systems.

System A — Geographic k-NN
    Edge rule:    k=3 nearest stations by Haversine distance,
                  symmetrised via union (edge {i,j} if j is in N_k(i)
                  OR i is in N_k(j)).
    Edge weight:  Gaussian kernel A_ij = exp(-d_ij^2 / (2 σ^2)),
                  σ = median of connected-pair distances.
    Justification: weather systems are spatially coherent; nearby
                   stations share wind regimes, so their local models
                   should be close in parameter space.

System B — Correlation k-NN
    Edge rule:    k=3 most-correlated neighbours by Pearson correlation
                  of wind speed over the TRAINING period, symmetrised
                  via union.
    Edge weight:  A_ij = max(0, ρ_ij).
    Justification: meteorological similarity rather than geographic
                   proximity. Two coastal stations 400 km apart may
                   share more wind behaviour than a coastal–inland
                   pair 100 km apart.

Two important pitfalls handled explicitly:
    * Correlations are computed on TRAINING rows only (architecture
      pitfall #3 — using full data would leak val/test info into the
      graph structure).
    * We correlate raw wind series, NOT power labels — correlating
      labels would partly encode the power-curve formula into the graph.

Inputs:
    data/selected_stations.json  - station metadata (fmisid, lat, lon)
    data/dataset/<fmisid>.csv    - standardised features + split column

Outputs:
    data/graphs/system_a.npz     - adjacency (n*n), fmisids array
    data/graphs/system_b.npz
    data/graphs/_summary.json    - graph statistics
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


K_NEIGHBOURS_DEFAULT: int = 3
MIN_CORR_OVERLAP: int = 100   # require at least this many overlapping train rows for ρ


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_r_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * earth_r_km * math.asin(math.sqrt(a))


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distances (km). Vectorised, returns dense n×n matrix."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# k-NN selection (works for both "smaller-is-closer" and "larger-is-closer")
# ---------------------------------------------------------------------------
def knn_indicator(score_matrix: np.ndarray, k: int, larger_is_closer: bool) -> np.ndarray:
    """
    Return a boolean n×n indicator where row i marks the k nearest neighbours
    of i (excluding self). NOT yet symmetrised.

    For Haversine: larger_is_closer=False (smaller distance = closer).
    For correlation: larger_is_closer=True (higher ρ = more similar).
    """
    n = score_matrix.shape[0]
    indicator = np.zeros((n, n), dtype=bool)
    for i in range(n):
        scores = score_matrix[i].copy()
        scores[i] = -np.inf if larger_is_closer else np.inf   # exclude self
        if larger_is_closer:
            nn = np.argpartition(-scores, k)[:k]
        else:
            nn = np.argpartition(scores, k)[:k]
        indicator[i, nn] = True
    return indicator


def symmetrise_union(indicator: np.ndarray) -> np.ndarray:
    """Symmetrise k-NN indicator: edge {i,j} if either picks the other."""
    return indicator | indicator.T


# ---------------------------------------------------------------------------
# System A — Geographic
# ---------------------------------------------------------------------------
def build_system_a(
    lat: np.ndarray, lon: np.ndarray, k: int
) -> tuple[np.ndarray, dict]:
    """
    Build the geographic k-NN graph with Gaussian-kernel weights.
    Returns (adjacency matrix, info dict).
    """
    D = haversine_matrix(lat, lon)
    edge_mask = symmetrise_union(knn_indicator(D, k, larger_is_closer=False))
    # σ = median of distances on connected pairs (upper triangle, mask=True)
    iu = np.triu_indices_from(edge_mask, k=1)
    connected_dists = D[iu][edge_mask[iu]]
    sigma_km = float(np.median(connected_dists))

    A = np.zeros_like(D)
    W_full = np.exp(-(D ** 2) / (2.0 * sigma_km ** 2))
    A[edge_mask] = W_full[edge_mask]
    np.fill_diagonal(A, 0.0)

    info = {
        "type": "geographic_knn",
        "k": k,
        "weight_kernel": "gaussian",
        "sigma_km": round(sigma_km, 2),
        "distance_unit": "km",
    }
    return A, info


# ---------------------------------------------------------------------------
# System B — Correlation
# ---------------------------------------------------------------------------
def load_train_wind_wide(
    fmisids: list[int], dataset_dir: Path
) -> pd.DataFrame:
    """
    Load each station's training-set wind_speed_ms series and assemble
    into a wide DataFrame (index = timestamps, columns = fmisids).
    Outer join across stations: rows where a station has no training
    sample show up as NaN in that column.
    """
    series_list = []
    for fmisid in fmisids:
        path = dataset_dir / f"{fmisid}.csv"
        df = pd.read_csv(path, parse_dates=["time_utc"])
        train = df[df["split"] == "train"][["time_utc", "wind_speed_ms"]]
        s = train.set_index("time_utc")["wind_speed_ms"]
        s.name = fmisid
        series_list.append(s)
    return pd.concat(series_list, axis=1, join="outer")


def build_system_b(
    fmisids: list[int],
    dataset_dir: Path,
    k: int,
    min_overlap: int,
) -> tuple[np.ndarray, dict]:
    """
    Build the correlation k-NN graph using training-period wind correlations.
    Pearson is invariant to the (already applied) standardisation, so we can
    use the standardised wind column directly.
    """
    wide = load_train_wind_wide(fmisids, dataset_dir)
    n = len(fmisids)
    corr = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            si = wide[fmisids[i]]
            sj = wide[fmisids[j]]
            both = pd.concat([si, sj], axis=1, join="inner").dropna()
            if len(both) < min_overlap:
                corr[i, j] = corr[j, i] = np.nan
            else:
                rho = both.iloc[:, 0].corr(both.iloc[:, 1])
                corr[i, j] = corr[j, i] = float(rho)

    # k-NN by correlation. NaN-pairs (insufficient overlap) treated as -inf
    # so they never get picked as a neighbour.
    corr_safe = np.where(np.isnan(corr), -np.inf, corr)
    edge_mask = symmetrise_union(knn_indicator(corr_safe, k, larger_is_closer=True))

    A = np.zeros((n, n))
    pos = np.clip(corr, 0.0, None)        # max(0, ρ)
    A[edge_mask] = pos[edge_mask]
    A[np.isnan(A)] = 0.0
    np.fill_diagonal(A, 0.0)

    info = {
        "type": "correlation_knn",
        "k": k,
        "weight_kernel": "max(0, pearson_rho)",
        "correlation_target": "wind_speed_ms (training rows only)",
        "min_overlap_rows": min_overlap,
    }
    return A, info


# ---------------------------------------------------------------------------
# Stats and sanity
# ---------------------------------------------------------------------------
def graph_stats(A: np.ndarray) -> dict:
    """Compute descriptive statistics + sanity assertions."""
    n = A.shape[0]
    assert A.shape == (n, n), f"adjacency not square: {A.shape}"
    assert np.allclose(A, A.T), "adjacency not symmetric"
    assert (A >= 0).all(), "adjacency has negative weights"
    assert (np.diag(A) == 0).all(), "adjacency has self-loops"

    upper = A[np.triu_indices_from(A, k=1)]
    edge_mask = upper > 0
    n_edges = int(edge_mask.sum())

    degrees = (A > 0).sum(axis=1)
    weights_on_edges = upper[edge_mask]

    n_comp, _ = connected_components(csr_matrix(A > 0), directed=False)

    return {
        "n_nodes": int(n),
        "n_edges": n_edges,
        "n_components": int(n_comp),
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
        "degree_mean": round(float(degrees.mean()), 2),
        "weight_min": round(float(weights_on_edges.min()), 4) if n_edges else None,
        "weight_max": round(float(weights_on_edges.max()), 4) if n_edges else None,
        "weight_mean": round(float(weights_on_edges.mean()), 4) if n_edges else None,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_graph(
    path: Path, A: np.ndarray, fmisids: list[int], info: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        adjacency=A,
        fmisids=np.array(fmisids, dtype=np.int64),
        description=json.dumps(info),
    )
    print(f"  wrote {path}")


def load_station_metadata(path: Path) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    """Flatten data/selected_stations.json into ordered arrays."""
    selection = json.loads(path.read_text(encoding="utf-8"))
    fmisids: list[int] = []
    lats: list[float] = []
    lons: list[float] = []
    names: list[str] = []
    for region_info in selection.values():
        for s in region_info["stations"]:
            fmisids.append(int(s["fmisid"]))
            lats.append(float(s["lat"]))
            lons.append(float(s["lon"]))
            names.append(s["name"])
    return fmisids, np.array(lats), np.array(lons), names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stations", default="data/selected_stations.json")
    ap.add_argument("--dataset-dir", default="data/dataset")
    ap.add_argument("--out-dir", default="data/graphs")
    ap.add_argument("--k", type=int, default=K_NEIGHBOURS_DEFAULT)
    ap.add_argument("--min-overlap", type=int, default=MIN_CORR_OVERLAP,
                    help="Min overlapping training rows required for ρ_ij. Default: %(default)s")
    args = ap.parse_args()

    fmisids, lats, lons, _names = load_station_metadata(Path(args.stations))
    print(f"Loaded {len(fmisids)} stations from {args.stations}")
    out_dir = Path(args.out_dir)

    # System A
    print(f"\nBuilding System A (geographic k={args.k} NN, Gaussian kernel)...")
    A_geo, info_a = build_system_a(lats, lons, k=args.k)
    stats_a = graph_stats(A_geo)
    info_a.update(stats_a)
    print(f"  edges:      {stats_a['n_edges']}")
    print(f"  components: {stats_a['n_components']}")
    print(f"  σ:          {info_a['sigma_km']:.2f} km")
    print(f"  degree:     min={stats_a['degree_min']}, max={stats_a['degree_max']}, "
          f"mean={stats_a['degree_mean']}")
    print(f"  weights:    [{stats_a['weight_min']:.3f}, {stats_a['weight_max']:.3f}], "
          f"mean={stats_a['weight_mean']:.3f}")

    # System B
    print(f"\nBuilding System B (correlation k={args.k} NN, max(0,ρ) weights)...")
    A_cor, info_b = build_system_b(
        fmisids, Path(args.dataset_dir), k=args.k, min_overlap=args.min_overlap
    )
    stats_b = graph_stats(A_cor)
    info_b.update(stats_b)
    print(f"  edges:      {stats_b['n_edges']}")
    print(f"  components: {stats_b['n_components']}")
    print(f"  degree:     min={stats_b['degree_min']}, max={stats_b['degree_max']}, "
          f"mean={stats_b['degree_mean']}")
    print(f"  weights:    [{stats_b['weight_min']:.3f}, {stats_b['weight_max']:.3f}], "
          f"mean={stats_b['weight_mean']:.3f}")

    # Edge overlap between the two systems — how much do they agree?
    mask_a = (A_geo > 0)
    mask_b = (A_cor > 0)
    edges_a = {tuple(sorted(e)) for e in zip(*np.where(mask_a))}
    edges_b = {tuple(sorted(e)) for e in zip(*np.where(mask_b))}
    overlap = edges_a & edges_b
    print(f"\nEdge overlap A ∩ B: {len(overlap)} edges shared "
          f"out of {len(edges_a)} (A) and {len(edges_b)} (B)")

    # Save artifacts
    print()
    save_graph(out_dir / "system_a.npz", A_geo, fmisids, info_a)
    save_graph(out_dir / "system_b.npz", A_cor, fmisids, info_b)

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "fmisids": fmisids,
                "system_a": info_a,
                "system_b": info_b,
                "edge_overlap_count": len(overlap),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()