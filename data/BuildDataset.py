#!/usr/bin/env python3
"""
Phase 2: build the ML dataset from per-station raw CSVs.

For each station this script:
  1. Reindexes to the full hourly grid (gaps become explicit NaN rows).
  2. Forward-fills non-wind features for gaps <= 3 hours (per Step 1.3).
     Wind speed is never filled — its rows just get dropped later.
  3. Extrapolates 10m wind to hub height (Vestas V162 -> 100 m) using
     a project-wide shear exponent alpha = 0.14 (coastal default).
  4. Computes air density from temperature alone, using standard
     sea-level pressure P0 = 101325 Pa. Pressure is dropped project-wide
     because 2 of our 25 stations don't report it; documented in the
     report as a deliberate simplification (cost ~1% accuracy in rho).
  5. Applies the Vestas V162-6.0MW power curve -> label power_kw.
  6. Builds the 10-dim feature vector at time t:
       [temp, humidity, sin(wind_dir), cos(wind_dir), wind_speed_10m,
        icing_flag, sin(hour), cos(hour), sin(doy), cos(doy)]
     Note: feature uses RAW 10m wind speed; hub-height extrapolation is
     only used to compute the label (per architecture pitfall #8).
  7. Time-shifts: features at t paired with power_kw at t+1.
  8. Drops pairs where any feature or label is NaN.

Global pieces:
  * Chronological split: calendar-date boundaries shared across all
    stations (train 70% / val 15% / test 15%).
  * StandardScaler fitted on the concatenated training data from all
    stations, applied to every station's full X. This keeps feature
    semantics consistent across nodes, which is what makes the GTVMin
    graph penalty ||w(i) - w(j)||^2 meaningful.

Output layout:
  data/dataset/<fmisid>.csv   - one row per valid (X, y) pair:
       columns: time_utc, <10 standardized feature columns>,
                power_kw, split
  data/dataset/_scaler.json   - per-feature global mean and std
  data/dataset/_meta.json     - all preprocessing parameters

Usage:
    uv run python data/BuildDataset.py \
        --stations-dir data/stations \
        --out-dir data/dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
P0_PA: float = 101_325.0          # standard sea-level pressure
R_D: float = 287.05               # specific gas constant for dry air
T_KELVIN_OFFSET: float = 273.15
RHO_REF: float = 1.225            # reference air density (kg/m^3)
HUB_HEIGHT_M: float = 100.0       # Vestas V162-6.0MW typical hub height
REF_HEIGHT_M: float = 10.0        # FMI 10m wind measurement height

# Vestas V162-6.0MW manufacturer specs
V_CIN_MS: float = 3.0
V_RATED_MS: float = 13.0
V_COUT_MS: float = 25.0
P_RATED_KW: float = 6000.0

# Project-wide shear exponent. Documented simplification — see report.
DEFAULT_SHEAR_ALPHA: float = 0.14

# Train/val split fractions of total calendar range (test = remainder).
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15

# Feature column names (order is the feature vector index).
FEATURE_NAMES: list[str] = [
    "temperature_c",
    "humidity_pct",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_speed_ms",
    "icing_flag",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def hub_height_wind(v_10m: np.ndarray, alpha: float) -> np.ndarray:
    """Power-law extrapolation from REF_HEIGHT to HUB_HEIGHT."""
    return v_10m * (HUB_HEIGHT_M / REF_HEIGHT_M) ** alpha


def air_density(temp_c: np.ndarray) -> np.ndarray:
    """
    Ideal-gas air density at standard pressure.

    rho(t) = P0 / (R_d * T(t))

    We use standard sea-level pressure because 2/25 stations don't
    report pressure; using a constant across all stations keeps the
    label formula uniform. Temperature variation accounts for the
    dominant ~4% of air-density variation in temperate climates; the
    residual ~1-2% from pressure is documented as ignored.
    """
    return P0_PA / (R_D * (temp_c + T_KELVIN_OFFSET))


def vestas_v162_power(v_hub: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Vestas V162-6.0MW power curve, vectorised.

    Below cut-in or above cut-out: 0.
    Cubic ramp between cut-in and rated wind speed (scaled by air
    density relative to 1.225 kg/m^3).
    Rated power between rated and cut-out (also density-scaled).
    """
    p = np.zeros_like(v_hub, dtype=float)

    ramp = (v_hub >= V_CIN_MS) & (v_hub < V_RATED_MS)
    rated = (v_hub >= V_RATED_MS) & (v_hub <= V_COUT_MS)

    cubic_num = v_hub[ramp] ** 3 - V_CIN_MS ** 3
    cubic_den = V_RATED_MS ** 3 - V_CIN_MS ** 3
    p[ramp] = P_RATED_KW * (rho[ramp] / RHO_REF) * (cubic_num / cubic_den)
    p[rated] = P_RATED_KW * (rho[rated] / RHO_REF)

    # NaN propagation: if v_hub or rho is NaN, result is NaN
    invalid = np.isnan(v_hub) | np.isnan(rho)
    p[invalid] = np.nan
    return p


# ---------------------------------------------------------------------------
# Per-station processing
# ---------------------------------------------------------------------------
def process_station(
    raw_csv: Path,
    shear_alpha: float = DEFAULT_SHEAR_ALPHA,
) -> pd.DataFrame:
    """
    Read one station's raw CSV and produce a tidy frame with columns:
        time_utc, <FEATURE_NAMES>, power_kw

    The returned frame contains only the valid (X, y) time-shifted pairs
    (no NaN in any feature or label). It is NOT yet standardised; split
    assignment and scaling happen at the global level in main().
    """
    df = pd.read_csv(raw_csv, parse_dates=["time_utc"])
    df = df.set_index("time_utc").sort_index()

    # Reindex to a contiguous hourly grid so the t -> t+1 shift is
    # semantically a one-hour-ahead forecast (not "next non-NaN row").
    start = df.index.min().floor("h")
    end = df.index.max().ceil("h")
    full_index = pd.date_range(start, end, freq="h", tz="UTC")
    df = df.reindex(full_index)

    # Forward-fill non-wind features for short gaps (architecture Step 1.3).
    # Wind speed is intentionally NOT filled — its NaNs propagate to
    # invalid rows, which we drop below.
    fill_cols = ["temperature_c", "humidity_pct", "wind_dir_deg"]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].ffill(limit=3)

    # Compute label series on the full grid.
    v_10m = df["wind_speed_ms"].to_numpy(dtype=float)
    temp_c = df["temperature_c"].to_numpy(dtype=float)
    rho = air_density(temp_c)
    v_hub = hub_height_wind(v_10m, shear_alpha)
    power_kw = vestas_v162_power(v_hub, rho)

    # Build feature matrix on the full grid.
    rh = df["humidity_pct"].to_numpy(dtype=float)
    wind_dir_rad = np.deg2rad(df["wind_dir_deg"].to_numpy(dtype=float))
    # icing_flag is well-defined only where temp and humidity are present
    icing_flag = (
        (temp_c >= -5.0) & (temp_c <= 1.0) & (rh > 95.0)
    ).astype(float)
    # but if either temp or rh is NaN we should not pretend icing=0
    icing_flag[np.isnan(temp_c) | np.isnan(rh)] = np.nan

    hour = df.index.hour.to_numpy()
    doy = df.index.dayofyear.to_numpy()

    feat = np.column_stack(
        [
            temp_c,
            rh,
            np.sin(wind_dir_rad),
            np.cos(wind_dir_rad),
            v_10m,                                  # raw 10m, not v_hub (pitfall #8)
            icing_flag,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * doy / 365.0),
            np.cos(2 * np.pi * doy / 365.0),
        ]
    )

    # Time shift: features at t (rows 0..N-2) paired with power at t+1 (rows 1..N-1).
    X_full = feat[:-1, :]
    y_full = power_kw[1:]
    times = full_index[:-1]

    # Drop rows where any feature or the label is NaN.
    valid = (~np.isnan(X_full).any(axis=1)) & (~np.isnan(y_full))

    out = pd.DataFrame(X_full[valid], columns=FEATURE_NAMES)
    out.insert(0, "time_utc", times[valid])
    out["power_kw"] = y_full[valid]
    return out


# ---------------------------------------------------------------------------
# Splits and scaling
# ---------------------------------------------------------------------------
def compute_split_boundaries(
    times: pd.DatetimeIndex,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Convert (train_frac, val_frac) into absolute timestamp cutoffs based
    on the global calendar range. Same boundaries are then applied to
    every station.
    """
    t_min = times.min()
    t_max = times.max()
    span = t_max - t_min
    train_end = t_min + span * train_frac
    val_end = t_min + span * (train_frac + val_frac)
    return train_end, val_end


def assign_split(
    times: pd.Series, train_end: pd.Timestamp, val_end: pd.Timestamp
) -> pd.Series:
    """Map each timestamp to 'train' / 'val' / 'test' based on cutoffs."""
    s = pd.Series(index=times.index, dtype="object")
    s[times < train_end] = "train"
    s[(times >= train_end) & (times < val_end)] = "val"
    s[times >= val_end] = "test"
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stations-dir", default="data/stations",
                    help="Per-station raw CSV directory. Default: %(default)s")
    ap.add_argument("--out-dir", default="data/dataset",
                    help="Where to write processed per-station CSVs. Default: %(default)s")
    ap.add_argument("--shear-alpha", type=float, default=DEFAULT_SHEAR_ALPHA,
                    help="Wind-shear exponent (project-wide). Default: %(default)s")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC,
                    help="Fraction of calendar range for training. Default: %(default)s")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC,
                    help="Fraction for validation (rest is test). Default: %(default)s")
    args = ap.parse_args()

    stations_dir = Path(args.stations_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all per-station CSVs (anything matching <digits>.csv)
    raw_paths = sorted(
        p for p in stations_dir.glob("*.csv")
        if p.stem.isdigit()
    )
    if not raw_paths:
        raise SystemExit(f"No per-station CSVs found in {stations_dir}")
    print(f"Found {len(raw_paths)} station CSVs in {stations_dir}\n")

    # ---- pass 1: feature engineering per station -------------------------
    per_station: dict[int, pd.DataFrame] = {}
    for p in raw_paths:
        fmisid = int(p.stem)
        df = process_station(p, shear_alpha=args.shear_alpha)
        per_station[fmisid] = df
        print(f"  [{fmisid}] valid pairs: {len(df):>5}  "
              f"(power range {df['power_kw'].min():>6.0f} .. "
              f"{df['power_kw'].max():>6.0f} kW)")

    # ---- global split boundaries from union of timestamps -----------------
    all_times = pd.concat([d["time_utc"] for d in per_station.values()])
    train_end, val_end = compute_split_boundaries(
        pd.DatetimeIndex(all_times), args.train_frac, args.val_frac
    )
    print(f"\nGlobal split boundaries:")
    print(f"  train: < {train_end}")
    print(f"  val:   [{train_end}, {val_end})")
    print(f"  test:  >= {val_end}")

    # Assign split labels per station.
    for fmisid, d in per_station.items():
        d["split"] = assign_split(d["time_utc"], train_end, val_end)

    # ---- global scaler fit on training rows from all stations -------------
    train_X = np.vstack(
        [d.loc[d["split"] == "train", FEATURE_NAMES].to_numpy()
         for d in per_station.values()]
    )
    mu = train_X.mean(axis=0)
    sigma = train_X.std(axis=0, ddof=0)
    # Defensive: any feature with zero variance gets sigma=1 to avoid div0.
    zero_var = sigma < 1e-12
    if zero_var.any():
        affected = [FEATURE_NAMES[i] for i, z in enumerate(zero_var) if z]
        print(f"\n⚠ Zero-variance features in training set (sigma=1 used): {affected}")
        sigma[zero_var] = 1.0

    print(f"\nGlobal scaler (fit on {len(train_X):,} training rows from "
          f"{len(per_station)} stations):")
    for name, m, s in zip(FEATURE_NAMES, mu, sigma):
        print(f"  {name:<18s} mean={m:>10.4f}  std={s:>10.4f}")

    # ---- pass 2: apply scaler, write per-station CSVs ---------------------
    print()
    split_counts_total = {"train": 0, "val": 0, "test": 0}
    for fmisid, d in per_station.items():
        X = d[FEATURE_NAMES].to_numpy()
        Xs = (X - mu) / sigma
        out = d.copy()
        out[FEATURE_NAMES] = Xs

        out_path = out_dir / f"{fmisid}.csv"
        out.to_csv(out_path, index=False)

        counts = out["split"].value_counts().to_dict()
        for k in split_counts_total:
            split_counts_total[k] += counts.get(k, 0)
        print(f"  [{fmisid}] train={counts.get('train', 0):>5}  "
              f"val={counts.get('val', 0):>5}  test={counts.get('test', 0):>5}  "
              f"-> {out_path}")

    print(f"\nTotals across all stations: "
          f"train={split_counts_total['train']:,}  "
          f"val={split_counts_total['val']:,}  "
          f"test={split_counts_total['test']:,}")

    # ---- sidecar metadata -------------------------------------------------
    scaler_path = out_dir / "_scaler.json"
    scaler_path.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "mean": mu.tolist(),
                "std": sigma.tolist(),
                "fit_on_n_rows": int(len(train_X)),
                "fit_on_n_stations": len(per_station),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta_path = out_dir / "_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "shear_alpha": args.shear_alpha,
                "hub_height_m": HUB_HEIGHT_M,
                "ref_height_m": REF_HEIGHT_M,
                "turbine_model": "Vestas V162-6.0MW",
                "turbine_params": {
                    "v_cin": V_CIN_MS,
                    "v_rated": V_RATED_MS,
                    "v_cout": V_COUT_MS,
                    "p_rated_kw": P_RATED_KW,
                },
                "air_density": {
                    "model": "ideal-gas, standard pressure (no pressure feature)",
                    "P0_Pa": P0_PA,
                    "R_d": R_D,
                },
                "feature_names": FEATURE_NAMES,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "train_end_utc": str(train_end),
                "val_end_utc": str(val_end),
                "split_total_rows": split_counts_total,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {scaler_path}")
    print(f"Wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()