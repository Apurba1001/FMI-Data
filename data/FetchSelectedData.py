#!/usr/bin/env python3
"""
Phase 1, Step 1.2: full time-series fetch for the 25 selected stations.

Reads data/selected_stations.json, queries FMI multipointcoverage in
chunks, filters at the FMI side via explicit fmisid arguments, pivots
to wide format, and writes one CSV per station to
data/stations/<fmisid>.csv.

Output schema (per-station CSV, one row per hourly timestamp):
    time_utc, temperature_c, pressure_hpa, humidity_pct,
    wind_speed_ms, wind_dir_deg

Missing values are kept as empty cells; the Phase 2 feature-engineering
step is responsible for drop/fill logic (architecture doc Step 1.3).

WHY fmisid INSTEAD OF bbox:
    A bbox query returns every station inside Finland (~200) for the
    full chunk window. For 7-day chunks × ~13 parameters × hourly, the
    response is large enough that FMI closes the connection without
    responding (RemoteDisconnected). Filtering by fmisid at the query
    level means FMI only assembles data for our 25 stations — ~8× less
    work for FMI, and reliably under their response limits.

Usage:
    uv run python data/FetchSelectedData.py \
        --stations data/selected_stations.json \
        --start 2023-01-01 \
        --end   2025-01-01 \
        --out-dir data/stations
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from fmiopendata.wfs import download_stored_query


# FMI parameter name (as returned by multipointcoverage) -> our column name.
NEEDED_PARAMS: dict[str, str] = {
    "Air temperature": "temperature_c",
    "Pressure (msl)": "pressure_hpa",
    "Relative humidity": "humidity_pct",
    "Wind speed": "wind_speed_ms",
    "Wind direction": "wind_dir_deg",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def iso_z(t: dt.datetime) -> str:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    t = t.astimezone(dt.timezone.utc)
    return t.isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_chunks(
    start: dt.datetime, end: dt.datetime, days: int
) -> Iterable[tuple[dt.datetime, dt.datetime]]:
    cur = start
    delta = dt.timedelta(days=days)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt


def load_selected_fmisids(path: Path) -> list[int]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    out: list[int] = []
    for region_info in selection.values():
        for s in region_info["stations"]:
            out.append(int(s["fmisid"]))
    return out


# ---------------------------------------------------------------------------
# FMI query (one chunk, with retry)
# ---------------------------------------------------------------------------
def query_chunk(
    fmisids: list[int],
    start: dt.datetime,
    end: dt.datetime,
    retries: int,
    backoff_s: float,
) -> list[dict]:
    """
    Query FMI for one time chunk, filtered to selected stations via
    explicit fmisid arguments. Retries connection errors with exponential
    backoff.
    """
    args = (
        [f"fmisid={fmisid}" for fmisid in fmisids]
        + [
            f"starttime={iso_z(start)}",
            f"endtime={iso_z(end)}",
            "timestep=60",
            "timeseries=True",
        ]
    )

    obs = None
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            obs = download_stored_query(
                "fmi::observations::weather::multipointcoverage",
                args=args,
            )
            last_exc = None
            break
        except Exception as e:                       # noqa: BLE001
            last_exc = e
            if attempt < retries:
                wait = backoff_s * (2 ** (attempt - 1))
                print(f" (retry {attempt}/{retries} in {wait:.0f}s)",
                      end="", flush=True)
                time.sleep(wait)
    if obs is None:
        assert last_exc is not None
        raise last_exc

    fmisid_set = set(fmisids)  # defensive: filter again in case FMI returns extras
    meta = obs.location_metadata

    rows: list[dict] = []
    for station_name, station_data in obs.data.items():
        m = meta.get(station_name, {})
        fmisid = m.get("fmisid")
        if fmisid is None or int(fmisid) not in fmisid_set:
            continue

        times = station_data.get("times", [])
        if not times:
            continue

        for param, payload in station_data.items():
            if param == "times":
                continue
            if param not in NEEDED_PARAMS:
                continue
            values = payload.get("values", [])
            n = min(len(times), len(values))
            col = NEEDED_PARAMS[param]
            for i in range(n):
                rows.append(
                    {
                        "fmisid": int(fmisid),
                        "time_utc": times[i],
                        "param": col,
                        "value": values[i],
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Pivot + write
# ---------------------------------------------------------------------------
def pivot_and_save(long_df: pd.DataFrame, out_dir: Path) -> dict[int, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_cols = list(NEEDED_PARAMS.values())

    summary: dict[int, dict] = {}
    for fmisid, group in long_df.groupby("fmisid", sort=True):
        wide = group.pivot_table(
            index="time_utc",
            columns="param",
            values="value",
            aggfunc="first",
        )
        for col in expected_cols:
            if col not in wide.columns:
                wide[col] = float("nan")
        wide = wide[expected_cols].sort_index().reset_index()

        out_path = out_dir / f"{fmisid}.csv"
        wide.to_csv(out_path, index=False)

        n_rows = len(wide)
        n_complete = wide[expected_cols].dropna().shape[0]
        completeness = (n_complete / n_rows * 100.0) if n_rows else 0.0
        summary[int(fmisid)] = {
            "rows": n_rows,
            "complete_rows": n_complete,
            "completeness_pct": round(completeness, 1),
            "path": str(out_path),
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stations", default="data/selected_stations.json")
    ap.add_argument("--start", default="2023-01-01",
                    help="Inclusive start date (YYYY-MM-DD). Default: %(default)s")
    ap.add_argument("--end", default="2025-01-01",
                    help="Exclusive end date (YYYY-MM-DD). Default: %(default)s")
    ap.add_argument("--chunk-days", type=int, default=7,
                    help="Chunk size in days. Default: %(default)s")
    ap.add_argument("--out-dir", default="data/stations",
                    help="Where to write per-station CSVs. Default: %(default)s")
    ap.add_argument("--retries", type=int, default=3,
                    help="Retry attempts per chunk on connection errors. Default: %(default)s")
    ap.add_argument("--backoff", type=float, default=2.0,
                    help="Initial backoff seconds (doubles each retry). Default: %(default)s")
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end).replace(tzinfo=dt.timezone.utc)
    if end <= start:
        raise SystemExit("--end must be strictly after --start")

    fmisids = load_selected_fmisids(Path(args.stations))
    print(f"Loaded {len(fmisids)} selected FMISIDs from {args.stations}")
    print(f"  {sorted(fmisids)}")
    print(f"Time range: {iso_z(start)} -> {iso_z(end)}  (end exclusive)")

    chunks = list(iter_chunks(start, end, args.chunk_days))
    print(f"Chunks: {len(chunks)} × {args.chunk_days} day(s)")
    print(f"Filtering at FMI via fmisid args (not bbox)\n")

    all_rows: list[dict] = []
    failed: list[tuple[str, str, str]] = []

    for i, (cs, ce) in enumerate(chunks, 1):
        print(f"[{i:>3}/{len(chunks)}] {iso_z(cs)} -> {iso_z(ce)}",
              end="  ", flush=True)
        try:
            rows = query_chunk(fmisids, cs, ce, args.retries, args.backoff)
        except Exception as e:                       # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append((iso_z(cs), iso_z(ce), str(e)))
            continue
        print(f"{len(rows):>6} rows")
        all_rows.extend(rows)

    if failed:
        print(f"\n⚠ {len(failed)} chunk(s) failed:")
        for cs, ce, msg in failed:
            print(f"  {cs} -> {ce}: {msg}")

    if not all_rows:
        print("No data fetched at all. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal long-format rows: {len(all_rows):,}")

    long_df = pd.DataFrame(all_rows)
    long_df["time_utc"] = pd.to_datetime(long_df["time_utc"], utc=True)

    got = set(long_df["fmisid"].unique().tolist())
    missing = set(fmisids) - got
    if missing:
        print(f"⚠ No data for FMISIDs: {sorted(missing)}")
    else:
        print(f"All {len(fmisids)} stations have at least some data ✓")

    print(f"\nPivoting and writing per-station CSVs to {args.out_dir}/ ...")
    summary = pivot_and_save(long_df, Path(args.out_dir))

    print("\nPer-station summary:")
    print(f"  {'fmisid':>7}  {'rows':>7}  {'complete':>8}  {'%':>6}")
    for fmisid in sorted(summary):
        s = summary[fmisid]
        print(f"  {fmisid:>7}  {s['rows']:>7}  {s['complete_rows']:>8}  "
              f"{s['completeness_pct']:>5.1f}%")

    out_dir = Path(args.out_dir)
    (out_dir / "_fetch_summary.json").write_text(
        json.dumps(
            {
                "start": iso_z(start),
                "end": iso_z(end),
                "chunk_days": args.chunk_days,
                "fmisids": sorted(fmisids),
                "stations": summary,
                "failed_chunks": [
                    {"start": cs, "end": ce, "error": err}
                    for cs, ce, err in failed
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir / '_fetch_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()