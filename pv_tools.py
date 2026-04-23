"""
pv_tools.py -- PV-sizing computational tools
=============================================

Self-contained toolbox that the pipeline calls *before* the LLM to
pre-compute numeric results the model can reference.  Every function
reads only from the ``data/`` directory and returns plain dicts / lists
so the caller can serialise them into prompt context.

Tool inventory
--------------
4A  load_household_profile_from_eia   – real 8760-h load profile (kW)
4B  build_synthetic_load_profile      – fallback synthetic 8760-h profile
4C  irradiance_shape_factor           – hourly G(t)/G_ref via sine model
4D  fetch_irradiance_annual           – mean annual GHI (kWh/m²/yr)
4E  build_hourly_tariffs              – 8760 hourly TOU prices ($/kWh)
4F  build_hourly_pv_output            – 8760-h AC PV output (kW)
4G  select_panel / select_battery     – hardware selection helpers
4H  run_dispatch_simulation           – rule-based 8760-h dispatch
4I  compute_economics                 – 10-year NPV financial model
4J  run_all_tools                     – orchestrator that runs 4A–4I
"""

from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
#  DATA PATHS
# ═══════════════════════════════════════════════════════════════

_DATA_DIR = Path(__file__).resolve().parent / "data"

EIA_LOAD_PATH = _DATA_DIR / "San_Diego_Load_EIA_Fixed.csv"
TOU_DR_PATH   = _DATA_DIR / "tou_dr_daily_2021_2025.csv"
TOU_DR1_PATH  = _DATA_DIR / "tou_dr1_daily_2021_2025.csv"
TOU_DR2_PATH  = _DATA_DIR / "tou_dr2_daily_2021_2025.csv"

# Fallback: EIA file may also live beside household_generator.py
_ALT_EIA_PATH = Path(__file__).resolve().parent / "data_extraction" / "San_Diego_Load_EIA_Fixed.csv"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# ═══════════════════════════════════════════════════════════════
#  PHYSICAL / FINANCIAL CONSTANTS
# ═══════════════════════════════════════════════════════════════

SDGE_TOTAL_CUSTOMERS     = 1_040_149
COASTAL_LON_REF          = -117.25
CITY_CENTER_LAT          = 32.7157
CITY_CENTER_LON          = -117.1611

G_REF_W_M2               = 1000.0
PR_PERFORMANCE_RATIO     = 0.80
INSTALLATION_COST_RATE   = 0.10
FEDERAL_ITC_RATE         = 0.30
UTILITY_INFLATION_RATE   = 0.06
DISCOUNT_RATE            = 0.07
O_AND_M_COST_PER_W_YR   = 0.005
NEM_EXPORT_CREDIT        = 0.10
INVERTER_REPLACEMENT_USD = 2000.0
INVERTER_REPLACEMENT_YR  = 10
ANALYSIS_YEARS           = 10
SDGE_DAILY_FIXED_FEE     = 0.345

EV_CHARGER_POWER_KW      = 7.2
EV_DAILY_ENERGY_KWH      = 14.0
EV_CHARGE_START_HOUR      = 22
EV_CHARGE_END_HOUR        = 6

# ═══════════════════════════════════════════════════════════════
#  EQUIPMENT CATALOGS
# ═══════════════════════════════════════════════════════════════

SOLAR_PANEL_CATALOG: List[Dict[str, Any]] = [
    # length_m × width_m are physical dimensions (portrait orientation, longer side first).
    # cells_in_series × cells_in_parallel = total cells per panel.
    # Standard mono/poly panels: all cells in series (cells_in_parallel = 1).
    {"manufacturer": "REC Group",      "model": "Alpha Pure",  "efficiency_percent": 22.6, "cost_per_wp_usd": 2.85, "temp_coeff_pct_per_c": -0.26, "panel_power_w": 405, "area_m2": 1.79, "length_m": 1.730, "width_m": 1.034, "cells": 60, "cells_in_series": 60, "cells_in_parallel": 1, "degradation_rate": 0.005},
    {"manufacturer": "JA Solar",       "model": "DeepBlue",    "efficiency_percent": 21.5, "cost_per_wp_usd": 2.90, "temp_coeff_pct_per_c": -0.35, "panel_power_w": 395, "area_m2": 1.84, "length_m": 1.769, "width_m": 1.040, "cells": 60, "cells_in_series": 60, "cells_in_parallel": 1, "degradation_rate": 0.006},
    {"manufacturer": "Trina Solar",    "model": "Vertex S",    "efficiency_percent": 21.8, "cost_per_wp_usd": 2.90, "temp_coeff_pct_per_c": -0.34, "panel_power_w": 400, "area_m2": 1.83, "length_m": 1.762, "width_m": 1.039, "cells": 60, "cells_in_series": 60, "cells_in_parallel": 1, "degradation_rate": 0.006},
    {"manufacturer": "Canadian Solar", "model": "TOPHiKu7",    "efficiency_percent": 22.5, "cost_per_wp_usd": 3.10, "temp_coeff_pct_per_c": -0.30, "panel_power_w": 420, "area_m2": 1.87, "length_m": 1.780, "width_m": 1.051, "cells": 66, "cells_in_series": 66, "cells_in_parallel": 1, "degradation_rate": 0.005},
    {"manufacturer": "Silfab Solar",   "model": "Prime",       "efficiency_percent": 22.1, "cost_per_wp_usd": 3.15, "temp_coeff_pct_per_c": -0.33, "panel_power_w": 410, "area_m2": 1.85, "length_m": 1.776, "width_m": 1.041, "cells": 60, "cells_in_series": 60, "cells_in_parallel": 1, "degradation_rate": 0.005},
    {"manufacturer": "Jinko Solar",    "model": "Tiger Neo",   "efficiency_percent": 23.8, "cost_per_wp_usd": 3.20, "temp_coeff_pct_per_c": -0.29, "panel_power_w": 440, "area_m2": 1.85, "length_m": 1.762, "width_m": 1.050, "cells": 66, "cells_in_series": 66, "cells_in_parallel": 1, "degradation_rate": 0.005},
    {"manufacturer": "LONGi Solar",    "model": "Hi-MO 6",     "efficiency_percent": 23.3, "cost_per_wp_usd": 3.35, "temp_coeff_pct_per_c": -0.29, "panel_power_w": 435, "area_m2": 1.87, "length_m": 1.797, "width_m": 1.040, "cells": 66, "cells_in_series": 66, "cells_in_parallel": 1, "degradation_rate": 0.005},
    {"manufacturer": "Maxeon Solar",   "model": "Maxeon 7",    "efficiency_percent": 22.8, "cost_per_wp_usd": 3.50, "temp_coeff_pct_per_c": -0.27, "panel_power_w": 430, "area_m2": 1.89, "length_m": 1.812, "width_m": 1.044, "cells": 66, "cells_in_series": 66, "cells_in_parallel": 1, "degradation_rate": 0.004},
    {"manufacturer": "Aiko Solar",     "model": "Neostar 2P",  "efficiency_percent": 24.3, "cost_per_wp_usd": 3.75, "temp_coeff_pct_per_c": -0.24, "panel_power_w": 460, "area_m2": 1.89, "length_m": 1.812, "width_m": 1.044, "cells": 66, "cells_in_series": 66, "cells_in_parallel": 1, "degradation_rate": 0.004},
]

BATTERY_CATALOG: List[Dict[str, Any]] = [
    {"manufacturer": "Tesla",     "model": "Powerwall 3",       "usable_capacity_kwh": 13.5, "max_charge_power_kw": 11.5, "max_discharge_power_kw": 11.5, "round_trip_efficiency_pct": 97.5, "cycle_life": 4000, "cost_usd": 11500, "degradation_rate": 0.010},
    {"manufacturer": "Enphase",   "model": "IQ Battery 5P",     "usable_capacity_kwh":  5.0, "max_charge_power_kw":  3.84,"max_discharge_power_kw":  3.84,"round_trip_efficiency_pct": 96.0, "cycle_life": 4000, "cost_usd":  6000, "degradation_rate": 0.012},
    {"manufacturer": "Generac",   "model": "PWRcell M6",        "usable_capacity_kwh":  9.0, "max_charge_power_kw":  6.7, "max_discharge_power_kw":  6.7, "round_trip_efficiency_pct": 96.5, "cycle_life": 3500, "cost_usd": 10000, "degradation_rate": 0.012},
    {"manufacturer": "SolarEdge", "model": "Home Battery 48V",  "usable_capacity_kwh":  9.7, "max_charge_power_kw":  5.0, "max_discharge_power_kw":  5.0, "round_trip_efficiency_pct": 94.5, "cycle_life": 6000, "cost_usd":  9500, "degradation_rate": 0.010},
    {"manufacturer": "Panasonic", "model": "EverVolt H Series", "usable_capacity_kwh": 17.1, "max_charge_power_kw":  7.6, "max_discharge_power_kw":  7.6, "round_trip_efficiency_pct": 97.0, "cycle_life": 6000, "cost_usd": 15000, "degradation_rate": 0.010},
]


# ═══════════════════════════════════════════════════════════════
#  4A  EIA LOAD LOADER
# ═══════════════════════════════════════════════════════════════

def load_household_profile_from_eia(
    latitude: float,
    longitude: float,
    annual_kwh_override: float | None = None,
    num_evs: int = 0,
    num_people: int = 3,
    num_daytime_occupants: int = 1,
) -> tuple[list[float], float]:
    """
    Build a real 8760-h household load profile (kW) from EIA regional data.

    Method
    ------
    1. Load San_Diego_Load_EIA_Fixed.csv (hourly MW UTC).
       Localise to America/Los_Angeles.
       Downscale: avg_kw = MW_Load * 1000 / SDGE_TOTAL_CUSTOMERS.

    2. Average across ALL full years (>= 8000 rows) by hour-of-year position.

    3. Apply 9 location-based variability factors + EV + daytime occupants.

    4. Add +/-3 %% hourly noise (deterministic from lat/lon seed).

    5. Rescale to annual_kwh_override if supplied.

    Returns
    -------
    (hourly_kw: list[float], annual_kwh: float)
    """
    eia_path = EIA_LOAD_PATH if EIA_LOAD_PATH.is_file() else _ALT_EIA_PATH
    if not eia_path.is_file():
        raise FileNotFoundError(
            f"EIA file not found at {EIA_LOAD_PATH} or {_ALT_EIA_PATH}"
        )

    df = pd.read_csv(eia_path)
    df["dt_utc"]   = pd.to_datetime(df["Timestamp_UTC"])
    df["dt_local"] = df["dt_utc"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
    df["kw"]       = df["MW_Load"] * 1000.0 / SDGE_TOTAL_CUSTOMERS
    df["year"]     = df["dt_local"].dt.year
    df["doy"]      = df["dt_local"].dt.dayofyear
    df["hour"]     = df["dt_local"].dt.hour
    df["hoy"]      = ((df["doy"] - 1) * 24 + df["hour"]).clip(0, 8759)

    full_years = df["year"].value_counts()
    full_years = full_years[full_years >= 8000].index.tolist()
    df_full    = df[df["year"].isin(full_years)]
    avg_hoy    = (
        df_full.groupby("hoy")["kw"]
        .mean()
        .reindex(range(8760))
        .interpolate(method="linear")
        .ffill().bfill()
    )
    profile = avg_hoy.values.copy()

    # Variability factors
    loc_seed = int(hashlib.sha256(f"{latitude}_{longitude}".encode()).hexdigest(), 16) % (2**32)
    rng_hh  = np.random.RandomState(loc_seed)
    rng_sol = np.random.RandomState(loc_seed + 1000)
    rng_mg  = np.random.RandomState(loc_seed + 3000)

    dc = longitude - COASTAL_LON_REF
    if   dc >= 0.15:  f1 = 1.25
    elif dc >= 0.10:  f1 = 1.05 + (dc - 0.10) * 4.0
    elif dc >= 0:     f1 = 0.95 + dc * 1.0
    elif dc >= -0.05: f1 = 0.90 + (dc + 0.05) * 1.0
    else:             f1 = 0.85

    if   latitude < 32.60: f2 = 1.10
    elif latitude < 32.70: f2 = 1.05
    elif latitude < 32.85: f2 = 1.00
    elif latitude < 32.95: f2 = 0.95
    else:                  f2 = 0.90

    f3 = 1.0 + (max(0, longitude - COASTAL_LON_REF) + max(0, latitude - 32.70) * 2.0) * 0.15
    f4 = 0.70 + 0.10 * min(num_people, 6)

    dist_c = math.sqrt((latitude - CITY_CENTER_LAT)**2 + (longitude - CITY_CENTER_LON)**2)
    if   dist_c < 0.03: f5 = 0.7
    elif dist_c < 0.08: f5 = 0.9
    elif dist_c < 0.15: f5 = 1.1
    else:               f5 = 1.3

    is_coastal    = longitude < -117.20
    is_north      = latitude  >  32.80
    is_urban_core = dist_c    <   0.05
    if   is_coastal and is_north: f6 = 1.15
    elif is_coastal:              f6 = 1.05
    elif is_urban_core:           f6 = 1.25
    elif longitude > -117.00:     f6 = 0.95
    else:                         f6 = 1.10

    is_south = latitude < 32.75
    is_urban = dist_c   < 0.10
    mg_prob  = 0.25 if (is_south and is_urban) else (0.15 if is_urban else 0.10)
    f9 = rng_mg.uniform(1.20, 1.50) if rng_mg.random() < mg_prob else 1.0

    profile *= f1 * f2 * f3 * f4 * f5 * f6 * f9

    # F7: Solar adoption
    sol_prob = (0.35 if (is_coastal and is_north)
                else 0.20 if (is_coastal or longitude > -117.00)
                else 0.05 if is_urban_core else 0.15)
    if rng_sol.random() < sol_prob:
        hrs       = np.arange(8760) % 24
        intensity = np.clip(np.sin((hrs - 6) * np.pi / 12), 0, 1)
        intensity[(hrs < 6) | (hrs > 18)] = 0
        max_red   = rng_sol.uniform(0.4, 0.7)
        profile  *= 1.0 - intensity * (1.0 - max_red)

    # F8: EV charging
    if num_evs > 0:
        hrs = np.arange(8760) % 24
        hours_needed = int(np.ceil((EV_DAILY_ENERGY_KWH * num_evs) / EV_CHARGER_POWER_KW))
        end_h = (EV_CHARGE_START_HOUR + hours_needed) % 24
        if EV_CHARGE_START_HOUR >= end_h:
            is_chg = (hrs >= EV_CHARGE_START_HOUR) | (hrs < end_h)
        else:
            is_chg = (hrs >= EV_CHARGE_START_HOUR) & (hrs < end_h)
        profile += np.where(is_chg, EV_CHARGER_POWER_KW * num_evs, 0.0)

    # F10: Daytime occupants
    if num_daytime_occupants > 0:
        hrs = np.arange(8760) % 24
        daytime_mask = (hrs >= 9) & (hrs < 17)
        occupancy_multiplier = 1.0 + (0.05 * num_daytime_occupants)
        profile = np.where(daytime_mask, profile * occupancy_multiplier, profile)

    # Noise
    profile *= np.random.RandomState(loc_seed).normal(1.0, 0.03, size=8760)
    profile  = np.clip(profile, 0, None)

    # Rescale
    prof_sum = float(profile.sum())
    if annual_kwh_override is not None and annual_kwh_override > 0 and prof_sum > 0:
        profile *= annual_kwh_override / prof_sum
        ann_kwh  = annual_kwh_override
    else:
        ann_kwh = round(prof_sum, 1)

    return profile.tolist(), ann_kwh


# ═══════════════════════════════════════════════════════════════
#  4B  SYNTHETIC LOAD FALLBACK
# ═══════════════════════════════════════════════════════════════

def _estimate_ev_hourly_demand(num_evs: int) -> list[float]:
    """24-element hourly EV demand (kW). Overnight Level-2 charging."""
    if num_evs == 0:
        return [0.0] * 24
    ch_hours = int(np.ceil(EV_DAILY_ENERGY_KWH / EV_CHARGER_POWER_KW))
    if EV_CHARGE_START_HOUR < EV_CHARGE_END_HOUR:
        ch = list(range(EV_CHARGE_START_HOUR, EV_CHARGE_END_HOUR))
    else:
        ch = list(range(EV_CHARGE_START_HOUR, 24)) + list(range(0, EV_CHARGE_END_HOUR))
    ch = ch[:ch_hours]
    p = [0.0] * 24
    for h in ch:
        p[h] = EV_CHARGER_POWER_KW * num_evs
    return p


def build_synthetic_load_profile(
    annual_kwh: float,
    num_evs: int,
    num_people: int,
    num_daytime_occupants: int,
) -> list[float]:
    """Synthetic 8760-h household load profile anchored to annual_kwh."""
    shape = np.array([
        0.40, 0.35, 0.32, 0.30, 0.30, 0.35,
        0.50, 0.70, 0.85, 0.80, 0.75, 0.72,
        0.70, 0.68, 0.65, 0.65, 0.70, 0.90,
        1.00, 0.95, 0.85, 0.75, 0.65, 0.50,
    ])
    ev24   = np.array(_estimate_ev_hourly_demand(num_evs))
    ev_ann = ev24.sum() * 365
    y0     = datetime(2024, 1, 1)
    raw: list[float] = []
    for h in range(8760):
        dt = y0 + timedelta(hours=h)
        sm = 1.20 if 6 <= dt.month <= 10 else (0.95 if dt.month in (3, 4, 5, 11) else 1.00)
        om = 1.0 + 0.05 * num_daytime_occupants if 9 <= dt.hour < 17 else 1.0
        ps = 0.70 + 0.10 * min(num_people, 6)
        raw.append(float(shape[dt.hour] * sm * om * ps))
    arr   = np.array(raw)
    scale = max(annual_kwh - ev_ann, 0.0) / arr.sum() if arr.sum() > 0 else 1.0
    return [float(raw[h] * scale + ev24[h % 24]) for h in range(8760)]


# ═══════════════════════════════════════════════════════════════
#  4C  IRRADIANCE SHAPE
# ═══════════════════════════════════════════════════════════════

def irradiance_shape_factor(hour_of_day: int, day_of_year: int) -> float:
    """
    Fractional G(t)/G_ref via sine-wave daylight model.
    Sunset varies seasonally: 18 + 2*sin(2pi*(doy-80)/365).
    Returns 0 outside daylight hours.
    """
    sunrise = 6.0
    sunset  = 18.0 + 2.0 * np.sin(2.0 * np.pi * (day_of_year - 80) / 365.0)
    if hour_of_day < sunrise or hour_of_day >= sunset:
        return 0.0
    return max(0.0, float(np.sin(np.pi * (hour_of_day - sunrise) / (sunset - sunrise))))


# ═══════════════════════════════════════════════════════════════
#  4D  ANNUAL IRRADIANCE
# ═══════════════════════════════════════════════════════════════

def fetch_irradiance_annual(latitude: float, longitude: float) -> float:
    """
    Mean annual GHI (kWh/m2/yr) from Open-Meteo 5-year archive.
    Falls back to SD average 2080 kWh/m2/yr on any failure.
    """
    try:
        import requests as _req
        end   = datetime.now() - timedelta(days=7)
        start = end.replace(year=end.year - 5)
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date":   end.strftime("%Y-%m-%d"),
            "daily": "shortwave_radiation_sum", "timezone": "auto",
        }
        r = _req.get(OPEN_METEO_URL, params=params, timeout=60)
        r.raise_for_status()
        vals = [v for v in r.json()["daily"]["shortwave_radiation_sum"] if v is not None]
        if not vals:
            raise ValueError("empty response")
        return round(float(np.mean(vals)) / 3.6 * 365, 1)
    except Exception:
        return 2080.0


# ═══════════════════════════════════════════════════════════════
#  4E  HOURLY TARIFFS
# ═══════════════════════════════════════════════════════════════

def build_hourly_tariffs(rate_plan: str, year: int = 2024) -> list[float]:
    """
    8760 hourly SDG&E TOU prices ($/kWh).

    Period mapping (all plans):
        On-Peak        16:00-21:00 every day
        Super-Off-Peak 00:00-06:00 + Mar/Apr 10:00-14:00
        Off-Peak       all other hours
    """
    rate_plan = rate_plan.upper()

    if rate_plan in ("TOU_DR", "TOU_DR1"):
        path = TOU_DR_PATH if rate_plan == "TOU_DR" else TOU_DR1_PATH
        df = pd.read_csv(path, parse_dates=["date"])
        lk = (
            df[df["date"].dt.year == year]
            .set_index("date")[["on_peak_$/kwh", "off_peak_$/kwh", "super_off_peak_$/kwh"]]
        )
        if lk.empty:
            lk = df.set_index("date")[["on_peak_$/kwh", "off_peak_$/kwh", "super_off_peak_$/kwh"]]

        out: list[float] = []
        y0 = datetime(year, 1, 1)
        for h in range(8760):
            dt = y0 + timedelta(hours=h)
            hr, mo = dt.hour, dt.month
            on_p  = (16 <= hr < 21)
            sup_p = (hr < 6) or (mo in (3, 4) and 10 <= hr < 14)
            dk = pd.Timestamp(dt.date())
            if dk not in lk.index:
                dk = min(lk.index, key=lambda d, _dk=dk: abs((d - _dk).days))
            row = lk.loc[dk]
            if on_p:
                out.append(float(row["on_peak_$/kwh"]))
            elif sup_p:
                val = row.get("super_off_peak_$/kwh", row["off_peak_$/kwh"])
                out.append(float(val) if pd.notna(val) else float(row["off_peak_$/kwh"]))
            else:
                out.append(float(row["off_peak_$/kwh"]))
        return out

    elif rate_plan == "TOU_DR2":
        df = pd.read_csv(TOU_DR2_PATH, parse_dates=["date"])
        lk = (
            df[df["date"].dt.year == year]
            .set_index("date")[["on_peak_$/kwh", "off_peak_$/kwh"]]
        )
        if lk.empty:
            lk = df.set_index("date")[["on_peak_$/kwh", "off_peak_$/kwh"]]

        out = []
        y0 = datetime(year, 1, 1)
        for h in range(8760):
            dt = y0 + timedelta(hours=h)
            hr = dt.hour
            on_p = (16 <= hr < 21)
            dk = pd.Timestamp(dt.date())
            if dk not in lk.index:
                dk = min(lk.index, key=lambda d, _dk=dk: abs((d - _dk).days))
            row = lk.loc[dk]
            out.append(float(row["on_peak_$/kwh"]) if on_p else float(row["off_peak_$/kwh"]))
        return out

    else:
        raise ValueError(f"Unknown rate_plan '{rate_plan}'. Use TOU_DR, TOU_DR1, or TOU_DR2.")


# ═══════════════════════════════════════════════════════════════
#  4F  PV OUTPUT
# ═══════════════════════════════════════════════════════════════

def build_hourly_pv_output(
    panel: Dict[str, Any],
    n_panels: int,
    irradiance_kwh_m2_yr: float,
) -> list[float]:
    """
    8760-h AC PV output (kW).
    P_array(t) = n_panels * panel_kw * shape(t) * PR.
    Normalised so annual sum = n_panels * panel_kw * irradiance * PR.
    """
    pkw = panel["panel_power_w"] / 1000.0
    raw = [
        n_panels * pkw * irradiance_shape_factor(h % 24, (h // 24) % 365 + 1) * PR_PERFORMANCE_RATIO
        for h in range(8760)
    ]
    tgt = n_panels * pkw * irradiance_kwh_m2_yr * PR_PERFORMANCE_RATIO
    tot = sum(raw)
    if tot > 0:
        raw = [v * tgt / tot for v in raw]
    return raw


# ═══════════════════════════════════════════════════════════════
#  4G  HARDWARE SELECTION
# ═══════════════════════════════════════════════════════════════

def select_panel(panel_brand: str | None) -> Dict[str, Any]:
    """None -> best efficiency/cost ratio. String -> exact manufacturer match."""
    if panel_brand is None:
        return max(SOLAR_PANEL_CATALOG, key=lambda p: p["efficiency_percent"] / p["cost_per_wp_usd"])
    m = [p for p in SOLAR_PANEL_CATALOG if p["manufacturer"].lower() == panel_brand.lower()]
    if not m:
        available = [p["manufacturer"] for p in SOLAR_PANEL_CATALOG]
        raise ValueError(f"Brand '{panel_brand}' not found. Available: {available}")
    return m[0]


def select_battery(required_kwh: float) -> Optional[Dict[str, Any]]:
    """Most cost-effective unit >= required_kwh. Returns None if required_kwh <= 0.5."""
    if required_kwh <= 0.5:
        return None
    cands = [b for b in BATTERY_CATALOG if b["usable_capacity_kwh"] >= required_kwh]
    if not cands:
        cands = BATTERY_CATALOG
    return min(cands, key=lambda b: b["cost_usd"] / b["usable_capacity_kwh"])


# ═══════════════════════════════════════════════════════════════
#  4H  DISPATCH SIMULATION
# ═══════════════════════════════════════════════════════════════

def run_dispatch_simulation(
    hourly_load_kw: list[float],
    hourly_pv_kw: list[float],
    hourly_tariffs: list[float],
    battery: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Rule-based 8760-h dispatch.
    Surplus: charge battery first, then export remainder.
    Deficit: discharge battery first, then import remainder.
    """
    has_b = battery is not None
    cap   = battery["usable_capacity_kwh"]       if has_b else 0.0
    maxc  = battery["max_charge_power_kw"]       if has_b else 0.0
    maxd  = battery["max_discharge_power_kw"]    if has_b else 0.0
    eta   = battery["round_trip_efficiency_pct"] / 100.0 if has_b else 1.0
    soc   = cap * 0.5

    tot_imp = tot_exp = imp_cost = exp_cred = bat_cyc = 0.0

    for h in range(8760):
        net = hourly_load_kw[h] - hourly_pv_kw[h]
        if net < 0:
            sur  = -net
            ckw  = min(maxc, (cap - soc) / eta) if has_b else 0.0
            ckw  = min(sur, ckw)
            soc += ckw * eta
            bat_cyc += ckw
            export = sur - ckw
            gbuy   = 0.0
        else:
            dkw  = min(maxd, soc) if has_b else 0.0
            dkw  = min(net, dkw)
            soc -= dkw
            gbuy   = net - dkw
            export = 0.0
        soc       = max(0.0, min(cap, soc))
        tot_imp  += gbuy
        tot_exp  += export
        imp_cost += gbuy * hourly_tariffs[h]
        exp_cred += export * NEM_EXPORT_CREDIT

    return {
        "annual_import_kwh":  round(tot_imp,  1),
        "annual_export_kwh":  round(tot_exp,  1),
        "import_cost_usd":    round(imp_cost, 2),
        "export_credit_usd":  round(exp_cred, 2),
        "battery_kwh_cycled": round(bat_cyc,  1),
    }


# ═══════════════════════════════════════════════════════════════
#  4I  ECONOMICS
# ═══════════════════════════════════════════════════════════════

def compute_economics(
    dispatch: Dict[str, Any],
    panel: Dict[str, Any],
    n_panels: int,
    battery: Optional[Dict[str, Any]],
    battery_units: int,
    annual_load_kwh: float,
    avg_tariff: float,
    with_battery: bool,
) -> Dict[str, Any]:
    """
    10-year NPV financial model.

    Year 0 capex: gross = (pv + battery) * (1 + install_rate), net = gross * (1 - ITC).
    Years 1-10:   traditional bill grows at utility_inflation;
                  solar bill uses degraded PV, inflated tariff, O&M, inverter.
    Payback = first year cumulative savings >= net_capex.
    NPV = -net_capex + sum(savings / (1+r)^y).
    """
    array_w  = n_panels * panel["panel_power_w"]
    pv_cost  = array_w * panel["cost_per_wp_usd"]
    bat_cost = (battery["cost_usd"] * battery_units) if battery else 0.0
    hw       = pv_cost + bat_cost
    install  = hw * INSTALLATION_COST_RATE
    gross    = hw + install
    net_cap  = gross * (1.0 - FEDERAL_ITC_RATE)

    fixed_ann = SDGE_DAILY_FIXED_FEE * 365
    trad_y1   = annual_load_kwh * avg_tariff + fixed_ann
    solar_y1  = dispatch["import_cost_usd"] - dispatch["export_credit_usd"] + fixed_ann

    cumul = 0.0
    payback: int | None = None
    npv = -net_cap
    rows: list[Dict[str, Any]] = []

    for y in range(1, ANALYSIS_YEARS + 1):
        inf  = (1 + UTILITY_INFLATION_RATE) ** (y - 1)
        disc = (1 + DISCOUNT_RATE) ** y
        trad = trad_y1 * inf
        scl  = (1 - panel["degradation_rate"]) ** (y - 1)
        imp  = annual_load_kwh - (annual_load_kwh - dispatch["annual_import_kwh"]) * scl
        exp  = dispatch["annual_export_kwh"] * scl
        sol  = imp * avg_tariff * inf - exp * NEM_EXPORT_CREDIT + fixed_ann
        om   = O_AND_M_COST_PER_W_YR * array_w * inf
        inv  = INVERTER_REPLACEMENT_USD if y == INVERTER_REPLACEMENT_YR else 0.0
        tot_sol = sol + om + inv
        sav  = trad - tot_sol
        cumul += sav
        if payback is None and cumul >= net_cap:
            payback = y
        npv += sav / disc
        rows.append({
            "year": y,
            "trad_bill_usd":     round(trad,    2),
            "solar_total_usd":   round(tot_sol, 2),
            "net_savings_usd":   round(sav,     2),
            "cumulative_savings": round(cumul,  2),
        })

    sav_y1 = trad_y1 - solar_y1 - O_AND_M_COST_PER_W_YR * array_w
    return {
        "scenario":                                   "with_battery" if with_battery else "pv_only",
        "panel_manufacturer":                         panel["manufacturer"],
        "panel_model":                                panel["model"],
        "n_panels":                                   n_panels,
        "system_kw_dc":                               round(n_panels * panel["panel_power_w"] / 1000, 2),
        "total_pv_cost_usd":                          round(pv_cost,   2),
        "total_battery_cost_usd":                     round(bat_cost,  2),
        "total_installation_cost_usd":                round(install,   2),
        "gross_capex_usd":                            round(gross,     2),
        "net_capex_after_itc_usd":                    round(net_cap,   2),
        "annual_grid_energy_import_kwh":              dispatch["annual_import_kwh"],
        "annual_grid_energy_export_kwh":              dispatch["annual_export_kwh"],
        "annual_electricity_bill_with_system_usd":    round(solar_y1,  2),
        "annual_electricity_bill_without_system_usd": round(trad_y1,   2),
        "annual_savings_usd":                         round(sav_y1,    2),
        "simple_payback_years":                       payback if payback else ANALYSIS_YEARS + 1,
        "npv_usd":                                    round(npv,       2),
        "ten_year_breakdown":                         rows,
    }


# ═══════════════════════════════════════════════════════════════
#  BRAND COMPARISON HELPER
# ═══════════════════════════════════════════════════════════════

def _compare_all_brands(
    roof_length_m: float,
    roof_breadth_m: float,
    budget_usd: float,
    annual_kwh: float,
    hourly_load: list,
    tariffs: list,
    avg_tariff: float,
    irradiance: float,
    battery: Optional[Dict[str, Any]],
    battery_units: int,
) -> List[Dict[str, Any]]:
    """
    Run 10-yr NPV economics for EVERY panel in SOLAR_PANEL_CATALOG under
    the same roof, budget, and load constraints and return a list sorted
    by NPV descending (best brand first).

    Each entry contains the key metrics so the LLM can explain the choice.
    """
    rows: List[Dict[str, Any]] = []

    for p in SOLAR_PANEL_CATALOG:
        # Roof capacity for this panel's actual dimensions
        layout = _compute_roof_layout(
            roof_length_m, roof_breadth_m, p["length_m"], p["width_m"]
        )
        max_roof = layout["max_panels_by_roof_dimensions"]

        # Budget capacity: use gross cost (incl. 10% install) so CAPEX <= budget pre-ITC
        p_kw             = p["panel_power_w"] / 1000.0
        cost_per_panel   = p["panel_power_w"] * p["cost_per_wp_usd"]
        row_battery      = battery
        row_bat_units    = battery_units
        bat_gross        = (row_battery["cost_usd"] * row_bat_units * (1 + INSTALLATION_COST_RATE)) if row_battery else 0.0
        cost_per_panel_gross = cost_per_panel * (1 + INSTALLATION_COST_RATE)
        budget_for_pv    = max(0.0, budget_usd - bat_gross)
        max_budget       = int(budget_for_pv // cost_per_panel_gross) if cost_per_panel_gross > 0 else 0

        # Feasibility fallback: drop battery if it leaves zero room for PV.
        if max_budget < 1 and row_battery is not None:
            row_battery   = None
            row_bat_units = 0
            bat_gross     = 0.0
            budget_for_pv = float(budget_usd)
            max_budget    = int(budget_for_pv // cost_per_panel_gross) if cost_per_panel_gross > 0 else 0

        # Sizing: aim for 70 % offset as the budget-aware target
        ann_prod_pp      = p_kw * irradiance * PR_PERFORMANCE_RATIO
        panels_70        = math.ceil(annual_kwh * 0.70 / ann_prod_pp) if ann_prod_pp > 0 else 0
        n_panels         = min(panels_70, max_budget, max_roof)

        if n_panels < 1:
            rows.append({
                "rank":             None,
                "manufacturer":     p["manufacturer"],
                "model":            p["model"],
                "n_panels":         0,
                "system_kw_dc":     0.0,
                "efficiency_pct":   p["efficiency_percent"],
                "cost_per_wp_usd":  p["cost_per_wp_usd"],
                "net_capex_usd":    0.0,
                "annual_savings_usd": 0.0,
                "payback_years":    None,
                "npv_10yr_usd":     -999999.0,
                "note":             "0 panels fit within roof/budget",
            })
            continue

        pv_h  = build_hourly_pv_output(p, n_panels, irradiance)
        disp  = run_dispatch_simulation(hourly_load, pv_h, tariffs, row_battery)
        econ  = compute_economics(
            disp, p, n_panels, row_battery, row_bat_units,
            annual_kwh, avg_tariff,
            with_battery=row_battery is not None,
        )

        rows.append({
            "rank":               None,          # filled after sorting
            "manufacturer":       p["manufacturer"],
            "model":              p["model"],
            "n_panels":           n_panels,
            "system_kw_dc":       econ["system_kw_dc"],
            "efficiency_pct":     p["efficiency_percent"],
            "cost_per_wp_usd":    p["cost_per_wp_usd"],
            "net_capex_usd":      econ["net_capex_after_itc_usd"],
            "annual_savings_usd": econ["annual_savings_usd"],
            "payback_years":      econ["simple_payback_years"],
            "npv_10yr_usd":       econ["npv_usd"],
            "note":               "",
        })

    # Sort by NPV descending; ties broken by annual savings
    rows.sort(key=lambda r: (r["npv_10yr_usd"], r["annual_savings_usd"]), reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return rows


# ═══════════════════════════════════════════════════════════════
#  ROOF LAYOUT HELPER
# ═══════════════════════════════════════════════════════════════

_PANEL_GAP_M = 0.02  # 2 cm between panels (racking clearance)


def _compute_roof_layout(
    roof_length_m: float,
    roof_breadth_m: float,
    panel_length_m: float,
    panel_width_m: float,
) -> Dict[str, Any]:
    """
    Compute how many panels fit on a rectangular roof in both
    portrait and landscape orientations and return the better one.

    Portrait  : panel long side (length_m) runs along roof_length.
    Landscape : panel short side (width_m) runs along roof_length
                (panel rotated 90°).

    A 2 cm gap between adjacent panels is assumed for racking hardware.
    """
    gap = _PANEL_GAP_M

    # Portrait: length along roof_length, width along roof_breadth
    cols_p = int((roof_length_m  + gap) / (panel_length_m + gap))
    rows_p = int((roof_breadth_m + gap) / (panel_width_m  + gap))
    total_p = cols_p * rows_p

    # Landscape: width along roof_length, length along roof_breadth
    cols_l = int((roof_length_m  + gap) / (panel_width_m  + gap))
    rows_l = int((roof_breadth_m + gap) / (panel_length_m + gap))
    total_l = cols_l * rows_l

    if total_p >= total_l:
        return {
            "orientation":         "portrait",
            "panels_along_length": cols_p,
            "panels_along_breadth": rows_p,
            "max_panels_by_roof_dimensions": total_p,
            "alt_orientation":     "landscape",
            "alt_panels_along_length": cols_l,
            "alt_panels_along_breadth": rows_l,
            "alt_max_panels":      total_l,
        }
    else:
        return {
            "orientation":         "landscape",
            "panels_along_length": cols_l,
            "panels_along_breadth": rows_l,
            "max_panels_by_roof_dimensions": total_l,
            "alt_orientation":     "portrait",
            "alt_panels_along_length": cols_p,
            "alt_panels_along_breadth": rows_p,
            "alt_max_panels":      total_p,
        }


# ═══════════════════════════════════════════════════════════════
#  4J  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_all_tools(
    latitude: float,
    longitude: float,
    num_evs: int = 0,
    num_people: int = 3,
    num_daytime_occupants: int = 1,
    budget_usd: float = 25000.0,
    roof_length_m: float = 8.0,
    roof_breadth_m: float = 6.25,
    rate_plan: str = "TOU_DR",
    panel_brand: str | None = None,
) -> Dict[str, Any]:
    """
    Run the full tool chain (4A-4I) and return a summary dict suitable
    for injection into the LLM prompt.

    This is the single entry-point the pipeline calls.

    Parameters
    ----------
    roof_length_m, roof_breadth_m : float
        Physical roof dimensions (metres).  Area = length × breadth.
        Panels are laid out in both portrait and landscape orientations;
        the orientation that fits more panels is used for sizing.
    """
    roof_area_m2 = round(roof_length_m * roof_breadth_m, 3)
    auto_brand   = panel_brand is None           # track whether brand was auto-selected

    # 4A: load profile (needed before brand comparison so all brands use same load)
    hourly_load, annual_kwh = load_household_profile_from_eia(
        latitude, longitude,
        num_evs=num_evs,
        num_people=num_people,
        num_daytime_occupants=num_daytime_occupants,
    )

    # 4D: irradiance (use SD default to avoid network call in pipeline)
    irradiance = 2080.0

    # 4E: tariffs
    tariffs = build_hourly_tariffs(rate_plan, year=2024)
    avg_tariff = round(sum(tariffs) / len(tariffs), 4)

    # nighttime load fraction – needed before battery selection and brand comparison
    hrs_arr    = np.arange(8760) % 24
    night_mask = (hrs_arr < 6) | (hrs_arr >= 20)
    load_arr   = np.array(hourly_load)
    nighttime_frac     = round(float(load_arr[night_mask].sum() / load_arr.sum()), 3) if load_arr.sum() > 0 else 0.0
    battery_kwh_needed = nighttime_frac * annual_kwh / 365 if nighttime_frac > 0.30 else 0.0
    battery      = select_battery(battery_kwh_needed)
    battery_units = 1 if battery else 0

    # 4G: brand selection — compare all brands by NPV when brand is auto
    brand_comparison: Optional[List[Dict[str, Any]]] = None
    if auto_brand:
        brand_comparison = _compare_all_brands(
            roof_length_m, roof_breadth_m,
            budget_usd, annual_kwh, hourly_load,
            tariffs, avg_tariff, irradiance,
            battery, battery_units,
        )
        # Pick the NPV winner (rank 1)
        best_row = brand_comparison[0]
        panel = next(
            (p for p in SOLAR_PANEL_CATALOG
             if p["manufacturer"] == best_row["manufacturer"] and p["model"] == best_row["model"]),
            None,
        )
        if panel is None:          # fallback – should never happen
            panel = select_panel(None)
    else:
        panel = select_panel(panel_brand)

    # Roof layout using actual panel dimensions
    roof_layout = _compute_roof_layout(
        roof_length_m, roof_breadth_m,
        panel["length_m"], panel["width_m"],
    )
    max_panels_by_roof = roof_layout["max_panels_by_roof_dimensions"]

    # Cell breakdown for the selected panel
    cell_info = {
        "cells_per_panel":   panel["cells"],
        "cells_in_series":   panel["cells_in_series"],
        "cells_in_parallel": panel["cells_in_parallel"],
        "panel_length_m":    panel["length_m"],
        "panel_width_m":     panel["width_m"],
    }

    # sizing
    panel_kw = panel["panel_power_w"] / 1000.0
    annual_prod_per_panel = panel_kw * irradiance * PR_PERFORMANCE_RATIO
    panels_100 = math.ceil(annual_kwh / annual_prod_per_panel) if annual_prod_per_panel > 0 else 0
    panels_70  = math.ceil(annual_kwh * 0.70 / annual_prod_per_panel) if annual_prod_per_panel > 0 else 0

    # Budget uses gross cost (incl. 10% install) so CAPEX <= budget pre-ITC.
    # The tool surfaces TWO candidate recommended sizings — PV-only and
    # PV+battery — so the agent can reason about the tradeoff. Panel count
    # is prioritised: PV-only sizes against the full budget; PV+battery
    # reserves battery capex first and then fits panels into what remains.
    cost_per_panel       = panel["panel_power_w"] * panel["cost_per_wp_usd"]
    cost_per_panel_gross = cost_per_panel * (1 + INSTALLATION_COST_RATE)
    full_bat_gross       = (battery["cost_usd"] * battery_units * (1 + INSTALLATION_COST_RATE)) if battery else 0.0

    # PV-only candidate
    max_panels_by_budget_pv_only = int(budget_usd // cost_per_panel_gross) if cost_per_panel_gross > 0 else 0
    n_rec_pv_only = min(panels_70, max_panels_by_budget_pv_only, max_panels_by_roof)

    # PV+battery candidate
    budget_for_pv_with_bat        = max(0.0, budget_usd - full_bat_gross)
    max_panels_by_budget_with_bat = int(budget_for_pv_with_bat // cost_per_panel_gross) if cost_per_panel_gross > 0 else 0
    n_rec_with_bat = min(panels_70, max_panels_by_budget_with_bat, max_panels_by_roof) if battery is not None else 0

    # Default n_rec (used by downstream recommended_scenario): pick PV+battery
    # when the battery has room to also deliver at least one panel; otherwise
    # fall back to PV-only so we never report a zero-panel system.
    if battery is not None and n_rec_with_bat >= 1:
        n_rec               = n_rec_with_bat
        rec_battery         = battery
        rec_battery_units   = battery_units
        bat_gross           = full_bat_gross
    else:
        n_rec               = n_rec_pv_only
        rec_battery         = None
        rec_battery_units   = 0
        bat_gross           = 0.0

    budget_for_pv        = max(0.0, budget_usd - bat_gross)
    max_panels_by_budget = int(budget_for_pv // cost_per_panel_gross) if cost_per_panel_gross > 0 else 0
    n_opt = min(panels_100, max_panels_by_roof)

    results: Dict[str, Any] = {
        "panel_selected": {
            "manufacturer":      panel["manufacturer"],
            "model":             panel["model"],
            "power_w":           panel["panel_power_w"],
            "efficiency_pct":    panel["efficiency_percent"],
            "cost_per_wp_usd":   panel["cost_per_wp_usd"],
            "area_m2":           panel["area_m2"],
            "length_m":          panel["length_m"],
            "width_m":           panel["width_m"],
            "cells_per_panel":   panel["cells"],
            "cells_in_series":   panel["cells_in_series"],
            "cells_in_parallel": panel["cells_in_parallel"],
        },
        "battery_selected": {
            "manufacturer": battery["manufacturer"],
            "model": battery["model"],
            "capacity_kwh": battery["usable_capacity_kwh"],
            "cost_usd": battery["cost_usd"],
        } if battery else None,
        "load_profile_summary": {
            "annual_kwh": round(annual_kwh, 1),
            "peak_kw": round(max(hourly_load), 2),
            "avg_kw": round(annual_kwh / 8760, 3),
            "nighttime_load_fraction": nighttime_frac,
        },
        "irradiance_kwh_m2_yr": irradiance,
        "tariff_summary": {
            "rate_plan": rate_plan,
            "avg_tariff_usd_kwh": avg_tariff,
            "on_peak_avg": round(float(np.mean([tariffs[h] for h in range(8760) if 16 <= h % 24 < 21])), 4),
            "off_peak_avg": round(float(np.mean([tariffs[h] for h in range(8760) if not (16 <= h % 24 < 21)])), 4),
        },
        "roof_summary": {
            "roof_length_m":   round(roof_length_m, 3),
            "roof_breadth_m":  round(roof_breadth_m, 3),
            "roof_area_m2":    roof_area_m2,
            **roof_layout,
        },
        "sizing": {
            "panels_for_100pct":                 panels_100,
            "panels_for_70pct":                  panels_70,
            "max_panels_by_roof":                max_panels_by_roof,
            "max_panels_by_budget":              max_panels_by_budget,
            "max_panels_by_budget_pv_only":      max_panels_by_budget_pv_only,
            "max_panels_by_budget_with_battery": max_panels_by_budget_with_bat,
            "n_rec_pv_only":                     n_rec_pv_only,
            "n_rec_with_battery":                n_rec_with_bat,
            "annual_prod_per_panel_kwh":         round(annual_prod_per_panel, 1),
        },
        "brand_selection": {
            "mode":                 "auto" if auto_brand else "user_specified",
            "selected_manufacturer": panel["manufacturer"],
            "selected_model":        panel["model"],
            "comparison_table":      brand_comparison,  # None when user specified a brand
        },
    }

    # 4F + 4H + 4I for recommended scenario (default pick: PV+battery when
    # affordable, PV-only otherwise — see sizing block above).
    pv_rec = build_hourly_pv_output(panel, n_rec, irradiance)
    disp_rec = run_dispatch_simulation(hourly_load, pv_rec, tariffs, rec_battery)
    econ_rec = compute_economics(disp_rec, panel, n_rec, rec_battery, rec_battery_units,
                                 annual_kwh, avg_tariff, with_battery=rec_battery is not None)
    econ_rec["total_cells_on_roof"]      = n_rec * panel["cells"]
    econ_rec["total_cells_in_series"]    = n_rec * panel["cells_in_series"]
    econ_rec["total_cells_in_parallel"]  = n_rec * panel["cells_in_parallel"]
    econ_rec["sizing_mode"]              = "pv_plus_battery" if rec_battery is not None else "pv_only"
    results["recommended_scenario"] = econ_rec

    # Alternative recommended sizing: always provide a PV-only candidate so the
    # agent can reason about the trade-off between more panels vs. a battery.
    if rec_battery is None:
        results["recommended_pv_only_scenario"] = None     # same as default
    else:
        pv_alt  = build_hourly_pv_output(panel, n_rec_pv_only, irradiance)
        disp_alt = run_dispatch_simulation(hourly_load, pv_alt, tariffs, None)
        econ_alt = compute_economics(disp_alt, panel, n_rec_pv_only, None, 0,
                                     annual_kwh, avg_tariff, with_battery=False)
        econ_alt["total_cells_on_roof"]      = n_rec_pv_only * panel["cells"]
        econ_alt["total_cells_in_series"]    = n_rec_pv_only * panel["cells_in_series"]
        econ_alt["total_cells_in_parallel"]  = n_rec_pv_only * panel["cells_in_parallel"]
        econ_alt["sizing_mode"]              = "pv_only"
        results["recommended_pv_only_scenario"] = econ_alt

    # 4F + 4H + 4I for optimal scenario (no battery to keep it simple)
    pv_opt = build_hourly_pv_output(panel, n_opt, irradiance)
    disp_opt = run_dispatch_simulation(hourly_load, pv_opt, tariffs, None)
    econ_opt = compute_economics(disp_opt, panel, n_opt, None, 0,
                                 annual_kwh, avg_tariff, with_battery=False)
    econ_opt["total_cells_on_roof"]      = n_opt * panel["cells"]
    econ_opt["total_cells_in_series"]    = n_opt * panel["cells_in_series"]
    econ_opt["total_cells_in_parallel"]  = n_opt * panel["cells_in_parallel"]
    results["optimal_scenario"] = econ_opt

    # ── Battery analysis: always compare PV-only vs PV+battery ────────────────
    # Uses the recommended panel count (n_rec) as the base.
    # Always selects the most cost-effective battery regardless of the nighttime
    # threshold so the LLM can make an informed add/skip decision.
    nighttime_kwh_per_day = nighttime_frac * annual_kwh / 365.0

    # Pick the battery to analyse (always pick one for comparison purposes)
    bat_for_analysis = select_battery(max(nighttime_kwh_per_day, 4.0))

    # PV-only baseline for n_rec panels
    disp_pv_only = run_dispatch_simulation(hourly_load, pv_rec, tariffs, None)
    econ_pv_only = compute_economics(disp_pv_only, panel, n_rec, None, 0,
                                     annual_kwh, avg_tariff, with_battery=False)

    # PV + battery for n_rec panels
    if bat_for_analysis:
        disp_with_bat = run_dispatch_simulation(hourly_load, pv_rec, tariffs, bat_for_analysis)
        econ_with_bat = compute_economics(disp_with_bat, panel, n_rec,
                                          bat_for_analysis, 1,
                                          annual_kwh, avg_tariff, with_battery=True)

        extra_savings_yr = round(
            econ_with_bat["annual_savings_usd"] - econ_pv_only["annual_savings_usd"], 2
        )
        bat_net_cost = round(bat_for_analysis["cost_usd"] * (1.0 - FEDERAL_ITC_RATE), 2)
        bat_incremental_payback = (
            round(bat_net_cost / extra_savings_yr, 1)
            if extra_savings_yr > 0 else None
        )
        # Reduced grid imports thanks to the battery
        import_reduction_kwh = round(
            econ_pv_only["annual_grid_energy_import_kwh"]
            - econ_with_bat["annual_grid_energy_import_kwh"], 1
        )
        # Self-consumption: fraction of PV output consumed on-site
        total_pv_kwh = n_rec * panel["panel_power_w"] / 1000.0 * irradiance * PR_PERFORMANCE_RATIO
        self_cons_pct = round(
            100.0 * (1.0 - econ_with_bat["annual_grid_energy_export_kwh"] / max(total_pv_kwh, 1)), 1
        )

        # Decision logic
        if bat_incremental_payback is not None and bat_incremental_payback <= 12 and extra_savings_yr >= 250:
            decision = "add_battery"
        elif extra_savings_yr >= 100 and nighttime_kwh_per_day >= 3.0:
            decision = "evaluate_later"
        else:
            decision = "pv_only"

        bat_analysis: Dict[str, Any] = {
            "battery_analysed": {
                "manufacturer":    bat_for_analysis["manufacturer"],
                "model":           bat_for_analysis["model"],
                "capacity_kwh":    bat_for_analysis["usable_capacity_kwh"],
                "gross_cost_usd":  bat_for_analysis["cost_usd"],
                "net_cost_after_itc_usd": bat_net_cost,
                "round_trip_efficiency_pct": bat_for_analysis["round_trip_efficiency_pct"],
                "cycle_life":      bat_for_analysis["cycle_life"],
            },
            "pv_only_annual_savings_usd":        econ_pv_only["annual_savings_usd"],
            "pv_only_annual_import_kwh":         econ_pv_only["annual_grid_energy_import_kwh"],
            "pv_only_net_capex_usd":             econ_pv_only["net_capex_after_itc_usd"],
            "pv_only_payback_years":             econ_pv_only["simple_payback_years"],
            "pv_plus_battery_annual_savings_usd": econ_with_bat["annual_savings_usd"],
            "pv_plus_battery_annual_import_kwh":  econ_with_bat["annual_grid_energy_import_kwh"],
            "pv_plus_battery_net_capex_usd":      econ_with_bat["net_capex_after_itc_usd"],
            "pv_plus_battery_payback_years":      econ_with_bat["simple_payback_years"],
            "extra_annual_savings_usd":           extra_savings_yr,
            "import_reduction_kwh":               import_reduction_kwh,
            "self_consumption_pct_with_battery":  self_cons_pct,
            "nighttime_load_kwh_per_day":         round(nighttime_kwh_per_day, 2),
            "nighttime_load_fraction":            nighttime_frac,
            "battery_incremental_payback_years":  bat_incremental_payback,
            "decision":                           decision,
        }
    else:
        bat_analysis = {
            "battery_analysed": None,
            "decision": "pv_only",
            "note": "No suitable battery found in catalog for this load profile.",
        }

    results["battery_analysis"] = bat_analysis

    return results
