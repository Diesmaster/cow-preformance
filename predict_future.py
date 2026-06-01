"""
predict_future.py
=================
Predict future weights for cattle still on the farm (salesDate > today).

For each active cow whose cow_id exists in the training dataframe:
  1. Take the last real observation as the baseline.
  2. Freeze the ration (feed settings) at that last observation's daily rates.
  3. Iteratively predict ADG over 14-day intervals until the cow's salesDate,
     feeding each predicted weight as input to the next interval.
  4. Save: per-cow weight trajectory plots, a combined CSV, and a summary table.

Usage
-----
# Panel model (default), Kalman-smoothed, all models
python predict_future.py --model-type panel

# OLS model, specific model name, no Kalman
python predict_future.py --model-type ols --no-kalman --model-name limousine_model1

# Custom output directory
python predict_future.py --model-type panel --output-dir results/forecast
"""

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from data_processor.DataProcessor import DataProcessing
from base_models.OLSModel import OLSModel
from base_models.PanelOLS import PanelOLSModel
from base_models.MixedEffectsModel import MixedEffectsModel
from models.models import models, OLS_models

try:
    from consts.consts import tdn_table
except ImportError:
    tdn_table = {
        'silage': 0.65, 'grass': 0.55, 'slobber': 0.72,
        'SP2A mix': 0.75, 'SP2B mix': 0.75, 'Rice Hay': 0.50,
        'SMG Mixfeed S14': 0.72, 'Ampas Tahu': 0.65,
    }


# ---------------------------------------------------------------------------
# Feature recomputation
# ---------------------------------------------------------------------------

def _safe_log(x, fallback=0.0):
    """log(x) returning fallback when x <= 0."""
    return math.log(x) if x and x > 0 else fallback


def _safe_div(a, b, fallback=0.0):
    return a / b if b and b != 0 else fallback


def recompute_features(template_row: pd.Series,
                       new_weight: float,
                       sim_date: date,
                       entry_date: date,
                       start_weight: float,
                       day_diff: int = 14,
                       feed_pct: float = None,
                       last_ration_dmi_pd: float = None) -> pd.Series:
    """
    Return a new feature row for a synthetic future 14-day window.

    Strategy
    --------
    - Feed features: frozen at their *per-day* rates from the template, then
      multiplied by the new day_diff to get window totals.
    - Weight-dependent features: recomputed from new_weight.
    - Medical: no new BEF, vaccinations, or deworming; hormone decay updated.
    - Time: daysOnFeedNow, day_diff, etc. updated to simulation date.
    """
    row = template_row.copy()

    # ── time ────────────────────────────────────────────────────────────────
    row['day_diff']        = day_diff
    row['day_diff_2']      = day_diff ** 2
    row['day_diff_recp']   = day_diff ** 2   # matches original formula
    row['theoritical_error_adg'] = 20 / day_diff

    days_on_feed = (sim_date - entry_date).days
    row['daysOnFeedNow']   = days_on_feed
    row['daysOnFeedNow_2'] = days_on_feed ** 2
    row['daysOnFeedNow_r'] = days_on_feed ** 0.5
    row['daysOnFeed_then'] = days_on_feed + day_diff

    # ── weight & metabolic weight ────────────────────────────────────────────
    row['weight']           = new_weight
    mw = (new_weight * 0.96) ** 0.75
    row['metabolic_weight'] = mw

    inc_ratio = _safe_div(new_weight, start_weight)
    row['increase_ratio']        = inc_ratio
    row['increase_ratio_r']      = inc_ratio ** 0.5
    row['increase_ratio_dt']     = _safe_div(inc_ratio, day_diff)
    row['increase_ratio_dt_r']   = _safe_div(inc_ratio, day_diff) ** 0.5
    row['exp_increase_ratio']    = math.exp(inc_ratio) if inc_ratio < 700 else math.exp(700)
    row['ln_increase_ratio_dt']  = _safe_div(_safe_log(inc_ratio), day_diff)
    row['mw_ratio']              = mw * inc_ratio
    row['originWeight_dt']       = _safe_div(row.get('originWeight', 0), day_diff)
    row['originWeight_mw']       = row.get('originWeight', 0) * mw

    # breed-specific metabolic weights
    breed = row.get('breed', 'Other')
    for b in ['Simental', 'Simmental', 'Limousin', 'Other']:
        row[f'metabolic_weight_{b}']      = mw if breed in (b,) else 0
        row[f'metabolic_weight_{b}_mw']   = mw ** 2 if breed in (b,) else 0
        avg_dmi_pd = row.get('avg_dm_intake_per_day', 1) or 1
        row[f'metabolic_weight_{b}_ddmi'] = _safe_div(mw, avg_dmi_pd) if breed in (b,) else 0

    # ── reconstruct feed totals from recipe + feedPercentageOfWeight ────────────
    # New total DMI = new_weight * feed_pct/100 (same recipe, weight drives intake).
    # All per-component rates scale proportionally (ration ratios unchanged).
    old_dmi_pd = (last_ration_dmi_pd
                  if last_ration_dmi_pd and last_ration_dmi_pd > 0
                  else row.get('avg_dm_intake_per_day') or row.get('total_dmi_dt') or 1)

    if feed_pct and feed_pct > 0 and old_dmi_pd > 0:
        new_dmi_pd = new_weight * feed_pct / 100
        feed_scale = new_dmi_pd / old_dmi_pd
    else:
        # Fallback: frozen rates (no feed_pct available)
        new_dmi_pd = row.get('avg_dm_intake_per_day') or row.get('total_dmi_dt') or old_dmi_pd
        feed_scale = 1.0

    # Apply scale to all per-day TDN rates (recipe composition stays fixed)
    tdn_silage  = row.get('tdn_silage_dt',  0) * feed_scale * day_diff
    tdn_rumput  = row.get('tdn_rumput_dt',  0) * feed_scale * day_diff
    tdn_slobber = row.get('tdn_slobber_dt', 0) * feed_scale * day_diff
    tdn_sp2a    = row.get('tdn_SP2A_dt',    0) * feed_scale * day_diff
    tdn_sp2b    = row.get('tdn_SP2B_dt',    0) * feed_scale * day_diff
    tdn_ricehay = row.get('tdn_ricehay_dt', 0) * feed_scale * day_diff
    total_tdn   = tdn_silage + tdn_rumput + tdn_slobber + tdn_sp2a + tdn_sp2b + tdn_ricehay

    # Also update the per-day *_dt row values so next iteration inherits scaled rates
    row['tdn_silage_dt']  = row.get('tdn_silage_dt',  0) * feed_scale
    row['tdn_rumput_dt']  = row.get('tdn_rumput_dt',  0) * feed_scale
    row['tdn_slobber_dt'] = row.get('tdn_slobber_dt', 0) * feed_scale
    row['tdn_SP2A_dt']    = row.get('tdn_SP2A_dt',    0) * feed_scale
    row['tdn_SP2B_dt']    = row.get('tdn_SP2B_dt',    0) * feed_scale
    row['tdn_ricehay_dt'] = row.get('tdn_ricehay_dt', 0) * feed_scale
    row['avg_dm_intake_per_day'] = new_dmi_pd

    total_dmi = new_dmi_pd * day_diff
    row['total_dmi']      = total_dmi
    row['total_dm_intake'] = total_dmi
    row['total_dmi_2']    = total_dmi ** 2
    row['1_total_dmi']    = _safe_div(1, total_dmi)
    row['1_total_dmi_dt'] = _safe_div(1, _safe_div(total_dmi, day_diff))
    row['total_dmi_dt']   = _safe_div(total_dmi, day_diff)
    row['total_dmi_dt_2'] = _safe_div(total_dmi, day_diff) ** 2
    row['r_total_dmi_dt'] = _safe_div(total_dmi, day_diff) ** 0.5
    row['total_dmi_dw']   = _safe_div(total_dmi, new_weight)
    row['total_dmi_log']  = _safe_log(total_dmi)
    row['total_dmi_log_dt'] = _safe_log(_safe_div(total_dmi, day_diff))
    row['total_dmi_log_dw'] = _safe_log(_safe_div(total_dmi, new_weight))
    row['total_dmi_log_dmi'] = _safe_log(total_dmi) * total_dmi
    row['day_diff_ddmi']   = _safe_div(day_diff, total_dmi)
    row['day_diff_2_ddmi'] = _safe_div(day_diff ** 2, total_dmi)
    row['weight_ddmi']     = _safe_div(_safe_div(total_dmi, day_diff), new_weight)
    row['weight_ddmi_ddmi']= _safe_div(1, day_diff * new_weight)

    # avg daily intake stays frozen (ration unchanged)
    avg_dmi_pd = row.get('avg_dm_intake_per_day', _safe_div(total_dmi, day_diff)) or 1
    row['avg_real_dm_inake_per_weight_per_day'] = _safe_div(avg_dmi_pd, new_weight) * 100
    row['avg_real_dm_inake_per_mw_per_day']     = _safe_div(avg_dmi_pd, mw) * 100 if mw > 0 else 0
    row['originWeight_ddmi'] = _safe_div(row.get('originWeight', 0), avg_dmi_pd)
    row['mw_per_ddmi']       = _safe_div(mw, avg_dmi_pd)

    # ── TDN totals & ratios ──────────────────────────────────────────────────
    row['total_tdn']         = total_tdn
    tdn_ratio = _safe_div(total_tdn, total_dmi)
    row['tdn_ratio']         = tdn_ratio
    row['total_tdn_ratio']   = tdn_ratio
    row['tdn_ratio_r_3']     = tdn_ratio ** (1/3) if tdn_ratio > 0 else 0

    row['total_tdn_dt']      = _safe_div(total_tdn, day_diff)
    row['total_tdn_dt_2']    = _safe_div(total_tdn, day_diff) ** 2
    row['total_tdn_dt_3']    = _safe_div(total_tdn, day_diff) ** 3
    row['total_tdn_dt_log']  = _safe_log(_safe_div(total_tdn, day_diff))
    row['total_tdn_mw_dt']   = _safe_div(_safe_div(total_tdn, mw), day_diff) if mw > 0 else 0
    row['total_tdn_mw_dt_2'] = row['total_tdn_mw_dt'] ** 2
    row['total_tdn_mw_dt_3'] = row['total_tdn_mw_dt'] ** 3
    row['total_tdn_mw_dt_log']= _safe_log(row['total_tdn_mw_dt'])
    row['total_tdn_mw']      = _safe_div(total_tdn, mw) if mw > 0 else 0
    row['total_tdn_mw_2']    = row['total_tdn_mw'] ** 2
    row['total_tdn_mw_3']    = row['total_tdn_mw'] ** 3
    row['total_tdn_2']       = total_tdn ** 2
    row['total_tdn_3']       = total_tdn ** 3
    row['total_tdn_squared'] = total_tdn ** 2
    row['1_total_tdn']       = _safe_div(1, total_tdn)
    row['total_tdn_squared_ddmi'] = _safe_div(total_tdn ** 2, total_dmi)
    row['total_tdn_3_ddmi']  = _safe_div(total_tdn ** 3, total_dmi)
    row['total_tdn_2_ddmi']  = _safe_div(total_tdn ** 2, total_dmi)

    row['total_tdn_greens_over_mw']    = _safe_div(tdn_silage + tdn_rumput, mw) if mw > 0 else 0
    row['total_tdn_greens_over_mw_dt'] = _safe_div(row['total_tdn_greens_over_mw'], day_diff)

    # per-tdn
    row['per_tdn_silage']      = _safe_div(tdn_silage, total_tdn)
    row['per_tdn_rumput']      = _safe_div(tdn_rumput, total_tdn)
    row['per_tdn_slobber']     = _safe_div(tdn_slobber, total_tdn)
    row['per_tdn_concentrates']= _safe_div(tdn_sp2a + tdn_sp2b + tdn_slobber, total_tdn)

    # tdn_*_dt (frozen ration → same per-day value)
    # tdn_silage_dt / tdn_rumput_dt / etc. stay unchanged in template

    # mw-dependent tdn features
    row['tdn_silage_over_mw_dt']  = _safe_div(_safe_div(tdn_silage, mw), day_diff) if mw > 0 else 0
    row['tdn_rumput_over_mw_dt']  = _safe_div(_safe_div(tdn_rumput, mw), day_diff) if mw > 0 else 0
    row['tdn_slobber_over_mw_dt'] = _safe_div(_safe_div(tdn_slobber, mw), day_diff) if mw > 0 else 0
    row['tdn_SP2A_over_mw_dt']    = _safe_div(_safe_div(tdn_sp2a, mw), day_diff) if mw > 0 else 0
    row['tdn_ricehay_over_mw_dt'] = _safe_div(_safe_div(tdn_ricehay, mw), day_diff) if mw > 0 else 0
    row['tdn_silage_over_mw']     = _safe_div(tdn_silage, mw) if mw > 0 else 0
    row['tdn_rumput_over_mw']     = _safe_div(tdn_rumput, mw) if mw > 0 else 0
    row['tdn_slobber_over_mw']    = _safe_div(tdn_slobber, mw) if mw > 0 else 0
    row['tdn_silage_over_mw_dt_2']  = row['tdn_silage_over_mw_dt'] ** 2
    row['tdn_rumput_over_mw_dt_2']  = row['tdn_rumput_over_mw_dt'] ** 2
    row['tdn_slobber_over_mw_dt_2'] = row['tdn_slobber_over_mw_dt'] ** 2
    row['tdn_slobber_over_mw_2']    = row['tdn_slobber_over_mw'] ** 2
    row['silage_rumput_mw']    = row['tdn_silage_over_mw'] * row['tdn_rumput_over_mw']
    row['silage_x_rumput_tdn'] = row['tdn_rumput_over_mw_dt'] * row['tdn_slobber_over_mw_dt']
    row['silage_rumput_dt_r_dmi'] = _safe_div(tdn_silage * tdn_rumput, day_diff)
    row['silage_rumput']          = tdn_silage * tdn_rumput
    greens_slobber_mw = _safe_div(row['total_tdn_greens_over_mw'], row['tdn_slobber_over_mw'])
    row['greens_slobber_mw']   = greens_slobber_mw
    row['greens_slobber_mw_2'] = greens_slobber_mw
    row['tdn_slobber_daysonfeed'] = row['tdn_slobber_over_mw_dt'] * days_on_feed

    # over_mw_daysinfeedlot_dt aliases
    row['tdn_silage_over_mw_daysinfeedlot_dt']  = _safe_div(tdn_silage, mw) if mw > 0 else 0
    row['tdn_rumput_over_mw_daysinfeedlot_dt']  = _safe_div(tdn_rumput, mw) if mw > 0 else 0
    row['tdn_slobber_over_mw_daysinfeedlot_dt'] = _safe_div(tdn_slobber, mw) if mw > 0 else 0
    row['tdn_SP2A_over_mw_daysinfeedlot_dt']    = _safe_div(tdn_sp2a, mw) if mw > 0 else 0
    row['tdn_ricehay_over_mw_daysinfeedlot_dt'] = _safe_div(tdn_ricehay, mw) if mw > 0 else 0

    # dt_ratio (divided by exp_increase_ratio)
    exp_inc = row['exp_increase_ratio']
    row['tdn_silage_dt_ratio']  = _safe_div(row.get('tdn_silage_dt',  0), exp_inc)
    row['tdn_rumput_dt_ratio']  = _safe_div(row.get('tdn_rumput_dt',  0), exp_inc)
    row['tdn_slobber_dt_ratio'] = _safe_div(row.get('tdn_slobber_dt', 0), exp_inc)
    row['tdn_SP2A_dt_ratio']    = _safe_div(row.get('tdn_SP2A_dt',    0), exp_inc)
    row['tdn_ricehay_dt_ratio'] = _safe_div(row.get('tdn_ricehay_dt', 0), exp_inc)

    # dmi interactions
    row['mw_dmi']         = mw * total_dmi
    row['mw_dmi_dt']      = _safe_div(mw * total_dmi, day_diff)
    row['mw_dmi_dt_ratio']= row['mw_dmi_dt'] * inc_ratio
    row['mw_dmi_dt_ratio_2'] = row['mw_dmi_dt_ratio'] ** 2
    row['ln_mw_dmi_dt_ratio']  = _safe_log(max(row['mw_dmi_dt_ratio'], 1e-10))
    row['ln_mw_dmi_dt_ratio_2']= row['ln_mw_dmi_dt_ratio'] ** 2
    row['mw_dmi_dt_2']    = row['mw_dmi_dt'] ** 2
    row['mw_ratio_dmi']   = inc_ratio * total_dmi
    row['mw_ratio_dmi_dt']= _safe_div(row['mw_ratio_dmi'], day_diff)
    row['mw_dmi_dt_dstartweight'] = _safe_div(row['mw_dmi_dt'], start_weight)
    row['increase_ratio_dt_dmi']  = _safe_div(inc_ratio, day_diff) * total_dmi
    row['increase_ratio_dt_r_dmi']= _safe_div(inc_ratio, day_diff) ** 0.5 * total_dmi
    row['day_diff_2_dmi'] = day_diff ** 2 * total_dmi
    row['day_diff_dmi']   = day_diff * total_dmi
    row['day_diff_dmi_log']= _safe_log(day_diff * total_dmi)

    # concentrates
    tdn_concentrats_dt = row.get('tdn_SP2A_dt', 0) + row.get('tdn_slobber_dt', 0)
    row['tdn_concentrats_dt']        = tdn_concentrats_dt
    row['tdn_concentrats_dt_r_dmi']  = tdn_concentrats_dt
    row['tdn_concentrats_2_dt']      = (row.get('tdn_SP2A_dt', 0)**2
                                        + row.get('tdn_slobber_dt', 0)**2)
    row['tdn_concentrats']           = tdn_sp2a + tdn_sp2b + tdn_slobber

    # tdn_*_dt_2_3 (normalised by total_tdn_dt^(2/3))
    total_tdn_dt = _safe_div(total_tdn, day_diff)
    denom_23 = total_tdn_dt ** (2/3) if total_tdn_dt > 0 else 1
    row['tdn_silage_dt_2_3']       = _safe_div(row.get('tdn_silage_dt',  0), denom_23)
    row['tdn_rumput_dt_2_3']       = _safe_div(row.get('tdn_rumput_dt',  0), denom_23)
    row['tdn_slobber_dt_2_3']      = _safe_div(row.get('tdn_slobber_dt', 0), denom_23)
    row['tdn_concentrats_dt_2_3']  = _safe_div(tdn_concentrats_dt, denom_23)

    # tdn_*_dt_r_dmi_mw (divided by inc_ratio per original code)
    row['tdn_silage_dt_r_dmi_mw']  = _safe_div(row.get('tdn_silage_dt', 0) * total_dmi, inc_ratio)
    row['tdn_rumput_dt_r_dmi_mw']  = _safe_div(row.get('tdn_rumput_dt', 0) * total_dmi, inc_ratio)
    row['tdn_slobber_dt_r_dmi_mw'] = _safe_div(row.get('tdn_slobber_dt',0) * total_dmi, inc_ratio)
    row['tdn_SP2A_dt_r_dmi_mw']    = _safe_div(row.get('tdn_SP2A_dt',  0) * total_dmi, inc_ratio)
    row['tdn_ricehay_dt_r_dmi_mw'] = _safe_div(row.get('tdn_silage_dt', 0) * total_dmi, inc_ratio**2)  # matches original
    row['tdn_SMG_dt_r_dmi_mw']     = row.get('tdn_SMG_dt_r_dmi_mw', 0)   # SMG stays frozen
    row['tdn_SP2B_dt_r_dmi_mw']    = _safe_div(row.get('tdn_SP2B_dt',  0) * total_dmi, inc_ratio)
    row['tdn_greens_dt_r_dmi_mw']  = _safe_div((row.get('tdn_silage_dt', 0)
                                                 + row.get('tdn_rumput_dt', 0))
                                                * total_dmi, 1)

    # sdmi / r_dmi features
    tdn_ratio_r_3 = tdn_ratio ** (1/3) if tdn_ratio > 0 else 1
    dm_silage_ratio  = row.get('dm_silage_ratio',  0)
    dm_rumput_ratio  = row.get('dm_rumput_ratio',  0)
    dm_slobber_ratio = row.get('dm_slobber_ratio', 0)
    dm_ricehay_ratio = row.get('dm_ricehay_ratio', 0)
    dm_sp2a_ratio    = row.get('dm_SP2A_mix_ratio',0)
    dm_concentrats_r = row.get('dm_concentrats',   0)

    def _sdmi(tdn_tbl_key, dm_ratio):
        return _safe_div(tdn_table.get(tdn_tbl_key, 0) * dm_ratio
                         * total_dmi ** (4/3), tdn_ratio_r_3)

    row['tdn_silage_dt_r_dmi']    = _safe_div(row.get('tdn_silage_dt', 0)
                                               * total_dmi**(4/3), tdn_ratio_r_3)
    row['tdn_silage_dt_sdmi']     = _sdmi('silage',   dm_silage_ratio)
    row['tdn_rumput_dt_sdmi']     = _sdmi('grass',    dm_rumput_ratio)
    row['tdn_slobber_dt_sdmi']    = _sdmi('slobber',  dm_slobber_ratio)
    row['tdn_ricehay_dt_sdmi']    = _sdmi('Rice Hay', dm_ricehay_ratio)
    row['tdn_SP2A_dt_r_dmi']      = _sdmi('SP2A mix', dm_sp2a_ratio)
    row['tdn_ricehay_dt_r_dmi']   = _sdmi('Rice Hay', dm_ricehay_ratio)

    tdn_concentrats_ratio = row.get('tdn_concentrats_ratio', 0)
    row['tdn_concentrats_dt_sdmi'] = _safe_div(
        (tdn_concentrats_ratio * dm_concentrats_r) * total_dmi**(4/3), tdn_ratio_r_3
    )
    row['tdn_concentrats_dt_3_r_dmi'] = _safe_div(
        tdn_concentrats_dt**3 * total_dmi**(4/3), tdn_ratio_r_3
    )
    row['tdn_rumput_log']           = _safe_log(row.get('tdn_rumput_dt', 0))
    row['tdn_rumput_log_r_dmi']     = _safe_div(
        row.get('tdn_rumput_dt', 0) * total_dmi**(4/3), tdn_ratio_r_3
    )
    row['tdn_rumput_dt_r_dmi']      = _safe_div(
        row['tdn_rumput_log'] * total_dmi**(4/3), tdn_ratio_r_3
    )
    row['tdn_rumput_dt_r_dmi_log']  = _safe_log(
        row.get('tdn_rumput_dt', 0) * total_dmi * tdn_ratio_r_3
    )
    row['tdn_silage_dt_r_dmi_log']  = _safe_log(
        row.get('tdn_silage_dt', 0) * total_dmi
    ) if row.get('tdn_silage_dt', 0) > 0 else 0

    # total_tdn interactions with dmi
    avg_rdmi = row.get('avg_real_dm_inake_per_weight_per_day', 0)
    row['total_tdn_1_dt_dmi'] = total_tdn_dt * avg_rdmi
    row['total_tdn_2_dt_dmi'] = total_tdn_dt**2 * avg_rdmi
    row['total_tdn_3_dt_dmi'] = total_tdn_dt**3 * avg_rdmi
    row['total_tdn_3_dt']     = row['total_tdn_mw_dt'] ** 3
    row['total_tdn_dt_2_sdmi']= _safe_div(tdn_ratio**2 * (total_dmi/day_diff)**3, 1)

    # tdn log features
    row['tdn_silage_dt_log']     = _safe_log(row.get('tdn_silage_dt', 0))
    row['tdn_rumput_dt_log']     = _safe_log(row.get('tdn_rumput_dt', 0))
    row['tdn_slobber_dt_log']    = _safe_log(row.get('tdn_slobber_dt', 0))
    row['tdn_slobber_over_mw_dt_log'] = _safe_log(row['tdn_slobber_over_mw_dt'])
    row['tdn_total_mw']          = _safe_div(total_tdn, mw) if mw > 0 else 0
    row['1_over_tdn_rumput']     = _safe_div(1, tdn_rumput)
    row['tdn_silage_dt_2']       = row.get('tdn_silage_dt', 0)**2
    row['tdn_rumput_dt_2']       = row.get('tdn_rumput_dt', 0)**2
    row['tdn_slobber_dt_2']      = row.get('tdn_slobber_dt', 0)**2
    row['tdn_silage_over_mw_dt_ddmi'] = _safe_div(row['tdn_silage_over_mw_dt'],
                                                    _safe_div(total_dmi, day_diff))
    row['tdn_rumput_over_mw_dt_ddmi'] = _safe_div(row['tdn_rumput_over_mw_dt'],
                                                    _safe_div(total_dmi, day_diff))
    row['tdn_slobber_over_mw_dt_ddmi']= _safe_div(row['tdn_slobber_over_mw_dt'],
                                                    _safe_div(total_dmi, day_diff))
    row['tdn_silage_ddmi']  = _safe_div(tdn_silage, total_dmi)
    row['tdn_rumput_ddmi']  = _safe_div(tdn_rumput, total_dmi)
    row['tdn_slobber_ddmi'] = _safe_div(tdn_slobber, total_dmi)
    row['per_slobber_dm_dmi'] = (row.get('per_slobber_dm', 0)
                                 * row.get('avg_real_dm_inake_per_weight_per_day', 0))
    row['silage_dm_log']  = _safe_log(row.get('silage_dm',  0))
    row['grass_dm_log']   = _safe_log(row.get('grass_dm',   0))
    row['slobber_dm_log'] = _safe_log(row.get('slobber_dm', 0))

    # ── medical (frozen / decayed) ───────────────────────────────────────────
    # No new interventions assumed for future windows
    row['hasBEF']              = False
    row['hasBEF_dt']           = 0.0
    row['hasBEF_dmi_dt']       = 0.0
    row['hasBEF_dmi_dt_log']   = 0.0
    row['hasBEF_dmi']          = 0.0
    row['hasBEF_dmi_dt_2']     = 0.0
    row['hasBEF_ddmi']         = 0.0
    row['gotHNMVaccination']   = False
    row['gotHNMVaccination_dt']= 0.0
    row['gotHNMVaccination_dmi_dt'] = 0.0
    row['gotDewormed']         = False
    row['gotDewormed_dt']      = 0.0
    row['gotDewormed_dmi_dt']  = 0.0
    row['gotAppetiteBoost']    = False
    row['gotAppetiteBoost_dt'] = 0.0
    row['gotAppetiteBoost_dmi_dt'] = 0.0
    row['gotWorms']            = False

    # Deworming: days since last treatment keeps advancing
    prev_dsd = template_row.get('DaysSinceDewormed', 999)
    if prev_dsd is not None and prev_dsd >= 0:
        new_dsd = prev_dsd + day_diff
        row['DaysSinceDewormed']    = new_dsd
        row['DaysSinceDewormed_dt'] = _safe_div(new_dsd, day_diff)
        row['gotDewormed_dt']       = 0.0

    # Hormone decay advancement
    prev_hormone_days = template_row.get('gotHormonesLast', 999)
    if prev_hormone_days is not None and prev_hormone_days >= 0:
        new_hd = prev_hormone_days + day_diff
        row['gotHormonesLast'] = new_hd
        T_MIN, T_PEAK, LAMB = 3, 30, 0.02
        if new_hd < T_MIN:
            h = 0.0
        elif new_hd < T_PEAK:
            h = (new_hd - T_MIN) / (T_PEAK - T_MIN)
        else:
            h = math.exp(-LAMB * (new_hd - T_PEAK))
        row['hormone_effect'] = h
        if new_hd >= 14:
            row['gotHormones']        = True
            row['1_over_hormones']    = _safe_div(1, new_hd)
            row['hormone_adjustment'] = 0
        else:
            row['gotHormones']        = False
            row['1_over_hormones']    = 0.0
            row['hormone_adjustment'] = 1

    got_hormones = bool(row.get('gotHormones', False))
    row['gotHormones_dt']        = _safe_div(int(got_hormones), day_diff)
    row['gotHormones_dmi_dt']    = _safe_div(int(got_hormones), day_diff) * total_dmi
    row['gotHormones_dt_mw']     = row['gotHormones_dt'] * mw
    row['1_over_hormones_dt']    = _safe_div(row.get('1_over_hormones', 0), day_diff)
    row['hormone_adjustment_dt'] = _safe_div(row.get('hormone_adjustment', 0), day_diff)
    row['hormone_effect_dt']     = _safe_div(row.get('hormone_effect', 0), day_diff)
    row['hormone_effect_dt_dmi'] = row['hormone_effect_dt'] * total_dmi

    return row


# ---------------------------------------------------------------------------
# Entity-effect estimation (for Panel/Mixed models)
# ---------------------------------------------------------------------------

def estimate_entity_effect(fitted_model, df_train: pd.DataFrame,
                            independent_attr: list, dependent_attr: str,
                            cow_id: str) -> float:
    """
    Compute the cow-specific intercept as mean(y - Xβ) over training rows.
    Works for any model type without relying on internal attributes.
    """
    try:
        params = fitted_model.results.params
        cow_df = df_train[df_train['cow_id'] == cow_id].copy()
        if len(cow_df) == 0:
            return 0.0

        # Align parameter names with available columns
        shared = [c for c in independent_attr if c in cow_df.columns
                  and c in params.index]
        if not shared:
            return 0.0

        X = cow_df[shared].fillna(0).values
        beta = params[shared].values
        y = cow_df[dependent_attr].values
        residuals = y - (X @ beta)
        return float(np.mean(residuals))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main prediction loop
# ---------------------------------------------------------------------------

def predict_future_for_model(model_name: str,
                              model_config: dict,
                              model_type: str,
                              df: pd.DataFrame,
                              processor: DataProcessing,
                              output_dir: str,
                              kalman: bool,
                              use_cv: bool,
                              k_folds: int,
                              interval_days: int = 14) -> pd.DataFrame:
    """
    Fit model on historical data, identify active cows, run iterative forecast.

    Returns
    -------
    pd.DataFrame  All predictions for all cows (long format).
    """
    today = date.today()
    dependent_attr  = model_config['depended_attr']
    independent_attr = model_config['indpended_attr']

    print(f"\n{'='*80}")
    print(f"FUTURE PREDICTION: {model_name}")
    print(f"  Dependent:   {dependent_attr}")
    print(f"  Independent: {len(independent_attr)} features")
    print(f"  Interval:    {interval_days} days")
    print(f"{'='*80}")

    # ── 1. Fit model on all historical data ──────────────────────────────────
    print(f"\n[1/4] Fitting {model_type.upper()} model on historical data...")
    try:
        if model_type == 'ols':
            fitted = OLSModel(independent_attr, dependent_attr, 1, model_name)
            fitted.fit(df)
        elif model_type == 'mixed':
            fitted = MixedEffectsModel(independent_attr, dependent_attr, 1,
                                       model_name, group_col='cow_id')
            if use_cv:
                fitted.fit_with_cv(df, k=k_folds)
            else:
                fitted.fit(df)
        else:  # panel
            fitted = PanelOLSModel(independent_attr, dependent_attr, 1,
                                   model_name, group_col='cow_id',
                                   time_col='pred_date',
                                   entity_effects=True, time_effects=False)
            if use_cv:
                fitted.fit_with_cv(df, k=k_folds)
            else:
                fitted.fit(df)
        print("  ✓ Model fitted successfully")
    except Exception as e:
        print(f"  ✗ Model fit failed: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

    # ── 2. Identify active cows ───────────────────────────────────────────────
    print(f"\n[2/4] Identifying active cows (salesDate > {today})...")

    # Build cow_id → (salesDate, entryDate, startWeight) mapping from objects
    cow_meta = {}
    for doc_id, cow_dict in processor.objects.items():
        cd = cow_dict['cow_data']
        cattle_id = cd.cattleId

        # salesDate
        sales_date = None
        for attr in ('salesDate', 'sale_date', 'saleDate'):
            val = getattr(cd, attr, None)
            if val:
                try:
                    sales_date = datetime.strptime(str(val)[:10], '%Y-%m-%d').date()
                except ValueError:
                    pass
                break

        # entryDate / startWeight
        entry_date = None
        for attr in ('entryDate', 'entry_date', 'arrivalDate'):
            val = getattr(cd, attr, None)
            if val:
                try:
                    entry_date = datetime.strptime(str(val)[:10], '%Y-%m-%d').date()
                except ValueError:
                    pass
                break

        entry_weight = getattr(cd, 'entryWeight', None)
        origin_weight = getattr(cd, 'originWeight', None)
        start_weight = entry_weight or origin_weight

        # Feed percentage of body weight and last ration from feed history
        feed_history = cow_dict.get('feed_history_data')
        feed_pct = None
        last_ration_dmi_per_day = None
        if feed_history and feed_history.data:
            last_entry = feed_history.data[-1]
            feed_pct = last_entry.get('feedPercentageOfWeight') or last_entry.get('percentageOfBodyWeight')
            last_ration_dmi_per_day = last_entry.get('dryMatterIntakePerCow')  # per-day DMI at last entry

        cow_meta[cattle_id] = {
            'salesDate':            sales_date,
            'entryDate':            entry_date,
            'startWeight':          start_weight,       # first feedlot weighing → for increase_ratio
            'weight':               getattr(cd, 'weight', None),  # actual live weight → forecast baseline
            'feed_pct':             feed_pct,           # % of body weight → scales DMI with new weight
            'last_ration_dmi_pd':   last_ration_dmi_per_day,  # reference DMI/day for scaling
            'doc_id':               doc_id,
        }

    # Cows in training data
    known_cows = set(df['cow_id'].unique())

    active_cows = []
    for cid, meta in cow_meta.items():
        sd = meta.get('salesDate')
        if sd is None or sd <= today:
            continue
        if cid not in known_cows:
            continue
        active_cows.append(cid)

    print(f"  Total cows in model:  {len(known_cows)}")
    print(f"  Active (unsold) cows: {len(active_cows)}")
    if not active_cows:
        print("  → No active cows found. Check that salesDate is populated and > today.")
        return pd.DataFrame()

    # ── 3. Iterative prediction per cow ──────────────────────────────────────
    print(f"\n[3/4] Running iterative {interval_days}-day forecasts...")

    all_predictions = []
    params = fitted.results.params

    for cow_id in active_cows:
        meta = cow_meta[cow_id]
        sale_date          = meta['salesDate']
        entry_date         = meta['entryDate']
        feed_pct           = meta.get('feed_pct')
        last_ration_dmi_pd = meta.get('last_ration_dmi_pd')

        # Last real observation for this cow (used for features + dates)
        cow_hist = df[df['cow_id'] == cow_id].sort_values('pred_date')
        if len(cow_hist) == 0:
            print(f"  ⚠  {cow_id}: no historical rows, skipping")
            continue

        last_row  = cow_hist.iloc[-1].copy()
        last_date = pd.to_datetime(last_row['pred_date']).date()

        # ── Two distinct weight concepts ──────────────────────────────────
        # current_weight: actual live weight to start the forecast from.
        #   → meta['weight'] = cd.weight (most recent real weight on the cow object)
        #   → fallback to last df row weight if None
        meta_weight = meta.get('weight')
        current_weight = (float(meta_weight) if meta_weight is not None
                          else float(last_row.get('weight', last_row.get('pred_weight', 0))))

        # ratio_base_weight: denominator for increase_ratio = weight / startWeight.
        #   Must match training: startWeight = weight_history.data[0]['weight']
        #   (the very first feedlot weighing, stored as 'startWeight' in every df row)
        ratio_base_weight = float(last_row.get('startWeight',
                                               last_row.get('originWeight', current_weight)))

        # ── Fallback: entry_date ─────────────────────────────────────────
        if entry_date is None:
            try:
                first_row  = cow_hist.iloc[0]
                date_str   = str(first_row.get('date', first_row.get('pred_date', '')))[:10]
                dof        = int(first_row.get('daysOnFeedNow', 0))
                entry_date = (datetime.strptime(date_str, '%Y-%m-%d')
                              - timedelta(days=dof)).date()
            except Exception:
                entry_date = last_date

        # Estimate entity effect for this cow
        entity_effect = estimate_entity_effect(
            fitted, df, independent_attr, dependent_attr, cow_id
        )

        # ── iterative forecast ────────────────────────────────────────────
        current_date = last_date
        template_row = last_row.copy()
        cow_preds    = []

        # Record baseline (actual current weight from CowData)
        cow_preds.append({
            'cow_id':          cow_id,
            'date':            current_date,
            'weight':          current_weight,
            'predicted_adg':   None,
            'weight_gain':     None,
            'actual_days':     None,
            'is_partial':      False,
            'is_forecast':     False,
            'days_to_sale':    (sale_date - current_date).days,
            'sale_date':       sale_date,
        })

        step = 0
        while current_date < sale_date:
            next_date = current_date + timedelta(days=interval_days)
            is_partial = next_date > sale_date
            if is_partial:
                next_date = sale_date
            actual_diff = (next_date - current_date).days
            if actual_diff <= 0:
                break

            # Always build features as if it's a full interval_days window —
            # this keeps all ratio/dt features consistent regardless of whether
            # we're in a partial tail period.
            feat_row = recompute_features(
                template_row, current_weight,
                current_date, entry_date, ratio_base_weight,
                day_diff=interval_days,         # <-- always full 14 days
                feed_pct=feed_pct,
                last_ration_dmi_pd=last_ration_dmi_pd,
            )

            # Predict ADG
            try:
                shared_cols = [c for c in independent_attr
                               if c in feat_row.index and c in params.index]
                if not shared_cols:
                    print(f"  ⚠  {cow_id} step {step}: no shared feature columns")
                    break

                X_vec    = np.array([feat_row[c] if pd.notna(feat_row[c])
                                     else 0.0 for c in shared_cols])
                beta     = params[shared_cols].values
                pred_adg = float(np.dot(X_vec, beta)) + entity_effect
            except Exception as e:
                print(f"  ⚠  {cow_id} step {step} prediction error: {e}")
                break

            # Weight gain uses actual days remaining (may be < interval for last step)
            weight_gain = pred_adg * actual_diff
            new_weight  = current_weight + weight_gain

            cow_preds.append({
                'cow_id':          cow_id,
                'date':            next_date,
                'weight':          new_weight,
                'predicted_adg':   pred_adg,        # ADG as if full 14-day period
                'weight_gain':     weight_gain,     # actual gain (days may be < 14)
                'actual_days':     actual_diff,     # days in this interval
                'is_partial':      is_partial,      # flag for last step
                'is_forecast':     True,
                'days_to_sale':    (sale_date - next_date).days,
                'sale_date':       sale_date,
            })

            # Advance state
            template_row  = feat_row
            current_weight = new_weight
            current_date   = next_date
            step          += 1

        print(f"  ✓ {cow_id}: {step} forecast steps  |  "
              f"start {cow_preds[0]['weight']:.1f} kg → predicted {current_weight:.1f} kg "
              f"(sale: {sale_date})")

        all_predictions.extend(cow_preds)

    if not all_predictions:
        return pd.DataFrame()

    pred_df = pd.DataFrame(all_predictions)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    pred_df['model'] = model_name

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    print(f"\n[4/4] Saving outputs to '{output_dir}'...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _save_plots(pred_df, model_name, out_path)
    _save_csv(pred_df, model_name, out_path)
    _save_summary(pred_df, model_name, out_path)

    return pred_df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _plot_single_cow(axes_row, cow_id, cow_df):
    """
    Fill a 2-axis row (weight trajectory | ADG bars) for one cow.
    axes_row: [ax_weight, ax_adg]
    """
    ax, ax2 = axes_row
    cow_df  = cow_df.sort_values('date')
    hist    = cow_df[~cow_df['is_forecast']]
    fc      = cow_df[cow_df['is_forecast']]
    sale_date = cow_df['sale_date'].iloc[0]

    # ── weight trajectory ─────────────────────────────────────────────────
    if not hist.empty:
        ax.scatter(hist['date'], hist['weight'], color='#2c6fad',
                   zorder=5, label='Last known', s=50)
    if not fc.empty:
        # Distinguish partial (last) step visually
        if 'is_partial' in fc.columns:
            partial_mask = fc['is_partial'].fillna(False).astype(bool)
        else:
            partial_mask = pd.Series(False, index=fc.index)
        full_fc    = fc[~partial_mask]
        partial_fc = fc[partial_mask]

        ax.plot(fc['date'], fc['weight'], color='#e05c2a',
                linewidth=2, marker='o', markersize=4, label='Forecast')
        if not partial_fc.empty:
            ax.scatter(partial_fc['date'], partial_fc['weight'],
                       color='#e05c2a', marker='*', s=80, zorder=6,
                       label='Partial interval')
        ax.fill_between(fc['date'],
                        fc['weight'] * 0.97, fc['weight'] * 1.03,
                        alpha=0.12, color='#e05c2a')

    final_w = float(fc['weight'].iloc[-1]) if not fc.empty else float(hist['weight'].iloc[-1])
    ax.axvline(pd.Timestamp(sale_date), color='#2a9d2a', linestyle='--',
               linewidth=1.4, label=f'Sale {sale_date}')
    ax.annotate(f'{final_w:.0f} kg',
                xy=(pd.Timestamp(sale_date), final_w),
                xytext=(-38, 10), textcoords='offset points',
                fontsize=8, color='#e05c2a',
                arrowprops=dict(arrowstyle='->', color='#e05c2a', lw=0.8))

    ax.set_title(str(cow_id), fontsize=9, fontweight='bold')
    ax.set_ylabel('Weight (kg)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── ADG bars ─────────────────────────────────────────────────────────
    if not fc.empty and fc['predicted_adg'].notna().any():
        bar_colors = []
        for _, r in fc.iterrows():
            if r.get('is_partial', False):
                bar_colors.append('#f0a500')   # amber for partial step
            elif r['predicted_adg'] >= 0:
                bar_colors.append('#2ca02c')
            else:
                bar_colors.append('#d62728')

        ax2.bar(fc['date'], fc['predicted_adg'],
                color=bar_colors, width=10, alpha=0.82)
        ax2.axhline(0, color='black', linewidth=0.6)
        mean_adg = fc['predicted_adg'].mean()
        ax2.axhline(mean_adg, color='orange', linestyle='--',
                    linewidth=1.1, label=f'μ={mean_adg:.3f}')
        ax2.legend(fontsize=6)

    ax2.set_ylabel('ADG (kg/day)', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.25, axis='y')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')


def _save_plots(pred_df: pd.DataFrame, model_name: str, out_path: Path,
                cows_per_page: int = 6):
    """
    All cows together in one combined grid figure (saved as a single PNG).
    Layout: each cow gets one row of 2 panels [weight | ADG].
    If there are many cows the figure is split into pages of `cows_per_page` rows.
    """
    plots_dir = out_path / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    cow_ids = sorted(pred_df['cow_id'].unique())
    n_cows  = len(cow_ids)

    if n_cows == 0:
        return

    # Split into pages so the figure stays readable
    pages = [cow_ids[i:i + cows_per_page]
             for i in range(0, n_cows, cows_per_page)]

    for page_idx, page_cows in enumerate(pages):
        n_rows  = len(page_cows)
        fig_h   = max(4 * n_rows, 6)
        fig, axes = plt.subplots(n_rows, 2,
                                 figsize=(16, fig_h),
                                 squeeze=False)

        page_label = f' (page {page_idx + 1}/{len(pages)})' if len(pages) > 1 else ''
        fig.suptitle(f'Future Weight Forecast — {model_name}{page_label}',
                     fontsize=13, fontweight='bold', y=1.01)

        for row_idx, cow_id in enumerate(page_cows):
            cow_df = pred_df[pred_df['cow_id'] == cow_id]
            _plot_single_cow([axes[row_idx, 0], axes[row_idx, 1]], cow_id, cow_df)

        # Hide unused rows if page is not full (shouldn't happen, but defensive)
        for row_idx in range(len(page_cows), n_rows):
            axes[row_idx, 0].set_visible(False)
            axes[row_idx, 1].set_visible(False)

        plt.tight_layout()

        suffix = f'_p{page_idx + 1}' if len(pages) > 1 else ''
        plot_file = plots_dir / f'{model_name}{suffix}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Combined plot ({len(page_cows)} cows) → {plot_file}")

    print(f"  ✓ All plots saved → {plots_dir}/")


def _save_csv(pred_df: pd.DataFrame, model_name: str, out_path: Path):
    """Single CSV with all cows, all intervals."""
    csv_file = out_path / f'{model_name}_future_predictions.csv'
    cols = ['model', 'cow_id', 'date', 'weight', 'predicted_adg',
            'weight_gain', 'actual_days', 'is_partial',
            'days_to_sale', 'sale_date', 'is_forecast']
    out_cols = [c for c in cols if c in pred_df.columns]
    pred_df[out_cols].to_csv(csv_file, index=False, float_format='%.4f')
    print(f"  ✓ Combined CSV  → {csv_file}")


def _save_summary(pred_df: pd.DataFrame, model_name: str, out_path: Path):
    """Summary table: one row per cow with final predicted weight at sale."""
    forecast_only = pred_df[pred_df['is_forecast']]

    if forecast_only.empty:
        print("  ⚠  No forecast rows to summarise.")
        return

    rows = []
    for cow_id, cdf in pred_df.groupby('cow_id'):
        hist = cdf[~cdf['is_forecast']]
        fc   = cdf[cdf['is_forecast']].sort_values('date')

        baseline_weight = float(hist['weight'].iloc[-1]) if not hist.empty else None
        baseline_date   = hist['date'].iloc[-1] if not hist.empty else None
        final_weight    = float(fc['weight'].iloc[-1]) if not fc.empty else baseline_weight
        sale_date       = cdf['sale_date'].iloc[0]
        n_intervals     = len(fc)
        mean_adg        = float(fc['predicted_adg'].mean()) if not fc.empty else None
        total_gain      = (final_weight - baseline_weight
                           if final_weight is not None and baseline_weight is not None else None)
        days_remaining  = (pd.Timestamp(sale_date) - (baseline_date
                           if baseline_date else pd.Timestamp(sale_date))).days

        rows.append({
            'cow_id':           cow_id,
            'baseline_date':    baseline_date,
            'baseline_weight_kg': round(baseline_weight, 1) if baseline_weight else None,
            'sale_date':        sale_date,
            'days_remaining':   days_remaining,
            'n_forecast_intervals': n_intervals,
            'predicted_final_weight_kg': round(final_weight, 1) if final_weight else None,
            'predicted_total_gain_kg':   round(total_gain, 1) if total_gain else None,
            'mean_predicted_adg': round(mean_adg, 4) if mean_adg else None,
        })

    summary_df = (pd.DataFrame(rows)
                  .sort_values('predicted_final_weight_kg', ascending=False)
                  .reset_index(drop=True))

    summary_file = out_path / f'{model_name}_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    # Print table to console
    print(f"\n  ── Summary: {model_name} ──")
    print(f"  {'Cow ID':<20} {'Baseline':>9} {'Sale Date':>12} "
          f"{'Days':>6} {'Pred. Final':>12} {'Total Gain':>11} {'Mean ADG':>9}")
    print(f"  {'-'*85}")
    for _, r in summary_df.iterrows():
        print(f"  {str(r['cow_id']):<20} "
              f"{str(r['baseline_weight_kg']):>9} "
              f"{str(r['sale_date']):>12} "
              f"{str(r['days_remaining']):>6} "
              f"{str(r['predicted_final_weight_kg']):>12} "
              f"{str(r['predicted_total_gain_kg']):>11} "
              f"{str(r['mean_predicted_adg']):>9}")

    print(f"\n  ✓ Summary CSV  → {summary_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Predict future weights for cattle still on the farm.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: panel model, Kalman-smoothed, all models
  python predict_future.py

  # OLS model, no Kalman smoothing
  python predict_future.py --model-type ols --no-kalman

  # Specific model, custom output folder
  python predict_future.py --model-name limousine_model1 --output-dir results/forecast

  # Disable cross-validation
  python predict_future.py --model-type panel --no-cv

  # Change forecast interval (default 14 days)
  python predict_future.py --interval 21
        """
    )

    parser.add_argument('--model-type', choices=['ols', 'panel', 'mixed'],
                        default='panel')
    parser.add_argument('--kalman',    dest='kalman',  action='store_true')
    parser.add_argument('--no-kalman', dest='kalman',  action='store_false')
    parser.set_defaults(kalman=True)
    parser.add_argument('--cut-tails', action='store_true', default=False)
    parser.add_argument('--measurement-noise', type=float, default=None)
    parser.add_argument('--model-name', type=str, default=None,
                        help='Run only this model (default: all)')
    parser.add_argument('--n-weighings', type=int, nargs='+', default=[1])
    parser.add_argument('--cv',    dest='use_cv', action='store_true')
    parser.add_argument('--no-cv', dest='use_cv', action='store_false')
    parser.set_defaults(use_cv=True)
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--interval', type=int, default=14,
                        help='Forecast interval in days (default: 14)')
    parser.add_argument('--output-dir', type=str,
                        default='results/future_predictions',
                        help='Output directory (default: results/future_predictions)')

    args = parser.parse_args()

    # ── Configuration banner ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("FUTURE WEIGHT PREDICTION")
    print("="*80)
    print(f"  Model type:      {args.model_type.upper()}")
    print(f"  Kalman:          {args.kalman}")
    print(f"  Interval:        {args.interval} days")
    print(f"  Output:          {args.output_dir}")
    print(f"  Cross-val:       {args.use_cv}  (k={args.k_folds})")
    print(f"  Today:           {date.today()}")
    print("="*80 + "\n")

    # ── Load & process data (once) ──────────────────────────────────────────
    processor = DataProcessing()
    dfs = processor.get_dfs(
        n_weighings=args.n_weighings,
        measurement_noise=args.measurement_noise,
        apply_smoothing=args.kalman,
        cut_tails=args.cut_tails,
    )

    # ── Select model configs ────────────────────────────────────────────────
    model_configs = OLS_models if args.model_type == 'ols' else models

    if args.model_name:
        if args.model_name not in model_configs:
            print(f"Error: '{args.model_name}' not in config. "
                  f"Available: {list(model_configs.keys())}")
            return
        model_configs = {args.model_name: model_configs[args.model_name]}

    # ── Run forecasts ───────────────────────────────────────────────────────
    all_results = []

    for model_name, model_config in model_configs.items():
        if model_config.get('pass', False) and args.model_name is None:
            continue

        # Build full model name (same logic as main.py)
        full_name = model_name
        if args.kalman:
            full_name = 'Kal_' + full_name
        elif not args.cut_tails:
            full_name = 'Raw_' + full_name
        if args.cut_tails:
            full_name = 'Cut_' + full_name

        for n, df in dfs.items():
            # Breed filter (same as main.py filter_breed)
            mn_lower = model_name.lower()
            if mn_lower.startswith('simental'):
                df_f = df[
                    df['breed'].isin(['Simental', 'Simmental']) &
                    (df['pred_adgLatest_average'] < 3)
                ].copy()
            elif mn_lower.startswith('limousine'):
                df_f = df[df['breed'].isin(['Limousin', 'Limousine'])].copy()
            else:
                df_f = df[df['breed'].isin(
                    ['Limousin', 'Limousine', 'Simental', 'Simmental']
                )].copy()

            unique = model_config.get('unique', False)
            if unique:
                df_f = df_f.drop_duplicates(subset='docId', keep='first').reset_index(drop=True)

            if len(df_f) == 0:
                print(f"  ⚠  No data for {full_name} (n={n})")
                continue

            result = predict_future_for_model(
                model_name    = f'{full_name}_n{n}',
                model_config  = model_config,
                model_type    = args.model_type,
                df            = df_f,
                processor     = processor,
                output_dir    = args.output_dir,
                kalman        = args.kalman,
                use_cv        = args.use_cv,
                k_folds       = args.k_folds,
                interval_days = args.interval,
            )
            if not result.empty:
                all_results.append(result)

    # ── Master combined file ─────────────────────────────────────────────────
    if all_results:
        master = pd.concat(all_results, ignore_index=True)
        master_path = Path(args.output_dir) / 'ALL_MODELS_future_predictions.csv'
        master.to_csv(master_path, index=False, float_format='%.4f')
        print(f"\n✓ Master CSV (all models) → {master_path}")

    print("\n" + "="*80)
    print("PREDICTION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
