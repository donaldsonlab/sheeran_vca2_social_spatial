"""
xsession_utils.py — Shared utilities for cross-session analysis.

Provides data loading, metric extraction, statistical testing, and plotting
functions used by the behavioral, population, and single-neuron summary notebooks.

All session metadata is imported from session_config.py which must live in the
same directory (or be on sys.path).

Usage in a notebook:
    import sys, os
    sys.path.insert(0, r"C:\\path\\to\\this\\directory")   # if needed
    from session_config import *
    from xsession_utils import *

Written by WMS and Claude Opus 4.6; all Claude-generated code reviewed by WMS
Last updated: WMS/Claude Code 9/19/26

"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests

from session_config import (
    SESSION_CONFIG,
    SESSION_SUBSETS,
    BEHAVIOR_RESULTS_REL,
    SINGLE_CELL_RESULTS_REL,
    POPULATION_RESULTS_DIR_REL,
    PPT_SESSION_MAP
)


# =============================================================================
# 1. Generic I/O Helpers
# =============================================================================

def load_pickle(filepath):
    """
    Safely load a pickle file.

    Returns None (without raising) if the file is missing or unreadable.
    """
    if filepath is None:
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


# =============================================================================
# 2. Behavioral Data Loading
# =============================================================================

def load_behavior_results(bento_path, filename=None):
    """
    Load a behavioral results pickle for a single session.

    Parameters
    ----------
    bento_path : str or None
        Path to the session's bento folder.
    filename : str, optional
        Relative path to the pickle within the bento folder.
        Defaults to BEHAVIOR_RESULTS_REL from session_config.

    Returns
    -------
    dict or None
    """
    if filename is None:
        filename = BEHAVIOR_RESULTS_REL
    if bento_path is None:
        return None
    return load_pickle(os.path.join(bento_path, filename))


def load_all_behavior_data(animal_paths_dict, filename=None, verbose=True):
    """
    Load behavioral results for every animal × session.

    Parameters
    ----------
    animal_paths_dict : dict
        {animal_id: [path_session0, path_session1, …, path_session5]}
    filename : str, optional
        Relative pickle path (default BEHAVIOR_RESULTS_REL).
    verbose : bool
        Print per-session status.

    Returns
    -------
    all_data : dict
        Nested dict  all_data[animal_id][session_idx] → results dict | None
    """
    if filename is None:
        filename = BEHAVIOR_RESULTS_REL

    all_data = {}
    for animal_id, session_paths in animal_paths_dict.items():
        all_data[animal_id] = {}
        for session_idx, path in enumerate(session_paths):
            results = load_behavior_results(path, filename)
            all_data[animal_id][session_idx] = results
            if verbose:
                label = SESSION_CONFIG['labels'][session_idx]
                status = "✓" if results is not None else "✗"
                print(f"  {animal_id} | {label}: {status}")
        if verbose:
            print()
    return all_data


# =============================================================================
# 2.5 Single-cell data loading
# =============================================================================

def load_session_single_cell_data(bento_path):
    if bento_path is None:
        return None
    return load_pickle(os.path.join(bento_path, SINGLE_CELL_RESULTS_REL))

def load_all_single_cell_data(animal_paths_dict, verbose=True):
    all_data = {}
    for animal_id, session_paths in animal_paths_dict.items():
        all_data[animal_id] = {}
        for s_idx, path in enumerate(session_paths):
            all_data[animal_id][s_idx] = load_session_single_cell_data(path)
            if verbose:
                sl = SESSION_CONFIG['short_labels'][s_idx]
                status = "✓" if all_data[animal_id][s_idx] is not None else "✗"
                print(f"  {animal_id} | {sl}: {status}")
    return all_data


# =============================================================================
# 3. Population Data Loading
# =============================================================================

def load_session_population_data(bento_path,decoding_types):
    """
    Load all population-level results (sparsity + all decoding types) for one session.

    Returns
    -------
    dict or None
        Keys include 'sparsity' and every key in DECODING_TYPES.
    """
    if bento_path is None:
        return None

    pop_path = os.path.join(bento_path, POPULATION_RESULTS_DIR_REL)
    if not os.path.exists(pop_path):
        print(f"  Population results folder not found: {pop_path}")
        return None

    data = {}
    data['sparsity'] = load_pickle(os.path.join(pop_path, 'neural_event_sparsity_metrics.pkl'))

    for decode_type, cfg in decoding_types.items():
        data[decode_type] = load_pickle(os.path.join(pop_path, cfg['filename']))

    data['mahalanobis_distance'] = load_pickle(os.path.join(pop_path, 'mahalanobis_distance_comparisons.pkl'))

    return data


def load_all_population_data(animal_paths_dict, decoding_types, verbose=True):
    """
    Load population results for every animal × session.

    Returns
    -------
    all_data : dict
        Nested dict  all_data[animal_id][session_idx] → session dict | None
    """
    all_data = {}
    for animal_id, session_paths in animal_paths_dict.items():
        if verbose:
            print(f"\nLoading {animal_id}...")
        all_data[animal_id] = {}
        for session_idx, path in enumerate(session_paths):
            session_data = load_session_population_data(path,decoding_types)
            all_data[animal_id][session_idx] = session_data
            if verbose:
                sl = SESSION_CONFIG['short_labels'][session_idx]
                if session_data is None:
                    print(f"  {sl}: ✗")
                else:
                    loaded = [k for k, v in session_data.items() if v is not None]
                    print(f"  {sl}: ✓ ({', '.join(loaded)})")
    return all_data


# =============================================================================
# 4. Metric Extraction — Behavioral
# =============================================================================

def _make_session_row(animal_id, session_idx):
    """Return a dict with the standard session-identifying columns."""
    return {
        'animal': animal_id,
        'session_idx': session_idx,
        'session_label': SESSION_CONFIG['labels'][session_idx],
        'short_label': SESSION_CONFIG['short_labels'][session_idx],
        'day': SESSION_CONFIG['days'][session_idx],
    }


def extract_metric(all_data, metric_name, session_indices=None, animals=None):
    """
    Extract a single behavioral metric across animals/sessions → tidy DataFrame.

    Parameters
    ----------
    all_data : dict
        Nested dict from load_all_behavior_data().
    metric_name : str
        Key in the per-session results dict (e.g. 'partner_preference_index').
    session_indices : list[int] or None
        Sessions to include (default: all 6).
    animals : list[str] or None
        Animal IDs to include (default: all).

    Returns
    -------
    pd.DataFrame
        Columns: animal, session_idx, session_label, short_label, day, value, metric
    """
    if session_indices is None:
        session_indices = list(range(len(SESSION_CONFIG['labels'])))
    if animals is None:
        animals = list(all_data.keys())

    rows = []
    for animal_id in animals:
        if animal_id not in all_data:
            continue
        for s_idx in session_indices:
            row = _make_session_row(animal_id, s_idx)
            session_data = all_data[animal_id].get(s_idx)
            if session_data is None:
                row['value'] = np.nan
            elif metric_name not in session_data:
                print(f"Warning: '{metric_name}' not found in {animal_id} session {s_idx}")
                row['value'] = np.nan
            else:
                row['value'] = session_data[metric_name]
            row['metric'] = metric_name
            rows.append(row)
    return pd.DataFrame(rows)


def extract_multiple_metrics(all_data, metric_names, session_indices=None, animals=None):
    """
    Extract several behavioral metrics into one tidy DataFrame.

    Returns a DataFrame with an additional 'metric' column distinguishing them.
    """
    dfs = [extract_metric(all_data, m, session_indices, animals) for m in metric_names]
    return pd.concat(dfs, ignore_index=True)


def extract_derived_metric(all_data, metric_name, func, session_indices=None, animals=None):
    """
    Extract a metric computed *from* the stored keys, rather than read from one.

    Same output schema as extract_metric(), so the result drops directly into
    results_compilation_loop and the plotting/stats helpers.

    Parameters
    ----------
    all_data : dict
        Nested dict from load_all_behavior_data().
    metric_name : str
        Name for the derived quantity (used for the 'metric' column and labels).
    func : callable
        func(session_dict) -> float. Receives one session's flat results dict.
        Raising KeyError for an absent input key is fine — it is caught and
        reported the same way extract_metric reports a missing key.
    session_indices : list[int] or None
        Sessions to include (default: all 6).
    animals : list[str] or None
        Animal IDs to include (default: all).

    Returns
    -------
    pd.DataFrame
        Columns: animal, session_idx, session_label, short_label, day, value, metric
    """
    if session_indices is None:
        session_indices = list(range(len(SESSION_CONFIG['labels'])))
    if animals is None:
        animals = list(all_data.keys())

    rows = []
    for animal_id in animals:
        if animal_id not in all_data:
            continue
        for s_idx in session_indices:
            row = _make_session_row(animal_id, s_idx)
            session_data = all_data[animal_id].get(s_idx)
            if session_data is None:
                row['value'] = np.nan
            else:
                try:
                    row['value'] = func(session_data)
                except KeyError as e:
                    print(f"Warning: {e} needed by '{metric_name}' "
                          f"not found in {animal_id} session {s_idx}")
                    row['value'] = np.nan
            row['metric'] = metric_name
            rows.append(row)
    return pd.DataFrame(rows)


# ── Derived behavioral quantities ────────────────────────────────────────────
# Each takes one session's flat results dict and returns a float.
# See extract_derived_metric() for why these are derived rather than stored.

def total_partner_int_sec(d):
    """Total seconds of active partner cup investigation."""
    return d['n_active_partner_ints'] * d['avg_active_partner_int_length']


def total_novel_int_sec(d):
    """Total seconds of active novel cup investigation."""
    return d['n_active_novel_ints'] * d['avg_active_novel_int_length']


def total_int_sec(d):
    """Total seconds of active cup investigation, both stimuli combined."""
    return total_partner_int_sec(d) + total_novel_int_sec(d)


# Nominal length of the "first five minutes" window, in seconds. behavior_analysis.ipynb
# cell 25 builds that window with get_offset_frame(..., time_offset=300, units='sec'), which
# picks the frame closest to start + 300 s -- so the true window is within one frame (~0.02 s)
# of this value.
S1_FIRST_FIVE_SEC = 300.0


def s1_first5_sec_partner(d):
    """Seconds of the first five minutes of s1 spent actively investigating the partner."""
    return (d['s1_first_five_min_n_partner_ints']
            * d['s1_first_five_min_avg_partner_int_length'])


def s1_first5_sec_novel(d):
    """Seconds of the first five minutes of s1 spent actively investigating the novel animal."""
    return (d['s1_first_five_min_n_novel_ints']
            * d['s1_first_five_min_avg_novel_int_length'])


def s1_first5_prop_time_partner(d):
    """Fraction of the first five minutes of s1 spent actively investigating the partner."""
    return s1_first5_sec_partner(d) / S1_FIRST_FIVE_SEC


def s1_first5_prop_time_novel(d):
    """Fraction of the first five minutes of s1 spent actively investigating the novel animal."""
    return s1_first5_sec_novel(d) / S1_FIRST_FIVE_SEC


def derived_extractor(func):
    """
    Wrap a derived-quantity function into an extract_metric-compatible callable.

    Lets results_compilation_loop take a derived metric through its `extractor`
    argument without special-casing:

        results_compilation_loop(..., metric='total_partner_int_sec',
                                 extractor=derived_extractor(total_partner_int_sec))
    """
    def _extractor(all_data, metric_name, session_indices=None, animals=None):
        return extract_derived_metric(all_data, metric_name, func,
                                      session_indices=session_indices,
                                      animals=animals)
    return _extractor


# =============================================================================
# 5. Metric Extraction — Population / Decoding + single cell
# =============================================================================

def extract_sparsity_metric(all_data, metric_name, session_indices=None, animals=None):
    """
    Extract a sparsity metric (e.g. 'median_rate', 'prop_pl_pts_empty') across
    animals/sessions → tidy DataFrame.

    Parameters
    ----------
    all_data : dict
        Output from load_all_population_data().
    metric_name : str
        Key inside the 'sparsity' sub-dict.
    """
    if session_indices is None:
        session_indices = list(range(len(SESSION_CONFIG['labels'])))
    if animals is None:
        animals = list(all_data.keys())

    rows = []
    for animal_id in animals:
        if animal_id not in all_data:
            continue
        for s_idx in session_indices:
            row = _make_session_row(animal_id, s_idx)
            session_data = all_data[animal_id].get(s_idx)
            if session_data is None or session_data.get('sparsity') is None:
                row['value'] = np.nan
            elif metric_name not in session_data['sparsity']:
                row['value'] = np.nan
            else:
                row['value'] = session_data['sparsity'][metric_name]
            row['metric'] = metric_name
            rows.append(row)
    return pd.DataFrame(rows)

def extract_MD_metric(all_data, metric_name, session_indices=None, animals=None):
    """
    Extract a metric from the Mahalanobis distance dictionaries across
    animals/sessions → tidy DataFrame.

    Parameters
    ----------
    all_data : dict
        Output from load_all_population_data().
    metric_name : str
        Key inside the 'mahalanobis_distance' sub-dict.
    """
    if session_indices is None:
        session_indices = list(range(len(SESSION_CONFIG['labels'])))
    if animals is None:
        animals = list(all_data.keys())

    rows = []
    for animal_id in animals:
        if animal_id not in all_data:
            continue
        for s_idx in session_indices:
            row = _make_session_row(animal_id, s_idx)
            session_data = all_data[animal_id].get(s_idx)
            if session_data is None or session_data.get('mahalanobis_distance') is None:
                row['value'] = np.nan
            elif metric_name not in session_data['mahalanobis_distance']:
                row['value'] = np.nan
            else:
                row['value'] = session_data['mahalanobis_distance'][metric_name]
            row['metric'] = metric_name
            rows.append(row)
    return pd.DataFrame(rows)


def extract_decoding_metric(all_data, decode_type, null_options, metric='accuracy',
                            session_indices=None, animals=None,
                            include_null=True, null_type='circular_aware'):
    """
    Extract decoding performance across animals/sessions → tidy DataFrame.

    Parameters
    ----------
    decode_type : str
        Key in DECODING_TYPES (e.g. 'identity', 'location_social_within').
    metric : str
        'accuracy', 'f1', or 'auroc'.
    include_null : bool
        Whether to attach null-distribution summary columns.
    null_type : str
        'circular', 'circular_aware', or 'shuffle'.

    Returns
    -------
    pd.DataFrame
        Always includes: animal, session_idx, short_label, day, decode_type,
        metric, true_value, true_std, n_neurons, total_pts,
        mean_train_pts_per_iter.
        If include_null: also null_mean, null_std, null_values, p_value.
    """
    if session_indices is None:
        session_indices = list(range(len(SESSION_CONFIG['labels'])))
    if animals is None:
        animals = list(all_data.keys())

    null_key = null_options.get(null_type, 'null_shift_aware_results')

    rows = []
    for animal_id in animals:
        if animal_id not in all_data:
            continue
        for s_idx in session_indices:
            row = _make_session_row(animal_id, s_idx)
            row['decode_type'] = decode_type
            row['metric'] = metric

            session_data = all_data[animal_id].get(s_idx)

            if session_data is None or session_data.get(decode_type) is None:
                row.update({
                    'true_value': np.nan, 'true_std': np.nan,
                    'null_mean': np.nan, 'null_std': np.nan,
                    'null_values': [], 'p_value': np.nan,
                    'n_neurons': np.nan, 'total_pts': np.nan,
                    'mean_train_pts_per_iter': np.nan,
                })
            else:
                decode_data = session_data[decode_type]
                true_results = decode_data.get('true_decoding_results', {})

                row['true_value'] = true_results.get(f'mean_{metric}', np.nan)
                row['true_std'] = true_results.get(f'std_{metric}', np.nan)
                row['n_neurons'] = decode_data.get('n_neurons_subsampled', np.nan)

                pts_keys = [k for k in decode_data.keys() if k.startswith('num_pts_')]
                row['total_pts'] = sum(decode_data.get(k, 0) for k in pts_keys)
                row['mean_train_pts_per_iter'] = np.mean(
                    true_results.get('n_train_per_iter', np.nan)
                )

                if include_null and null_key in decode_data:
                    null_results = decode_data[null_key]
                    null_metric_key = (
                        f'null_{metric}s' if metric != 'accuracy' else 'null_accuracies'
                    )
                    row['null_mean'] = null_results.get(f'mean_null_{metric}', np.nan)
                    row['null_std'] = null_results.get(f'std_null_{metric}', np.nan)
                    row['null_values'] = null_results.get(null_metric_key, [])

                    null_type_key = null_type.replace(
                        'circular_aware', 'shift_aware'
                    ).replace('circular', 'shift')
                    p_key = f'p_val_{null_type_key}_{metric}'
                    row['p_value'] = decode_data.get(p_key, np.nan)
                else:
                    row.update({
                        'null_mean': np.nan, 'null_std': np.nan,
                        'null_values': [], 'p_value': np.nan,
                    })

            rows.append(row)
    return pd.DataFrame(rows)


def extract_all_decoding_types(all_data, decoding_types, null_options, metric='accuracy', session_idx=None,
                               animals=None, null_type='circular_aware'):
    """
    Extract every decoding type for a single session → combined DataFrame.

    Useful for side-by-side comparison plots of location vs. identity.
    """
    if session_idx is None:
        raise ValueError("Must specify a single session_idx")
    dfs = []
    for dt in decoding_types:
        dfs.append(
            extract_decoding_metric(
                all_data, dt, metric=metric,
                session_indices=[session_idx], animals=animals,
                include_null=True, null_type=null_type, null_options=null_options
            )
        )
    return pd.concat(dfs, ignore_index=True)

def make_ppt_lookup(ppt_df, score_types=('cup', 'chamber')):
    """
    Reshape a wide PPT scores table into a tidy DataFrame
    keyed on (animal, session_idx) for easy merging with neural data.

    Parameters
    ----------
    ppt_df : pd.DataFrame
        The imported Excel sheet with columns like SS_ppref_score_cup, etc.
    score_types : tuple of str
        Which score variants to include ('cup', 'chamber', or both).

    Returns
    -------
    pd.DataFrame
        Columns: animal, session_idx, ppref_score_cup, ppref_score_chamber
    """
    rows = []
    for prefix, s_idx in PPT_SESSION_MAP.items():
        for _, animal_row in ppt_df.iterrows():
            row = {
                'animal': str(animal_row['Animal']),
                'session_idx': s_idx
            }
            for score_type in score_types:
                col = f'{prefix}_ppref_score_{score_type}'
                row[f'ppref_score_{score_type}'] = animal_row.get(col, np.nan)
            rows.append(row)

    return pd.DataFrame(rows)


def make_ppt_cup_time_frame(ppt_df, stimulus, metric_name=None, zone='cup'):
    """
    Pull one stimulus' raw PPT zone-occupancy column into an extract_metric()-shaped frame.

    make_ppt_lookup() returns the *preference score* (partner/(partner+novel)); this returns
    the underlying per-stimulus proportion.

    Parameters
    ----------
    ppt_df : pd.DataFrame
        The imported Excel sheet, with an 'Animal' column.
    stimulus : str
        'partner' or 'novel'.
    metric_name : str or None
        Value for the 'metric' column (default: f'ppt_{stimulus}_{zone}_time').
    zone : str
        'cup' or 'chamber'.

    Returns
    -------
    pd.DataFrame
        Columns: animal, session_idx, session_label, short_label, day, value, metric
    """
    if stimulus not in ('partner', 'novel'):
        raise ValueError(f"stimulus must be 'partner' or 'novel', got '{stimulus}'")
    if metric_name is None:
        metric_name = f'ppt_{stimulus}_{zone}_time'

    rows = []
    for prefix, s_idx in PPT_SESSION_MAP.items():
        col = f'{prefix}_{stimulus}_{zone}'
        if col not in ppt_df.columns:
            print(f"Warning: '{col}' not found in the PPT table; session {s_idx} will be NaN")
        for _, animal_row in ppt_df.iterrows():
            row = _make_session_row(str(animal_row['Animal']), s_idx)
            row['value'] = pd.to_numeric(animal_row.get(col, np.nan), errors='coerce')
            row['metric'] = metric_name
            rows.append(row)

    return pd.DataFrame(rows).sort_values(['animal', 'session_idx']).reset_index(drop=True)


def make_ppt_seconds_frame(ppt_long_df, stimulus, metric_name=None):
    """
    Pull one stimulus' PPT cup-occupancy *seconds* into an extract_metric()-shaped frame.

    Sibling of make_ppt_cup_time_frame(), which reads the original wide workbook
    (imaging_animal_ppt_scores.xlsx, one row per animal, {prefix}_{stimulus}_cup columns of
    proportions). This reads the reconstructed seconds workbook
    (good_imaging_animal_ppt_scores_seconds.xlsx), which is *long* -- one row per
    animal-session, with an explicit session_idx -- so the two layouts need separate readers.

    Parameters
    ----------
    ppt_long_df : pd.DataFrame
        The imported seconds workbook: one row per animal-session, with at least
        'animal', 'session_idx', and 'total_{stimulus}_cup_seconds' columns.
    stimulus : str
        'partner' or 'novel'.
    metric_name : str or None
        Value for the 'metric' column (default: f'ppt_{stimulus}_cup_seconds').

    Returns
    -------
    pd.DataFrame
        Columns: animal, session_idx, session_label, short_label, day, value, metric
    """
    if stimulus not in ('partner', 'novel'):
        raise ValueError(f"stimulus must be 'partner' or 'novel', got '{stimulus}'")
    if metric_name is None:
        metric_name = f'ppt_{stimulus}_cup_seconds'

    col = f'total_{stimulus}_cup_seconds'
    for required in ('animal', 'session_idx', col):
        if required not in ppt_long_df.columns:
            raise KeyError(f"'{required}' not found in the PPT seconds table; "
                           f"columns are {list(ppt_long_df.columns)}")

    expected_sessions = set(PPT_SESSION_MAP.values())

    rows = []
    for _, animal_row in ppt_long_df.iterrows():
        s_idx = int(animal_row['session_idx'])
        if s_idx not in expected_sessions:
            print(f"Warning: session_idx {s_idx} is not a PPT session "
                  f"({sorted(expected_sessions)}); check the workbook")

        # Labels come from SESSION_CONFIG, not the spreadsheet -- one source of truth. The
        # sheet carries its own short_label/day, so cross-check them and complain loudly
        # rather than silently preferring one.
        row = _make_session_row(str(animal_row['animal']), s_idx)
        for sheet_col, built_col in (('short_label', 'short_label'), ('day', 'day')):
            if sheet_col in ppt_long_df.columns and not pd.isna(animal_row[sheet_col]):
                sheet_val, config_val = animal_row[sheet_col], row[built_col]
                if str(sheet_val) != str(config_val):
                    print(f"Warning: {animal_row['animal']} session {s_idx} has "
                          f"{sheet_col}='{sheet_val}' in the workbook but "
                          f"'{config_val}' in SESSION_CONFIG; using SESSION_CONFIG")

        row['value'] = pd.to_numeric(animal_row[col], errors='coerce')
        row['metric'] = metric_name
        rows.append(row)

    return pd.DataFrame(rows).sort_values(['animal', 'session_idx']).reset_index(drop=True)


TUNING_CATEGORIES_SOCIAL = [
    'partner_cells', 'novel_cells', 'left_cells', 'right_cells',
    'partner_left_mixed', 'partner_right_mixed', 
    'novel_left_mixed', 'novel_right_mixed'
]

TUNING_CATEGORIES_EMPTY = ['left_empty1_cells', 'right_empty1_cells']

# Location categories for social trial 1 (same analysis approach as e1, but during social trial 1)
# Addresses reviewer request: compare location coding between e1 and s1
TUNING_CATEGORIES_SOCIAL1 = ['left_social1_cells', 'right_social1_cells']

def extract_tuning_proportions(all_sc_data, categories=None, 
                                filter_mixed_by_within_trial=False, collapse_mixed_categories=False):
    """
    Build a tidy df: one row per (animal, session, category).
    Columns: animal, session_idx, short_label, day, category, n_cells, numcells, proportion
    
    In addition to the standard categories, this function adds four collapsed
    mixed-cell categories that aggregate the four conjunctive subtypes:
        - partner_mixed = partner_left_mixed + partner_right_mixed
        - novel_mixed   = novel_left_mixed + novel_right_mixed
        - left_mixed    = partner_left_mixed + novel_left_mixed
        - right_mixed   = partner_right_mixed + novel_right_mixed
    
    Parameters
    ----------
    all_sc_data : dict
        Nested dict: all_sc_data[animal_id][session_idx] -> single_cell_tuning dict.
    categories : list[str] or None
        Standard categories to extract. Defaults to TUNING_CATEGORIES_SOCIAL +
        TUNING_CATEGORIES_EMPTY + TUNING_CATEGORIES_SOCIAL1.
    filter_mixed_by_within_trial : bool
        If True, restrict mixed cell categories (both standard and collapsed) to
        only those cells that passed the within-trial dF/F check stored in 
        single_cell_tuning['mixed_within_trial_dff_confirmed']. Cells that 
        failed the check are excluded. Pure tuning categories (partner_cells,
        novel_cells, left_cells, right_cells) and empty1 categories are 
        unaffected. If the within-trial check key is missing for a session,
        all mixed cells in that session are kept (with a warning printed).
    """
    if categories is None:
        categories = TUNING_CATEGORIES_SOCIAL + TUNING_CATEGORIES_EMPTY + TUNING_CATEGORIES_SOCIAL1
    
    # Categories that are mixed cell types (both standard and collapsed share these source keys)
    mixed_categories_standard = ['partner_left_mixed', 'partner_right_mixed',
                                  'novel_left_mixed', 'novel_right_mixed']
    
    collapsed_mixed = {
        'partner_mixed': ['partner_left_mixed', 'partner_right_mixed'],
        'novel_mixed':   ['novel_left_mixed', 'novel_right_mixed'],
        'left_mixed':    ['partner_left_mixed', 'novel_left_mixed'],
        'right_mixed':   ['partner_right_mixed', 'novel_right_mixed'],
    }
    
    def get_filtered_mixed_cells(sc_data, mixed_cat):
        """
        Return the cell list for a mixed category, filtered by within-trial 
        check if requested. Returns the unfiltered list if filtering is off
        or if the within-trial check is unavailable.
        """
        raw_cells = sc_data.get(mixed_cat, [])
        if not filter_mixed_by_within_trial:
            return raw_cells
        
        within_trial_dict = sc_data.get('mixed_within_trial_dff_confirmed')
        if within_trial_dict is None:
            return raw_cells
        
        cat_check = within_trial_dict.get(mixed_cat)
        if cat_check is None or 'confirmed' not in cat_check:
            return raw_cells
        
        confirmed_set = set(cat_check['confirmed'])
        return [c for c in raw_cells if c in confirmed_set]
    
    rows = []
    for animal_id, sessions in all_sc_data.items():
        for s_idx, sc_data in sessions.items():
            if sc_data is None:
                continue
            numcells = sc_data.get('numcells', np.nan)
            
            # Standard categories
            for cat in categories:
                if cat not in sc_data:
                    continue
                
                # Apply within-trial filter only to mixed categories
                if cat in mixed_categories_standard:
                    cell_list = get_filtered_mixed_cells(sc_data, cat)
                else:
                    cell_list = sc_data[cat]
                
                row = _make_session_row(animal_id, s_idx)
                row['category'] = cat
                row['n_tuned_cells'] = len(cell_list)
                row['n_total_cells'] = numcells
                row['proportion'] = len(cell_list) / numcells if numcells > 0 else np.nan
                rows.append(row)
            
            # Collapsed mixed categories
            if collapse_mixed_categories:
                for collapsed_name, source_cats in collapsed_mixed.items():
                    if not any(sc in sc_data for sc in source_cats):
                        continue
                
                    combined_cells = set()
                    for sc in source_cats:
                        if sc in sc_data:
                            combined_cells.update(get_filtered_mixed_cells(sc_data, sc))
                
                    row = _make_session_row(animal_id, s_idx)
                    row['category'] = collapsed_name
                    row['n_tuned_cells'] = len(combined_cells)
                    row['n_total_cells'] = numcells
                    row['proportion'] = len(combined_cells) / numcells if numcells > 0 else np.nan
                    rows.append(row)
    
    return pd.DataFrame(rows)

def collapse_tuning_categories(tuning_df):
    """Add aggregate categories: any_identity, any_location, any_mixed, any_tuned."""
    pivot = tuning_df.pivot_table(
        index=['animal', 'session_idx', 'short_label', 'day'],
        columns='category', values='n_tuned_cells', fill_value=0
    ).reset_index()
    
    pivot['any_identity'] = pivot.get('partner_cells', 0) + pivot.get('novel_cells', 0)
    pivot['any_location'] = pivot.get('left_cells', 0) + pivot.get('right_cells', 0)
    pivot['any_mixed'] = (pivot.get('partner_left_mixed', 0) + pivot.get('partner_right_mixed', 0) +
                         pivot.get('novel_left_mixed', 0) + pivot.get('novel_right_mixed', 0))
    pivot['any_tuned'] = pivot['any_identity'] + pivot['any_location'] + pivot['any_mixed']
    return pivot

def summarize_tuning_proportions(tuning_df):
    summ_df = (tuning_df.groupby(['session_idx', 'short_label', 'category'])
            .agg(mean_prop=('proportion', 'mean'),
                 sem_prop=('proportion', lambda x: x.std() / np.sqrt(x.count())),
                 n_animals=('proportion', 'count'),
                 total_n_tuned_cells=('n_tuned_cells', 'sum'),
                 total_all_recorded_cells=('n_total_cells','sum'))
            .reset_index())
    summ_df['total_prop']=summ_df['total_n_tuned_cells']/summ_df['total_all_recorded_cells']
    return summ_df

def compute_supertype_proportions(tuning_df):
    """
    Compute per-animal, per-session proportions for identity-tuned, 
    location-tuned, and mixed-selective supertypes by summing the 
    contributing standard categories.
    
    Parameters
    ----------
    tuning_df : pd.DataFrame
        Output of extract_tuning_proportions() with per-animal rows.
    
    Returns
    -------
    pd.DataFrame
        Same tidy format as tuning_df, with three new category values:
        'identity_supertype', 'location_supertype', 'mixed_supertype'.
    """
    supertype_groups = {
        'identity_supertype': ['partner_cells', 'novel_cells'],
        'location_supertype': ['left_cells', 'right_cells'],
        'mixed_supertype':    ['partner_left_mixed', 'partner_right_mixed',
                               'novel_left_mixed', 'novel_right_mixed'],
    }
    
    rows = []
    for (animal, s_idx), group in tuning_df.groupby(['animal', 'session_idx']):
        for super_name, source_cats in supertype_groups.items():
            matching = group[group['category'].isin(source_cats)]
            if len(matching) == 0:
                continue
            row = {
                'animal': animal,
                'session_idx': s_idx,
                'short_label': matching['short_label'].iloc[0],
                'day': matching['day'].iloc[0],
                'category': super_name,
                'n_tuned_cells': matching['n_tuned_cells'].sum(),
                'n_total_cells': matching['n_total_cells'].iloc[0],
            }
            row['proportion'] = row['n_tuned_cells'] / row['n_total_cells'] if row['n_total_cells'] > 0 else np.nan
            rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# 6. DataFrame Reshaping Helpers
# =============================================================================

def pivot_for_comparison(df, session_indices):
    """
    Pivot a tidy metric DataFrame for pairwise comparison between exactly 2 sessions.
    The df must contain values from exactly one metric

    Parameters
    ----------
    df : pd.DataFrame
        Output from extract_metric() (must contain 'value' column).
    session_indices : list of 2 ints

    Returns
    -------
    pd.DataFrame
        Animals as rows, session short_labels as columns, values as entries.
    """
    if len(session_indices) != 2:
        raise ValueError("session_indices must contain exactly 2 sessions")
    subset = df[df['session_idx'].isin(session_indices)].copy()
    pivot_df = subset.pivot(index='animal', columns='short_label', values='value')
    col_order = [SESSION_CONFIG['short_labels'][i] for i in session_indices]
    return pivot_df[col_order]

def merge_ppt_scores_neural_data(neural_data_df,ppt_df):
    """
    fxn to merge a tidy dataframe with partner preference scores by animal and session index to a tidy neural data df with a given metric by animal and session idx
    neural_data_df must have 'animal','session_idx', and 'value' in its columns
    ppt_df must have 'animal', 'session_idx', 'ppref_score_cup' and 'ppref_score_chamber' in its columns
    
    """
    merged_df = neural_data_df.merge(ppt_df,on=['animal','session_idx'],how='left')
    merged_df = merged_df.dropna(subset=['value','ppref_score_cup','ppref_score_chamber']).reset_index(drop=True)
    return merged_df


# =============================================================================
# 7. Statistical Testing
# =============================================================================

def paired_comparison(df, session_indices, test='ttest', alternative='two-sided'):
    """
    Paired test between two sessions on a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Output from extract_metric() (single metric).
    session_indices : list of 2 ints
    test : str
        'ttest' or 'wilcoxon'.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    dict or None
    """
    pivot_df = pivot_for_comparison(df, session_indices)
    complete_df = pivot_df.dropna()

    if len(complete_df) <= 2:
        print(f"Warning: Only {len(complete_df)} animals have complete data for both sessions; need >= 3")
        return None

    col1, col2 = complete_df.columns
    v1, v2 = complete_df[col1].values, complete_df[col2].values

    if np.all(v1 == v2): #tests would throw errors if all values in both columns are equal --> handle this gracefully
        return {
            'test': 'not run (all values equal)',
            'alternative': alternative,
            'statistic': 0.0,
            'p_value': 1.0,
            'n': len(complete_df),
            'session_labels': [col1, col2],
            'means': [v1.mean(), v2.mean()],
            'stds': [v1.std(), v2.std()],
            'difference_mean': 0.0,
            'difference_std': 0.0,
            'data': complete_df,
        }

    if test == 'ttest':
        stat, p_val = stats.ttest_rel(v1, v2, alternative=alternative)
        test_name = 'Paired t-test'
    elif test == 'wilcoxon':
        stat, p_val = stats.wilcoxon(v1, v2, alternative=alternative)
        test_name = 'Wilcoxon signed-rank'
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        'test': test_name,
        'alternative': alternative,
        'statistic': stat,
        'p_value': p_val,
        'n': len(complete_df),
        'session_labels': [col1, col2],
        'means': [v1.mean(), v2.mean()],
        'stds': [v1.std(), v2.std()],
        'difference_mean': (v2 - v1).mean(),
        'difference_std': (v2 - v1).std(),
        'data': complete_df,
    }


def one_sample_vs_chance(df, session_idx, chance_level=0.5,
                         test='ttest', alternative='two-sided'):
    """
    One-sample test of a behavioural metric against a chance level.

    Parameters
    ----------
    df : pd.DataFrame
        Output from extract_metric().
    session_idx : int
    chance_level : float
    test : str
        'ttest' or 'wilcoxon'.

    Returns
    -------
    dict or None
    """
    subset = df[df['session_idx'] == session_idx].dropna(subset=['value'])
    values = subset['value'].values

    if len(values) <= 2:
        print(f"Warning: Only {len(values)} animals have data for session {session_idx}; need >= 3")
        return None

    if test == 'ttest':
        stat, p_val = stats.ttest_1samp(values, chance_level, alternative=alternative)
        test_name = 'One-sample t-test'
    elif test == 'wilcoxon':
        stat, p_val = stats.wilcoxon(values - chance_level, alternative=alternative)
        test_name = 'One-sample Wilcoxon'
    else:
        raise ValueError(f"Unknown test: {test}")

    session_label = SESSION_CONFIG['short_labels'][session_idx]
    return {
        'test': test_name,
        'alternative': alternative,
        'chance_level': chance_level,
        'statistic': stat,
        'p_value': p_val,
        'n': len(values),
        'session_label': session_label,
        'mean': values.mean(),
        'std': values.std(),
        'sem': values.std() / np.sqrt(len(values)),
    }

def friedmantest_across_sessions(df):
    """
    Function to take in any df with the values of some metric across combinations of animals and sessions and perform a Friedman test for repeated samples
    """
    
    # Pivot so each column is a session, each row is an animal
    pivot = df.pivot(index='animal',columns='session_idx',values='value').dropna()
    
    n_dropped = df['animal'].nunique() - pivot.shape[0]
    if n_dropped > 0:
        print(f"  Note: {n_dropped} animal(s) dropped due to incomplete sessions")

    # Unpack columns as separate arrays
    arrays = [pivot[col].values for col in pivot.columns]
    stat, p = friedmanchisquare(*arrays)

    res = {
        'statistic':stat,
        'p_value':p,
        'n_animals':pivot.shape[0],
        'n_sessions':pivot.shape[1],
           }

    return res

def compare_metrics_within_sessions(df_a, df_b, label_a='Metric A', label_b='Metric B',
                                     session_indices=None, test='wilcoxon',
                                     alternative='two-sided', addl_info=None, alpha=0.05):
    """
    Compare two metrics within each session across animals (paired test).
    """
    if session_indices is None:
        session_indices = sorted(
            set(df_a['session_idx'].unique()) & set(df_b['session_idx'].unique())
        )

    merged = df_a[['animal', 'session_idx', 'value']].merge(
        df_b[['animal', 'session_idx', 'value']],
        on=['animal', 'session_idx'], suffixes=(f'_{label_a}', f'_{label_b}')
    )

    rows = []
    for s_idx in session_indices:
        s_data = merged[merged['session_idx'] == s_idx].dropna()

        row = {
            'session_idx': s_idx,
            'short_label': SESSION_CONFIG['short_labels'][s_idx],
            'n': len(s_data),
            f'mean_{label_a}': s_data[f'value_{label_a}'].mean() if len(s_data) > 0 else np.nan,
            f'mean_{label_b}': s_data[f'value_{label_b}'].mean() if len(s_data) > 0 else np.nan,
        }
        if addl_info:
            row['info']=addl_info

        if len(s_data) > 2:
            v_a = s_data[f'value_{label_a}'].values
            v_b = s_data[f'value_{label_b}'].values

            if test == 'wilcoxon':
                stat, p = stats.wilcoxon(v_a, v_b, alternative=alternative)
                row['test'] = 'Wilcoxon signed-rank'
            elif test == 'ttest':
                stat, p = stats.ttest_rel(v_a, v_b, alternative=alternative)
                row['test'] = 'Paired t-test'
            else:
                raise ValueError(f"Unknown test: {test}")

            row['statistic'] = stat
            row['p_value'] = p
        else:
            row['statistic'] = np.nan
            row['p_value'] = np.nan
            row['test'] = test

        rows.append(row)

    results_df = pd.DataFrame(rows)

    valid_mask = results_df['p_value'].notna()
    if valid_mask.sum() > 0:
        rejected, p_corr, _, _ = multipletests(
            results_df.loc[valid_mask, 'p_value'], alpha=alpha, method='fdr_bh')
        results_df['p_fdr'] = np.nan
        results_df.loc[valid_mask, 'p_fdr'] = p_corr
        results_df['significant'] = False
        results_df.loc[valid_mask, 'significant'] = rejected
    results_df = results_df.rename(columns={'p_value': 'p_uncorrected'})

    return results_df

def compare_decoding_sessions(all_data, decode_type, session_indices, null_options, decoding_types,
                              metric='accuracy', test='ttest',
                              alternative='two-sided', min_pseudotrials=None, verbose=True):
    """
    Paired comparison of decoding performance between two sessions.

    Parameters
    ----------
    all_data : dict
        Output from load_all_population_data().
    decode_type : str
    session_indices : list of 2 ints
    metric : str
    test : str
        'ttest' or 'wilcoxon'.
    verbose : bool

    Returns
    -------
    dict or None
    """
    if len(session_indices) != 2:
        raise ValueError("Must provide exactly 2 session indices")

    df = extract_decoding_metric(all_data, decode_type, null_options=null_options, metric=metric,
                                 session_indices=session_indices) #note -- defaults to all animals, include_null=True, null_type='circular_aware'
    if min_pseudotrials is not None:
        df = df[df['mean_train_pts_per_iter']>=min_pseudotrials]

    pivot = df.pivot(index='animal', columns='short_label', values='true_value').dropna()

    if len(pivot) < 2:
        print(f"Insufficient data: only {len(pivot)} animals with both sessions")
        return None

    col1, col2 = pivot.columns
    v1, v2 = pivot[col1].values, pivot[col2].values

    if test == 'ttest':
        stat, p = stats.ttest_rel(v1, v2, alternative=alternative)
        test_name = 'Paired t-test'
    elif test == 'wilcoxon':
        stat, p = stats.wilcoxon(v1, v2, alternative=alternative)
        test_name = 'Wilcoxon'
    else:
        raise ValueError(f"Unknown test: {test}")

    results = {
        'test': test_name, 'decode_type': decode_type, 'metric': metric,
        'sessions': [col1, col2],
        'means': [v1.mean(), v2.mean()], 'stds': [v1.std(), v2.std()],
        'statistic': stat, 'p_value': p, 'n': len(pivot),
    }

    if verbose:
        _print_decoding_comparison(results,decoding_types=decoding_types)
    return results


def test_decoding_vs_chance(all_data, decode_type, decoding_types, session_idx, null_options,
                            metric='accuracy', chance=0.5,
                            test='ttest', alternative='greater', verbose=True):
    """
    Test whether decoding performance exceeds chance for one session.
    """
    df = extract_decoding_metric(all_data, decode_type, null_options=null_options, metric=metric,
                                 session_indices=[session_idx])
    values = df['true_value'].dropna().values

    if len(values) < 2:
        print(f"Insufficient data: only {len(values)} animals")
        return None

    if test == 'ttest':
        stat, p = stats.ttest_1samp(values, chance, alternative=alternative)
        test_name = 'One-sample t-test'
    elif test == 'wilcoxon':
        stat, p = stats.wilcoxon(values - chance, alternative=alternative)
        test_name = 'One-sample Wilcoxon'
    else:
        raise ValueError(f"Unknown test: {test}")

    session_label = SESSION_CONFIG['short_labels'][session_idx]

    if verbose:
        print(f"\n{decoding_types[decode_type]['label']} — {session_label}")
        print(f"{test_name} vs chance ({chance}), n={len(values)}")
        print(f"Mean: {values.mean():.4f} ± {values.std():.4f}")
        print(f"Statistic: {stat:.4f}, p = {p:.4f}")
        print(f"Significance: {_sig_stars(p)}\n")

    return {'test': test_name, 'mean': values.mean(), 'std': values.std(),
            'statistic': stat, 'p_value': p, 'n': len(values)}


def test_decoding_group_significance(all_data, decode_type, decoding_types, metric='accuracy',
                                     session_indices=None, null_type='circular_aware',
                                     animals=None, alpha=0.05, min_pseudotrials=None, verbose=True):
    """
    Wilcoxon signed-rank tests (true > null) per session with BH-FDR correction.

    For each session, computes paired differences (true_mean − null_mean) per
    animal and runs a one-sided Wilcoxon test. Applies Benjamini-Hochberg FDR
    correction across the tested sessions. Has optional functionality to filter sessions by a
    minimum threshold of mean # of pseudotrials available to the SVM for training.

    Returns
    -------
    pd.DataFrame
        One row per session with group-level statistics, uncorrected and
        FDR-corrected p-values, and per-animal data lists.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    if session_indices is None:
        session_indices = [1, 2, 3, 4, 5]

    null_key_map = {
        'circular': 'null_shift_results',
        'circular_aware': 'null_shift_aware_results',
        'shuffle': 'null_shuffle_results',
    }
    null_results_key = null_key_map.get(null_type)
    if null_results_key is None:
        raise ValueError(f"null_type must be one of {list(null_key_map.keys())}")

    null_metric_key = f'null_{metric}s' if metric != 'accuracy' else 'null_accuracies'

    if animals is None:
        animals = list(all_data.keys())

    session_results = []
    for s_idx in session_indices:
        true_values, null_means, animal_ids = [], [], []
        for animal_id in animals:
            session_data = all_data.get(animal_id, {}).get(s_idx)
            if session_data is None:
                continue
            decode_data = session_data.get(decode_type)
            if decode_data is None:
                continue
            true_results = decode_data.get('true_decoding_results', {})
            true_val = true_results.get(f'mean_{metric}')
            if true_val is None or np.isnan(true_val):
                continue
            null_results = decode_data.get(null_results_key)
            if null_results is None:
                continue
            null_vals = null_results.get(null_metric_key, [])
            if len(null_vals) == 0:
                continue

            if min_pseudotrials is not None:
                mean_train_pts = np.mean(true_results.get('n_train_per_iter', [np.nan]))
                if np.isnan(mean_train_pts) or mean_train_pts < min_pseudotrials:
                    continue

            true_values.append(true_val)
            null_means.append(np.mean(null_vals))
            animal_ids.append(animal_id)

        true_values = np.array(true_values)
        null_means_arr = np.array(null_means)
        diffs = true_values - null_means_arr
        n = len(diffs)

        row = {
            'session_idx': s_idx,
            'short_label': SESSION_CONFIG['short_labels'][s_idx],
            'n_animals': n,
            'mean_true': np.mean(true_values) if n > 0 else np.nan,
            'mean_null': np.mean(null_means_arr) if n > 0 else np.nan,
            'mean_diff': np.mean(diffs) if n > 0 else np.nan,
            'animal_ids': animal_ids,
            'animal_true_values': true_values.tolist(),
            'animal_null_means': null_means_arr.tolist(),
            'animal_diffs': diffs.tolist(),
        }

        if n >= 4:
            try:
                stat, p = wilcoxon(diffs, alternative='greater')
                row['wilcoxon_stat'] = stat
                row['p_uncorrected'] = p
            except ValueError:
                row['wilcoxon_stat'] = np.nan
                row['p_uncorrected'] = np.nan
        else:
            row['wilcoxon_stat'] = np.nan
            row['p_uncorrected'] = np.nan
            if verbose and n > 0:
                print(f"  {row['short_label']}: n={n}, too few for Wilcoxon")

        session_results.append(row)

    results_df = pd.DataFrame(session_results)

    # BH-FDR correction
    valid_mask = results_df['p_uncorrected'].notna()
    if valid_mask.sum() > 0:
        valid_pvals = results_df.loc[valid_mask, 'p_uncorrected'].values
        rejected, p_corrected, _, _ = multipletests(valid_pvals, alpha=alpha, method='fdr_bh')
        results_df['p_fdr'] = np.nan
        results_df.loc[valid_mask, 'p_fdr'] = p_corrected
        results_df['significant'] = False
        results_df.loc[valid_mask, 'significant'] = rejected
    else:
        results_df['p_fdr'] = np.nan
        results_df['significant'] = False

    if verbose:
        decode_label = decoding_types[decode_type]['label']
        print(f"\n{'='*70}")
        print(f"Group-level significance: {decode_label} ({metric})")
        print(f"Null model: {null_type} | FDR alpha: {alpha}")
        print(f"{'='*70}")
        for _, r in results_df.iterrows():
            n = r['n_animals']
            lbl = r['short_label']
            if n == 0:
                print(f"  {lbl}: No data")
                continue
            sig = ''
            if not np.isnan(r.get('p_fdr', np.nan)):
                sig = ' ' + _sig_stars(r['p_fdr'])
            print(f"\n  {lbl} (n={n}):")
            print(f"    True:  {r['mean_true']:.4f}")
            print(f"    Null:  {r['mean_null']:.4f}")
            print(f"    Diff:  {r['mean_diff']:.4f}")
            if not np.isnan(r.get('p_uncorrected', np.nan)):
                print(f"    p (uncorrected): {r['p_uncorrected']:.4f}")
                print(f"    p (FDR):         {r['p_fdr']:.4f}{sig}")
            else:
                print(f"    Wilcoxon: not run (n < 4)")
        print(f"\n{'='*70}")

    return results_df

def correlate_across_within_timepoints(df,x_col,y_col,comparison='across_sessions',
                                       session_indices=None,min_n=3,addl_info=None,alpha=0.05):
    """
    Compute Pearson correlations between two columns within each session or across sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'session_idx', x_col, and y_col columns.
    x_col : str
        Column name for the predictor (e.g. 'mean_train_pts_per_iter' or 'ppref_score_cup').
    y_col : str
        Column name for the outcome (e.g. 'value').
    comparison : str
        'across_sessions' or 'within_sessions'. Indicates whether to pool data from across all sessions to perform one Pearson's correlation 
        or to leave them segregated by session and to perform a Pearson's correlation on each session individually. Will also perform a Benjamini-Hochberg
        correction for multiple comparisons if the latter.
    session_indices : list[int] or None
        Sessions to include. If None, uses all unique sessions in df.
    min_n : int
        Minimum number of valid data points per session to run correlation.
    addl_info : str
        Any additional label to append to the output df
    Returns
    -------
    pd.DataFrame
        One row per session with columns: session_idx, short_label, n, r, p_value,
        x_mean, y_mean.
    """
    if session_indices is None: 
        session_indices = sorted(df['session_idx'].unique())

    rows = []
    data = df.dropna(subset=[x_col, y_col])
    data = data[data['session_idx'].isin(session_indices)]

    if comparison=='across_sessions':
        row = {'n':len(data)}
        if addl_info:
            row['info']=addl_info
        if len(data) >= min_n:
            r, p = stats.pearsonr(data[x_col], data[y_col])
            row['r'] = r
            row['p_value'] = p
        else:
            row['r'] = np.nan
            row['p_value'] = np.nan
            print(f"n={len(data)}, too few for correlation (need >= {min_n})")
        row[f'{x_col}_mean'] = data[x_col].mean() if len(data) > 0 else np.nan
        row[f'{x_col}_std'] = data[x_col].std() if len(data) > 0 else np.nan
        row[f'{y_col}_mean'] = data[y_col].mean() if len(data) > 0 else np.nan
        row[f'{y_col}_std'] = data[y_col].std() if len(data) > 0 else np.nan
        rows.append(row)

    elif comparison=='within_sessions':
        for s_idx in session_indices:
            s_data = data[data['session_idx'] == s_idx]
            row = {
                'n':len(s_data), 
                'session_idx':s_idx, 
                'short_label':SESSION_CONFIG['short_labels'][s_idx], 
                }
            if addl_info:
                row['info']=addl_info
            if len(s_data) >= min_n:
                r, p = stats.pearsonr(s_data[x_col], s_data[y_col])
                row['r'] = r
                row['p_value'] = p
            else:
                row['r'] = np.nan
                row['p_value'] = np.nan
                print(f"  {row['short_label']}: n={len(s_data)}, too few for correlation (need >= {min_n})")

            row[f'{x_col}_mean'] = s_data[x_col].mean() if len(s_data) > 0 else np.nan
            row[f'{x_col}_std'] = s_data[x_col].std() if len(s_data) > 0 else np.nan
            row[f'{y_col}_mean'] = s_data[y_col].mean() if len(s_data) > 0 else np.nan
            row[f'{y_col}_std'] = s_data[y_col].std() if len(s_data) > 0 else np.nan
            rows.append(row)
            
    else:
        raise ValueError(f"comparison must be 'across_sessions' or 'within_sessions', got {comparison}")
    
    summary_df = pd.DataFrame(rows)

    if comparison=='within_sessions': #Benjamini-Hochberg correction if multiple comparisons
        valid_mask = summary_df['p_value'].notna()
        if valid_mask.sum() > 0:
            rejected, p_corrected, _, _ = multipletests(
                summary_df.loc[valid_mask, 'p_value'], alpha=alpha, method='fdr_bh')
            summary_df['p_fdr'] = np.nan
            summary_df.loc[valid_mask, 'p_fdr'] = p_corrected
            summary_df['significant'] = False
            summary_df.loc[valid_mask, 'significant'] = rejected
        summary_df = summary_df.rename(columns={'p_value':'p_uncorrected'})

    return summary_df
# =============================================================================
# 8. Pretty-Print Helpers
# =============================================================================

def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


def print_comparison_results(results, metric_name=''):
    """Pretty-print output from paired_comparison() or one_sample_vs_chance()."""
    if results is None:
        print("No results to display (insufficient data)")
        return

    print(f"\n{'='*60}")
    if metric_name:
        print(f"Metric: {metric_name}")
    print(f"Test: {results['test']} ({results['alternative']})")
    print(f"n = {results['n']} animals")
    print(f"-"*60)

    if 'session_labels' in results:  # paired
        s1, s2 = results['session_labels']
        m1, m2 = results['means']
        sd1, sd2 = results['stds']
        print(f"{s1}: {m1:.4f} ± {sd1:.4f}")
        print(f"{s2}: {m2:.4f} ± {sd2:.4f}")
        print(f"Difference ({s2} − {s1}): "
              f"{results['difference_mean']:.4f} ± {results['difference_std']:.4f}")
    else:  # one-sample
        print(f"{results['session_label']}: {results['mean']:.4f} ± {results['std']:.4f}")
        print(f"vs chance level: {results['chance_level']}")

    print(f"-"*60)
    print(f"Statistic: {results['statistic']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Significance: {_sig_stars(results['p_value'])}")
    print(f"{'='*60}\n")


def _print_decoding_comparison(results,decoding_types):
    """Pretty-print output from compare_decoding_sessions()."""
    dt = results['decode_type']
    print(f"\n{'='*60}")
    print(f"{decoding_types[dt]['label']} — {results['metric']}")
    print(f"{results['test']}, n={results['n']}")
    print(f"-"*60)
    c1, c2 = results['sessions']
    m1, m2 = results['means']
    s1, s2 = results['stds']
    print(f"{c1}: {m1:.4f} ± {s1:.4f}")
    print(f"{c2}: {m2:.4f} ± {s2:.4f}")
    print(f"-"*60)
    print(f"Statistic: {results['statistic']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Significance: {_sig_stars(results['p_value'])}")
    print(f"{'='*60}\n")


# =============================================================================
# 9. Utility: List Available Metrics
# =============================================================================

def list_available_metrics(all_data):
    """
    Scan a nested data dict and print every unique key found.

    Works for behavioral data (flat dicts) and population data (nested dicts).
    """
    all_keys = set()
    for animal_id, sessions in all_data.items():
        for s_idx, data in sessions.items():
            if data is None:
                continue
            if isinstance(data, dict):
                all_keys.update(data.keys())
    print("Available metrics:\n")
    for k in sorted(all_keys):
        print(f"  - {k}")
    return sorted(all_keys)

# =============================================================================
#10. Single-cell confound checks
# =============================================================================

def extract_sampling_balance(all_sc_data):
    """Pull the sampling_df from each session into a tidy df."""
    rows = []
    for animal_id, sessions in all_sc_data.items():
        for s_idx, sc_data in sessions.items():
            if sc_data is None:
                continue
            sampling_df = sc_data.get('significant_cells_by_behavior_sampling_per_comparison')
            if sampling_df is None:
                continue
            for comp_label, comp_row in sampling_df.iterrows():
                row = _make_session_row(animal_id, s_idx)
                row['comparison'] = comp_label
                row['ratio'] = comp_row['ratio']
                row['n_sig_class0'] = comp_row['n_sig_class0']
                row['n_sig_class1'] = comp_row['n_sig_class1']
                # Asymmetry: positive = more sig cells for class with more frames
                row['sig_asymmetry'] = comp_row['n_sig_class1'] - comp_row['n_sig_class0']
                row['log_ratio'] = np.log(comp_row['ratio']) if comp_row['ratio'] > 0 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def extract_e1_s1_interaction_time(animal_paths_dict, session_indices=None, verbose=True):
    """
    Total active cup-interaction time during e1 and s1, per animal/session.

    Parameters
    ----------
    animal_paths_dict : dict
        ANIMAL_PATHS-shaped dict (e.g. ANIMAL_PATHS_REVISIONS).
    session_indices : list[int] or None
        Sessions to include (default: all).
    verbose : bool
        Print a line for each session whose files could not be read.

    Returns
    -------
    pd.DataFrame
        One row per (animal, session, trial_type). Columns: animal, session_idx,
        session_label, short_label, day, trial_type ('e1' or 's1'), left_sec,
        right_sec, total_sec, n_bouts
    """
    if session_indices is None:
        session_indices = list(range(len(SESSION_CONFIG['labels'])))

    trial_files = [
        ('e1', 'active_leftcup_interactions_empty1.csv',
               'active_rightcup_interactions_empty1.csv'),
        ('s1', 'active_leftcup_interactions_socialtrials.csv',
               'active_rightcup_interactions_socialtrials.csv'),
    ]

    rows = []
    for animal_id, session_paths in animal_paths_dict.items():
        for s_idx in session_indices:
            bento_path = session_paths[s_idx]
            if bento_path is None:
                continue

            prep_path = os.path.join(bento_path, 'analysis_results', 'data_prep')
            raster_file = os.path.join(prep_path, 'aligned_rasters.csv')
            if not os.path.exists(raster_file):
                if verbose:
                    print(f"  {animal_id} session {s_idx}: no aligned_rasters.csv, skipping")
                continue

            beh_rast = pd.read_csv(raster_file, index_col=0, header=0)
            s1_cols = beh_rast.columns[beh_rast.loc['social1'] == 1]
            s1_start, s1_end = int(s1_cols[0]), int(s1_cols[-1])

            for trial_type, left_file, right_file in trial_files:
                sides = {}
                for side, fname in [('left', left_file), ('right', right_file)]:
                    fpath = os.path.join(prep_path, fname)
                    if not os.path.exists(fpath):
                        break
                    bouts = pd.read_csv(fpath, index_col=0, header=0)
                    if trial_type == 's1':
                        # the social CSVs span s1 and s2 -- keep only bouts inside s1
                        bouts = bouts[(bouts['start'] >= s1_start) &
                                      (bouts['end'] <= s1_end)]
                    sides[side] = bouts

                if len(sides) < 2:
                    if verbose:
                        print(f"  {animal_id} session {s_idx}: missing a {trial_type} "
                              f"bout file, skipping {trial_type}")
                    continue

                row = _make_session_row(animal_id, s_idx)
                row['trial_type'] = trial_type
                row['left_sec'] = float(sides['left']['length'].sum())
                row['right_sec'] = float(sides['right']['length'].sum())
                row['total_sec'] = row['left_sec'] + row['right_sec']
                row['n_bouts'] = int(len(sides['left']) + len(sides['right']))
                rows.append(row)

    return pd.DataFrame(rows)


def pearson_ci(r, n, alpha=0.05):
    """
    Fisher z-transform confidence interval for a Pearson correlation.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient.
    n : int
        Number of paired observations.
    alpha : float
        Two-sided significance level (default 0.05 -> 95% CI).

    Returns
    -------
    (lo, hi) : tuple of float
        Interval bounds, or (nan, nan) where the transform is undefined (n < 4,
        |r| >= 1, or non-finite r).
    """
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)))