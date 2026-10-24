"""
session_config.py — Central configuration for WMS calcium imaging analysis (vCA2 social identity coding project)

Contains all session metadata, animal paths, decoding type definitions,
and named session subsets. Imported by vca2_utils.py and all analysis notebooks.

Usage:
    from session_config import *
    # or
    from session_config import SESSION_CONFIG, ANIMAL_PATHS, SESSION_SUBSETS

Last updated: WMS 4/6/26
"""

import os
import pandas as pd

# =============================================================================
# Session Definitions
# =============================================================================
# Each session has a consistent index (0-5) used across all notebooks.

SESSION_CONFIG = {
    'labels': [
        'empty_chamb',
        'ss_novel',
        'ss_familiar',
        'os_novel',
        'os_familiar_presep',
        'os_familiar_postsep',
    ],
    'short_labels': ['EC', 'SS-N', 'SS-F', 'OS-N', 'OS-F-Pre', 'OS-F-Post'],
    'days': [0, 1, 2, 8, 9, 23],
    'colors': ['#545454', '#73CFED', '#048BA8', '#F05D7F', '#D31F30', '#8B0000'],
    'has_social_stimuli': [False, True, True, True, True, True],
}

NUM_SESSIONS = len(SESSION_CONFIG['labels'])

# Quick-reference DataFrame (session_idx as index)
SESSION_REF = pd.DataFrame(SESSION_CONFIG)
SESSION_REF.index.name = 'session_idx'


# =============================================================================
# Named Session Subsets
# =============================================================================
# Predefined index lists for common comparisons.

SESSION_SUBSETS = {
    'all':              [0, 1, 2, 3, 4, 5],
    'social':           [1, 2, 3, 4, 5],
    'social_presep':    [1, 2, 3, 4],
    'same_sex':         [1, 2],
    'opposite_sex':     [3, 4, 5],
    'os_presep':        [3, 4],
    'novel':            [1, 3],
    'familiar':         [2, 4, 5],
    'familiar_presep':  [2, 4],
    'os_familiar':      [4, 5],
}


def get_session_info(subset_name):
    """Return the SESSION_REF rows for a named subset."""
    indices = SESSION_SUBSETS[subset_name]
    return SESSION_REF.iloc[indices].copy()


# =============================================================================
# Animal Session Paths
# =============================================================================
# Keys = animal IDs (str), Values = list of 6 bento-folder paths (None if missing).
# All downstream results files are located relative to these bento paths.

ANIMAL_PATHS = {
    '8946': [
        None,  # empty_chamb (session 0)
        r"C:\miniscope_files\analyzed\8946\2025_10_29\12_14_20\bento",   # ss_novel
        r"C:\miniscope_files\analyzed\8946\2025_10_30\12_09_10\bento",   # ss_familiar
        r"C:\miniscope_files\analyzed\8946\2025_11_05\11_27_56\bento",   # os_novel
        r"C:\miniscope_files\analyzed\8946\2025_11_06\11_26_57\bento",   # os_familiar_presep
        r"C:\miniscope_files\analyzed\8946\2025_11_20\11_51_01\bento",   # os_familiar_postsep
    ],
    '8929': [
        None,
        r"C:\miniscope_files\analyzed\8929\2025_10_29\15_38_35\bento",
        r"C:\miniscope_files\analyzed\8929\2025_10_30\15_43_40\bento",
        r"C:\miniscope_files\analyzed\8929\2025_11_05\15_06_48\bento",
        r"C:\miniscope_files\analyzed\8929\2025_11_06\15_03_15\bento",
        r"C:\miniscope_files\analyzed\8929\2025_11_20\15_14_19\bento",
    ],
    '8689': [
        None,
        r"C:\miniscope_files\analyzed\8689\2025_08_12\12_11_23\bento",
        r"C:\miniscope_files\analyzed\8689\2025_08_13\11_44_57\bento",
        r"C:\miniscope_files\analyzed\8689\2025_08_19\12_22_31\bento",
        r"C:\miniscope_files\analyzed\8689\2025_08_20\12_04_29\bento",
        r"C:\miniscope_files\analyzed\8689\2025_09_03\12_16_29\bento",
    ],
    '8020': [
        None,
        r"C:\miniscope_files\analyzed\8020\2025_04_03\15_48_37\bento",
        r"C:\miniscope_files\analyzed\8020\2025_04_04\15_52_44\bento",
        r"C:\miniscope_files\analyzed\8020\2025_04_10\16_23_52\bento",
        r"C:\miniscope_files\analyzed\8020\2025_04_11\15_59_33\bento",
        r"C:\miniscope_files\analyzed\8020\2025_04_25\16_12_33\bento",
    ],
    '7981': [
        None,
        r"C:\miniscope_files\analyzed\7981\2025_03_24\15_32_02\bento",
        r"C:\miniscope_files\analyzed\7981\2025_03_25\bento",
        r"C:\miniscope_files\analyzed\7981\2025_03_31\15_36_09\bento",
        r"C:\miniscope_files\analyzed\7981\2025_04_01\15_50_43\bento",
        r"C:\miniscope_files\analyzed\7981\2025_04_15\15_48_29\bento",
    ],
    '7845': [
        None,
        r"C:\miniscope_files\analyzed\7845\2025_01_14\13_45_23\bento",
        r"C:\miniscope_files\analyzed\7845\2025_01_15\13_00_07\bento",
        r"C:\miniscope_files\analyzed\7845\2025_01_21\14_11_38\bento",
        r"C:\miniscope_files\analyzed\7845\2025_01_22\13_45_49\bento",
        r"C:\miniscope_files\analyzed\7845\2025_02_05\14_00_22\bento",
    ],
    '6175': [
        None,
        r"C:\miniscope_files\analyzed\6175\2024_02_07\13_21_54\bento",
        r"C:\miniscope_files\analyzed\6175\2024_02_08\13_24_50\bento",
        r"C:\miniscope_files\analyzed\6175\2024_02_14\13_24_01\bento",
        r"C:\miniscope_files\analyzed\6175\2024_02_15\13_32_05\bento",
        r"C:\miniscope_files\analyzed\6175\2024_02_29\13_37_01\bento",
    ],
}


# =============================================================================
# Default Result-File Paths (relative to bento folder)
# =============================================================================

BEHAVIOR_RESULTS_REL = 'analysis_results/behavioral_results/behavior_results.pkl'
SINGLE_CELL_RESULTS_REL = 'analysis_results/single_cell_results/single_cell_res.pkl' #single_cell_tuning dict
POPULATION_RESULTS_DIR_REL = 'analysis_results/population_results'

# Cross-session / cross-animal results directory
CROSS_SESSION_RESULTS_DIR = r"C:\miniscope_files\analyzed\xsessionxanimal_results"


##session map for partner preference score comparisons:
##ppt scores for each imaging animal should live at r"\\penc8.rc.int.colorado.edu\DonaldsonLab\Sheeran\experiments\hippocampus miniscope imaging\imaging animals ppt results\imaging_animal_ppt_scores.xlsx"
PPT_SESSION_MAP = {
    'SS': 2,           # PPT follows ss_familiar
    'OS_presep': 4,    # PPT follows os_familiar_presep
    'OS_postsep': 5,   # PPT follows os_familiar_postsep
}


## base colors
PARTNER_COLOR = '#4B2E8D'
NOVEL_COLOR = '#07AC4B'
LEFT_COLOR = '#5c433d'
RIGHT_COLOR = '#6cc0ec'

#mixed cells -- these are all 40% lighter than the base colors
PARTNER_MIXED_COLOR = '#9382bb'
NOVEL_MIXED_COLOR = '#6acd93'
LEFT_MIXED_COLOR = '#9d8e8b'
RIGHT_MIXED_COLOR = '#a7d9f4'
