"""
utils.py

various general purpose functions for 1-photon calcium imaging / behavioral data analysis

written by Will Sheeran (University of Colorado, Boulder) with assistance from Claude Sonnet 4.5, Opus 4.5, and Opus 4.6. All Claude-generated code vetted by WMS.

last edit 3/3/2026

"""

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

def plot_trace(fluoro_traces,cell_id,downsample=1,data_type='Cdata',figsize=(12,3),save_path=None,show=True):
    """
    Plots continuous fluorescence activity (y) for a single cell over recorded frames (x).

    parameters:
    ---
    fluoro_traces: pd.DataFrame
        df with rows=cells and columns=frames (or other time series); values are fluorescence data to be plotted (eg C data or z-scored dF/F)
    cell_id: int
        index of df to plot (=cell to plot)
    downsample: int, optional
        plot every nth frame (default=1 --> no downsampling)
        higher values (e.g., 5) --> faster plotting
    data_type: str, optional
        what kind of trace is being plotted? Options are:
            'Cdata' ('raw' or 'denoised' fluorescence values from CNMF-E in arbitrary units)
            'c_z' (z-scored c data)
            'dff' (dF/F; change in fluorescence from baseline normalized to baseline)
            'dff_z' (z-scored dF/F)
    figsize: tuple, optional
        (width, height)
    save_path: str, optional
        path to save figure. If =None, will not save
    show: bool, optional
        display the plot or not

    returns:
    ---
    fig, ax: matplotlib figure and axis objects
    """
    if data_type not in ['Cdata', 'c_z', 'dff', 'dff_z']:
        raise ValueError(f"data_type must be 'Cdata', 'c_z', 'dff', or 'dff_z', got '{data_type}'")

    #extract trace
    trace=fluoro_traces.iloc[cell_id].values[::downsample]
    frames=np.arange(0,len(fluoro_traces.columns),downsample)

    #create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frames, trace, linewidth=0.5, rasterized=True) #rasterizing speeds up rendering

    ax.set_xlabel('frames', fontsize=11)
    if data_type=='Cdata':
        ax.set_ylabel('fluorescence (a.u.)', fontsize=11)
    elif data_type=='c_z':
        ax.set_ylabel('norm. fluorescence (σ)',fontsize=11)
    elif data_type=='dff':
        ax.set_ylabel('ΔF/F',fontsize=11)
    elif data_type=='dff_z':
        ax.set_ylabel('ΔF/F', fontsize=11)

    ax.set_title(f'Cell {cell_id} - Calcium Trace', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def get_transient_timestamps_mod(neural_data, thresh_type="zscore", std_thresh=5, localMaxNumPoints=10):
    """
    written by Mostafa El-Kalliny and Kathleen Murphy (CU Boulder) and SMTB undergrads
    lightly edited here

    Converts an array of continuous time series (e.g., traces or S)
    into lists of timestamps where activity exceeds some threshold.

    E.g., to convert fluorescence data into discritized 'event' data
​
    :parameters
    ---
    neural_data: (neuron, time) array
        Neural time series, (e.g., C or S).
​
    thresh_type: str
        Type of thresholding ("zscore" or "zscore_localMax" or "scipy_peak").
​
    std_thresh: float
        Number of standard deviations above the mean to define threshold.
​
    :returns
    ---
    event_times: list of length neuron
        Each entry in the list contains the timestamps of a neuron's
        activity.
​
    event_mags: list of length neuron
        Event magnitudes.
​
    bool_arr: ndarray of bool
        Boolean array indicating whether a value exceeds the threshold.
    """
    # Compute thresholds (z-scores) for each neuron.
    neural_array = np.asarray(neural_data, dtype=np.float32)
    stds = np.std(neural_array, axis=1)
    means = np.mean(neural_array, axis=1)
    thresh = means + std_thresh * stds
    
    # Initialize event times, magnitudes, and boolean array.
    event_times = []
    event_mags = []
    bool_arr = np.zeros_like(neural_array, dtype=bool)
    
    for index, (neuron, t) in enumerate(zip(neural_array, thresh)):
        event_indices = []
        
        if thresh_type == "zscore":
            event_indices = np.where(neuron > t)[0]
        
        elif thresh_type == "zscore_localMax":
            for i in range(1, len(neuron) - 1):
                if (
                    neuron[i] > t
                    and all(0 <= j < len(neuron) and neuron[i] > neuron[j] for j in range(i - localMaxNumPoints, i))
                    and all(0 <= j < len(neuron) and neuron[i] > neuron[j] for j in range(i + 1, i + localMaxNumPoints + 1))
                ):
                    event_indices.append(i)
                    
        elif thresh_type == "scipy_peak":
            local_max = scipy.signal.argrelextrema(neuron, np.greater, order=localMaxNumPoints)[0]
            event_indices = local_max[neuron[local_max] > t]          

        event_times.append(np.array(event_indices))
        event_mags.append(neuron[event_indices])
        bool_arr[index, event_indices] = True

    return event_times, event_mags, bool_arr 


def plot_events(neural_data,neuron,event_times,event_mags,downsample=1,trace_type=None,figsize=(12,4),save_path=None,show=True):
    """
    plotting code to label defined events on top of a calcium trace

    parameters:
    ----
    neural_data: pd.DataFrame
        df with rows=cells and columns=frames; fluorescence data (e.g., Cdata, dF/F)
    neuron: int
        index of cell to plot
    event_times: list
        output from get_transient_timestamps_mod() - list of event indices per neuron
    event_mags: list
        output from get_transient_timestamps_mod() - list of event magnitudes per neuron
    downsample: int, optional
        plot every nth frame (default=1 --> no downsampling)
    trace_type: str, optional
        'Cdata', 'c_z', 'dff', 'dff_z', or None (if you don't want to specify)
    figsize : tuple, optional
        (width, height)
    save_path: str, optional
        path to save figure. If None, will not save
    show: bool, optional
        display the plot or not
        
    returns:
    ----
    fig, ax: matplotlib figure and axis objects
    
    """
    trace=neural_data.iloc[neuron].values #eg Cdata df
    event_inds = event_times[neuron] #output of get_transient_timestamps_mod
    event_vals = event_mags[neuron] #output of get_transient_timestamps_mod

    trace_plot = trace[::downsample]
    frames = np.arange(0, len(trace), downsample)

    if downsample > 1:
        event_mask = event_inds % downsample == 0
        event_inds_plot = event_inds[event_mask]
        event_vals_plot = event_vals[event_mask]
    else:
        event_inds_plot = event_inds
        event_vals_plot = event_vals

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frames, trace_plot, 'k-', linewidth=1, alpha=0.8, 
            rasterized=True, label='calcium trace')
    ax.scatter(event_inds_plot, event_vals_plot, c='red', s=30, 
               alpha=0.7, label='detected events')
    
    ax.set_xlabel('frames', fontsize=11)
    
    # Set y-label based on data type
    if trace_type is not None:
        if trace_type == 'Cdata':
            ax.set_ylabel('Fluorescence (a.u.)', fontsize=11)
        elif trace_type=='c_z':
            ax.set_ylabel('norm. fluorescence (σ)',fontsize=11)
        elif trace_type == 'dff':
            ax.set_ylabel('ΔF/F', fontsize=11)
        elif trace_type == 'dff_z':
            ax.set_ylabel('ΔF/F', fontsize=11)
        else:
            ax.set_ylabel('units', fontsize=11)
    else:
            ax.set_ylabel('units', fontsize=11)
    
    ax.set_title(f'Cell {neuron} - Events on Trace', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def calculate_dff_percentile(Cdata, percentile=10):
    """
    Calculate dF/F using sesion-wide non-zero percentile as F0
    """
    def nonzero_percentile(arr):
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            return np.nan
        return np.percentile(nonzero, percentile)
    
    baseline = Cdata.apply(nonzero_percentile, axis=1)
    baseline_expanded = pd.DataFrame(
    np.tile(baseline.values[:, np.newaxis], Cdata.shape[1]),
        index=Cdata.index,
        columns=Cdata.columns
    )
    
    dff = (Cdata - baseline_expanded) / baseline_expanded
    
    return dff, baseline_expanded

def calculate_dff_median(Cdata):
    """
    Calculate dF/F using sesion-wide non-zero median as F0
    """
    def nonzero_median(arr):
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            return np.nan
        return np.median(nonzero)
    
    baseline = Cdata.apply(nonzero_median, axis=1)
    baseline_expanded = pd.DataFrame(
    np.tile(baseline.values[:, np.newaxis], Cdata.shape[1]),
        index=Cdata.index,
        columns=Cdata.columns
    )
    
    dff = (Cdata - baseline_expanded) / baseline_expanded
    
    return dff, baseline_expanded


def traces_plot(fluoro_traces,n_traces,x_range=None,data_type='Cdata',figsize=(12,3),color=True,offset_scale='auto', 
                fps=None,scalebar_time_sec=None,scalebar_ylen=None,
                save_path=None,show=True):
    """
    Plots continuous fluorescence activity (y) for multiple cells over a specified range of frames (x).

    parameters:
    ---
    fluoro_traces: pd.DataFrame
        df with rows=cells and columns=frames (or other time series); values are fluorescence data to be plotted (eg C data or z-scored dF/F)
    n_traces: int or array-like
        if int: how many (randomly selected) rows of fluoro_traces to plot
        if array-like: which specific rows to plot
    x_range: array-like or None
        which frames (or whatever the time axis is in fluoro_traces) to plot over
    data_type: str, optional
        what kind of trace is being plotted? Options are:
            'Cdata' ('raw' or 'denoised' fluorescence values from CNMF-E in arbitrary units)
            'c_z' (z-scored c data)
            'dff' (dF/F; change in fluorescence from baseline normalized to baseline)
            'dff_z' (z-scored dF/F)
    figsize: tuple, optional
        (width, height)
    color: bool
        Whether to distinctly color each trace (True), otherwise leave all traces black (False)
    offset_scale: float or 'auto', optional
        vertical spacing between traces. If 'auto', calculated from data
    fps: float, optional
        frame rate (fps). Required for time-based scale bar
    scalebar_time_sec: float, optional
        scale bar duration in seconds (e.g., 10 for "10 s"). Requires fps
    scalebar_ylen: float, optional
        scale bar height in either standard deviations (if data_type= 'd_z' or 'dff_z') or dF/F units (if data_type='dff')
    save_path: str, optional
        path to save figure. If =None, will not save
    show: bool, optional
        display the plot or not

    returns:
    ---
    fig, ax: matplotlib figure and axis objects
    """
    if data_type not in ['Cdata', 'c_z', 'dff', 'dff_z']:
        raise ValueError(f"data_type must be 'Cdata', 'c_z', 'dff', or 'dff_z', got '{data_type}'")
    
    num_rows = len(fluoro_traces)
    if isinstance(n_traces, int):
        rows_to_plot = np.random.choice(num_rows, size=n_traces, replace=False)
    elif isinstance(n_traces, (list, np.ndarray)):
        rows_to_plot = np.array(n_traces)
    else:
        raise ValueError("n_traces must be int, list, or numpy array")
    
    #get the columns (frames) to plot
    if x_range is not None:
        plot_df = fluoro_traces.iloc[rows_to_plot, x_range].copy()
    else:
        plot_df = fluoro_traces.iloc[rows_to_plot, :].copy()
    
    num_plot_rows = len(rows_to_plot)
    if num_plot_rows == 0:
        raise ValueError("n_traces resulted in 0 traces to plot")
    
    #calculate offset between traces
    if offset_scale == 'auto':
        offset = 2 * np.max([np.ptp(plot_df.iloc[i].values) for i in range(num_plot_rows)])
        offset = max(offset, 0.1)  # Minimum offset to prevent tiny spacing
    else:
        offset = offset_scale
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if color:
        colors = plt.cm.Set1(np.linspace(0, 1, num_plot_rows))
    else:
        colors = ['black'] * num_plot_rows
    
    #plot traces with vertical offset
    t = np.arange(len(plot_df.columns))
    for i, row_idx in enumerate(rows_to_plot):
        trace = plot_df.iloc[i].values
        y_offset = i * offset
        ax.plot(t, trace + y_offset, color=colors[i], linewidth=0.75, gid=f'trace_{row_idx}', label=f'neuron_{row_idx}')
    
    #get rid of each subplot's grid / axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add y-axis label on the left
    if data_type == 'Cdata':
        ylabel_text = 'Fluorescence (a.u.)'
    elif data_type=='c_z':
        ylabel_text = 'norm. fluorescence'
    elif data_type == 'dff':
        ylabel_text = 'ΔF/F'
    elif data_type == 'dff_z':
        ylabel_text = 'ΔF/F'
    
    fig.text(0.02, 0.5, ylabel_text, va='center', rotation='vertical', fontsize=12)
    
    # Add scale bar
    scalebar_x = -0.02 * len(t)  # 5% from left edge
    scalebar_y = -0.2 * offset   # Below bottom trace
    
    # Horizontal bar (time)
    if scalebar_time_sec is not None and fps is not None:
        scalebar_frames = scalebar_time_sec * fps
        ax.plot([scalebar_x, scalebar_x + scalebar_frames], 
                [scalebar_y, scalebar_y], 'k-', linewidth=2)
        ax.text(scalebar_x + scalebar_frames/2, scalebar_y - 0.3*offset, 
                f'{scalebar_time_sec} s', ha='center', va='top', fontsize=10)
    
    # Vertical bar (amplitude)
    if (data_type == 'dff_z' or data_type == 'c_z') and scalebar_ylen is not None:
        scalebar_height = scalebar_ylen
        ax.plot([scalebar_x, scalebar_x], 
                [scalebar_y, scalebar_y + scalebar_height], 'k-', linewidth=2)
        
        label = f'{scalebar_ylen} σ'
        
        ax.text(scalebar_x - 0.01*len(t), scalebar_y + scalebar_height/1.5, 
                label, ha='right', va='center', fontsize=10)

    elif data_type == 'dff' and scalebar_ylen is not None:
        scalebar_height = scalebar_ylen
        ax.plot([scalebar_x, scalebar_x], 
                [scalebar_y, scalebar_y + scalebar_height], 'k-', linewidth=2)
        
        label = f'{int(scalebar_ylen * 100)}% ΔF/F'

        ax.text(scalebar_x - 0.01*len(t), scalebar_y + scalebar_height/1.5, 
                label, ha='right', va='center', fontsize=10)
    
    # Set appropriate limits to include scale bar
    ax.set_ylim(scalebar_y - offset, (num_plot_rows) * offset)
    ax.set_xlim(-0.05 * len(t), len(t) * 1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig, ax

def add_social_interaction_behaviors(raster_df, partner_side_social1):
    """
    Add social interaction behavior rows to raster_df based on partner/novel locations.
    
    In most of WMS's recordings in the 3-chamber PPT apparatus, the partner and novel animals start on opposite sides during
    the social1 trial, then swap locations for the social2 trial. This function automatically adds rows to the behavioral
    annotation df that track different kinds of interactions with each stimulus animal.
    
    parameters:
    ----
    raster_df : pd.DataFrame
        Behavioral raster with rows=behaviors, columns=frames, values=0/1
        Must contain rows: 'left', 'right', 'left_cup', 'right_cup', 
                          'social1', 'social2', 'active'
    partner_side_social1 : str
        Which side partner is in for the social1 trial. Either 'left' or 'right'
    
    returns:
    ----
    raster_df : pd.DataFrame
        Updated raster with 6 new behavior rows added:
        - 'partner_chamb': test animal in partner's chamber
        - 'novel_chamb': test animal in novel's chamber
        - 'partner_cup': test animal at partner's cup
        - 'novel_cup': test animal at novel's cup
        - 'partner_active': test animal actively interacting with partner
        - 'novel_active': test animal actively interacting with novel
    """
    #input validation
    if partner_side_social1 not in ['left', 'right']:
        raise ValueError("partner_side_social1 must be 'left' or 'right'")
    
    required_rows = ['left', 'right', 'left_cup', 'right_cup', 
                     'social1', 'social2', 'active']
    missing_rows = [r for r in required_rows if r not in raster_df.index]
    if missing_rows:
        raise ValueError(f"raster_df missing required rows: {missing_rows}")
    
    #determine sides for each trial
    if partner_side_social1 == 'left':
        partner_social1_chamb = 'left'
        partner_social1_cup = 'left_cup'
        partner_social2_chamb = 'right'
        partner_social2_cup = 'right_cup'
        novel_social1_chamb = 'right'
        novel_social1_cup = 'right_cup'
        novel_social2_chamb = 'left'
        novel_social2_cup = 'left_cup'
    else:  #partner starts on right
        partner_social1_chamb = 'right'
        partner_social1_cup = 'right_cup'
        partner_social2_chamb = 'left'
        partner_social2_cup = 'left_cup'
        novel_social1_chamb = 'left'
        novel_social1_cup = 'left_cup'
        novel_social2_chamb = 'right'
        novel_social2_cup = 'right_cup'
    
    #extract relevant rows
    social1 = raster_df.loc['social1']
    social2 = raster_df.loc['social2']
    active = raster_df.loc['active']
    
    #initialize new behavior rows
    n_frames = raster_df.shape[1]
    partner_chamb = pd.Series(0, index=raster_df.columns, name='partner_chamb')
    novel_chamb = pd.Series(0, index=raster_df.columns, name='novel_chamb')
    partner_cup = pd.Series(0, index=raster_df.columns, name='partner_cup')
    novel_cup = pd.Series(0, index=raster_df.columns, name='novel_cup')
    partner_active = pd.Series(0, index=raster_df.columns, name='partner_active')
    novel_active = pd.Series(0, index=raster_df.columns, name='novel_active')
    
    #partner chamber (social1 + social2)
    partner_chamb_mask = (
        ((raster_df.loc[partner_social1_chamb] == 1) & (social1 == 1)) |
        ((raster_df.loc[partner_social2_chamb] == 1) & (social2 == 1))
    )
    partner_chamb[partner_chamb_mask] = 1
    
    #novel chamber (social1 + social2)
    novel_chamb_mask = (
        ((raster_df.loc[novel_social1_chamb] == 1) & (social1 == 1)) |
        ((raster_df.loc[novel_social2_chamb] == 1) & (social2 == 1))
    )
    novel_chamb[novel_chamb_mask] = 1
    
    #partner cup (social1 + social2)
    partner_cup_mask = (
        ((raster_df.loc[partner_social1_cup] == 1) & (social1 == 1)) |
        ((raster_df.loc[partner_social2_cup] == 1) & (social2 == 1))
    )
    partner_cup[partner_cup_mask] = 1
    
    #novel cup (social1 + social2)
    novel_cup_mask = (
        ((raster_df.loc[novel_social1_cup] == 1) & (social1 == 1)) |
        ((raster_df.loc[novel_social2_cup] == 1) & (social2 == 1))
    )
    novel_cup[novel_cup_mask] = 1
    
    #partner active (active + partner chamber)
    partner_active[partner_chamb_mask & (active == 1)] = 1
    
    #novel active (active + novel chamber)
    novel_active[novel_chamb_mask & (active == 1)] = 1
    
    # Add new rows to raster_df
    raster_df.loc['partner_chamb'] = partner_chamb
    raster_df.loc['novel_chamb'] = novel_chamb
    raster_df.loc['partner_cup'] = partner_cup
    raster_df.loc['novel_cup'] = novel_cup
    raster_df.loc['partner_active'] = partner_active
    raster_df.loc['novel_active'] = novel_active
    
    print(f"Added 6 social interaction behaviors (partner started on {partner_side_social1} in social1)")
    
    return raster_df

def align_feeds_flexible(reference_ts, **other_feeds):
    '''
    Takes timestamp .csv files and aligns all feeds to the reference timestamp
    
    Args:
        reference_ts: path to reference timestamp CSV (usually miniscope)
        **other_feeds: keyword arguments where key=feed_name, value=path to CSV
                      e.g., cam1='path/to/cam1.csv', cam2='path/to/cam2.csv'
    
    Returns:
        align_df: dataframe with reference timestamps and corresponding aligned frames
    '''
    # Read reference timestamps
    reference_df = pd.read_csv(reference_ts)
    
    # Start with reference dataframe
    align_df = reference_df.copy()
    
    #handle edge cases where some time stamps get out of order because the camera's buffer ran full
    if not align_df["Time Stamp (ms)"].is_monotonic_increasing:
        diffs_adf = align_df['Time Stamp (ms)'].diff()
        problematic_rows_adf = align_df[diffs_adf < 0].index-1 #you need to isolate the frame that randomly skipped ahead, which will be the previous row from the row with the negative diff
        #granted, this assumes that when frames get out of order they always skip ahead, which is the behavior I've seen. If a different behavior happens, this shoudl get caught by the assert statement below
        problematic_frame_numbers_adf=align_df.loc[problematic_rows_adf,"Frame Number"].values
        print(f"Scope: Dropping {len(problematic_rows_adf)} out-of-order timestamp(s) -- Frame numbers : {problematic_frame_numbers_adf}")
        align_df = align_df.drop(problematic_rows_adf).reset_index(drop=True)
        assert align_df["Time Stamp (ms)"].is_monotonic_increasing, "Scope timestamps still not monotonic after dropping out-of-order frames"
    else:
        print(f"No timestamp(s) needed to drop from scope -- no out of order frames :)")

    # Align each additional feed to the reference
    for feed_name, feed_path in other_feeds.items():
        feed_df = pd.read_csv(feed_path)

        #handle edge cases where some time stamps get out of order because the camera's buffer ran full
        if not feed_df["Time Stamp (ms)"].is_monotonic_increasing:
            diffs = feed_df['Time Stamp (ms)'].diff()
            problematic_rows = feed_df[diffs < 0].index-1 #you need to isolate the frame that randomly skipped ahead, which will be the previous row from the row with the negative diff
            problematic_frame_numbers=feed_df.loc[problematic_rows,"Frame Number"].values
            print(f"{feed_name}: Dropping {len(problematic_rows)} out-of-order timestamp(s) -- Frame numbers : {problematic_frame_numbers}")
            feed_df = feed_df.drop(problematic_rows).reset_index(drop=True)
            assert feed_df["Time Stamp (ms)"].is_monotonic_increasing, "Cam timestamps still not monotonic after dropping out-of-order frames"
        else:
            print(f"No timestamp(s) needed to drop from {feed_name} -- no out of order frames :)")


        align_df = pd.merge_asof(align_df, feed_df, 
                                on="Time Stamp (ms)", 
                                direction="nearest", 
                                suffixes=['', f'_{feed_name}'])
    
    # Remove Buffer Index columns
    align_df = align_df[align_df.columns.drop(list(align_df.filter(regex='Buffer Index')))]
    
    return align_df

def approx_fr(timestamps_df,ts_col='Time Stamp (ms)',input_unit='ms'):
    '''
    function to define the approximate frame rate of a given feed
    parameters:
    ----
    timestamps_df: pd.DataFrame
        a df where each row = one frame and at least one column containing timestamps.
        This timestamps column will typically be called "Time Stamp (ms)" as generated by the UCLA Miniscope DAQ software
    ts_col: str
        name of the column to use as the timestamps column
    input_unit: str
        The unit of time for the timestamps column. Can be 'ms' (milliseconds) or 'sec' (seconds)
    returns:
    ----
    fr: float
        the approximated frame rate in frames/sec
    '''
    cols = timestamps_df.columns
    
    if ts_col not in cols:
        raise ValueError(f"DataFrame missing column '{ts_col}'!")

    if input_unit not in ['ms', 'sec']:
        raise ValueError(f"input_unit must be 'ms' or 'sec', got '{input_unit}'")
    
    time_diffs = timestamps_df[ts_col].diff().dropna()
    
    if len(time_diffs) == 0:
        raise ValueError("Need at least 2 frames to calculate frame rate")
    
    median_interval = time_diffs.median()
    
    if input_unit == 'ms':
        fr = 1000 / median_interval  #ms per frame --> frames per second
    elif  input_unit == 'sec':
        fr = 1 / median_interval

    return fr

def make_behavior_bouts(arr,fr):
    '''
    function to take an array, detect when the array does not monotonically increase by one, split into 'bouts' accordingly.
    Used for taking arrays of frame numbers corresponding to different behavioral bouts to a dataframe where each row=a bout
    and columns=information about each bout
    
    param:
    ----
    arr: array of all recorded frame numbers associated with a given behavior
    fr: recorded frame rate in frames/sec
    returns:
    ----
    bout_df: pd.Dataframe with columns = start (start of bout), end (end of bout), and length (length of bout in seconds)
    '''
    if len(arr) == 0:
        return pd.DataFrame(columns=['start', 'end', 'length'])
    
    # Handle single frame
    if len(arr) == 1:
        return pd.DataFrame([[arr[0], arr[0], 1/fr]], columns=['start', 'end', 'length'])

    start=arr[0]
    series=[]

    for i in range(1,len(arr)):
        if arr[i] != arr[i-1] + 1:
            end = arr[i-1]
            series.append([start,end,end-start+1])
            start=arr[i]

    end=arr[-1]
    series.append([start, end, end - start + 1])

    bout_df = pd.DataFrame(series,columns=['start','end','length'])
    #convert length from frames to seconds
    bout_df['length']=bout_df['length']/fr
    
    return bout_df

def get_bout_subtypes(bouts1,bouts2,label,length_min=1):
    '''
    function to take two interaction bouts dfs (produced by make_behavior_bouts) and produce a new df containing only bouts shared across the two input dfs to produce subtypes/classes of bouts
    can also filter for a minimum bout length time and add a label for this subtype
    WMS use-case: find which partner or novel bouts occurred at the left or right cup

    parameters:
    ----
    bouts1 = pd.DataFrame
        product of make_behavior_bouts() - a df indicating the start frame, stop frame, and duration (columns) of each behavior bout (rows) for a specific type of behavior (e.g., partner interactions)
    bouts2 = pd.DataFrame
        same as bouts1 but a different behavior (e.g., left cup interactions)
    length_min = numeric
        a minimum length for a bout to be included in the new df
    label = string
        a new label to be appended to each bout in the new df
    
    returns:
    ----
    subtype_bouts = pd.DataFrame
        the new df with the bouts that are shared among the two parent types (e.g., partner/left-cup bouts)
    '''
    matches = []

    for idx1, row1 in bouts1.iterrows():
        for idx2, row2 in bouts2.iterrows():
            if (row1['start'] == row2['start'] and 
                row1['end'] == row2['end'] and
                row1['length']>=length_min):

                matches.append({
                    'start': int(row1['start']),
                    'end': int(row1['end']),
                    'length': row1['length']
                })

    subtype_bouts = pd.DataFrame(matches)
    subtype_bouts['type']=label

    return subtype_bouts

def check_inclusion_threshold(*args,min_int_number=2):
    '''
    function to determine whether behavioral data for a given session passes an inclusion threshold (e.g., are there at least two 1 second-long interactions for each type)
    *args are behavior bout dfs (three columns)

    '''
    int_dfs = [*args]
    for int_df in int_dfs:
        num_ints = int_df.shape[0]
        int_type = int_df['type'][0]
        if num_ints < min_int_number:
            print(f'WARNING: {int_type} has fewer than {min_int_number} defined interactions')
        else:
            print(f'{int_type} has at least {min_int_number} defined interactions')

def make_pseudotrials(ints_df,fr,entire=True,length=1):
    '''
    function to create bins, or "pseudotrials", of a defined, consistent length from many variably lengthed behavioral bouts
    The term 'pseudotrial' is taken from Boyle et al., Neuron 2024

    parameters:
    ----
    ints_df: pd.DataFrame
        df with rows = bouts, and columns containing at least start frame, stop frame, and duration
    fr: float
        framerate
    entire: bool
        whether to create pseudovectors tiling the entirety of each bout (True) or whether to just create a single pseudotrial from the first portion of the bout (False)
        default = True
    length: float or int
        how long (in seconds) each pseudotrial should be
        default = 1
    '''
    new_rows = []

    frame_length = round(fr*length)

    for idx, row in ints_df.iterrows():
        start_frame = row['start']
        end_frame = row['end']

        if entire == True: #iterate over the range from start to end in steps specified by the length argument
            for frame in range(start_frame, end_frame, frame_length):
                new_row={
                    'start': frame, #start frame for new row
                    'end': min(frame+(frame_length-1), end_frame), #end frame for the new row, capped at the original end_frame
                    'int': idx
                }
                #make sure to skip PVs that aren't the specified length (i.e., skip the final PV created for a given interaction that only covers the remainder)
                if new_row['end']-new_row['start'] == (frame_length-1):
                    new_rows.append(new_row)
        else:
            new_row = {
                'start': start_frame,
                'end': min(start_frame+(frame_length-1),end_frame),
                'int': idx
            }
            if new_row['end']-new_row['start'] == (frame_length-1):
                new_rows.append(new_row)
        
    new_df = pd.DataFrame(new_rows)
    if 'type' in ints_df.columns:
        new_df['type'] = ints_df['type'].iloc[0]
    #new_df['type'] = ints_df['type'][0] #add the int type as inherited from the input df

    return new_df

def validate_pseudotrials(pts_df, neural_data):
    '''Confirm all pseudotrial frames exist in neural data. Safety measure to check that pseudotrials don't start or end on dropped scope frames (would induce issues in slicing out the respective calcium imaging data)'''
    valid_frames = set(neural_data.columns)
    col_type = type(neural_data.columns[0])
    
    invalid = []
    for idx, row in pts_df.iterrows():
        start = col_type(row['start'])
        end = col_type(row['end'])
        if start not in valid_frames or end not in valid_frames:
            invalid.append(idx)
    
    if invalid:
        print(f"Warning: {len(invalid)} pseudotrials reference missing frames")
    else:
        print("All pseudotrials validated")
    
    return len(invalid) == 0


def check_empty_pseudotrials(pts_df, sData, buffer_frames=0, verbose=True):
    '''
    Check how many pseudotrials contain no neural events.
    
    Parameters:
    -----------
    pts_df : pd.DataFrame
        Pseudotrials dataframe with 'start', 'end' columns
    sData : pd.DataFrame
        Binarized event data with rows = neurons, columns = frames
    buffer_frames : int
        Number of frames to expand window on each side when checking for events.
        Default = 0 (exact pseudotrial boundaries)
    verbose : bool
        Whether to print summary statistics
    
    Returns:
    --------
    event_sums : list
        Total event count for each pseudotrial (with buffer)
    n_empty : int
        Number of pseudotrials with zero events
    empty_indices : list
        Indices of empty pseudotrials
    '''
    col_type = type(sData.columns[0])
    
    # Get valid frame range for bounds checking
    all_frames = sData.columns.astype(int)
    min_frame, max_frame = all_frames.min(), all_frames.max()
    
    event_sums = []
    empty_indices = []
    
    for idx, row in pts_df.iterrows():
        # Apply buffer, respecting data bounds
        pt_start = max(int(row['start']) - buffer_frames, min_frame)
        pt_end = min(int(row['end']) + buffer_frames, max_frame)
        
        pt_start = col_type(pt_start)
        pt_end = col_type(pt_end)
        
        sData_pt = sData.loc[:, pt_start:pt_end]
        pt_sum = sData_pt.values.sum()
        event_sums.append(pt_sum)
        
        if pt_sum == 0:
            empty_indices.append(idx)
    
    n_empty = len(empty_indices)
    n_total = len(pts_df)
    pct_empty = (n_empty / n_total) * 100 if n_total > 0 else 0
    
    if verbose:
        pt_type = pts_df['type'].iloc[0] if 'type' in pts_df.columns else 'unknown'
        print(f"--- {pt_type} pseudotrials (buffer={buffer_frames} frames) ---")
        print(f"Total pseudotrials: {n_total}")
        print(f"Empty (no events):  {n_empty} ({pct_empty:.1f}%)")
        print(f"With events:        {n_total - n_empty} ({100 - pct_empty:.1f}%)")
    
    return event_sums, n_empty, empty_indices

def sample_pseudotrials_balanced(df, n_samples):
    """
    written by Claude Sonnet 4.5, verified by WMS

    Sample pseudotrials in round-robin fashion across interactions.
    Takes one pseudotrial from each interaction before cycling back,
    without replacement.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Pseudotrial dataframe with 'int' column identifying interaction bouts
    n_samples : int
        Total number of pseudotrials to sample
    
    Returns:
    --------
    sampled_df : pd.DataFrame
        Dataframe with sampled pseudotrials
    """
    # Create dictionary: {int_id: [list of available row indices]}
    available_by_int = {}
    for int_id in df['int'].unique():
        # Get all row indices for this interaction
        available_by_int[int_id] = df[df['int'] == int_id].index.tolist()
    
    sampled_indices = []
    
    # Keep cycling through interactions until we have enough samples
    while len(sampled_indices) < n_samples:
        # Get interactions that still have available pseudotrials
        active_ints = [int_id for int_id, indices in available_by_int.items() 
                       if len(indices) > 0]
        
        if len(active_ints) == 0:
            print(f"Warning: Only {len(sampled_indices)} pseudotrials available, requested {n_samples}")
            break
        
        # One round: sample one from each active interaction
        for int_id in sorted(active_ints):  # Sort for consistency
            if len(sampled_indices) >= n_samples:
                break
            
            # Randomly choose one available pseudotrial from this interaction
            chosen_idx = np.random.choice(available_by_int[int_id])
            sampled_indices.append(chosen_idx)
            
            # Remove this pseudotrial from available pool
            available_by_int[int_id].remove(chosen_idx)
    
    # Return sampled dataframe
    return df.loc[sampled_indices].reset_index(drop=True)

def traces_pv_by_bout(bouts_df,neural_data):
    '''
    function to get the continuous neural traces and mean population vector for each behavioral bout
    parameters:
    ----
    bouts_df: pd.DataFrame
        df where rows = individual bouts and columns at least = 'start' and 'end' in frames
    neural_data: pd_DataFrame
        df with rows = neurons, columns = frames, values = neural data

    returns:
    ----
    traces: dict
        the continuous neural activity corresponding to each bout
    mean_pvs: dict
       mean population vectors corresponding to each bout
    '''
    # Ensure column type consistency for .loc slicing
    col_type = type(neural_data.columns[0])

    traces = {}
    mean_pvs = {}
    for idx, row in bouts_df.iterrows():
        bout_start = col_type(row['start'])
        bout_end = col_type(row['end'])

        #using .loc instead of .iloc because we've possibly dropped some frames without re-indexing (intentionally) so we want to call the column names 
        bout_neural = neural_data.loc[:,bout_start:bout_end].copy() #.loc indexing is inclusive so no need to add +1 to bout_end

        if bout_neural.shape[1] == 0:
            print(f"Warning: bout {idx} ({bout_start}-{bout_end}) returned no frames")

        traces[idx]=bout_neural

        #now get the mean population vector for this interval
        pv = bout_neural.mean(axis=1)
        mean_pvs[idx]=pv

    return traces, mean_pvs

#ef nested_pie_chart():

def plot_traces_with_bouts(neural_data, bout_dict, neuron_indices=None, 
                          frame_window=None, bout_color='identity', vertical_spacing=10, 
                          smooth_sigma=5, figsize=(14, 8), data_type='Cdata',
                          save_path=None):
    """
    Plot calcium traces with behavioral bout overlays.
    
    Parameters:
    -----------
    neural_data : pd.DataFrame
        Neural data with rows=neurons, columns=frames (z-scored)
    bout_dict : dict
        Dictionary mapping bout type names to DataFrames with 'start', 'end' columns
        Example: {'partner': partner_df, 'novel': novel_df, 'left': left_df}
    neuron_indices : list or None
        List of neuron indices to plot. If None, plots first 5 neurons
    frame_window : tuple or None
        (start_frame, end_frame) to zoom into specific time window. If None, plots all
    vertical_spacing : float
        Vertical spacing between neuron traces
    smooth_sigma : float
        Gaussian smoothing sigma (set to 0 for no smoothing)
    figsize : tuple
        Figure size (width, height)
    save_path : str or None
        Path to save figure. If None, doesn't save
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Default color scheme for bout types
    identity_colors = {
        'partner': '#4B2E8D',
        'novel': '#07AC4B',
        'partner_left': '#4B2E8D',
        'partner_right': '#4B2E8D',
        'p_l': '#4B2E8D',
        'p_r': '#4B2E8D',
        'novel_left': '#07AC4B',
        'novel_right': '#07AC4B',
        'n_l': '#07AC4B',
        'n_r': '#07AC4B'
    }

    location_colors = {
        'left': '#5c433d',
        'right': '#6cc0ec',
        'partner_left': '#5c433d',
        'partner_right': '#6cc0ec',
        'p_l': '#5c433d',
        'p_r': '#6cc0ec',
        'novel_left': '#5c433d',
        'novel_right': '#6cc0ec',
        'n_l': '#5c433d',
        'n_r': '#6cc0ec'
    }
    
    # Select neurons to plot
    if neuron_indices is None:
        neuron_indices = list(range(min(5, len(neural_data))))
    ncells = len(neuron_indices)
    
    # Get time axis
    t = neural_data.columns.astype(float)
    
    # Apply time window if specified
    if frame_window is not None:
        start_frame, end_frame = frame_window
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        mask = (t >= start_frame) & (t <= end_frame)
        t = t[mask]
        neural_subset = neural_data.loc[:, mask]
    else:
        neural_subset = neural_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bout overlays FIRST (so they're behind traces)
    for bout_type, bout_df in bout_dict.items():
        if bout_color=='identity':
            color = identity_colors.get(bout_type, '#CCCCCC')
        elif bout_color=='location':
            color = location_colors.get(bout_type, '#CCCCCC')
        else:
            color='#CCCCCC'

        for idx, event in bout_df.iterrows():
            x_start = event['start']
            x_end = event['end']
            
            # Only plot if in time window
            if frame_window is None or (x_start <= end_frame and x_end >= start_frame):
                ax.axvspan(x_start, x_end, color=color, alpha=0.3, 
                          label=bout_type if idx == 0 else "")
    
    # Plot neural traces
    for plot_idx, neuron_idx in enumerate(neuron_indices):
        vertical_offset = plot_idx * vertical_spacing
        
        # Get trace for this neuron
        trace = neural_subset.iloc[neuron_idx].values
        
        # Apply smoothing if requested
        if smooth_sigma > 0:
            trace = gaussian_filter1d(trace, sigma=smooth_sigma)
        
        # Plot
        ax.plot(t, trace + vertical_offset, color='black', 
               alpha=0.8, linewidth=1.2, label=f'Neuron {neuron_idx}')
        
        # Add neuron ID label on the left
        ax.text(t[0] - (t[-1] - t[0]) * 0.02, vertical_offset, 
               f'{neuron_idx}', va='center', ha='right', fontsize=9)
    
    # Formatting
    ax.set_ylim(-vertical_spacing * 0.5, ncells * vertical_spacing + vertical_spacing * 0.5)
    ax.set_xlabel('Frames', fontsize=12)

    if data_type == 'Cdata':
        ylabel_text = 'Fluorescence (a.u.)'
    elif data_type=='c_z':
        ylabel_text = 'norm. fluorescence'
    elif data_type == 'dff':
        ylabel_text = 'ΔF/F'
    elif data_type == 'dff_z':
        ylabel_text = 'ΔF/F (z-scored)'

    ax.set_ylabel(ylabel_text, fontsize=12)
    ax.set_title('Neural Activity During Social Interactions', fontsize=14, weight='bold')
    
    # Add legend (only unique bout types)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig, ax

def get_offset_frame(timestamps,start_frame,time_offset,units='sec'):
    '''
    from a timestamps df, get the timestamp that is a specified amount of time away from a specific timestamp

    Params:
    ----
    timestamps: pd>DataFrame
        df with columns 'Frame Number' and 'Time Stamp (ms)' as generated by the UCLA minsicope recording software
    start_frame: int
        frame # to find time difference from
    time_offset: float
        time difference from start_frame to find the output frame
    
    Returns:
    ----
    out_frame: int
        frame # closest to the target time
    end_timestamp: float
        actual timestamp at out_frame
    actual_offset_sec: float
        the time offset calculated, in seconds
    '''
    start_ts = timestamps.loc[timestamps['Frame Number']==start_frame, 'Time Stamp (ms)'].values[0]
    if units=='sec':
        target_ts = start_ts + (time_offset*1000)
    elif units=='ms':
        target_ts = start_ts + time_offset
    else:
        raise ValueError(f"'units' must be 'ms' or 'sec', got '{units}'")

    timestamps['diff'] = np.abs(timestamps['Time Stamp (ms)'] - target_ts)
    closest_idx = timestamps['diff'].idxmin()

    out_frame = timestamps.loc[closest_idx, 'Frame Number']
    end_timestamp = timestamps.loc[closest_idx, 'Time Stamp (ms)']

    actual_offset_sec = (end_timestamp - start_ts) / 1000

    timestamps.drop(columns=['diff'], inplace=True)

    return int(out_frame), end_timestamp, actual_offset_sec

def plot_behavior_raster(raster_df, behaviors=['partner_active', 'novel_active', 
                                                'partner_chamb', 'novel_chamb'],
                          fr=50,colors=None, separator_frame=None,
                          figsize=(16, 6), set_title=True, title='Test Animal Behavior during Imaging Session',
                          save_path=None, show=True):
    """
    Create a raster plot showing behavioral bouts from a behavioral raster dataframe
    
    Parameters:
    -----------
    raster_df : pd.DataFrame
        Behavioral raster with rows=behaviors and columns=frames
    behaviors : list
        List of behavior row names to plot (top to bottom order)
    colors : dict, optional
        Color mapping for each behavior. If None, uses default colors
    separator_frame : int, optional
        Frame number to draw a vertical separator line (e.g., to separate trials)
        If None, no separator is drawn
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Default colors
    if colors is None:
        colors = {
            'partner_active': '#4B2E8D',
            'novel_active': '#07AC4B',
            'partner_chamb': '#BBA3CE',
            'novel_chamb': '#9BD091',
            'left': '#5c433d',
            'right': '#6cc0ec',
            'active': '#E74C3C'
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_behaviors = len(behaviors)
    
    # Plot each behavior
    for i, behavior in enumerate(behaviors):
        if behavior not in raster_df.index:
            print(f"Warning: '{behavior}' not found in raster_df, skipping...")
            continue
        
        # Get frames where behavior = 1
        behavior_arr = np.asarray(raster_df.loc[behavior])
        behavior_frames = np.where(behavior_arr == 1)[0]
        frame_indices = raster_df.columns[behavior_frames].astype(int)
        
        if len(frame_indices) > 0:
            # Make bouts from continuous frame sequences
            bouts = make_behavior_bouts(frame_indices,fr)
            
            # Y position (inverted so first behavior is at top)
            y_pos = n_behaviors - i
            
            # Plot each bout as a horizontal bar
            for _, bout in bouts.iterrows():
                ax.barh(y_pos, width=bout['end'] - bout['start'], 
                       left=bout['start'], height=0.8,
                       color=colors.get(behavior, 'gray'), 
                       edgecolor='none', alpha=0.8)
    
    # Add vertical separator line if specified
    if separator_frame is not None:
        ax.axvline(x=separator_frame, color='black', linestyle='--', 
                  linewidth=2, alpha=0.7, zorder=10)
    
    # Set y-axis labels
    ytick_positions = [n_behaviors - i for i in range(n_behaviors)]
    ytick_labels = [behavior.replace('_', ' ').title() for behavior in behaviors]
    
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    

    # Convert x-axis from frames to seconds
    xticks_frames = ax.get_xticks()
    xticks_seconds = xticks_frames / fr
    ax.set_xticks(xticks_frames, labels=[f'{x:.0f}' for x in xticks_seconds])

    max_frame = raster_df.shape[1] - 1  # Last frame index
    ax.set_xlim(0, max_frame)

    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Behavior', fontsize=12)
    if set_title == True:
        ax.set_title(title, fontsize=14, weight='bold')
    
    ax.set_ylim(0.5, n_behaviors + 0.5)
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=colors.get(beh, 'gray'), 
                                     label=beh.replace('_', ' ').title()) 
                      for beh in behaviors if beh in raster_df.index]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def merge_close_bouts(bouts_df, threshold_sec=1.0, fr=30, dropped_frame_info=None):
    '''
    Merge bouts of the same type that are separated by less than threshold_sec.
    Optionally skip merges that would bridge over dropped frames.
    
    Parameters:
    -----------
    bouts_df : pd.DataFrame
        DataFrame with columns 'start', 'end', 'type', etc.
    threshold_sec : float
        Maximum gap (in seconds) between bouts to merge
    fr : float
        Frame rate for converting threshold to frames
    dropped_frame_info : dict or None
        Output from get_dropped_frame_info(). If provided, merges that would
        bridge over dropped frames are skipped.
    
    Returns:
    --------
    merged_df : pd.DataFrame
        DataFrame with close bouts merged
    merge_count : int
        Number of merges performed
    skipped_due_to_drops : int
        Number of potential merges skipped due to dropped frames
    '''
    if bouts_df.empty:
        return bouts_df.copy(), 0, 0
    
    threshold_frames = threshold_sec * fr
    
    # Sort by start frame
    sorted_df = bouts_df.sort_values('start').reset_index(drop=True)
    
    merged_bouts = []
    merge_count = 0
    skipped_due_to_drops = 0
    
    # Initialize with first bout
    current_bout = sorted_df.iloc[0].to_dict()
    
    for i in range(1, len(sorted_df)):
        next_bout = sorted_df.iloc[i]
        gap = next_bout['start'] - current_bout['end']
        
        # Check if bouts are close enough to potentially merge
        if gap <= threshold_frames:
            # Check if gap contains dropped frames
            if dropped_frame_info is not None:
                gap_has_drops = False
                gap_start = int(current_bout['end']) + 1
                gap_end = int(next_bout['start']) - 1
                
                if gap_start <= gap_end:  # There is actually a gap
                    for drop_start, drop_end in dropped_frame_info['dropped_ranges']:
                        # Check if dropped range overlaps with gap
                        if drop_start <= gap_end and drop_end >= gap_start:
                            gap_has_drops = True
                            break
                
                if gap_has_drops:
                    # Don't merge - save current bout and start new one
                    merged_bouts.append(current_bout)
                    current_bout = next_bout.to_dict()
                    skipped_due_to_drops += 1
                    continue
            
            # Safe to merge - extend current bout
            current_bout['end'] = next_bout['end']
            merge_count += 1
        else:
            # Gap too large - save current bout and start new one
            merged_bouts.append(current_bout)
            current_bout = next_bout.to_dict()
    
    # Don't forget the last bout
    merged_bouts.append(current_bout)
    
    merged_df = pd.DataFrame(merged_bouts)
    
    # Recalculate length if it exists
    if 'length' in merged_df.columns:
        merged_df['length'] = (merged_df['end'] - merged_df['start'] + 1) / fr
    
    return merged_df, merge_count, skipped_due_to_drops

def get_event_rates(sData, fr):
    '''
    Calculate event rates for each neuron from binarized event data.
    
    Parameters:
    -----------
    sData : pd.DataFrame
        Binarized event data with rows = neurons, columns = frames
    fr : float
        Frame rate in Hz
    
    Returns:
    --------
    event_rates : pd.Series
        Event rate (events/second) for each neuron
    median_rate : float
        Median event rate across all neurons
    mean_rate : float
        Mean event rate across all neurons
    '''
    n_frames = sData.shape[1]
    duration_sec = n_frames / fr
    
    # Sum events per neuron, divide by duration
    events_per_neuron = sData.sum(axis=1)
    event_rates = events_per_neuron / duration_sec
    
    median_rate = event_rates.median()
    mean_rate = event_rates.mean()
    
    return event_rates, median_rate, mean_rate

def get_dropped_frame_info(neural_data):
    '''
    Identify dropped frames from discontinuities in neural data column names.
    
    Parameters:
    -----------
    neural_data : pd.DataFrame
        Neural data with columns = frame numbers (may be non-contiguous)
    
    Returns:
    --------
    info : dict
        'valid_frames': set of frame numbers that exist in the data
        'dropped_ranges': list of (start, end) tuples for each dropped interval
                          (these are the missing frame numbers, inclusive)
    '''
    # Get frame numbers as integers
    frame_nums = np.array([int(col) for col in neural_data.columns])
    valid_frames = set(frame_nums)
    
    # Find gaps
    diffs = np.diff(frame_nums)
    gap_indices = np.where(diffs > 1)[0]
    
    dropped_ranges = []
    for idx in gap_indices:
        # Frame before gap
        last_valid = frame_nums[idx]
        # Frame after gap
        first_valid_after = frame_nums[idx + 1]
        # Dropped frames are everything in between
        dropped_start = last_valid + 1
        dropped_end = first_valid_after - 1
        dropped_ranges.append((dropped_start, dropped_end))
    
    return {
        'valid_frames': valid_frames,
        'dropped_ranges': dropped_ranges
    }
