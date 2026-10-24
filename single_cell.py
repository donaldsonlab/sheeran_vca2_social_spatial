"""single_cell.py

python functions written to assist in analysis of individual neurons' calcium activity with respect to animal behavior

written by a combination of Will Sheeran (WMS) and Claude Opus 4.6 in response to prompts from WMS; all Claude-generated code is vetted and validated by WMS

last edit 3/27/2026

"""

#imports
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import scipy.signal
import utils
import decoding
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve, auc

def events_on_position_plot(sData, xy_df, beh_rast, neurons, trials='social', only_ints=False, 
                            color_by='none', figsize=None, save_path=None, show=True):
    '''
    Plot discrete calcium events (from S data) on the animal's XY position during any combination of trials.

    Parameters:
    -----------
    sData : pd.DataFrame
        Binarized event data (rows=neurons, columns=frames)
    xy_df : pd.DataFrame
        XY position data (rows='X','Y', columns=frames)
    beh_rast : pd.DataFrame
        Behavioral raster (rows=behaviors, columns=frames)
    neurons : int or list of int
        Neuron index or indices to plot. If list, creates subplots.
    trials : str
        Which trial(s) to plot: 'social' (both social trials combined),
        'empty1', 'social1', 'social2', or 'empty2'
    only_ints : bool
        If True, only show events during active interactions and highlight
        interaction trajectory segments in black
    color_by : str
        How to color events:
        - 'none': all events in red
        - 'trial': pink=social1, cyan=social2 (only valid when trials='social')
        - 'identity': purple=partner, green=novel (only valid for social trials; implies only_ints=True)
    figsize : tuple or None
        Figure size. If None, auto-calculated based on number of neurons.
    save_path : str or None
        Path to save figure
    show : bool
        Whether to display the plot

    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    '''
    if color_by not in ['none', 'trial', 'identity']:
        raise ValueError(f"color_by must be 'none', 'trial', or 'identity', got '{color_by}'")
    
    if color_by == 'trial' and trials != 'social':
        raise ValueError("color_by='trial' only valid when trials='social'")
    
    if color_by == 'identity' and trials in ['empty1', 'empty2']:
        raise ValueError("color_by='identity' not valid for empty trials (no partner/novel distinction)")
    
    if color_by == 'identity':
        only_ints = True
    
    # Handle single neuron input
    if isinstance(neurons, (int, np.integer)):
        neurons = [neurons]
    
    n_neurons = len(neurons)
    
    # Determine subplot layout
    if n_neurons == 1:
        nrows, ncols = 1, 1
    elif n_neurons <= 4:
        nrows, ncols = 1, n_neurons
    else:
        ncols = min(4, n_neurons)
        nrows = int(np.ceil(n_neurons / ncols))
    
    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    # --- Shared data (compute once) ---
    xyT = xy_df.T.copy()
    
    # Build trial mask
    if trials == 'social':
        trial_mask = (beh_rast.loc['social1'] == 1) | (beh_rast.loc['social2'] == 1)
    else:
        expected_trials = ['empty1', 'social1', 'social2', 'empty2']
        if trials not in expected_trials:
            raise ValueError(f"'trials' must be 'social' or one of {expected_trials}, got '{trials}'")
        trial_mask = (beh_rast.loc[trials] == 1)
    
    trial_cols = beh_rast.columns[trial_mask]
    trial_xyT = xyT.loc[trial_cols, :]
    
    # Active interaction frames within selected trials (if needed)
    if only_ints:
        active_trial_mask = trial_mask & (beh_rast.loc['active'] == 1)
        active_trial_cols = beh_rast.columns[active_trial_mask]
        active_trial_xyT = xyT.loc[active_trial_cols, :]
        
        # Compute bout groupings for black trajectory overlay
        frame_diff = active_trial_xyT.index.to_series().astype(int).diff()
        bout_id = (frame_diff > 1).cumsum()
    
    # Trial-specific frames for color_by='trial'
    if color_by == 'trial':
        if trials == 'social':
            s1_cols = beh_rast.columns[beh_rast.loc['social1'] == 1]
            s2_cols = beh_rast.columns[beh_rast.loc['social2'] == 1]
            if only_ints:
                s1_cols = beh_rast.columns[(beh_rast.loc['social1'] == 1) & (beh_rast.loc['active'] == 1)]
                s2_cols = beh_rast.columns[(beh_rast.loc['social2'] == 1) & (beh_rast.loc['active'] == 1)]
        else:
            raise ValueError(f"To label events by trial, 'trials' must = 'social'")
            
    # Identity-specific frames for color_by='identity'
    if color_by == 'identity':
        partner_cols = beh_rast.columns[trial_mask & (beh_rast.loc['partner_active'] == 1)]
        novel_cols = beh_rast.columns[trial_mask & (beh_rast.loc['novel_active'] == 1)]
    
    # --- Per-neuron plotting ---
    for i, neuron in enumerate(neurons):
        ax = axes_flat[i]
        
        # Background trajectory
        ax.plot(trial_xyT['X'], trial_xyT['Y'], linestyle='-', linewidth=0.8, 
                alpha=0.65, color='gray', zorder=1)
        
        # darker overlay for active interaction segments
        if only_ints:
            for _, bout in active_trial_xyT.groupby(bout_id):
                ax.plot(bout['X'], bout['Y'], color='black', linewidth=0.8, alpha=0.65, zorder=2)
        
        # Plot events
        if color_by == 'trial':
            s1_sd = sData.loc[:, s1_cols]
            s2_sd = sData.loc[:, s2_cols]
            
            s1_event_cols = s1_sd.columns[s1_sd.loc[neuron] == 1]
            s2_event_cols = s2_sd.columns[s2_sd.loc[neuron] == 1]
            
            ax.plot(xyT.loc[s1_event_cols, 'X'], xyT.loc[s1_event_cols, 'Y'], 'o', 
                    markersize=4, color='deeppink', zorder=3, label='Social 1')
            ax.plot(xyT.loc[s2_event_cols, 'X'], xyT.loc[s2_event_cols, 'Y'], 'o', 
                    markersize=4, color='aqua', zorder=3, label='Social 2')
        
        elif color_by == 'identity':
            partner_sd = sData.loc[:, partner_cols]
            novel_sd = sData.loc[:, novel_cols]
            
            partner_event_cols = partner_sd.columns[partner_sd.loc[neuron] == 1]
            novel_event_cols = novel_sd.columns[novel_sd.loc[neuron] == 1]
            
            ax.plot(xyT.loc[partner_event_cols, 'X'], xyT.loc[partner_event_cols, 'Y'], 'o', 
                    markersize=4, color='#6750A1', zorder=3, label='Partner')
            ax.plot(xyT.loc[novel_event_cols, 'X'], xyT.loc[novel_event_cols, 'Y'], 'o', 
                    markersize=4, color='#55B948', zorder=3, label='Novel')
        
        else:  # color_by == 'none'
            if only_ints:
                source_sd = sData.loc[:, active_trial_cols]
            else:
                source_sd = sData.loc[:, trial_cols]
            
            event_cols = source_sd.columns[source_sd.loc[neuron] == 1]
            ax.plot(xyT.loc[event_cols, 'X'], xyT.loc[event_cols, 'Y'], 'o', 
                    markersize=4, color='red', zorder=3, label='Events')
        
        ax.set_aspect('equal')
        ax.set_title(f'Cell {neuron}', fontsize=11)
        ax.set_xlabel('focal animal position (mm)')

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            if only_ints:
                handles.append(Line2D([0], [0], color='black', linewidth=1.5, alpha=0.65))
                labels.append('Active investigation')
            ax.legend(handles, labels, fontsize=8, loc='best')
    
    # Hide unused subplots
    for j in range(n_neurons, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes

def plot_traces_with_bouts(neural_data, bout_dict, neuron_indices=None, 
                          frame_window=None, bout_color='identity', vertical_spacing=10, 
                          smooth_sigma=5, figsize=(14, 8), data_type='Cdata',
                          show_window_separators=True, save_path=None, show=True):
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
    frame_window : tuple, list of tuples, or None
        Single (start_frame, end_frame) tuple, a list of such tuples for 
        multiple discontinuous windows plotted contiguously, or None for all frames.
    bout_color : str
        'identity' or 'location' — determines color scheme for bout overlays
    vertical_spacing : float
        Vertical spacing between neuron traces
    smooth_sigma : float
        Gaussian smoothing sigma (set to 0 for no smoothing)
    figsize : tuple
        Figure size (width, height)
    data_type : str
        'Cdata', 'c_z', 'dff', or 'dff_z' — determines y-axis label
    show_window_separators : bool
        If True and multiple frame_windows provided, draw dashed vertical lines
        at window boundaries
    save_path : str or None
        Path to save figure. If None, doesn't save
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Default color schemes
    identity_colors = {
        'partner': '#4B2E8D', 'novel': '#07AC4B',
        'partner_left': '#4B2E8D', 'partner_right': '#4B2E8D',
        'p_l': '#4B2E8D', 'p_r': '#4B2E8D',
        'novel_left': '#07AC4B', 'novel_right': '#07AC4B',
        'n_l': '#07AC4B', 'n_r': '#07AC4B'
    }
    location_colors = {
        'left': '#5c433d', 'right': '#6cc0ec',
        'partner_left': '#5c433d', 'partner_right': '#6cc0ec',
        'p_l': '#5c433d', 'p_r': '#6cc0ec',
        'novel_left': '#5c433d', 'novel_right': '#6cc0ec',
        'n_l': '#5c433d', 'n_r': '#6cc0ec',
        'empty1_left': '#5c433d', 'l_e1': '#5c433d',
        'empty1_right': '#6cc0ec', 'r_e1': '#6cc0ec'
    }
    
    # Label mapping for legend (collapse subclasses into parent classes)
    identity_labels = {
        'partner': 'Partner', 'novel': 'Novel',
        'partner_left': 'Partner', 'partner_right': 'Partner',
        'p_l': 'Partner', 'p_r': 'Partner',
        'novel_left': 'Novel', 'novel_right': 'Novel',
        'n_l': 'Novel', 'n_r': 'Novel'
    }

    location_labels = {
        'left': 'Left', 'right': 'Right',
        'partner_left': 'Left', 'partner_right': 'Right',
        'p_l': 'Left', 'p_r': 'Right',
        'novel_left': 'Left', 'novel_right': 'Right',
        'n_l': 'Left', 'n_r': 'Right',
        'l_e1': 'Left', 'r_e1': 'Right'
    }

    # Select neurons to plot
    if neuron_indices is None:
        neuron_indices = list(range(min(5, len(neural_data))))
    ncells = len(neuron_indices)
    
    # --- Normalize frame_window to a list of (start, end) tuples ---
    if frame_window is None:
        t = neural_data.columns.astype(float)
        windows = [(t[0], t[-1])]
    elif isinstance(frame_window, tuple):
        windows = [frame_window]
    else:
        windows = list(frame_window)
    
    # --- Build concatenated data across windows ---
    # For each window, extract the neural data and track where each window
    # starts/ends in the concatenated x-axis (using original frame numbers)
    t_full = neural_data.columns.astype(float)
    
    concat_t = []          # concatenated x-axis values (original frame numbers)
    concat_traces = {}     # neuron_idx -> concatenated trace values
    separator_positions = []  # x-positions for separator lines
    
    # Initialize traces dict
    for neuron_idx in neuron_indices:
        concat_traces[neuron_idx] = []
    
    running_offset = 0  # cumulative offset to close gaps between windows
    
    frame_map = {}
    for w_idx, (w_start, w_end) in enumerate(windows):
        w_start, w_end = int(w_start), int(w_end)
        mask = (t_full >= w_start) & (t_full <= w_end)
        window_t = t_full[mask]
        
        if len(window_t) == 0:
            continue
        
        # Offset: shift this window's frames so they start right after the previous window
        if w_idx == 0:
            # First window — no offset needed, use original frame numbers
            offset = 0
        else:
            # Gap between end of previous concatenated data and start of this window
            offset = running_offset - window_t[0]
        
        shifted_t = window_t + offset
        concat_t.append(shifted_t)

        #add 
        for orig_frame in window_t:
            frame_map[orig_frame] = orig_frame + offset
        
        # Extract traces for this window
        window_neural = neural_data.loc[:, mask]
        for neuron_idx in neuron_indices:
            trace = window_neural.iloc[neuron_idx].values
            if smooth_sigma > 0:
                trace = gaussian_filter1d(trace, sigma=smooth_sigma)
            concat_traces[neuron_idx].append(trace)
        
        # Track separator position (at the boundary between this and next window)
        running_offset = shifted_t[-1] + 1  # +1 so next window doesn't overlap
        if w_idx < len(windows) - 1:
            separator_positions.append(shifted_t[-1])
    
    # Flatten concatenated arrays
    concat_t = np.concatenate(concat_t)
    for neuron_idx in neuron_indices:
        concat_traces[neuron_idx] = np.concatenate(concat_traces[neuron_idx])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- Plot bout overlays ---
    for bout_type, bout_df in bout_dict.items():
        if bout_color == 'identity':
            color = identity_colors.get(bout_type, '#CCCCCC')
        elif bout_color == 'location':
            color = location_colors.get(bout_type, '#CCCCCC')
        else:
            color = '#CCCCCC'
        
        if bout_color == 'identity':
            legend_label = identity_labels.get(bout_type, bout_type)
        elif bout_color == 'location':
            legend_label = location_labels.get(bout_type, bout_type)
        else:
            legend_label = bout_type

        first_plotted = False
        for idx, event in bout_df.iterrows():
            x_start_orig = event['start']
            x_end_orig = event['end']
            
            # For each window, check if this bout overlaps and plot the overlapping portion
            for w_start, w_end in windows:
                w_start, w_end = int(w_start), int(w_end)
                
                # Clip bout to window boundaries
                clipped_start = max(x_start_orig, w_start)
                clipped_end = min(x_end_orig, w_end)
                
                if clipped_start <= clipped_end:
                    # Map to shifted coordinates
                    if clipped_start in frame_map and clipped_end in frame_map:
                        shifted_start = frame_map[clipped_start]
                        shifted_end = frame_map[clipped_end]
                    else:
                        # Find nearest mapped frames
                        mapped_frames = np.array(list(frame_map.keys()))
                        start_nearest = mapped_frames[np.argmin(np.abs(mapped_frames - clipped_start))]
                        end_nearest = mapped_frames[np.argmin(np.abs(mapped_frames - clipped_end))]
                        shifted_start = frame_map[start_nearest]
                        shifted_end = frame_map[end_nearest]
                    
                    label = legend_label if not first_plotted else ""
                    ax.axvspan(shifted_start, shifted_end, color=color, alpha=0.3, label=label)
                    first_plotted = True
    
    # --- Plot neural traces ---
    for plot_idx, neuron_idx in enumerate(neuron_indices):
        vertical_offset = plot_idx * vertical_spacing
        trace = concat_traces[neuron_idx]
        
        ax.plot(concat_t, trace + vertical_offset, color='black', 
               alpha=0.8, linewidth=1.2, label='')
        
        ax.text(concat_t[0] - (concat_t[-1] - concat_t[0]) * 0.02, vertical_offset, 
               f'{neuron_idx}', va='center', ha='right', fontsize=9)
    
    # --- Draw window separators ---
    if show_window_separators and len(separator_positions) > 0:
        for sep_x in separator_positions:
            ax.axvline(x=sep_x, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, zorder=5)
    
    # Formatting
    ax.set_ylim(-vertical_spacing * 0.5, ncells * vertical_spacing + vertical_spacing * 0.5)
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_yticks([])
    
    if data_type == 'Cdata':
        ylabel_text = 'Fluorescence (a.u.)'
    elif data_type == 'c_z':
        ylabel_text = 'norm. fluorescence'
    elif data_type == 'dff':
        ylabel_text = 'ΔF/F'
    elif data_type == 'dff_z':
        ylabel_text = 'ΔF/F (z-scored)'
    else:
        ylabel_text = 'Signal'
    
    ax.set_ylabel(ylabel_text, fontsize=12)
    ax.set_title('Neural Activity During Active Interactions', fontsize=14, weight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def get_mean_dff_by_subclass(ints_dict,neural_data):
    '''
    function to get the average activity of each neuron across all frames of a given interaction subclass, for any combination of interaction subclasses
    '''
    mean_dffs_by_subclass = {}
    for subclass, ints_df in ints_dict.items():
        if len(ints_df) == 0:
            print(f"Warning: {subclass} has no bouts, skipping")
            mean_dffs_by_subclass[subclass] = pd.Series(np.nan, index=neural_data.index)
            continue
        subclass_traces, _ = utils.traces_pv_by_bout(ints_df,neural_data)
        subclass_traces_total = [trace_df for trace_df in subclass_traces.values()]
        subclass_traces_total_df = pd.concat(subclass_traces_total,axis=1)
        mean_dffs_subclass = subclass_traces_total_df.mean(axis=1)
        mean_dffs_by_subclass[subclass]=mean_dffs_subclass

    return mean_dffs_by_subclass

def plot_activity_heatmap(mean_dffs_dict, neuron, mode='social', cmap='RdBu_r', 
                          symmetric_cbar=True, vmin=None, vmax=None, figsize=None, save_path=None, show=True):
    '''
    Plot a heatmap of a neuron's mean activity across interaction subclasses.

    Parameters:
    -----------
    mean_dffs_dict : dict
        Output of get_mean_dff_by_subclass(). Keys are subclass labels, 
        values are pd.Series with one value per neuron.
        For social mode: expects keys 'p_l', 'p_r', 'n_l', 'n_r'
        For empty mode: expects keys 'l_e1', 'r_e1'
    neuron : int
        Neuron index to plot
    mode : str
        'social' for 2x2 identity × location heatmap
        'empty' for 1x2 location-only heatmap
    cmap : str
        Matplotlib colormap name
    symmetric_cbar : bool
        If True, center the colorbar at 0 so positive and negative 
        deviations are equally visible
    figsize : tuple or None
        Figure size. If None, auto-determined by mode.
    save_path : str or None
        Path to save figure
    show : bool
        Whether to display the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    '''
    if mode == 'social':
        # Build 2x2 matrix: rows = identity (partner, novel), cols = location (left, right)
        expected_keys = ['p_l', 'p_r', 'n_l', 'n_r']
        missing = [k for k in expected_keys if k not in mean_dffs_dict]
        if missing:
            raise ValueError(f"Missing subclass keys for social mode: {missing}")
        
        data = np.array([
            [mean_dffs_dict['p_l'].iloc[neuron], mean_dffs_dict['p_r'].iloc[neuron]],
            [mean_dffs_dict['n_l'].iloc[neuron], mean_dffs_dict['n_r'].iloc[neuron]]
        ])
        row_labels = ['Partner', 'Novel']
        col_labels = ['Left', 'Right']
        title = f'Cell {neuron} — Social Trial Activity'
        
        if figsize is None:
            figsize = (5, 4)
    
    elif mode == 'empty':
        expected_keys = ['l_e1', 'r_e1']
        missing = [k for k in expected_keys if k not in mean_dffs_dict]
        if missing:
            raise ValueError(f"Missing subclass keys for empty mode: {missing}")
        
        data = np.array([
            [mean_dffs_dict['l_e1'].iloc[neuron], mean_dffs_dict['r_e1'].iloc[neuron]]
        ])
        row_labels = ['Empty 1']
        col_labels = ['Left', 'Right']
        title = f'Cell {neuron} — Empty Trial Activity'
        
        if figsize is None:
            figsize = (5, 2.5)
    
    else:
        raise ValueError(f"mode must be 'social' or 'empty', got '{mode}'")
    
    # Handle NaN values
    if np.any(np.isnan(data)):
        print(f"Warning: NaN values in heatmap for cell {neuron} (missing subclass data)")
    
    # Determine color limits and normalization
    if vmin is not None and vmax is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    elif symmetric_cbar:
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax
        norm = None  # default linear norm is fine when symmetric
    else:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        norm = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if norm is not None:
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean z-scored ΔF/F', fontsize=10)
    
    # Label axes
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    
    # Annotate each cell with its value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                # Choose text color based on background intensity
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                        fontsize=12, fontweight='bold', color=text_color)
    
    ax.set_title(title, fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def construct_auROC_imports(ints0_df,ints1_df,neural_data):
    '''
    function to prepare imports for an auROC calculation between two types of interaction
    '''
    #construct the behavior raster -- ints0_df = 0 class, ints1_df = 1 class
    type0 = ints0_df['type'].iloc[0]
    type1 = ints1_df['type'].iloc[0]

    col_type = type(neural_data.columns[0])
    
    all_ints = pd.concat([ints0_df,ints1_df]).sort_values('start').reset_index(drop=True)
    total_rast = []
    all_neural_data_per_int = []
    for idx, row in all_ints.iterrows():
        int_start = row['start']
        int_end = row['end']

        num_frames = int(int_end)-int(int_start)+1 #.loc is inclusive so we need to add 1 here to align the length
        if row['type']==type0:
            row_rast = np.tile(0,num_frames)
        else:
            row_rast = np.tile(1,num_frames)
        total_rast.append(row_rast)

        #slice out the neural data for each int in the same order
        neural_int = neural_data.loc[:,col_type(int_start):col_type(int_end)].copy()
        
        #confirm that the sliced neural data for this bout is indeed the expected # of frames long (i.e., it did not skip over any drops due to .loc indexing)
        assert neural_int.shape[1] == num_frames, f"Bout {idx} ({int_start}-{int_end}): expected {num_frames} frames, got {neural_int.shape[1]}"

        all_neural_data_per_int.append(neural_int)
        
    total_rast = np.concatenate(total_rast)
    all_neural_data_per_int = pd.concat(all_neural_data_per_int,axis=1)
    #neuron_neural_data_per_int = all_neural_data_per_int.iloc[neuron].values

    return total_rast, all_neural_data_per_int

def auROC_vs_null(subclass_bouts_dict, neural_data,
                  cutoff=5, null_size=1000, null_method='circular',
                  discontinuity_frames=None, verbose=True):
    '''
    Generate null distribution for auROC by circularly permuting neural data.
    
    Parameters:
    -----------
    subclass_bouts_dict : dict
        Dictionary mapping subclass names to behavior bout (interaction) DataFrames.
    neural_data : pd.DataFrame
        Neural data with rows=neurons, columns=frames (column names may be 
        non-contiguous frame numbers due to dropped frames)
    cutoff : int
        %ile above which and below which to consider a neuron significantly tuned
    null_size : int
        Number of null permutations to run
    null_method : str
        'circular': simple circular shift of all neurons by the same amount
        'circular_aware': reject permutations where discontinuities land in bouts
    discontinuity_frames : list of frame numbers, optional
        Frame numbers (column names) where discontinuities exist (e.g., dropped frames, 
        trial boundaries). These mark frames where there is a discontinuity BEFORE that frame.
        Required if null_method='circular_aware'.
    verbose : bool
        Print progress updates
    
    Returns:
    --------
 
    '''
    
    if null_method not in ['circular', 'circular_aware']:
        raise ValueError("null_method must be 'circular' or 'circular_aware'")
    
    if null_method == 'circular_aware' and discontinuity_frames is None:
        raise ValueError("discontinuity_frames required for 'circular_aware' method")
    
    n_frames = neural_data.shape[1]
    n_neurons = neural_data.shape[0]

    if len(subclass_bouts_dict) != 2:
        raise ValueError("'subclass_bouts_dict' must contain exactly 2 bouts dataframes")

    #get each subclass and its corresponding bouts, and assign each as the 0 class or the 1 class
    subclasses = list(subclass_bouts_dict.keys())
    class0_df, class1_df = subclass_bouts_dict[subclasses[0]], subclass_bouts_dict[subclasses[1]]

    #get the imports for the auROC calculation for the true score
    rast, neural_data_per_bout = construct_auROC_imports(class0_df,class1_df,neural_data)

    #perform auROC calculation --> get true auROC score for each neuron
    true_scores = []
    for _, values in neural_data_per_bout.iterrows():
        auROC_score = roc_auc_score(rast,values)
        true_scores.append(auROC_score)
    
    # For circular_aware, we need to work in positional indices
    if null_method == 'circular_aware':
        # Create mapping from frame numbers (column names) to positional indices
        col_to_pos = {col: i for i, col in enumerate(neural_data.columns)}

        col_type = type(neural_data.columns[0])
        
        # Get all bout intervals as POSITIONS, not frame numbers
        all_bout_positions = []
        for bouts_df in subclass_bouts_dict.values():
            for _, row in bouts_df.iterrows():
                start_pos = col_to_pos[col_type(row['start'])]
                end_pos = col_to_pos[col_type(row['end'])]
                all_bout_positions.append((start_pos, end_pos))
        
        # Convert discontinuity frame numbers to positions
        discontinuity_positions = []
        for f in discontinuity_frames:
            f_key = col_type(f)
            if f_key in col_to_pos:
                discontinuity_positions.append(col_to_pos[f_key])

    null_auROCs = pd.DataFrame(np.zeros((neural_data.shape[0], null_size)))
    n_rejected = 0
    perm_idx = 0
    success_iters = 0
    
    while success_iters < null_size:
        if verbose and perm_idx % 100 == 0:
            print(f"Permutation {perm_idx}, completed: {success_iters}/{null_size}")
        
        perm_idx += 1
        
        # Safety valve to prevent infinite loops
        if perm_idx > null_size * 3:
            print(f"Warning: Too many rejected permutations. Stopping early.")
            break
        
        # Generate same shift for all neurons
        shift = np.random.randint(0, n_frames)
        shifts = np.array([shift] * n_neurons)
        
        # For circular_aware, check if any discontinuity lands in a bout
        if null_method == 'circular_aware':
            reject = False
            
            # After np.roll(data, shift):
            # - Original position i lands at position (i + shift) % n_frames
            # - Original position 0 lands at position shift
            # - The seam (where original end meets original beginning) is at position shift
            
            # Check if seam lands within any bout
            seam_position = shift % n_frames
            for bout_start, bout_end in all_bout_positions:
                if bout_start <= seam_position <= bout_end:
                    reject = True
                    break
            
            # Check additional discontinuities (e.g., dropped frames, trial boundaries)
            if not reject and discontinuity_positions:
                for disc_pos in discontinuity_positions:
                    # Original discontinuity at position disc_pos now appears at position
                    # (disc_pos + shift) % n_frames in the shifted data
                    shifted_disc_position = (disc_pos + shift) % n_frames
                    
                    for bout_start, bout_end in all_bout_positions:
                        if bout_start <= shifted_disc_position <= bout_end:
                            reject = True
                            break
                    
                    if reject:
                        break
            
            if reject:
                n_rejected += 1
                continue
        
        # Apply circular shift to neural data
        neural_data_shifted = decoding.circular_shift_neural_data(neural_data, shifts)

        #get imports for auROC
        _, shift_neural_data_per_bout = construct_auROC_imports(class0_df,class1_df,neural_data_shifted)

        shift_scores = []
        for _, values in shift_neural_data_per_bout.iterrows():
            auROC_shift = roc_auc_score(rast,values)
            shift_scores.append(auROC_shift)
        null_auROCs.iloc[:,success_iters]=shift_scores

        success_iters += 1

    null_dist_comp = pd.DataFrame(np.zeros((neural_data.shape[0], 3)),columns=['low','high','true'])

    pos_cells = []
    neg_cells = []

    null_auROCs = null_auROCs.iloc[:, :success_iters] #just in case the safety valve triggers so we don't have a ton of 0s at the end
    
    for idx, row in null_auROCs.iterrows():
        low = np.percentile(row,cutoff)
        high = np.percentile(row,(100-cutoff))
        true = true_scores[idx]
        newrow = [low,high,true]
        null_dist_comp.loc[idx]=newrow

        if true>high:
            pos_cells.append(idx)
        elif true<low:
            neg_cells.append(idx)
            
    true_scores_df = pd.DataFrame({'auROC':true_scores},index=neural_data_per_bout.index)
    
    results = {
        'n_iters':success_iters,
        '0_class':subclasses[0],
        '1_class':subclasses[1],
        '1_class_cells':pos_cells,
        '0_class_cells':neg_cells,
        'null_dist_comp':null_dist_comp,
        'all_real_scores':true_scores_df,
        'all_null_scores':null_auROCs
    }
    
    return results

def plot_auROC_vs_null(auroc_results, neuron, bins=30, figsize=(6, 4), save_path=None, show=True):
    '''
    Plot a neuron's true auROC score against its null distribution.

    Parameters:
    -----------
    auroc_results : dict
        Output from auROC_vs_null()
    neuron : int
        Neuron index to plot
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure
    show : bool
        Whether to display the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    null_scores = auroc_results['all_null_scores'].loc[neuron].values
    true_score = auroc_results['all_real_scores'].loc[neuron, 'auROC']
    cutoff_low = auroc_results['null_dist_comp'].loc[neuron, 'low']
    cutoff_high = auroc_results['null_dist_comp'].loc[neuron, 'high']
    class0 = auroc_results['0_class']
    class1 = auroc_results['1_class']

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(null_scores, bins=bins, color='gray', alpha=0.7, edgecolor='white', label='Null distribution')
    ax.axvline(true_score, color='mediumblue', linewidth=2, linestyle='-', label=f'True auROC ({true_score:.3f})')
    ax.axvline(cutoff_low, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label=f'Lower cutoff ({cutoff_low:.3f})')
    ax.axvline(cutoff_high, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label=f'Upper cutoff ({cutoff_high:.3f})')

    # Determine significance
    if true_score > cutoff_high:
        sig_text = f'Significant: tuned to {class1}'
    elif true_score < cutoff_low:
        sig_text = f'Significant: tuned to {class0}'
    else:
        sig_text = 'Not significant'

    ax.set_xlabel('auROC', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Cell {neuron} — {class0} vs {class1}\n{sig_text}', fontsize=12)
    ax.legend(fontsize=9, loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def plot_auROC_curve(auroc_results, neural_data, subclass_bouts_dict, neuron, 
                     figsize=(5, 5), save_path=None, show=True):
    '''
    Plot the ROC curve for a single neuron's discrimination between two interaction classes.

    Parameters:
    -----------
    auroc_results : dict
        Output from auROC_vs_null() — used for class labels and the true auROC score
    neural_data : pd.DataFrame
        Neural data with rows=neurons, columns=frames (e.g., z-scored dF/F)
    subclass_bouts_dict : dict
        The same bout dictionary passed to auROC_vs_null() (2 keys, each mapping to a bouts DataFrame)
    neuron : int
        Neuron index to plot
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure
    show : bool
        Whether to display the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    class0 = auroc_results['0_class']
    class1 = auroc_results['1_class']

    subclasses = list(subclass_bouts_dict.keys())
    class0_df = subclass_bouts_dict[subclasses[0]]
    class1_df = subclass_bouts_dict[subclasses[1]]

    rast, neural_data_per_bout = construct_auROC_imports(class0_df, class1_df, neural_data)
    neuron_values = neural_data_per_bout.iloc[neuron].values

    fpr, tpr, _ = roc_curve(rast, neuron_values)
    auroc_score = auroc_results['all_real_scores'].loc[neuron, 'auROC']

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='mediumblue', linewidth=3, label=f'auROC = {auroc_score:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linewidth=2, linestyle='--', label='Chance (0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'Cell {neuron} — {class0} vs {class1}', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def plot_tuning_pie(tuning_dict, numcells=None, mode='social', 
                    figsize=(10, 8), title=None, save_path=None, show=True):
    '''
    Plot a nested pie chart summarizing neuron tuning classifications.

    Parameters:
    -----------
    tuning_dict : dict
        For mode='social': must contain keys 'partner_cells', 'novel_cells', 
        'left_cells', 'right_cells', 'partner_left_mixed', 'partner_right_mixed',
        'novel_left_mixed', 'novel_right_mixed' — each a list of neuron indices.
        For mode='empty': must contain keys 'left_cells', 'right_cells' — each
        a list of neuron indices.
        If 'numcells' is a key in tuning_dict, it will be used automatically.
    numcells : int or None
        Total number of neurons in the session. If None, pulled from tuning_dict['numcells'].
    mode : str
        'social' for identity/location/mixed classification
        'empty' for location-only classification
    figsize : tuple
        Figure size
    title : str or None
        Custom title. If None, auto-generated.
    save_path : str or None
        Path to save figure
    show : bool
        Whether to display the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    # Resolve numcells
    if numcells is None:
        if 'numcells' in tuning_dict:
            numcells = tuning_dict['numcells']
        else:
            raise ValueError("numcells must be provided either as argument or as key in tuning_dict")

    # Color definitions
    identity_colors = {'partner': '#4B2E8D', 'novel': '#07AC4B'}
    location_colors = {'left': '#5c433d', 'right': '#6cc0ec'}
    untuned_color = '#2E2E2E'

    if mode == 'social':
        # Count cells in each category
        n_partner = len(tuning_dict['partner_cells'])
        n_novel = len(tuning_dict['novel_cells'])
        n_left = len(tuning_dict['left_cells'])
        n_right = len(tuning_dict['right_cells'])
        n_pl_mixed = len(tuning_dict['partner_left_mixed'])
        n_pr_mixed = len(tuning_dict['partner_right_mixed'])
        n_nl_mixed = len(tuning_dict['novel_left_mixed'])
        n_nr_mixed = len(tuning_dict['novel_right_mixed'])

        n_identity = n_partner + n_novel
        n_location = n_left + n_right
        n_mixed = n_pl_mixed + n_pr_mixed + n_nl_mixed + n_nr_mixed
        n_tuned = n_identity + n_location + n_mixed
        n_untuned = numcells - n_tuned

        # Inner ring: broad categories
        inner_sizes = [n_identity, n_location, n_mixed, n_untuned]
        inner_labels_raw = ['Identity\nSelective', 'Location\nSelective', 
                            'Mixed\nSelective', 'Untuned']
        inner_colors = ['#BABABA', '#787777', '#505050', untuned_color]

        # Outer ring: specific subtypes
        outer_sizes = [n_partner, n_novel, n_left, n_right,
                       n_pl_mixed, n_pr_mixed, n_nl_mixed, n_nr_mixed, n_untuned]
        outer_labels_raw = ['Partner', 'Novel', 'Left', 'Right',
                            'Partner/Left', 'Partner/Right', 
                            'Novel/Left', 'Novel/Right', 'Untuned']
        # Mixed cells get identity color as fill; location dots added via hatch
        outer_colors = [identity_colors['partner'], identity_colors['novel'],
                        location_colors['left'], location_colors['right'],
                        identity_colors['partner'], identity_colors['partner'],
                        identity_colors['novel'], identity_colors['novel'],
                        untuned_color]
        # Track which outer wedges are mixed and their location hatch color
        # Indices: 4=p_l, 5=p_r, 6=n_l, 7=n_r
        mixed_wedge_hatches = {
            4: "#645B58",
            5: location_colors['right'],
            6: "#645B58",
            7: location_colors['right'],
        }

    elif mode == 'empty':
        n_left = len(tuning_dict['left_empty1_cells'])
        n_right = len(tuning_dict['right_empty1_cells'])
        n_tuned = n_left + n_right
        n_untuned = numcells - n_tuned

        # Inner ring: tuned vs untuned
        inner_sizes = [n_tuned, n_untuned]
        inner_labels_raw = ['Location\nSelective', 'Untuned']
        inner_colors = ['#787777', untuned_color]

        # Outer ring: left vs right vs untuned
        outer_sizes = [n_left, n_right, n_untuned]
        outer_labels_raw = ['Left', 'Right', 'Untuned']
        outer_colors = [location_colors['left'], location_colors['right'], untuned_color]

    else:
        raise ValueError(f"mode must be 'social' or 'empty', got '{mode}'")

    # Filter out zero-count categories to avoid cluttered labels
    inner_labels = [f'{name}\nn={size} ({size/numcells*100:.1f}%)' if size > 0 else ''
                    for name, size in zip(inner_labels_raw, inner_sizes)]
    outer_labels = [f'{name}\nn={size} ({size/numcells*100:.1f}%)' if size > 0 else ''
                    for name, size in zip(outer_labels_raw, outer_sizes)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Inner pie (donut)
    wedges1, texts1 = ax.pie(
        inner_sizes,
        labels=inner_labels,
        colors=inner_colors,
        startangle=90,
        radius=0.7,
        wedgeprops=dict(width=0.3, edgecolor='white'),
        textprops={'fontsize': 9},
        labeldistance=0.35
    )

    # Outer pie
    wedges2, texts2 = ax.pie(
        outer_sizes,
        labels=outer_labels,
        colors=outer_colors,
        startangle=90,
        radius=1.0,
        wedgeprops=dict(width=0.35, edgecolor='white'),
        textprops={'fontsize': 10},
        labeldistance=1.2
    )

    # Add location-colored dot hatching to mixed cell wedges
    if mode == 'social':
        import matplotlib as mpl
        for wedge_idx, hatch_color in mixed_wedge_hatches.items():
            if outer_sizes[wedge_idx] > 0:
                wedges2[wedge_idx].set_hatch('++')
                wedges2[wedge_idx].set_edgecolor(hatch_color)
                wedges2[wedge_idx].set_linewidth(0.5)
        # matplotlib hatching uses edgecolor, so re-apply white edges to non-mixed wedges
        for i, wedge in enumerate(wedges2):
            if i not in mixed_wedge_hatches:
                wedge.set_edgecolor('white')

    for text in texts1:
        text.set_horizontalalignment('center')
    for text in texts2:
        text.set_horizontalalignment('center')

    if title is None:
        if mode == 'social':
            title = 'Neuron Classification — Social Trials'
        else:
            title = 'Neuron Classification — Empty Trial'
            
    ax.text(-1, -0.8, f'Total n = {numcells}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title(title, fontsize=13, weight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

