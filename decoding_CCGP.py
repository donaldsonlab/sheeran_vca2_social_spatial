"""decoding.py

python functions written to assist with decoding behavioral variables from 1-photon calcium imaging data

written by Claude Opus 4.5 in response to prompts from Will Sheeran (University of Colorado, Boulder); all code is vetted and validated by WMS

last edit 2/6/2026

"""

#imports
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal
import utils
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def extract_population_vectors(pts_df, neural_data, col_type):
    '''
    Extract mean population vectors for each pseudotrial. Very similar to utils.traces_pv_by_bout() but more direct and better output for the decoder function below
    
    Parameters:
    -----------
    pts_df : pd.DataFrame
        Pseudotrial DataFrame with 'start' and 'end' columns
    neural_data : pd.DataFrame
        Neural data with rows=neurons, columns=frames
    col_type : type
        Type of neural_data column index (for proper slicing)
    
    Returns:
    --------
    pvs : np.ndarray
        Array of shape (n_pseudotrials, n_neurons)
    '''
    pvs = []
    for idx, row in pts_df.iterrows():
        start = col_type(row['start'])
        end = col_type(row['end'])
        
        bout_neural = neural_data.loc[:, start:end]
        pv = bout_neural.mean(axis=1).values
        pvs.append(pv)
    
    return np.array(pvs)

#primary decoding function
def population_decoder(subclass_pts_dict, class_assignments, neural_data,
                       n_iterations=100, test_fraction=0.5,
                       min_train_per_class=2, random_seed=None, C=1.0,
                       n_neurons_subsample=None):
    '''
    function written by Claude Opus 4.5 (vetted by WMS)

    
    Decode binary classes from population vectors using SVM,
    with interaction-level train/test separation.
    
    Flexible to handle any number of subclasses grouped into two classes.
    
    Note that the default parameters require each subclass to have a minimum of two discrete interactions with three total pseudotrials across them

    Parameters:
    -----------
    subclass_pts_dict : dict
        Dictionary mapping subclass names to pseudotrial DataFrames.
        Each DataFrame must have columns: 'start', 'end', 'int', 'type'
        
        Example (4 subclass):
            {'pr': pr_pts, 'pl': pl_pts, 'nr': nr_pts, 'nl': nl_pts}
        
        Example (2 subclass):
            {'left': left_pts, 'right': right_pts}
    
    class_assignments : dict
        Dictionary specifying which subclasses belong to each binary class.
        Must have exactly two keys (class labels), each mapping to a list of subclass names (which are the 'type' column of the pseudotrial DataFrames).
        
        Example (identity decoding):
            {'partner': ['pr', 'pl'], 'novel': ['nr', 'nl']}
        
        Example (location decoding):
            {'right': ['pr', 'nr'], 'left': ['pl', 'nl']}
        
        Example (location during the first empty trial):
            {'left': ['l_e1'], 'right': ['r_e1']}
    
    neural_data : pd.DataFrame
        Neural data timeseries with rows=neurons, columns=frames (e.g., z-scored dF/F)
    n_iterations : int
        Number of random train/test splits to perform
    test_fraction : float
        Fraction of interactions to assign to test set (default=0.5)
    min_train_per_class : int
        Minimum pseudotrials per subclass in training set; skip iteration if not met
    random_seed : int or None
        Random seed for reproducibility
    C : int or float
        regularization paramter for SVM (default = 1.0)
    n_neurons_subsample : int or None
        If provided, randomly subsample this many neurons at each iteration.
        Useful for comparing across sessions/animals with different neuron counts.
        Must be <= number of neurons in neural_data.
    
    Returns:
    --------
    results : dict
        'accuracies': list of accuracy for each valid iteration
        'mean_accuracy': mean across iterations
        'std_accuracy': std across iterations
        'n_valid_iterations': how many iterations met minimum criteria
        'n_skipped_iterations': how many iterations were skipped
        'n_train_per_iter': list of training set sizes per iteration
        'n_test_per_iter': list of test set sizes per iteration
        'class_labels': tuple of (class_0_name, class_1_name)
    '''
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_neurons_total = neural_data.shape[0]
    if n_neurons_subsample is not None:
        if n_neurons_subsample > n_neurons_total:
            raise ValueError(f"n_neurons_subsample ({n_neurons_subsample}) > available neurons ({n_neurons_total})")
        if n_neurons_subsample < 1:
            raise ValueError("n_neurons_subsample must be >= 1")

    # Validate class_assignments
    if len(class_assignments) != 2:
        raise ValueError("class_assignments must have exactly 2 classes")
    
    class_names = list(class_assignments.keys())
    class_0_name, class_1_name = class_names[0], class_names[1]
    
    # Validate that all referenced subclasses exist
    all_subclasses = class_assignments[class_0_name] + class_assignments[class_1_name]
    for subclass in all_subclasses:
        if subclass not in subclass_pts_dict:
            raise ValueError(f"Subclass '{subclass}' in class_assignments not found in subclass_pts_dict")
    
    # Copy and reset index for each subclass (wms translation -- if you've assigned a subclass to a binary category and that same subclass has a corresponding pts df in the dict, copy that df here)
    subclass_pts = {
        key: df.copy().reset_index(drop=True) 
        for key, df in subclass_pts_dict.items()
        if key in all_subclasses
    }
    
    # Get unique interactions for each subclass
    subclass_interactions = {
        key: df['int'].unique().tolist() 
        for key, df in subclass_pts.items()
    }
    
    # Check minimum interactions per subclass
    for subclass, ints in subclass_interactions.items():
        if len(ints) < 2:
            print(f"Warning: Subclass '{subclass}' has only {len(ints)} interaction(s).")
            print("Cannot perform train/test split. Need at least 2 interactions per subclass.")
            return None
    
    # Column type for neural data indexing
    col_type = type(neural_data.columns[0])
    
    accuracies = []
    f1_scores = []
    auroc_scores = []
    n_train_list = []
    n_test_list = []
    n_neurons_list = []
    weights_list = []
    neuron_indices_list = []
    skipped_iterations = 0
    
    for iteration in range(n_iterations):

        # Subsample neurons if requested (different random subset each iteration)
        if n_neurons_subsample is not None:
            neuron_indices = np.random.choice(n_neurons_total, size=n_neurons_subsample, replace=False)
            neural_data_iter = neural_data.iloc[neuron_indices, :]
        else:
            neural_data_iter = neural_data
        
        n_neurons_list.append(neural_data_iter.shape[0])

        train_pts_by_subclass = {}
        test_pts_by_subclass = {}
        
        # Step 1: Split interactions into train/test for each subclass
        for subclass in all_subclasses:
            ints = subclass_interactions[subclass].copy()
            np.random.shuffle(ints)
            
            n_test = max(1, int(len(ints) * test_fraction))
            # Ensure at least 1 interaction in train
            n_test = min(n_test, len(ints) - 1)
            
            test_ints = ints[:n_test]
            train_ints = ints[n_test:]
            
            pts_df = subclass_pts[subclass]
            
            train_pts_by_subclass[subclass] = pts_df[pts_df['int'].isin(train_ints)].copy()
            test_pts_by_subclass[subclass] = pts_df[pts_df['int'].isin(test_ints)].copy()

        #end of step 1 - train_ and test_pts_by_subclass have all pts from all subclasses but split half and half into each structure at the level of interaction

        # Step 2: Balance training set - downsample to smallest subclass
        min_train = min(len(df) for df in train_pts_by_subclass.values())
        
        if min_train < min_train_per_class:
            skipped_iterations += 1
            continue
        
        train_pts_balanced = {}
        for subclass, df in train_pts_by_subclass.items():
            if len(df) > min_train:
                train_pts_balanced[subclass] = utils.sample_pseudotrials_balanced(df, min_train)
            else:
                train_pts_balanced[subclass] = df.reset_index(drop=True)
        
        # Step 3: Balance test set - downsample to smallest subclass
        min_test = min(len(df) for df in test_pts_by_subclass.values())
        
        if min_test < 1:
            skipped_iterations += 1
            continue
        
        test_pts_balanced = {}
        for subclass, df in test_pts_by_subclass.items():
            if len(df) > min_test:
                test_pts_balanced[subclass] = utils.sample_pseudotrials_balanced(df, min_test)
            else:
                test_pts_balanced[subclass] = df.reset_index(drop=True)
        
        # Step 4: Extract population vectors and build X, y arrays
        X_train_class0 = []
        X_train_class1 = []
        X_test_class0 = []
        X_test_class1 = []
        
        for subclass in class_assignments[class_0_name]:
            pvs_train = extract_population_vectors(train_pts_balanced[subclass], neural_data_iter, col_type)
            pvs_test = extract_population_vectors(test_pts_balanced[subclass], neural_data_iter, col_type)
            X_train_class0.append(pvs_train)
            X_test_class0.append(pvs_test)
        
        for subclass in class_assignments[class_1_name]:
            pvs_train = extract_population_vectors(train_pts_balanced[subclass], neural_data_iter, col_type)
            pvs_test = extract_population_vectors(test_pts_balanced[subclass], neural_data_iter, col_type)
            X_train_class1.append(pvs_train)
            X_test_class1.append(pvs_test)
        
        X_train_class0 = np.vstack(X_train_class0)
        X_train_class1 = np.vstack(X_train_class1)
        X_test_class0 = np.vstack(X_test_class0)
        X_test_class1 = np.vstack(X_test_class1)
        
        X_train = np.vstack([X_train_class0, X_train_class1])
        X_test = np.vstack([X_test_class0, X_test_class1])
        
        # Step 5: Create labels
        y_train = np.array([0]*len(X_train_class0) + [1]*len(X_train_class1))
        y_test = np.array([0]*len(X_test_class0) + [1]*len(X_test_class1))
        
        # Step 6: Scale features and train SVM -- wms note - I don't think I want to do this bc data is already z-scored within this session
        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_test_scaled = scaler.transform(X_test)
        
        clf = LinearSVC(C=C, max_iter=10000)
        clf.fit(X_train, y_train)
        
        # Store SVM weights for this iteration
        weights = clf.coef_[0].copy()  # Shape: (n_neurons_used,)
        weights_list.append(weights)

        if n_neurons_subsample is not None:
            neuron_indices_list.append(neuron_indices.copy())
        
        #clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test)
        y_scores = clf.decision_function(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if len(np.unique(y_test)) == 2:
            auroc = roc_auc_score(y_test, y_scores)
        else:
            auroc = np.nan

        #accuracy = clf.score(X_test_scaled, y_test)
        #accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        auroc_scores.append(auroc)
        n_train_list.append(len(y_train))
        n_test_list.append(len(y_test))
    
    # Compile results
    results = {
        'accuracies': accuracies,
        'mean_accuracy': np.mean(accuracies) if accuracies else np.nan,
        'std_accuracy': np.std(accuracies) if accuracies else np.nan,
        'f1_scores': f1_scores,
        'mean_f1': np.mean(f1_scores) if f1_scores else np.nan,
        'std_f1': np.std(f1_scores) if f1_scores else np.nan,
        'auroc_scores': auroc_scores,
        'mean_auroc': np.nanmean(auroc_scores) if auroc_scores else np.nan,  # nanmean to handle edge cases
        'std_auroc': np.nanstd(auroc_scores) if auroc_scores else np.nan,
        'n_valid_iterations': len(accuracies),
        'n_skipped_iterations': skipped_iterations,
        'n_train_per_iter': n_train_list,
        'n_test_per_iter': n_test_list,
        'n_neurons_per_iter': n_neurons_list,
        'class_labels': (class_0_name, class_1_name)
    }
    # SVM weights across iterations
    if n_neurons_subsample is None:
        # All iterations used same neurons - can stack into matrix
        results['weights_matrix'] = np.vstack(weights_list) if weights_list else None
        # Shape: (n_valid_iterations, n_neurons)
    else:
        # Different neurons each iteration - return as lists
        results['weights_per_iter'] = weights_list
        results['neuron_indices_per_iter'] = neuron_indices_list
    
    return results

def compute_decoding_importance(results):
    '''
    Compute the importance of individual neurons to a linear SVM's decoding of a variable pair (averaged across all cross-validation folds)
    
    Parameters:
    -----------
    results : dict
        Output from population_decoder with weights_matrix
        (requires n_neurons_subsample=None during decoding)
    
    Returns:
    --------
    coding_importance : np.ndarray
        Shape (n_neurons,) - abs value of the mean SVM weight across all cross-validations for each neuron
    mean_weights : np.ndarray
        Shape (n_neurons,) - same as coding_importance but with the original signs of the weights
    '''
    if 'weights_matrix' not in results or results['weights_matrix'] is None:
        raise ValueError("No weights_matrix found. Run decoder with n_neurons_subsample=None")
    
    weights_matrix = results['weights_matrix']  # (n_iterations, n_neurons)
    
    mean_weights = np.mean(weights_matrix, axis=0)
    decoding_importance = np.abs(mean_weights)
    
    return decoding_importance, mean_weights #in case you want to have the signs of each for some reason

def generate_null_distribution(subclass_pts_dict, class_assignments, neural_data,
                                n_permutations=500, n_decoder_iterations=20,
                                null_method='circular', discontinuity_frames=None,
                                test_fraction=0.5, min_train_per_class=2,
                                random_seed=None, verbose=True, C=1.0, n_neurons_subsample=None):
    '''
    Generate null distribution for decoding accuracy by circularly permuting neural data.
    
    Parameters:
    -----------
    subclass_pts_dict : dict
        Dictionary mapping subclass names to pseudotrial DataFrames.
        Same format as for population_decoder()
    class_assignments : dict
        Dictionary specifying which subclasses belong to each binary class.
        Same format as for population_decoder()
    neural_data : pd.DataFrame
        Neural data with rows=neurons, columns=frames (column names may be 
        non-contiguous frame numbers due to dropped frames)
    n_permutations : int
        Number of null permutations to run
    n_decoder_iterations : int
        Number of decoder iterations per permutation (fewer than real decoding is fine)
    null_method : str
        'circular': simple circular shift of all neurons by the same amount
        'circular_aware': reject permutations where discontinuities land in bouts
    discontinuity_frames : list of frame numbers, optional
        Frame numbers (column names) where discontinuities exist (e.g., dropped frames, 
        trial boundaries). These mark frames where there is a discontinuity BEFORE that frame.
        Required if null_method='circular_aware'.
    test_fraction : float
        Fraction of interactions to assign to test set
    min_train_per_class : int
        Minimum pseudotrials per subclass in training set
    random_seed : int or None
        Random seed for reproducibility
    verbose : bool
        Print progress updates
    
    Returns:
    --------
    null_results : dict
        'null_accuracies': list of mean accuracy for each permutation
        'mean_null': mean of null distribution
        'std_null': std of null distribution
        'n_permutations_completed': how many permutations were successful
        'n_permutations_rejected': how many were rejected (circular_aware only)
    '''
    import numpy as np
    import pandas as pd
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if null_method not in ['circular', 'circular_aware']:
        raise ValueError("null_method must be 'circular' or 'circular_aware'")
    
    if null_method == 'circular_aware' and discontinuity_frames is None:
        raise ValueError("discontinuity_frames required for 'circular_aware' method")
    
    n_frames = neural_data.shape[1]
    n_neurons = neural_data.shape[0]
    
    # For circular_aware, we need to work in positional indices
    if null_method == 'circular_aware':
        # Create mapping from frame numbers (column names) to positional indices
        col_to_pos = {col: i for i, col in enumerate(neural_data.columns)}

        col_type = type(neural_data.columns[0])
        
        # Get all bout intervals as POSITIONS, not frame numbers
        all_bout_positions = []
        for subclass, pts_df in subclass_pts_dict.items():
            if subclass in [s for subclasses in class_assignments.values() for s in subclasses]:
                for _, row in pts_df.iterrows():
                    start_pos = col_to_pos[col_type(row['start'])]
                    end_pos = col_to_pos[col_type(row['end'])]
                    all_bout_positions.append((start_pos, end_pos))
        
        # Convert discontinuity frame numbers to positions
        discontinuity_positions = []
        for f in discontinuity_frames:
            f_key = col_type(f)
            if f_key in col_to_pos:
                discontinuity_positions.append(col_to_pos[f_key])
    
    null_accuracies = []
    null_f1s = []
    null_aurocs = []
    n_rejected = 0
    perm_idx = 0
    
    while len(null_accuracies) < n_permutations:
        if verbose and perm_idx % 50 == 0:
            print(f"Permutation {perm_idx}, completed: {len(null_accuracies)}/{n_permutations}")
        
        perm_idx += 1
        
        # Safety valve to prevent infinite loops
        if perm_idx > n_permutations * 10:
            print(f"Warning: Too many rejected permutations. Stopping early.")
            break
        
        # Generate same shift for all neurons (preserves population dynamics)
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
        neural_data_shifted = circular_shift_neural_data(neural_data, shifts)
        
        # Run decoder on shifted data
        results = population_decoder(
            subclass_pts_dict=subclass_pts_dict,
            class_assignments=class_assignments,
            neural_data=neural_data_shifted,
            n_iterations=n_decoder_iterations,
            test_fraction=test_fraction,
            min_train_per_class=min_train_per_class,
            random_seed=None,  # Don't fix seed within permutations
            C=C,
            n_neurons_subsample=n_neurons_subsample
        )
        
        if results is not None and results['n_valid_iterations'] > 0:
            null_accuracies.append(results['mean_accuracy'])
            null_f1s.append(results['mean_f1'])
            null_aurocs.append(results['mean_auroc'])
    
    if verbose:
        print(f"Completed {len(null_accuracies)} permutations, rejected {n_rejected}")
    
    null_results = {
        'null_accuracies': null_accuracies,
        'mean_null_accuracy': np.mean(null_accuracies) if null_accuracies else np.nan,
        'std_null_accuracy': np.std(null_accuracies) if null_accuracies else np.nan,
        'null_f1s': null_f1s,
        'mean_null_f1': np.mean(null_f1s) if null_f1s else np.nan,
        'std_null_f1': np.std(null_f1s) if null_f1s else np.nan,
        'null_aurocs': null_aurocs,
        'mean_null_auroc': np.mean(null_aurocs) if null_aurocs else np.nan,
        'std_null_auroc': np.std(null_aurocs) if null_aurocs else np.nan,
        'n_permutations_completed': len(null_accuracies),
        'n_permutations_rejected': n_rejected
    }
    
    return null_results


def circular_shift_neural_data(neural_data, shifts):
    '''
    Circularly shift each neuron's trace by a specified amount.
    
    Parameters:
    -----------
    neural_data : pd.DataFrame
        Neural data with rows=neurons, columns=frames
    shifts : array-like
        Shift amount for each neuron (length must equal number of neurons)
    
    Returns:
    --------
    shifted_data : pd.DataFrame
        Shifted neural data with same structure as input
    '''
    import numpy as np
    import pandas as pd
    
    shifted_array = np.zeros_like(neural_data.values)
    
    for i, shift in enumerate(shifts):
        shifted_array[i, :] = np.roll(neural_data.iloc[i].values, shift)
    
    shifted_data = pd.DataFrame(
        shifted_array,
        index=neural_data.index,
        columns=neural_data.columns
    )
    
    return shifted_data


def compute_p_value(observed_score, null_scores):
    '''
    Compute p-value: proportion of null accuracies >= observed.
    
    Parameters:
    -----------
    observed_score : float
        Accuracy from real decoding
    null_scores : list
        List of scores from null distribution
    
    Returns:
    --------
    p_value : float
    '''
    
    null_scores = np.array(null_scores)
    p_value = np.mean(null_scores >= observed_score)
    
    return p_value

def generate_null_label_shuffle(subclass_pts_dict, class_assignments, neural_data,
                                 n_permutations=500, n_decoder_iterations=20,
                                 test_fraction=0.5, min_train_per_class=2,
                                 shuffle_level='interaction', random_seed=None, 
                                 verbose=True, C=1.0, n_neurons_subsample=None):
    '''
    Generate null distribution by shuffling behavioral labels.
    
    Parameters:
    -----------
    subclass_pts_dict : dict
        Dictionary mapping subclass names to pseudotrial DataFrames
    class_assignments : dict
        Dictionary specifying which subclasses belong to each binary class
    neural_data : pd.DataFrame
        Neural data with rows=neurons, columns=frames
    n_permutations : int
        Number of null permutations to run
    n_decoder_iterations : int
        Number of decoder iterations per permutation
    test_fraction : float
        Fraction of interactions to assign to test set
    min_train_per_class : int
        Minimum pseudotrials per subclass in training set
    shuffle_level : str
        'interaction' - shuffle which interactions belong to which subclass (preserves bout structure)
        'pseudotrial' - shuffle individual pseudotrial labels (destroys bout structure)
    random_seed : int or None
        Random seed for reproducibility
    verbose : bool
        Print progress updates
    
    Returns:
    --------
    null_results : dict
        'null_accuracies': list of mean accuracy for each permutation
        'mean_null': mean of null distribution
        'std_null': std of null distribution
        'n_permutations_completed': how many permutations were successful
    '''
  
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if shuffle_level not in ['interaction', 'pseudotrial']:
        raise ValueError("shuffle_level must be 'interaction' or 'pseudotrial'")
    
    # Get all subclasses involved in this comparison
    all_subclasses = []
    for class_name, subclasses in class_assignments.items():
        all_subclasses.extend(subclasses)
    
    null_accuracies = []
    null_f1s = []
    null_aurocs = []
    
    for perm_idx in range(n_permutations):
        if verbose and perm_idx % 50 == 0:
            print(f"Permutation {perm_idx}/{n_permutations}")
        
        if shuffle_level == 'interaction':
            # Collect all interactions with their pseudotrials
            all_interactions = []
            for subclass in all_subclasses:
                pts_df = subclass_pts_dict[subclass]
                for int_id in pts_df['int'].unique():
                    int_pts = pts_df[pts_df['int'] == int_id].copy()
                    all_interactions.append({
                        'original_subclass': subclass,
                        'pts': int_pts
                    })
            
            # Shuffle subclass assignments
            shuffled_subclasses = [d['original_subclass'] for d in all_interactions]
            np.random.shuffle(shuffled_subclasses)
            
            # Rebuild subclass_pts_dict with shuffled assignments
            # Track next available int ID for each subclass
            next_int_id = {subclass: 0 for subclass in all_subclasses}
            shuffled_pts_dict = {subclass: [] for subclass in all_subclasses}

            for i, interaction in enumerate(all_interactions):
                new_subclass = shuffled_subclasses[i]
                pts = interaction['pts'].copy()
                pts['type'] = new_subclass
                # Assign a new unique int ID for this subclass
                pts['int'] = next_int_id[new_subclass]
                next_int_id[new_subclass] += 1
                shuffled_pts_dict[new_subclass].append(pts)
            
            # Concatenate
            shuffled_subclass_pts = {}
            for subclass in all_subclasses:
                if shuffled_pts_dict[subclass]:
                    shuffled_subclass_pts[subclass] = pd.concat(shuffled_pts_dict[subclass], ignore_index=True)
                else:
                    shuffled_subclass_pts[subclass] = pd.DataFrame(columns=['start', 'end', 'int', 'type'])
        
        else:  # pseudotrial level 
            # Collect all pseudotrials
            all_pts = []
            for subclass in all_subclasses:
                pts_df = subclass_pts_dict[subclass].copy()
                pts_df['type'] = subclass
                all_pts.append(pts_df)
            all_pts_combined = pd.concat(all_pts, ignore_index=True)
            
            # Shuffle subclass labels
            shuffled_types = all_pts_combined['type'].values.copy()
            np.random.shuffle(shuffled_types)
            all_pts_combined['type'] = shuffled_types
            
            # Split back into subclass dicts
            shuffled_subclass_pts = {}
            for subclass in all_subclasses:
                subclass_df = all_pts_combined[all_pts_combined['type'] == subclass].copy()
                if len(subclass_df) > 0:
                    # Each pseudotrial gets its own unique int ID
                    subclass_df['int'] = range(len(subclass_df))
                shuffled_subclass_pts[subclass] = subclass_df.reset_index(drop=True)
        
        # Check if we have enough data in each subclass after shuffling
        valid_shuffle = True
        for subclass in all_subclasses:
            if subclass not in shuffled_subclass_pts or len(shuffled_subclass_pts[subclass]) == 0:
                valid_shuffle = False
                break
            n_ints = shuffled_subclass_pts[subclass]['int'].nunique()
            if n_ints < 2:
                valid_shuffle = False
                break
        
        if not valid_shuffle:
            continue
        
        # Run decoder on shuffled labels
        results = population_decoder(
            subclass_pts_dict=shuffled_subclass_pts,
            class_assignments=class_assignments,
            neural_data=neural_data,
            n_iterations=n_decoder_iterations,
            test_fraction=test_fraction,
            min_train_per_class=min_train_per_class,
            random_seed=None,
            C=C,
            n_neurons_subsample=n_neurons_subsample
        )
        
        if results is not None and results['n_valid_iterations'] > 0:
            null_accuracies.append(results['mean_accuracy'])
            null_f1s.append(results['mean_f1'])
            null_aurocs.append(results['mean_auroc'])
    
    if verbose:
        print(f"Completed {len(null_accuracies)} permutations")
    
    null_results = {
        'null_accuracies': null_accuracies,
        'mean_null_accuracy': np.mean(null_accuracies) if null_accuracies else np.nan,
        'std_null_accuracy': np.std(null_accuracies) if null_accuracies else np.nan,
        'null_f1s': null_f1s,
        'mean_null_f1': np.mean(null_f1s) if null_f1s else np.nan,
        'std_null_f1': np.std(null_f1s) if null_f1s else np.nan,
        'null_aurocs': null_aurocs,
        'mean_null_auroc': np.nanmean(null_aurocs) if null_aurocs else np.nan, 
        'std_null_auroc': np.nanstd(null_aurocs) if null_aurocs else np.nan,
        'n_permutations_completed': len(null_accuracies),
        'shuffle_level': shuffle_level
}
    
    return null_results

# ---------------------------------------------------------------------------
# CCGP (Cross-Condition Generalization Performance) — added by MEK
# ---------------------------------------------------------------------------

def ccgp_decoder(
    subclass_pts_dict,
    train_class_assignments,
    test_class_assignments,
    neural_data,
    n_iterations=100,
    min_train_per_class=2,
    random_seed=None,
    C=1.0,
    n_neurons_subsample=None,
):
    """
    Decode a binary dichotomy with a cross-condition train/test split.

    The dichotomy (e.g., partner vs. novel) is fixed. What varies is which
    context (e.g., location) is used for training vs. testing.

    Example — identity CCGP, fold A:
        train_class_assignments = {'partner': ['pl'], 'novel': ['nl']}
        test_class_assignments  = {'partner': ['pr'], 'novel': ['nr']}

    The test set is fixed as all pseudotrials from the held-out context.
    Iterations serve to average over different random downsamples of the
    balanced train and test sets, giving stable estimates.

    Parameters
    ----------
    subclass_pts_dict : dict
        Maps subclass name -> pseudotrial DataFrame (start/end/int/type columns).
    train_class_assignments : dict
        Exactly 2 keys (class labels), each mapping to a list of subclass names
        used for TRAINING. E.g. {'partner': ['pl'], 'novel': ['nl']}.
    test_class_assignments : dict
        Same 2 class labels, subclasses used for TESTING (held-out context).
        E.g. {'partner': ['pr'], 'novel': ['nr']}.
        Key order must match train_class_assignments.
    neural_data : pd.DataFrame
        Rows = neurons, columns = frames.
    n_iterations : int
        Number of random downsampling iterations to average over.
    min_train_per_class : int
        Minimum pseudotrials per class in training set; skip iteration if not met.
    random_seed : int or None
    C : float
        SVM regularization parameter.
    n_neurons_subsample : int or None
        If set, randomly subsample this many neurons each iteration.

    Returns
    -------
    results : dict or None
        None if no valid iterations completed.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    train_classes = list(train_class_assignments.keys())
    test_classes  = list(test_class_assignments.keys())
    if set(train_classes) != set(test_classes):
        raise ValueError(
            f"train and test class_assignments must have the same class labels. "
            f"Got train={train_classes}, test={test_classes}"
        )
    if len(train_classes) != 2:
        raise ValueError("class_assignments must have exactly 2 classes")

    class_0, class_1 = train_classes[0], train_classes[1]

    train_subclasses = train_class_assignments[class_0] + train_class_assignments[class_1]
    test_subclasses  = test_class_assignments[class_0]  + test_class_assignments[class_1]
    all_subclasses   = train_subclasses + test_subclasses

    for sc in all_subclasses:
        if sc not in subclass_pts_dict:
            raise ValueError(f"Subclass '{sc}' not found in subclass_pts_dict")

    subclass_pts = {
        k: df.copy().reset_index(drop=True)
        for k, df in subclass_pts_dict.items()
        if k in all_subclasses
    }

    n_neurons_total = neural_data.shape[0]
    if n_neurons_subsample is not None:
        if n_neurons_subsample > n_neurons_total:
            raise ValueError(
                f"n_neurons_subsample ({n_neurons_subsample}) > "
                f"available neurons ({n_neurons_total})"
            )

    col_type = type(neural_data.columns[0])

    accuracies, f1_scores, auroc_scores = [], [], []
    skipped = 0

    for _ in range(n_iterations):

        if n_neurons_subsample is not None:
            neuron_idx = np.random.choice(n_neurons_total, size=n_neurons_subsample, replace=False)
            nd = neural_data.iloc[neuron_idx, :]
        else:
            nd = neural_data

        # --- Training set ---
        train_c0_dfs = [subclass_pts[sc] for sc in train_class_assignments[class_0]]
        train_c1_dfs = [subclass_pts[sc] for sc in train_class_assignments[class_1]]
        n_c0 = sum(len(df) for df in train_c0_dfs)
        n_c1 = sum(len(df) for df in train_c1_dfs)
        min_train = min(n_c0, n_c1)

        if min_train < min_train_per_class:
            skipped += 1
            continue

        X_train_c0 = _ccgp_vstack_subsample(train_c0_dfs, min_train, nd, col_type)
        X_train_c1 = _ccgp_vstack_subsample(train_c1_dfs, min_train, nd, col_type)
        X_train = np.vstack([X_train_c0, X_train_c1])
        y_train = np.array([0] * len(X_train_c0) + [1] * len(X_train_c1))

        # --- Test set (full held-out context) ---
        test_c0_dfs = [subclass_pts[sc] for sc in test_class_assignments[class_0]]
        test_c1_dfs = [subclass_pts[sc] for sc in test_class_assignments[class_1]]
        n_test_c0 = sum(len(df) for df in test_c0_dfs)
        n_test_c1 = sum(len(df) for df in test_c1_dfs)
        min_test = min(n_test_c0, n_test_c1)

        if min_test < 1:
            skipped += 1
            continue

        X_test_c0 = _ccgp_vstack_subsample(test_c0_dfs, min_test, nd, col_type)
        X_test_c1 = _ccgp_vstack_subsample(test_c1_dfs, min_test, nd, col_type)
        X_test = np.vstack([X_test_c0, X_test_c1])
        y_test = np.array([0] * len(X_test_c0) + [1] * len(X_test_c1))

        # --- Fit and evaluate ---
        clf = LinearSVC(C=C, max_iter=10000)
        clf.fit(X_train, y_train)
        y_pred   = clf.predict(X_test)
        y_scores = clf.decision_function(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        if len(np.unique(y_test)) == 2:
            auroc_scores.append(roc_auc_score(y_test, y_scores))
        else:
            auroc_scores.append(np.nan)

    if not accuracies:
        return None

    return {
        'accuracies':           accuracies,
        'mean_accuracy':        np.mean(accuracies),
        'std_accuracy':         np.std(accuracies),
        'f1_scores':            f1_scores,
        'mean_f1':              np.mean(f1_scores),
        'std_f1':               np.std(f1_scores),
        'auroc_scores':         auroc_scores,
        'mean_auroc':           np.nanmean(auroc_scores),
        'std_auroc':            np.nanstd(auroc_scores),
        'n_valid_iterations':   len(accuracies),
        'n_skipped_iterations': skipped,
        'class_labels':         (class_0, class_1),
        'train_context':        train_subclasses,
        'test_context':         test_subclasses,
    }


def _ccgp_vstack_subsample(dfs, n, neural_data, col_type):
    """
    Stack population vectors from a list of pseudotrial DataFrames,
    then randomly subsample to exactly n rows. Private helper for ccgp_decoder.
    """
    pvs = np.vstack([
        extract_population_vectors(df, neural_data, col_type)
        for df in dfs if len(df) > 0
    ])
    idx = np.random.choice(len(pvs), size=n, replace=False)
    return pvs[idx]


def ccgp_both_folds(
    subclass_pts_dict,
    fold_A_train, fold_A_test,
    fold_B_train, fold_B_test,
    neural_data,
    n_iterations=100,
    min_train_per_class=2,
    random_seed=None,
    C=1.0,
    n_neurons_subsample=None,
):
    """
    Run ccgp_decoder for both folds and return averaged metrics.

    For identity CCGP:
        fold_A_train = {'partner': ['pl'], 'novel': ['nl']}
        fold_A_test  = {'partner': ['pr'], 'novel': ['nr']}
        fold_B_train = {'partner': ['pr'], 'novel': ['nr']}
        fold_B_test  = {'partner': ['pl'], 'novel': ['nl']}

    Returns
    -------
    results : dict with 'fold_A', 'fold_B', and metrics pooled across both folds.
    """
    seed_A = random_seed
    seed_B = (random_seed + 1) if random_seed is not None else None

    res_A = ccgp_decoder(
        subclass_pts_dict=subclass_pts_dict,
        train_class_assignments=fold_A_train,
        test_class_assignments=fold_A_test,
        neural_data=neural_data,
        n_iterations=n_iterations,
        min_train_per_class=min_train_per_class,
        random_seed=seed_A,
        C=C,
        n_neurons_subsample=n_neurons_subsample,
    )

    res_B = ccgp_decoder(
        subclass_pts_dict=subclass_pts_dict,
        train_class_assignments=fold_B_train,
        test_class_assignments=fold_B_test,
        neural_data=neural_data,
        n_iterations=n_iterations,
        min_train_per_class=min_train_per_class,
        random_seed=seed_B,
        C=C,
        n_neurons_subsample=n_neurons_subsample,
    )

    acc_vals, f1_vals, auroc_vals = [], [], []
    for res in [res_A, res_B]:
        if res is not None:
            acc_vals.extend(res['accuracies'])
            f1_vals.extend(res['f1_scores'])
            auroc_vals.extend(res['auroc_scores'])

    return {
        'fold_A':        res_A,
        'fold_B':        res_B,
        'mean_accuracy': np.mean(acc_vals)       if acc_vals   else np.nan,
        'std_accuracy':  np.std(acc_vals)        if acc_vals   else np.nan,
        'mean_f1':       np.mean(f1_vals)        if f1_vals    else np.nan,
        'std_f1':        np.std(f1_vals)         if f1_vals    else np.nan,
        'mean_auroc':    np.nanmean(auroc_vals)  if auroc_vals else np.nan,
        'std_auroc':     np.nanstd(auroc_vals)   if auroc_vals else np.nan,
    }


def generate_ccgp_null_condition_shuffle(
    subclass_pts_dict,
    fold_A_train, fold_A_test,
    fold_B_train, fold_B_test,
    neural_data,
    n_permutations=1000,
    n_decoder_iterations=20,
    min_train_per_class=2,
    random_seed=None,
    C=1.0,
    n_neurons_subsample=None,
    verbose=True,
):
    """
    Null distribution for CCGP via independent per-condition neuron index shuffle.

    For each permutation, each condition (pr, pl, nr, nl) gets an independent
    random permutation of neuron indices applied to its pseudotrial population
    vectors. This destroys cross-condition geometry (and therefore generalization)
    while preserving within-condition structure and overall decodability.

    Follows Boyle et al. / Biane et al. null model construction.
    Expected to produce null distributions centered at ~0.5.

    Parameters
    ----------
    subclass_pts_dict : dict
        Maps subclass name -> pseudotrial DataFrame (start/end/int/type columns).
    fold_A_train / fold_A_test / fold_B_train / fold_B_test : dict
        Class assignments for each fold.
    neural_data : pd.DataFrame
        Rows = neurons, columns = frames.
    n_permutations : int
        Number of null permutations (Boyle uses 1000).
    n_decoder_iterations : int
        Number of downsampling iterations per permutation.
    min_train_per_class : int
    random_seed : int or None
    C : float
    n_neurons_subsample : int or None
    verbose : bool

    Returns
    -------
    null_results : dict
        Same structure as other null functions for drop-in compatibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    all_subclasses = list(set(
        fold_A_train[list(fold_A_train.keys())[0]] +
        fold_A_train[list(fold_A_train.keys())[1]] +
        fold_A_test[list(fold_A_test.keys())[0]]   +
        fold_A_test[list(fold_A_test.keys())[1]]
    ))

    n_neurons = neural_data.shape[0]
    col_type  = type(neural_data.columns[0])

    # Pre-extract all population vectors per condition once
    # shape per condition: (n_pseudotrials, n_neurons)
    condition_pvs = {}
    for sc in all_subclasses:
        pts_df = subclass_pts_dict[sc]
        pvs = extract_population_vectors(pts_df, neural_data, col_type)
        condition_pvs[sc] = pvs

    null_accuracies, null_f1s, null_aurocs = [], [], []

    for perm_idx in range(n_permutations):
        if verbose and perm_idx % 100 == 0:
            print(f"  CCGP condition shuffle null: {perm_idx}/{n_permutations}")

        # Independent neuron permutation per condition
        shuffled_pvs = {}
        for sc in all_subclasses:
            perm = np.random.permutation(n_neurons)
            shuffled_pvs[sc] = condition_pvs[sc][:, perm]

        res = _ccgp_both_folds_from_pvs(
            shuffled_pvs=shuffled_pvs,
            fold_A_train=fold_A_train, fold_A_test=fold_A_test,
            fold_B_train=fold_B_train, fold_B_test=fold_B_test,
            n_iterations=n_decoder_iterations,
            min_train_per_class=min_train_per_class,
            C=C,
            n_neurons_subsample=n_neurons_subsample,
            n_neurons=n_neurons,
        )

        if res is not None and not np.isnan(res['mean_accuracy']):
            null_accuracies.append(res['mean_accuracy'])
            null_f1s.append(res['mean_f1'])
            null_aurocs.append(res['mean_auroc'])

    if verbose:
        print(f"  CCGP condition shuffle null done: {len(null_aurocs)} completed")

    return {
        'null_accuracies':          null_accuracies,
        'mean_null_accuracy':       np.mean(null_accuracies) if null_accuracies else np.nan,
        'std_null_accuracy':        np.std(null_accuracies)  if null_accuracies else np.nan,
        'null_f1s':                 null_f1s,
        'mean_null_f1':             np.mean(null_f1s)        if null_f1s        else np.nan,
        'std_null_f1':              np.std(null_f1s)         if null_f1s        else np.nan,
        'null_aurocs':              null_aurocs,
        'mean_null_auroc':          np.nanmean(null_aurocs)  if null_aurocs     else np.nan,
        'std_null_auroc':           np.nanstd(null_aurocs)   if null_aurocs     else np.nan,
        'n_permutations_completed': len(null_aurocs),
        'null_method':              'condition_neuron_shuffle',
    }


def _ccgp_both_folds_from_pvs(
    shuffled_pvs,
    fold_A_train, fold_A_test,
    fold_B_train, fold_B_test,
    n_iterations,
    min_train_per_class,
    C,
    n_neurons_subsample,
    n_neurons,
):
    """
    Run both CCGP folds directly from pre-extracted (shuffled) population vectors.
    Private helper for generate_ccgp_null_condition_shuffle.

    shuffled_pvs : dict mapping subclass -> np.ndarray of shape (n_pts, n_neurons)
    """
    def _run_fold(train_class_assignments, test_class_assignments):
        class_0, class_1 = list(train_class_assignments.keys())
        train_sc_c0 = train_class_assignments[class_0]
        train_sc_c1 = train_class_assignments[class_1]
        test_sc_c0  = test_class_assignments[class_0]
        test_sc_c1  = test_class_assignments[class_1]

        accuracies, f1_scores, auroc_scores = [], [], []
        skipped = 0

        for _ in range(n_iterations):
            if n_neurons_subsample is not None:
                neuron_idx = np.random.choice(n_neurons, size=n_neurons_subsample, replace=False)
            else:
                neuron_idx = np.arange(n_neurons)

            # Training set
            pvs_train_c0 = np.vstack([shuffled_pvs[sc][:, neuron_idx] for sc in train_sc_c0])
            pvs_train_c1 = np.vstack([shuffled_pvs[sc][:, neuron_idx] for sc in train_sc_c1])
            n_c0 = len(pvs_train_c0)
            n_c1 = len(pvs_train_c1)
            min_train = min(n_c0, n_c1)

            if min_train < min_train_per_class:
                skipped += 1
                continue

            idx_c0 = np.random.choice(n_c0, size=min_train, replace=False)
            idx_c1 = np.random.choice(n_c1, size=min_train, replace=False)
            X_train = np.vstack([pvs_train_c0[idx_c0], pvs_train_c1[idx_c1]])
            y_train = np.array([0] * min_train + [1] * min_train)

            # Test set
            pvs_test_c0 = np.vstack([shuffled_pvs[sc][:, neuron_idx] for sc in test_sc_c0])
            pvs_test_c1 = np.vstack([shuffled_pvs[sc][:, neuron_idx] for sc in test_sc_c1])
            n_test_c0 = len(pvs_test_c0)
            n_test_c1 = len(pvs_test_c1)
            min_test = min(n_test_c0, n_test_c1)

            if min_test < 1:
                skipped += 1
                continue

            idx_tc0 = np.random.choice(n_test_c0, size=min_test, replace=False)
            idx_tc1 = np.random.choice(n_test_c1, size=min_test, replace=False)
            X_test = np.vstack([pvs_test_c0[idx_tc0], pvs_test_c1[idx_tc1]])
            y_test = np.array([0] * min_test + [1] * min_test)

            clf = LinearSVC(C=C, max_iter=10000)
            clf.fit(X_train, y_train)
            y_pred   = clf.predict(X_test)
            y_scores = clf.decision_function(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            if len(np.unique(y_test)) == 2:
                auroc_scores.append(roc_auc_score(y_test, y_scores))
            else:
                auroc_scores.append(np.nan)

        if not accuracies:
            return None

        return {
            'accuracies':   accuracies,
            'f1_scores':    f1_scores,
            'auroc_scores': auroc_scores,
        }

    res_A = _run_fold(fold_A_train, fold_A_test)
    res_B = _run_fold(fold_B_train, fold_B_test)

    acc_vals, f1_vals, auroc_vals = [], [], []
    for res in [res_A, res_B]:
        if res is not None:
            acc_vals.extend(res['accuracies'])
            f1_vals.extend(res['f1_scores'])
            auroc_vals.extend(res['auroc_scores'])

    if not acc_vals:
        return None

    return {
        'fold_A':        res_A,
        'fold_B':        res_B,
        'mean_accuracy': np.mean(acc_vals),
        'std_accuracy':  np.std(acc_vals),
        'mean_f1':       np.mean(f1_vals),
        'std_f1':        np.std(f1_vals),
        'mean_auroc':    np.nanmean(auroc_vals),
        'std_auroc':     np.nanstd(auroc_vals),
    }


def generate_ccgp_null_shift(
    subclass_pts_dict,
    fold_A_train, fold_A_test,
    fold_B_train, fold_B_test,
    neural_data,
    n_permutations=500,
    n_decoder_iterations=20,
    null_method='circular_aware',
    discontinuity_frames=None,
    min_train_per_class=2,
    random_seed=None,
    C=1.0,
    n_neurons_subsample=None,
    verbose=True,
):
    """
    Circular shift null distribution for CCGP (both folds averaged per permutation).
    Kept for reference but condition_neuron_shuffle is preferred for CCGP.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if null_method not in ['circular', 'circular_aware']:
        raise ValueError("null_method must be 'circular' or 'circular_aware'")
    if null_method == 'circular_aware' and discontinuity_frames is None:
        raise ValueError("discontinuity_frames required for 'circular_aware'")

    n_frames  = neural_data.shape[1]
    n_neurons = neural_data.shape[0]

    if null_method == 'circular_aware':
        col_to_pos = {col: i for i, col in enumerate(neural_data.columns)}
        col_type   = type(neural_data.columns[0])

        all_subclasses_used = set(
            fold_A_train[list(fold_A_train.keys())[0]] +
            fold_A_train[list(fold_A_train.keys())[1]] +
            fold_A_test[list(fold_A_test.keys())[0]]   +
            fold_A_test[list(fold_A_test.keys())[1]]
        )
        all_bout_positions = []
        for sc in all_subclasses_used:
            for _, row in subclass_pts_dict[sc].iterrows():
                s = col_to_pos.get(col_type(row['start']))
                e = col_to_pos.get(col_type(row['end']))
                if s is not None and e is not None:
                    all_bout_positions.append((s, e))

        disc_positions = []
        for f in discontinuity_frames:
            fk = col_type(f)
            if fk in col_to_pos:
                disc_positions.append(col_to_pos[fk])

    null_accuracies, null_f1s, null_aurocs = [], [], []
    n_rejected = 0
    perm_idx   = 0

    while len(null_accuracies) < n_permutations:
        if verbose and perm_idx % 50 == 0:
            print(f"  CCGP shift null: {perm_idx} attempts, {len(null_accuracies)}/{n_permutations} completed")

        perm_idx += 1
        if perm_idx > n_permutations * 10:
            print("Warning: too many rejected permutations. Stopping early.")
            break

        shift = np.random.randint(0, n_frames)

        if null_method == 'circular_aware':
            reject = False
            seam   = shift % n_frames
            for bs, be in all_bout_positions:
                if bs <= seam <= be:
                    reject = True
                    break
            if not reject:
                for dp in disc_positions:
                    shifted_dp = (dp + shift) % n_frames
                    for bs, be in all_bout_positions:
                        if bs <= shifted_dp <= be:
                            reject = True
                            break
                    if reject:
                        break
            if reject:
                n_rejected += 1
                continue

        shifts     = np.array([shift] * n_neurons)
        nd_shifted = circular_shift_neural_data(neural_data, shifts)

        res = ccgp_both_folds(
            subclass_pts_dict=subclass_pts_dict,
            fold_A_train=fold_A_train, fold_A_test=fold_A_test,
            fold_B_train=fold_B_train, fold_B_test=fold_B_test,
            neural_data=nd_shifted,
            n_iterations=n_decoder_iterations,
            min_train_per_class=min_train_per_class,
            random_seed=None,
            C=C,
            n_neurons_subsample=n_neurons_subsample,
        )

        if not np.isnan(res['mean_accuracy']):
            null_accuracies.append(res['mean_accuracy'])
            null_f1s.append(res['mean_f1'])
            null_aurocs.append(res['mean_auroc'])

    if verbose:
        print(f"  CCGP shift null done: {len(null_accuracies)} completed, {n_rejected} rejected")

    return {
        'null_accuracies':          null_accuracies,
        'mean_null_accuracy':       np.mean(null_accuracies) if null_accuracies else np.nan,
        'std_null_accuracy':        np.std(null_accuracies)  if null_accuracies else np.nan,
        'null_f1s':                 null_f1s,
        'mean_null_f1':             np.mean(null_f1s)        if null_f1s        else np.nan,
        'std_null_f1':              np.std(null_f1s)         if null_f1s        else np.nan,
        'null_aurocs':              null_aurocs,
        'mean_null_auroc':          np.nanmean(null_aurocs)  if null_aurocs     else np.nan,
        'std_null_auroc':           np.nanstd(null_aurocs)   if null_aurocs     else np.nan,
        'n_permutations_completed': len(null_accuracies),
        'n_permutations_rejected':  n_rejected,
        'null_method':              null_method,
    }


def generate_ccgp_null_shuffle(
    subclass_pts_dict,
    fold_A_train, fold_A_test,
    fold_B_train, fold_B_test,
    neural_data,
    n_permutations=500,
    n_decoder_iterations=20,
    min_train_per_class=2,
    random_seed=None,
    C=1.0,
    n_neurons_subsample=None,
    verbose=True,
):
    """
    Interaction-level label shuffle null for CCGP.
    Kept for reference but condition_neuron_shuffle is preferred.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    all_subclasses = list(set(
        fold_A_train[list(fold_A_train.keys())[0]] +
        fold_A_train[list(fold_A_train.keys())[1]] +
        fold_A_test[list(fold_A_test.keys())[0]]   +
        fold_A_test[list(fold_A_test.keys())[1]]
    ))

    null_accuracies, null_f1s, null_aurocs = [], [], []

    for perm_idx in range(n_permutations):
        if verbose and perm_idx % 50 == 0:
            print(f"  CCGP shuffle null: {perm_idx}/{n_permutations}")

        all_interactions = []
        for sc in all_subclasses:
            pts_df = subclass_pts_dict[sc]
            for int_id in pts_df['int'].unique():
                all_interactions.append({
                    'original_subclass': sc,
                    'pts': pts_df[pts_df['int'] == int_id].copy(),
                })

        shuffled_labels = [d['original_subclass'] for d in all_interactions]
        np.random.shuffle(shuffled_labels)

        next_int_id       = {sc: 0 for sc in all_subclasses}
        shuffled_pts_dict = {sc: [] for sc in all_subclasses}

        for i, interaction in enumerate(all_interactions):
            new_sc      = shuffled_labels[i]
            pts         = interaction['pts'].copy()
            pts['type'] = new_sc
            pts['int']  = next_int_id[new_sc]
            next_int_id[new_sc] += 1
            shuffled_pts_dict[new_sc].append(pts)

        valid = True
        shuffled_subclass_pts = {}
        for sc in all_subclasses:
            if shuffled_pts_dict[sc]:
                df = pd.concat(shuffled_pts_dict[sc], ignore_index=True)
            else:
                df = pd.DataFrame(columns=['start', 'end', 'int', 'type'])
            if df['int'].nunique() < 2:
                valid = False
                break
            shuffled_subclass_pts[sc] = df

        if not valid:
            continue

        res = ccgp_both_folds(
            subclass_pts_dict=shuffled_subclass_pts,
            fold_A_train=fold_A_train, fold_A_test=fold_A_test,
            fold_B_train=fold_B_train, fold_B_test=fold_B_test,
            neural_data=neural_data,
            n_iterations=n_decoder_iterations,
            min_train_per_class=min_train_per_class,
            random_seed=None,
            C=C,
            n_neurons_subsample=n_neurons_subsample,
        )

        if not np.isnan(res['mean_accuracy']):
            null_accuracies.append(res['mean_accuracy'])
            null_f1s.append(res['mean_f1'])
            null_aurocs.append(res['mean_auroc'])

    if verbose:
        print(f"  CCGP shuffle null done: {len(null_accuracies)} completed")

    return {
        'null_accuracies':          null_accuracies,
        'mean_null_accuracy':       np.mean(null_accuracies) if null_accuracies else np.nan,
        'std_null_accuracy':        np.std(null_accuracies)  if null_accuracies else np.nan,
        'null_f1s':                 null_f1s,
        'mean_null_f1':             np.mean(null_f1s)        if null_f1s        else np.nan,
        'std_null_f1':              np.std(null_f1s)         if null_f1s        else np.nan,
        'null_aurocs':              null_aurocs,
        'mean_null_auroc':          np.nanmean(null_aurocs)  if null_aurocs     else np.nan,
        'std_null_auroc':           np.nanstd(null_aurocs)   if null_aurocs     else np.nan,
        'n_permutations_completed': len(null_aurocs),
    }


def ccgp_decoding_loop(
    comparison_label,
    subclass_pts_dict,
    fold_A_train,
    fold_A_test,
    fold_B_train,
    fold_B_test,
    neural_data=None,
    min_train_per_class=2,
    n_iterations_true=100,
    n_iterations_null=20,
    n_permutations_null=1000,
    discontinuity_frames=None,
    random_seed=None,
    C=1.0,
    n_neurons_subsample=None,
    null_shift=False,
    null_shift_aware=False,
    null_shuffle=False,
    null_condition_shuffle=True,
):
    """
    Run CCGP decoding (both folds) plus null distributions.
    Mirrors Will's decoding_loop() in structure and return format.

    Default null is now condition_neuron_shuffle (Boyle et al.),
    which produces null distributions centered at ~0.5.
    """
    if neural_data is None:
        raise ValueError("neural_data must be provided")

    res_dict = {}

    for key, df in subclass_pts_dict.items():
        res_dict[f'num_pts_{key}'] = len(df)

    res_dict['subsampled_neurons']   = 'yes' if n_neurons_subsample is not None else 'no'
    res_dict['n_neurons_subsampled'] = n_neurons_subsample if n_neurons_subsample is not None else neural_data.shape[0]

    # --- True decoding ---
    true_results = ccgp_both_folds(
        subclass_pts_dict=subclass_pts_dict,
        fold_A_train=fold_A_train, fold_A_test=fold_A_test,
        fold_B_train=fold_B_train, fold_B_test=fold_B_test,
        neural_data=neural_data,
        n_iterations=n_iterations_true,
        min_train_per_class=min_train_per_class,
        random_seed=random_seed,
        C=C,
        n_neurons_subsample=n_neurons_subsample,
    )
    print(
        f"{comparison_label} CCGP true results: "
        f"acc={true_results['mean_accuracy']:.3f}±{true_results['std_accuracy']:.3f}  "
        f"auroc={true_results['mean_auroc']:.3f}±{true_results['std_auroc']:.3f}"
    )
    res_dict['true_decoding_results'] = true_results

    # --- Condition neuron shuffle null (Boyle et al.) ---
    if null_condition_shuffle:
        print("============")
        print(f"{comparison_label}: running condition neuron shuffle null")
        null_condition_res = generate_ccgp_null_condition_shuffle(
            subclass_pts_dict=subclass_pts_dict,
            fold_A_train=fold_A_train, fold_A_test=fold_A_test,
            fold_B_train=fold_B_train, fold_B_test=fold_B_test,
            neural_data=neural_data,
            n_permutations=n_permutations_null,
            n_decoder_iterations=n_iterations_null,
            min_train_per_class=min_train_per_class,
            random_seed=random_seed,
            C=C,
            n_neurons_subsample=n_neurons_subsample,
            verbose=True,
        )
        p_acc   = compute_p_value(true_results['mean_accuracy'], null_condition_res['null_accuracies'])
        p_f1    = compute_p_value(true_results['mean_f1'],       null_condition_res['null_f1s'])
        p_auroc = compute_p_value(true_results['mean_auroc'],    null_condition_res['null_aurocs'])
        print(
            f"{comparison_label} condition shuffle null: "
            f"mean_acc={null_condition_res['mean_null_accuracy']:.3f}  "
            f"p_acc={p_acc:.3f}  p_auroc={p_auroc:.3f}"
        )
        res_dict['null_condition_shuffle_results']    = null_condition_res
        res_dict['p_val_condition_shuffle_accuracy']  = p_acc
        res_dict['p_val_condition_shuffle_f1']        = p_f1
        res_dict['p_val_condition_shuffle_auroc']     = p_auroc

    # --- Naive circular shift null ---
    if null_shift:
        print("============")
        print(f"{comparison_label}: running naive circular shift null")
        null_shift_res = generate_ccgp_null_shift(
            subclass_pts_dict=subclass_pts_dict,
            fold_A_train=fold_A_train, fold_A_test=fold_A_test,
            fold_B_train=fold_B_train, fold_B_test=fold_B_test,
            neural_data=neural_data,
            n_permutations=n_permutations_null,
            n_decoder_iterations=n_iterations_null,
            null_method='circular',
            discontinuity_frames=discontinuity_frames,
            min_train_per_class=min_train_per_class,
            random_seed=random_seed,
            C=C,
            n_neurons_subsample=n_neurons_subsample,
            verbose=True,
        )
        p_acc   = compute_p_value(true_results['mean_accuracy'], null_shift_res['null_accuracies'])
        p_f1    = compute_p_value(true_results['mean_f1'],       null_shift_res['null_f1s'])
        p_auroc = compute_p_value(true_results['mean_auroc'],    null_shift_res['null_aurocs'])
        print(
            f"{comparison_label} naive shift null: "
            f"mean_acc={null_shift_res['mean_null_accuracy']:.3f}  "
            f"p_acc={p_acc:.3f}  p_auroc={p_auroc:.3f}"
        )
        res_dict['null_shift_results']   = null_shift_res
        res_dict['p_val_shift_accuracy'] = p_acc
        res_dict['p_val_shift_f1']       = p_f1
        res_dict['p_val_shift_auroc']    = p_auroc

    # --- Discontinuity-aware circular shift null ---
    if null_shift_aware:
        print("============")
        print(f"{comparison_label}: running discontinuity-aware circular shift null")
        null_shift_aware_res = generate_ccgp_null_shift(
            subclass_pts_dict=subclass_pts_dict,
            fold_A_train=fold_A_train, fold_A_test=fold_A_test,
            fold_B_train=fold_B_train, fold_B_test=fold_B_test,
            neural_data=neural_data,
            n_permutations=n_permutations_null,
            n_decoder_iterations=n_iterations_null,
            null_method='circular_aware',
            discontinuity_frames=discontinuity_frames,
            min_train_per_class=min_train_per_class,
            random_seed=random_seed,
            C=C,
            n_neurons_subsample=n_neurons_subsample,
            verbose=True,
        )
        p_acc   = compute_p_value(true_results['mean_accuracy'], null_shift_aware_res['null_accuracies'])
        p_f1    = compute_p_value(true_results['mean_f1'],       null_shift_aware_res['null_f1s'])
        p_auroc = compute_p_value(true_results['mean_auroc'],    null_shift_aware_res['null_aurocs'])
        print(
            f"{comparison_label} aware shift null: "
            f"mean_acc={null_shift_aware_res['mean_null_accuracy']:.3f}  "
            f"p_acc={p_acc:.3f}  p_auroc={p_auroc:.3f}"
        )
        res_dict['null_shift_aware_results']    = null_shift_aware_res
        res_dict['p_val_shift_aware_accuracy']  = p_acc
        res_dict['p_val_shift_aware_f1']        = p_f1
        res_dict['p_val_shift_aware_auroc']     = p_auroc

    # --- Label shuffle null ---
    if null_shuffle:
        print("============")
        print(f"{comparison_label}: running label shuffle null")
        null_shuffle_res = generate_ccgp_null_shuffle(
            subclass_pts_dict=subclass_pts_dict,
            fold_A_train=fold_A_train, fold_A_test=fold_A_test,
            fold_B_train=fold_B_train, fold_B_test=fold_B_test,
            neural_data=neural_data,
            n_permutations=n_permutations_null,
            n_decoder_iterations=n_iterations_null,
            min_train_per_class=min_train_per_class,
            random_seed=random_seed,
            C=C,
            n_neurons_subsample=n_neurons_subsample,
            verbose=True,
        )
        p_acc   = compute_p_value(true_results['mean_accuracy'], null_shuffle_res['null_accuracies'])
        p_f1    = compute_p_value(true_results['mean_f1'],       null_shuffle_res['null_f1s'])
        p_auroc = compute_p_value(true_results['mean_auroc'],    null_shuffle_res['null_aurocs'])
        print(
            f"{comparison_label} label shuffle null: "
            f"mean_acc={null_shuffle_res['mean_null_accuracy']:.3f}  "
            f"p_acc={p_acc:.3f}  p_auroc={p_auroc:.3f}"
        )
        res_dict['null_shuffle_results']    = null_shuffle_res
        res_dict['p_val_shuffle_accuracy']  = p_acc
        res_dict['p_val_shuffle_f1']        = p_f1
        res_dict['p_val_shuffle_auroc']     = p_auroc

    return res_dict