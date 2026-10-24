# sheeran_vca2_social_spatial
Companion repository for Sheeran et al. manuscript (vCA2 calcium imaging). Please contact Will Sheeran (william.sheeran@colorado.edu) and/or Zoe Donaldson (zoe.donaldson@colorado.edu) for questions.

See also **zenodo link** for accompanying processed data.

### Notebooks:
1. neural_behavioral_data_prep.ipynb : notebook that reshapes and aligns CNMF-E / Caiman outputs with behavioral data (outputs from Bento and ezTrack)
2. behavior_analysis.ipynb : notebook that computes metrics from imaging session behavioral data
3. imaging_session_behavior_summary.ipynb : notebook that compiles behavioral results across animals/sessions to perform summary statistics and plotting
4. single_cell_analysis.ipynb : notebook that defines single cell tuning profiles for a given imaging session's neuronal population
5. single_cell_summary.ipynb : notebook that compiles cell tuning results across animals/sessions to perform summary statistics and plotting
6. population_analysis.ipynb : notebook that performs SVM-based linear decoding and mahalanobis distance ratio analyses on a given imaging session's population activity
7. population_analysis_summary.ipynb : notebook that compiles standard linear decoding and mahalanobis distance ratio results across animals/sessions to perform summary statistics and plotting
8. CellReg_import.ipynb : notebook that interfaces CellReg outputs with single_cell_analysis outputs to perform FOV/tuning stability analyses
9. LMMs_for_decoding_performance.ipynb : notebook that runs linear mixed effects models on outputs from population_analysis.ipynb
10. ccgp_decoding.ipynb : notebook that performs SVM-based linear decoding of CCGP
11. geometry.ipynb : notebook that performs and compiles statistics on analyses of neural population geometry, including PCA, point-to-plane, parallelism, and outputs of CCGP decoding 
   

### .py files:
1. utils.py : general-purpose utility functions
2. xsesssion_utils.py : utility functions for cross-session comparisons/summary statistics
3. session_config.py : configuration file to assist with cross-session comparisons
4. single_cell.py : utility functions for defining single-cell tuning profiles and related analyses
5. decoding.py : utility functions for SVM-based linear decoding
6. decoding_CCGP.py : utility functions for SVM-based linear decoding of CCGP
