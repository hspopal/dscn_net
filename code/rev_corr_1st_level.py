# -*- coding: utf-8 -*-
"""
Spyder Editor

First level reverse correlation analysis with nilearn
"""

import sys
import os
import pandas as pd

from nilearn.maskers import NiftiLabelsMasker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

##########################################################################
# Set up
##########################################################################

# Take script inputs
subj = 'sub-NET'+str(sys.argv[1])
task = 'net'

# For beta testings
#subj = 'sub-NET011'

# Define fmriprep template space
template = 'MNI152NLin6Asym'

# Set BIDS project directory
bids_dir = '/data/neuron/NET/hpopal/'
os.chdir(bids_dir)

# Set output directory
outp_dir = bids_dir + 'derivatives/reverse_correlation/subject_data/'


##########################################################################
# Set scan specific paramters
##########################################################################

tr = 1.25  # repetition time is 1 second
#n_scans = 241  # the acquisition comprises 128 scans
#frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
slice_time_ref = 0.5

##########################################################################
# Find subject specific data
##########################################################################

print('Starting 1st-level analysis for '+subj)

# Make participant-specific directory for output if it doesn't exist
if not os.path.exists(outp_dir):
    os.makedirs(outp_dir)
    
    
# Find QC data for participant
qc_data = pd.read_csv(os.path.join(bids_dir, 'derivatives', 'participants-qc.csv'))

# Filter for only participant QC data
subj_qc_data = qc_data[qc_data['participant_id'] == subj]
subj_qc_data = subj_qc_data[subj_qc_data['Run'] != 'T1']


# Grab subject's T1 as a mask to keep analysis in subject space
subj_t1 = bids_dir+'derivatives/fmriprep/'+subj+'/anat/'+subj+'_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'

# Set path to subject specific fmriprep output
fmri_run_data_dir = bids_dir+'derivatives/fmriprep/'+subj+'/func/'


# Define parcellation
cb_atlas = bids_dir + 'derivatives/rois/Nettekoven_2023/atl-NettekovenAsym32_space-MNI152NLin2009cSymC_dseg.nii'
cb_atlas_labels = pd.read_csv('derivatives/rois/Nettekoven_2023/atl-NettekovenAsym32.lut', sep=' ',
                             names=['Value', 'R', 'G', 'B', 'Label'])


# Instantiate the masker with label image and label values
masker = NiftiLabelsMasker(
    labels_img=cb_atlas,
    labels=cb_atlas_labels['Label'],
    mask_img=subj_t1,
    standardize="zscore_sample",
    detrend=True,
    high_pass=0.01,
    t_r=1.25, 
    strategy='mean')



##########################################################################
# Loop through funcitonal runs
##########################################################################

for n in range(len(subj_qc_data)):
    
    # Specify run number
    run_num = subj_qc_data['Run'].iloc[n][-1]
    
    # Find preprocessed functional run
    func_run = bids_dir + '/derivatives/fmriprep/'+subj+'/func/'+subj+'_task-'+task+'_run-'+run_num+'_space-'+template+'_desc-smoothAROMAnonaggr_bold.nii.gz'
    
    cb_data_ts = masker.fit_transform(func_run)

    cb_data_ts_df = pd.DataFrame(cb_data_ts, columns=cb_atlas_labels['Label'][1:])
    cb_data_ts_df.to_csv(outp_dir+subj+'_run-'+run_num+'_cb_atlas_ts.csv', index=False)




