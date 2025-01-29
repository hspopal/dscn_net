#!/bin/bash

export BIDS_DIR="/data/neuron/NET/hpopal"
cd ${BIDS_DIR}

subj_list=( 011 012 014 015 016 017 018 019 021 022 
            023 025 026 029 030 032 033 034 035 036 
            037 038 039 040 041 042 046 047 048 049 
            050 052 056 058 059 062 064 068 070 072 
            073 075 077 078 079 081 083 084 086 087 
            090 091 094 095 096 098 099 101 102 103 
            104 105 107 108 109 )





##########################################################################
# Set up BIDS
##########################################################################

# Create symbolic links for the dicoms in BIDS format
for subj in "${subj_list[@]}"; do
    mkdir sourcedata/${subj}
    ln -s /data/neuron/TRW/original/${subj}/2* sourcedata/${subj}/
done


# Convert dicoms to niftis, using Heudiconv

# First we will create the heuristic.py file that will be used to pull dicoms of 
# the same type of scans together (e.g. anat, func, fmap)
singularity exec \
    --bind /data/neuron/TRW:/base \
    /software/neuron/Containers/heudiconv_latest.sif \
    heudiconv \
    -d /base/original/RED_TRW_{subject}/*/*/*.dcm \
    -o /base/reprocessed \
    -f reproin \
    -s 001 \
    -c none \
    --overwrite 

# Manually edit the heuristics.py file located in .heudiconv/001/info/

# Use heudiconv to create niftis
for subj in "${subj_list[@]}"; do
    sh code/preprocessing_TRW-bswift2.sh -s ${subj} -n
done


# Run fmriprep on bswift2
for subj in "${subj_list[@]}"; do
    sh code/preprocessing_TRW-bswift2.sh -s ${subj} -f
done


# Transfer preprocessed data back to lab server
for subj in "${subj_list[@]}"; do
    sh code/preprocessing_TRW-bswift2.sh -s ${subj} -t
done



# MRIQC
singularity run --cleanenv \
    -B /data/neuron/TRW/reprocessed:/base \
    /software/neuron/Containers/mriqc-23.1.1.sif \
    /base /base/derivatives/mriqc \
    participant --participant-label REDTRW001 \
    -w /data/neuron/TRW/reprocessed/archive/work
    
ssh ${uname}@bswift2-login.umd.edu "sbatch --export=uname="$uname",subID="$subID" --job-name=mriqc_"$subID" --mail-user="${uname}"@umd.edu --output="$bswift_dir"/derivatives/log/mriqc_sub-${proj_abr}"$subID".log ${bswift_dir}/code/mriqc_TRW-bswift2.sh"
ssh hpopal@bswift2-login.umd.edu "sbatch --export=uname=hpopal,subID="001" --job-name=mriqc_"001" --mail-user=hpopal@umd.edu --output=/data/software-research/hpopal/TRW/derivatives/log/mriqc_sub-REDTRW001.log /data/software-research/hpopal/TRW/code/mriqc_TRW-bswift2.sh"


