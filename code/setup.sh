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


# Transfer fmriprep files to local computer for QC
for subj in "${subj_list[@]}"; do
    mkdir sub-NET${subj}
    mkdir sub-NET${subj}/figures
    scp -r neuron.umd.edu:/data/neuron/NET/fmriprep/sub-NET${subj}/figures sub-NET${subj}/
done

scp -r neuron.umd.edu:/data/neuron/NET/fmriprep/sub-NET\*.html ./



# Complete video transcriptions and annotations

video_list=( grapes_hd.avi wedding_hd.avi awkward_hd.avi 
             ballet_hd.avi penguin_hd.avi sharktank_hd.avi 
             basketball_hd.avi onion_hd.avi cooking_hd.avi 
             mars_hd.avi afv_hd.avi cleese_hd.avi 
             partly_cloudy_hd.avi the_present_hd.avi despicable_hd.avi )

for video in "${video_list[@]}"; do
    python code/video_transcription.py \
        -i derivatives/task-naturalistic/stimuli/${video} \
        -o derivatives/video_analysis/
done


python code/video_transcription.py \
        -i derivatives/task-naturalistic/stimuli/awkward_hd.avi \
        -o derivatives/video_analysis/

for subj in "${subj_list[@]}"; do
    python code/rev_corr_1st_level.py ${subj}
done

