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




