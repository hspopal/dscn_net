#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:57:47 2025

@author: hpopal






Whisper
https://github.com/openai/whisper

"""


import os
import pandas as pd

#import moviepy.editor as mp
from moviepy.editor import *
import whisper

import argparse


beta = False
clip_start = False
clip_end = False

if beta:
    proj_dir = '/Users/hpopal/Google Drive/My Drive/dscn_lab/projects/net/'
    os.chdir(proj_dir)
    input_path = proj_dir + 'derivatives/task-naturalistic/stimuli/grapes_hd.avi'
    outp_dir = proj_dir + 'derivatives/video_analysis/'
    clip_name=os.path.splitext(str(input_path))[0].split('/')[-1]
else:

    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Add input file and output path argument
    parser.add_argument("-i", "--Input", help = "Input video file")
    parser.add_argument("-o", "--Output", help = "Output path")
    
    # Add optional time segment arguments
    parser.add_argument("-s", "--Start_time", help = "Video start time")
    parser.add_argument("-e", "--End_time", help = "Video end time")
    
    
    # Read arguments from command line
    args = parser.parse_args()
    

    if args.Start_time:
        clip_start = args.Start_time
        
    if args.End_time:
        clip_end = args.End_time
    
    # Get absolute path of input file
    print(args.Input)
    input_path = os.path.abspath(args.Input)


    # Set paths
    outp_dir = os.path.abspath(args.Output) + '/'

    
    clip_name=os.path.splitext(str(args.Input))[0].split('/')[-1]
    print(clip_name)
    if clip_start and clip_end:    
        clip_name = clip_name+'_'+str(clip_start)+'-'+str(clip_end)
    

video_clip = VideoFileClip(input_path)


# Extract the audio  from the video
if clip_start and clip_end:
    audio_clip = video_clip.subclip(clip_start, clip_end)
    audio_path = clip_name+'_'+str(clip_start)+'-'+str(clip_end)+'.wav'
else:
    audio_clip = video_clip.audio
    audio_path = clip_name+'.wav'


print(audio_path)

audio_clip.write_audiofile(outp_dir+'audio_files/'+audio_path, codec='pcm_s16le')



# Transcribe
model = whisper.load_model("small")
result = model.transcribe(outp_dir+'audio_files/'+audio_path, word_timestamps=True)

# Export transcription
if clip_start and clip_end:
    trans_file_name = outp_dir+'transcriptions/'+clip_name+'_transcript_'+str(clip_start)+'-'+str(clip_end)+'.txt'
else:
    trans_file_name=outp_dir+'transcriptions/'+clip_name+'_transcript.txt'

with open(trans_file_name, 'w') as f:
    f.write(result['text'])
    

# Get time stamps of sentences

transcript_ts = pd.DataFrame(columns=['Text','Start_time','End_time'])

for idx in range(len(result['segments'])):
    transcript_ts.loc[idx, 'Text'] = result['segments'][idx]['text']
    transcript_ts.loc[idx, 'Start_time'] = result['segments'][idx]['start']
    transcript_ts.loc[idx, 'End_time'] = result['segments'][idx]['end']
    
# Export
if clip_start and clip_end:
    time_file_name = clip_name+'_timestamps_'+str(clip_start)+'-'+str(clip_end)+'.csv'
else:
    time_file_name = clip_name+'_timestamps.csv'

transcript_ts.to_csv(outp_dir+'transcriptions/'+time_file_name)





##########################################################################

# Sentiment Analysis

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download corpus and other relevant data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# Define function to preprocess text
def preprocess(text):
    # Lowercase and then tokenize (split into words) the text
    tokens = word_tokenize(text.lower())

    # Remove stop words (common words with minimal content - mostly grammatical words)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens (reduce words to base lemmas - i.e., form found in dictionary)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


# Set analyzer model parameters
analyzer = SentimentIntensityAnalyzer()
def polarity(text):
    pol = analyzer.polarity_scores(text)
    return pol



# Set relevant features
sentiment_feats = ['neg','neu','pos']

# Analyze text
for i_row in range(len(transcript_ts)):
    cleaned_text = preprocess(transcript_ts.loc[i_row,'Text'])
    transcript_ts.loc[i_row,'Text_clean'] = cleaned_text

    temp_sent = polarity(cleaned_text)
    for feat in sentiment_feats:
        transcript_ts.loc[i_row,feat] = temp_sent[feat]





##########################################################################

# Export Annotations
if clip_start and clip_end:
    time_file_name = clip_name+'_annotations_'+str(clip_start)+'-'+str(clip_end)+'.csv'
else:
    time_file_name = clip_name+'_annotations.csv'

transcript_ts.to_csv(outp_dir+'annotations/'+time_file_name)





##########################################################################

# Facial features
# currently just emotions

from feat import Detector


# Import video stimuli timing
stimuli_info = pd.read_csv('derivatives/task-naturalistic/stimuli/video_timing.csv')
video_onset = stimuli_info[stimuli_info['Video'] == clip_name]['Onset'].iloc[0]


detector = Detector(face_model='retinaface', landmark_model='mobilefacenet', 
                    au_model='svm', emotion_model='resmasknet', 
                    facepose_model='img2pose')

fps = video_clip.fps
video_feat = detector.detect_video(input_path, skip_frames=fps)  # one per TR

video_feat = video_feat.reset_index(drop=True)

# Capture emotion data
emotions_df = video_feat.emotions
emotions_df['frame'] = video_feat['frame'].to_list()
emotions_df = emotions_df.groupby('frame').mean()
emotions_df = emotions_df.reset_index(drop=True)
emotions_df.index.name = 'TR'
emotions_df['Onset'] = 0.0
emotions_df.loc[0,'Onset'] = video_onset

for idx in emotions_df.index:
    emotions_df.loc[idx,'Onset'] = video_onset + idx

# Add onset time


if clip_start and clip_end:
    time_file_name = clip_name+'_annotations-feat_'+str(clip_start)+'-'+str(clip_end)+'.csv'
else:
    time_file_name = clip_name+'_annotations-feat.csv'

emotions_df.to_csv(outp_dir+'annotations/'+time_file_name)




