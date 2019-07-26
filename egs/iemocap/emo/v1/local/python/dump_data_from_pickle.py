#from
#https://github.com/zh794390558/IEMOCAP-Emotion-Detection

import os
import sys
import random
import numpy as np
import pickle
import copy
import wave
from scipy.io import wavfile

from features import *
from helper import *
import kaldiio

''' dump wav, text, label from pkl '''

emotions_used = np.array(['ang', 'neu', 'sad', 'exc', 'hap'])
root_path = os.path.realpath(os.getcwd())
data_path = os.path.join(root_path, 'data')
sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04']
framerate = 16000
dump_dir= os.path.join(data_path, 'dump')

def save_wav(data, filename, rate=framerate):
    assert data.dtype == np.int16, data.dtype
    wavfile.write(filename, rate, data)

def save_text(data, filename):
    with open(filename, 'w') as f:
        f.write(data)

def save_label(data, filename):
    with open(filename, 'w') as f:
        f.write(data)

with open(os.path.join(data_path,'data_collected.pickle'), 'rb') as handle:
    datas = pickle.load(handle)
    # data : dict_keys(['start', 'end', 'id', 'v', 'a', 'd', 'emotion', 'emo_evo', 'signal', 'transcription', 'mocap_hand', 'mocap_rot', 'mocap_head'])
    for data in datas:
        # Ses01F_impro01_F000  Excuse me.  neu
        key = data['id']
        samples = data['signal'] # int16 
        text = data['transcription']
        label = data['emotion']
        if label in emotions_used:
            if label == 'exc':
                label = 'hap'

            if key[0:5] in sessions:
                dirpath = os.path.join(dump_dir, 'train', label, key)
            else:
                dirpath = os.path.join(dump_dir, 'eval', label, key)

            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            filepath = os.path.join(dirpath, key) + '.wav'
            save_wav(np.array(samples, dtype=np.int16), filepath)
            filepath = os.path.join(dirpath, key) + '.txt'
            save_text(text, filepath)
            filepath = os.path.join(dirpath, key) + '.label'
            save_label(label, filepath)
