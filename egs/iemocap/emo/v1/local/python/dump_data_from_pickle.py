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

from dump_all_data import gen_all_data
from helper import *
import kaldiio

''' dump wav, text, label from pkl '''

emotions_used = np.array(['ang', 'neu', 'sad', 'hap'])
root_path = os.path.realpath(os.getcwd())
data_path = os.path.join(root_path, 'data')
sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04']
framerate = 16000
dump_dir= os.path.join(data_path, 'dump')
all_dir = os.path.join(dump_dir, 'all')

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
    for data in datas:
        # Ses01F_impro01_F000  Excuse me.  neu
        key = data['id']
        samples = data['signal'] # int16 
        text = data['transcription']
        label = data['emotion']

        if label in emotions_used:
            if key[7:12] == 'impro':
                dirpath = os.path.join(dump_dir, 'impro')
            else:
                dirpath = os.path.join(dump_dir, 'script')

            if key[0:5] in sessions:
                dirpath = os.path.join(dirpath, 'train', label, key)
            else:
                dirpath = os.path.join(dirpath, 'eval', label, key)

            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

            filepath = os.path.join(dirpath, key) + '.wav'
            save_wav(np.array(samples, dtype=np.int16), filepath)

            filepath = os.path.join(dirpath, key) + '.txt'
            save_text(text, filepath)

            filepath = os.path.join(dirpath, key) + '.label'
            save_label(label, filepath)

gen_all_data(all_dir, dump_dir)
