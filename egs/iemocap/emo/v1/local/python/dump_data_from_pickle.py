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

emotions_used = np.array(['ang', 'neu', 'sad', 'hap'])
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
    for data in datas:
        # Ses01F_impro01_F000  Excuse me.  neu
        key = data['id']
        samples = data['signal'] # int16 
        text = data['transcription']
        label = data['emotion']
        all_dirpath = os.path.join(dump_dir, 'all')

        if label in emotions_used:
            if key[7:12] == 'impro':
                dirpath = os.path.join(dump_dir, 'impro')
            else:
                dirpath = os.path.join(dump_dir, 'script')

            if key[0:5] in sessions:
                dirpath = os.path.join(dirpath, 'train', label, key)
                all_dirpath = os.path.join(all_dirpath, 'train',label, key)
            else:
                dirpath = os.path.join(dirpath, 'eval', label, key)
                all_dirpath = os.path.join(all_dirpath, 'eval',label, key)

            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
                os.makedirs(all_dirpath)

            filepath = os.path.join(dirpath, key) + '.wav'
            save_wav(np.array(samples, dtype=np.int16), filepath)
            link_file = os.path.join(all_dirpath, key) + '.wav'
            os.symlink(filepath, link_file)

            filepath = os.path.join(dirpath, key) + '.txt'
            save_text(text, filepath)
            link_file = os.path.join(all_dirpath, key) + '.txt'
            os.symlink(filepath, link_file)

            filepath = os.path.join(dirpath, key) + '.label'
            save_label(label, filepath)
            link_file = os.path.join(all_dirpath, key) + '.label'
            os.symlink(filepath, link_file)
