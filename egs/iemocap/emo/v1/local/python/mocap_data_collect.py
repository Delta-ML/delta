#from
#https://github.com/zh794390558/IEMOCAP-Emotion-Detection

import os
import sys
import tqdm
import numpy as np

import wave
import copy
import math
import pickle

from multiprocessing import Pool, Lock, cpu_count, Manager

from sklearn.preprocessing import label_binarize
from helper import *

''' collection data from corpus '''

iemocap_path = sys.argv[1]
root_path = os.path.realpath(os.getcwd())
data_path = os.path.join(root_path, 'data')
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
path_map = {'Ses01':'Session1', 'Ses02':'Session2','Ses03':'Session3','Ses04':'Session4','Ses05':'Session5'}
framerate = 16000

def get_mocap_rot(path_to_mocap_rot, filename, start,end):
    f = open(os.path.join(path_to_mocap_rot, filename), 'r').read()
    f = np.array(f.split('\n'))
    mocap_rot = []
    mocap_rot_avg = []
    f = f[2:]
    counter = 0
    for data in f:
        counter+=1
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        if(float(data2[1])>start and float(data2[1])<end):
            mocap_rot_avg.append(np.array(data2[2:]).astype(np.float))
            
    mocap_rot_avg = np.array_split(np.array(mocap_rot_avg), 200)
    for spl in mocap_rot_avg:
        if spl.size != 0:
            mocap_rot.append(np.mean(spl, axis=0))
    return np.array(mocap_rot)

def get_mocap_hand(path_to_mocap_hand, filename, start,end):
    f = open(os.path.join(path_to_mocap_hand, filename), 'r').read()
    f = np.array(f.split('\n'))
    mocap_hand = []
    mocap_hand_avg = []
    f = f[2:]
    counter = 0
    for data in f:
        counter+=1
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        if(float(data2[1])>start and float(data2[1])<end):
            mocap_hand_avg.append(np.array(data2[2:]).astype(np.float))
            
    mocap_hand_avg = np.array_split(np.array(mocap_hand_avg), 200)
    for spl in mocap_hand_avg:
        if spl.size != 0:
            mocap_hand.append(np.mean(spl, axis=0))
    return np.array(mocap_hand)

def get_mocap_head(path_to_mocap_head, filename, start,end):
    f = open(os.path.join(path_to_mocap_head, filename), 'r').read()
    f = np.array(f.split('\n'))
    mocap_head = []
    mocap_head_avg = []
    f = f[2:]
    counter = 0
    for data in f:
        counter+=1
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        if(float(data2[1])>start and float(data2[1])<end):
            mocap_head_avg.append(np.array(data2[2:]).astype(np.float))
            
    mocap_head_avg = np.array_split(np.array(mocap_head_avg), 200)
    for spl in mocap_head_avg:
        if spl.size != 0:
            mocap_head.append(np.mean(spl, axis=0))
    return np.array(mocap_head)

def collect(f):
    path_to_wav = os.path.join(iemocap_path, path_map[f[0:5]], 'dialog', 'wav')
    path_to_emotions = os.path.join(iemocap_path, path_map[f[0:5]], 'dialog', 'EmoEvaluation')
    path_to_transcriptions = os.path.join(iemocap_path, path_map[f[0:5]], 'dialog', 'transcriptions')
    path_to_mocap_hand = os.path.join(iemocap_path, path_map[f[0:5]], 'dialog', 'MOCAP_hand')
    path_to_mocap_rot = os.path.join(iemocap_path, path_map[f[0:5]], 'dialog', 'MOCAP_rotated')
    path_to_mocap_head = os.path.join(iemocap_path, path_map[f[0:5]], 'dialog', 'MOCAP_head')

    mocap_f = f
    if (f== 'Ses05M_script01_1b'):
        mocap_f = 'Ses05M_script01_1'

    wav = get_audio(path_to_wav, f + '.wav')
    transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
    emotions = get_emotions(path_to_emotions, f + '.txt')
    sample = split_wav(wav, emotions)

    for ie, e in enumerate(emotions):
        '''if 'F' in e['id']:
            e['signal'] = sample[ie]['left']
        else:
            e['signal'] = sample[ie]['right']'''

        e['signal'] = sample[ie]['left']
        e.pop("left", None)
        e.pop("right", None)
        e['transcription'] = transcriptions[e['id']]
        e['mocap_hand'] = get_mocap_hand(path_to_mocap_hand, mocap_f + '.txt', e['start'], e['end'])
        e['mocap_rot'] = get_mocap_rot(path_to_mocap_rot, mocap_f + '.txt', e['start'], e['end'])
        e['mocap_head'] = get_mocap_head(path_to_mocap_head, mocap_f + '.txt', e['start'], e['end'])
        lock.acquire()
        if e['id'] not in ids:
            data.append(e)
            ids[e['id']] = 1
        lock.release()

def read_iemocap_mocap():
    for session in sessions:
        path_to_wav = os.path.join(iemocap_path, session, 'dialog', 'wav')
        files2 = os.listdir(path_to_wav)

        files = []
        for f in files2:
            if f.endswith(".wav"):
                if 'perturb' not in f:
                    if f[0] == '.':
                        files.append(f[2:-4])
                    else:
                        files.append(f[:-4])
        print('Collect ', session)
        with Pool(cpu_num) as p:
            r = list(tqdm.tqdm(p.imap(collect, files), total=len(files)))
                        
    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]
    
cpu_num = cpu_count()
manager = Manager()
data = manager.list()
ids = manager.dict()
lock=Lock()

data = read_iemocap_mocap()

with open(os.path.join(data_path, 'data_collected.pickle'), 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

