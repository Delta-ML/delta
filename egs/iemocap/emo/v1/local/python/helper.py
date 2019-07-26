#from
#https://github.com/zh794390558/IEMOCAP-Emotion-Detection

import os
import csv
import wave
import sys
import numpy as np
import pandas as pd
import glob
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def split_wav(wav, emotions):
    (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav

    left = samples[0::nchannels]
    right = samples[1::nchannels]

    frames = []
    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']

        e['right'] = right[int(start * framerate):int(end * framerate)]
        e['left'] = left[int(start * framerate):int(end * framerate)]

        frames.append({'left': e['left'], 'right': e['right']})
    return frames


def get_field(data, key):
    return np.array([e[key] for e in data])

def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):

    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask


def convert_gt_from_array_to_list(gt_batch, gt_batch_mask=None):

    B, L = gt_batch.shape
    gt_batch = gt_batch.astype('int')
    gts = []
    for i in range(B):
        if gt_batch_mask is None:
            l = L
        else:
            l = int(gt_batch_mask[i, :].sum())
        gts.append(gt_batch[i, :l].tolist())
    return gts

def get_audio(path_to_wav, filename):
    wav = wave.open(os.path.join(path_to_wav, filename), mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    assert framerate == 16000
    content = wav.readframes(nframes)
    samples = np.fromstring(content, dtype=np.int16)
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples


def get_transcriptions(path_to_transcriptions, filename):
    f = open(os.path.join(path_to_transcriptions, filename), 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1+2:]
        transcription[ind_id] = ind_ts
    return transcription

def get_transcriptions_align(path_to_transcriptions, filename,
                             path_to_transcriptions_to_align):
    f = open(os.path.join(path_to_transcriptions, filename), 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    print("path_to_transcriptions:", path_to_transcriptions)
    print("path_to_transcriptions_to_align:", path_to_transcriptions_to_align)
    print("filename:", filename)
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1+2:]
        ind_ts = text_to_word_sequence(ind_ts,filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',
                                       lower=True,split=" ")
        align_t = []
        #print("ind_id_text:", g, flush=True)
        #print("ind_id:", ind_id, flush=True)
        align_file_name = path_to_transcriptions_to_align + filename[:-4] + "/" + str(ind_id) + ".wdseg"
        if os.path.exists(align_file_name):
            f_align = open(align_file_name, 'r').read()
            f_align = np.array(f_align.split('\n'))
            #print("f_align", f_align, flush=True)
            for i in range(2, len(f_align) - 3):
                # print(f'f_align{i}', f_align[i])
                w = f_align[i].split()[3].split("(")[0]
                w = text_to_word_sequence(w,filters='!"#$%&()*+,-./:;=>?@[\\]^`{|}~\t\n',
                                           lower=True,split=" ")[0]
                if w in ind_ts:
                    align_t.append({'word': w,
                                    'SFrm': f_align[i].split()[0],
                                    'Efrm': f_align[i].split()[1]})
            #print("align_t", align_t, flush=True)
            #print("w_list", ind_ts, flush=True)
            assert len(align_t) == len(ind_ts)
            transcription[ind_id] = {'ind_ts': ind_ts,
                                   'align_t': align_t}
    return transcription

def get_emotions(path_to_emotions, filename):
    f = open(os.path.join(path_to_emotions, filename), 'r').read()
    f = np.array(f.split('\n'))
    idx = f == ''
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i]+1:idx_n[i+1]]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                        head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])
        
        j = 1
        emos = []
        while g[j][0] == "C":
            head = g[j]
            start_idx = head.find("\t") + 1
            evoluator_emo = []
            idx = head.find(";", start_idx)
            while idx != -1:
                evoluator_emo.append(head[start_idx:idx].strip().lower()[:3])
                start_idx = idx + 1
                idx = head.find(";", start_idx)
            emos.append(evoluator_emo)
            j += 1

        emotion.append({'start': start_time,
                        'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo,
                        'emo_evo': emos})
    return emotion
