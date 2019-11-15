# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
'''

from __future__ import division
import numpy
from scipy.fftpack import dct

from delta.data.feat.python_speech_features import sigproc

# numpy.log() is not recognized by pylint
# pylint: disable=assignment-from-no-return

# pylint: disable=unsupported-assignment-operation
# pylint: disable=unused-argument
# pylint: disable=unused-variable
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments


def mfcc(signal,
         samplerate=16000,
         winlen=0.025,
         winstep=0.01,
         numcep=13,
         nfilt=26,
         nfft=512,
         lowfreq=0,
         highfreq=None,
         preemph=0.97,
         ceplifter=22,
         append_energy=True,
         winfunc=lambda x: numpy.ones((x,))):
  """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
  feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                       lowfreq, highfreq, preemph, winfunc)
  feat = numpy.log(feat)
  feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
  feat = lifter(feat, ceplifter)
  if append_energy:
    feat[:, 0] = numpy.log(
        energy)  # replace first cepstral coefficient with log of frame energy
  return feat


def powerspec(signal,
              samplerate=16000,
              winlen=0.025,
              winstep=0.01,
              nfft=512,
              lowfreq=0,
              highfreq=None,
              preemph=0.97,
              winfunc=lambda x: numpy.ones((x,))):
  """Compute power spectorgram features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns:  first is a numpy array of size (NUMFRAMES by nfft) containing power spectrogram.
    """
  highfreq = highfreq or samplerate / 2
  signal = sigproc.preemphasis(signal, preemph)
  frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate,
                            winfunc)
  pspec = sigproc.powspec(frames, nfft)
  return pspec


def logpowerspec(ps, norm=1):
  """Compute the log power spectorgram features from powerspec.

    :param ps: power spectrum
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
  ps[ps <= 1e-30] = 1e-30
  lps = 10 * numpy.log10(ps)
  if norm:
    return lps - numpy.max(lps)
  return lps


def fbank(signal,
          samplerate=16000,
          winlen=0.025,
          winstep=0.01,
          nfilt=26,
          nfft=512,
          lowfreq=0,
          highfreq=None,
          preemph=0.97,
          winfunc=lambda x: numpy.ones((x,))):
  """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
  pspec = powerspec(signal, samplerate, winlen, winstep, nfft, lowfreq,
                    highfreq, preemph, winfunc)
  energy = numpy.sum(pspec, 1)  # this stores the total energy in each frame
  energy = numpy.where(energy == 0,
                       numpy.finfo(float).eps,
                       energy)  # if energy is zero, we get problems with log

  fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
  feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
  feat = numpy.where(feat == 0,
                     numpy.finfo(float).eps,
                     feat)  # if feat is zero, we get problems with log

  return feat, energy


def logfbank(signal,
             samplerate=16000,
             winlen=0.025,
             winstep=0.01,
             nfilt=26,
             nfft=512,
             lowfreq=0,
             highfreq=None,
             preemph=0.97,
             winfunc=lambda x: numpy.ones((x,))):
  """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
  feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                       lowfreq, highfreq, preemph, winfunc)
  return numpy.log(feat)


def logfbank_from_powspec(pspec,
                          samplerate=16000,
                          nfilt=26,
                          nfft=512,
                          lowfreq=0,
                          highfreq=None):
  ''' Log fbank from power spectrum. '''
  energy = numpy.sum(pspec, 1)  # this stores the total energy in each frame
  energy = numpy.where(energy == 0,
                       numpy.finfo(float).eps,
                       energy)  # if energy is zero, we get problems with log

  fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
  feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
  feat = numpy.where(feat == 0,
                     numpy.finfo(float).eps,
                     feat)  # if feat is zero, we get problems with log

  return numpy.log(feat)


def ssc(signal,
        samplerate=16000,
        winlen=0.025,
        winstep=0.01,
        nfilt=26,
        nfft=512,
        lowfreq=0,
        highfreq=None,
        preemph=0.97,
        winfunc=lambda x: numpy.ones((x,))):
  """Compute Spectral Subband Centroid features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
  highfreq = highfreq or samplerate / 2
  signal = sigproc.preemphasis(signal, preemph)
  frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate,
                            winfunc)
  pspec = sigproc.powspec(frames, nfft)
  pspec = numpy.where(pspec == 0,
                      numpy.finfo(float).eps,
                      pspec)  # if things are all zeros we get problems

  fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
  feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
  R = numpy.tile(
      numpy.linspace(1, samplerate / 2, numpy.size(pspec, 1)),
      (numpy.size(pspec, 0), 1))

  return numpy.dot(pspec * R, fb.T) / feat


def hz2mel(hz):
  """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
  return 2595 * numpy.log10(1 + hz / 700.)


def mel2hz(mel):
  """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
  return 700 * (10**(mel / 2595.0) - 1)


def get_filterbanks(nfilt=20,
                    nfft=512,
                    samplerate=16000,
                    lowfreq=0,
                    highfreq=None):
  """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
  highfreq = highfreq or samplerate / 2
  assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

  # compute points evenly spaced in mels
  lowmel = hz2mel(lowfreq)
  highmel = hz2mel(highfreq)
  melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
  # our points are in Hz, but we use fft bins, so we have to convert
  #  from Hz to fft bin number
  bins = numpy.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

  fbanks = numpy.zeros([nfilt, nfft // 2 + 1])
  for j in range(0, nfilt):
    for i in range(int(bins[j]), int(bins[j + 1])):
      fbanks[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
    for i in range(int(bins[j + 1]), int(bins[j + 2])):
      fbanks[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
  return fbanks


def lifter(cepstra, L=22):
  """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
  if L > 0:
    nframes, ncoeff = numpy.shape(cepstra)
    n = numpy.arange(ncoeff)
    lift = 1 + (L / 2.) * numpy.sin(numpy.pi * n / L)
    return lift * cepstra
  # values of L <= 0, do nothing
  return cepstra


def delta(feat, N):
  """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
  if N < 1:
    raise ValueError('N must be an integer >= 1')
  NUMFRAMES = len(feat)
  denominator = 2 * sum([i**2 for i in range(1, N + 1)])
  delta_feat = numpy.empty_like(feat)
  padded = numpy.pad(
      feat, ((N, N), (0, 0)), mode='edge')  # padded version of feat
  for t in range(NUMFRAMES):
    delta_feat[t] = numpy.dot(
        numpy.arange(-N, N + 1),
        padded[t:t + 2 * N +
               1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
  return delta_feat
