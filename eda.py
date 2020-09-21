import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import sys

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle("Time Series", size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

def plot_logmel(logmel):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('log Mel spectrogram', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(logmel.keys())[i])
            axes[x, y].imshow(list(logmel.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

def envelop(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def get_log_mel_spectrogram_vector(y,
                                   n_mels=64,
                                   frames=5,
                                   n_fft=1024,
                                   hop_length=512,
                                   power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    #y, sr = librosa.load(file_name, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array

df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)


#down sampling and nomalization， 下采样，这里首先降低采样率，从44100降到16000， 然后使用envelop函数过滤
#阀值0.005一下的值
if len(os.listdir('refinement')) == 0:
    for f in tqdm(df.fname):
        print(f)
        signal, rate =librosa.load('wavfiles_/'+f, sr=16000)
        mask = envelop(signal, rate, 0.005)
        wavfile.write(filename='refinement/'+f, rate=rate, data=signal[mask])

for f in df.index:
    rate, signal = wavfile.read('wavfiles_/'+f)
    df.at[f, 'olength'] = signal.shape[0]/rate
    y, sr = librosa.load('refinement/'+f, sr=None)
    df.at[f, 'rlength'] = y.shape[0]/sr

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['rlength'].mean()

# fig, ax = plt.subplots()
# ax.set_title('class distribution', y=1.08)
# ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
# ax.axis('equal')
# plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}
logmels = {}

for c in classes:
    wav_file = df[df.label == c]
    wav_name = wav_file.iloc[0, 0]

    signal, rate = librosa.load('refinement/'+wav_name, sr=44100)
    mask = envelop(signal, rate, 0.005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel
    logmels[c] = get_log_mel_spectrogram_vector(signal,
                                                n_mels=64,
                                                frames=5,
                                                n_fft=1104,
                                                hop_length=552,
                                                power=2.0)

plot_signals(signals)
plt.show()
#
plot_fft(fft)
plt.show()
# #
plot_fbank(fbank)
plt.show()
# #
plot_mfccs(mfccs)
plt.show()

plot_logmel(logmels)
plt.show()


