from moviepy import editor
import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.io import wavfile
from scipy import misc


SAMPLES_PER_VIDEO = 3
DATA_PATH = '/media/bonilla/HDD_2TB_basura/databases'
DATABASE = 'AVE_Dataset/AVE'
RESULT = 'AVE_Dataset/Processed'
DB_PATH = os.path.join(DATA_PATH, DATABASE)
OUTPUT_PATH = os.path.join(DATA_PATH, RESULT)
videos = glob.glob(os.path.join(DB_PATH, '*'))


def mel_2_audio(mel, sr, n_fft=2048, hop_length=512):
    mel = librosa.db_to_power(mel)
    audio = librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    audio = normalize([audio], norm="max")
    audio = np.clip(audio, -1, 1)
    return audio.flatten()


def audio_2_mel(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    if len(audio.shape) == 2 and audio.shape[-1] == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2
    S = librosa.feature.melspectrogram(np.asfortranarray(audio), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


for video_path in videos:
    print(video_path)
    video = editor.VideoFileClip(video_path)
    audio = video.audio
    duration = video.duration
    sr = audio.fps
    audio = audio.to_soundarray()
    sample = duration / SAMPLES_PER_VIDEO
    for s in range(SAMPLES_PER_VIDEO):
        result_size = 512
        hop_len = int(442856/result_size)
        frame_RGB = video.get_frame(int(sample * s))
        mel = audio_2_mel(audio, sr, n_fft=2048, hop_length=hop_len, n_mels=result_size)

        mean = mel.mean()
        std = mel.std()

        print(mel.shape, mean, std)

        plt.figure(0)
        plt.imshow(mel)

        org_size = mel.shape

        frame_RGB = misc.imresize(frame_RGB, (256, 256))
        mel = misc.imresize(mel, (512, 512), interp='nearest')
        mel = misc.imresize(mel, (org_size[0], org_size[1])).astype('float32')
        mel_o = mel.copy()
        #
        mean_res = mel.mean()
        mel -= mean_res
        std_res = mel.std()
        mel /= std_res
        #
        mel *= 20.  # std
        mel += -50.  # mean

        audio_rec = mel_2_audio(mel, sr, n_fft=2048, hop_length=hop_len)
        wavfile.write('original.wav', rate=sr, data=audio)
        wavfile.write('result.wav', rate=sr, data=audio_rec)

        plt.figure(1)
        plt.imshow(frame_RGB)
        plt.figure(2)
        plt.imshow(mel)
        plt.show()
