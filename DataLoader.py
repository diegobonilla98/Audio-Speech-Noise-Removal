import cv2
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.io import wavfile
import soundfile
from scipy import misc
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DataLoader:
    def __init__(self, root_path, reload_possible_audios=False):
        self.AUDIO_SEC_LENGTH = 10
        self.RESULT_SIZE = 512
        self.HOP_LEN = int(442856 / self.RESULT_SIZE)
        if root_path is not None:
            self.ROOT_PATH = root_path
            self.PEOPLE = os.listdir(os.path.join(self.ROOT_PATH))

            if reload_possible_audios:
                self._enough_length_audios()
            self.AUDIOS = np.array(open('train_available_data.txt', 'r').read().split('\n')[:-1])
            self.NUM_AUDIOS = len(self.AUDIOS)
            self.NOISE = np.array(glob.glob(os.path.join(*os.path.split(self.ROOT_PATH)[:-1], 'noise', '*')))
            self.NUM_NOISES = len(self.NOISE)

    def _treat_audio(self, data, sr, new_sr=44100, is_noise=False):
        if len(data.shape) == 2 and data.shape[-1] == 2:
            data = (data[:, 0] + data[:, 1]) / 2
        data = data.flatten()
        if is_noise:
            seq_len = data.shape[0]
            start = np.random.randint(low=0, high=seq_len - sr * self.AUDIO_SEC_LENGTH)
            data = data[start: start + sr * self.AUDIO_SEC_LENGTH]
        else:
            data = data[:sr * self.AUDIO_SEC_LENGTH]
        if sr != new_sr:
            return librosa.resample(data, sr, new_sr)
        else:
            return data

    @staticmethod
    def _mel_2_audio(mel, sr=44100, n_fft=2048, hop_length=512, do_power=True):
        if do_power:
            mel = librosa.db_to_power(mel)
        audio = librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_fft=n_fft, hop_length=hop_length)
        audio = normalize([audio], norm="max")
        audio = np.clip(audio, -1, 1)
        return audio.flatten()

    @staticmethod
    def _audio_2_mel(audio, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
        S = librosa.feature.melspectrogram(np.asfortranarray(audio), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB

    def _enough_length_audios(self):
        file = open('train_available_data.txt', 'w')
        for person in self.PEOPLE:
            texts = list(map(lambda x: os.path.join(self.ROOT_PATH, person, x), os.listdir(os.path.join(self.ROOT_PATH, person))))
            for t in texts:
                audios = glob.glob(os.path.join(t, '*.flac'))
                for a in audios:
                    y, sr = soundfile.read(a, always_2d=True)
                    seconds = y.shape[0] / sr
                    if seconds >= self.AUDIO_SEC_LENGTH:
                        file.write(a)
                        file.write('\n')
        file.close()

    def sample_results(self, batch_size, save_path, generator_model, num_epoch):
        try:
            os.mkdir(os.path.join(save_path, str(num_epoch)))
        except FileExistsError:
            pass
        x, y = self.load_batch(batch_size)
        res = generator_model.predict(x)

        data_mel = np.uint8(res[0, :, :, 0] * 255)
        cv2.imwrite(os.path.join(save_path, str(num_epoch), f'predicted_mask.jpg'), cv2.resize(data_mel, (self.RESULT_SIZE, self.RESULT_SIZE)))

        data_mel_ = np.uint8(res[0, :, :, 1] * 255)
        cv2.imwrite(os.path.join(save_path, str(num_epoch), f'predicted_voice_mask.jpg'), cv2.resize(data_mel_, (self.RESULT_SIZE, self.RESULT_SIZE)))

        # m = cv2.adaptiveThreshold(data_mel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)
        # _, m = cv2.threshold(data_mel, 90, 255, cv2.THRESH_BINARY)
        # _, m = cv2.threshold(data_mel, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data_mel = cv2.morphologyEx(data_mel, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        cv2.imwrite(os.path.join(save_path, str(num_epoch), f'predicted_mask_thresh.jpg'), cv2.resize(data_mel, (self.RESULT_SIZE, self.RESULT_SIZE)))

        r = (np.clip(x[0, :, :, 0] * (1 - data_mel / 255.), a_min=0, a_max=1.) - 1) * 80

        data_mel = np.uint8((r / 80 + 1) * 255)
        cv2.imwrite(os.path.join(save_path, str(num_epoch), f'reconstructed.jpg'), cv2.resize(data_mel, (self.RESULT_SIZE, self.RESULT_SIZE)))

        data_mel = np.uint8((x - np.min(x)) / (np.max(x) - np.min(x)) * 255)[0, :, :, 0]
        cv2.imwrite(os.path.join(save_path, str(num_epoch), f'input.jpg'), cv2.resize(data_mel, (self.RESULT_SIZE, self.RESULT_SIZE)))

        data_mel = np.uint8((y - np.min(y)) / (np.max(y) - np.min(y)) * 255)[0, :, :, 0]
        cv2.imwrite(os.path.join(save_path, str(num_epoch), f'real_mask.jpg'), cv2.resize(data_mel, (self.RESULT_SIZE, self.RESULT_SIZE)))

        x = (x[0, :, :, 0] - 1) * 80
        audio_rec = self._mel_2_audio(x, n_fft=2048, hop_length=self.HOP_LEN)
        wavfile.write(os.path.join(save_path, str(num_epoch), f'original.wav'), rate=44100, data=audio_rec)

        audio_rec = self._mel_2_audio(r, n_fft=2048, hop_length=self.HOP_LEN)
        wavfile.write(os.path.join(save_path, str(num_epoch), f'restored.wav'), rate=44100, data=audio_rec)

    def evaluate_audio(self, audio_path, model, save_path='.'):
        if audio_path[-3:] == 'mp3':
            data, sr = librosa.load(audio_path)
        else:
            data, sr = soundfile.read(audio_path, always_2d=True)
        data = self._treat_audio(data, sr)
        data_mel = self._audio_2_mel(data, n_fft=2048, hop_length=self.HOP_LEN, n_mels=self.RESULT_SIZE)
        data_mel = np.uint8((data_mel - np.min(data_mel)) / (np.max(data_mel) - np.min(data_mel)) * 255)
        data_mel = cv2.resize(data_mel, (self.RESULT_SIZE, self.RESULT_SIZE)).astype('float32') / 255.

        data_mel_in = data_mel.reshape((1, self.RESULT_SIZE, self.RESULT_SIZE, 1))

        result = model.predict(data_mel_in)[0, :, :, 0]
        r = (np.clip(data_mel * (1 - result), a_min=0, a_max=1.) - 1) * 80
        audio_rec = self._mel_2_audio(r, n_fft=2048, hop_length=self.HOP_LEN)
        wavfile.write(os.path.join(save_path, 'restored.wav'), rate=44100, data=audio_rec)

    def load_batch(self, batch_size):
        X = []
        y = []
        data_rand_idx = random.sample(range(self.NUM_AUDIOS), batch_size)
        noise_rand_idx = np.random.randint(low=0, high=self.NUM_NOISES, size=(batch_size,))
        audios_path = self.AUDIOS[data_rand_idx]
        noises_path = self.NOISE[noise_rand_idx]
        for a, n in zip(audios_path, noises_path):
            data, sr_d = soundfile.read(a, always_2d=True)
            data = self._treat_audio(data, sr_d)
            data_mel = self._audio_2_mel(data, n_fft=2048, hop_length=self.HOP_LEN, n_mels=self.RESULT_SIZE)
            data_mel = np.uint8((data_mel - np.min(data_mel)) / (np.max(data_mel) - np.min(data_mel)) * 255)
            data_mel = cv2.resize(data_mel, (self.RESULT_SIZE, self.RESULT_SIZE))

            if n[-3:] == 'mp3':
                noise, sr_n = librosa.load(n)
            else:
                noise, sr_n = soundfile.read(n, always_2d=True)
            noise = self._treat_audio(noise, sr_n, is_noise=True)
            mix_val = np.random.normal(0.3, 0.1)
            noise_combined = noise * mix_val + data * (1 - mix_val)
            data_noise = self._audio_2_mel(noise_combined, n_fft=2048, hop_length=self.HOP_LEN, n_mels=self.RESULT_SIZE)
            data_noise = np.uint8((data_noise - np.min(data_noise)) / (np.max(data_noise) - np.min(data_noise)) * 255)
            data_noise = cv2.resize(data_noise, (self.RESULT_SIZE, self.RESULT_SIZE))

            diff = cv2.absdiff(data_noise, data_mel)
            _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            X.append(np.expand_dims(data_noise.astype('float32') / 255., axis=-1))
            y.append(np.expand_dims(diff.astype('float32') / 255., axis=-1))

        return np.array(X), np.array(y)


# data_loader = DataLoader(root_path='/media/bonilla/HDD_2TB_basura/databases/LibriSpeech/train-clean-100')
# test = data_loader.load_batch(16)
