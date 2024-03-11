import random
import soundfile as sf
import numpy as np

from torch.autograd.variable import *
from torch.utils.data import Dataset
import glob



class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.near_path = self.args.tr_nearend_path
        self.far_path = self.args.tr_farend_path
        self.rir_path = self.args.tr_rir_path
        self.len = self.args.num_train

        self.far_list = sorted(glob.glob(self.far_path + '/*_lpb.wav'))[:-1000]
        self.near_list = sorted(glob.glob(self.near_path + '/*'))[:-1000]

        self.ser = self.args.ser
        self.speech_wav_len = args.speech_len * args.sample_rate

    def __len__(self):
        return self.len

    def mix2signal(self, sig1, sig2, snr):
        alpha = np.sqrt(
            (np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + 1.0e-8)) / 10.0 ** (snr / 10.0) + 1.0e-8)
        return alpha

    def __getitem__(self, idx):

        # far_end / echo
        far_path = random.choice(self.far_list)
        far_wav, _ = sf.read(far_path)
        echo_wav, _ = sf.read(far_path.replace('_lpb.wav', '_mic.wav'))
        random_idx = random.randint(1, len(far_wav))
        far_wav = np.roll(far_wav, random_idx)
        echo_wav = np.roll(echo_wav, random_idx)
        if len(far_wav) >= self.speech_wav_len:
            far_wav = far_wav[:self.speech_wav_len]
            echo_wav = echo_wav[:self.speech_wav_len]
        else:
            far_wav = np.pad(far_wav, (self.speech_wav_len-len(far_wav), 0), mode='constant', constant_values=0)
            echo_wav = np.pad(echo_wav, (self.speech_wav_len-len(echo_wav), 0), mode='constant', constant_values=0)

        # near_end
        near_wav, _ = sf.read(random.choice(self.near_list))
        near_wav = np.roll(near_wav, random.randint(1, len(near_wav)))
        if len(near_wav) >= self.speech_wav_len:
            near_wav = near_wav[:self.speech_wav_len]
        else:
            near_wav = np.pad(near_wav, (self.speech_wav_len-len(near_wav), 0), mode='constant', constant_values=0)

        random_for_DT_or_ST = random.random()
        if random_for_DT_or_ST < 0.25:
            # far-end single talk
            near_wav *= 0
            mix_wav = echo_wav
        elif random_for_DT_or_ST >= 0.25 and random_for_DT_or_ST < 0.5:
            # near-end single talk
            mix_wav = near_wav
        else:
            # double-talk
            Ser = random.randint(self.ser[0], self.ser[1])
            alpha = self.mix2signal(near_wav, echo_wav, Ser)
            echo_wav = alpha * echo_wav
            mix_wav = near_wav + echo_wav

        sample = (Variable(torch.FloatTensor(mix_wav.astype('float32'))),
                  Variable(torch.FloatTensor(far_wav.astype('float32'))),
                  Variable(torch.FloatTensor(echo_wav.astype('float32'))),
                  Variable(torch.FloatTensor(near_wav.astype('float32'))),
                  )

        return sample

class EvalDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.near_path = self.args.tr_nearend_path
        self.far_path = self.args.tr_farend_path
        self.rir_path = self.args.tr_rir_path
        self.len = self.args.num_eval

        self.far_list = sorted(glob.glob(self.far_path + '/*_lpb.wav'))[-1000: -500]
        self.near_list = sorted(glob.glob(self.near_path + '/*'))[-1000: -500]

        self.ser = range(self.args.ser[0], self.args.ser[1])
        self.speech_wav_len = args.speech_len * args.sample_rate

    def __len__(self):
        return self.len

    def mix2signal(self, sig1, sig2, snr):
        alpha = np.sqrt(
            (np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + 1.0e-8)) / 10.0 ** (snr / 10.0) + 1.0e-8)
        return alpha

    def __getitem__(self, idx):

        # far_end / echo
        far_path = self.far_list[idx % len(self.far_list)]
        far_wav, _ = sf.read(far_path)
        echo_wav, _ = sf.read(far_path.replace('_lpb.wav', '_mic.wav'))
        if len(far_wav) >= self.speech_wav_len:
            far_wav = far_wav[:self.speech_wav_len]
            echo_wav = echo_wav[:self.speech_wav_len]
        else:
            far_wav = np.pad(far_wav, (self.speech_wav_len-len(far_wav), 0), mode='constant', constant_values=0)
            echo_wav = np.pad(echo_wav, (self.speech_wav_len-len(echo_wav), 0), mode='constant', constant_values=0)

        # near_end
        near_wav, _ = sf.read(self.near_list[idx % len(self.near_list)])
        if len(near_wav) >= self.speech_wav_len:
            near_wav = near_wav[:self.speech_wav_len]
        else:
            near_wav = np.pad(near_wav, (self.speech_wav_len-len(near_wav), 0), mode='constant', constant_values=0)


        Ser = self.ser[idx % len(self.ser)]
        alpha = self.mix2signal(near_wav, echo_wav, Ser)
        echo_wav = alpha * echo_wav

        choice_for_DT_or_ST = idx % 5
        if choice_for_DT_or_ST == 3:
            # far-end single talk
            near_wav *= 0
            mix_wav = echo_wav
        elif choice_for_DT_or_ST == 4:
            # near-end single talk
            mix_wav = near_wav
        else:
            # double-talk
            mix_wav = near_wav + echo_wav

        sample = (Variable(torch.FloatTensor(mix_wav.astype('float32'))),
                  Variable(torch.FloatTensor(far_wav.astype('float32'))),
                  Variable(torch.FloatTensor(near_wav.astype('float32'))),
                  )

        return sample

class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.near_path = self.args.tr_nearend_path
        self.far_path = self.args.tr_farend_path
        self.len = self.args.num_test

        self.far_list = sorted(glob.glob(self.far_path + '/*_lpb.wav'))[-500:]
        self.near_list = sorted(glob.glob(self.near_path + '/*'))[-500:]

        self.ser = range(self.args.ser[0], self.args.ser[1])
        self.speech_wav_len = args.speech_len * args.sample_rate

    def __len__(self):
        return self.len

    def mix2signal(self, sig1, sig2, snr):
        alpha = np.sqrt(
            (np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + 1.0e-8)) / 10.0 ** (snr / 10.0) + 1.0e-8)
        return alpha

    def __getitem__(self, idx):

        # far_end / echo
        far_path = self.far_list[idx % len(self.far_list)]
        far_wav, _ = sf.read(far_path)
        echo_wav, _ = sf.read(far_path.replace('_lpb.wav', '_mic.wav'))
        if len(far_wav) >= self.speech_wav_len:
            far_wav = far_wav[:self.speech_wav_len]
            echo_wav = echo_wav[:self.speech_wav_len]
        else:
            far_wav = np.pad(far_wav, (self.speech_wav_len - len(far_wav), 0), mode='constant', constant_values=0)
            echo_wav = np.pad(echo_wav, (self.speech_wav_len - len(echo_wav), 0), mode='constant', constant_values=0)

        # near_end
        near_wav, _ = sf.read(self.near_list[idx % len(self.near_list)])
        if len(near_wav) >= self.speech_wav_len:
            near_wav = near_wav[:self.speech_wav_len]
        else:
            near_wav = np.pad(near_wav, (self.speech_wav_len - len(near_wav), 0), mode='constant', constant_values=0)

        Ser = self.ser[idx % len(self.ser)]
        alpha = self.mix2signal(near_wav, echo_wav, Ser)
        echo_wav = alpha * echo_wav

        choice_for_DT_or_ST = idx % 5
        type = None
        if choice_for_DT_or_ST == 3:
            # far-end single talk
            near_wav *= 0
            mix_wav = echo_wav
            type = 'FE_ST'
        elif choice_for_DT_or_ST == 4:
            # near-end single talk
            mix_wav = near_wav
            type = 'NE_ST'
        else:
            # double-talk
            mix_wav = near_wav + echo_wav
            type = 'DT'

        sample = (Variable(torch.FloatTensor(mix_wav.astype('float32'))),
                  Variable(torch.FloatTensor(far_wav.astype('float32'))),
                  Variable(torch.FloatTensor(near_wav.astype('float32'))),
                  type
                  )

        return sample

