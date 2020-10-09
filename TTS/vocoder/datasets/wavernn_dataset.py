import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset
from multiprocessing import Manager


class WaveRNNDataset(Dataset):
    def __init__(self,
                 ap,
                 items,
                 seq_len,
                 pad_short,
                 mode,
                 mulaw=False,
                 conv_pad=2,
                 is_training=False,
                 use_noise_augment=False,
                 use_cache=False,
                 verbose=False):
        """WaveRNN dataset with a special collate() function.
        Args:
            ap (TTS.utils.AudioProcessor): AudioProcessor object.
            items (list): list of dataset items.
            seq_len (int): length of randomly selected segment from each voice signal.
            pad_short (int): length of padding applied to short samples.
            mode (str): model training mode which defines output type as quantized or continuous.
            mulaw (bool): if true voice signal is quantized using mulaw method.
            conv_path (int): padding applied to keep context in receptive field as applying convolutions to conditional features.
            is_training (bool): enable/disable training mode.
            use_noise_augment (bool): enable/disable random noise augmentation.
            use_cache (bool): enable/disable in memory cache. If dataset is big, might cause OOM error.
            verbose (bool): enable/disable logging on terminal.
        """

        self.ap = ap
        self.item_list = items
        self.compute_feat = not isinstance(items[0], (tuple, list))
        self.seq_len = seq_len
        self.pad_short = pad_short
        self.mode = mode
        self.mulaw = mulaw
        self.conv_pad = conv_pad
        self.is_training = is_training
        self.use_cache = use_cache
        self.use_noise_augment = use_noise_augment
        self.verbose = verbose
        self.hop_len = ap.hop_length

        if self.use_cache:
            self.create_feature_cache()

        assert seq_len % self.hop_len == 0, " [!] seq_len has to be a multiple of hop_len."
        self.feat_frame_len = seq_len // self.hop_len + (2 * conv_pad)

    def quantize_wav(self, wav):
        """quantize the audio depending on model's mode"""
        if self.mode in ['gauss', 'mold']:
            # continious output
            quant = wav
        elif type(self.mode) is int and self.mulaw:
            # mulaw quantized output
            quant = self.ap.mulaw_encode(wav, self.mode)
            quant = quant.astype(np.int32)
        elif type(self.mode) is int:
            # quantized output
            quant = self.ap.quantize(wav)
            quant = quant.clip(0, 2**self.mode - 1)
            quant = quant.astype(np.int32)
        return quant

    def create_feature_cache(self):
        """setup data loader in memory cache"""
        self.manager = Manager()
        self.cache = self.manager.list()
        self.cache += [None for _ in range(len(self.item_list))]

    @staticmethod
    def find_wav_files(path):
        return glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        mel, audio = self.load_item(idx)
        return (mel, audio)

    def load_item(self, idx):
        """ load (feat, mel) couple """
        if self.compute_feat:
            # compute features from wav
            wavpath = self.item_list[idx]
            # print(wavpath)

            if self.use_cache and self.cache[idx] is not None:
                audio, mel = self.cache[idx]
            else:
                audio = self.ap.load_wav(wavpath)

                if len(audio) < self.seq_len :
                    audio = np.pad(audio, (0, self.seq_len + self.pad_short - len(audio)), \
                            mode='constant', constant_values=0.0)

                mel = self.ap.melspectrogram(audio)
        else:

            # load precomputed features
            wavpath, feat_path = self.item_list[idx]

            if self.use_cache and self.cache[idx] is not None:
                audio, mel = self.cache[idx]
            else:
                audio = self.ap.load_wav(wavpath)
                mel = np.load(feat_path)

        # correct the audio length wrt padding applied in stft
        audio = np.pad(audio, (0, self.hop_len), mode="edge")
        audio = audio[:mel.shape[-1] * self.hop_len]
        assert mel.shape[-1] * self.hop_len == audio.shape[
            -1], f' [!] {mel.shape[-1] * self.hop_len} vs {audio.shape[-1]}'


        # quantize output
        audio = self.quantize_wav(audio)

        if self.use_noise_augment and self.is_training and self.return_segments:
            audio = audio + (1 / 32768) * np.rand(audio.shape)

        # audio = torch.from_numpy(audio).float().unsqueeze(0)
        # mel = torch.from_numpy(mel).float().squeeze(0)
        return (mel, audio)

    def collate(self, batch):
        """Collate samples generated by each worker and gather the next batch"""

        mel_win = self.feat_frame_len
        seq_len = self.seq_len
        pad = self.conv_pad  # padding against resnet

        # check if min mel length is larger than required size
        min_mel_len = np.min([x[0].shape[-1] for x in batch])
        assert min_mel_len > mel_win

        # set offsets for picking a segment
        max_offsets = [x[0].shape[-1] - mel_win for x in batch]
        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        sig_offsets = [(offset + pad) * self.hop_len for offset in mel_offsets]

        # create sample segments
        mels = [
            x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win]
            for i, x in enumerate(batch)
        ]
        wavs = [
            x[1][sig_offsets[i]:sig_offsets[i] + seq_len + 1]
            for i, x in enumerate(batch)
        ]

        # create batch
        mels = np.stack(mels).astype(np.float32)

        if self.mode in ['gauss', 'mold']:
            # continious output
            wavs = np.stack(wavs).astype(np.float32)
            wavs = torch.FloatTensor(wavs)
            x = wavs[:, :seq_len]
        elif isinstance(self.mode, int):
            # quantized output
            wavs = np.stack(wavs).astype(np.int64)
            wavs = torch.LongTensor(wavs)
            x = 2 * wavs[:, :seq_len].float() / (2**self.mode - 1.0) - 1.0

        # set label
        y = wavs[:, 1:]
        mels = torch.FloatTensor(mels)
        return x, mels, y
