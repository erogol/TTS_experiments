import librosa
import soundfile as sf
import numpy as np
import scipy.io
import scipy.signal


class AudioProcessor(object):
    def __init__(self,
                 sample_rate=None,
                 num_mels=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 hop_length=None,
                 win_length=None,
                 num_freq=None,
                 power=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_db=60,
                 sound_norm=False,
                 use_cuda=False,
                 **_):

        print(" > Setting up Audio Processor...")
        # setup class attributed
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.num_freq = num_freq
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.sound_norm = sound_norm
        # setup stft parameters
        if hop_length is None:
            self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        else:
            self.hop_length = hop_length
            self.win_length = win_length
            self.n_fft = (self.num_freq - 1) * 2
        members = vars(self)
        # print class attributes
        for key, value in members.items():
            print(" | > {}:{}".format(key, value))
        # create spectrogram utils
        self.mel_basis = self._build_mel_basis()
        self.inv_mel_basis = np.linalg.pinv(self._build_mel_basis())

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    ### setting up the parameters ###
    def _build_mel_basis(self, ):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            self.n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        factor = self.frame_length_ms / self.frame_shift_ms
        assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(hop_length * factor)
        return n_fft, hop_length, win_length
    
    ### DB and AMP conversion ###
    def amp_to_db(self, x):
        return np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x)

    ### SPECTROGRAM ###
    def linear_to_mel(self, spectrogram):
        return np.dot(self.mel_basis, spectrogram)

    def mel_to_linear(self, mel_spec):
        return np.maximum(1e-10, np.dot(self.inv_mel_basis, mel_spec))

    def spectrogram(self, y):
        D = self._stft(y)
        S = self.amp_to_db(np.abs(D))
        return S

    def melspectrogram(self, y):
        D = self._stft(y)
        S = self.amp_to_db(self.linear_to_mel(np.abs(D)))
        return S

    ### INV SPECTROGRAM ###
    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        S = self.db_to_amp(spectrogram)
        return self._griffin_lim(S**self.power)

    def inv_melspectrogram(self, spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        S = self.db_to_amp(spectrogram)
        S = self.mel_to_linear(S)  # Convert back to linear
        return self._griffin_lim(S**self.power)

    def out_linear_to_mel(self, linear_spec):
        S = self._denormalize(linear_spec)
        S = self._db_to_amp(S + self.ref_level_db)
        S = self._linear_to_mel(np.abs(S))
        S = self._amp_to_db(S) - self.ref_level_db
        mel = self._normalize(S)
        return mel

    ### STFT and ISTFT ###
    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode='constant'
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)
    
    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    ### Audio processing ###
    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.01 sec margin """
        margin = int(self.sample_rate * 0.01)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=self.trim_db, frame_length=self.win_length, hop_length=self.hop_length)[0]
    
    def sound_norm(self, x):
        return x / abs(x).max() * 0.9

    ### save and load ###
    def load_wav(self, filename, sr=None):
        if sr is None:
            x, sr = sf.read(filename)
        else:
            x, sr = librosa.load(filename, sr=sr)
        return x
    
    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def quantize(x, bits):
        return (x + 1.) * (2**bits - 1) / 2

    @staticmethod
    def dequantize(x, bits):
        return 2 * x / (2**bits - 1) - 1

    @staticmethod
    def mulaw_encode(wav, qc):
        mu = 2 ** qc - 1
        # wav_abs = np.minimum(np.abs(wav), 1.0)
        signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1. + mu)
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return np.floor(signal,)

    @staticmethod
    def mulaw_decode(wav, qc):
        """Recovers waveform from quantized values."""
        mu = 2 ** qc - 1
        x = np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)
        return x
