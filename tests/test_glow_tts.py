import os
import time
import unittest

import torch
from tests import get_tests_input_path
from torch import nn, optim
from TTS.tts.layers.losses import GlowTTSLoss
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.io import load_config

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(get_tests_input_path(), 'test_config.json'))


class GlowTTSTrainTest(unittest.TestCase):
    def test_train_step(self):  #pylint: disable=no-self-use
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8, )).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, c.audio['num_mels'], 30).to(device)
        mel_postnet_spec = torch.rand(8, c.audio['num_mels'], 30).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        mel_lengths[-1] = 30

        criterion = GlowTTSLoss().to(device)
        model = GlowTTS(num_chars=24,
                        hidden_channels=192,
                        filter_channels=768,
                        filter_channels_dp=256,
                        out_channels=80,
                        kernel_size=3,
                        num_heads=2,
                        num_layers_enc=6,
                        dropout_p=0.1,
                        num_blocks_dec=12,
                        kernel_size_dec=5,
                        dilation_rate=1,
                        num_block_layers=4,
                        dropout_p_dec=0.05,
                        num_speakers=0,
                        c_in_channels=0,
                        num_splits=4,
                        num_sqz=2,
                        sigmoid_scale=False,
                        rel_attn_winndow_size=4,
                        input_length=None,
                        mean_only=True,
                        hidden_channels_enc=192,
                        hidden_channels_dec=192,
                        prenet=True).to(device)

        # training
        model.train()
        count = 0
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for i in range(10):
            (
                z, logdet, y_mean, y_log_scale
            ), attn, o_dur_log, o_total_dur = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths)
            optimizer.zero_grad()
            loss_dict = criterion(z, y_mean, y_log_scale, logdet, mel_lengths, o_dur_log, o_total_dur, input_lengths)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

        # inference
        start = time.time()
        (
            y, logdet, y_mean, y_log_scale
        ), attn, o_dur_log, o_total_dur = model.inference(
            input_dummy, input_lengths)
        print(f"inference runtime: {time.time() - start}")
