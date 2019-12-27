import os
import unittest
import shutil
import torch
import numpy as np

from torch.utils.data import DataLoader
from TTS.utils.generic_utils import load_config
from TTS.utils.audio import AudioProcessor
from TTS.datasets import TTSDataset
from TTS.datasets.preprocess import ljspeech

#pylint: disable=unused-variable

file_path = os.path.dirname(os.path.realpath(__file__))
OUTPATH = os.path.join(file_path, "outputs/loader_tests/")
os.makedirs(OUTPATH, exist_ok=True)
c = load_config(os.path.join(file_path, 'test_config.json'))
ok_ljspeech = os.path.exists(c.data_path)

DATA_EXIST = True
if not os.path.exists(c.data_path):
    DATA_EXIST = False

class TestTTSDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTTSDataset, self).__init__(*args, **kwargs)
        self.max_loader_iter = 4
        self.ap = AudioProcessor(**c.audio)

    def _create_dataloader(self, batch_size, r, bgs):
        items = ljspeech(c.data_path,'metadata.csv')
        dataset = TTSDataset.MyDataset(
            r,
            c.text_cleaner,
            ap=self.ap,
            meta_data=items, 
            batch_group_size=bgs,
            min_seq_len=c.min_seq_len,
            max_seq_len=float("inf"),
            use_phonemes=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=True,
            num_workers=c.num_loader_workers)
        return dataloader, dataset

    def test_loader(self):
        if ok_ljspeech:
            dataloader, dataset = self._create_dataloader(2, c.r, 0)

            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
                text_input = data[0]
                text_lengths = data[1]
                speaker_name = data[2]
                linear_input = data[3]
                mel_input = data[4]
                mel_lengths = data[5]
                stop_target = data[6]
                item_idx = data[7]

                neg_values = text_input[text_input < 0]
                check_count = len(neg_values)
                assert check_count == 0, \
                    " !! Negative values in text_input: {}".format(check_count)
                # TODO: more assertion here
                assert type(speaker_name[0]) is str
                assert linear_input.shape[0] == c.batch_size
                assert linear_input.shape[2] == self.ap.num_freq
                assert mel_input.shape[0] == c.batch_size
                assert mel_input.shape[2] == c.audio['num_mels']
                # check normalization ranges
                if self.ap.symmetric_norm:
                    assert mel_input.max() <= self.ap.max_norm
                    assert mel_input.min() >= -self.ap.max_norm
                    assert mel_input.min() < 0
                else:
                    assert mel_input.max() <= self.ap.max_norm
                    assert mel_input.min() >= 0