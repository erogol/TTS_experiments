import re
import os
import glob
import torch
import shutil
import datetime
import subprocess
import importlib
import numpy as np
from collections import Counter

from TTS.utils.generic_utils import check_argument


def split_dataset(items):
    is_multi_speaker = False
    speakers = [item[2] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    eval_split_size = 500 if len(items) * 0.01 > 500 else int(
        len(items) * 0.01)
    assert eval_split_size > 0, " [!] You do not have enough samples to train. You need at least 100 samples."
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        # most stupid code ever -- Fix it !
        while len(items_eval) < eval_split_size:
            speakers = [item[-1] for item in items]
            speaker_counter = Counter(speakers)
            item_idx = np.random.randint(0, len(items))
            if speaker_counter[items[item_idx][-1]] > 1:
                items_eval.append(items[item_idx])
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]


# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.to(sequence_length.device)
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    # B x T_max
    return seq_range_expand < seq_length_expand


def to_camel(text):
    text = text.capitalize()
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), text)


def setup_model(num_chars, num_speakers, c):
    print(" > Using model: {}".format(c.model))
    MyModel = importlib.import_module('TTS.tts.models.' + c.model.lower())
    MyModel = getattr(MyModel, to_camel(c.model))
    if c.model.lower() in "tacotron":
        model = MyModel(num_chars=num_chars,
                        num_speakers=num_speakers,
                        r=c.r,
                        postnet_output_dim=int(c.audio['fft_size'] / 2 + 1),
                        decoder_output_dim=c.audio['num_mels'],
                        gst=c.use_gst,
                        memory_size=c.memory_size,
                        attn_type=c.attention_type,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        forward_attn=c.use_forward_attn,
                        trans_agent=c.transition_agent,
                        forward_attn_mask=c.forward_attn_mask,
                        location_attn=c.location_attn,
                        attn_K=c.attention_heads,
                        separate_stopnet=c.separate_stopnet,
                        bidirectional_decoder=c.bidirectional_decoder,
                        double_decoder_consistency=c.double_decoder_consistency,
                        ddc_r=c.ddc_r)
    elif c.model.lower() == "tacotron2":
        model = MyModel(num_chars=num_chars,
                        num_speakers=num_speakers,
                        r=c.r,
                        postnet_output_dim=c.audio['num_mels'],
                        decoder_output_dim=c.audio['num_mels'],
                        gst=c.use_gst,
                        attn_type=c.attention_type,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        forward_attn=c.use_forward_attn,
                        trans_agent=c.transition_agent,
                        forward_attn_mask=c.forward_attn_mask,
                        location_attn=c.location_attn,
                        attn_K=c.attention_heads,
                        separate_stopnet=c.separate_stopnet,
                        bidirectional_decoder=c.bidirectional_decoder,
                        double_decoder_consistency=c.double_decoder_consistency,
                        ddc_r=c.ddc_r)
    elif c.model.lower() == "glow_tts":
        model = MyModel(num_chars=num_chars,
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
                        num_speakers=num_speakers,
                        c_in_channels=0,
                        num_splits=4,
                        num_sqz=2,
                        sigmoid_scale=False,
                        rel_attn_winndow_size=4,
                        input_length=None,
                        mean_only=True,
                        hidden_channels_enc=192,
                        hidden_channels_dec=192,
                        prenet=True)
    return model

class KeepAverage():
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def items(self):
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if name not in self.avg_values:
            # add value if not exist before
            self.add_value(name, init_val=value)
        else:
            # else update existing value
            if weighted_avg:
                self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
                self.iters[name] += 1
            else:
                self.avg_values[name] = self.avg_values[name] * \
                    self.iters[name] + value
                self.iters[name] += 1
                self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)


def check_config(c):
    check_argument('model', c, enum_list=['tacotron', 'tacotron2', 'glow_tts'], restricted=True, val_type=str)
    check_argument('run_name', c, restricted=True, val_type=str)
    check_argument('run_description', c, val_type=str)

    # AUDIO
    check_argument('audio', c, restricted=True, val_type=dict)

    # audio processing parameters
    check_argument('num_mels', c['audio'], restricted=True, val_type=int, min_val=10, max_val=2056)
    check_argument('fft_size', c['audio'], restricted=True, val_type=int, min_val=128, max_val=4058)
    check_argument('sample_rate', c['audio'], restricted=True, val_type=int, min_val=512, max_val=100000)
    check_argument('frame_length_ms', c['audio'], restricted=True, val_type=float, min_val=10, max_val=1000, alternative='win_length')
    check_argument('frame_shift_ms', c['audio'], restricted=True, val_type=float, min_val=1, max_val=1000, alternative='hop_length')
    check_argument('preemphasis', c['audio'], restricted=True, val_type=float, min_val=0, max_val=1)
    check_argument('min_level_db', c['audio'], restricted=True, val_type=int, min_val=-1000, max_val=10)
    check_argument('ref_level_db', c['audio'], restricted=True, val_type=int, min_val=0, max_val=1000)
    check_argument('power', c['audio'], restricted=True, val_type=float, min_val=1, max_val=5)
    check_argument('griffin_lim_iters', c['audio'], restricted=True, val_type=int, min_val=10, max_val=1000)

    # vocabulary parameters
    check_argument('characters', c, restricted=False, val_type=dict)
    check_argument('pad', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('eos', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('bos', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('characters', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('phonemes', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('punctuations', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)

    # normalization parameters
    check_argument('signal_norm', c['audio'], restricted=True, val_type=bool)
    check_argument('symmetric_norm', c['audio'], restricted=True, val_type=bool)
    check_argument('max_norm', c['audio'], restricted=True, val_type=float, min_val=0.1, max_val=1000)
    check_argument('clip_norm', c['audio'], restricted=True, val_type=bool)
    check_argument('mel_fmin', c['audio'], restricted=True, val_type=float, min_val=0.0, max_val=1000)
    check_argument('mel_fmax', c['audio'], restricted=True, val_type=float, min_val=500.0)
    check_argument('spec_gain', c['audio'], restricted=True, val_type=float, min_val=1, max_val=100)
    check_argument('do_trim_silence', c['audio'], restricted=True, val_type=bool)
    check_argument('trim_db', c['audio'], restricted=True, val_type=int)

    # training parameters
    check_argument('batch_size', c, restricted=True, val_type=int, min_val=1)
    check_argument('eval_batch_size', c, restricted=True, val_type=int, min_val=1)
    check_argument('r', c, restricted=True, val_type=int, min_val=1)
    check_argument('gradual_training', c, restricted=False, val_type=list)
    check_argument('loss_masking', c, restricted=True, val_type=bool)
    # check_argument('grad_accum', c, restricted=True, val_type=int, min_val=1, max_val=100)

    # validation parameters
    check_argument('run_eval', c, restricted=True, val_type=bool)
    check_argument('test_delay_epochs', c, restricted=True, val_type=int, min_val=0)
    check_argument('test_sentences_file', c, restricted=False, val_type=str)

    # optimizer
    check_argument('noam_schedule', c, restricted=False, val_type=bool)
    check_argument('grad_clip', c, restricted=True, val_type=float, min_val=0.0)
    check_argument('epochs', c, restricted=True, val_type=int, min_val=1)
    check_argument('lr', c, restricted=True, val_type=float, min_val=0)
    check_argument('wd', c, restricted=True, val_type=float, min_val=0)
    check_argument('warmup_steps', c, restricted=True, val_type=int, min_val=0)
    check_argument('seq_len_norm', c, restricted=True, val_type=bool)

    # tacotron prenet
    check_argument('memory_size', c, restricted=True, val_type=int, min_val=-1)
    check_argument('prenet_type', c, restricted=True, val_type=str, enum_list=['original', 'bn'])
    check_argument('prenet_dropout', c, restricted=True, val_type=bool)

    # attention
    check_argument('attention_type', c, restricted=True, val_type=str, enum_list=['graves', 'original'])
    check_argument('attention_heads', c, restricted=True, val_type=int)
    check_argument('attention_norm', c, restricted=True, val_type=str, enum_list=['sigmoid', 'softmax'])
    check_argument('windowing', c, restricted=True, val_type=bool)
    check_argument('use_forward_attn', c, restricted=True, val_type=bool)
    check_argument('forward_attn_mask', c, restricted=True, val_type=bool)
    check_argument('transition_agent', c, restricted=True, val_type=bool)
    check_argument('transition_agent', c, restricted=True, val_type=bool)
    check_argument('location_attn', c, restricted=True, val_type=bool)
    check_argument('bidirectional_decoder', c, restricted=True, val_type=bool)
    check_argument('double_decoder_consistency', c, restricted=True, val_type=bool)
    check_argument('ddc_r', c, restricted='double_decoder_consistency' in c.keys(), min_val=1, max_val=7, val_type=int)

    # stopnet
    check_argument('stopnet', c, restricted=True, val_type=bool)
    check_argument('separate_stopnet', c, restricted=True, val_type=bool)

    # tensorboard
    check_argument('print_step', c, restricted=True, val_type=int, min_val=1)
    check_argument('tb_plot_step', c, restricted=True, val_type=int, min_val=1)
    check_argument('save_step', c, restricted=True, val_type=int, min_val=1)
    check_argument('checkpoint', c, restricted=True, val_type=bool)
    check_argument('tb_model_param_stats', c, restricted=True, val_type=bool)

    # dataloading
    # pylint: disable=import-outside-toplevel
    from TTS.tts.utils.text import cleaners
    check_argument('text_cleaner', c, restricted=True, val_type=str, enum_list=dir(cleaners))
    check_argument('enable_eos_bos_chars', c, restricted=True, val_type=bool)
    check_argument('num_loader_workers', c, restricted=True, val_type=int, min_val=0)
    check_argument('num_val_loader_workers', c, restricted=True, val_type=int, min_val=0)
    check_argument('batch_group_size', c, restricted=True, val_type=int, min_val=0)
    check_argument('min_seq_len', c, restricted=True, val_type=int, min_val=0)
    check_argument('max_seq_len', c, restricted=True, val_type=int, min_val=10)

    # paths
    check_argument('output_path', c, restricted=True, val_type=str)

    # multi-speaker gst
    check_argument('use_speaker_embedding', c, restricted=True, val_type=bool)
    check_argument('style_wav_for_test', c, restricted=True, val_type=str)
    check_argument('use_gst', c, restricted=True, val_type=bool)

    # datasets - checking only the first entry
    check_argument('datasets', c, restricted=True, val_type=list)
    for dataset_entry in c['datasets']:
        check_argument('name', dataset_entry, restricted=True, val_type=str)
        check_argument('path', dataset_entry, restricted=True, val_type=str)
        check_argument('meta_file_train', dataset_entry, restricted=True, val_type=str)
        check_argument('meta_file_val', dataset_entry, restricted=True, val_type=str)
