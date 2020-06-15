import copy
from abc import ABC, abstractmethod

import torch
from torch import nn

from TTS.utils.generic_utils import sequence_mask


class TacotronAbstract(ABC, nn.Module):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r,
                 postnet_output_dim=80,
                 decoder_output_dim=80,
                 attn_type='original',
                 attn_win=False,
                 attn_norm="softmax",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 attn_K=5,
                 separate_stopnet=True,
                 bidirectional_decoder=False,
                 double_decoder_consistency=False,
                 ddc_r=None,
                 gst=False,
                 use_decoder_mask=False,
                 use_attn_mask=False):
        """ Abstract Tacotron class """
        super().__init__()
        self.num_chars = num_chars
        self.r = r
        self.decoder_output_dim = decoder_output_dim
        self.postnet_output_dim = postnet_output_dim
        self.gst = gst
        self.num_speakers = num_speakers
        self.bidirectional_decoder = bidirectional_decoder
        self.double_decoder_consistency = double_decoder_consistency
        self.ddc_r = ddc_r
        self.attn_type = attn_type
        self.attn_win = attn_win
        self.attn_norm = attn_norm
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent
        self.forward_attn_mask = forward_attn_mask
        self.location_attn = location_attn
        self.attn_K = attn_K
        self.separate_stopnet = separate_stopnet
        self.use_decoder_mask = use_decoder_mask
        self.use_attn_mask = use_attn_mask

        # layers
        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.postnet = None

        # global style token
        if self.gst:
            self.gst_layer = None

        # model states
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

        # additional layers
        self.decoder_backward = None
        self.coarse_decoder = None

    #############################
    # INIT FUNCTIONS
    #############################

    def _init_states(self):
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

    def _init_backward_decoder(self):
        self.decoder_backward = copy.deepcopy(self.decoder)

    def _init_coarse_decoder(self):
        self.coarse_decoder = copy.deepcopy(self.decoder)
        self.coarse_decoder.r_init = self.ddc_r
        self.coarse_decoder.set_r(self.ddc_r)

    #############################
    # CORE FUNCTIONS
    #############################

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    #############################
    # COMMON COMPUTE FUNCTIONS
    #############################

    def compute_masks(self, text_lengths, mel_lengths):
        """Compute masks  against sequence paddings."""
        # B x T_in_max (boolean)
        device = text_lengths.device
        input_mask = sequence_mask(text_lengths).to(device) if self.use_attn_mask else None
        output_mask = None
        if mel_lengths is not None and self.use_decoder_mask:
            max_len = mel_lengths.max()
            r = self.decoder.r
            max_len = max_len + (r - (max_len % r)) if max_len % r > 0 else max_len
            output_mask = sequence_mask(mel_lengths, max_len=max_len).to(device)
        return input_mask, output_mask

    def _backward_pass(self, mel_specs, encoder_outputs, mask):
        """ Run backwards decoder """
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask,
            self.speaker_embeddings_projected)
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2).contiguous()
        return decoder_outputs_b, alignments_b

    def _coarse_decoder_pass(self, mel_specs, encoder_outputs, alignments,
                             input_mask):
        """ Double Decoder Consistency """
        T = mel_specs.shape[1]
        if T % self.coarse_decoder.r > 0:
            padding_size = self.coarse_decoder.r - (T % self.coarse_decoder.r)
            mel_specs = torch.nn.functional.pad(mel_specs,
                                                (0, 0, 0, padding_size, 0, 0))
        decoder_outputs_backward, alignments_backward, _ = self.coarse_decoder(
            encoder_outputs.detach(), mel_specs, input_mask)
        # scale_factor = self.decoder.r_init / self.decoder.r
        alignments_backward = torch.nn.functional.interpolate(
            alignments_backward.transpose(1, 2),
            size=alignments.shape[1],
            mode='nearest').transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward.transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward[:, :T, :]
        return decoder_outputs_backward, alignments_backward

    #############################
    # EMBEDDING FUNCTIONS
    #############################

    def compute_speaker_embedding(self, speaker_ids):
        """ Compute speaker embedding vectors """
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(
                " [!] Model has speaker embedding layer but speaker_id is not provided"
            )
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            self.speaker_embeddings = self.speaker_embedding(speaker_ids).unsqueeze(1)
        if hasattr(self, "speaker_project_mel") and speaker_ids is not None:
            self.speaker_embeddings_projected = self.speaker_project_mel(
                self.speaker_embeddings).squeeze(1)

    def compute_gst(self, inputs, mel_specs):
        """ Compute global style token """
        # pylint: disable=not-callable
        gst_outputs = self.gst_layer(mel_specs)
        inputs = self._add_speaker_embedding(inputs, gst_outputs)
        return inputs

    @staticmethod
    def _add_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = outputs + speaker_embeddings_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, speaker_embeddings_], dim=-1)
        return outputs
