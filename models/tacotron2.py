import copy
import torch
from math import sqrt
from torch import nn
from TTS.layers.tacotron2 import Encoder, Decoder, Postnet
from TTS.utils.generic_utils import sequence_mask


# TODO: match function arguments with tacotron
class Tacotron2(nn.Module):
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
                 bidirectional_decoder=False):
        super(Tacotron2, self).__init__()
        self.postnet_output_dim = postnet_output_dim
        self.decoder_output_dim = decoder_output_dim
        self.r = r
        self.bidirectional_decoder = bidirectional_decoder
        decoder_dim = 512 if num_speakers > 1 else 512
        encoder_dim = 512 if num_speakers > 1 else 512
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        # embedding layer
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)
        std = sqrt(2.0 / (num_chars + 512))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, 512)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
            self.speaker_embeddings = None
            self.speaker_embeddings_projected = None
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(decoder_dim, self.decoder_output_dim, r, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet, proj_speaker_dim)
        if self.bidirectional_decoder:
            self.decoder_backward = copy.deepcopy(self.decoder)
        self.postnet = Postnet(self.postnet_output_dim)

    def _init_states(self):
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, mel_specs=None, mel_lengths=None, speaker_ids=None):
        self._init_states()
        # compute mask for padding
        # B x T_in_max (boolean)
        input_mask = sequence_mask(text_lengths).to(text.device)
        if mel_lengths is not None:
            max_len = mel_lengths.max()
            r = self.decoder.r
            max_len = max_len + (r - (max_len % r)) if max_len % r > 0 else max_len 
            output_mask = sequence_mask(mel_lengths, max_len=max_len).to(text.device)
        # B x D_embed x T_in_max
        embedded_inputs = self.embedding(text).transpose(1, 2)
        # B x T_in_max x D_en
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        # adding speaker embeddding to encoder output 
        # TODO: try adding speaker embedding to character embedding 
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)
        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)
        # B x mel_dim x T_out -- B x T_out//r x T_in -- B x T_out//r
        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, input_mask)
        # sequence masking
        if mel_lengths is not None:
            decoder_outputs = decoder_outputs * output_mask.unsqueeze(1).expand_as(decoder_outputs) 
        # B x mel_dim x T_out 
        postnet_outputs = self.postnet(decoder_outputs)
        # sequence masking
        if mel_lengths is not None:
            postnet_outputs = postnet_outputs * output_mask.unsqueeze(1).expand_as(postnet_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        # B x T_out x mel_dim -- B x T_out x mel_dim -- B x T_out//r x T_in
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_inference(mel_specs, encoder_outputs, input_mask)
            return decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    @torch.no_grad()
    def inference(self, text, speaker_ids=None):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

    def inference_truncated(self, text, speaker_ids=None):
        """
        Preserve model states for continuous inference
        """
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)
        mel_outputs, alignments, stop_tokens = self.decoder.inference_truncated(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

    def _backward_inference(self, mel_specs, encoder_outputs, mask):
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask,
            self.speaker_embeddings_projected)
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2)
        return decoder_outputs_b, alignments_b

    def _add_speaker_embedding(self, encoder_outputs, speaker_ids):
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(" [!] Model has speaker embedding layer but speaker_id is not provided")
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            speaker_embeddings = self.speaker_embedding(speaker_ids)

            speaker_embeddings.unsqueeze_(1)
            speaker_embeddings = speaker_embeddings.expand(encoder_outputs.size(0),
                                                           encoder_outputs.size(1),
                                                           -1)
            encoder_outputs = encoder_outputs + speaker_embeddings
        return encoder_outputs
