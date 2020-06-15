from math import sqrt

import torch
from torch import nn

from TTS.layers.gst_layers import GST
from TTS.layers.tacotron2 import Decoder, Encoder, Postnet
from TTS.models.tacotron_abstract import TacotronAbstract


# TODO: match function arguments with tacotron
class Tacotron2(TacotronAbstract):
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
        super(Tacotron2,
              self).__init__(num_chars, num_speakers, r, postnet_output_dim,
                             decoder_output_dim, attn_type, attn_win,
                             attn_norm, prenet_type, prenet_dropout,
                             forward_attn, trans_agent, forward_attn_mask,
                             location_attn, attn_K, separate_stopnet,
                             bidirectional_decoder, double_decoder_consistency,
                             ddc_r, gst, use_decoder_mask, use_attn_mask)
        decoder_in_features = 512 if num_speakers > 1 else 512
        encoder_in_features = 512 if num_speakers > 1 else 512
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        # base layers
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)
        std = sqrt(2.0 / (num_chars + 512))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, 512)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(encoder_in_features)
        self.decoder = Decoder(decoder_in_features, self.decoder_output_dim, r, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet, proj_speaker_dim)
        self.postnet = Postnet(self.postnet_output_dim)
        # global style token layers
        if self.gst:
            gst_embedding_dim = encoder_in_features
            self.gst_layer = GST(num_mel=80,
                                 num_heads=4,
                                 num_style_tokens=10,
                                 embedding_dim=gst_embedding_dim)
        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self._init_coarse_decoder()

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, mel_specs=None, mel_lengths=None, speaker_ids=None):
        self._init_states()
        # compute mask for padding
        # B x T_in_max (boolean)
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x D_embed x T_in_max
        embedded_inputs = self.embedding(text).transpose(1, 2)
        # B x T_in_max x D_en
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        # adding speaker embeddding to encoder output
        # TODO: multi-speaker
        # B x speaker_embed_dim
        if speaker_ids is not None:
            self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            # B x T_in x embed_dim + speaker_embed_dim
            encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                          self.speaker_embeddings)
        if input_mask is not None:
            encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)
        # global style token
        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs)
        # B x mel_dim x T_out -- B x T_out//r x T_in -- B x T_out//r
        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, input_mask)
        # sequence masking
        if mel_lengths is not None:
            decoder_outputs = decoder_outputs * output_mask.unsqueeze(1).expand_as(decoder_outputs)
        # B x mel_dim x T_out
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        # sequence masking
        if output_mask is not None:
            postnet_outputs = postnet_outputs * output_mask.unsqueeze(1).expand_as(postnet_outputs)
        # B x T_out x mel_dim -- B x T_out x mel_dim -- B x T_out//r x T_in
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_pass(mel_specs, encoder_outputs, input_mask)
            return decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        if self.double_decoder_consistency:
            decoder_outputs_backward, alignments_backward = self._coarse_decoder_pass(mel_specs, encoder_outputs, alignments, input_mask)
            return  decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    @torch.no_grad()
    def inference(self, text, speaker_ids=None):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        if speaker_ids is not None:
            self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                          self.speaker_embeddings)
        decoder_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

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


    def _speaker_embedding_pass(self, encoder_outputs, speaker_ids):
        # TODO: multi-speaker
        # if hasattr(self, "speaker_embedding") and speaker_ids is None:
        #     raise RuntimeError(" [!] Model has speaker embedding layer but speaker_id is not provided")
        # if hasattr(self, "speaker_embedding") and speaker_ids is not None:

        #     speaker_embeddings = speaker_embeddings.expand(encoder_outputs.size(0),
        #                                                    encoder_outputs.size(1),
        #                                                    -1)
        #     encoder_outputs = encoder_outputs + speaker_embeddings
        # return encoder_outputs
        pass
