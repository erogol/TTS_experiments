import torch
from torch import nn

from TTS.tts.layers.gst_layers import GST
from TTS.tts.layers.tacotron2 import Decoder, Encoder, Postnet
from TTS.tts.models.tacotron_abstract import TacotronAbstract

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
                 speaker_embedding_dim=None,
                 gst=False,
                 gst_embedding_dim=512,
                 gst_num_heads=4,
                 gst_style_tokens=10,
                 disable_gst=False,
                 gst_use_speaker_embedding=False):
        super(Tacotron2,
              self).__init__(num_chars, num_speakers, r, postnet_output_dim,
                             decoder_output_dim, attn_type, attn_win,
                             attn_norm, prenet_type, prenet_dropout,
                             forward_attn, trans_agent, forward_attn_mask,
                             location_attn, attn_K, separate_stopnet,
                             bidirectional_decoder, double_decoder_consistency,
                             ddc_r, gst, gst_embedding_dim, gst_num_heads, gst_style_tokens,
                             disable_gst, gst_use_speaker_embedding)

        # init layer dims
        decoder_in_features = 512
        encoder_in_features = 512

        if speaker_embedding_dim is None:
            # if speaker_embedding_dim is None we need use the nn.Embedding, with default speaker_embedding_dim
            self.embeddings_per_sample = False
            speaker_embedding_dim = 512
        else:
            # if speaker_embedding_dim is not None we need use speaker embedding per sample
            self.embeddings_per_sample = True

        # speaker and gst embeddings is concat in decoder input
        if num_speakers > 1:
            decoder_in_features = decoder_in_features + speaker_embedding_dim # add speaker embedding dim
        if self.gst:
            decoder_in_features = decoder_in_features + gst_embedding_dim # add gst embedding dim

        # embedding layer
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)

        # speaker embedding layer
        if num_speakers > 1:
            if not self.embeddings_per_sample:
                self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
                self.speaker_embedding.weight.data.normal_(0, 0.3)

        # base model layers
        self.encoder = Encoder(encoder_in_features)
        self.decoder = Decoder(decoder_in_features, self.decoder_output_dim, r, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet)
        self.postnet = Postnet(self.postnet_output_dim)

        # global style token layers
        if self.gst:
            self.gst_layer = GST(num_mel=80,
                                num_heads=gst_num_heads,
                                num_style_tokens=gst_style_tokens,
                                gst_embedding_dim=gst_embedding_dim,
                                speaker_embedding_dim=speaker_embedding_dim if self.embeddings_per_sample and self.gst_use_speaker_embedding else None)

        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self.coarse_decoder = Decoder(
                decoder_in_features, self.decoder_output_dim, ddc_r, attn_type,
                attn_win, attn_norm, prenet_type, prenet_dropout, forward_attn,
                trans_agent, forward_attn_mask, location_attn, attn_K,
                separate_stopnet)

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, mel_specs=None, mel_lengths=None, speaker_ids=None, speaker_embeddings=None):
        # compute mask for padding
        # B x T_in_max (boolean)
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x D_embed x T_in_max
        embedded_inputs = self.embedding(text).transpose(1, 2)
        # B x T_in_max x D_en
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, 
                                                mel_specs if not self.disable_gst else None, # if not disable gst pass wav reference
                                                speaker_embeddings if self.gst_use_speaker_embedding else None)

        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            else:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = torch.unsqueeze(speaker_embeddings, 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)

        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)

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
    def inference(self, text, speaker_ids=None, style_mel=None, speaker_embeddings=None):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs,
                                                style_mel if not self.disable_gst else None, # if not disable gst pass wav reference
                                                speaker_embeddings if self.gst_use_speaker_embedding else None)
        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)

        decoder_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    def inference_truncated(self, text, speaker_ids=None, style_mel=None, speaker_embeddings=None):
        """
        Preserve model states for continuous inference
        """
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)

        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs,
                                                style_mel if not self.disable_gst else None, # if not disable gst pass wav reference
                                                speaker_embeddings if self.gst_use_speaker_embedding else None)

        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)

        mel_outputs, alignments, stop_tokens = self.decoder.inference_truncated(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens
