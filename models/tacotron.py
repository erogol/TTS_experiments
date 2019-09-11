# coding: utf-8
from torch import nn
from TTS.layers.tacotron import Encoder, Decoder, PostCBHG
from TTS.utils.generic_utils import sequence_mask


class Tacotron(nn.Module):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r=5,
                 linear_dim=1025,
                 mel_dim=80,
                 memory_size=5,
                 attn_win=False,
                 attn_norm="sigmoid",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 separate_stopnet=True):
        super(Tacotron, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(num_chars, 256)
        self.embedding.weight.data.normal_(0, 0.3)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, 256)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(256)
        self.decoder = Decoder(256, mel_dim, r, memory_size, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, separate_stopnet)
        self.postnet = PostCBHG(mel_dim)
        self.last_linear = nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim)
        
    def forward(self, characters, text_lengths, mel_specs, speaker_ids=None):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        self.encoder_outputs = self.encoder(inputs)
        self.encoder_outputs = self._add_speaker_embedding(self.encoder_outputs,
                                                      speaker_ids)
        mel_outputs, alignments, stop_tokens = self.decoder(
            self.encoder_outputs, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def inference(self, characters, speaker_ids=None):
        B = characters.size(0)
        inputs = self.embedding(characters)
        self.encoder_outputs = self.encoder(inputs)
        self.encoder_outputs = self._add_speaker_embedding(self.encoder_outputs,
                                                      speaker_ids)
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            self.encoder_outputs)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def inference_duration(self, characters, model_duration, speaker_ids=None):
        B = characters.size(0)
        inputs = self.embedding(characters)
        self.encoder_outputs = self.encoder(inputs)
        self.encoder_outputs = self._add_speaker_embedding(self.encoder_outputs,
                                                      speaker_ids)
        pred = model_duration.inference(self.encoder_outputs)
        alignments = model_duration.compute_alignment(pred[0])
        context = model_duration.length_regulation(self.encoder_outputs[0], pred[0])
        mel_outputs = self.decoder.inference_duration(inputs,
            context)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        alignments = alignments.T[None, :]
        return mel_outputs, linear_outputs, alignments

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
