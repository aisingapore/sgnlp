from math import inf
import torch
import torch.nn as nn
from transformers import PreTrainedModel


from .config import CSGConfig
from .modules.conv_decoder import ConvDecoder
from .modules.conv_encoder import ConvEncoder
from .utils import Buffer


class CSGPreTrainedModel(PreTrainedModel):

    config_class = CSGConfig
    base_model_prefix = "csg"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CSGModel(CSGPreTrainedModel):
    def __init__(self, config: CSGConfig):
        super().__init__(config)
        self.config = config
        self.encoder = ConvEncoder(
            num_embeddings=config.source_vocab_size,
            embedding_dim=config.embedding_dim,
            max_seq_len=config.src_max_seq_len,
            padding_idx=config.padding_idx,
            token_dropout=config.dropout,
            hidden_dim=config.hidden_dim,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            num_conv_layers=config.num_encoders,
        )
        self.auxencoder = ConvEncoder(
            num_embeddings=config.source_vocab_size,
            embedding_dim=config.embedding_dim,
            max_seq_len=config.ctx_max_seq_len,
            padding_idx=config.padding_idx,
            token_dropout=config.dropout,
            hidden_dim=config.hidden_dim,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            num_conv_layers=config.num_aux_encoders,
        )
        self.decoder = ConvDecoder(
            num_embeddings=config.target_vocab_size,
            embedding_dim=config.embedding_dim,
            max_seq_len=config.trg_max_seq_len,
            padding_idx=config.padding_idx,
            token_dropout=config.dropout,
            hidden_dim=config.hidden_dim,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            num_conv_layers=config.num_decoders,
        )

    def forward(self, *args, **kwargs):
        assert False, "Please use the decode method to get model predictions."

    def decode(self, source_ids, context_ids):

        assert (
            source_ids.shape[0] == 1
        ), "This method currently only supports single sentence inputs."

        encoder_out_dict = self.encoder(source_ids)
        auxencoder_out_dict = self.auxencoder(context_ids)
        hypotheses, seq_scores = self._beam_search_decode_one(
            encoder_out_dict=encoder_out_dict,
            auxencoder_out_dict=auxencoder_out_dict,
            beam_size=self.config.beam_size,
            max_len=source_ids.shape[1] + 1,
        )

        best_sentence_indices = hypotheses[0, 1:].tolist()
        best_sentence_indices = best_sentence_indices[
            : best_sentence_indices.index(self.config.eos_idx)
        ]

        return best_sentence_indices

    def _beam_search_decode_one(
        self,
        encoder_out_dict,
        auxencoder_out_dict,
        beam_size,
        max_len,
    ):
        incremental_state_buffer = Buffer(max_len=self.config.num_decoders)
        prev_output_tokens = torch.tile(
            torch.LongTensor([self.config.eos_idx]), (beam_size, 1)
        )

        # Special set up for first step
        decoder_output = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out_dict=encoder_out_dict,
            auxencoder_out_dict=auxencoder_out_dict,
            incremental_state=incremental_state_buffer,
        )
        seq_values, topk_cand = torch.topk(
            torch.log_softmax(decoder_output[:, -1:, :], dim=2)[0, :, :].data, beam_size
        )
        prev_output_tokens = torch.cat(
            (prev_output_tokens, topk_cand.reshape(-1, 1)), dim=1
        )

        for step in range(max_len):
            print(prev_output_tokens.tolist(), "\n")
            decoder_output = self.decoder(
                prev_output_tokens=prev_output_tokens,
                encoder_out_dict=encoder_out_dict,
                auxencoder_out_dict=auxencoder_out_dict,
                incremental_state=incremental_state_buffer,
            )
            log_proba = torch.log_softmax(decoder_output[:, :, :], dim=2)
            log_proba[:, :, self.config.padding_idx] = -inf
            seq_values = (
                log_proba.reshape(beam_size, -1) + seq_values.reshape(beam_size, -1)
            ).reshape(-1)
            seq_values, topk_cand = torch.topk(seq_values.data, beam_size)

            prev_sequence_indices = torch.div(
                topk_cand, self.config.target_vocab_size, rounding_mode="floor"
            )
            new_token_vocab_indices = topk_cand % self.config.target_vocab_size
            prev_output_tokens = torch.cat(
                (
                    prev_output_tokens[prev_sequence_indices],
                    new_token_vocab_indices.reshape(beam_size, -1),
                ),
                dim=1,
            )

            for idx in range(self.config.num_decoders):
                temp = incremental_state_buffer.get_first_element()[
                    prev_sequence_indices, :, :
                ]
                incremental_state_buffer.add_element(temp)

        return prev_output_tokens, seq_values
