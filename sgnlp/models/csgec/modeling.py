from math import inf
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .config import CsgConfig
from .modules.conv_decoder import ConvDecoder
from .modules.conv_encoder import ConvEncoder
from .utils import Buffer, Beam


class CsgPreTrainedModel(PreTrainedModel):
    config_class = CsgConfig
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


class CsgModel(CsgPreTrainedModel):
    def __init__(self, config: CsgConfig):
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

    def decode(self, batch_source_ids, batch_context_ids):
        batch_output = []

        for text_source_ids, text_context_ids in zip(
            batch_source_ids, batch_context_ids
        ):
            text_output = []
            for sentence_source_ids, sentence_context_ids in zip(
                text_source_ids, text_context_ids
            ):
                encoder_out_dict = self.encoder(sentence_source_ids.reshape(1, -1))
                auxencoder_out_dict = self.auxencoder(
                    sentence_context_ids.reshape(1, -1)
                )
                best_sentence_indices = self._decode_one(
                    encoder_out_dict=encoder_out_dict,
                    auxencoder_out_dict=auxencoder_out_dict,
                    beam_size=self.config.beam_size,
                    max_len=len(sentence_source_ids)
                    + 5,  # +5 is arbitrary. Normally we'd add 1 for the EOS token
                )
                text_output += [best_sentence_indices[1:-1]]
            batch_output.append(text_output)

        return batch_output

    def _decode_one(
        self,
        encoder_out_dict,
        auxencoder_out_dict,
        beam_size,
        max_len=None,
    ):
        max_len = max_len if max_len is not None else self.config.trg_max_seq_len
        incremental_state_buffer = Buffer(max_len=self.config.num_decoders)
        finalised_cands_beam = Beam(beam_size=beam_size)
        num_cands = beam_size * 2

        prev_output_tokens = torch.tile(
            torch.LongTensor([self.config.eos_idx]), (beam_size, 1)
        )

        for step in range(max_len):
            decoder_output = self.decoder(
                prev_output_tokens=prev_output_tokens,
                encoder_out_dict=encoder_out_dict,
                auxencoder_out_dict=auxencoder_out_dict,
                incremental_state=incremental_state_buffer,
            )

            if step == 0:
                seq_values, topk_cand = torch.topk(
                    torch.log_softmax(decoder_output[:, -1:, :], dim=2)[0, :, :].data,
                    num_cands,
                )

                # Check which candidates have been finalised by checking whether
                # the EOS token has been generated
                unfinalised_indices = topk_cand != self.config.eos_idx
                finalised_indices = topk_cand == self.config.eos_idx

                # If there are any finalised candidates, add them to the beam
                if torch.any(finalised_indices):
                    finalised_cand_lst = topk_cand[finalised_indices].reshape(-1)
                    finalised_cand_lst = torch.cat(
                        (
                            torch.tile(
                                torch.LongTensor([self.config.eos_idx]),
                                (finalised_cand_lst.shape[0], 1),
                            ),
                            finalised_cand_lst,
                        ),
                        dim=1,
                    ).tolist()

                    # Normalise scores based on the number of tokens generated
                    finalised_scores_lst = seq_values[finalised_indices] / (step + 1)
                    finalised_score_lst = finalised_scores_lst.reshape(-1).tolist()

                    # Add the finalised candidates
                    finalised_cands_beam.add_elements(
                        finalised_score_lst, finalised_cand_lst
                    )

                # We will continue the search on beam_size unfinalised candidates
                # Ie, we continue with the topk unfinalised candidates
                unfinalised_cand_lst = topk_cand[unfinalised_indices][:beam_size]
                seq_values = seq_values[unfinalised_indices][:beam_size]

                prev_output_tokens = torch.cat(
                    (
                        prev_output_tokens,
                        unfinalised_cand_lst.reshape(-1, 1),
                    ),
                    dim=1,
                )
            else:
                log_proba = torch.log_softmax(decoder_output, dim=2)
                log_proba[:, :, self.config.padding_idx] = -inf
                seq_values = (
                    log_proba.reshape(beam_size, -1) + seq_values.reshape(beam_size, -1)
                ).reshape(-1)
                seq_values, topk_cand = torch.topk(seq_values.data, num_cands)

                # Compute the new token indices first
                new_token_vocab_indices = topk_cand % self.config.target_vocab_size
                prev_sequence_indices = torch.div(
                    topk_cand,
                    self.config.target_vocab_size,
                    rounding_mode="floor",
                )

                # Check which candidates have been finalised by checking whether
                # the EOS token has been generated
                unfinalised_indices = new_token_vocab_indices != self.config.eos_idx
                finalised_indices = new_token_vocab_indices == self.config.eos_idx

                # If there are any finalised candidates, add them to the beam
                if torch.any(finalised_indices):
                    finalised_new_token_vocab_indices = new_token_vocab_indices[
                        finalised_indices
                    ].reshape(-1, 1)
                    finalised_prev_sequence_indices = prev_sequence_indices[
                        finalised_indices
                    ]
                    finalised_cand_lst = torch.cat(
                        (
                            prev_output_tokens[finalised_prev_sequence_indices],
                            finalised_new_token_vocab_indices,
                        ),
                        dim=1,
                    ).tolist()

                    # Normalise scores based on the number of tokens generated
                    finalised_scores_lst = seq_values[finalised_indices] / (step + 1)
                    finalised_score_lst = finalised_scores_lst.reshape(-1).tolist()

                    # Add the elements to the beam
                    finalised_cands_beam.add_elements(
                        finalised_score_lst, finalised_cand_lst
                    )

                # Retain only the top beam_size worth of unfinalised hypotheses
                unfinalised_new_token_vocab_indices = new_token_vocab_indices[
                    unfinalised_indices
                ][:beam_size].reshape(beam_size, -1)
                unfinalised_prev_sequence_indices = prev_sequence_indices[
                    unfinalised_indices
                ][:beam_size]
                prev_output_tokens = torch.cat(
                    (
                        prev_output_tokens[unfinalised_prev_sequence_indices],
                        unfinalised_new_token_vocab_indices,
                    ),
                    dim=1,
                )

                seq_values = seq_values[unfinalised_indices][:beam_size]

                for idx in range(self.config.num_decoders):
                    temp = incremental_state_buffer.get_first_element()[
                        unfinalised_prev_sequence_indices, :, :
                    ]
                    incremental_state_buffer.add_element(temp)

                # If the score of the unfinalised outputs are smaller than the score
                # of the worst finalised candidate, then it can't get any better and we
                # stop the generation
                if (
                    (seq_values / max_len)[0] < finalised_cands_beam.get_lowest_score()
                ) and len(finalised_cands_beam.get_elements()) != 0:
                    break

        return finalised_cands_beam.get_best_element()["indices"]
