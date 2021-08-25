from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as R

# Using ELMo contextual word embeddings
# from allennlp.modules.elmo import batch_to_ids

from torch.autograd import Variable
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .config import RSTPointerNetworkConfig, RSTParsingNetConfig


@dataclass
class RSTPointerNetworkModelOutput(ModelOutput):
    pass


class RSTPointerNetworkPreTrainedModel(PreTrainedModel):
    config_class = RSTPointerNetworkConfig
    base_model_prefix = "RSTPointerNetwork"


class RSTPointerNetworkModel(RSTPointerNetworkPreTrainedModel):
    def __init__(self, config, elmo):
        super(RSTPointerNetworkModel, self).__init__()

        self.word_dim = config.word_dim
        self.hidden_dim = config.hidden_dim
        self.dropout_prob = config.dropout_prob
        self.is_bi_encoder_rnn = config.is_bi_encoder_rnn
        self.num_rnn_layers = config.rnn_layers
        self.rnn_type = config.rnn_type
        self.with_finetuning = config.with_finetuning

        self.nnDropout = nn.Dropout(config.dropout_prob)

        self.is_batch_norm = config.is_batch_norm

        if self.rnn_type in ['LSTM', 'GRU']:
            self.decoder_rnn = getattr(nn, self.rnn_type)(
                input_size=2 * self.hidden_dim if self.is_bi_encoder_rnn else self.hidden_dim,
                hidden_size=2 * self.hidden_dim if self.is_bi_encoder_rnn else self.hidden_dim,
                num_layers=self.rnn_layers,
                dropout=self.dropout_prob,
                batch_first=True)

            self.encoder_rnn = getattr(nn, self.rnn_type)(
                input_size=self.word_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.rnn_layers,
                bidirectional=self.is_bi_encoder_rnn,
                dropout=self.dropout_prob,
                batch_first=True)
        else:
            raise ValueError('rnn_type should be one of [\'LSTM\', \'GRU\'].')

        self.use_cuda = config.use_cuda

        if self.is_bi_encoder_rnn:
            self.num_encoder_bi = 2
        else:
            self.num_encoder_bi = 1

        self.elmo = elmo

    def init_hidden(self, hsize, batchsize):

        if self.rnn_type == 'LSTM':

            h_0 = Variable(torch.zeros(self.num_encoder_bi * self.num_rnn_layers, batchsize, hsize))
            c_0 = Variable(torch.zeros(self.num_encoder_bi * self.num_rnn_layers, batchsize, hsize))

            if self.use_cuda:
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

            return (h_0, c_0)
        else:

            h_0 = Variable(torch.zeros(self.num_encoder_bi * self.num_rnn_layers, batchsize, hsize))

            if self.use_cuda:
                h_0 = h_0.cuda()

            return h_0

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        # Sort first if ONNX exportability is needed
        x_packed = R.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h

    def pointer_encoder(self, batch_x, batch_x_lens):
        batch_norm = nn.BatchNorm1d(self.word_dim, affine=False, track_running_stats=False)

        batch_size = len(batch_x)

        # # to convert input to ELMo embeddings
        # character_ids = batch_to_ids(batch_x)
        # if self.use_cuda:
        #     character_ids = character_ids.cuda()
        # embeddings = self.elmo(character_ids)
        # batch_x_elmo = embeddings['elmo_representations'][0]  # two layers output  [batch,length,d_elmo]
        # if self.use_cuda:
        #     batch_x_elmo = batch_x_elmo.cuda()

        # X = batch_x_elmo
        X = batch_x
        if self.is_batch_norm:
            X = X.permute(0, 2, 1)  # N C L
            X = batch_norm(X)
            X = X.permute(0, 2, 1)  # N L C

        X = self.nnDropout(X)

        encoder_lstm_co_h_o = self.init_hidden(self.hidden_dim, batch_size)
        output_encoder, hidden_states_encoder = self._run_rnn_packed(self.encoder_rnn, X, batch_x_lens,
                                                                     encoder_lstm_co_h_o)  # batch_first=True
        output_encoder = output_encoder.contiguous()
        output_encoder = self.nnDropout(output_encoder)

        return output_encoder, hidden_states_encoder

    def pointer_layer(self, encoder_states, cur_decoder_state):
        """
        :param encoder_states:  [Length, hidden_size]
        :param cur_decoder_state:  [hidden_size,1]
        """

        # we use simple dot product attention to computer pointer
        attention_pointer = torch.matmul(encoder_states, cur_decoder_state).unsqueeze(1)
        attention_pointer = attention_pointer.permute(1, 0)

        att_weights = F.softmax(attention_pointer)
        logits = F.log_softmax(attention_pointer)

        return logits, att_weights

    def neg_log_likelihood(self, BatchX, IndexX, IndexY, lens):
        encoder_hn, encoder_h_end = self.pointer_encoder(BatchX, lens)

        loss = self.training_decoder(encoder_hn, encoder_h_end, IndexX, IndexY, lens)

        return loss

    def test_decoder(self, h_n, h_end, batch_x_lens, batch_y_index=None):
        """
        :param h_n: all hidden states
        :param h_end: final hidden state
        :param batch_x_lens: lengths of x (i.e. number of tokens)
        :param batch_y_index: optional. provide to get loss metric.
        :return: A tuple containing the following values:
                batch_start_boundaries: array of start tokens for each predicted edu
                batch_end_boundaries: array of end tokens for each predicted edu
                batch_align_matrix: -
                batch_loss: optional metric loss. calculated if batch_y_index is provided.
        """
        total_loops = 0

        batch_start_boundaries = []
        batch_end_boundaries = []
        batch_align_matrix = []

        batch_size = len(batch_x_lens)

        # calculate batch loss if y_index is provided
        if batch_y_index is not None:
            loss_function = nn.NLLLoss()
            batch_loss = 0
        else:
            batch_loss = None

        for i in range(batch_size):
            cur_len = batch_x_lens[i]
            cur_encoder_hn = h_n[i, 0:cur_len, :]  # length * encoder_hidden_size
            cur_end_boundary = cur_len - 1  # end boundary is index of last token

            cur_y_index = batch_y_index[i] if batch_y_index is not None else None

            cur_end_boundaries = []
            cur_start_boundaries = []
            cur_align_matrix = []

            if self.rnn_type == 'LSTM':  # need h_end,c_end
                h_end = h_end[0].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers, -1)
                c_end = h_end[1].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers, -1)

                cur_h0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                cur_c0 = c_end[i].unsqueeze(0).permute(1, 0, 2)

                h_pass = (cur_h0, cur_c0)
            else:  # only need h_end
                h_end = h_end.permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers, -1)
                cur_h0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = cur_h0

            loop_hc = h_pass
            loop_in = cur_encoder_hn[0, :].unsqueeze(0).unsqueeze(0)  # [ 1, 1, encoder_hidden_size] (first start)

            cur_start_boundary = 0
            loop_j = 0

            while True:
                loop_o, loop_hc = self.decoder_rnn(loop_in, loop_hc)

                predict_range = list(range(cur_start_boundary, cur_len))
                cur_encoder_hn_back = cur_encoder_hn[predict_range, :]
                cur_logits, cur_weights = self.pointer_layer(cur_encoder_hn_back, loop_o.squeeze(0).squeeze(0))

                cur_align_vector = np.zeros(cur_len)
                cur_align_vector[predict_range] = cur_weights.data.cpu().numpy()[0]
                cur_align_matrix.append(cur_align_vector)

                _, top_i = cur_logits.data.topk(1)
                pred_index = top_i[0][0]
                ori_pred_index = pred_index + cur_start_boundary

                # Calculate loss
                if batch_y_index is not None:
                    if loop_j > len(cur_y_index) - 1:
                        cur_ground_y = cur_y_index[-1]
                    else:
                        cur_ground_y = cur_y_index[loop_j]

                    cur_ground_y_var = Variable(torch.LongTensor([max(0, int(cur_ground_y) - cur_start_boundary)]))
                    if self.use_cuda:
                        cur_ground_y_var = cur_ground_y_var.cuda()

                    batch_loss += loss_function(cur_logits, cur_ground_y_var)

                if cur_end_boundary <= ori_pred_index:
                    cur_end_boundaries.append(cur_end_boundary)
                    cur_start_boundaries.append(cur_start_boundary)
                    total_loops = total_loops + 1
                    break
                else:
                    cur_end_boundaries.append(ori_pred_index)

                    loop_in = cur_encoder_hn[ori_pred_index + 1, :].unsqueeze(0).unsqueeze(0)
                    cur_start_boundaries.append(cur_start_boundary)

                    cur_start_boundary = ori_pred_index + 1  # start =  pred_end + 1

                    loop_j = loop_j + 1
                    total_loops = total_loops + 1

            # For each instance in batch
            batch_end_boundaries.append(cur_end_boundaries)
            batch_start_boundaries.append(cur_start_boundaries)
            batch_align_matrix.append(cur_align_matrix)

        batch_end_boundaries = np.array(batch_end_boundaries)
        batch_start_boundaries = np.array(batch_start_boundaries)
        batch_align_matrix = np.array(batch_align_matrix)

        batch_loss = batch_loss / total_loops if batch_y_index is not None else None

        return batch_start_boundaries, batch_end_boundaries, batch_align_matrix, batch_loss

    def forward(self, x_batch, x_lens, y_batch=None):
        encoder_h_n, encoder_h_end = self.pointer_encoder(x_batch, x_lens)
        batch_start_boundaries, batch_end_boundaries, _, batch_loss = self.test_decoder(encoder_h_n, encoder_h_end,
                                                                                        x_lens, y_batch)
        return batch_loss, batch_start_boundaries, batch_end_boundaries


@dataclass
class RSTParsingNetModelOutput(ModelOutput):
    pass


class RSTParsingNetPreTrainedModel(PreTrainedModel):
    config_class = RSTParsingNetConfig
    base_model_prefix = "RSTPointerNetwork"


class RSTParsingNetModel(RSTParsingNetPreTrainedModel):
    pass
