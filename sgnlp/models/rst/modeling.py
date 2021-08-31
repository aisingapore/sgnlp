from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as R
from torch.autograd import Variable
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .config import RSTPointerNetworkConfig, RSTParsingNetConfig
from .modules.encoder_rnn import EncoderRNN
from .modules.decoder_rnn import DecoderRNN
from .modules.pointer_attention import PointerAtten
from .modules.classifier import LabelClassifier
from .modules.type import DiscourseTreeNode, DiscourseTreeSplit
from .utils import get_relation_and_nucleus


@dataclass
class RSTPointerNetworkModelOutput(ModelOutput):
    batch_loss: float = None
    batch_start_boundaries: np.array = None
    batch_end_boundaries: np.array = None


class RSTPointerNetworkPreTrainedModel(PreTrainedModel):
    config_class = RSTPointerNetworkConfig
    base_model_prefix = "RSTPointerNetwork"


class RSTPointerNetworkModel(RSTPointerNetworkPreTrainedModel):
    def __init__(self, config: RSTPointerNetworkConfig):
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
        return RSTPointerNetworkModelOutput(batch_loss, batch_start_boundaries, batch_end_boundaries)


@dataclass
class RSTParsingNetModelOutput(ModelOutput):
    loss_tree_batch: np.array = None
    loss_label_batch: np.array = None
    split_batch: List[List[DiscourseTreeSplit]] = None


class RSTParsingNetPreTrainedModel(PreTrainedModel):
    config_class = RSTParsingNetConfig
    base_model_prefix = "RSTPointerNetwork"


class RSTParsingNetModel(RSTParsingNetPreTrainedModel):
    def __init__(self, config):
        super(RSTParsingNetModel, self).__init__()
        self.batch_size = config.batch_size
        self.word_dim = config.word_dim
        self.hidden_size = config.hidden_size
        self.decoder_input_size = config.decoder_input_size
        self.atten_model = config.atten_model
        self.device = config.device
        self.classifier_input_size = config.classifier_input_size
        self.classifier_hidden_size = config.classifier_hidden_size
        self.highorder = config.highorder
        self.classes_label = config.classes_label
        self.classifier_bias = config.classifier_bias
        self.rnn_layers = config.rnn_layers
        self.encoder = EncoderRNN(
            word_dim=self.word_dim,
            hidden_size=self.hidden_size,
            device=self.device,
            rnn_layers=self.rnn_layers,
            nnDropout=config.dropout_e)
        self.decoder = DecoderRNN(
            input_size=self.decoder_input_size,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=config.dropout_d)
        self.pointer = PointerAtten(
            atten_model=self.atten_model,
            hidden_size=self.hidden_size)
        self.getlabel = LabelClassifier(
            input_size=self.classifier_input_size,
            hidden_size=self.classifier_hidden_size,
            classes_label=self.classes_label,
            bias=self.classifier_bias,
            dropout=config.dropout_c)

    def forward(self, input_sentence, edu_breaks, label_index, parsing_index, generate_splits=True):
        # Obtain encoder outputs and last hidden states
        encoder_outputs, last_hidden_states = self.encoder(input_sentence)

        loss_function = nn.NLLLoss()
        loss_label_batch = 0
        loss_tree_batch = 0
        loop_label_batch = 0
        loop_tree_batch = 0
        cur_label = []
        label_batch = []
        cur_tree = []
        tree_batch = []

        if generate_splits:
            splits_batch = []

        for i in range(len(edu_breaks)):

            cur_label_index = label_index[i]
            cur_label_index = torch.tensor(cur_label_index)
            cur_label_index = cur_label_index.to(self.device)
            cur_parsing_index = parsing_index[i]

            if len(edu_breaks[i]) == 1:
                # For a sentence containing only ONE EDU, it has no
                # corresponding relation label and parsing tree break.
                tree_batch.append([])
                label_batch.append([])

                if generate_splits:
                    splits_batch.append([])

            elif len(edu_breaks[i]) == 2:
                # Take the last hidden state of an EDU as the representation of
                # this EDU. The dimension: [2,hidden_size]
                cur_encoder_outputs = encoder_outputs[i][edu_breaks[i]]

                #  Directly run the classifier to obain predicted label
                input_left = cur_encoder_outputs[0].unsqueeze(0)
                input_right = cur_encoder_outputs[1].unsqueeze(0)
                relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                _, topindex = relation_weights.topk(1)
                label_predict = int(topindex[0][0])
                tree_batch.append([0])
                label_batch.append([label_predict])

                loss_label_batch = loss_label_batch + loss_function(log_relation_weights, cur_label_index)
                loop_label_batch = loop_label_batch + 1

                if generate_splits:
                    nuclearity_left, nuclearity_right, relation_left, relation_right = \
                        get_relation_and_nucleus(label_predict)

                    split = DiscourseTreeSplit(
                        left=DiscourseTreeNode(span=(0, 0), ns_type=nuclearity_left, label=relation_left),
                        right=DiscourseTreeNode(span=(1, 1), ns_type=nuclearity_right, label=relation_right)
                    )

                    splits_batch.append([split])

            else:
                # Take the last hidden state of an EDU as the representation of this EDU
                # The dimension: [NO_EDU,hidden_size]
                cur_encoder_outputs = encoder_outputs[i][edu_breaks[i]]

                edu_index = [x for x in range(len(cur_encoder_outputs))]
                stacks = ['__StackRoot__', edu_index]

                # cur_decoder_input: [1,1,hidden_size]
                # Alternative way is to take the last one as the input. You need to prepare data accordingly for training
                cur_decoder_input = cur_encoder_outputs[0].unsqueeze(0).unsqueeze(0)

                # Obtain last hidden state
                temptest = torch.transpose(last_hidden_states, 0, 1)[i].unsqueeze(0)
                cur_last_hidden_states = torch.transpose(temptest, 0, 1)
                cur_last_hidden_states = cur_last_hidden_states.contiguous()

                cur_decoder_hidden = cur_last_hidden_states
                loop_index = 0

                if generate_splits:
                    splits = []
                if self.highorder:
                    cur_sibling = {}

                while stacks[-1] != '__StackRoot__':
                    stack_head = stacks[-1]

                    if len(stack_head) < 3:

                        # Predict relation label
                        input_left = cur_encoder_outputs[stack_head[0]].unsqueeze(0)
                        input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(0)
                        relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                        _, topindex = relation_weights.topk(1)
                        label_predict = int(topindex[0][0])
                        cur_label.append(label_predict)

                        # For 2 EDU case, we directly point the first EDU
                        # as the current parsing tree break
                        cur_tree.append(stack_head[0])

                        # To keep decoder hidden states consistent
                        _, cur_decoder_hidden = self.decoder(cur_decoder_input, cur_decoder_hidden)

                        # Align ground truth label
                        if loop_index > (len(cur_parsing_index) - 1):
                            cur_label_true = cur_label_index[-1]
                        else:
                            cur_label_true = cur_label_index[loop_index]

                        loss_label_batch = loss_label_batch + loss_function(log_relation_weights,
                                                                            cur_label_true.unsqueeze(0))
                        loop_label_batch = loop_label_batch + 1
                        loop_index = loop_index + 1
                        del stacks[-1]

                        if generate_splits:
                            # To generate a tree structure
                            nuclearity_left, nuclearity_right, relation_left, relation_right = \
                                get_relation_and_nucleus(label_predict)

                            cur_split = DiscourseTreeSplit(
                                left=DiscourseTreeNode(span=(stack_head[0], stack_head[0]), ns_type=nuclearity_left,
                                                       label=relation_left),
                                right=DiscourseTreeNode(span=(stack_head[-1], stack_head[-1]), ns_type=nuclearity_right,
                                                        label=relation_right)
                            )

                            splits.append(cur_split)

                    else:
                        # Length of stack_head >= 3
                        # Alternative way is to take the last one as the input. You need to prepare data accordingly for training
                        cur_decoder_input = cur_encoder_outputs[stack_head[0]].unsqueeze(0).unsqueeze(0)

                        if self.highorder:
                            if loop_index != 0:
                                # Incoperate Parents information
                                cur_decoder_input_P = cur_encoder_outputs[stack_head[-1]]
                                # To incorperate Sibling information
                                if str(stack_head) in cur_sibling.keys():
                                    cur_decoder_input_S = cur_encoder_outputs[cur_sibling[str(stack_head)]]

                                    inputs_all = torch.cat(
                                        (cur_decoder_input.squeeze(0), cur_decoder_input_S.unsqueeze(0), \
                                         cur_decoder_input_P.unsqueeze(0)), 0)
                                    new_inputs_all = torch.matmul(
                                        F.softmax(torch.matmul(inputs_all, inputs_all.transpose(0, 1)), 0), inputs_all)
                                    cur_decoder_input = new_inputs_all[0, :] + new_inputs_all[1, :] + new_inputs_all[2,
                                                                                                      :]
                                    cur_decoder_input = cur_decoder_input.unsqueeze(0).unsqueeze(0)

                                    # cur_decoder_input = cur_decoder_input + cur_decoder_input_P + cur_decoder_input_S
                                else:
                                    inputs_all = torch.cat(
                                        (cur_decoder_input.squeeze(0), cur_decoder_input_P.unsqueeze(0)), 0)
                                    new_inputs_all = torch.matmul(
                                        F.softmax(torch.matmul(inputs_all, inputs_all.transpose(0, 1)), 0), inputs_all)
                                    cur_decoder_input = new_inputs_all[0, :] + new_inputs_all[1, :]
                                    cur_decoder_input = cur_decoder_input.unsqueeze(0).unsqueeze(0)

                        # Predict the parsing tree break
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, cur_decoder_hidden)
                        atten_weights, log_atten_weights = self.pointer(cur_encoder_outputs[stack_head[:-1]],
                                                                        cur_decoder_output.squeeze(0).squeeze(0))
                        _, topindex_tree = atten_weights.topk(1)
                        tree_predict = int(topindex_tree[0][0]) + stack_head[0]
                        cur_tree.append(tree_predict)

                        # Predict the Label
                        input_left = cur_encoder_outputs[tree_predict].unsqueeze(0)
                        input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(0)
                        relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                        _, topindex_label = relation_weights.topk(1)
                        label_predict = int(topindex_label[0][0])
                        cur_label.append(label_predict)

                        # Align ground true label and tree
                        if loop_index > (len(cur_parsing_index) - 1):
                            cur_label_true = cur_label_index[-1]
                            cur_tree_true = cur_parsing_index[-1]
                        else:
                            cur_label_true = cur_label_index[loop_index]
                            cur_tree_true = cur_parsing_index[loop_index]

                        temp_ground = max(0, (int(cur_tree_true) - int(stack_head[0])))
                        if temp_ground >= (len(stack_head) - 1):
                            temp_ground = stack_head[-2] - stack_head[0]
                        # Compute Tree Loss
                        cur_ground_index = torch.tensor([temp_ground])
                        cur_ground_index = cur_ground_index.to(self.device)
                        loss_tree_batch = loss_tree_batch + loss_function(log_atten_weights, cur_ground_index)

                        # Compute Classifier Loss
                        loss_label_batch = loss_label_batch + loss_function(log_relation_weights,
                                                                            cur_label_true.unsqueeze(0))

                        # Stacks stuff
                        stack_down = stack_head[(tree_predict - stack_head[0] + 1):]
                        stack_top = stack_head[:(tree_predict - stack_head[0] + 1)]
                        del stacks[-1]
                        loop_label_batch = loop_label_batch + 1
                        loop_tree_batch = loop_tree_batch + 1
                        loop_index = loop_index + 1

                        # Sibling information
                        if self.highorder:
                            if len(stack_down) > 2:
                                cur_sibling.update({str(stack_down): stack_top[-1]})

                        # Remove ONE-EDU part
                        if len(stack_down) > 1:
                            stacks.append(stack_down)
                        if len(stack_top) > 1:
                            stacks.append(stack_top)

                        if generate_splits:
                            nuclearity_left, nuclearity_right, relation_left, relation_right = \
                                get_relation_and_nucleus(label_predict)

                            cur_split = DiscourseTreeSplit(
                                left=DiscourseTreeNode(span=(stack_head[0], tree_predict), ns_type=nuclearity_left,
                                                       label=relation_left),
                                right=DiscourseTreeNode(span=(tree_predict + 1, stack_head[-1]),
                                                        ns_type=nuclearity_right, label=relation_right)
                            )

                            splits.append(cur_split)

                tree_batch.append(cur_tree)
                label_batch.append(cur_label)
                if generate_splits:
                    splits_batch.append(splits)

        if loop_label_batch != 0:
            loss_label_batch = loss_label_batch / loop_label_batch
            loss_label_batch = loss_label_batch.detach().cpu().numpy()

        if loss_tree_batch != 0:
            loss_tree_batch = loss_tree_batch / loop_tree_batch
            loss_tree_batch = loss_tree_batch.detach().cpu().numpy()

        return RSTParsingNetModelOutput(loss_tree_batch, loss_label_batch, (splits_batch if generate_splits else None))
