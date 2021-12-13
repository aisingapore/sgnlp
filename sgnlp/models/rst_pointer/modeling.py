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

from .config import RstPointerSegmenterConfig, RstPointerParserConfig
from .modules.classifier import LabelClassifier
from .modules.decoder_rnn import DecoderRNN
from .modules.elmo import initialize_elmo
from .modules.encoder_rnn import EncoderRNN
from .modules.pointer_attention import PointerAtten
from .modules.type import DiscourseTreeNode, DiscourseTreeSplit
from .utils import get_relation_and_nucleus


@dataclass
class RstPointerSegmenterModelOutput(ModelOutput):
    loss: float = None
    start_boundaries: np.array = None
    end_boundaries: np.array = None


class RstPointerSegmenterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RstPointerSegmenterConfig
    base_model_prefix = "rst_pointer_segmenter"

    def _init_weights(self, module):
        pass


class RstPointerSegmenterModel(RstPointerSegmenterPreTrainedModel):
    """This model performs discourse segmentation.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Args:
        config (:class:`~sgnlp.models.rst_pointer.RstPointerSegmenterConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.

    Example::

        from sgnlp.models.rst_pointer import RstPointerSegmenterConfig, RstPointerSegmenterModel

        # Method 1: Loading a default model
        segmenter_config = RstPointerSegmenterConfig()
        segmenter = RstPointerSegmenterModel(segmenter_config)

        # Method 2: Loading from pretrained
        segmenter_config = RstPointerSegmenterConfig.from_pretrained(
            'https://storage.googleapis.com/sgnlp/models/rst_pointer/segmenter/config.json')
        segmenter = RstPointerSegmenterModel.from_pretrained(
            'https://storage.googleapis.com/sgnlp/models/rst_pointer/segmenter/pytorch_model.bin',
            config=segmenter_config)

    """

    def __init__(self, config: RstPointerSegmenterConfig):
        super().__init__(config)

        self.word_dim = config.word_dim
        self.hidden_dim = config.hidden_dim
        self.dropout_prob = config.dropout_prob
        self.use_bilstm = config.use_bilstm
        self.num_rnn_layers = config.num_rnn_layers
        self.rnn_type = config.rnn_type
        self.is_batch_norm = config.is_batch_norm

        self.dropout = nn.Dropout(config.dropout_prob)
        self.embedding, self.word_dim = initialize_elmo(config.elmo_size)

        if self.rnn_type in ["LSTM", "GRU"]:
            self.decoder_rnn = getattr(nn, self.rnn_type)(
                input_size=2 * self.hidden_dim if self.use_bilstm else self.hidden_dim,
                hidden_size=2 * self.hidden_dim if self.use_bilstm else self.hidden_dim,
                num_layers=self.num_rnn_layers,
                dropout=self.dropout_prob,
                batch_first=True,
            )

            self.encoder_rnn = getattr(nn, self.rnn_type)(
                input_size=self.word_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_rnn_layers,
                bidirectional=self.use_bilstm,
                dropout=self.dropout_prob,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type should be one of ['LSTM', 'GRU'].")

        if self.use_bilstm:
            self.num_encoder_bi = 2
        else:
            self.num_encoder_bi = 1

    def init_hidden(self, hsize, batchsize):
        if self.rnn_type == "LSTM":
            h_0 = Variable(
                torch.zeros(self.num_encoder_bi * self.num_rnn_layers, batchsize, hsize)
            ).to(self.device)
            c_0 = Variable(
                torch.zeros(self.num_encoder_bi * self.num_rnn_layers, batchsize, hsize)
            ).to(self.device)

            return h_0, c_0
        else:
            h_0 = Variable(
                torch.zeros(self.num_encoder_bi * self.num_rnn_layers, batchsize, hsize)
            ).to(self.device)

            return h_0

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        # Sort first if ONNX exportability is needed
        x_packed = R.pack_padded_sequence(
            x, x_lens, batch_first=True, enforce_sorted=False
        )

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h

    def pointer_encoder(self, sentences_ids, sentences_lens):
        batch_norm = nn.BatchNorm1d(
            self.word_dim, affine=False, track_running_stats=False
        )

        batch_size = len(sentences_ids)

        embeddings = self.embedding(sentences_ids)
        batch_x_elmo = embeddings["elmo_representations"][
            0
        ]  # two layers output  [batch,length,d_elmo]

        x = batch_x_elmo
        if self.is_batch_norm:
            x = x.permute(0, 2, 1)  # N C L
            x = batch_norm(x)
            x = x.permute(0, 2, 1)  # N L C

        x = self.dropout(x)

        encoder_lstm_co_h_o = self.init_hidden(self.hidden_dim, batch_size)
        output_encoder, hidden_states_encoder = self._run_rnn_packed(
            self.encoder_rnn, x, sentences_lens, encoder_lstm_co_h_o
        )  # batch_first=True
        output_encoder = output_encoder.contiguous()
        output_encoder = self.dropout(output_encoder)

        return output_encoder, hidden_states_encoder

    def pointer_layer(self, encoder_states, cur_decoder_state):
        # we use simple dot product attention to computer pointer
        attention_pointer = torch.matmul(encoder_states, cur_decoder_state).unsqueeze(1)
        attention_pointer = attention_pointer.permute(1, 0)

        att_weights = F.softmax(attention_pointer, dim=1)
        logits = F.log_softmax(attention_pointer, dim=1)

        return logits, att_weights

    def decoder(self, h_n, h_end, batch_x_lens, batch_y=None):
        """
        Args:
            h_n: all hidden states
            h_end: final hidden state
            batch_x_lens: lengths of x (i.e. number of tokens)
            batch_y: optional. provide to get loss metric.

        Returns:
            A tuple containing the following values:
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
        if batch_y is not None:
            loss_function = nn.NLLLoss()
            batch_loss = 0
        else:
            batch_loss = None

        for i in range(batch_size):
            cur_len = batch_x_lens[i]
            cur_encoder_hn = h_n[i, 0:cur_len, :]  # length * encoder_hidden_size
            cur_end_boundary = cur_len - 1  # end boundary is index of last token

            cur_y_index = batch_y[i] if batch_y is not None else None

            cur_end_boundaries = []
            cur_start_boundaries = []
            cur_align_matrix = []

            if self.rnn_type == "LSTM":  # need h_end,c_end
                h_end = (
                    h_end[0]
                    .permute(1, 0, 2)
                    .contiguous()
                    .view(batch_size, self.num_rnn_layers, -1)
                )
                c_end = (
                    h_end[1]
                    .permute(1, 0, 2)
                    .contiguous()
                    .view(batch_size, self.num_rnn_layers, -1)
                )

                cur_h0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                cur_c0 = c_end[i].unsqueeze(0).permute(1, 0, 2)

                h_pass = (cur_h0, cur_c0)
            else:  # only need h_end
                h_end = (
                    h_end.permute(1, 0, 2)
                    .contiguous()
                    .view(batch_size, self.num_rnn_layers, -1)
                )
                cur_h0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = cur_h0

            loop_hc = h_pass
            loop_in = (
                cur_encoder_hn[0, :].unsqueeze(0).unsqueeze(0)
            )  # [ 1, 1, encoder_hidden_size] (first start)

            cur_start_boundary = 0
            loop_j = 0

            while True:
                loop_o, loop_hc = self.decoder_rnn(loop_in, loop_hc)

                predict_range = list(range(cur_start_boundary, cur_len))
                cur_encoder_hn_back = cur_encoder_hn[predict_range, :]
                cur_logits, cur_weights = self.pointer_layer(
                    cur_encoder_hn_back, loop_o.squeeze(0).squeeze(0)
                )

                cur_align_vector = np.zeros(cur_len)
                cur_align_vector[predict_range] = cur_weights.data.cpu().numpy()[0]
                cur_align_matrix.append(cur_align_vector)

                _, top_i = cur_logits.data.topk(1)
                pred_index = top_i[0][0].item()
                ori_pred_index = pred_index + cur_start_boundary

                # Calculate loss
                if batch_y is not None:
                    if loop_j > len(cur_y_index) - 1:
                        cur_ground_y = cur_y_index[-1]
                    else:
                        cur_ground_y = cur_y_index[loop_j]

                    cur_ground_y_var = Variable(
                        torch.LongTensor(
                            [max(0, int(cur_ground_y) - cur_start_boundary)]
                        )
                    ).to(self.device)

                    batch_loss += loss_function(cur_logits, cur_ground_y_var)

                if cur_end_boundary <= ori_pred_index:
                    cur_end_boundaries.append(cur_end_boundary)
                    cur_start_boundaries.append(cur_start_boundary)
                    total_loops = total_loops + 1
                    break
                else:
                    cur_end_boundaries.append(ori_pred_index)

                    loop_in = (
                        cur_encoder_hn[ori_pred_index + 1, :].unsqueeze(0).unsqueeze(0)
                    )
                    cur_start_boundaries.append(cur_start_boundary)

                    cur_start_boundary = ori_pred_index + 1  # start =  pred_end + 1

                    loop_j = loop_j + 1
                    total_loops = total_loops + 1

            # For each instance in batch
            batch_end_boundaries.append(cur_end_boundaries)
            batch_start_boundaries.append(cur_start_boundaries)
            batch_align_matrix.append(cur_align_matrix)

        batch_loss = batch_loss / total_loops if batch_y is not None else None

        return (
            batch_start_boundaries,
            batch_end_boundaries,
            batch_align_matrix,
            batch_loss,
        )

    def forward(self, tokenized_sentence_ids, sentence_lens, labels=None):
        """
        Args:
            tokenized_sentence_ids: Token IDs.
            sentence_lens: Sentence lengths.
            labels: Optional. Provide if loss is needed.

        Returns:
            output (:class:`~sgnlp.models.rst_pointer.modeling.RstPointerSegmenterModelOutput`)
        """
        encoder_h_n, encoder_h_end = self.pointer_encoder(
            tokenized_sentence_ids, sentence_lens
        )
        start_boundaries, end_boundaries, _, loss = self.decoder(
            encoder_h_n, encoder_h_end, sentence_lens, labels
        )
        return RstPointerSegmenterModelOutput(loss, start_boundaries, end_boundaries)


@dataclass
class RstPointerParserModelOutput(ModelOutput):
    loss_tree: np.array = None
    loss_label: np.array = None
    splits: List[List[DiscourseTreeSplit]] = None


class RstPointerParserPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RstPointerParserConfig
    base_model_prefix = "rst_pointer_parser"

    def _init_weights(self, module):
        pass


class RstPointerParserModel(RstPointerParserPreTrainedModel):
    """This model performs discourse parsing.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Args:
        config (:class:`~sgnlp.models.rst_pointer.RstPointerParserConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.

    Example::

        from sgnlp.models.rst_pointer import RstPointerParserConfig, RstPointerParserModel

        # Method 1: Loading a default model
        parser_config = RstPointerParserConfig()
        parser = RstPointerParserModel(parser_config)

        # Method 2: Loading from pretrained
        parser_config = RstPointerParserConfig.from_pretrained(
            'https://storage.googleapis.com/sgnlp/models/rst_pointer/parser/config.json')
        parser = RstPointerParserModel.from_pretrained(
            'https://storage.googleapis.com/sgnlp/models/rst_pointer/parser/pytorch_model.bin',
            config=parser_config)

    """

    def __init__(self, config: RstPointerParserConfig):
        super().__init__(config)
        self.word_dim = config.word_dim
        self.hidden_size = config.hidden_size
        self.decoder_input_size = config.decoder_input_size
        self.atten_model = config.atten_model
        self.classifier_input_size = config.classifier_input_size
        self.classifier_hidden_size = config.classifier_hidden_size
        self.highorder = config.highorder
        self.classes_label = config.classes_label
        self.classifier_bias = config.classifier_bias
        self.rnn_layers = config.rnn_layers

        self.embedding, self.word_dim = initialize_elmo(config.elmo_size)

        self.encoder = EncoderRNN(
            word_dim=self.word_dim,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=config.dropout_e,
        )
        self.decoder = DecoderRNN(
            input_size=self.decoder_input_size,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=config.dropout_d,
        )
        self.pointer = PointerAtten(
            atten_model=self.atten_model, hidden_size=self.hidden_size
        )
        self.classifier = LabelClassifier(
            input_size=self.classifier_input_size,
            classifier_hidden_size=self.classifier_hidden_size,
            classes_label=self.classes_label,
            bias=self.classifier_bias,
            dropout=config.dropout_c,
        )

    def forward(
        self,
        input_sentence_ids,
        edu_breaks,
        sentence_lengths,
        label_index=None,
        parsing_index=None,
        generate_splits=True,
    ):
        """
        Args:
            input_sentence_ids: Input sentence IDs.
            edu_breaks: Token positions of edu breaks.
            sentence_lengths: Lengths of sentences.
            label_index: Label IDs. Needed only if loss needs to be computed.
            parsing_index: Parsing IDs. Needed only if loss needs to be computed.
            generate_splits: Whether to return splits.

        Returns:
            output (:class:`~sgnlp.models.rst_pointer.modeling.RstPointerParserModelOutput`)
        """
        # Obtain encoder outputs and last hidden states
        embeddings = self.embedding(input_sentence_ids)
        encoder_outputs, last_hidden_states = self.encoder(embeddings, sentence_lengths)

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

        calculate_loss = True if (label_index and parsing_index) else False

        for i in range(len(edu_breaks)):

            if calculate_loss:
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
                relation_weights, log_relation_weights = self.classifier(
                    input_left, input_right
                )
                _, topindex = relation_weights.topk(1)
                label_predict = int(topindex[0][0])
                tree_batch.append([0])
                label_batch.append([label_predict])

                if calculate_loss:
                    loss_label_batch = loss_label_batch + loss_function(
                        log_relation_weights, cur_label_index
                    )
                    loop_label_batch = loop_label_batch + 1

                if generate_splits:
                    (
                        nuclearity_left,
                        nuclearity_right,
                        relation_left,
                        relation_right,
                    ) = get_relation_and_nucleus(label_predict)

                    split = DiscourseTreeSplit(
                        left=DiscourseTreeNode(
                            span=(0, 0), ns_type=nuclearity_left, label=relation_left
                        ),
                        right=DiscourseTreeNode(
                            span=(1, 1), ns_type=nuclearity_right, label=relation_right
                        ),
                    )

                    splits_batch.append([split])

            else:
                # Take the last hidden state of an EDU as the representation of this EDU
                # The dimension: [NO_EDU,hidden_size]
                cur_encoder_outputs = encoder_outputs[i][edu_breaks[i]]

                edu_index = [x for x in range(len(cur_encoder_outputs))]
                stacks = ["__StackRoot__", edu_index]

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

                while stacks[-1] != "__StackRoot__":
                    stack_head = stacks[-1]

                    if len(stack_head) < 3:

                        # Predict relation label
                        input_left = cur_encoder_outputs[stack_head[0]].unsqueeze(0)
                        input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(0)
                        relation_weights, log_relation_weights = self.classifier(
                            input_left, input_right
                        )
                        _, topindex = relation_weights.topk(1)
                        label_predict = int(topindex[0][0])
                        cur_label.append(label_predict)

                        # For 2 EDU case, we directly point the first EDU
                        # as the current parsing tree break
                        cur_tree.append(stack_head[0])

                        # To keep decoder hidden states consistent
                        _, cur_decoder_hidden = self.decoder(
                            cur_decoder_input, cur_decoder_hidden
                        )

                        # Align ground truth label
                        if calculate_loss:
                            if loop_index > (len(cur_parsing_index) - 1):
                                cur_label_true = cur_label_index[-1]
                            else:
                                cur_label_true = cur_label_index[loop_index]

                            loss_label_batch = loss_label_batch + loss_function(
                                log_relation_weights, cur_label_true.unsqueeze(0)
                            )
                            loop_label_batch = loop_label_batch + 1
                        loop_index = loop_index + 1
                        del stacks[-1]

                        if generate_splits:
                            # To generate a tree structure
                            (
                                nuclearity_left,
                                nuclearity_right,
                                relation_left,
                                relation_right,
                            ) = get_relation_and_nucleus(label_predict)

                            cur_split = DiscourseTreeSplit(
                                left=DiscourseTreeNode(
                                    span=(stack_head[0], stack_head[0]),
                                    ns_type=nuclearity_left,
                                    label=relation_left,
                                ),
                                right=DiscourseTreeNode(
                                    span=(stack_head[-1], stack_head[-1]),
                                    ns_type=nuclearity_right,
                                    label=relation_right,
                                ),
                            )

                            splits.append(cur_split)

                    else:
                        # Length of stack_head >= 3
                        # Alternative way is to take the last one as the input. You need to prepare data accordingly for training
                        cur_decoder_input = (
                            cur_encoder_outputs[stack_head[0]].unsqueeze(0).unsqueeze(0)
                        )

                        if self.highorder:
                            if loop_index != 0:
                                # Incorporate Parents information
                                cur_decoder_input_P = cur_encoder_outputs[
                                    stack_head[-1]
                                ]
                                # To incorporate Sibling information
                                if str(stack_head) in cur_sibling.keys():
                                    cur_decoder_input_S = cur_encoder_outputs[
                                        cur_sibling[str(stack_head)]
                                    ]

                                    inputs_all = torch.cat(
                                        (
                                            cur_decoder_input.squeeze(0),
                                            cur_decoder_input_S.unsqueeze(0),
                                            cur_decoder_input_P.unsqueeze(0),
                                        ),
                                        0,
                                    )
                                    new_inputs_all = torch.matmul(
                                        F.softmax(
                                            torch.matmul(
                                                inputs_all, inputs_all.transpose(0, 1)
                                            ),
                                            0,
                                        ),
                                        inputs_all,
                                    )
                                    cur_decoder_input = (
                                        new_inputs_all[0, :]
                                        + new_inputs_all[1, :]
                                        + new_inputs_all[2, :]
                                    )
                                    cur_decoder_input = cur_decoder_input.unsqueeze(
                                        0
                                    ).unsqueeze(0)

                                    # cur_decoder_input = cur_decoder_input + cur_decoder_input_P + cur_decoder_input_S
                                else:
                                    inputs_all = torch.cat(
                                        (
                                            cur_decoder_input.squeeze(0),
                                            cur_decoder_input_P.unsqueeze(0),
                                        ),
                                        0,
                                    )
                                    new_inputs_all = torch.matmul(
                                        F.softmax(
                                            torch.matmul(
                                                inputs_all, inputs_all.transpose(0, 1)
                                            ),
                                            0,
                                        ),
                                        inputs_all,
                                    )
                                    cur_decoder_input = (
                                        new_inputs_all[0, :] + new_inputs_all[1, :]
                                    )
                                    cur_decoder_input = cur_decoder_input.unsqueeze(
                                        0
                                    ).unsqueeze(0)

                        # Predict the parsing tree break
                        cur_decoder_output, cur_decoder_hidden = self.decoder(
                            cur_decoder_input, cur_decoder_hidden
                        )
                        atten_weights, log_atten_weights = self.pointer(
                            cur_encoder_outputs[stack_head[:-1]],
                            cur_decoder_output.squeeze(0).squeeze(0),
                        )
                        _, topindex_tree = atten_weights.topk(1)
                        tree_predict = int(topindex_tree[0][0]) + stack_head[0]
                        cur_tree.append(tree_predict)

                        # Predict the Label
                        input_left = cur_encoder_outputs[tree_predict].unsqueeze(0)
                        input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(0)
                        relation_weights, log_relation_weights = self.classifier(
                            input_left, input_right
                        )
                        _, topindex_label = relation_weights.topk(1)
                        label_predict = int(topindex_label[0][0])
                        cur_label.append(label_predict)

                        # Align ground true label and tree
                        if calculate_loss:
                            if loop_index > (len(cur_parsing_index) - 1):
                                cur_label_true = cur_label_index[-1]
                                cur_tree_true = cur_parsing_index[-1]
                            else:
                                cur_label_true = cur_label_index[loop_index]
                                cur_tree_true = cur_parsing_index[loop_index]

                            temp_ground = max(
                                0, (int(cur_tree_true) - int(stack_head[0]))
                            )
                            if temp_ground >= (len(stack_head) - 1):
                                temp_ground = stack_head[-2] - stack_head[0]
                            # Compute Tree Loss
                            cur_ground_index = torch.tensor([temp_ground])
                            cur_ground_index = cur_ground_index.to(self.device)
                            loss_tree_batch = loss_tree_batch + loss_function(
                                log_atten_weights, cur_ground_index
                            )

                            # Compute Classifier Loss
                            loss_label_batch = loss_label_batch + loss_function(
                                log_relation_weights, cur_label_true.unsqueeze(0)
                            )
                            loop_label_batch = loop_label_batch + 1
                            loop_tree_batch = loop_tree_batch + 1

                        # Stacks stuff
                        stack_down = stack_head[(tree_predict - stack_head[0] + 1) :]
                        stack_top = stack_head[: (tree_predict - stack_head[0] + 1)]
                        del stacks[-1]
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
                            (
                                nuclearity_left,
                                nuclearity_right,
                                relation_left,
                                relation_right,
                            ) = get_relation_and_nucleus(label_predict)

                            cur_split = DiscourseTreeSplit(
                                left=DiscourseTreeNode(
                                    span=(stack_head[0], tree_predict),
                                    ns_type=nuclearity_left,
                                    label=relation_left,
                                ),
                                right=DiscourseTreeNode(
                                    span=(tree_predict + 1, stack_head[-1]),
                                    ns_type=nuclearity_right,
                                    label=relation_right,
                                ),
                            )

                            splits.append(cur_split)

                tree_batch.append(cur_tree)
                label_batch.append(cur_label)
                if generate_splits:
                    splits_batch.append(splits)

        if calculate_loss:
            if loop_label_batch != 0:
                loss_label_batch = loss_label_batch / loop_label_batch
                loss_label_batch = loss_label_batch.detach().cpu().numpy()

            if loss_tree_batch != 0:
                loss_tree_batch = loss_tree_batch / loop_tree_batch
                loss_tree_batch = loss_tree_batch.detach().cpu().numpy()
        else:
            loss_tree_batch = None
            loss_label_batch = None

        return RstPointerParserModelOutput(
            loss_tree_batch,
            loss_label_batch,
            (splits_batch if generate_splits else None),
        )

    def forward_train(
        self,
        input_sentence_ids_batch,
        edu_breaks_batch,
        label_index_batch,
        parsing_index_batch,
        decoder_input_index_batch,
        parents_index_batch,
        sibling_index_batch,
        sentence_lengths,
    ):
        # TODO: This function should ideally be combined with the forward function.
        #   There are significant overlap in code, but also some significant differences in the logic
        #   which makes refactoring them difficult.
        #   To retain the original code's fidelity, this function is used for the training forward pass.

        # Obtain encoder outputs and last hidden states
        embeddings = self.embedding(input_sentence_ids_batch)
        encoder_outputs, last_hiddenstates = self.encoder(embeddings, sentence_lengths)
        loss_function = nn.NLLLoss()
        loss_label_batch = 0
        loss_tree_batch = 0
        loop_label_batch = 0
        loop_tree_batch = 0

        for i in range(input_sentence_ids_batch.shape[0]):
            cur_label_index = label_index_batch[i]
            cur_label_index = torch.tensor(cur_label_index)
            cur_label_index = cur_label_index.to(self.device)
            cur_parsing_index = parsing_index_batch[i]
            cur_decoder_input_index = decoder_input_index_batch[i]
            cur_parents_index = parents_index_batch[i]
            cur_sibling_index = sibling_index_batch[i]

            if len(edu_breaks_batch[i]) == 1:
                continue

            elif len(edu_breaks_batch[i]) == 2:
                # Take the last hidden state of an EDU as the representation of
                # this EDU. The dimension: [2,hidden_size]
                cur_encoder_outputs = encoder_outputs[i][edu_breaks_batch[i]]

                # Use the last hidden state of a span to predict the relation
                # beween these two span.
                input_left = cur_encoder_outputs[0].unsqueeze(0)
                input_right = cur_encoder_outputs[1].unsqueeze(0)
                _, log_relation_weights = self.classifier(input_left, input_right)

                loss_label_batch = loss_label_batch + loss_function(
                    log_relation_weights, cur_label_index
                )
                loop_label_batch = loop_label_batch + 1

            else:
                # Take the last hidden state of an EDU as the representation of this EDU
                # The dimension: [NO_EDU,hidden_size]
                cur_encoder_outputs = encoder_outputs[i][edu_breaks_batch[i]].to(
                    self.device
                )

                # Obtain last hidden state of encoder
                temp = torch.transpose(last_hiddenstates, 0, 1)[i].unsqueeze(0)
                cur_last_hiddenstates = torch.transpose(temp, 0, 1)
                cur_last_hiddenstates = cur_last_hiddenstates.contiguous()

                if self.highorder:
                    # Incorporate parents information
                    cur_decoder_inputs_P = cur_encoder_outputs[cur_parents_index]
                    cur_decoder_inputs_P[0] = 0

                    # Incorporate sibling information
                    cur_decoder_inputs_S = torch.zeros(
                        [len(cur_sibling_index), cur_encoder_outputs.shape[1]]
                    ).to(self.device)
                    for n, s_idx in enumerate(cur_sibling_index):
                        if s_idx != 99:
                            cur_decoder_inputs_S[n] = cur_encoder_outputs[s_idx]

                    # Original input
                    cur_decoder_inputs = cur_encoder_outputs[cur_decoder_input_index]

                    # One-layer self attention
                    inputs_all = torch.cat(
                        (
                            cur_decoder_inputs.unsqueeze(0).transpose(0, 1),
                            cur_decoder_inputs_S.unsqueeze(0).transpose(0, 1),
                            cur_decoder_inputs_P.unsqueeze(0).transpose(0, 1),
                        ),
                        1,
                    )
                    new_inputs_all = torch.matmul(
                        F.softmax(
                            torch.matmul(inputs_all, inputs_all.transpose(1, 2)), 1
                        ),
                        inputs_all,
                    )
                    cur_decoder_inputs = (
                        new_inputs_all[:, 0, :]
                        + new_inputs_all[:, 1, :]
                        + new_inputs_all[:, 2, :]
                    )

                else:
                    cur_decoder_inputs = cur_encoder_outputs[cur_decoder_input_index]

                # Obtain decoder outputs
                cur_decoder_outputs, _ = self.decoder(
                    cur_decoder_inputs.unsqueeze(0), cur_last_hiddenstates
                )
                cur_decoder_outputs = cur_decoder_outputs.squeeze(0)

                edu_index = [x for x in range(len(cur_encoder_outputs))]
                stacks = ["__StackRoot__", edu_index]

                for j in range(len(cur_decoder_outputs)):

                    if stacks[-1] != "__StackRoot__":
                        stack_head = stacks[-1]

                        if len(stack_head) < 3:

                            # We remove this from stacks after compute the
                            # relation between these two EDUS

                            # Compute Classifier Loss
                            input_left = cur_encoder_outputs[
                                cur_parsing_index[j]
                            ].unsqueeze(0)
                            input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(
                                0
                            )
                            _, log_relation_weights = self.classifier(
                                input_left, input_right
                            )

                            loss_label_batch = loss_label_batch + loss_function(
                                log_relation_weights, cur_label_index[j].unsqueeze(0)
                            )

                            del stacks[-1]
                            loop_label_batch = loop_label_batch + 1

                        else:  # Length of stack_head >= 3
                            # Compute Tree Loss
                            # We don't attend to the last EDU of a span to be parsed
                            _, log_atten_weights = self.pointer(
                                cur_encoder_outputs[stack_head[:-1]],
                                cur_decoder_outputs[j],
                            )
                            cur_ground_index = torch.tensor(
                                [int(cur_parsing_index[j]) - int(stack_head[0])]
                            )
                            cur_ground_index = cur_ground_index.to(self.device)
                            loss_tree_batch = loss_tree_batch + loss_function(
                                log_atten_weights, cur_ground_index
                            )

                            # Compute Classifier Loss
                            input_left = cur_encoder_outputs[
                                cur_parsing_index[j]
                            ].unsqueeze(0)
                            input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(
                                0
                            )
                            _, log_relation_weights = self.classifier(
                                input_left, input_right
                            )

                            loss_label_batch = loss_label_batch + loss_function(
                                log_relation_weights, cur_label_index[j].unsqueeze(0)
                            )

                            # Stacks stuff
                            stack_down = stack_head[
                                (cur_parsing_index[j] - stack_head[0] + 1) :
                            ]
                            stack_top = stack_head[
                                : (cur_parsing_index[j] - stack_head[0] + 1)
                            ]
                            del stacks[-1]
                            loop_label_batch = loop_label_batch + 1
                            loop_tree_batch = loop_tree_batch + 1

                            # Remove ONE-EDU part, TWO-EDU span will be removed after classifier in next step
                            if len(stack_down) > 1:
                                stacks.append(stack_down)
                            if len(stack_top) > 1:
                                stacks.append(stack_top)

        if loop_label_batch != 0:
            loss_label_batch = loss_label_batch / loop_label_batch

        if loss_tree_batch != 0:
            loss_tree_batch = loss_tree_batch / loop_tree_batch

        return loss_tree_batch, loss_label_batch
