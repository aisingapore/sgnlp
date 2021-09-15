import os
import copy
import pickle
import torch
import random
import logging
import numpy as np
from typing import List

from sgnlp.models.rst.preprocess import RSTPreprocessor
from .modeling import RstPointerParserModel, RstPointerParserConfig
from .utils import parse_args_and_load_config
from .data_class import RstPointerParserTrainArgs, RstPointerSegmenterTrainArgs
from .modules.type import DiscourseTreeNode, DiscourseTreeSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup(seed):
    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_span_dict(discourse_tree_splits: List[DiscourseTreeSplit]):
    span_dict = {}

    for split in discourse_tree_splits:
        left_span_key, left_node_value = get_span_key_and_node_value(split.left)
        span_dict[left_span_key] = left_node_value
        right_span_key, right_node_value = get_span_key_and_node_value(split.right)
        span_dict[right_span_key] = right_node_value

    return span_dict


def get_span_key_and_node_value(node: DiscourseTreeNode):
    span_key = f'{node.span[0]}-{node.span[1]}'
    node_value = [node.label, node.ns_type]
    return span_key, node_value


def get_measurement(discourse_tree_splits_1, discourse_tree_splits_2):
    span_dict1 = get_span_dict(discourse_tree_splits_1)
    span_dict2 = get_span_dict(discourse_tree_splits_2)
    num_correct_relations = 0
    num_correct_nuclearity = 0
    num_spans_1 = len(span_dict1)
    num_spans_2 = len(span_dict2)

    # no of right spans
    matching_spans = list(set(span_dict1.keys()).intersection(set(span_dict2.keys())))
    num_matching_spans = len(matching_spans)

    # count matching relations and nuclearity
    for span in matching_spans:
        if span_dict1[span][0] == span_dict2[span][0]:
            num_correct_relations += 1
        if span_dict1[span][1] == span_dict2[span][1]:
            num_correct_nuclearity += 1

    return num_matching_spans, num_correct_relations, num_correct_nuclearity, num_spans_1, num_spans_2


def get_batch_measure(input_splits_batch, golden_metric_batch):
    num_matching_spans = 0
    num_correct_relations = 0
    num_correct_nuclearity = 0
    num_spans_input = 0
    num_spans_golden = 0

    for input_splits, golden_splits in zip(input_splits_batch, golden_metric_batch):
        if input_splits and golden_splits:
            # if both splits have values in the list
            _num_matching_spans, _num_correct_relations, _num_correct_nuclearity, _num_spans_input, _num_spans_golden \
                = get_measurement(input_splits, golden_splits)

            num_matching_spans += _num_matching_spans
            num_correct_relations += _num_correct_relations
            num_correct_nuclearity += _num_correct_nuclearity
            num_spans_input += _num_spans_input
            num_spans_golden += _num_spans_golden

        elif input_splits and not golden_splits:
            # each split has 2 spans
            num_spans_input += len(input_splits) * 2

        elif not input_splits and golden_splits:
            num_spans_golden += len(golden_splits) * 2

    return num_matching_spans, num_correct_relations, num_correct_nuclearity, num_spans_input, num_spans_golden


def get_micro_measure(correct_span, correct_relation, correct_nuclearity, no_system, no_golden):
    # Compute Micro-average measure
    # Span
    precision_span = correct_span / no_system
    recall_span = correct_span / no_golden
    f1_span = (2 * correct_span) / (no_golden + no_system)

    # Relation
    precision_relation = correct_relation / no_system
    recall_relation = correct_relation / no_golden
    f1_relation = (2 * correct_relation) / (no_golden + no_system)

    # Nuclearity
    precision_nuclearity = correct_nuclearity / no_system
    recall_nuclearity = correct_nuclearity / no_golden
    f1_nuclearity = (2 * correct_nuclearity) / (no_golden + no_system)

    return (precision_span, recall_span, f1_span), (precision_relation, recall_relation, f1_relation), \
           (precision_nuclearity, recall_nuclearity, f1_nuclearity)


def get_batch_data_training(input_sentences, edu_breaks, decoder_input, relation_label,
                            parsing_breaks, golden_metric, parents_index, sibling, batch_size):
    # change them into np.array
    input_sentences = np.array(input_sentences, dtype="object")
    edu_breaks = np.array(edu_breaks, dtype="object")
    decoder_input = np.array(decoder_input, dtype="object")
    relation_label = np.array(relation_label, dtype="object")
    parsing_breaks = np.array(parsing_breaks, dtype="object")
    golden_metric = np.array(golden_metric, dtype="object")
    parents_index = np.array(parents_index, dtype="object")
    sibling = np.array(sibling, dtype="object")

    if len(decoder_input) < batch_size:
        batch_size = len(decoder_input)

    sample_indices = random.sample(range(len(decoder_input)), batch_size)
    # Get batch data
    input_sentences_batch = copy.deepcopy(input_sentences[sample_indices])
    edu_breaks_batch = copy.deepcopy(edu_breaks[sample_indices])
    decoder_input_batch = copy.deepcopy(decoder_input[sample_indices])
    relation_label_batch = copy.deepcopy(relation_label[sample_indices])
    parsing_breaks_batch = copy.deepcopy(parsing_breaks[sample_indices])
    golden_metric_batch = copy.deepcopy(golden_metric[sample_indices])
    parents_index_batch = copy.deepcopy(parents_index[sample_indices])
    sibling_batch = copy.deepcopy(sibling[sample_indices])

    # Get sorted
    lengths_batch = np.array([len(sent) for sent in input_sentences_batch])
    idx = np.argsort(lengths_batch)
    idx = idx[::-1]

    # Convert them back to list
    input_sentences_batch = input_sentences_batch[idx].tolist()
    edu_breaks_batch = edu_breaks_batch[idx].tolist()
    decoder_input_batch = decoder_input_batch[idx].tolist()
    relation_label_batch = relation_label_batch[idx].tolist()
    parsing_breaks_batch = parsing_breaks_batch[idx].tolist()
    golden_metric_batch = golden_metric_batch[idx].tolist()
    parents_index_batch = parents_index_batch[idx].tolist()
    sibling_batch = sibling_batch[idx].tolist()

    return input_sentences_batch, edu_breaks_batch, decoder_input_batch, relation_label_batch, \
           parsing_breaks_batch, golden_metric_batch, parents_index_batch, sibling_batch


def get_batch_data(input_sentences, edu_breaks, decoder_input, relation_label,
                   parsing_breaks, golden_metric, batch_size):
    # change them into np.array
    input_sentences = np.array(input_sentences, dtype="object")
    edu_breaks = np.array(edu_breaks, dtype="object")
    decoder_input = np.array(decoder_input, dtype="object")
    relation_label = np.array(relation_label, dtype="object")
    parsing_breaks = np.array(parsing_breaks, dtype="object")
    golden_metric = np.array(golden_metric, dtype="object")

    if len(decoder_input) < batch_size:
        batch_size = len(decoder_input)
    sample_indices = random.sample(range(len(decoder_input)), batch_size)

    # Get batch data
    input_sentences_batch = copy.deepcopy(input_sentences[sample_indices])
    edu_breaks_batch = copy.deepcopy(edu_breaks[sample_indices])
    decoder_input_batch = copy.deepcopy(decoder_input[sample_indices])
    relation_label_batch = copy.deepcopy(relation_label[sample_indices])
    parsing_breaks_batch = copy.deepcopy(parsing_breaks[sample_indices])
    golden_metric_batch = copy.deepcopy(golden_metric[sample_indices])

    # Get sorted
    lengths_batch = np.array([len(sent) for sent in input_sentences_batch])
    idx = np.argsort(lengths_batch)
    idx = idx[::-1]

    # Convert them back to list
    input_sentences_batch = input_sentences_batch[idx].tolist()
    edu_breaks_batch = edu_breaks_batch[idx].tolist()
    decoder_input_batch = decoder_input_batch[idx].tolist()
    relation_label_batch = relation_label_batch[idx].tolist()
    parsing_breaks_batch = parsing_breaks_batch[idx].tolist()
    golden_metric_batch = golden_metric_batch[idx].tolist()

    return input_sentences_batch, edu_breaks_batch, decoder_input_batch, relation_label_batch, parsing_breaks_batch, golden_metric_batch


class Train(object):
    def __init__(self, model, train_input_sentences, train_edu_breaks, train_decoder_input,
                 train_relation_label, train_parsing_breaks, train_golden_metric,
                 train_parents_index, train_sibling_index,
                 test_input_sentences, test_edu_breaks, test_decoder_input,
                 test_relation_label, test_parsing_breaks, test_golden_metric,
                 batch_size, eval_size, epochs, lr, lr_decay_epoch, weight_decay,
                 save_path, device):

        self.model = model
        self.train_input_sentences = train_input_sentences
        self.train_edu_breaks = train_edu_breaks
        self.train_decoder_input = train_decoder_input
        self.train_relation_label = train_relation_label
        self.train_parsing_breaks = train_parsing_breaks
        self.train_golden_metric = train_golden_metric
        self.train_parents_index = train_parents_index
        self.train_sibling_index = train_sibling_index
        self.test_input_sentences = test_input_sentences
        self.test_edu_breaks = test_edu_breaks
        self.test_decoder_input = test_decoder_input
        self.test_relation_label = test_relation_label
        self.test_parsing_breaks = test_parsing_breaks
        self.test_golden_metric = test_golden_metric
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay_epoch = lr_decay_epoch
        self.weight_decay = weight_decay
        self.save_path = save_path

        # Preprocessor
        self.preprocessor = RSTPreprocessor(device=device)

    def get_training_eval(self):
        # Obtain eval_size samples of training data to evaluate the model in
        # every epoch

        # Convert to np.array
        train_input_sentences = np.array(self.train_input_sentences)
        train_edu_breaks = np.array(self.train_edu_breaks)
        train_decoder_input = np.array(self.train_decoder_input)
        train_relation_label = np.array(self.train_relation_label)
        train_parsing_breaks = np.array(self.train_parsing_breaks)
        train_golden_metric = np.array(self.train_golden_metric)

        sample_indices = random.sample(range(len(self.train_parsing_breaks)), self.eval_size)

        eval_input_sentences = train_input_sentences[sample_indices].tolist()
        eval_edu_breaks = train_edu_breaks[sample_indices].tolist()
        eval_decoder_input = train_decoder_input[sample_indices].tolist()
        eval_relation_label = train_relation_label[sample_indices].tolist()
        eval_parsing_breaks = train_parsing_breaks[sample_indices].tolist()
        eval_golden_metric = train_golden_metric[sample_indices].tolist()

        return eval_input_sentences, eval_edu_breaks, eval_decoder_input, eval_relation_label, \
               eval_parsing_breaks, eval_golden_metric

    def get_accuracy(self, input_sentences, edu_breaks, decoder_input, relation_label,
                     parsing_breaks, golden_metric):

        num_loops = int(np.ceil(len(edu_breaks) / self.batch_size))

        loss_tree_all = []
        loss_label_all = []
        correct_span = 0
        correct_relation = 0
        correct_nuclearity = 0
        no_system = 0
        no_golden = 0

        for loop in range(num_loops):

            start_position = loop * self.batch_size
            end_position = (loop + 1) * self.batch_size
            if end_position > len(edu_breaks):
                end_position = len(edu_breaks)

            input_sentences_batch, edu_breaks_batch, _, \
            relation_label_batch, parsing_breaks_batch, golden_metric_splits_batch = \
                get_batch_data(input_sentences[start_position:end_position],
                               edu_breaks[start_position:end_position],
                               decoder_input[start_position:end_position],
                               relation_label[start_position:end_position],
                               parsing_breaks[start_position:end_position],
                               golden_metric[start_position:end_position], self.batch_size)

            input_sentences_ids_batch, sentence_lengths = self.preprocessor(input_sentences_batch)

            model_output = self.model.forward(
                input_sentence=input_sentences_ids_batch,
                edu_breaks=edu_breaks_batch,
                label_index=relation_label_batch,
                parsing_index=parsing_breaks_batch,
                sentence_lengths=sentence_lengths,
                generate_splits=True
            )

            loss_tree_all.append(model_output.loss_tree_batch)
            loss_label_all.append(model_output.loss_label_batch)
            correct_span_batch, correct_relation_batch, correct_nuclearity_batch, \
            no_system_batch, no_golden_batch = get_batch_measure(model_output.split_batch,
                                                                 golden_metric_splits_batch)

            correct_span = correct_span + correct_span_batch
            correct_relation = correct_relation + correct_relation_batch
            correct_nuclearity = correct_nuclearity + correct_nuclearity_batch
            no_system = no_system + no_system_batch
            no_golden = no_golden + no_golden_batch

        span_points, relation_points, nuclearity_points = get_micro_measure(
            correct_span, correct_relation, correct_nuclearity, no_system, no_golden)

        return np.mean(loss_tree_all), np.mean(loss_label_all), span_points, relation_points, nuclearity_points

    def learning_rate_adjust(self, optimizer, epoch, lr_decay=0.5, lr_decay_epoch=50):
        if (epoch % lr_decay_epoch == 0) and (epoch != 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_decay

    def train(self):

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=self.lr, betas=(0.9, 0.9), weight_decay=self.weight_decay)

        num_iterations = int(np.ceil(len(self.train_parsing_breaks) / self.batch_size))

        try:
            os.mkdir(self.save_path)
        except:
            pass

        best_F_relation = 0
        best_F_span = 0
        for current_epoch in range(self.epochs):
            self.learning_rate_adjust(optimizer, current_epoch, 0.8, self.lr_decay_epoch)

            for current_iteration in range(num_iterations):
                input_sentences_batch, edu_breaks_batch, decoder_input_batch, \
                relation_label_batch, parsing_breaks_batch, _, parents_index_batch, \
                sibling_batch = get_batch_data_training(
                    self.train_input_sentences, self.train_edu_breaks,
                    self.train_decoder_input, self.train_relation_label,
                    self.train_parsing_breaks, self.train_golden_metric,
                    self.train_parents_index, self.train_sibling_index, self.batch_size)

                self.model.zero_grad()

                input_sentences_ids_batch, sentence_lengths = self.preprocessor(input_sentences_batch)

                loss_tree_batch, loss_label_batch = self.model.forward_train(
                    input_sentence_ids_batch=input_sentences_ids_batch,
                    edu_breaks_batch=edu_breaks_batch,
                    label_index_batch=relation_label_batch,
                    parsing_index_batch=parsing_breaks_batch,
                    decoder_input_index_batch=decoder_input_batch,
                    parents_index_batch=parents_index_batch,
                    sibling_index_batch=sibling_batch,
                    sentence_lengths=sentence_lengths
                )

                loss = loss_tree_batch + loss_label_batch
                loss.backward()

                cur_loss = float(loss.item())

                logger.info(f'Epoch: {current_epoch + 1}/{self.epochs}, '
                            f'iteration: {current_iteration + 1}/{num_iterations}, '
                            f'loss: {cur_loss:.3f}')

                # To avoid gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                optimizer.step()

            # Convert model to eval
            self.model.eval()

            # Obtain Training (development) data
            eval_input_sentences, eval_edu_breaks, eval_decoder_input, eval_relation_label, \
            eval_parsing_breaks, eval_golden_metric = self.get_training_eval()

            # Eval on training (development) data
            loss_tree_eval, loss_label_eval, span_points_eval, relation_points_eval, nuclearity_points_eval = \
                self.get_accuracy(eval_input_sentences, eval_edu_breaks,
                                  eval_decoder_input, eval_relation_label,
                                  eval_parsing_breaks, eval_golden_metric)

            # Eval on Testing data
            loss_tree_test, loss_label_test, span_points_test, relation_points_test, nuclearity_points_test = \
                self.get_accuracy(self.test_input_sentences, self.test_edu_breaks,
                                  self.test_decoder_input, self.test_relation_label,
                                  self.test_parsing_breaks, self.test_golden_metric)

            # Unfold numbers
            # Test
            P_span, R_span, F_span = span_points_test
            P_relation, R_relation, F_relation = relation_points_test
            P_nuclearity, R_nuclearity, F_nuclearity = nuclearity_points_test
            # Training Eval
            _, _, F_span_eval = span_points_eval
            _, _, F_relation_eval = relation_points_eval
            _, _, F_nuclearity_eval = nuclearity_points_eval

            # Relation will take the priority consideration
            if F_relation > best_F_relation:
                best_epoch = current_epoch
                # relation
                best_F_relation = F_relation
                best_P_relation = P_relation
                best_R_relation = R_relation
                # span
                best_F_span = F_span
                best_P_span = P_span
                best_R_span = R_span
                # nuclearity
                best_F_nuclearity = F_nuclearity
                best_P_nuclearity = P_nuclearity
                best_R_nuclearity = R_nuclearity

            # Saving data
            save_data = [current_epoch, loss_tree_eval, loss_label_eval,
                         F_span_eval, F_relation_eval, F_nuclearity_eval,
                         loss_tree_test, loss_label_test, F_span, F_relation, F_nuclearity]

            # Log evaluation and test metrics
            self.log_metrics(log_prefix='Metrics on train sample --',
                             loss_tree=loss_tree_eval, loss_label=loss_label_eval,
                             f1_span=F_span_eval, f1_relation=F_relation_eval, f1_nuclearity=F_nuclearity_eval)
            self.log_metrics(log_prefix='Metrics on test data --',
                             loss_tree=loss_tree_test, loss_label=loss_label_test,
                             f1_span=F_span, f1_relation=F_relation, f1_nuclearity=F_nuclearity)

            logger.info(f'End of epoch {current_epoch + 1}')
            file_name = f'span_bs_{self.batch_size}_es_{self.eval_size}_lr_{self.lr}_' \
                        f'lrdc_{self.lr_decay_epoch}_wd_{self.weight_decay}.txt'

            with open(os.path.join(self.save_path, file_name), 'a+') as f:
                f.write(','.join(map(str, save_data)) + '\n')

            # Saving model
            if best_epoch == current_epoch:
                torch.save(self.model, os.path.join(self.save_path, f'epoch_{current_epoch + 1}.torchsave'))

            # Convert back to training
            self.model.train()

        return best_epoch, best_F_relation, best_P_relation, best_R_relation, best_F_span, \
               best_P_span, best_R_span, best_F_nuclearity, best_P_nuclearity, best_R_nuclearity

    def log_metrics(self, log_prefix, loss_tree, loss_label, f1_span, f1_relation, f1_nuclearity):
        logger.info(f'{log_prefix} \n'
                    f'\t'
                    f'loss_tree: {loss_tree:.3f}, loss_label: {loss_label:.3f} \n'
                    f'\t'
                    f'f1_span: {f1_span:.3f}, f1_relation: {f1_relation:.3f}, f1_nuclearity: {f1_nuclearity:.3f}')


def train_parser(cfg: RstPointerParserTrainArgs) -> None:
    logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.info(f'===== Training RST Pointer Parser =====')

    # Setup
    setup(seed=cfg.seed)

    train_data_dir = cfg.train_data_dir
    test_data_dir = cfg.test_data_dir
    save_dir = cfg.save_dir
    batch_size = cfg.batch_size
    hidden_size = cfg.hidden_size
    rnn_layers = cfg.rnn_layers
    dropout_e = cfg.dropout_e
    dropout_d = cfg.dropout_d
    dropout_c = cfg.dropout_c
    atten_model = cfg.atten_model
    classifier_input_size = cfg.classifier_input_size
    classifier_hidden_size = cfg.classifier_hidden_size
    classifier_bias = cfg.classifier_bias
    elmo_size = cfg.elmo_size
    seed = cfg.seed
    eval_size = cfg.eval_size
    epochs = cfg.epochs
    lr = cfg.lr
    lr_decay_epoch = cfg.lr_decay_epoch
    weight_decay = cfg.weight_decay
    highorder = cfg.highorder

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:" + str(cfg.gpu_id) if USE_CUDA else "cpu")
    logger.info(f'Using CUDA: {USE_CUDA}')

    logger.info('Loading training and test data...')
    # Load Training data
    tr_input_sentences = pickle.load(open(os.path.join(train_data_dir, "tokenized_sentences.pickle"), "rb"))
    tr_edu_breaks = pickle.load(open(os.path.join(train_data_dir, "edu_breaks.pickle"), "rb"))
    tr_decoder_input = pickle.load(open(os.path.join(train_data_dir, "decoder_input_index.pickle"), "rb"))
    tr_relation_label = pickle.load(open(os.path.join(train_data_dir, "relation_index.pickle"), "rb"))
    tr_parsing_breaks = pickle.load(open(os.path.join(train_data_dir, "splits_order.pickle"), "rb"))
    tr_golden_metric = pickle.load(open(os.path.join(train_data_dir, "discourse_tree_splits.pickle"), "rb"))
    tr_parents_index = pickle.load(open(os.path.join(train_data_dir, "parent_index.pickle"), "rb"))
    tr_sibling_index = pickle.load(open(os.path.join(train_data_dir, "sibling_index.pickle"), "rb"))

    # Load Testing data
    test_input_sentences = pickle.load(open(os.path.join(test_data_dir, "tokenized_sentences.pickle"), "rb"))
    test_edu_breaks = pickle.load(open(os.path.join(test_data_dir, "edu_breaks.pickle"), "rb"))
    test_decoder_input = pickle.load(open(os.path.join(test_data_dir, "decoder_input_index.pickle"), "rb"))
    test_relation_label = pickle.load(open(os.path.join(test_data_dir, "relation_index.pickle"), "rb"))
    test_parsing_breaks = pickle.load(open(os.path.join(test_data_dir, "splits_order.pickle"), "rb"))
    test_golden_metric = pickle.load(open(os.path.join(test_data_dir, "discourse_tree_splits.pickle"), "rb"))

    # To save model and data
    file_name = f'seed_{seed}_batchSize_{batch_size}_elmo_{elmo_size}_attenModel_{atten_model}' \
                f'_rnnLayers_{rnn_layers}_rnnHiddenSize_{hidden_size}_classifierHiddenSize_{classifier_hidden_size}'

    model_save_dir = os.path.join(save_dir, file_name)

    logger.info('--------------------------------------------------------------------')
    logger.info('Starting model training...')
    logger.info('--------------------------------------------------------------------')
    # Initialize model
    # model_config = RstPointerParserConfig.from_pretrained(cfg.model_config_path)
    model_config = RstPointerParserConfig(
        batch_size=batch_size,
        hidden_size=hidden_size,
        decoder_input_size=hidden_size,
        atten_model=atten_model,
        device=device,
        classifier_input_size=classifier_input_size,
        classifier_hidden_size=classifier_hidden_size,
        highorder=highorder,
        classes_label=39,
        classifier_bias=classifier_bias,
        rnn_layers=rnn_layers,
        dropout_e=dropout_e,
        dropout_d=dropout_d,
        dropout_c=dropout_c,
        elmo_size=elmo_size
    )

    model = RstPointerParserModel(model_config)
    model = model.to(device)
    model.embedding.to(device)  # Elmo layer doesn't get put onto device automatically

    trainer = Train(model, tr_input_sentences, tr_edu_breaks, tr_decoder_input,
                    tr_relation_label, tr_parsing_breaks, tr_golden_metric,
                    tr_parents_index, tr_sibling_index,
                    test_input_sentences, test_edu_breaks, test_decoder_input,
                    test_relation_label, test_parsing_breaks, test_golden_metric,
                    batch_size, eval_size, epochs, lr, lr_decay_epoch,
                    weight_decay, model_save_dir, device)

    best_epoch, best_F_relation, best_P_relation, best_R_relation, best_F_span, \
    best_P_span, best_R_span, best_F_nuclearity, best_P_nuclearity, \
    best_R_nuclearity = trainer.train()

    logger.info('--------------------------------------------------------------------')
    logger.info('Model training completed!')
    logger.info('--------------------------------------------------------------------')
    logger.info('Processing...')
    logger.info(f'The best F1 points for Relation is: {best_F_relation:.3f}.')
    logger.info(f'The best F1 points for Nuclearity is: {best_F_nuclearity:.3f}')
    logger.info(f'The best F1 points for Span is: {best_F_span:.3f}')

    # Save result
    with open(os.path.join(save_dir, 'results.csv'), 'a') as f:
        f.write(file_name + ',' + ','.join(map(str, [best_epoch, best_F_relation,
                                                     best_P_relation, best_R_relation, best_F_span,
                                                     best_P_span, best_R_span, best_F_nuclearity,
                                                     best_P_nuclearity, best_R_nuclearity])) + '\n')


def train_segmenter(cfg: RstPointerSegmenterTrainArgs) -> None:
    pass


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if isinstance(cfg, RstPointerSegmenterTrainArgs):
        train_segmenter(cfg)
        print(cfg)
    if isinstance(cfg, RstPointerParserTrainArgs):
        train_parser(cfg)
        print(cfg)
