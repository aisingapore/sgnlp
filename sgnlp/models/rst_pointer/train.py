import copy
import logging
import os
import pickle
import random
from typing import List

import numpy as np
import torch

from sgnlp.utils.csv_writer import CsvWriter
from .data_class import RstPointerParserTrainArgs, RstPointerSegmenterTrainArgs
from .modeling import (
    RstPointerParserModel,
    RstPointerParserConfig,
    RstPointerSegmenterModel,
    RstPointerSegmenterConfig,
)
from .modules.type import DiscourseTreeNode, DiscourseTreeSplit
from .preprocess import RstPreprocessor
from .utils import parse_args_and_load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Shared functions
def setup(seed):
    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # See: https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=50):
    if (epoch % lr_decay_epoch == 0) and (epoch != 0):
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_decay


# Parser training code
def get_span_dict(discourse_tree_splits: List[DiscourseTreeSplit]):
    span_dict = {}

    for split in discourse_tree_splits:
        left_span_key, left_node_value = get_span_key_and_node_value(split.left)
        span_dict[left_span_key] = left_node_value
        right_span_key, right_node_value = get_span_key_and_node_value(split.right)
        span_dict[right_span_key] = right_node_value

    return span_dict


def get_span_key_and_node_value(node: DiscourseTreeNode):
    span_key = f"{node.span[0]}-{node.span[1]}"
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

    return (
        num_matching_spans,
        num_correct_relations,
        num_correct_nuclearity,
        num_spans_1,
        num_spans_2,
    )


def get_batch_measure(input_splits_batch, golden_metric_batch):
    num_matching_spans = 0
    num_correct_relations = 0
    num_correct_nuclearity = 0
    num_spans_input = 0
    num_spans_golden = 0

    for input_splits, golden_splits in zip(input_splits_batch, golden_metric_batch):
        if input_splits and golden_splits:
            # if both splits have values in the list
            (
                _num_matching_spans,
                _num_correct_relations,
                _num_correct_nuclearity,
                _num_spans_input,
                _num_spans_golden,
            ) = get_measurement(input_splits, golden_splits)

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

    return (
        num_matching_spans,
        num_correct_relations,
        num_correct_nuclearity,
        num_spans_input,
        num_spans_golden,
    )


def get_micro_measure(
    correct_span, correct_relation, correct_nuclearity, no_system, no_golden
):
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

    return (
        (precision_span, recall_span, f1_span),
        (precision_relation, recall_relation, f1_relation),
        (precision_nuclearity, recall_nuclearity, f1_nuclearity),
    )


# TODO: Change data sampling
def get_batch_data_training(
    input_sentences,
    edu_breaks,
    decoder_input,
    relation_label,
    parsing_breaks,
    golden_metric,
    parents_index,
    sibling,
    batch_size,
):
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

    return (
        input_sentences_batch,
        edu_breaks_batch,
        decoder_input_batch,
        relation_label_batch,
        parsing_breaks_batch,
        golden_metric_batch,
        parents_index_batch,
        sibling_batch,
    )


def get_batch_data(
    input_sentences,
    edu_breaks,
    decoder_input,
    relation_label,
    parsing_breaks,
    golden_metric,
    batch_size,
):
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

    return (
        input_sentences_batch,
        edu_breaks_batch,
        decoder_input_batch,
        relation_label_batch,
        parsing_breaks_batch,
        golden_metric_batch,
    )


def get_accuracy(
    model,
    preprocessor,
    input_sentences,
    edu_breaks,
    decoder_input,
    relation_label,
    parsing_breaks,
    golden_metric,
    batch_size,
):
    num_loops = int(np.ceil(len(edu_breaks) / batch_size))

    loss_tree_all = []
    loss_label_all = []
    correct_span = 0
    correct_relation = 0
    correct_nuclearity = 0
    no_system = 0
    no_golden = 0

    for loop in range(num_loops):
        start_idx = loop * batch_size
        end_idx = (loop + 1) * batch_size
        if end_idx > len(edu_breaks):
            end_idx = len(edu_breaks)

        (
            input_sentences_batch,
            edu_breaks_batch,
            _,
            relation_label_batch,
            parsing_breaks_batch,
            golden_metric_splits_batch,
        ) = get_batch_data(
            input_sentences[start_idx:end_idx],
            edu_breaks[start_idx:end_idx],
            decoder_input[start_idx:end_idx],
            relation_label[start_idx:end_idx],
            parsing_breaks[start_idx:end_idx],
            golden_metric[start_idx:end_idx],
            batch_size,
        )

        input_sentences_ids_batch, sentence_lengths = preprocessor.get_elmo_char_ids(
            input_sentences_batch
        )
        input_sentences_ids_batch = input_sentences_ids_batch.to(device=model.device)

        model_output = model(
            input_sentence_ids=input_sentences_ids_batch,
            edu_breaks=edu_breaks_batch,
            sentence_lengths=sentence_lengths,
            label_index=relation_label_batch,
            parsing_index=parsing_breaks_batch,
            generate_splits=True,
        )

        loss_tree_all.append(model_output.loss_tree)
        loss_label_all.append(model_output.loss_label)
        (
            correct_span_batch,
            correct_relation_batch,
            correct_nuclearity_batch,
            no_system_batch,
            no_golden_batch,
        ) = get_batch_measure(model_output.splits, golden_metric_splits_batch)

        correct_span = correct_span + correct_span_batch
        correct_relation = correct_relation + correct_relation_batch
        correct_nuclearity = correct_nuclearity + correct_nuclearity_batch
        no_system = no_system + no_system_batch
        no_golden = no_golden + no_golden_batch

    span_points, relation_points, nuclearity_points = get_micro_measure(
        correct_span, correct_relation, correct_nuclearity, no_system, no_golden
    )

    return (
        np.mean(loss_tree_all),
        np.mean(loss_label_all),
        span_points,
        relation_points,
        nuclearity_points,
    )


def train_parser(cfg: RstPointerParserTrainArgs) -> None:
    logger.info(f"===== Training RST Pointer Parser =====")

    # Setup
    setup(seed=cfg.seed)

    train_data_dir = cfg.train_data_dir
    test_data_dir = cfg.test_data_dir
    save_dir = cfg.save_dir
    batch_size = cfg.batch_size
    hidden_size = cfg.hidden_size
    rnn_layers = cfg.num_rnn_layers
    dropout_e = cfg.dropout_e
    dropout_d = cfg.dropout_d
    dropout_c = cfg.dropout_c
    atten_model = cfg.atten_model
    classifier_input_size = cfg.classifier_input_size
    classifier_hidden_size = cfg.classifier_hidden_size
    classifier_bias = cfg.classifier_bias
    elmo_size = cfg.elmo_size
    epochs = cfg.epochs
    lr = cfg.lr
    lr_decay_epoch = cfg.lr_decay_epoch
    weight_decay = cfg.weight_decay
    highorder = cfg.highorder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create directory and files
    os.makedirs(save_dir, exist_ok=True)
    best_results_writer = CsvWriter(
        file_path=os.path.join(save_dir, "best_results.csv"),
        fieldnames=[
            "best_epoch",
            "f1_relation",
            "precision_relation",
            "recall_relation",
            "f1_span",
            "precision_span",
            "recall_span",
            "f1_nuclearity",
            "precision_nuclearity",
            "recall_nuclearity",
        ],
    )
    results_writer = CsvWriter(
        file_path=os.path.join(save_dir, "results.csv"),
        fieldnames=[
            "current_epoch",
            "loss_tree_test",
            "loss_label_test",
            "f1_span",
            "f1_relation",
            "f1_nuclearity",
        ],
    )

    logger.info("Loading training and test data...")
    # Load Training data
    tr_input_sentences = pickle.load(
        open(os.path.join(train_data_dir, "tokenized_sentences.pickle"), "rb")
    )
    tr_edu_breaks = pickle.load(
        open(os.path.join(train_data_dir, "edu_breaks.pickle"), "rb")
    )
    tr_decoder_input = pickle.load(
        open(os.path.join(train_data_dir, "decoder_input_index.pickle"), "rb")
    )
    tr_relation_label = pickle.load(
        open(os.path.join(train_data_dir, "relation_index.pickle"), "rb")
    )
    tr_parsing_breaks = pickle.load(
        open(os.path.join(train_data_dir, "splits_order.pickle"), "rb")
    )
    tr_golden_metric = pickle.load(
        open(os.path.join(train_data_dir, "discourse_tree_splits.pickle"), "rb")
    )
    tr_parents_index = pickle.load(
        open(os.path.join(train_data_dir, "parent_index.pickle"), "rb")
    )
    tr_sibling_index = pickle.load(
        open(os.path.join(train_data_dir, "sibling_index.pickle"), "rb")
    )

    # Load Testing data
    test_input_sentences = pickle.load(
        open(os.path.join(test_data_dir, "tokenized_sentences.pickle"), "rb")
    )
    test_edu_breaks = pickle.load(
        open(os.path.join(test_data_dir, "edu_breaks.pickle"), "rb")
    )
    test_decoder_input = pickle.load(
        open(os.path.join(test_data_dir, "decoder_input_index.pickle"), "rb")
    )
    test_relation_label = pickle.load(
        open(os.path.join(test_data_dir, "relation_index.pickle"), "rb")
    )
    test_parsing_breaks = pickle.load(
        open(os.path.join(test_data_dir, "splits_order.pickle"), "rb")
    )
    test_golden_metric = pickle.load(
        open(os.path.join(test_data_dir, "discourse_tree_splits.pickle"), "rb")
    )

    logger.info("--------------------------------------------------------------------")
    logger.info("Starting model training...")
    logger.info("--------------------------------------------------------------------")
    # Initialize model
    model_config = RstPointerParserConfig(
        hidden_size=hidden_size,
        decoder_input_size=hidden_size,
        atten_model=atten_model,
        classifier_input_size=classifier_input_size,
        classifier_hidden_size=classifier_hidden_size,
        highorder=highorder,
        classes_label=39,
        classifier_bias=classifier_bias,
        rnn_layers=rnn_layers,
        dropout_e=dropout_e,
        dropout_d=dropout_d,
        dropout_c=dropout_c,
        elmo_size=elmo_size,
    )

    model = RstPointerParserModel(model_config)
    model = model.to(device)

    preprocessor = RstPreprocessor()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.9),
        weight_decay=weight_decay,
    )

    num_iterations = int(np.ceil(len(tr_parsing_breaks) / batch_size))

    best_f1_relation = 0
    best_f1_span = 0
    for current_epoch in range(epochs):
        adjust_learning_rate(optimizer, current_epoch, 0.8, lr_decay_epoch)

        for current_iteration in range(num_iterations):
            (
                input_sentences_batch,
                edu_breaks_batch,
                decoder_input_batch,
                relation_label_batch,
                parsing_breaks_batch,
                _,
                parents_index_batch,
                sibling_batch,
            ) = get_batch_data_training(
                tr_input_sentences,
                tr_edu_breaks,
                tr_decoder_input,
                tr_relation_label,
                tr_parsing_breaks,
                tr_golden_metric,
                tr_parents_index,
                tr_sibling_index,
                batch_size,
            )

            model.zero_grad()

            (
                input_sentences_ids_batch,
                sentence_lengths,
            ) = preprocessor.get_elmo_char_ids(input_sentences_batch)
            input_sentences_ids_batch = input_sentences_ids_batch.to(
                device=model.device
            )

            loss_tree_batch, loss_label_batch = model.forward_train(
                input_sentence_ids_batch=input_sentences_ids_batch,
                edu_breaks_batch=edu_breaks_batch,
                label_index_batch=relation_label_batch,
                parsing_index_batch=parsing_breaks_batch,
                decoder_input_index_batch=decoder_input_batch,
                parents_index_batch=parents_index_batch,
                sibling_index_batch=sibling_batch,
                sentence_lengths=sentence_lengths,
            )

            loss = loss_tree_batch + loss_label_batch
            loss.backward()

            cur_loss = float(loss.item())

            logger.info(
                f"Epoch: {current_epoch + 1}/{epochs}, "
                f"iteration: {current_iteration + 1}/{num_iterations}, "
                f"loss: {cur_loss:.3f}"
            )

            # To avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

        # Convert model to eval
        model.eval()

        # Eval on Testing data
        (
            loss_tree_test,
            loss_label_test,
            span_points_test,
            relation_points_test,
            nuclearity_points_test,
        ) = get_accuracy(
            model,
            preprocessor,
            test_input_sentences,
            test_edu_breaks,
            test_decoder_input,
            test_relation_label,
            test_parsing_breaks,
            test_golden_metric,
            batch_size,
        )

        # Unfold numbers
        # Test
        precision_span, recall_span, f1_span = span_points_test
        precision_relation, recall_relation, f1_relation = relation_points_test
        precision_nuclearity, recall_nuclearity, f1_nuclearity = nuclearity_points_test

        # Relation will take the priority consideration
        if f1_relation > best_f1_relation:
            best_epoch = current_epoch
            # relation
            best_f1_relation = f1_relation
            best_precision_relation = precision_relation
            best_recall_relation = recall_relation
            # span
            best_f1_span = f1_span
            best_precision_span = precision_span
            best_recall_span = recall_span
            # nuclearity
            best_f1_nuclearity = f1_nuclearity
            best_precision_nuclearity = precision_nuclearity
            best_recall_nuclearity = recall_nuclearity

        # Log evaluation and test metrics
        epoch_metrics = {
            "current_epoch": current_epoch,
            "loss_tree_test": loss_tree_test,
            "loss_label_test": loss_label_test,
            "f1_span": f1_span,
            "f1_relation": f1_relation,
            "f1_nuclearity": f1_nuclearity,
        }
        logger.info(f"Test metrics: {epoch_metrics}")

        results_writer.writerow(epoch_metrics)

        # Saving model
        if best_epoch == current_epoch:
            model.save_pretrained(save_dir)

        # Convert back to training
        model.train()

    logger.info("--------------------------------------------------------------------")
    logger.info("Model training completed!")
    logger.info("--------------------------------------------------------------------")
    logger.info(f"The best F1 points for Relation is: {best_f1_relation:.3f}.")
    logger.info(f"The best F1 points for Nuclearity is: {best_f1_nuclearity:.3f}")
    logger.info(f"The best F1 points for Span is: {best_f1_span:.3f}")

    best_results_writer.writerow(
        {
            "best_epoch": best_epoch,
            "f1_relation": best_f1_relation,
            "precision_relation": best_precision_relation,
            "recall_relation": best_recall_relation,
            "f1_span": best_f1_span,
            "precision_span": best_precision_span,
            "recall_span": best_recall_span,
            "f1_nuclearity": best_f1_nuclearity,
            "precision_nuclearity": best_precision_nuclearity,
            "recall_nuclearity": best_recall_nuclearity,
        }
    )


# Segmenter training code
def sample_a_sorted_batch_from_numpy(input_x, output_y, batch_size):
    input_x = np.array(input_x, dtype="object")
    output_y = np.array(output_y, dtype="object")

    if batch_size is not None:
        select_index = random.sample(range(len(output_y)), batch_size)
    else:
        select_index = np.array(range(len(output_y)))

    batch_x = copy.deepcopy(input_x[select_index])
    batch_y = copy.deepcopy(output_y[select_index])

    all_lens = np.array([len(x) for x in batch_x])

    idx = np.argsort(all_lens)
    idx = idx[::-1]  # decreasing

    batch_x = batch_x[idx]

    batch_y = batch_y[idx]

    # decoder input
    batch_x_index = []

    for i in range(len(batch_y)):
        cur_y = batch_y[i]

        temp = [x + 1 for x in cur_y]
        temp.insert(0, 0)
        temp.pop()
        batch_x_index.append(temp)

    all_lens = all_lens[idx]

    return batch_x, batch_x_index, batch_y, all_lens


def get_batch_test(x, y, batch_size):
    x = np.array(x, dtype="object")
    y = np.array(y, dtype="object")

    if batch_size is not None:
        select_index = random.sample(range(len(y)), batch_size)
    else:
        select_index = np.array(range(len(y)))

    batch_x = copy.deepcopy(x[select_index])
    batch_y = copy.deepcopy(y[select_index])

    all_lens = np.array([len(x) for x in batch_x])

    return batch_x, batch_y, all_lens


def sample_batch(x, y, sample_size):
    select_index = random.sample(range(len(y)), sample_size)
    x = np.array(x, dtype="object")
    y = np.array(y, dtype="object")

    return x[select_index], y[select_index]


def get_batch_micro_metric(pre_b, ground_b):
    all_c = []
    all_r = []
    all_g = []
    for i in range(len(ground_b)):
        index_of_1 = np.array(ground_b[i])
        index_pre = pre_b[i]

        index_pre = np.array(index_pre)

        end_b = index_of_1[-1]
        index_pre = index_pre[index_pre != end_b]
        index_of_1 = index_of_1[index_of_1 != end_b]

        no_correct = len(np.intersect1d(list(index_of_1), list(index_pre)))
        all_c.append(no_correct)
        all_r.append(len(index_pre))
        all_g.append(len(index_of_1))

    return all_c, all_r, all_g


# Unused
def get_batch_metric(pre_b, ground_b):
    b_pr = []
    b_re = []
    b_f1 = []
    for i, cur_seq_y in enumerate(ground_b):
        index_of_1 = np.where(cur_seq_y == 1)[0]
        index_pre = pre_b[i]

        no_correct = len(np.intersect1d(index_of_1, index_pre))

        cur_pre = no_correct / len(index_pre)
        cur_rec = no_correct / len(index_of_1)
        cur_f1 = 2 * cur_pre * cur_rec / (cur_pre + cur_rec)

        b_pr.append(cur_pre)
        b_re.append(cur_rec)
        b_f1.append(cur_f1)

    return b_pr, b_re, b_f1


def check_accuracy(model, preprocessor, x, y, batch_size):
    num_loops = int(np.ceil(len(y) / batch_size))

    all_ave_loss = []
    all_start_boundaries = []
    all_end_boundaries = []
    all_index_decoder_y = []
    all_x_save = []

    all_c = []
    all_r = []
    all_g = []
    for i in range(num_loops):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        if end_idx > len(y):
            end_idx = len(y)

        batch_x, batch_y, all_lens = get_batch_test(
            x[start_idx:end_idx], y[start_idx:end_idx], None
        )

        input_sentences_ids_batch, _ = preprocessor.get_elmo_char_ids(batch_x)
        input_sentences_ids_batch = input_sentences_ids_batch.to(device=model.device)

        output = model(input_sentences_ids_batch, all_lens, batch_y)
        batch_ave_loss = output.loss
        batch_start_boundaries = output.start_boundaries
        batch_end_boundaries = output.end_boundaries

        all_ave_loss.extend([batch_ave_loss.cpu().data.numpy()])
        all_start_boundaries.extend(batch_start_boundaries)
        all_end_boundaries.extend(batch_end_boundaries)

        ba_c, ba_r, ba_g = get_batch_micro_metric(batch_end_boundaries, batch_y)

        all_c.extend(ba_c)
        all_r.extend(ba_r)
        all_g.extend(ba_g)

    ba_pre = np.sum(all_c) / np.sum(all_r)
    ba_rec = np.sum(all_c) / np.sum(all_g)
    ba_f1 = 2 * ba_pre * ba_rec / (ba_pre + ba_rec)

    return (
        np.mean(all_ave_loss),
        ba_pre,
        ba_rec,
        ba_f1,
        (all_x_save, all_index_decoder_y, all_start_boundaries, all_end_boundaries),
    )


def train_segmenter(cfg: RstPointerSegmenterTrainArgs) -> None:
    logger.info(f"===== Training RST Pointer Segmenter =====")

    setup(seed=cfg.seed)

    train_data_dir = cfg.train_data_dir
    test_data_dir = cfg.test_data_dir
    save_dir = cfg.save_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    hidden_dim = cfg.hidden_dim
    rnn_type = cfg.rnn
    num_rnn_layers = cfg.num_rnn_layers
    lr = cfg.lr
    dropout = cfg.dropout
    wd = cfg.weight_decay
    batch_size = cfg.batch_size
    lr_decay_epoch = cfg.lr_decay_epoch
    elmo_size = cfg.elmo_size
    epochs = cfg.epochs

    use_bilstm = cfg.use_bilstm
    is_batch_norm = cfg.use_batch_norm

    tr_x = pickle.load(
        open(os.path.join(train_data_dir, "tokenized_sentences.pickle"), "rb")
    )
    tr_y = pickle.load(open(os.path.join(train_data_dir, "edu_breaks.pickle"), "rb"))

    dev_x = pickle.load(
        open(os.path.join(test_data_dir, "tokenized_sentences.pickle"), "rb")
    )
    dev_y = pickle.load(open(os.path.join(test_data_dir, "edu_breaks.pickle"), "rb"))

    model_config = RstPointerSegmenterConfig(
        hidden_dim=hidden_dim,
        dropout_prob=dropout,
        use_bilstm=use_bilstm,
        num_rnn_layers=num_rnn_layers,
        rnn_type=rnn_type,
        is_batch_norm=is_batch_norm,
        elmo_size=elmo_size,
    )
    model = RstPointerSegmenterModel(model_config)
    model.to(device=device)

    preprocessor = RstPreprocessor()

    # Arbitrary eval_size
    eval_size = len(dev_x) * 2 // 3

    test_train_x, test_train_y = sample_batch(tr_x, tr_y, eval_size)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd
    )

    num_iterations = int(np.round(len(tr_y) / batch_size))

    os.makedirs(save_dir, exist_ok=True)
    best_results_writer = CsvWriter(
        file_path=os.path.join(save_dir, "best_results.csv"),
        fieldnames=["best_epoch", "precision", "recall", "f1"],
    )
    results_writer = CsvWriter(
        file_path=os.path.join(save_dir, "results.csv"),
        fieldnames=[
            "current_epoch",
            "train_loss",
            "train_precision",
            "train_recall",
            "train_f1",
            "dev_loss",
            "dev_precision",
            "dev_recall",
            "dev_f1",
        ],
    )

    best_epoch = 0
    best_f1 = 0

    for current_epoch in range(epochs):
        adjust_learning_rate(optimizer, current_epoch, 0.8, lr_decay_epoch)

        track_epoch_loss = []
        for current_iter in range(num_iterations):
            (
                batch_x,
                batch_x_index,
                batch_y,
                all_lens,
            ) = sample_a_sorted_batch_from_numpy(tr_x, tr_y, batch_size)

            model.zero_grad()

            input_sentences_ids_batch, _ = preprocessor.get_elmo_char_ids(batch_x)
            input_sentences_ids_batch = input_sentences_ids_batch.to(
                device=model.device
            )

            output = model(input_sentences_ids_batch, all_lens, batch_y)
            loss = output.loss
            loss_value = float(loss.data)

            track_epoch_loss.append(loss_value)
            logger.info(
                f"Epoch: {current_epoch + 1}/{epochs}, "
                f"iteration: {current_iter + 1}/{num_iterations}, "
                f"loss: {loss_value:.3f}"
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        model.eval()

        logger.info(
            "Running end of epoch evaluations on sample train data and test data..."
        )
        tr_batch_ave_loss, tr_pre, tr_rec, tr_f1, tr_visdata = check_accuracy(
            model, preprocessor, test_train_x, test_train_y, batch_size
        )

        dev_batch_ave_loss, dev_pre, dev_rec, dev_f1, dev_visdata = check_accuracy(
            model, preprocessor, dev_x, dev_y, batch_size
        )
        _, _, _, all_end_boundaries = dev_visdata

        logger.info(
            f"train sample -- loss: {tr_batch_ave_loss:.3f}, "
            f"precision: {tr_pre:.3f}, recall: {tr_rec:.3f}, f1: {tr_f1:.3f}"
        )
        logger.info(
            f"test sample -- loss: {dev_batch_ave_loss:.3f}, "
            f"precision: {dev_pre:.3f}, recall: {dev_rec:.3f}, f1: {dev_f1:.3f}"
        )

        if best_f1 < dev_f1:
            best_f1 = dev_f1
            best_rec = dev_rec
            best_pre = dev_pre
            best_epoch = current_epoch

        results_writer.writerow(
            {
                "current_epoch": current_epoch,
                "train_loss": tr_batch_ave_loss,
                "train_precision": tr_pre,
                "train_recall": tr_rec,
                "train_f1": tr_f1,
                "dev_loss": dev_batch_ave_loss,
                "dev_precision": dev_pre,
                "dev_recall": dev_rec,
                "dev_f1": dev_f1,
            }
        )

        if current_epoch == best_epoch:
            logger.info("Saving best model...")
            model.save_pretrained(save_dir)

            with open(os.path.join(save_dir, "best_segmentation.pickle"), "wb") as f:
                pickle.dump(all_end_boundaries, f)

        model.train()

    best_results_writer.writerow(
        {
            "best_epoch": best_epoch,
            "precision": best_pre,
            "recall": best_rec,
            "f1": best_f1,
        }
    )


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if isinstance(cfg, RstPointerSegmenterTrainArgs):
        train_segmenter(cfg)
        print(cfg)
    if isinstance(cfg, RstPointerParserTrainArgs):
        train_parser(cfg)
        print(cfg)
