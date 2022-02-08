import os
import csv
import logging
import torch
from torch.utils.data import TensorDataset
from .tokenization import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2,
                 input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, input_mask,
                 label_id, stance_position, label_mask):
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.segment_ids1 = segment_ids1
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.input_ids3 = input_ids3
        self.input_mask3 = input_mask3
        self.segment_ids3 = segment_ids3
        self.input_ids4 = input_ids4
        self.input_mask4 = input_mask4
        self.segment_ids4 = segment_ids4
        self.input_mask = input_mask
        self.label_id = label_id
        self.stance_position = stance_position
        self.label_mask = label_mask


class InputStanceFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2,
                 input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, input_mask,
                 label_id, stance_position, label_mask):
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.segment_ids1 = segment_ids1
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.input_ids3 = input_ids3
        self.input_mask3 = input_mask3
        self.segment_ids3 = segment_ids3
        self.input_ids4 = input_ids4
        self.input_mask4 = input_mask4
        self.segment_ids4 = segment_ids4
        self.input_mask = input_mask
        self.label_id = label_id
        self.stance_position = stance_position
        self.label_mask = label_mask


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_tweet_num, max_tweet_len):
    """Loads a data file into a list of `InputBatch`s."""

    # max_tweet_len = 20  # the number of words in each tweet (42,12) for rumor verification
    # max_tweet_num = 25  # the number of tweets in each bucket
    max_bucket_num = 4  # the
    # number of buckets in each thread

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tweetlist = example.text_a
        # tweetlist = tweetlist[:max_tweet_num]
        # labellist = example.label[:max_tweet_num]
        label = example.label

        tweets_tokens = []
        for i, cur_tweet in enumerate(tweetlist):
            tweet = tweetlist[i]
            if tweet == '':
                break
            tweet_token = tokenizer.tokenize(tweet)
            if len(tweet_token) >= max_tweet_len - 1:
                tweet_token = tweet_token[:(max_tweet_len - 2)]
            tweets_tokens.append(tweet_token)

        if len(tweets_tokens) <= max_tweet_num:
            tweets_tokens1 = tweets_tokens
            tweets_tokens2, tweets_tokens3, tweets_tokens4 = [], [], []
        elif len(tweets_tokens) > max_tweet_num and len(tweets_tokens) <= max_tweet_num * 2:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:]
            tweets_tokens3, tweets_tokens4 = [], []
        elif len(tweets_tokens) > max_tweet_num * 2 and len(tweets_tokens) <= max_tweet_num * 3:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2:]
            tweets_tokens4 = []
        elif len(tweets_tokens) > max_tweet_num * 3 and len(tweets_tokens) <= max_tweet_num * 4:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2:max_tweet_num * 3]
            tweets_tokens4 = tweets_tokens[max_tweet_num * 3:]
        else:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2:max_tweet_num * 3]
            tweets_tokens4 = tweets_tokens[max_tweet_num * 3:max_tweet_num * 4]

        input_tokens1, input_ids1, input_mask1, segment_ids1, stance_position1, label_mask1 = \
            bucket_rumor_conversion(tweets_tokens1, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens2, input_ids2, input_mask2, segment_ids2, stance_position2, label_mask2 = \
            bucket_rumor_conversion(tweets_tokens2, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens3, input_ids3, input_mask3, segment_ids3, stance_position3, label_mask3 = \
            bucket_rumor_conversion(tweets_tokens3, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens4, input_ids4, input_mask4, segment_ids4, stance_position4, label_mask4 = \
            bucket_rumor_conversion(tweets_tokens4, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)

        stance_position = []
        stance_position.extend(stance_position1)
        stance_position.extend(stance_position2)
        stance_position.extend(stance_position3)
        stance_position.extend(stance_position4)
        input_mask = []
        input_mask.extend(input_mask1)
        input_mask.extend(input_mask2)
        input_mask.extend(input_mask3)
        input_mask.extend(input_mask4)
        label_mask = []
        label_mask.extend(label_mask1)
        label_mask.extend(label_mask2)
        label_mask.extend(label_mask3)
        label_mask.extend(label_mask4)

        label_id = label_map[example.label]

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in input_tokens1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids1]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask1]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids1]))
            logger.info("label: %s" % (label))

        features.append(
            InputFeatures(input_ids1=input_ids1, input_mask1=input_mask1, segment_ids1=segment_ids1,
                          input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2,
                          input_ids3=input_ids3, input_mask3=input_mask3, segment_ids3=segment_ids3,
                          input_ids4=input_ids4, input_mask4=input_mask4, segment_ids4=segment_ids4,
                          input_mask=input_mask, label_id=label_id, stance_position=stance_position,
                          label_mask=label_mask))
    return features


def bucket_rumor_conversion(tweets_tokens, tokenizer, max_tweet_num, max_tweet_len, max_seq_length):
    ntokens = []
    input_tokens = []
    input_ids = []
    input_mask = []
    segment_ids = []
    stance_position = []
    label_mask = []
    if tweets_tokens != []:
        ntokens.append("[CLS]")
        # input_tokens.extend(ntokens) # avoid having two [CLS] at the begining
        # segment_ids.append(0) #########no need to add this line
        stance_position.append(0)
        label_mask.append(1)
    for i, tweet_token in enumerate(tweets_tokens):
        if i != 0:
            ntokens = []
            ntokens.append("[CLS]")
            stance_position.append(len(input_ids))
            label_mask.append(1)
        ntokens.extend(tweet_token)
        ntokens.append("[SEP]")
        input_tokens.extend(ntokens)  # just for printing out
        input_tokens.extend("[padpadpad]")  # just for printing out
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        while len(tweet_input_ids) < max_tweet_len:
            tweet_input_ids.append(0)
            tweet_input_mask.append(0)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [i % 2] * len(tweet_input_ids)

    cur_tweet_num = len(tweets_tokens)
    pad_tweet_length = max_tweet_num - cur_tweet_num
    for j in range(pad_tweet_length):
        ntokens = []
        ntokens.append("[CLS]")
        ntokens.append("[SEP]")
        stance_position.append(len(input_ids))
        label_mask.append(0)
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        tweet_input_ids = tweet_input_ids + [0] * (max_tweet_len - 2)
        tweet_input_mask = tweet_input_mask + [0] * (max_tweet_len - 2)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [(cur_tweet_num + j) % 2] * max_tweet_len

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_tokens, input_ids, input_mask, segment_ids, stance_position, label_mask


def bucket_conversion(tweets_tokens, labels, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length):
    ntokens = []
    input_tokens = []
    input_ids = []
    input_mask = []
    segment_ids = []
    label_ids = []
    label_mask = []
    stance_position = []
    if labels != []:
        ntokens.append("[CLS]")
        # input_tokens.extend(ntokens) # avoid having two [CLS] at the begining
        # segment_ids.append(0) #########no need to add this line
        label_ids.append(label_map[labels[0]])
        stance_position.append(0)
        label_mask.append(1)
    for i, tweet_token in enumerate(tweets_tokens):
        if i != 0:
            ntokens = []
            ntokens.append("[CLS]")
            label_ids.append(label_map[labels[i]])
            stance_position.append(len(input_ids))
            label_mask.append(1)
        ntokens.extend(tweet_token)
        ntokens.append("[SEP]")
        input_tokens.extend(ntokens)  # just for printing out
        input_tokens.extend("[padpadpad]")  # just for printing out
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        while len(tweet_input_ids) < max_tweet_len:
            tweet_input_ids.append(0)
            tweet_input_mask.append(0)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [i % 2] * len(tweet_input_ids)

    cur_tweet_num = len(tweets_tokens)
    pad_tweet_length = max_tweet_num - cur_tweet_num
    for j in range(pad_tweet_length):
        ntokens = []
        ntokens.append("[CLS]")
        ntokens.append("[SEP]")
        label_ids.append(0)
        stance_position.append(len(input_ids))
        label_mask.append(0)
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        tweet_input_ids = tweet_input_ids + [0] * (max_tweet_len - 2)
        tweet_input_mask = tweet_input_mask + [0] * (max_tweet_len - 2)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [(cur_tweet_num + j) % 2] * max_tweet_len

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_tokens, input_ids, input_mask, segment_ids, label_ids, stance_position, label_mask


def convert_stance_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_tweet_num, max_tweet_len):
    """Loads a data file into a list of `InputBatch`s."""

    # max_tweet_len = 20  # the number of words in each tweet (42,12) performs best
    # max_tweet_num = 25  # the number of tweets in each bucket
    max_bucket_num = 4  # the number of buckets in each thread
    label_map_dict = {'0': 'B-DENY', '1': 'B-SUPPORT', '2': 'B-QUERY', '3': 'B-COMMENT'}

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tweetlist = example.text_a
        # tweetlist = tweetlist[:max_tweet_num]
        # labellist = example.label[:max_tweet_num]
        labellist = example.label[:max_tweet_num * max_bucket_num]

        tweets_tokens = []
        labels = []
        for i, label in enumerate(labellist):
            tweet = tweetlist[i]
            if tweet == '':
                break
            tweet_token = tokenizer.tokenize(tweet)
            if len(tweet_token) >= max_tweet_len - 1:
                tweet_token = tweet_token[:(max_tweet_len - 2)]
            tweets_tokens.append(tweet_token)
            label_1 = label
            labels.append(label_map_dict[label_1])

        if len(labels) <= max_tweet_num:
            tweets_tokens1 = tweets_tokens
            labels1 = labels
            tweets_tokens2, labels2, tweets_tokens3, labels3, tweets_tokens4, labels4 = [], [], [], [], [], []
        elif len(labels) > max_tweet_num and len(labels) <= max_tweet_num * 2:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:]
            labels2 = labels[max_tweet_num:]
            tweets_tokens3, labels3, tweets_tokens4, labels4 = [], [], [], []
        elif len(labels) > max_tweet_num * 2 and len(labels) <= max_tweet_num * 3:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num * 2]
            labels2 = labels[max_tweet_num:max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2:]
            labels3 = labels[max_tweet_num * 2:]
            tweets_tokens4, labels4 = [], []
        elif len(labels) > max_tweet_num * 3 and len(labels) <= max_tweet_num * 4:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num * 2]
            labels2 = labels[max_tweet_num:max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2:max_tweet_num * 3]
            labels3 = labels[max_tweet_num * 2:max_tweet_num * 3]
            tweets_tokens4 = tweets_tokens[max_tweet_num * 3:]
            labels4 = labels[max_tweet_num * 3:]
        else:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num * 2]
            labels2 = labels[max_tweet_num:max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2:max_tweet_num * 3]
            labels3 = labels[max_tweet_num * 2:max_tweet_num * 3]
            tweets_tokens4 = tweets_tokens[max_tweet_num * 3:max_tweet_num * 4]
            labels4 = labels[max_tweet_num * 3:max_tweet_num * 4]

        input_tokens1, input_ids1, input_mask1, segment_ids1, label_ids1, stance_position1, label_mask1 = \
            bucket_conversion(tweets_tokens1, labels1, label_map, tokenizer, max_tweet_num, max_tweet_len,
                              max_seq_length)
        input_tokens2, input_ids2, input_mask2, segment_ids2, label_ids2, stance_position2, label_mask2 = \
            bucket_conversion(tweets_tokens2, labels2, label_map, tokenizer, max_tweet_num, max_tweet_len,
                              max_seq_length)
        input_tokens3, input_ids3, input_mask3, segment_ids3, label_ids3, stance_position3, label_mask3 = \
            bucket_conversion(tweets_tokens3, labels3, label_map, tokenizer, max_tweet_num, max_tweet_len,
                              max_seq_length)
        input_tokens4, input_ids4, input_mask4, segment_ids4, label_ids4, stance_position4, label_mask4 = \
            bucket_conversion(tweets_tokens4, labels4, label_map, tokenizer, max_tweet_num, max_tweet_len,
                              max_seq_length)

        label_ids = []
        label_ids.extend(label_ids1)
        label_ids.extend(label_ids2)
        label_ids.extend(label_ids3)
        label_ids.extend(label_ids4)
        stance_position = []
        stance_position.extend(stance_position1)
        stance_position.extend(stance_position2)
        stance_position.extend(stance_position3)
        stance_position.extend(stance_position4)
        label_mask = []
        label_mask.extend(label_mask1)
        label_mask.extend(label_mask2)
        label_mask.extend(label_mask3)
        label_mask.extend(label_mask4)
        input_mask = []
        input_mask.extend(input_mask1)
        input_mask.extend(input_mask2)
        input_mask.extend(input_mask3)
        input_mask.extend(input_mask4)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in input_tokens1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids1]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask1]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids1]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids1]))

        features.append(
            InputStanceFeatures(input_ids1=input_ids1, input_mask1=input_mask1, segment_ids1=segment_ids1,
                                input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2,
                                input_ids3=input_ids3, input_mask3=input_mask3, segment_ids3=segment_ids3,
                                input_ids4=input_ids4, input_mask4=input_mask4, segment_ids4=segment_ids4,
                                input_mask=input_mask, label_id=label_ids, stance_position=stance_position,
                                label_mask=label_mask))
    return features


def prepare_data_for_training(processor, stance_processor, tokenizer, train_config):

    label_list = processor.get_labels()
    num_labels = 3  # label 0 corresponds to padding, label in label_list starts from 1

    stance_label_list = stance_processor.get_labels()
    stance_num_labels = len(stance_label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1

    train_examples = processor.get_train_examples(train_config.data_dir)
    num_train_steps = int(
        len(train_examples) / train_config.train_batch_size / train_config.gradient_accumulation_steps * train_config.num_train_epochs)
    stance_train_examples = stance_processor.get_train_examples(train_config.data_dir2)
    # enrich stance_train_examples to be the same as train examples
    src_tgt_ratio = int(len(train_examples) / len(stance_train_examples))
    remaining_samp_len = len(train_examples) - src_tgt_ratio * len(stance_train_examples)
    extended_stance_train_examples = stance_train_examples * src_tgt_ratio
    extended_stance_train_examples.extend(stance_train_examples[:remaining_samp_len])

    assert len(train_examples) == len(extended_stance_train_examples)

    # rumor detection task
    train_features = convert_examples_to_features(
        train_examples, label_list, train_config.max_seq_length, tokenizer, train_config.max_tweet_num,
        train_config.max_tweet_length)

    all_input_ids1 = torch.tensor([f.input_ids1 for f in train_features], dtype=torch.int32)
    all_input_mask1 = torch.tensor([f.input_mask1 for f in train_features], dtype=torch.int32)
    all_segment_ids1 = torch.tensor([f.segment_ids1 for f in train_features], dtype=torch.int32)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in train_features], dtype=torch.int32)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in train_features], dtype=torch.int32)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in train_features], dtype=torch.int32)
    all_input_ids3 = torch.tensor([f.input_ids3 for f in train_features], dtype=torch.int32)
    all_input_mask3 = torch.tensor([f.input_mask3 for f in train_features], dtype=torch.int32)
    all_segment_ids3 = torch.tensor([f.segment_ids3 for f in train_features], dtype=torch.int32)
    all_input_ids4 = torch.tensor([f.input_ids4 for f in train_features], dtype=torch.int32)
    all_input_mask4 = torch.tensor([f.input_mask4 for f in train_features], dtype=torch.int32)
    all_segment_ids4 = torch.tensor([f.segment_ids4 for f in train_features], dtype=torch.int32)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.int32)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in train_features], dtype=torch.int32)

    # stance classification task
    stance_train_features = convert_stance_examples_to_features(extended_stance_train_examples,
                                                                stance_label_list, train_config.max_seq_length,
                                                                tokenizer,
                                                                train_config.max_tweet_num,
                                                                train_config.max_tweet_length)

    stance_all_input_ids1 = torch.tensor([f.input_ids1 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_mask1 = torch.tensor([f.input_mask1 for f in stance_train_features], dtype=torch.int32)
    stance_all_segment_ids1 = torch.tensor([f.segment_ids1 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_ids2 = torch.tensor([f.input_ids2 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_mask2 = torch.tensor([f.input_mask2 for f in stance_train_features], dtype=torch.int32)
    stance_all_segment_ids2 = torch.tensor([f.segment_ids2 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_ids3 = torch.tensor([f.input_ids3 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_mask3 = torch.tensor([f.input_mask3 for f in stance_train_features], dtype=torch.int32)
    stance_all_segment_ids3 = torch.tensor([f.segment_ids3 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_ids4 = torch.tensor([f.input_ids4 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_mask4 = torch.tensor([f.input_mask4 for f in stance_train_features], dtype=torch.int32)
    stance_all_segment_ids4 = torch.tensor([f.segment_ids4 for f in stance_train_features], dtype=torch.int32)
    stance_all_input_mask = torch.tensor([f.input_mask for f in stance_train_features], dtype=torch.int32)
    stance_all_label_ids = torch.tensor([f.label_id for f in stance_train_features], dtype=torch.long)
    # stance_all_stance_position = torch.tensor([f.stance_position for f in stance_train_features], dtype=torch.int32)
    stance_all_label_mask = torch.tensor([f.label_mask for f in stance_train_features], dtype=torch.int32)

    train_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                               all_input_ids2, all_input_mask2, all_segment_ids2,
                               all_input_ids3, all_input_mask3, all_segment_ids3,
                               all_input_ids4, all_input_mask4, all_segment_ids4,
                               all_input_mask, all_label_ids, all_label_mask,
                               stance_all_input_ids1, stance_all_input_mask1, stance_all_segment_ids1,
                               stance_all_input_ids2, stance_all_input_mask2, stance_all_segment_ids2,
                               stance_all_input_ids3, stance_all_input_mask3, stance_all_segment_ids3,
                               stance_all_input_ids4, stance_all_input_mask4, stance_all_segment_ids4,
                               stance_all_input_mask, stance_all_label_ids, stance_all_label_mask)
    # ''' for dev evaluation
    # rumor detection task
    eval_examples = processor.get_dev_examples(train_config.data_dir)
    print('dev data')
    eval_features = convert_examples_to_features(
        eval_examples, label_list, train_config.max_seq_length, tokenizer, train_config.max_tweet_num,
        train_config.max_tweet_length)

    all_input_ids1 = torch.tensor([f.input_ids1 for f in eval_features], dtype=torch.int32)
    all_input_mask1 = torch.tensor([f.input_mask1 for f in eval_features], dtype=torch.int32)
    all_segment_ids1 = torch.tensor([f.segment_ids1 for f in eval_features], dtype=torch.int32)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in eval_features], dtype=torch.int32)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in eval_features], dtype=torch.int32)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in eval_features], dtype=torch.int32)
    all_input_ids3 = torch.tensor([f.input_ids3 for f in eval_features], dtype=torch.int32)
    all_input_mask3 = torch.tensor([f.input_mask3 for f in eval_features], dtype=torch.int32)
    all_segment_ids3 = torch.tensor([f.segment_ids3 for f in eval_features], dtype=torch.int32)
    all_input_ids4 = torch.tensor([f.input_ids4 for f in eval_features], dtype=torch.int32)
    all_input_mask4 = torch.tensor([f.input_mask4 for f in eval_features], dtype=torch.int32)
    all_segment_ids4 = torch.tensor([f.segment_ids4 for f in eval_features], dtype=torch.int32)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.int32)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.int32)

    # stance classification task
    stance_eval_examples = stance_processor.get_test_examples(train_config.data_dir2)
    # enrich stance_train_examples to be the same as train examples
    eval_src_tgt_ratio = int(len(eval_examples) / len(stance_eval_examples))
    eval_remaining_samp_len = len(eval_examples) - eval_src_tgt_ratio * len(stance_eval_examples)
    extended_stance_eval_examples = stance_eval_examples * eval_src_tgt_ratio
    extended_stance_eval_examples.extend(stance_eval_examples[:eval_remaining_samp_len])
    stance_eval_features = convert_stance_examples_to_features(extended_stance_eval_examples,
                                                               stance_label_list, train_config.max_seq_length,
                                                               tokenizer,
                                                               train_config.max_tweet_num,
                                                               train_config.max_tweet_length)

    stance_all_input_ids1 = torch.tensor([f.input_ids1 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask1 = torch.tensor([f.input_mask1 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids1 = torch.tensor([f.segment_ids1 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_ids2 = torch.tensor([f.input_ids2 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask2 = torch.tensor([f.input_mask2 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids2 = torch.tensor([f.segment_ids2 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_ids3 = torch.tensor([f.input_ids3 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask3 = torch.tensor([f.input_mask3 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids3 = torch.tensor([f.segment_ids3 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_ids4 = torch.tensor([f.input_ids4 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask4 = torch.tensor([f.input_mask4 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids4 = torch.tensor([f.segment_ids4 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask = torch.tensor([f.input_mask for f in stance_eval_features], dtype=torch.int32)
    stance_all_label_ids = torch.tensor([f.label_id for f in stance_eval_features], dtype=torch.long)
    # stance_all_stance_position = torch.tensor([f.stance_position for f in stance_eval_features], dtype=torch.int32)
    stance_all_label_mask = torch.tensor([f.label_mask for f in stance_eval_features], dtype=torch.int32)

    eval_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                              all_input_ids2, all_input_mask2, all_segment_ids2,
                              all_input_ids3, all_input_mask3, all_segment_ids3,
                              all_input_ids4, all_input_mask4, all_segment_ids4,
                              all_input_mask, all_label_ids, all_label_mask,
                              stance_all_input_ids1, stance_all_input_mask1, stance_all_segment_ids1,
                              stance_all_input_ids2, stance_all_input_mask2, stance_all_segment_ids2,
                              stance_all_input_ids3, stance_all_input_mask3, stance_all_segment_ids3,
                              stance_all_input_ids4, stance_all_input_mask4, stance_all_segment_ids4,
                              stance_all_input_mask, stance_all_label_ids, stance_all_label_mask
                              )
    # ''' for test evaluation
    # rumor detection task
    test_examples = processor.get_test_examples(train_config.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, train_config.max_seq_length, tokenizer, train_config.max_tweet_num,
        train_config.max_tweet_length)

    all_input_ids1 = torch.tensor([f.input_ids1 for f in test_features], dtype=torch.int32)
    all_input_mask1 = torch.tensor([f.input_mask1 for f in test_features], dtype=torch.int32)
    all_segment_ids1 = torch.tensor([f.segment_ids1 for f in test_features], dtype=torch.int32)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in test_features], dtype=torch.int32)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in test_features], dtype=torch.int32)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in test_features], dtype=torch.int32)
    all_input_ids3 = torch.tensor([f.input_ids3 for f in test_features], dtype=torch.int32)
    all_input_mask3 = torch.tensor([f.input_mask3 for f in test_features], dtype=torch.int32)
    all_segment_ids3 = torch.tensor([f.segment_ids3 for f in test_features], dtype=torch.int32)
    all_input_ids4 = torch.tensor([f.input_ids4 for f in test_features], dtype=torch.int32)
    all_input_mask4 = torch.tensor([f.input_mask4 for f in test_features], dtype=torch.int32)
    all_segment_ids4 = torch.tensor([f.segment_ids4 for f in test_features], dtype=torch.int32)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.int32)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in test_features], dtype=torch.int32)

    # stance classification task
    stance_eval_examples = stance_processor.get_test_examples(train_config.data_dir2)
    # enrich stance_train_examples to be the same as train examples
    eval_src_tgt_ratio = int(len(test_examples) / len(stance_eval_examples))
    eval_remaining_samp_len = len(test_examples) - eval_src_tgt_ratio * len(stance_eval_examples)
    extended_stance_eval_examples = stance_eval_examples * eval_src_tgt_ratio
    extended_stance_eval_examples.extend(stance_eval_examples[:eval_remaining_samp_len])
    stance_eval_features = convert_stance_examples_to_features(extended_stance_eval_examples,
                                                               stance_label_list, train_config.max_seq_length,
                                                               tokenizer,
                                                               train_config.max_tweet_num,
                                                               train_config.max_tweet_length)

    stance_all_input_ids1 = torch.tensor([f.input_ids1 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask1 = torch.tensor([f.input_mask1 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids1 = torch.tensor([f.segment_ids1 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_ids2 = torch.tensor([f.input_ids2 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask2 = torch.tensor([f.input_mask2 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids2 = torch.tensor([f.segment_ids2 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_ids3 = torch.tensor([f.input_ids3 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask3 = torch.tensor([f.input_mask3 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids3 = torch.tensor([f.segment_ids3 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_ids4 = torch.tensor([f.input_ids4 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask4 = torch.tensor([f.input_mask4 for f in stance_eval_features], dtype=torch.int32)
    stance_all_segment_ids4 = torch.tensor([f.segment_ids4 for f in stance_eval_features], dtype=torch.int32)
    stance_all_input_mask = torch.tensor([f.input_mask for f in stance_eval_features], dtype=torch.int32)
    stance_all_label_ids = torch.tensor([f.label_id for f in stance_eval_features], dtype=torch.long)
    # stance_all_stance_position = torch.tensor([f.stance_position for f in stance_eval_features], dtype=torch.int32)
    stance_all_label_mask = torch.tensor([f.label_mask for f in stance_eval_features], dtype=torch.int32)

    test_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                              all_input_ids2, all_input_mask2, all_segment_ids2,
                              all_input_ids3, all_input_mask3, all_segment_ids3,
                              all_input_ids4, all_input_mask4, all_segment_ids4,
                              all_input_mask, all_label_ids, all_label_mask,
                              stance_all_input_ids1, stance_all_input_mask1, stance_all_segment_ids1,
                              stance_all_input_ids2, stance_all_input_mask2, stance_all_segment_ids2,
                              stance_all_input_ids3, stance_all_input_mask3, stance_all_segment_ids3,
                              stance_all_input_ids4, stance_all_input_mask4, stance_all_segment_ids4,
                              stance_all_input_mask, stance_all_label_ids, stance_all_label_mask
                              )
    return train_data, eval_data, test_data, num_train_steps
