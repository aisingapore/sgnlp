# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import json
import os
import csv
import logging
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from sgnlp.models.dual_bert.config import DualBertConfig
from .utils import classification_report

from .modeling import DualBert
from transformers import AdamW
from .optimization import BertAdam
from .preprocess import prepare_data_for_training, InputExample, DualBertPreprocessor
from sklearn.metrics import precision_recall_fscore_support

from .train_args import CustomDualBertTrainConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class RumorProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text = line[2].lower().split("|||||")
            label = line[1]
            examples.append(
                InputExample(text=text, label=label)
            )
        return examples


class StanceProcessor(DataProcessor):
    """Processor for the Stance Prediction data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "stance_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["B-DENY", "B-SUPPORT", "B-QUERY", "B-COMMENT"]
        # should be ["B-DENY", "B-SUPPORT", "B-QUERY", "B-COMMENT"], but it does not matter, and we can solve this
        # at the evaluation stage

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text = line[2].lower().split("|||||")
            label = line[1].split(",")
            examples.append(
                InputExample(text=text, label=label)
            )
        return examples


class DualBertDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_data):
        self.preprocessed_data = {k: v for k, v in preprocessed_data.items() if v is not None}

    def __len__(self):
        return len(list(self.preprocessed_data.values())[0])

    def __getitem__(self, idx):
        instance = {}
        for key in self.preprocessed_data.keys():
            instance[key] = self.preprocessed_data[key][idx]
        return instance


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def rumor_macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(
        true, preds, average="macro"
    )
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def load_train_config(trainConfigPath):
    """parser = argparse.ArgumentParser(description="Custom SA Model Training")
    parser.add_argument(
        "--train_config_path",
        type=str,
        required=True,
        help="Path to training config file.",
    )

    args = parser.parse_args()"""
    with open(trainConfigPath, "r") as f:
        config = json.load(f)
    train_config = CustomDualBertTrainConfig(**config)
    return train_config


def train_custom_dual_bert(train_config: CustomDualBertTrainConfig, model_config: DualBertConfig, tokenizer=None):
    train_config.seed = 16  # 42: very good, 2: good, 64, relatively good

    processors = {
        "semeval17": RumorProcessor,
        "pheme": RumorProcessor,
        "semeval17_stance": StanceProcessor,
    }

    task_name = train_config.task_name.lower()
    task_name2 = train_config.task_name2.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    if task_name2 not in processors:
        raise ValueError("Task 2 not found: %s" % task_name2)

    rumor_processor = RumorProcessor()
    stance_processor = StanceProcessor()

    rumor_label_list = rumor_processor.get_labels()
    stance_label_list = stance_processor.get_labels()

    # train_data, eval_data, test_data, num_train_steps = prepare_data_for_training(
    #     rumor_processor, stance_processor, tokenizer, train_config, model_config
    # )

    preprocessed_data = prepare_data_for_training(
        rumor_processor, stance_processor, tokenizer, train_config, model_config
    )

    rumor_train_dataset = DualBertDataset(preprocessed_data["rumor_train_features"])
    stance_train_dataset = DualBertDataset(preprocessed_data["stance_train_features"])
    rumor_eval_dataset = DualBertDataset(preprocessed_data["rumor_eval_features"])
    stance_eval_dataset = DualBertDataset(preprocessed_data["stance_eval_features"])
    rumor_test_dataset = DualBertDataset(preprocessed_data["rumor_test_features"])
    stance_test_dataset = DualBertDataset(preprocessed_data["stance_test_features"])

    if train_config.local_rank == -1 or train_config.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not train_config.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(train_config.local_rank)
        device = torch.device("cuda", train_config.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(train_config.local_rank != -1), train_config.fp16
        )
    )

    if train_config.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                train_config.gradient_accumulation_steps
            )
        )

    train_config.train_batch_size = int(
        train_config.train_batch_size / train_config.gradient_accumulation_steps
    )

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(train_config.seed)

    if not train_config.do_train and not train_config.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if (
            os.path.exists(train_config.output_dir)
            and os.listdir(train_config.output_dir)
            and train_config.do_train
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                train_config.output_dir
            )
        )
    os.makedirs(train_config.output_dir, exist_ok=True)

    # Prepare model
    print("The current multi-task learning model is our Dual Bert model...")
    config = DualBertConfig(
        rumor_num_labels=train_config.rumor_num_labels,
        stance_num_labels=train_config.stance_num_labels,
        max_tweet_num=train_config.max_tweet_num,
        max_tweet_length=train_config.max_tweet_length,
        convert_size=train_config.convert_size,
    )
    model = DualBert(config)
    model.init_bert()

    if train_config.fp16:
        model.half()
    model.to(device)
    if train_config.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    t_total = preprocessed_data["num_train_steps"]
    if train_config.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=train_config.learning_rate,
    )
    stance_optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=train_config.learning_rate,
    )
    # For testing with original optimizer
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=train_config.learning_rate,
    #                      warmup=0.1,
    #                      t_total=t_total)
    # stance_optimizer = BertAdam(optimizer_grouped_parameters,
    #                             lr=train_config.learning_rate,
    #                             warmup=0.1,
    #                             t_total=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_model_file = os.path.join(train_config.output_dir, "pytorch_model.bin")
    if train_config.do_train:
        print("training data")

        if train_config.local_rank == -1:
            rumor_train_sampler = RandomSampler(rumor_train_dataset)
            stance_train_sampler = RandomSampler(stance_train_dataset)
        else:
            rumor_train_sampler = DistributedSampler(rumor_train_dataset)
            stance_train_sampler = DistributedSampler(stance_train_dataset)

        rumor_train_dataloader = DataLoader(
            rumor_train_dataset, sampler=rumor_train_sampler, batch_size=train_config.train_batch_size
        )
        stance_train_dataloader = DataLoader(
            stance_train_dataset, sampler=stance_train_sampler, batch_size=train_config.train_batch_size
        )

        # Run prediction for full data
        rumor_eval_sampler = SequentialSampler(rumor_eval_dataset)
        rumor_eval_dataloader = DataLoader(
            rumor_eval_dataset, sampler=rumor_eval_sampler, batch_size=train_config.eval_batch_size
        )
        stance_eval_sampler = SequentialSampler(stance_eval_dataset)
        stance_eval_dataloader = DataLoader(
            stance_eval_dataset, sampler=stance_eval_sampler, batch_size=train_config.eval_batch_size
        )

        # Run prediction for full data
        rumor_test_sampler = SequentialSampler(rumor_test_dataset)
        rumor_test_dataloader = DataLoader(
            rumor_test_dataset, sampler=rumor_test_sampler, batch_size=train_config.eval_batch_size
        )
        stance_test_sampler = SequentialSampler(stance_test_dataset)
        stance_test_dataloader = DataLoader(
            stance_test_dataset, sampler=stance_test_sampler, batch_size=train_config.eval_batch_size
        )

        max_acc_f1 = 0.0
        logger.info("*************** Running training ***************")

        for train_idx in trange(int(train_config.num_train_epochs), desc="Epoch"):

            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Batch size = %d", train_config.train_batch_size)
            logger.info("  Num steps = %d", preprocessed_data["num_train_steps"])
            model.train()
            tr_loss = 0
            stance_tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            stance_nb_tr_examples, stance_nb_tr_steps = 0, 0
            for step, (rumor_batch, stance_batch) in enumerate(
                    tqdm(zip(rumor_train_dataloader, stance_train_dataloader), desc="Iteration")):
                rumor_batch = {k: v.to(device) for k, v in rumor_batch.items()}
                stance_batch = {k: v.to(device) for k, v in stance_batch.items()}
                # (
                #     input_ids1,
                #     input_mask1,
                #     segment_ids1,
                #     input_ids2,
                #     input_mask2,
                #     segment_ids2,
                #     input_ids3,
                #     input_mask3,
                #     segment_ids3,
                #     input_ids4,
                #     input_mask4,
                #     segment_ids4,
                #     input_mask,
                #     label_ids,
                #     label_mask,
                #     stance_input_ids1,
                #     stance_input_mask1,
                #     stance_segment_ids1,
                #     stance_input_ids2,
                #     stance_input_mask2,
                #     stance_segment_ids2,
                #     stance_input_ids3,
                #     stance_input_mask3,
                #     stance_segment_ids3,
                #     stance_input_ids4,
                #     stance_input_mask4,
                #     stance_segment_ids4,
                #     stance_input_mask,
                #     stance_label_ids,
                #     stance_label_mask,
                # ) = batch

                # optimize rumor detection task
                tmp_model_output = model(**rumor_batch)
                # tmp_model_output = model(
                #     input_ids1,
                #     segment_ids1,
                #     input_mask1,
                #     input_ids2,
                #     segment_ids2,
                #     input_mask2,
                #     input_ids3,
                #     segment_ids3,
                #     input_mask3,
                #     input_ids4,
                #     segment_ids4,
                #     input_mask4,
                #     input_mask,
                #     rumor_labels=label_ids,
                #     stance_label_mask=label_mask,
                # )
                loss = tmp_model_output.rumour_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if train_config.gradient_accumulation_steps > 1:
                    loss = loss / train_config.gradient_accumulation_steps

                if train_config.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                # nb_tr_examples += input_ids1.size(0)
                nb_tr_examples += rumor_batch["input_ids_buckets"][0].size(0)
                nb_tr_steps += 1
                if (step + 1) % train_config.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = train_config.learning_rate * warmup_linear(
                        global_step / t_total, train_config.warmup_proportion
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # optimize stance classification task
                tmp_model_output = model(**stance_batch)
                # tmp_model_output = model(
                #     stance_input_ids1,
                #     stance_segment_ids1,
                #     stance_input_mask1,
                #     stance_input_ids2,
                #     stance_segment_ids2,
                #     stance_input_mask2,
                #     stance_input_ids3,
                #     stance_segment_ids3,
                #     stance_input_mask3,
                #     stance_input_ids4,
                #     stance_segment_ids4,
                #     stance_input_mask4,
                #     stance_input_mask,
                #     stance_labels=stance_label_ids,
                #     stance_label_mask=stance_label_mask,
                # )
                stance_loss = tmp_model_output.stance_loss
                if n_gpu > 1:
                    stance_loss = stance_loss.mean()  # mean() to average on multi-gpu.
                if train_config.gradient_accumulation_steps > 1:
                    stance_loss = stance_loss / train_config.gradient_accumulation_steps

                if train_config.fp16:
                    stance_optimizer.backward(stance_loss)
                else:
                    stance_loss.backward()

                stance_tr_loss += stance_loss.item()
                # stance_nb_tr_examples += stance_input_ids1.size(0)
                stance_nb_tr_examples += stance_batch["input_ids_buckets"][0].size(0)
                stance_nb_tr_steps += 1
                if (step + 1) % train_config.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    stance_lr_this_step = train_config.learning_rate * warmup_linear(
                        global_step / t_total, train_config.warmup_proportion
                    )
                    for param_group in stance_optimizer.param_groups:
                        param_group["lr"] = stance_lr_this_step
                    stance_optimizer.step()
                    stance_optimizer.zero_grad()

            logger.info(
                "\n************************************************** Running evaluation on Dev Set****************************************"
            )
            # logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", train_config.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []

            stance_y_true = []
            stance_y_pred = []

            stance_label_map = {
                i: label for i, label in enumerate(stance_label_list, 1)
            }
            stance_label_map[0] = "PAD"

            # for (
            #         input_ids1,
            #         input_mask1,
            #         segment_ids1,
            #         input_ids2,
            #         input_mask2,
            #         segment_ids2,
            #         input_ids3,
            #         input_mask3,
            #         segment_ids3,
            #         input_ids4,
            #         input_mask4,
            #         segment_ids4,
            #         input_mask,
            #         label_ids,
            #         label_mask,
            #         stance_input_ids1,
            #         stance_input_mask1,
            #         stance_segment_ids1,
            #         stance_input_ids2,
            #         stance_input_mask2,
            #         stance_segment_ids2,
            #         stance_input_ids3,
            #         stance_input_mask3,
            #         stance_segment_ids3,
            #         stance_input_ids4,
            #         stance_input_mask4,
            #         stance_segment_ids4,
            #         stance_input_mask,
            #         stance_label_ids,
            #         stance_label_mask,
            # ) in tqdm(eval_dataloader, desc="Evaluating"):
            #     input_ids1 = input_ids1.to(device)
            #     input_mask1 = input_mask1.to(device)
            #     segment_ids1 = segment_ids1.to(device)
            #     input_ids2 = input_ids2.to(device)
            #     input_mask2 = input_mask2.to(device)
            #     segment_ids2 = segment_ids2.to(device)
            #     input_ids3 = input_ids3.to(device)
            #     input_mask3 = input_mask3.to(device)
            #     segment_ids3 = segment_ids3.to(device)
            #     input_ids4 = input_ids4.to(device)
            #     input_mask4 = input_mask4.to(device)
            #     segment_ids4 = segment_ids4.to(device)
            #     input_mask = input_mask.to(device)
            #     label_ids = label_ids.to(device)
            #     label_mask = label_mask.to(device)
            #
            #     stance_input_ids1 = stance_input_ids1.to(device)
            #     stance_input_mask1 = stance_input_mask1.to(device)
            #     stance_segment_ids1 = stance_segment_ids1.to(device)
            #     stance_input_ids2 = stance_input_ids2.to(device)
            #     stance_input_mask2 = stance_input_mask2.to(device)
            #     stance_segment_ids2 = stance_segment_ids2.to(device)
            #     stance_input_ids3 = stance_input_ids3.to(device)
            #     stance_input_mask3 = stance_input_mask3.to(device)
            #     stance_segment_ids3 = stance_segment_ids3.to(device)
            #     stance_input_ids4 = stance_input_ids4.to(device)
            #     stance_input_mask4 = stance_input_mask4.to(device)
            #     stance_segment_ids4 = stance_segment_ids4.to(device)
            #     stance_input_mask = stance_input_mask.to(device)
            #     stance_label_ids = stance_label_ids.to(device)
            #     # stance_stance_position = stance_stance_position.to(device)
            #     stance_label_mask = stance_label_mask.to(device)
            for rumor_batch, stance_batch in tqdm(zip(rumor_eval_dataloader, stance_eval_dataloader),
                                                  desc="Evaluating"):
                with torch.no_grad():
                    rumor_batch = {k: v.to(device) for k, v in rumor_batch.items()}
                    tmp_model_output = model(**rumor_batch)
                    tmp_eval_loss = tmp_model_output.rumour_loss
                    logits = tmp_model_output.rumour_logits
                    stance_logits = tmp_model_output.stance_logits

                logits = logits.detach().cpu().numpy()
                # label_ids = label_ids.to("cpu").numpy()
                label_ids = rumor_batch["rumor_label_ids"].to("cpu").numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                stance_logits = torch.argmax(F.log_softmax(stance_logits, dim=2), dim=2)
                stance_logits = stance_logits.detach().cpu().numpy()
                # stance_label_ids = stance_label_ids.to("cpu").numpy()
                stance_label_ids = stance_batch["stance_label_ids"].to("cpu").numpy()
                # stance_label_mask = stance_label_mask.to("cpu").numpy()
                stance_label_mask = stance_batch["stance_label_mask"].to("cpu").numpy()
                for i, mask in enumerate(stance_label_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if m:
                            temp_1.append(stance_label_map[stance_label_ids[i][j]])
                            temp_2.append(stance_label_map[stance_logits[i][j]])
                        else:
                            break
                    stance_y_true.append(temp_1)
                    stance_y_pred.append(temp_2)

                # nb_eval_examples += input_ids1.size(0)
                nb_eval_examples += rumor_batch["input_ids_buckets"][0].size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if train_config.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = rumor_macro_f1(true_label, pred_outputs)

            stance_report = classification_report(
                stance_y_true, stance_y_pred, digits=4
            )
            logger.info("\n***** Dev Stance Eval results *****")
            logger.info("\n%s", stance_report)
            stance_eval_true_label = np.concatenate(stance_y_true)
            stance_eval_pred_label = np.concatenate(stance_y_pred)
            stance_precision, stance_recall, stance_F_score = macro_f1(
                stance_eval_true_label, stance_eval_pred_label
            )
            print("stance_F-score: ", stance_F_score)

            result = {
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "f_score": F_score,
                "global_step": global_step,
                "loss": loss,
            }

            logger.info("\n***** Dev Rumor Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if (
                    eval_accuracy + F_score > max_acc_f1
            ):  # if eval_accuracy+F_score+stance_F_score >= max_acc_f1:
                # Save a trained model
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Only save the model it-self
                if train_config.do_train:
                    # torch.save(model_to_save.state_dict(), output_model_file)
                    model.save_pretrained(train_config.output_dir)
                max_acc_f1 = eval_accuracy + F_score
                # max_acc_f1 = eval_accuracy+F_score+stance_F_score

            logger.info(
                "\n************************************************** Running evaluation on Test Set****************************************"
            )
            # logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", train_config.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []

            stance_y_true = []
            stance_y_pred = []

            stance_label_map = {
                i: label for i, label in enumerate(stance_label_list, 1)
            }
            stance_label_map[0] = "PAD"

            # for (
            #         input_ids1,
            #         input_mask1,
            #         segment_ids1,
            #         input_ids2,
            #         input_mask2,
            #         segment_ids2,
            #         input_ids3,
            #         input_mask3,
            #         segment_ids3,
            #         input_ids4,
            #         input_mask4,
            #         segment_ids4,
            #         input_mask,
            #         label_ids,
            #         label_mask,
            #         stance_input_ids1,
            #         stance_input_mask1,
            #         stance_segment_ids1,
            #         stance_input_ids2,
            #         stance_input_mask2,
            #         stance_segment_ids2,
            #         stance_input_ids3,
            #         stance_input_mask3,
            #         stance_segment_ids3,
            #         stance_input_ids4,
            #         stance_input_mask4,
            #         stance_segment_ids4,
            #         stance_input_mask,
            #         stance_label_ids,
            #         stance_label_mask,
            # ) in tqdm(test_dataloader, desc="Evaluating"):
            #     input_ids1 = input_ids1.to(device)
            #     input_mask1 = input_mask1.to(device)
            #     segment_ids1 = segment_ids1.to(device)
            #     input_ids2 = input_ids2.to(device)
            #     input_mask2 = input_mask2.to(device)
            #     segment_ids2 = segment_ids2.to(device)
            #     input_ids3 = input_ids3.to(device)
            #     input_mask3 = input_mask3.to(device)
            #     segment_ids3 = segment_ids3.to(device)
            #     input_ids4 = input_ids4.to(device)
            #     input_mask4 = input_mask4.to(device)
            #     segment_ids4 = segment_ids4.to(device)
            #     input_mask = input_mask.to(device)
            #     label_ids = label_ids.to(device)
            #     label_mask = label_mask.to(device)
            #
            #     stance_input_ids1 = stance_input_ids1.to(device)
            #     stance_input_mask1 = stance_input_mask1.to(device)
            #     stance_segment_ids1 = stance_segment_ids1.to(device)
            #     stance_input_ids2 = stance_input_ids2.to(device)
            #     stance_input_mask2 = stance_input_mask2.to(device)
            #     stance_segment_ids2 = stance_segment_ids2.to(device)
            #     stance_input_ids3 = stance_input_ids3.to(device)
            #     stance_input_mask3 = stance_input_mask3.to(device)
            #     stance_segment_ids3 = stance_segment_ids3.to(device)
            #     stance_input_ids4 = stance_input_ids4.to(device)
            #     stance_input_mask4 = stance_input_mask4.to(device)
            #     stance_segment_ids4 = stance_segment_ids4.to(device)
            #     stance_input_mask = stance_input_mask.to(device)
            #     stance_label_ids = stance_label_ids.to(device)
            #     # stance_stance_position = stance_stance_position.to(device)
            #     stance_label_mask = stance_label_mask.to(device)
            for (rumor_batch, stance_batch) in tqdm(zip(rumor_test_dataloader, stance_test_dataloader),
                                                    desc="Evaluating"):

                with torch.no_grad():
                    tmp_model_output = model(**rumor_batch)
                    # tmp_model_output = model(
                    #     input_ids1,
                    #     segment_ids1,
                    #     input_mask1,
                    #     input_ids2,
                    #     segment_ids2,
                    #     input_mask2,
                    #     input_ids3,
                    #     segment_ids3,
                    #     input_mask3,
                    #     input_ids4,
                    #     segment_ids4,
                    #     input_mask4,
                    #     input_mask,
                    #     label_ids,
                    #     stance_label_mask=label_mask,
                    # )
                    tmp_eval_loss = tmp_model_output.rumour_loss
                    logits = tmp_model_output.rumour_logits

                    tmp_model_output = model(**stance_batch)
                    # tmp_model_output = model(
                    #     stance_input_ids1,
                    #     stance_segment_ids1,
                    #     stance_input_mask1,
                    #     stance_input_ids2,
                    #     stance_segment_ids2,
                    #     stance_input_mask2,
                    #     stance_input_ids3,
                    #     stance_segment_ids3,
                    #     stance_input_mask3,
                    #     stance_input_ids4,
                    #     stance_segment_ids4,
                    #     stance_input_mask4,
                    #     stance_input_mask,
                    #     stance_label_mask=stance_label_mask,
                    # )
                    stance_logits = tmp_model_output.stance_logits

                logits = logits.detach().cpu().numpy()
                # label_ids = label_ids.to("cpu").numpy()
                label_ids = rumor_batch["rumor_label_ids"].to("cpu").numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                stance_logits = torch.argmax(F.log_softmax(stance_logits, dim=2), dim=2)
                stance_logits = stance_logits.detach().cpu().numpy()
                stance_label_ids = stance_batch["stance_label_ids"].to("cpu").numpy()
                stance_label_mask = stance_batch["stance_label_mask"].to("cpu").numpy()
                for i, mask in enumerate(stance_label_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if m:
                            temp_1.append(stance_label_map[stance_label_ids[i][j]])
                            temp_2.append(stance_label_map[stance_logits[i][j]])
                        else:
                            break
                    stance_y_true.append(temp_1)
                    stance_y_pred.append(temp_2)

                # nb_eval_examples += input_ids1.size(0)
                nb_eval_examples += rumor_batch["input_ids_buckets"][0].size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if train_config.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = rumor_macro_f1(true_label, pred_outputs)

            stance_report = classification_report(
                stance_y_true, stance_y_pred, digits=4
            )
            logger.info("\n***** Test Stance Eval results *****")
            logger.info("\n%s", stance_report)
            stance_eval_true_label = np.concatenate(stance_y_true)
            stance_eval_pred_label = np.concatenate(stance_y_pred)
            stance_precision, stance_recall, stance_F_score = macro_f1(
                stance_eval_true_label, stance_eval_pred_label
            )
            print("stance_F-score: ", stance_F_score)

            result = {
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "f_score": F_score,
                "global_step": global_step,
                "loss": loss,
            }

            logger.info("\n***** Test Rumor Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

    # Load a trained model that you have fine-tuned

    # model_state_dict = torch.load(output_model_file)

    # config = DualBertConfig(rumor_num_labels=train_config.rumor_num_labels,
    #     stance_num_labels=train_config.stance_num_labels,
    #     max_tweet_num=train_config.max_tweet_num,
    #     max_tweet_length=train_config.max_tweet_length,
    #     convert_size=train_config.convert_size,)
    # model = DualBert(config)

    # model = DualBert.from_pretrained(
    #     train_config.bert_model,
    #     state_dict=model_state_dict,
    #     rumor_num_labels=train_config.rumor_num_labels,
    #     stance_num_labels=train_config.stance_num_labels,
    #     max_tweet_num=train_config.max_tweet_num,
    #     max_tweet_length=train_config.max_tweet_length,
    #     convert_size=train_config.convert_size,
    # )
    if os.listdir(train_config.output_dir):
        config = DualBertConfig.from_pretrained(os.path.join(train_config.output_dir, "config.json"))
        model = DualBert.from_pretrained(output_model_file, config=config)

    model.to(device)

    if train_config.do_eval and (
            train_config.local_rank == -1 or torch.distributed.get_rank() == 0
    ):
        eval_examples = rumor_processor.get_test_examples(train_config.data_dir)
        stance_eval_examples = stance_processor.get_test_examples(
            train_config.data_dir2
        )
        tweets_list = []
        stances_list = []
        for (ex_index, example) in enumerate(eval_examples):
            tweets_list.append(example.text)
        for (ex_index, example) in enumerate(stance_eval_examples):
            stances_list.append(example.label)

        preprocessor = DualBertPreprocessor(model_config, tokenizer)
        eval_features = preprocessor(eval_examples, task="rumor", label_names=rumor_label_list)
        # eval_features = convert_examples_to_features(
        #     eval_examples,
        #     rumor_label_list,
        #     train_config.max_seq_length,
        #     tokenizer,
        #     train_config.max_tweet_num,
        #     train_config.max_tweet_length,
        # )
        logger.info("\n***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", train_config.eval_batch_size)
        # all_input_ids1 = torch.tensor(
        #     [f.input_ids1 for f in eval_features], dtype=torch.int32
        # )
        # all_input_mask1 = torch.tensor(
        #     [f.input_mask1 for f in eval_features], dtype=torch.int32
        # )
        # all_segment_ids1 = torch.tensor(
        #     [f.segment_ids1 for f in eval_features], dtype=torch.int32
        # )
        # all_input_ids2 = torch.tensor(
        #     [f.input_ids2 for f in eval_features], dtype=torch.int32
        # )
        # all_input_mask2 = torch.tensor(
        #     [f.input_mask2 for f in eval_features], dtype=torch.int32
        # )
        # all_segment_ids2 = torch.tensor(
        #     [f.segment_ids2 for f in eval_features], dtype=torch.int32
        # )
        # all_input_ids3 = torch.tensor(
        #     [f.input_ids3 for f in eval_features], dtype=torch.int32
        # )
        # all_input_mask3 = torch.tensor(
        #     [f.input_mask3 for f in eval_features], dtype=torch.int32
        # )
        # all_segment_ids3 = torch.tensor(
        #     [f.segment_ids3 for f in eval_features], dtype=torch.int32
        # )
        # all_input_ids4 = torch.tensor(
        #     [f.input_ids4 for f in eval_features], dtype=torch.int32
        # )
        # all_input_mask4 = torch.tensor(
        #     [f.input_mask4 for f in eval_features], dtype=torch.int32
        # )
        # all_segment_ids4 = torch.tensor(
        #     [f.segment_ids4 for f in eval_features], dtype=torch.int32
        # )
        # all_input_mask = torch.tensor(
        #     [f.input_mask for f in eval_features], dtype=torch.int32
        # )
        # all_label_ids = torch.tensor(
        #     [f.label_id for f in eval_features], dtype=torch.long
        # )
        # all_label_mask = torch.tensor(
        #     [f.label_mask for f in eval_features], dtype=torch.int32
        # )

        eval_dataset = DualBertDataset(eval_features)
        # eval_data = TensorDataset(
        #     all_input_ids1,
        #     all_input_mask1,
        #     all_segment_ids1,
        #     all_input_ids2,
        #     all_input_mask2,
        #     all_segment_ids2,
        #     all_input_ids3,
        #     all_input_mask3,
        #     all_segment_ids3,
        #     all_input_ids4,
        #     all_input_mask4,
        #     all_segment_ids4,
        #     all_input_mask,
        #     all_label_ids,
        #     all_label_mask,
        # )
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=train_config.eval_batch_size
        )

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        true_label_list = []
        pred_label_list = []
        attention_probs_list = []
        stance_pred_label_list = []
        stance_label_map = {i: label for i, label in enumerate(stance_label_list, 1)}
        stance_label_map[0] = "PAD"
        # convert_stance_label_map = {-1: -1, 0: 0, 1: 2, 2: 1, 3: 3}  # this will solve the problem above

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            # input_ids1 = input_ids1.to(device)
            # input_mask1 = input_mask1.to(device)
            # segment_ids1 = segment_ids1.to(device)
            # input_ids2 = input_ids2.to(device)
            # input_mask2 = input_mask2.to(device)
            # segment_ids2 = segment_ids2.to(device)
            # input_ids3 = input_ids3.to(device)
            # input_mask3 = input_mask3.to(device)
            # segment_ids3 = segment_ids3.to(device)
            # input_ids4 = input_ids4.to(device)
            # input_mask4 = input_mask4.to(device)
            # segment_ids4 = segment_ids4.to(device)
            # input_mask = input_mask.to(device)
            # label_ids = label_ids.to(device)
            # label_mask = label_mask.to(device)

            with torch.no_grad():
                tmp_model_output = model(**batch)
                # tmp_model_output = model(
                #     input_ids1,
                #     segment_ids1,
                #     input_mask1,
                #     input_ids2,
                #     segment_ids2,
                #     input_mask2,
                #     input_ids3,
                #     segment_ids3,
                #     input_mask3,
                #     input_ids4,
                #     segment_ids4,
                #     input_mask4,
                #     input_mask,
                #     label_ids,
                #     stance_label_mask=label_mask,
                # )

                tmp_eval_loss = tmp_model_output.rumour_loss
                logits = tmp_model_output.rumour_logits
                stance_logits = tmp_model_output.stance_logits
                attention_probs = tmp_model_output.attention_probs

            logits = logits.detach().cpu().numpy()
            attention_probs = attention_probs.detach().cpu().numpy()
            label_ids = batch["rumor_label_ids"].to("cpu").numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            if train_config.mt_model == "DB":
                stance_logits = torch.argmax(F.log_softmax(stance_logits, dim=2), dim=2)
                stance_logits = stance_logits.detach().cpu().numpy()
                stance_label_mask = batch["stance_label_mask"].to("cpu").numpy()
                for i, mask in enumerate(stance_label_mask):
                    temp_1 = []
                    attention_probs_temp_1 = []
                    for j, m in enumerate(mask):
                        if m:
                            temp_1.append(str(stance_logits[i][j] - 1))
                            attention_probs_temp_1.append(str(attention_probs[i][j]))
                            # 4 to 3, 3 to 2, 2 to 1, 1 to 0, same as original
                        else:
                            break
                    stance_pred_label_list.append(temp_1)
                    attention_probs_list.append(attention_probs_temp_1)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            # nb_eval_examples += input_ids1.size(0)
            nb_eval_examples += batch["input_ids_buckets"][0].size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if train_config.do_train else None
        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)
        # stance_pred_outputs_numpy = np.concatenate(stance_pred_label_list)

        precision, recall, F_score = rumor_macro_f1(true_label, pred_outputs)
        result = {
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "precision": precision,
            "recall": recall,
            "f_score": F_score,
            "global_step": global_step,
            "loss": loss,
        }

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = open(os.path.join(train_config.output_dir, "pred.txt"), "w")
        fout_t = open(os.path.join(train_config.output_dir, "true.txt"), "w")

        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + "\n")
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + "\n")

        fout_p.close()
        fout_t.close()
        FR_stance_dict = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
        }  # deny, support, query, comment
        TR_stance_dict = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
        }  # deny, support, query, comment
        UR_stance_dict = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
        }  # deny, support, query, comment
        if (
                train_config.task_name.lower() == "semeval17"
                and train_config.mt_model == "DB"
        ):
            fout_analysis = open(
                os.path.join(train_config.output_dir, "analysis.txt"), "w"
            )
            for i in range(len(pred_label)):
                fout_analysis.write("Test Sample: " + str(i) + "\n")
                tweets = tweets_list[i]
                fout_analysis.write("|||||".join(tweets) + "\n")
                pred_stances = stance_pred_label_list[i]
                fout_analysis.write(
                    "predicted stance label: " + ",".join(pred_stances) + "\n"
                )
                stances = stances_list[i]
                fout_analysis.write(
                    "original stance label:  " + ",".join(stances) + "\n"
                )
                attstr = str(pred_label[i])
                fout_analysis.write("predicted rumor label: " + attstr + "\n")
                attstr = str(true_label[i])
                fout_analysis.write("true rumor label: " + attstr + "\n\n")
                if pred_label[i] == 0:
                    for stance in pred_stances:
                        FR_stance_dict[stance] += 1
                elif pred_label[i] == 1:
                    for stance in pred_stances:
                        TR_stance_dict[stance] += 1
                elif pred_label[i] == 2:
                    for stance in pred_stances:
                        UR_stance_dict[stance] += 1
            fr_total = float(
                FR_stance_dict["0"] + FR_stance_dict["1"] + FR_stance_dict["2"]
            )
            tr_total = float(
                TR_stance_dict["0"] + TR_stance_dict["1"] + TR_stance_dict["2"]
            )
            ur_total = float(
                UR_stance_dict["0"] + UR_stance_dict["1"] + UR_stance_dict["2"]
            )
            fout_analysis.write(
                "\n\n\nFalse Rumor------"
                + "deny: "
                + str(FR_stance_dict["0"] / fr_total if fr_total > 0 else 0)
                + "\t"
                + "support: "
                + str(FR_stance_dict["1"] / fr_total if fr_total > 0 else 0)
                + "\t"
                + "query: "
                + str(FR_stance_dict["2"] / fr_total if fr_total > 0 else 0)
                + "\n"
            )
            fout_analysis.write(
                "True Rumor------"
                + "deny: "
                + str(TR_stance_dict["0"] / tr_total if tr_total > 0 else 0)
                + "\t"
                + "support: "
                + str(TR_stance_dict["1"] / tr_total if tr_total > 0 else 0)
                + "\t"
                + "query: "
                + str(TR_stance_dict["2"] / tr_total if tr_total > 0 else 0)
                + "\n"
            )
            fout_analysis.write(
                "Unverified Rumor------"
                + "deny: "
                + str(UR_stance_dict["0"] / ur_total if ur_total > 0 else 0)
                + "\t"
                + "support: "
                + str(UR_stance_dict["1"] / ur_total if ur_total > 0 else 0)
                + "\t"
                + "query: "
                + str(UR_stance_dict["2"] / ur_total if ur_total > 0 else 0)
                + "\n"
            )

            fout_analysis.close()
        elif (
                train_config.task_name.lower() == "pheme" and train_config.mt_model == "DB"
        ):
            fout_analysis = open(
                os.path.join(train_config.output_dir, "analysis.txt"), "w"
            )
            for i in range(len(pred_label)):
                fout_analysis.write("Test Sample: " + str(i) + "\n")
                tweets = tweets_list[i]
                fout_analysis.write("|||||".join(tweets) + "\n")
                pred_stances = stance_pred_label_list[i]
                fout_analysis.write(
                    "predicted stance label: " + ",".join(pred_stances) + "\n"
                )
                attstr = str(pred_label[i])
                fout_analysis.write("predicted rumor label: " + attstr + "\n")
                attstr = str(true_label[i])
                fout_analysis.write("true rumor label: " + attstr + "\n\n")
                if pred_label[i] == 0:
                    for stance in pred_stances:
                        FR_stance_dict[stance] += 1
                elif pred_label[i] == 1:
                    for stance in pred_stances:
                        TR_stance_dict[stance] += 1
                elif pred_label[i] == 2:
                    for stance in pred_stances:
                        UR_stance_dict[stance] += 1

            fr_total = float(
                FR_stance_dict["0"] + FR_stance_dict["1"] + FR_stance_dict["2"]
            )
            tr_total = float(
                TR_stance_dict["0"] + TR_stance_dict["1"] + TR_stance_dict["2"]
            )
            ur_total = float(
                UR_stance_dict["0"] + UR_stance_dict["1"] + UR_stance_dict["2"]
            )
            fout_analysis.write(
                "\n\n\nFalse Rumor------"
                + "deny: "
                + str(FR_stance_dict["0"] / fr_total if fr_total > 0 else 0)
                + "\t"
                + "support: "
                + str(FR_stance_dict["1"] / fr_total if fr_total > 0 else 0)
                + "\t"
                + "query: "
                + str(FR_stance_dict["2"] / fr_total if fr_total > 0 else 0)
                + "\n"
            )
            fout_analysis.write(
                "True Rumor------"
                + "deny: "
                + str(TR_stance_dict["0"] / tr_total if tr_total > 0 else 0)
                + "\t"
                + "support: "
                + str(TR_stance_dict["1"] / tr_total if tr_total > 0 else 0)
                + "\t"
                + "query: "
                + str(TR_stance_dict["2"] / tr_total if tr_total > 0 else 0)
                + "\n"
            )
            fout_analysis.write(
                "Unverified Rumor------"
                + "deny: "
                + str(UR_stance_dict["0"] / ur_total if ur_total > 0 else 0)
                + "\t"
                + "support: "
                + str(UR_stance_dict["1"] / ur_total if ur_total > 0 else 0)
                + "\t"
                + "query: "
                + str(UR_stance_dict["2"] / ur_total if ur_total > 0 else 0)
                + "\n"
            )

            fout_analysis.close()

        output_eval_file = os.path.join(train_config.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("\n***** Test Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    # train_config = load_train_config("/Users/nus/Documents/Code/projects/SGnlp/sgnlp/sgnlp/models/dual_bert/train_config_local.json")
    train_config = load_train_config("/polyaxon-data/workspace/atenzer/CHT_demo/train_config.json")
    model_config = DualBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/config.json")
    train_custom_dual_bert(train_config, model_config)
