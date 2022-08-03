import pickle
import torch
import time
import os
import datetime
import random
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AdamW
from torch.optim.swa_utils import SWALR
from transformers import XLNetTokenizer
from modeling import MomentumModel


class MomentumDataset(Dataset):
    def __init__(self, fname, model, device, datatype, negs, max_len):
        self.fname = fname
        self.device = device
        self.data = pickle.load(open(fname, "rb"))
        random.shuffle(self.data)
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-{}-cased".format(model))
        self.truncount = 0
        self.datatype = datatype
        self.negs = negs
        self.max_len = max_len

    def pad_ids(self, ids):
        if len(ids) < self.max_len:
            padding_size = self.max_len - len(ids)
            padding = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                for i in range(padding_size)
            ]
            ids = ids + padding
        else:
            ids = ids[: self.max_len]
            self.truncount += 1

        return ids

    def prepare_data(self, idx):
        pos_doc = self.data[idx]["pos"]

        if self.datatype == "single":
            neg_docs = [self.data[idx]["neg"]]
        elif self.datatype == "multiple":
            neg_docs = self.data[idx]["negs"][: self.negs]
        else:
            raise Exception("Unexpected datatype")

        pos_span = pos_doc
        pos_span = " ".join(pos_span)
        pos_tokens = self.tokenizer.tokenize(pos_span)
        pos_ids = self.tokenizer.convert_tokens_to_ids(pos_tokens)
        pos_ids = self.pad_ids(pos_ids)

        neg_span_list = []
        for neg_doc in neg_docs:
            neg_span = neg_doc
            neg_span = " ".join(neg_span)
            neg_tokens = self.tokenizer.tokenize(neg_span)
            neg_ids = self.tokenizer.convert_tokens_to_ids(neg_tokens)
            neg_ids = self.pad_ids(neg_ids)
            neg_input = self.tokenizer.build_inputs_with_special_tokens(neg_ids)

            neg_span_list.append(torch.tensor(neg_input))

        pos_input = self.tokenizer.build_inputs_with_special_tokens(pos_ids)

        return torch.tensor(pos_input).to(self.device), torch.stack(neg_span_list).to(
            self.device
        )

    def get_slice(self, doc):
        try:
            end = random.choice(range(4, len(doc)))
            return doc[:end]
        except:
            return doc

    def prepare_train_data(self, data_list, num_negs):
        train_list = []
        for each_item in data_list:
            train_list.append(list(self.prepare_each_item(each_item, num_negs)))
        return train_list

    def prepare_each_item(self, train_data_item, num_negs):
        pos_doc = train_data_item["pos"]
        if self.datatype == "single":
            neg_docs = [train_data_item["neg"]]
        elif self.datatype == "multiple":
            neg_docs = train_data_item["negs"][:num_negs]

        pos_span = pos_doc
        pos_span = " ".join(pos_span)
        pos_tokens = self.tokenizer.tokenize(pos_span)
        pos_ids = self.tokenizer.convert_tokens_to_ids(pos_tokens)
        pos_ids = self.pad_ids(pos_ids)

        pos_slice = " ".join(self.get_slice(pos_doc))
        slice_tokens = self.tokenizer.tokenize(pos_slice)
        slice_ids = self.tokenizer.convert_tokens_to_ids(slice_tokens)
        slice_ids = self.pad_ids(slice_ids)

        neg_span_list = []
        for neg_doc in neg_docs:
            neg_span = neg_doc
            neg_span = " ".join(neg_span)
            neg_tokens = self.tokenizer.tokenize(neg_span)
            neg_ids = self.tokenizer.convert_tokens_to_ids(neg_tokens)
            neg_ids = self.pad_ids(neg_ids)

            neg_input = self.tokenizer.build_inputs_with_special_tokens(neg_ids)

            neg_span_list.append(torch.tensor(neg_input))

        pos_input = self.tokenizer.build_inputs_with_special_tokens(pos_ids)
        slice_input = self.tokenizer.build_inputs_with_special_tokens(slice_ids)

        pos_tensor = torch.tensor(pos_input).unsqueeze(0).to(self.device)
        slice_tensor = torch.tensor(slice_input).unsqueeze(0).to(self.device)
        neg_tensor_stack = torch.stack(neg_span_list).unsqueeze(0).to(self.device)

        return pos_tensor, slice_tensor, neg_tensor_stack

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.prepare_data(idx)


class LoadData:
    def __init__(
            self, fname, batch_size, model, device, datatype, negs, max_len, model_type
    ):
        self.fname = fname
        self.batch_size = batch_size
        self.dataset = MomentumDataset(fname, model, device, datatype, negs, max_len)

    def data_loader(self):
        data_sampler = SequentialSampler(self.dataset)
        loader = DataLoader(
            dataset=self.dataset, sampler=data_sampler, batch_size=self.batch_size
        )
        return loader


class TrainMomentumModel:
    def save_model(self, output_dir, step, accuracy):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        model_path = os.path.join(
            output_dir,
            "{}_seed-{}_bs-{}_lr-{}_step-{}_type-{}_acc-{}.mom".format(
                self.desc,
                self.seed,
                self.batch_size,
                self.learning_rate,
                step,
                self.model_size,
                accuracy,
            ),
        )
        # torch.save(self.xlnet_model.state_dict(), model_path)
        self.xlnet_model.save_pretrained(model_path)

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.model_size = args.model_size
        self.learning_rate = args.lr_start
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_file = args.train_file
        self.dev_file = args.dev_file
        if args.test_file:
            self.test_file = args.test_file
        else:
            self.test_file = args.dev_file
        self.negs = args.num_negs
        self.rank_negs = args.num_rank_negs
        self.train_steps = args.train_steps
        self.margin = args.margin
        self.desc = args.model_description
        self.seed = args.seed
        self.datatype = args.data_type
        self.max_len = args.max_len
        self.bestacc = 0.0
        self.model_type = args.coherence_model_type

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.xlnet_model = MomentumModel(args)
        self.xlnet_model.init_encoders()

        self.output_dir = args.output_dir + datetime.datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

        self.xlnet_model = self.xlnet_model.to(self.device)

        self.optimizer = AdamW(self.xlnet_model.parameters(), lr=self.learning_rate)
        self.scheduler = SWALR(
            self.optimizer,
            anneal_strategy="linear",
            anneal_epochs=args.lr_anneal_epochs,
            swa_lr=args.lr_end,
        )
        self.total_loss = 0.0

        self.eval_interval = args.eval_interval

    def get_ranked_negs(self, neg_scores):
        ranked_idx = sorted(
            range(len(neg_scores)), key=neg_scores.__getitem__, reverse=True
        )
        hard_negs = ranked_idx[: self.negs]
        return hard_negs

    def get_next_train_data(self, processed_exploration_data):
        self.xlnet_model.eval()

        next_train_data = []
        with torch.no_grad():
            for i, each_data in enumerate(processed_exploration_data):
                try:
                    pos_input, slice_input, neg_input = each_data
                except Exception as e:
                    print(e)
                    continue

                pos_score, neg_scores = self.xlnet_model.eval_forward(
                    pos_input, neg_input
                )
                pos_score = pos_score.to(torch.device("cpu"))
                neg_scores = neg_scores.to(torch.device("cpu"))

                next_neg_idx = self.get_ranked_negs(neg_scores)

                if len(next_neg_idx) < self.negs:
                    continue

                neg_data_list = torch.stack(
                    [neg_input[0][x] for x in next_neg_idx]
                ).unsqueeze(0)
                next_train_data.append([pos_input, slice_input, neg_data_list])

        return next_train_data

    def hard_negs_controller(self):
        start = time.time()
        train_data = MomentumDataset(
            self.train_file,
            self.model_size,
            self.device,
            self.datatype,
            self.negs,
            self.max_len,
        )
        init_train_data = train_data.data[: self.train_steps]
        total_iterations = len(train_data.data) // self.train_steps

        for iteration_index in range(total_iterations):
            full_time = time.asctime(time.localtime(time.time()))

            print(
                "ITERATION: {} TIME: {} LOSS: {}".format(
                    iteration_index, full_time, self.total_loss
                )
            )
            self.total_loss = 0.0

            if iteration_index == 0:
                processed_train_data_list = train_data.prepare_train_data(
                    init_train_data, self.negs
                )
                self.train_xlnet_model(processed_train_data_list, iteration_index)
                next_train_data = []
            else:
                start_index = iteration_index * self.train_steps
                end_index = start_index + self.train_steps

                processed_explore_data_list = train_data.prepare_train_data(
                    train_data.data[start_index:end_index], self.rank_negs
                )
                next_train_data = self.get_next_train_data(processed_explore_data_list)
                self.train_xlnet_model(next_train_data, iteration_index)

                if (self.train_steps * (iteration_index + 1)) % self.eval_interval == 0:
                    self.scheduler.step()
                    self.eval_model(
                        self.dev_file, self.train_steps * (iteration_index + 1), start
                    )

        self.eval_model(self.dev_file, self.train_steps * (iteration_index + 1), start)

    def train_xlnet_model(self, train_loader):
        self.xlnet_model.train()

        for step, data in enumerate(train_loader):

            self.optimizer.zero_grad()

            try:
                pos_input, slice_input, neg_input = data
            except Exception as e:
                print(e)
                continue

            combined_loss = self.xlnet_model(pos_input, slice_input, neg_input)
            combined_loss.backward()

            self.xlnet_model.update_momentum_encoder()
            self.optimizer.step()

            self.total_loss += combined_loss.item()

    def eval_model(self, data_file, step, start):

        print(self.desc, self.seed, "EVAL START")
        batch_size = self.batch_size
        self.xlnet_model.eval()
        test_data = LoadData(
            data_file,
            self.batch_size,
            self.model_size,
            self.device,
            self.datatype,
            self.negs,
            self.max_len,
            self.model_type,
        )
        test_loader = test_data.data_loader()

        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for data in test_loader:
                try:
                    pos_input, neg_inputs = data
                except Exception as e:
                    print(e)
                    continue

                pos_score, neg_scores = self.xlnet_model.eval_forward(
                    pos_input, neg_inputs
                )
                try:
                    max_neg_score = torch.max(neg_scores, -1).values
                except:
                    max_neg_score = max(neg_scores)

                if pos_score > max_neg_score:
                    correct += 1.0
                total += 1.0

        self.xlnet_model.train()
        end = time.time()
        full_time = time.asctime(time.localtime(end))
        acc = correct / total
        if data_file == self.dev_file:
            print(
                "DEV EVAL Time: {} Elapsed: {} Steps: {} Acc: {}".format(
                    full_time, end - start, step, acc
                )
            )
            if step > 0:
                self.bestacc = acc
                self.save_model(self.output_dir, step, acc)
        elif data_file == self.test_file:
            print(
                "Please evaluate the test file separately with the best saved checkpoint."
            )
            print(
                "TEST EVAL Time: {} Steps: {} Acc: {}".format(
                    full_time, end - start, step, acc
                )
            )

        return
