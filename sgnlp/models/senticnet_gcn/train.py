import datetime
import logging
import math
import pathlib

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim

from data_class import SenticNetGCNTrainArgs
from sgnlp.models.senticnet_gcn.modeling import SenticNetBertGCNPreTrainedModel
from utils import parse_args_and_load_config, set_random_seed


logging.basicConfig(level=logging.DEBUG)


class SenticNetGCNBaseTrainer:
    def __init__(self, config: SenticNetGCNTrainArgs):
        self.config = config
        self.global_max_acc = 0.0
        self.global_max_f1 = 0.0
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not self.config.device
            else torch.device(self.config.device)
        )
        # self.dataloader_train = # dataloader
        # self.dataloader_test = # dataloader
        if config.save_state_dict:
            self.save_state_dict_folder = pathlib.Path(self.config.saved_state_dict_folder_path)
            self.save_state_dict_folder.mkdir(exist_ok=True)

    def _create_initializers(self):
        initializers = {
            "xavier_uniform_": nn.init.xavier_uniform_,
            "xavier_normal_": nn.init.xavier_normal,
            "orthogonal": nn.init.orthogonal_,
        }
        return initializers[self.config.initializer]

    def _create_optimizer(self):
        optimizers = {
            "adadelta": optim.Adadelta,
            "adagrad": optim.Adagrad,
            "adam": optim.Adam,
            "adamax": optim.Adamax,
            "asgd": optim.ASGD,
            "rmsprop": optim.RMSprop,
            "sgd": optim.SGD,
        }
        return optimizers[self.config.optimizer]

    def _reset_params(self):
        raise NotImplementedError("Please call from derived class only.")

    def _evaluate_acc_f1(self):
        self.model.eval()
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for _, t_batch in enumerate(self.dataset_test):
                t_inputs = [t_batch[col].to(self.device) for col in t_batch.keys()]
                t_targets = t_batch["polarity"].to(self.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        test_acc = n_correct / n_total
        f1 = f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average="macro")
        return test_acc, f1

    def _save_state_dict(self):
        if self.config.save_state_dict:
            curr_dt = datetime.datetime.now()
            curr_dt_str = curr_dt.strftime("%Y-%m-%d_%H%M%S")
            filename = f"{self.config.model}_{curr_dt_str}.pkl"
            try:
                torch.save(self.model.state_dict(), self.save_state_dict_folder.joinpath(filename))
            except:
                raise Exception("Error saving model state dict!")

    def _train_epoch(self, criterion: function, optimizer: function):
        max_val_acc, max_val_f1 = 0, 0
        max_val_epoch = 0
        global_step = 0
        path = 0

        for epoch in range(self.config.epochs):
            n_correct, n_total = 0, 0
            self.model.train()
            for _, batch in enumerate(self.dataloader_train):
                global_step += 1
                optimizer.zero_grad()

                inputs = [batch[col].to(self.device) for col in batch.keys()]
                targets = batch["polarity"].to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)

                if global_step % self.config.log_step == 0:
                    pass  # TODO: how to merge both calculate for bert and non-bert

    def train(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self._create_optimizer()(_params, lr=self.config.learning_rate, weight_decay=self.config.l2reg)

        test_accs, test_f1s = [], []
        for i in range(self.config.repeats):
            logging.info(f"Start overall train loop : {i + 1}")

            self._reset_params()
            test_acc, test_f1 = self._train_epoch(criterion, optimizer)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)

            logging.info(f"Test_acc: {test_acc}, Test_f1: {test_f1}")
        test_accs_avg = np.sum(test_accs) / self.config.repeats
        test_f1s_avg = np.sum(test_f1s) / self.config.repeats
        max_accs = np.max(test_accs)
        max_f1s = np.max(test_f1s)

        logging.info(
            f"""
            Test acc average: {test_accs_avg}
            Test f1 average: {test_f1s_avg}
            Test acc max: {max_accs}
            Test f1 max: {max_f1s}
        """
        )


class SenticNetBertGCNTrainer(SenticNetGCNBaseTrainer):
    def __init__(self, config: SenticNetGCNTrainArgs):
        self.config = config

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != SenticNetBertGCNPreTrainedModel:
                for param in child.parameters():
                    if param.requires_grad:
                        if len(param.shape) > 1:
                            self._create_initializers(param)
                        else:
                            stdv = 1.0 / math.sqrt(param.shape[0])
                            nn.init.uniform_(param, a=-stdv, b=stdv)


class SenticNetGCNTrainer(SenticNetGCNBaseTrainer):
    def __init__(self, config: SenticNetGCNTrainArgs):
        self.config = config

    def _reset_params(self):
        for param in self.modelparameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    self._create_initializers(param)
                else:
                    stdv = 1.0 / math.sqrt(param.shape[0])
                    nn.init.uniform_(param, a=-stdv, b=stdv)


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
    trainer = SenticNetGCNTrainer(cfg) if cfg.model == "senticgcn" else SenticNetBertGCNTrainer(cfg)
    trainer.train()
