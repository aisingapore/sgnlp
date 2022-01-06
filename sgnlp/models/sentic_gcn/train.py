import logging
import math
import pathlib
from typing import Tuple, Union

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from config import SenticGCNBertConfig, SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from data_class import SenticGCNTrainArgs
from modeling import SenticGCNBertPreTrainedModel, SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel
from tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer
from utils import parse_args_and_load_config, set_random_seed, SenticGCNDatasetGenerator, BucketIterator


logging.basicConfig(level=logging.DEBUG)


class SenticGCNBaseTrainer:
    """
    Base Trainer class used for training SenticGCNModel and SenticGCNBertModel
    """

    def __init__(self, config: SenticGCNTrainArgs):
        self.config = config
        self.global_max_acc = 0.0
        self.global_max_f1 = 0.0
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not self.config.device
            else torch.device(self.config.device)
        )
        if config.save_state_dict:
            self.save_state_dict_folder = pathlib.Path(self.config.saved_state_dict_folder_path)
            self.save_state_dict_folder.mkdir(exist_ok=True)

    def _create_initializers(self):
        """
        Private helper method to instantiate initializer.
        """
        initializers = {
            "xavier_uniform_": nn.init.xavier_uniform_,
            "xavier_normal_": nn.init.xavier_normal,
            "orthogonal": nn.init.orthogonal_,
        }
        return initializers[self.config.initializer]

    def _create_optimizer(self):
        """
        Private helper method to instantiate optimzer.
        """
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

    def _reset_params(self) -> None:
        raise NotImplementedError("Please call from derived class only.")

    def _generate_data_loaders(
        self,
    ) -> Union[Tuple[DataLoader, DataLoader, DataLoader], Tuple[BucketIterator, BucketIterator, BucketIterator]]:
        raise NotImplementedError("Please call from derived class only.")

    def _create_tokenizer(self) -> Union[SenticGCNTokenizer, SenticGCNBertTokenizer]:
        raise NotImplementedError("Please call from derived class only.")

    def _create_embedding_model(self) -> Union[SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel]:
        raise NotImplementedError("Please call from derived class only.")

    def _evaluate_acc_f1(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Private helper method to evaluate accuracy and f1 score.

        Args:
            dataloader (DataLoader): input val and test dataloader

        Returns:
            Tuple[float, float]: return acc and f1 score
        """
        self.model.eval()
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for _, t_batch in enumerate(dataloader):
                t_inputs = [t_batch[col] for col in t_batch.keys() if col != "polarity"]
                t_targets = t_batch["polarity"]
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

    # def _save_state_dict(self, epoch: int) -> pathlib.Path:
    #     curr_dt = datetime.datetime.now()
    #     curr_dt_str = curr_dt.strftime("%Y-%m-%d_%H%M%S")
    #     filename = f"{self.config.model}_epoch_{epoch}_{curr_dt_str}.pkl"
    #     full_path = self.save_state_dict_folder.joinpath(filename)
    #     try:
    #         torch.save(self.model.state_dict(), full_path)
    #     except:
    #         raise Exception("Error saving model state dict!")
    #     return full_path

    def _train_epoch(
        self, criterion: function, optimizer: function, train_dataloader: DataLoader, val_dataloader: DataLoader
    ) -> pathlib.Path:
        # max_val_acc, max_val_f1 = 0, 0
        # max_val_epoch = 0
        # global_step = 0
        # path = None

        # for epoch in range(self.config.epochs):
        #     n_correct, n_total, loss_total = 0, 0, 0
        #     self.model.train()
        #     for _, batch in enumerate(train_dataloader):
        #         global_step += 1
        #         optimizer.zero_grad()

        #         inputs = [batch[col]["input_ids"] for col in batch.keys() if col != "polarity"]
        #         targets = batch["polarity"]["input_ids"]
        #         outputs = self.model(inputs)
        #         loss = criterion(outputs, targets)
        #         loss.backward()
        #         optimizer.step()

        #         n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        #         n_total += len(outputs)
        #         loss_total += loss.item() * len(outputs)

        #         if global_step % self.config.log_step == 0:
        #             train_acc = n_correct / n_total
        #             train_loss = loss_total / n_total
        #             logging.info(f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}")

        #     val_acc, val_f1 = self._evaluate_acc_f1(val_dataloader)
        #     logging.info(
        #         f"""
        #         Epoch: {epoch}
        #         Test Acc: {val_acc:.4f}
        #         Test Loss: {val_f1:.4f}
        #     """
        #     )
        #     if val_f1 > max_val_f1:
        #         max_val_f1 = val_f1

        #     if val_acc > max_val_acc:
        #         max_val_acc = val_acc
        #         max_val_epoch = epoch
        #         if self.config.save_state_dict:
        #             path = self._save_state_dict(epoch)
        #         logging.info(
        #             f"""
        #             Best model saved. Acc: {max_val_acc:.4f}, F1: {max_val_f1}, Epoch: {max_val_epoch}
        #         """
        #         )

        #     if epoch - max_val_epoch >= self.config.patience:
        #         logging.info(f"Early stopping")
        #         break
        # return path
        pass

    def train(self):
        # criterion = nn.CrossEntropyLoss()
        # _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = self._create_optimizer()(_params, lr=self.config.learning_rate, weight_decay=self.config.l2reg)

        # train_dataloader, val_dataloader, test_dataloader = self._generate_data_loaders()

        # test_accs, test_f1s = [], []
        # for i in range(self.config.repeats):
        #     logging.info(f"Start overall train loop : {i + 1}")

        #     self._reset_params()
        #     test_acc, test_f1 = self._train_epoch(criterion, optimizer, train_dataloader, val_dataloader)
        #     test_accs.append(test_acc)
        #     test_f1s.append(test_f1)

        #     logging.info(f"Test_acc: {test_acc}, Test_f1: {test_f1}")
        # test_accs_avg = np.sum(test_accs) / self.config.repeats
        # test_f1s_avg = np.sum(test_f1s) / self.config.repeats
        # max_accs = np.max(test_accs)
        # max_f1s = np.max(test_f1s)

        # logging.info(
        #     f"""
        #     Test acc average: {test_accs_avg}
        #     Test f1 average: {test_f1s_avg}
        #     Test acc max: {max_accs}
        #     Test f1 max: {max_f1s}
        # """
        # )
        pass


class SenticGCNBertTrainer(SenticGCNBaseTrainer):
    """
    Trainer class derived from SenticGCNBaseTrainer. Used for training SenticGCNBertModel.

    Args:
        config (SenticGCNTrainArgs): Training config for SenticGCNBertModel
    """

    def __init__(self, config: SenticGCNTrainArgs):
        super().__init__(config)
        self.config = config
        tokenizer = self._create_tokenizer()
        self.embed = self._create_embedding_model()
        data_gen = SenticGCNDatasetGenerator(config, tokenizer)
        self.train_data, self.val_data, self.test_data = data_gen.generate_datasets()
        del data_gen

    def _create_tokenizer(self) -> SenticGCNBertTokenizer:
        """
        Private method to construct tokenizer via the from_pretrained method.

        Returns:
            SenticGCNBertTokenizer: return a SenticGCNBertTokenizer instance.
        """
        return SenticGCNBertTokenizer.from_pretrained(self.config.tokenizer)

    def _create_embedding_model(self) -> SenticGCNBertEmbeddingModel:
        """
        Private helper method to create the bert based embedding models.

        Returns:
            SenticGCNBertEmbeddingModel: return instance of pretrained SenticGCNBertEmbeddingModel
        """
        config = SenticGCNBertEmbeddingConfig.from_pretrained(self.config.embedding_model)
        return SenticGCNBertEmbeddingModel.from_pretrained(self.config.embedding_model, config=config)

    def _reset_params(self):
        """
        Private helper method to reset model parameters.
        To be used during repeats train loop.
        """
        for child in self.model.children():
            if type(child) != SenticGCNBertPreTrainedModel:
                for param in child.parameters():
                    if param.requires_grad:
                        if len(param.shape) > 1:
                            self._create_initializers(param)
                        else:
                            stdv = 1.0 / math.sqrt(param.shape[0])
                            nn.init.uniform_(param, a=-stdv, b=stdv)

    def _generate_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Private helper method to generate train, val and test dataloaders.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: return train, val and test dataloaders.
        """
        train_dataloader = DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_data, batch_size=self.config.batch_size, shuffle=False)
        test_dataloader = DataLoader(self.test_data, batch_size=self.config.batch_size, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader


class SenticGCNTrainer(SenticGCNBaseTrainer):
    """
    Trainer class derived from SenticGCNBaseTrainer. Used for training SenticGCNModel.

    Args:
        config (SenticGCNTrainArgs): Training config for SenticGCNModel
    """

    def __init__(self, config: SenticGCNTrainArgs) -> None:
        super().__init__(config)
        self.config = config
        tokenizer = self._create_tokenizer()
        self.embed = self._create_embedding_model(tokenizer.vocab)
        data_gen = SenticGCNDatasetGenerator(config, tokenizer)
        self.train_data, self.val_data, self.test_data = data_gen.generate_datasets()
        del data_gen

    def _create_tokenizer(self) -> SenticGCNTokenizer:
        """
        Private method to construct tokenizer either via the from_pretrained method or
        constructing the tokenizer using input dataset files.

        Returns:
            SenticGCNTokenizer: return a SenticGCNTokenizer instance.
        """
        if not self.config.train_tokenizer:
            return SenticGCNTokenizer.from_pretrained(self.config.tokenizer)
        else:
            tokenizer = SenticGCNTokenizer(
                train_files=[self.config.dataset_train, self.config.dataset_test], train_vocab=True
            )
            if self.config.save_tokenizer:
                tokenizer.save_pretrained(self.config.save_tokenizer_path)
            return tokenizer

    def _create_embedding_model(self, vocab: dict[str, int]) -> SenticGCNEmbeddingModel:
        """
        Private method to construct embedding model either via the from_pretrained method or
        building the embedding model from word vector files. (e.g. GloVe word vectors)

        Args:
            vocab (dict[str, int]): dictionary of vocab from tokenizer

        Returns:
            SenticGCNEmbeddingModel: return a SenticGCNEmbeddingModel instance.
        """
        if not self.config.build_embedding_model:
            config_path = pathlib.Path(self.config.embedding_model).joinpath("config.json")
            embed_config = SenticGCNEmbeddingConfig.from_pretrained(config_path)
            embed_path = pathlib.Path(self.config.embedding_model).joinpath("pytorch_model.bin")
            return SenticGCNEmbeddingModel.from_pretrained(embed_path, config=embed_config)
        else:
            embedding_model = SenticGCNEmbeddingModel.build_embedding_model(
                self.config.word_vec_file_path, vocab, self.config.embed_dim
            )
            if self.config.save_embedding_model:
                embedding_model.save_pretrained(self.config.save_embedding_model_path)
            return embedding_model

    def _reset_params(self) -> None:
        """
        Private helper method to reset model parameters.
        To be used during repeats train loop.
        """
        for param in self.modelparameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    self._create_initializers(param)
                else:
                    stdv = 1.0 / math.sqrt(param.shape[0])
                    nn.init.uniform_(param, a=-stdv, b=stdv)

    def _generate_data_loaders(self) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:
        """
        Private helper method to generate train, val and test dataloaders.

        Returns:
            Tuple[BucketIterator, BucketIterator, BucketIterator]: return train, val and test bucketiterators.
        """
        train_dataloader = BucketIterator(self.train_data, batch_size=self.config.batch_size, shuffle=True)
        val_dataloader = BucketIterator(self.val_data, batch_size=self.config.batch_size, shuffle=False)
        test_dataloader = BucketIterator(self.test_data, batch_size=self.config.batch_size, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # cfg = parse_args_and_load_config()
    args = {
        "senticnet_word_file_path": "/Users/raymond/work/aimakerspace_sgnlp/sgnlp/models/sentic_asgcn/senticNet/senticnet_word.txt",
        "save_preprocessed_senticnet": True,
        "saved_preprocessed_senticnet_file_path": "senticnet/senticnet.pickle",
        "spacy_pipeline": "en_core_web_sm",
        "word_vec_file_path": "/Users/raymond/work/aimakerspace_sgnlp/sgnlp/models/sentic_asgcn/glove/glove.840B.300d.txt",
        "dataset_train": "/Users/raymond/work/aimakerspace_sgnlp/sgnlp/models/sentic_asgcn/datasets/semeval14/restaurant_train.raw",
        "dataset_test": "/Users/raymond/work/aimakerspace_sgnlp/sgnlp/models/sentic_asgcn/datasets/semeval14/restaurant_test.raw",
        "valset_ratio": 0,
        "model": "senticgcn",
        "save_best_model": True,
        "save_model_path": "senticgcn",
        "tokenizer": "senticgcn",
        "train_tokenizer": False,
        "save_tokenizer": False,
        "save_tokenizer_path": "senticgcn_tokenizer",
        "embedding_model": "senticgcn_embed_model",
        "build_embedding_model": False,
        "save_embedding_model": False,
        "save_embedding_model_path": "senticgcn_embed_model",
        "initializer": "xavier_uniform",
        "optimizer": "adam",
        "loss_function": "cross_entropy",
        "learning_rate": 0.001,
        "l2reg": 0.00001,
        "epochs": 100,
        "batch_size": 32,
        "log_step": 5,
        "embed_dim": 300,
        "hidden_dim": 300,
        "polarities_dim": 3,
        "dropout": 0.3,
        "save_results": True,
        "seed": 776,
        "device": "cuda",
        "repeats": 10,
        "patience": 5,
        "max_len": 85,
    }
    from data_class import SenticGCNTrainArgs

    cfg = SenticGCNTrainArgs(**args)
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
    trainer = SenticGCNTrainer(cfg) if cfg.model == "senticgcn" else SenticGCNBertTrainer(cfg)
    trainer.train()
