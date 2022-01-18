import datetime
import logging
import pathlib
import shutil
import tempfile
import urllib
from typing import List, Tuple, Union

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data_class import SenticGCNTrainArgs
from .config import SenticGCNBertConfig, SenticGCNConfig, SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from .modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel, SenticGCNModel, SenticGCNBertModel
from .tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer
from .utils import (
    SenticGCNDatasetGenerator,
    BucketIterator,
    parse_args_and_load_config,
    download_tokenizer_files,
    set_random_seed,
)


logging.basicConfig(level=logging.DEBUG)


class SenticGCNBaseEvaluator:
    """
    Base Evaluator class used for evaluating SenticGCNModel and SenticGCNBertModel
    """

    def __init__(self, config: SenticGCNTrainArgs) -> None:
        self.config = config.eval_args
        self.data_cols = config.data_cols
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not self.config["device"]
            else torch.device(self.config["device"])
        )

    def _create_tokenizer(
        self, tokenizer_class: Union[SenticGCNTokenizer, SenticGCNBertTokenizer]
    ) -> Union[SenticGCNTokenizer, SenticGCNBertTokenizer]:
        """
        Private method to construct tokenizer.
        Tokenizer can be created via download from cloud storage, from local storage
        or from HuggingFace repository.

        Args:
            tokenizer_class (Union[SenticGCNTokenizer, SenticGCNBertTokenizer]): tokenizer class type to create.

        Returns:
            Union[SenticGCNTokenizer, SenticGCNBertTokenizer]: return the tokenizer class instance.
        """
        if self.config["tokenizer"].startswith("https://") or self.config["tokenizer"].startswith("http://"):
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = pathlib.Path(tmpdir)
            download_tokenizer_files(self.config["tokenizer"], temp_dir)
            tokenizer_ = tokenizer_class.from_pretrained(temp_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            tokenizer_ = tokenizer_class.from_pretrained(self.config["tokenizer"])
        return tokenizer_

    def _create_model(
        self,
        model_name_path_or_folder: str,
        config_class: Union[
            SenticGCNConfig, SenticGCNBertConfig, SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
        ],
        model_class: Union[SenticGCNModel, SenticGCNBertModel, SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel],
    ) -> Union[SenticGCNModel, SenticGCNBertModel, SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel]:
        """
        Private method to construct models and embedding models.
        Model can be created via download from cloud storage via from_pretrained method, from local storage
        or from HuggingFace repository.

        Args:
            model_name_path_or_folder (str): cloud or local storage path to model files
            config_class (Union[SenticGCNConfig, SenticGCNBertConfig, SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig]):
                config class type
            model_class (Union[SenticGCNModel, SenticGCNBertModel, SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel]):
                model class type

        Returns:
            Union[SenticGCNModel, SenticGCNBertModel, SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel]:
                return model instance.
        """
        if model_name_path_or_folder.startswith("https://") or model_name_path_or_folder.startswith("http://"):
            config_url = urllib.parse.urljoin(model_name_path_or_folder, self.config["config_filename"])
            model_url = urllib.parse.urljoin(model_name_path_or_folder, self.config["model_filename"])
            config = config_class.from_pretrained(config_url)
            model = model_class.from_pretrained(model_url, config=config)
        else:
            # Load from local folder
            embed_model_name = pathlib.Path(model_name_path_or_folder)
            if embed_model_name.is_dir():
                config_path = embed_model_name.joinpath(self.config["config_filename"])
                model_path = embed_model_name.joinpath(self.config["model_filename"])
                config = config_class.from_pretrained(config_path)
                model = model_class.from_pretrained(model_path, config=config)
            else:
                # Load from HuggingFace model repository
                config = config_class.from_pretrained(model_name_path_or_folder)
                model = model_class.from_pretrained(model_name_path_or_folder, config=config)
        return model

    def _evaluate_acc_f1(self, dataloader: Union[DataLoader, BucketIterator]) -> Tuple[float, float]:
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
                # Generate embeddings
                t_batch["text_embeddings"] = self._generate_embeddings(t_batch)
                # Prepare input data and targets
                t_inputs = [t_batch[col].to(self.device) for col in self.data_cols]
                t_targets = t_batch["polarity"].to(self.device)
                # Inference
                t_outputs = self.model(t_inputs)
                # Calculate loss
                n_correct += (torch.argmax(t_outputs.logits, -1) == t_targets).sum().item()
                n_total += len(t_outputs.logits)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs.logits
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.logits), dim=0)
        test_acc = n_correct / n_total
        f1 = f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average="macro")
        return test_acc, f1

    def _save_results_to_file(self, acc_f1: List[str]) -> None:
        """
        Private method to save acc and f1 results to file.

        Args:
            acc_f1 (List[str]): list containing acc and f1 results
        """
        results = [
            f"Model: {self.config['model']}\n",
            f"Batch Size: {self.config['eval_batch_size']}\n",
            f"Random Seed: {self.config['seed']}\n",
        ]
        results = [*results, *acc_f1]
        results_folder = pathlib.Path(self.config["result_folder"])
        results_folder.mkdir(exist_ok=True)
        results_file = results_folder.joinpath(
            f"{self.config['model']}_{datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')}_results.txt"
        )
        with open(results_file, "a") as f:
            f.writelines(results)


class SenticGCNEvaluator(SenticGCNBaseEvaluator):
    """
    Evaluator class derived from SenticGCNBaseEvaluator.

    Args:
        config (SenticGCNTrainArgs): Config for SenticGCNModel
    """

    def __init__(self, config: SenticGCNTrainArgs) -> None:
        super().__init__(config)
        self.tokenizer = self._create_tokenizer(SenticGCNTokenizer)
        self.embedding_model = self._create_model(
            config.eval_args["embedding_model"], SenticGCNEmbeddingConfig, SenticGCNEmbeddingModel
        )
        self.model = self._create_model(config.eval_args["model_path"], SenticGCNConfig, SenticGCNModel)
        data_gen = SenticGCNDatasetGenerator(config, self.tokenizer, "test")
        self.raw_data = data_gen.generate_test_datasets()
        del data_gen

    def _generate_embeddings(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Private helper method to generate embeddings.

        Args:
            batch (List[torch.Tensor]): a batch of test dataset

        Returns:
            torch.Tensor: return embedding tensor
        """
        text_indices = batch["text_indices"].to(self.device)
        return self.embedding_model(text_indices)

    def evaluate(self) -> None:
        """
        Main evaluate method.
        """
        # Generate dataloaders
        test_dataloader = BucketIterator(self.raw_data, batch_size=self.config["eval_batch_size"], shuffle=False)
        # Evalute Acc and F1
        acc, f1 = self._evaluate_acc_f1(test_dataloader)
        logging.info(f"Evaluate Results -> Acc: {acc}, F1: {f1}")
        # Save results
        acc_f1 = [f"Acc: {acc}\n", f"F1: {f1}\n"]
        self._save_results_to_file(acc_f1)

        logging.info("Evaluation Complete!")


class SenticGCNBertEvaluator(SenticGCNBaseEvaluator):
    """
    Evaluator class derived from SenticGCNBaseEvaluator.

    Args:
        config (SenticGCNTrainArgs): Config for SenticGCNModel
    """

    def __init__(self, config: SenticGCNTrainArgs) -> None:
        super().__init__(config)
        self.tokenizer = self._create_tokenizer(SenticGCNBertTokenizer)
        self.embedding_model = self._create_model(
            config.eval_args["embedding_model"], SenticGCNBertEmbeddingConfig, SenticGCNBertEmbeddingModel
        )
        self.model = self._create_model(config.eval_args["model_path"], SenticGCNBertConfig, SenticGCNBertModel)
        data_gen = SenticGCNDatasetGenerator(config, self.tokenizer, "test")
        self.raw_data = data_gen.generate_test_datasets()
        del data_gen

    def _generate_embeddings(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Private helper method to generate embeddings.

        Args:
            batch (List[torch.Tensor]): a batch of test dataset

        Returns:
            torch.Tensor: return embedding tensor
        """
        text_bert_indices = batch["text_bert_indices"].to(self.device)
        bert_segment_indices = batch["bert_segment_indices"].to(self.device)

        return self.embedding_model(text_bert_indices, token_type_ids=bert_segment_indices)["last_hidden_state"]

    def evaluate(self) -> None:
        """
        Main evaluate method.
        """
        # Generate dataloaders
        test_dataloader = DataLoader(self.raw_data, batch_size=self.config["eval_batch_size"], shuffle=False)
        # Evaluate Acc and F1
        acc, f1 = self._evaluate_acc_f1(test_dataloader)
        logging.info(f"Evaluate Results -> Acc: {acc}, F1: {f1}")
        # Save results
        acc_f1 = [f"Acc: {acc}\n", f"F1: {f1}\n"]
        self._save_results_to_file(acc_f1)

        logging.info("Evaluation Complete!")


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if cfg.eval_args["seed"] is not None:
        set_random_seed(cfg.eval_args["seed"])
    evaluator = SenticGCNEvaluator(cfg) if cfg.eval_args["model"] == "senticgcn" else SenticGCNBertEvaluator(cfg)
    logging.info(f"Evaluating {cfg.eval_args['model']}")
    evaluator.evaluate()
