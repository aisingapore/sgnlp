import logging
import pathlib
import shutil
import tempfile
import urllib
from typing import Tuple, Union

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data_class import SenticGCNTrainArgs
from .config import SenticGCNBertConfig, SenticGCNConfig, SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from .modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel, SenticGCNModel, SenticGCNBertModel
from .tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer
from .utils import BucketIterator, parse_args_and_load_config, download_tokenizer_files, set_random_seed


logging.basicConfig(level=logging.DEBUG)


class SenticGCNBaseEvaluator:
    def __init__(self, config: SenticGCNTrainArgs) -> None:
        self.config = config["eval_args"]
        self.data_cols = config.data_cols
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not self.config["device"]
            else torch.device(config[self.config["device"]])
        )

    def _create_tokenizer(
        self, tokenizer_class: Union[SenticGCNTokenizer, SenticGCNBertTokenizer]
    ) -> Union[SenticGCNTokenizer, SenticGCNBertTokenizer]:
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
        embedding_config_class: Union[SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig],
        embedding_model_class: Union[SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel],
    ) -> Union[SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel]:
        if model_name_path_or_folder.startswith("https://") or model_name_path_or_folder.startswith("http://"):
            config_url = urllib.parse.urljoin(model_name_path_or_folder, self.config["config_filename"])
            model_url = urllib.parse.urljoin(model_name_path_or_folder, self.config["model_filename"])
            embed_config = embedding_config_class.from_pretrained(config_url)
            embed_model = embedding_model_class.from_pretrained(model_url, config=embed_config)
        else:
            # Load from local folder
            embed_model_name = pathlib.Path(model_name_path_or_folder)
            if embed_model_name.is_dir():
                config_path = embed_model_name.joinpath(self.config["config_filename"])
                model_path = embed_model_name.joinpath(self.config["model_filename"])
                embed_config = embedding_config_class.from_pretrained(config_path)
                embed_model = embedding_model_class.from_pretrained(model_path, config=embed_config)
            else:
                # Load from HuggingFace model repository
                embed_config = embedding_config_class.from_pretrained(model_name_path_or_folder)
                embed_model = embedding_model_class.from_pretrained(model_name_path_or_folder, config=embed_config)
        return embed_model

    def _evaluate_acc_f1(self, dataloader: Union[DataLoader, BucketIterator]) -> Tuple[float, float]:
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


class SenticGCNEvaluator(SenticGCNBaseEvaluator):
    def __init__(self, config: SenticGCNTrainArgs) -> None:
        super().__init__(config)
        self.tokenizer = self._create_tokenizer(SenticGCNTokenizer)
        self.embedding_model = self._create_model(
            config.eval_args["embedding_model"], SenticGCNEmbeddingConfig, SenticGCNEmbeddingModel
        )
        self.model = self._create_model(config.eval_args["model"], SenticGCNConfig, SenticGCNModel)

    def evaluate(self):
        pass


class SenticGCNBertEvaluator(SenticGCNBaseEvaluator):
    def __init__(self, config: SenticGCNTrainArgs) -> None:
        super().__init__(config)
        self.tokenizer = self._create_tokenizer(SenticGCNBertTokenizer)
        self.embedding_model = self._create_model(
            config.eval_args["embedding_model"], SenticGCNBertEmbeddingConfig, SenticGCNBertEmbeddingModel
        )
        self.model = self._create_model(config.eval_args["model"], SenticGCNBertConfig, SenticGCNBertModel)

    def evaluate(self):
        pass


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if cfg.eval_args["seed"] is not None:
        set_random_seed(cfg.eval_args["seed"])
    evaluator = SenticGCNEvaluator(cfg) if cfg.model == "senticgcn" else SenticGCNBertEvaluator(cfg)
    evaluator.evaluate()
