import argparse
import logging
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import BaseConfig, RumourVerificationConfig, StanceClassificationConfig
from .data_class import BaseArguments
from .modeling import (
    PreTrainedModel,
    RumourVerificationModel,
    StanceClassificationModel,
)
from .modules.report import (
    BaseEvaluationReport,
    RumourVerificationModelEvaluationReport,
    StanceClassificationModelEvaluationReport,
)
from .preprocess import (
    BasePreprocessor,
    RumourVerificationPreprocessor,
    StanceClassificationPreprocessor,
)
from .tokenization import (
    BaseTokenizer,
    RumourVerificationTokenizer,
    StanceClassificationTokenizer,
)
from .utils import (
    check_path_exists,
    load_rumour_verification_config,
    load_stance_classification_config,
    set_device_and_seed,
)

logger = logging.getLogger(__name__)


class BaseModelEvaluator:
    """This is used as base class for derived StanceClassificationModelEvaluator and RumourVerificationModelEvaluator.

    Args:
        cfg (BaseArguments): Arguments for model evaluation.
        is_stance (bool): Whether the model type is stance classification.
        base_config (Type[BaseConfig]): Model configurations.
        base_model (Type[PreTrainedModel]): Model instance.
        base_tokenizer (Type[BaseTokenizer]): Tokenizer for the model.
        base_preprocessor (Type[BasePreprocessor]): Preprocessor for the test set.
        base_reporter (Type[BaseEvaluationReport]): Model evaluation report instance.

    """

    def __init__(
        self,
        cfg: BaseArguments,
        is_stance: bool,
        base_config: Type[BaseConfig],
        base_model: Type[PreTrainedModel],
        base_tokenizer: Type[BaseTokenizer],
        base_preprocessor: Type[BasePreprocessor],
        base_reporter: Type[BaseEvaluationReport],
    ) -> None:
        self.cfg: BaseArguments = cfg
        self.is_stance: bool = is_stance
        self.base_config: Type[BaseConfig] = base_config
        self.base_model: Type[PreTrainedModel] = base_model
        self.base_tokenizer: Type[BaseTokenizer] = base_tokenizer
        self.base_preprocessor: Type[BasePreprocessor] = base_preprocessor
        self.base_reporter: Type[BaseEvaluationReport] = base_reporter

        self.y_true: List[List[str]] = []
        self.y_pred: List[List[str]] = []

    def evaluate(self) -> None:
        """Setup and perform model evaluation."""

        logger.info("***** Evaluating on test dataset *****")

        env = set_device_and_seed(no_cuda=self.cfg.no_cuda, seed=self.cfg.seed)

        model = self._create_model().to(env["device"])

        tokenizer = self.base_tokenizer.from_pretrained(
            self.cfg.bert_model,
            do_lower_case=self.cfg.do_lower_case,
        )

        preprocessor = self.base_preprocessor(
            tokenizer=tokenizer,
            is_stance=self.is_stance,
            batch_size=self.cfg.batch_size,
            train_path=self.cfg.train_path,
            dev_path=self.cfg.dev_path,
            test_path=self.cfg.test_path,
            local_rank=env["local_rank"],
            max_tweet_num=self.cfg.max_tweet_num,
            max_tweet_length=self.cfg.max_tweet_length,
            max_seq_length=self.cfg.max_seq_length,
            max_tweet_bucket=self.cfg.max_tweet_bucket,
        )

        self.evaluate_model(
            dataloader=preprocessor.get_test_dataloader(),
            model=model,
            device=env["device"],
            eval_file=Path(__file__).parent / self.cfg.output_dir / "eval_results.txt",
        )

    def _create_model(self) -> PreTrainedModel:
        """Create model instance.

        Returns:
            PreTrainedModel: Model instance.
        """
        config_file = Path(__file__).parent / self.cfg.output_dir / "config.json"
        model_file = Path(__file__).parent / self.cfg.output_dir / "pytorch_model.bin"
        check_path_exists("Model weights", model_file)
        check_path_exists("Model configuration", config_file)

        config = self.base_config.from_pretrained(config_file)
        model = self.base_model.from_pretrained(
            model_file,
            config=config,
        )
        return model

    def evaluate_model(
        self,
        dataloader: DataLoader,
        model: Type[PreTrainedModel],
        device: torch.device,
        eval_file: Path = None,
    ) -> float:
        """Perform model evaluation.

        Args:
            dataloader (DataLoader): Dataloader for the data to be evaluated.
            model (Type[PreTrainedModel]): Model instance.
            device (torch.device): Device type.
            eval_file (Path, optional): Path of model evaluation results file. Defaults to None.

        Returns:
            float: Macro F1-score for StanceClassificationModel and accuracy for RumourVerificationModel.
        """
        self.y_true = []
        self.y_pred = []
        nb_eval_steps, nb_eval_examples = 0, 0
        metrics: Dict[str, Union[float, int]] = {
            "eval_accuracy": 0,
            "eval_loss": 0.0,
        }
        LABEL_ID = -2 if self.is_stance else -1

        model.eval()

        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = [b.to(device) for b in batch]

            with torch.no_grad():
                if not self.is_stance:
                    loss = model(*inputs).loss
                inputs[LABEL_ID] = None
                logits = model(*inputs).logits

            label_ids_in_numpy = batch[LABEL_ID].detach().cpu().numpy()

            if self.is_stance:
                label_mask_in_numpy = batch[-1].detach().cpu().numpy()
                logits_in_numpy = (
                    torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                    .detach()
                    .cpu()
                    .numpy()
                )
                self._convert_stance_ids_to_labels(
                    label_mask=label_mask_in_numpy,
                    label_ids=label_ids_in_numpy,
                    logits=logits_in_numpy,
                )
            else:
                logits_in_numpy = np.argmax(logits.detach().cpu().numpy(), axis=1)
                self._convert_rumour_ids_to_labels(
                    label_ids=label_ids_in_numpy,
                    logits=logits_in_numpy,
                )
                metrics["eval_loss"] += loss.mean().item()
                nb_eval_steps += 1
                metrics["eval_accuracy"] += np.sum(
                    logits_in_numpy == label_ids_in_numpy
                )
                nb_eval_examples += batch[0].size(0)

        if not self.is_stance:
            metrics["eval_loss"] /= nb_eval_steps
            metrics["eval_accuracy"] /= nb_eval_examples

        return self.base_reporter(
            y_true=self.y_true,
            y_pred=self.y_pred,
            eval_file=eval_file,
            metrics=metrics,
        ).get_score()

    def _convert_rumour_ids_to_labels(
        self,
        label_ids: np.ndarray,
        logits: np.ndarray,
    ) -> None:
        """Create labels for the predictions generated by RumourVerificationModel.

        Args:
            label_ids (np.ndarray): IDs of ground truths.
            logits (np.ndarray): IDs of predictions.
        """
        ID_TO_LABEL: Dict[int, str] = {
            0: "FALSE",
            1: "TRUE",
            2: "UNVERIFIED",
        }
        self.y_true.append([ID_TO_LABEL[label_ids[i]] for i in range(len(label_ids))])
        self.y_pred.append([ID_TO_LABEL[logits[i]] for i in range(len(logits))])

    def _convert_stance_ids_to_labels(
        self,
        label_mask: np.ndarray,
        label_ids: np.ndarray,
        logits: np.ndarray,
    ) -> None:
        """Create labels for the predictions generated by StanceClassificationModel.

        Args:
            label_mask (np.ndarray): Identify whether elements are IDs or paddings.
            label_ids (np.ndarray): IDs of ground truths.
            logits (np.ndarray): IDs of predictions.
        """

        ID_TO_LABEL: Dict[int, str] = {
            1: "DENY",
            2: "SUPPORT",
            3: "QUERY",
            4: "COMMENT",
        }
        for i, mask in enumerate(label_mask):
            truth: List[str] = []
            prediction: List[str] = []
            for j, m in enumerate(mask):
                if m:
                    truth.append(ID_TO_LABEL[label_ids[i][j]])
                    prediction.append(ID_TO_LABEL[logits[i][j]])
                else:
                    break
            self.y_true.append(truth)
            self.y_pred.append(prediction)


class StanceClassificationModelEvaluator(BaseModelEvaluator):
    """Evaluate StanceClassificationModel."""

    def __init__(
        self,
        cfg: BaseArguments,
        is_stance: bool = True,
        base_config: Type[BaseConfig] = StanceClassificationConfig,
        base_model: Type[PreTrainedModel] = StanceClassificationModel,
        base_tokenizer: Type[BaseTokenizer] = StanceClassificationTokenizer,
        base_preprocessor: Type[BasePreprocessor] = StanceClassificationPreprocessor,
        base_reporter: Type[
            BaseEvaluationReport
        ] = StanceClassificationModelEvaluationReport,
        **kwargs,
    ) -> None:
        super().__init__(
            cfg=cfg,
            is_stance=is_stance,
            base_config=base_config,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            base_preprocessor=base_preprocessor,
            base_reporter=base_reporter,
            **kwargs,
        )


class RumourVerificationModelEvaluator(BaseModelEvaluator):
    """Evaluate RumourVerificationModel."""

    def __init__(
        self,
        cfg: BaseArguments,
        is_stance: bool = False,
        base_config: Type[BaseConfig] = RumourVerificationConfig,
        base_model: Type[PreTrainedModel] = RumourVerificationModel,
        base_tokenizer: Type[BaseTokenizer] = RumourVerificationTokenizer,
        base_preprocessor: Type[BasePreprocessor] = RumourVerificationPreprocessor,
        base_reporter: Type[
            BaseEvaluationReport
        ] = RumourVerificationModelEvaluationReport,
        **kwargs,
    ) -> None:
        super().__init__(
            cfg=cfg,
            is_stance=is_stance,
            base_config=base_config,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            base_preprocessor=base_preprocessor,
            base_reporter=base_reporter,
            **kwargs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stance", action="store_true", help="run stance classification"
    )
    group.add_argument("--rumour", action="store_true", help="run rumour verification")
    args = parser.parse_args()
    if args.stance:
        stance_config = load_stance_classification_config()
        StanceClassificationModelEvaluator(stance_config).evaluate()

    if args.rumour:
        rumour_config = load_rumour_verification_config()
        RumourVerificationModelEvaluator(rumour_config).evaluate()
