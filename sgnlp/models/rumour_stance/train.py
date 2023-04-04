import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .config import BaseConfig, RumourVerificationConfig, StanceClassificationConfig
from .data_class import BaseArguments
from .eval import (
    BaseModelEvaluator,
    RumourVerificationModelEvaluator,
    StanceClassificationModelEvaluator,
)
from .modeling import (
    PreTrainedModel,
    RumourVerificationModel,
    StanceClassificationModel,
)
from .modules.optimization import BertAdam
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
    load_rumour_verification_config,
    load_stance_classification_config,
    set_device_and_seed,
)

logger = logging.getLogger(__name__)


class BaseModelTrainer:
    """This is used as base class for derived StanceClassificationModelTrainer and RumourVerificationModelTrainer.

    Args:
        cfg (BaseArguments): Arguments for model training.
    """

    def __init__(
        self,
        cfg: BaseArguments,
        is_stance: bool,
        base_config: Type[BaseConfig],
        base_model: Type[PreTrainedModel],
        base_tokenizer: Type[BaseTokenizer],
        base_preprocessor: Type[BasePreprocessor],
        base_evaluator: Type[BaseModelEvaluator],
        base_reporter: Type[BaseEvaluationReport],
    ) -> None:
        self.cfg: BaseArguments = cfg
        self.is_stance: bool = is_stance
        self.base_config: Type[BaseConfig] = base_config
        self.base_model: Type[PreTrainedModel] = base_model
        self.base_tokenizer: Type[BaseTokenizer] = base_tokenizer
        self.base_preprocessor: Type[BasePreprocessor] = base_preprocessor
        self.base_evaluator: Type[BaseModelEvaluator] = base_evaluator
        self.base_reporter: Type[BaseEvaluationReport] = base_reporter

        self.env = set_device_and_seed(no_cuda=cfg.no_cuda, seed=cfg.seed)
        self.cfg.batch_size = int(
            self.cfg.batch_size / self.cfg.gradient_accumulation_steps
        )
        self.max_score = 0.0
        self.model_file = Path(Path(__file__).parent / self.cfg.output_dir)

    def train(self) -> None:
        """Setup and perform model training."""
        model = self._create_model()

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
            local_rank=self.env["local_rank"],
            max_tweet_num=self.cfg.max_tweet_num,
            max_tweet_length=self.cfg.max_tweet_length,
            max_seq_length=self.cfg.max_seq_length,
            max_tweet_bucket=self.cfg.max_tweet_bucket,
        )

        evaluator = self.base_evaluator(
            self.cfg,
            is_stance=self.is_stance,
            base_config=self.base_config,
            base_model=self.base_model,
            base_tokenizer=self.base_tokenizer,
            base_preprocessor=self.base_preprocessor,
            base_reporter=self.base_reporter,
        )

        train_dataloader, train_configs = self._get_dataloader_and_configs(
            model=model,
            preprocessor=preprocessor,
        )

        self._train_model(
            train_configs=train_configs,
            train_dataloader=train_dataloader,
            dev_dataloader=preprocessor.get_dev_dataloader(),
            test_dataloader=preprocessor.get_test_dataloader(),
            model=model,
            evaluator=evaluator,
        )

    def _create_model(self) -> PreTrainedModel:
        """Create model instance.

        Raises:
            ImportError: Occurs when distributed training is requested but apex is not installed.

        Returns:
            PreTrainedModel: Model instance.
        """
        model = self.base_model.from_pretrained(
            self.cfg.bert_model,
            cache_dir=Path(
                os.getenv(
                    "PYTORCH_PRETRAINED_BERT_CACHE",
                    Path.home() / ".pytorch_pretrained_bert",
                )
            )
            / "distributed_{}".format(self.env["local_rank"]),
        )

        if self.cfg.fp16:
            model.half()
        model.to(self.env["device"])

        if self.env["local_rank"] != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError as exc:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed training."
                ) from exc
            model = DDP(model)
        elif self.env["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        return model

    def _get_dataloader_and_configs(
        self,
        model: PreTrainedModel,
        preprocessor: BasePreprocessor,
    ) -> Tuple[DataLoader, Dict[str, Any]]:
        """Create dataloader for train set and setup configurations of model training.

        Args:
            model (PreTrainedModel): Model instance.
            preprocessor (BasePreprocessor): Preprocessor of train dataset.

        Returns:
            Tuple[DataLoader, Dict[str, Any]]: Dataloader for train set and configurations of model training.
        """
        train_configs: Dict[str, Any] = {}
        train_dataloader = preprocessor.get_train_dataloader()

        train_configs["num_train_steps"] = int(
            preprocessor.get_num_threads()
            / self.cfg.batch_size
            / self.cfg.gradient_accumulation_steps
            * self.cfg.num_train_epochs
        )
        train_configs["t_total"] = train_configs["num_train_steps"]
        if self.env["local_rank"] != -1:
            train_configs["t_total"] = (
                train_configs["t_total"] // torch.distributed.get_world_size()
            )

        train_configs["optimizer"] = self._create_optimizer(
            model=model,
            t_total=train_configs["t_total"],
        )

        self.model_file.mkdir(parents=True, exist_ok=True)

        return train_dataloader, train_configs

    def _create_optimizer(
        self,
        model: PreTrainedModel,
        t_total: int,
    ) -> BertAdam:
        """Create model optimizer.

        Args:
            model (PreTrainedModel): Model instance.
            t_total (int): Total number of training steps for the learning rate schedule.

        Raises:
            ImportError: Occurs when 16-bit float precision is requested for model training but apex is not installed.

        Returns:
            BertAdam: Model optimizer.
        """
        param_optimizer = list(model.named_parameters())
        NO_DECAY = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in NO_DECAY)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in NO_DECAY)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.cfg.fp16:
            try:
                from apex.optimizers import FP16_Optimizer, FusedAdam
            except ImportError as exc:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                ) from exc

            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.cfg.learning_rate,
                bias_correction=False,
                max_grad_norm=1.0,
            )
            if self.cfg.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(
                    optimizer, static_loss_scale=self.cfg.loss_scale
                )
        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=self.cfg.learning_rate,
                warmup=self.cfg.warmup_proportion,
                t_total=t_total,
            )
        return optimizer

    def _train_model(
        self,
        train_configs: Dict[str, Any],
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: PreTrainedModel,
        evaluator: BaseModelEvaluator,
    ) -> None:
        """Perform model training.

        Args:
            train_configs (Dict[str, Any]): Configuration of model training.
            train_dataloader (DataLoader): Dataloader for training set.
            dev_dataloader (DataLoader): Dataloader for development set.
            test_dataloader (DataLoader): Dataloader for test set.
            model (PreTrainedModel): Model instance.
            evaluator (BaseModelEvaluator): Model evaluation instance.
        """
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        logger.info("************ Training on train dataset ************")
        for train_idx in trange(int(self.cfg.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Num steps = %d", train_configs["num_train_steps"])
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = [b.to(self.env["device"]) for b in batch]
                first_input_ids = batch[0]
                loss = model(*batch).loss

                if self.env["n_gpu"] > 1:
                    loss = loss.mean()
                if self.cfg.gradient_accumulation_steps > 1:
                    loss = loss / self.cfg.gradient_accumulation_steps
                if self.cfg.fp16:
                    train_configs["optimizer"].backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += first_input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    lr_this_step = self.cfg.learning_rate * self.warmup_linear(
                        global_step / train_configs["t_total"],
                        self.cfg.warmup_proportion,
                    )
                    for param_group in train_configs["optimizer"].param_groups:
                        param_group["lr"] = lr_this_step
                    train_configs["optimizer"].step()
                    train_configs["optimizer"].zero_grad()
                    global_step += 1

            self._evalaute_dev_and_test_sets(
                evaluator=evaluator,
                dev_dataloader=dev_dataloader,
                test_dataloader=test_dataloader,
                model=model,
            )

        logger.info(f"Trained model saved at {self.model_file}")

    def warmup_linear(self, x: float, warmup: float = 0.002) -> float:
        """Calculate warmup_linear learning rate scheduler."""
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def _evalaute_dev_and_test_sets(
        self,
        evaluator: BaseModelEvaluator,
        dev_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: PreTrainedModel,
    ) -> None:
        """Evaluate model on both development and test sets.

        Args:
            evaluator (BaseModelEvaluator): Model evaluation instance.
            dev_dataloader (DataLoader): Dataloader for development set.
            test_dataloader (DataLoader): Dataloader for test set.
            model (PreTrainedModel): Model instance.
        """
        logger.info("***** Evaluating on dev dataset *****")
        eval_score = evaluator.evaluate_model(
            dataloader=dev_dataloader, model=model, device=self.env["device"]
        )
        if eval_score > self.max_score:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(self.model_file)
            self.max_score = eval_score

        logger.info("***** Evaluating on test dataset *****")
        evaluator.evaluate_model(
            dataloader=test_dataloader, model=model, device=self.env["device"]
        )


class StanceClassificationModelTrainer(BaseModelTrainer):
    """Train StanceClassificationModel."""

    def __init__(
        self,
        cfg: BaseArguments,
        is_stance: bool = True,
        base_config: Type[BaseConfig] = StanceClassificationConfig,
        base_model: Type[PreTrainedModel] = StanceClassificationModel,
        base_tokenizer: Type[BaseTokenizer] = StanceClassificationTokenizer,
        base_preprocessor: Type[BasePreprocessor] = StanceClassificationPreprocessor,
        base_evaluator: Type[BaseModelEvaluator] = StanceClassificationModelEvaluator,
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
            base_evaluator=base_evaluator,
            base_reporter=base_reporter,
            **kwargs,
        )


class RumourVerificationModelTrainer(BaseModelTrainer):
    """Train RumourVerificationModel."""

    def __init__(
        self,
        cfg: BaseArguments,
        is_stance: bool = False,
        base_config: Type[BaseConfig] = RumourVerificationConfig,
        base_model: Type[PreTrainedModel] = RumourVerificationModel,
        base_tokenizer: Type[BaseTokenizer] = RumourVerificationTokenizer,
        base_preprocessor: Type[BasePreprocessor] = RumourVerificationPreprocessor,
        base_evaluator: Type[BaseModelEvaluator] = RumourVerificationModelEvaluator,
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
            base_evaluator=base_evaluator,
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
        StanceClassificationModelTrainer(stance_config).train()

    if args.rumour:
        rumour_config = load_rumour_verification_config()
        RumourVerificationModelTrainer(rumour_config).train()
