import numpy as np
import pandas as pd
from transformers import Trainer
from sklearn.metrics import classification_report
from transformers.training_args import TrainingArguments

from .tokenization import RecconEmotionEntailmentTokenizer
from .modeling import RecconEmotionEntailmentModel
from .utils import (
    RecconEmotionEntailmentData,
    convert_df_to_dataset,
    parse_args_and_load_config,
)
from .data_class import RecconEmotionEntailmentArguments


def evaluate(cfg: RecconEmotionEntailmentArguments):
    """
    Method to evaluate a trained RecconEmotionEntailmentModel.

    Args:
        config (:obj:`RecconEmotionEntailmentArguments`):
            RecconEmotionEntailmentArguments config load from config file.

    Example::

            import json
            from sgnlp.models.emotion_entailment import evaluate
            from sgnlp.models.emotion_entailment.utils import parse_args_and_load_config

            cfg = parse_args_and_load_config('config/emotion_entailment_config.json')
            evaluate(cfg)
    """

    tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained(cfg.model_name)
    model = RecconEmotionEntailmentModel.from_pretrained(
        cfg.eval_args["trained_model_dir"]
    )

    test_df = pd.read_csv(cfg.eval_args["x_test_path"])
    test_dataset = convert_df_to_dataset(
        df=test_df,
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
    )

    predict_args = TrainingArguments(
        output_dir=cfg.eval_args["trained_model_dir"],
        per_device_eval_batch_size=cfg.eval_args["per_device_eval_batch_size"],
        no_cuda=cfg.eval_args["no_cuda"],
    )
    trainer = Trainer(model=model, args=predict_args)
    raw_pred, labels, _ = trainer.predict(RecconEmotionEntailmentData(test_dataset))
    pred = np.argmax(raw_pred, axis=1)

    with open(cfg.eval_args["results_path"], "w") as result_file:
        result_file.write(classification_report(y_true=labels, y_pred=pred))


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    evaluate(cfg)
