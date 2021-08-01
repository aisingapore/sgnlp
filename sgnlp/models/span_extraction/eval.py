import json
import os

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from .tokenization import RecconSpanExtractionTokenizer
from .modeling import RecconSpanExtractionModel
from .data_class import RecconSpanExtractionArguments
from .utils import (
    parse_args_and_load_config,
    RawResult,
    to_list,
    load_examples,
    write_predictions,
    calculate_results,
    evaluate_results,
)


def evaluate(cfg: RecconSpanExtractionArguments):
    """
    Method to evaluate a trained RecconSpanExtractionModel.

    Args:
        config (:obj:`RecconSpanExtractionArguments`):
            RecconSpanExtractionArguments config load from config file.

    Example::

            import json
            from sgnlp.models.span_extraction import evaluate
            from sgnlp.models.span_extraction.utils import parse_args_and_load_config

            cfg = parse_args_and_load_config('config/span_extraction_config.json')
            evaluate(cfg)
    """
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not cfg.eval_args["no_cuda"]
        else torch.device("cpu")
    )

    tokenizer = RecconSpanExtractionTokenizer.from_pretrained(cfg.model_name)
    model = RecconSpanExtractionModel.from_pretrained(
        cfg.eval_args["trained_model_dir"]
    )

    with open(cfg.test_data_path, "r") as f:
        test_json = json.load(f)

    eval_dataset, examples, features = load_examples(
        test_json, tokenizer, evaluate=True, output_examples=True
    )

    eval_sample = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sample, batch_size=cfg.eval_args["batch_size"]
    )

    eval_loss = 0.0
    nb_eval_steps = 0
    model.to(device)
    model.eval()

    all_results = []
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)
            eval_loss += outputs[0].mean().item()

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(
                    unique_id=unique_id,
                    start_logits=to_list(outputs[0][i]),
                    end_logits=to_list(outputs[1][i]),
                )
                all_results.append(result)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    prefix = "text"
    os.makedirs(cfg.eval_args["results_path"], exist_ok=True)

    output_prediction_file = os.path.join(
        cfg.eval_args["results_path"], "predictions_{}.json".format(prefix)
    )
    output_nbest_file = os.path.join(
        cfg.eval_args["results_path"], "nbest_predictions_{}.json".format(prefix)
    )
    output_null_log_odds_file = os.path.join(
        cfg.eval_args["results_path"], "null_odds_{}.json".format(prefix)
    )

    all_predictions, all_nbest_json, scores_diff_json = write_predictions(
        examples,
        features,
        all_results,
        cfg.eval_args["n_best_size"],
        cfg.eval_args["max_answer_length"],
        False,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        True,
        cfg.eval_args["null_score_diff_threshold"],
    )

    result, texts = calculate_results(test_json, all_predictions)

    r = evaluate_results(texts)

    with open(
        os.path.join(cfg.eval_args["results_path"], "results.txt"), "w"
    ) as result_file:
        result_file.write(r)


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    evaluate(cfg)
