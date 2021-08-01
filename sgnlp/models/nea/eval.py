import logging


from .data_class import NEAArguments
from .tokenization import NEATokenizer
from .utils import (
    NEATrainingArguments,
    init_model,
    parse_args_and_load_config,
    get_model_friendly_scores,
    pad_sequences_from_list,
    NEATrainer,
    NEADataset,
    build_compute_metrics_fn,
    load_test_dataset,
    process_results,
)

logging.basicConfig(level=logging.DEBUG)


def evaluate(cfg: NEAArguments) -> None:
    """Method for evaluating trained Neural Essay Assessor model.

    NEA evaluate method requires the use of NLTK 'punkt' package.
    Please download the required pacakges as shown in example below prior to running the evaluate method.

    Args:
        cfg (:obj:`NEAArguments`): NEAArguments config load from configuration file.

    Example::
        # Download NLTK package
        import nltk
        nltk.download('punkt')

        # From Code
        import json
        from sgnlp.models.nea.utils import parse_args_and_load_config
        from sgnlp.models.nea import evaluate
        cfg = parse_args_and_load_config('config/nea_config.json')
        evaluate(cfg)
    """
    logging.info(f"Evaluation arguments: {cfg}")

    (test_x, test_y, test_pmt) = load_test_dataset(cfg)

    # Preprocess data
    tokenizer = NEATokenizer.from_pretrained(cfg.tokenizer_args["save_folder"])
    test_x = tokenizer(test_x)["input_ids"]
    test_x = pad_sequences_from_list(test_x)
    test_y = get_model_friendly_scores(test_y, test_pmt)
    test_data = NEADataset(test_x, test_y)

    # initialise trainer
    model = init_model(cfg)
    model = model.from_pretrained(cfg.eval_args["trainer_args"]["output_dir"])
    compute_metrics_fn = build_compute_metrics_fn(cfg)
    eval_trainer_args = NEATrainingArguments(**cfg.eval_args["trainer_args"])
    trainer = NEATrainer(
        model=model, compute_metrics=compute_metrics_fn, args=eval_trainer_args
    )

    # Predict
    _, _, metrics = trainer.predict(test_data)
    logging.info(f"Evaluation metrics on test set: {metrics}")

    result_output = process_results(metrics)
    with open(cfg.eval_args["results_path"], "w") as result_file:
        result_file.write(result_output)


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    evaluate(cfg)
