import logging
import pathlib
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data_class import UFDArguments
from .modeling import (
    UFDAdaptorGlobalModel,
    UFDAdaptorDomainModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
)
from .utils import (
    load_trained_models,
    create_dataset_embedding,
    parse_args_and_load_config,
)


def test_ufd(
    test_data: List,
    adaptor_global: UFDAdaptorGlobalModel,
    adaptor_domain: UFDAdaptorDomainModel,
    maper: UFDCombineFeaturesMapModel,
    model: UFDClassifierModel,
    criterion: torch.nn.CrossEntropyLoss,
    cfg: UFDArguments,
) -> float:
    """Test step method for UFD evaluation

    Args:
        test_data (List): test dataset embeddings
        adaptor_global (UFDAdaptorGlobalModel): adaptor global model
        adaptor_domain (UFDAdaptorDomainModel): adaptor domain model
        maper (UFDCombineFeaturesMapModel): combine features map model
        model (UFDClassifierModel): classifier model
        criterion (torch.nn.CrossEntropyLoss): cross entropy loss criterion
        cfg (UFDArguments): UFDArguments config load from configuration file

    Returns:
        float: loss calculated for the test step
    """
    loss = 0
    acc = 0
    batch_size = cfg.eval_args["eval_batch_size"]
    device = torch.device(cfg.device)

    data = DataLoader(test_data, batch_size=batch_size)
    for i, pairs in enumerate(data):
        text = pairs[:, :, 1:]
        index = pairs[:, :, :1]
        text = torch.squeeze(text)
        index = torch.squeeze(index)
        index = index.long()
        text, index = text.to(device), index.to(device)
        with torch.no_grad():
            global_f, _ = adaptor_global(text)
            domain_f, _ = adaptor_domain(text)
            features = torch.cat((global_f, domain_f), dim=1)
            output = model(maper(features))
            l_crit = criterion(output, index)
            loss += l_crit.item()
            acc += (output.argmax(1) == index).sum().item()

    return loss / len(test_data), acc / len(test_data)


def evaluate(cfg: UFDArguments) -> None:
    """Main evaluation method for UFD models

    Args:
        config (:obj:`UFDArguments`):
            UFDArgument config load from configuration file.

    Example::
        from sgnlp.models.ufd import parse_args_and_load_config
        from sgnlp.models.ufd import evaluate
        cfg = parse_args_and_load_config('config/ufd_config.json')
        evaluate(cfg)
    """

    logging.basicConfig(
        filename=str(
            pathlib.Path(cfg.eval_args["result_folder"])
            / cfg.eval_args["result_filename"]
        ),
        level=logging.INFO,
        force=True,
    )

    device = torch.device(cfg.device)
    loss_criterion = torch.nn.CrossEntropyLoss().to(device)

    # load test data
    test_data = create_dataset_embedding(cfg, dataset_type="test")

    for source_domain in cfg.eval_args["source_domains"]:
        for target_language in cfg.eval_args["target_languages"]:
            for target_domain in cfg.eval_args["target_domains"]:
                if source_domain == target_domain:
                    continue

                (
                    adaptor_domain_model,
                    adaptor_global_model,
                    combine_features_map_model,
                    classifier_model,
                ) = load_trained_models(
                    cfg, source_domain, target_language, target_domain
                )

                test_loss, test_acc = test_ufd(
                    test_data[target_language][target_domain],
                    adaptor_global_model,
                    adaptor_domain_model,
                    combine_features_map_model,
                    classifier_model,
                    loss_criterion,
                    cfg,
                )

                logging.info(
                    f"Model trained on {cfg.eval_args['source_language']} {source_domain}, validated on {target_language} {target_domain} | Test acc: {test_acc}, Test loss: {test_loss}"
                )


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    evaluate(cfg)
