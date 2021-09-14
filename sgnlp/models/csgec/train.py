import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import set_seed

from .config import CSGConfig
from .modeling import CSGModel
from .tokenization import CSGTokenizerFast
from .utils import load_transform_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the cross sentence grammatical error correction model.")

    # experiment info
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--experiment_number", type=str, required=True)
    parser.add_argument("--log_frequency", type=int, default=100)

    # model parameters
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--source_tokenizer_json", required=True, type=str, default=None)
    parser.add_argument("--target_tokenizer_json", required=True, type=str, default=None)
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--pretrained_source_embedding_path", type=str, default=None)
    parser.add_argument("--pretrained_target_embedding_path", type=str, default=None)
    parser.add_argument("--max_sentence_length", type=int, default=150)
    parser.add_argument("--max_context_length", type=int, default=250)
    parser.add_argument("--padding_token", type=str, default="[PAD]")
    
    # optimizer parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_annealing_factor', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)

    # other parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_gpu', type=bool, default=True)

    args = parser.parse_args()

    return args


def main(args):

    # Log the experiment name and info
    logger.info(f"Experiment {args.experiment_name}-{args.experiment_number}")

    # Default to cpu if GPU is unavailable
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")

    # Create the output dir. Raises an error if the folder has been created
    experiment_output_dir = os.path.join(args.output_dir, args.experiment_name, args.experiment_number)
    assert not os.path.exists(experiment_output_dir), "Experiment folder exists. Please change experiment name and/r experiment number"
    os.makedirs(experiment_output_dir)

    # Save train arguments
    with open(os.path.join(experiment_output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Load config if provided
    if args.model_config_path:
        config = CSGConfig.from_json_file(json_file=args.model_config_path)
    else:
        config = CSGConfig()

    model = CSGModel(config)

    # Load pretrained embeddings if provided
    if (args.pretrained_source_embedding_path is not None) and (args.pretrained_target_embedding_path is not None):
        model.load_pretrained_embedding(args.pretrained_source_embedding_path, args.pretrained_target_embedding_path)

    # Move model to device
    model.train().to(device)

    # Load tokenizers
    source_tokenizer = CSGTokenizerFast(tokenizer_file=args.source_tokenizer_json, model_max_length=args.max_sentence_length, pad_token=args.padding_token)
    context_tokenizer = CSGTokenizerFast(tokenizer_file=args.source_tokenizer_json, model_max_length=args.max_context_length, pad_token=args.padding_token)
    target_tokenizer = CSGTokenizerFast(tokenizer_file=args.target_tokenizer_json, model_max_length=args.max_sentence_length, pad_token=args.padding_token)

    # Load and transform dataset
    train_dataloader = load_transform_dataset(args.train_data, args.batch_size, source_tokenizer, context_tokenizer, target_tokenizer)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_annealing_factor)
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(args.num_epoch):

        logger.info(f"starting epoch {epoch}")

        for i, batch in enumerate(tqdm(train_dataloader)):
            source_ids = batch["source_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            context_ids = batch["context_ids"].to(device)

            outputs = model(source_ids, context_ids, target_ids)
            outputs = outputs.reshape(-1, target_tokenizer.vocab_size)

            loss = criterion(outputs, target_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % args.log_frequency == 0:
                logger.info(f"epoch: {epoch} batch: {i} loss: {loss} lr: {scheduler.get_last_lr()}")
        
        scheduler.step()

        epoch_output_dir = os.path.join(experiment_output_dir, "epoch-" + str(epoch))
        os.makedirs(epoch_output_dir)
        model.save_pretrained(epoch_output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)

