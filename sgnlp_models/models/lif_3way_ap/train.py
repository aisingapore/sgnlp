import argparse
import logging
import os
import pathlib
import json
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from transformers import set_seed
from datetime import datetime

from sgnlp_models.models.lif_3way_ap.config import LIF3WayAPConfig
from sgnlp_models.models.lif_3way_ap.modeling import LIF3WayAPModel
from sgnlp_models.models.lif_3way_ap.preprocess import LIF3WayAPPreprocessor, lif_3way_ap_collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LIFMetrics:
    """
    Calculates roc auc score, macro f1 score, and optimal classification threshold (eps).
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add_batch(self, model_output, batch_data):
        self.y_true += batch_data["label"].tolist()
        self.y_pred += model_output["label_probs"].tolist()

    def compute(self):
        roc_auc = roc_auc_score(self.y_true, self.y_pred)

        eps_best = 0
        macro_f1_best = 0
        for i in range(1, 1000):
            eps = i / 1000
            eps_preds = [1 if score > eps else 0 for score in self.y_pred]
            macro_f1 = f1_score(self.y_true, eps_preds, average='macro')
            if macro_f1 > macro_f1_best:
                macro_f1_best = macro_f1
                eps_best = eps
        return roc_auc, macro_f1_best, eps_best


class LIFDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for identifying follow up questions.")

    parser.add_argument("--train_file", type=str, required=True, help="A json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A json file containing the validation data.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path.")

    # Model arguments
    parser.add_argument("--model_config_path", type=str, default=None,
                        help="Provide a config if you want to override the defaults. "
                             "To use bert encoder layer, specify in config file.")
    parser.add_argument("--pretrained_embeddings_path", type=str, default=None,
                        help="Provide a path to pretrained embeddings if you want to want to use them.")

    # Dataset preprocessing arguments
    parser.add_argument('--max_vocab_size', type=int, default=None, help='Maximum vocabulary size.')
    parser.add_argument('--num_context_answers', type=int, default=3,
                        help='Max number of questions and answers to consider.')

    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, default=25, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--seed', type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument('--use_gpu', type=bool, default=True, help="Whether you want to use GPU for training.")

    args = parser.parse_args()

    return args


def main(args):
    logger.info(f"Training arguments: {vars(args)}")

    # Defaults to cpu if gpu is unavailable
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    timestamped_output_dir = None
    if args.output_dir is not None:
        timestamped_output_dir = os.path.join(args.output_dir, datetime.utcnow().strftime('%Y%m%d.%H%M%S'))
        pathlib.Path(timestamped_output_dir).mkdir(parents=True, exist_ok=False)

    if args.seed is not None:
        set_seed(args.seed)

    # Load model config
    config = LIF3WayAPConfig.from_json_file(json_file=args.model_config_path)

    logger.info("Preprocessing datasets...")
    preprocessor = LIF3WayAPPreprocessor(min_word_padding_size=config.char_embedding_args["kernel_size"],
                                         num_context_answers=args.num_context_answers)

    with open(args.train_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
    train_data = preprocessor.process_dataset(dataset)
    train_tokenized_data = preprocessor.build_vocab(batch_data=train_data,
                                                    max_word_vocab_size=args.max_vocab_size)
    train_vectorized_data = preprocessor.vectorize(train_tokenized_data)
    train_dataset = LIFDataset(train_vectorized_data)

    preprocessor.save_vocab(timestamped_output_dir)

    with open(args.validation_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
    val_data = preprocessor.process_dataset(dataset)
    val_tokenized_data, _ = preprocessor.tokenize(val_data)
    val_vectorized_data = preprocessor.vectorize(val_tokenized_data)
    val_dataset = LIFDataset(val_vectorized_data)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=preprocessor.collate_fn)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=preprocessor.collate_fn)

    # Model init
    # Set num embedding according to vocab
    num_embeddings = len(preprocessor.word_vocab)
    config.word_embedding_args["num_embeddings"] = num_embeddings
    logger.info(f"Num word embeddings: {num_embeddings}")

    logger.info(f"Initializing model...")
    model = LIF3WayAPModel(config)

    # Load pretrained embeddings if provided
    if args.pretrained_embeddings_path:
        logger.info("Loading pretrained embeddings")
        model.word_embedding.load_pretrained_embeddings(file_path=args.pretrained_embeddings_path,
                                                        vocab=preprocessor.word_vocab)
    model.to(device=device)

    best_f1 = 0

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=1)

    for epoch in range(args.num_epoch):
        logger.info(f"Epoch: {epoch + 1}/{args.num_epoch}, lr: {optimizer.param_groups[0]['lr']}")

        train_metrics = LIFMetrics()
        total_epoch_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=device)

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs["loss"]  # TODO: Refactor output type?

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_epoch_train_loss += loss.item()
            train_metrics.add_batch(model_output=outputs, batch_data=batch)

        avg_epoch_train_loss = total_epoch_train_loss / (step + 1)
        roc_auc, macro_f1, best_eps = train_metrics.compute()
        logger.info(f"Train loss: {avg_epoch_train_loss:.3f}, roc_auc: {roc_auc:.3f}, macro_f1: {macro_f1:.3f}, "
                    f"best_eps: {best_eps:.3f}")

        val_metrics = LIFMetrics()
        total_epoch_val_loss = 0
        model.eval()
        for step, batch in enumerate(val_dataloader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=device)

            outputs = model(**batch)
            loss = outputs["loss"]  # TODO: Refactor output type?

            total_epoch_val_loss += loss.item()
            val_metrics.add_batch(model_output=outputs, batch_data=batch)

        avg_epoch_val_loss = total_epoch_val_loss / (step + 1)
        roc_auc, macro_f1, best_eps = val_metrics.compute()
        logger.info(f"Val loss: {avg_epoch_val_loss:.3f}, roc_auc: {roc_auc:.3f}, macro_f1: {macro_f1:.3f}, "
                    f"best_eps: {best_eps:.3f}")

        if timestamped_output_dir is not None and macro_f1 > best_f1:
            logger.info("Best f1 so far, saving model.")
            best_f1 = macro_f1
            model.save_pretrained(os.path.join(timestamped_output_dir, 'best_f1'))

        scheduler.step(roc_auc)

    # Save final model
    if timestamped_output_dir is not None:
        logger.info("Saving final model.")
        model.save_pretrained(os.path.join(timestamped_output_dir, 'final'))


if __name__ == "__main__":
    args = parse_args()
    main(args)
