"""
Training or finetuning a LSR model on DocRED dataset.
"""
import csv
import os
import json
import torch
import logging
import argparse
import numpy as np
from sklearn import metrics
from torch.utils.data.dataloader import DataLoader, default_collate
from transformers import set_seed

from .config import LsrConfig
from .modeling import LsrModel, LsrModelOutput
from .preprocess import LsrPreprocessor
from .utils import h_t_idx_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LsrMetrics:
    """
    For calculating metrics for LsrModel.
    Computes precision, recall, f1, auc for precision vs recall, and the optimal prediction threshold (theta).
    This is modified from the original.
    The original additionally computes an ignore score which ignores facts seen in training set.
    """

    def __init__(self, num_relations):
        self.num_relations = num_relations

        self.test_result = []
        self.total_recall = 0

    def add_batch(self, model_output: LsrModelOutput, batch_data):
        predict_re = torch.sigmoid(model_output.prediction)
        predict_re = predict_re.data.cpu().numpy()

        for i, label in enumerate(batch_data['labels']):
            self.total_recall += len(label)

            vertex_set_length = batch_data['entity_num_list'][i]  # the number of entities in each instance.
            for j, (h_idx, t_idx) in enumerate(h_t_idx_generator(vertex_set_length)):
                for r in range(1, self.num_relations):
                    result_tuple = (
                        (h_idx, t_idx, r) in label,
                        float(predict_re[i, j, r]),
                    )
                    self.test_result.append(result_tuple)

    def compute(self, input_theta=None):
        """
        Computes metrics based on data added so far.

        Args:
            input_theta (`optional`, `float`):
                Prediction threshold. Provide a value between 0 to 1 if you want to compute the precision and recall
                for that specific threshold. Otherwise the optimal based on f1 score will be computed for you.
        """
        # Sorts in descending order by predicted value
        self.test_result.sort(key=lambda x: x[1], reverse=True)

        precision = []
        recall = []
        correct = 0
        w = 0
        if self.total_recall == 0:
            self.total_recall = 1  # for test

        for i, item in enumerate(self.test_result):
            correct += item[0]
            recall.append(float(correct) / (i + 1))
            precision.append(float(correct) / self.total_recall)
            if input_theta is not None and item[1] > input_theta:
                w = i

        precision = np.asarray(precision, dtype='float32')
        recall = np.asarray(recall, dtype='float32')
        f1_arr = (2 * precision * recall / (precision + recall + 1e-20))
        auc = metrics.auc(x=precision, y=recall)

        if input_theta is None:
            f1 = f1_arr.max()
            f1_pos = f1_arr.argmax()
            best_theta = self.test_result[f1_pos][1]
            return best_theta, f1, precision[f1_pos], recall[f1_pos], auc
        else:
            return input_theta, f1_arr[w], precision[w], recall[w], auc


class DocredDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, preprocessor):
        self.data = json.load(open(json_file))
        self.preprocessed_data = preprocessor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = {}
        for key in self.preprocessed_data.keys():
            instance[key] = self.preprocessed_data[key][idx]
        return instance


def lsr_collate_fn(batch):
    """
    Manually processes labels and uses default for the rest.
    """
    labels = []
    for instance in batch:
        labels_instance = instance.pop("labels")
        labels.append(labels_instance)

    collated_data = default_collate(batch)

    collated_data["labels"] = labels
    return collated_data


class MyAdagrad(torch.optim.Optimizer):
    """
    Modification of the Adagrad optimizer that allows to specify an initial
    accumulator value. This mimics the behavior of the default Adagrad implementation
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial accumulator value.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulater value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, \
                        weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) * \
                               init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)

                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss


def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name in ['adagrad', 'myadagrad']:
        # use custom adagrad to allow for init accumulator value
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr, weight_decay=l2)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=l2)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2)
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a relation extraction model based on latent structure refinement.")

    parser.add_argument("--train_file", type=str, required=True, help="A json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A json file containing the validation data.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path.")
    parser.add_argument("--metadata_dir", type=str, required=True, help="Path to docred metadata directory.")

    # Model arguments
    parser.add_argument("--model_weights_path", type=str, default=None,
                        help="Provide a path to model weights if you want to finetune from a checkpoint.")
    parser.add_argument("--model_config_path", type=str, default=None,
                        help="Provide a config if you want to override the defaults. "
                             "To use bert encoder layer, specify in config file.")
    parser.add_argument("--pretrained_embeddings_path", type=str, default=None,
                        help="Provide a path to pretrained embeddings if you want to want to use them.")

    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning rate decay rate.')
    parser.add_argument('--decay_epoch', type=int, default=20, help='Decay learning rate after this epoch.')
    parser.add_argument('--evaluate_epoch', type=int, default=30, help='Evaluate after this epoch.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamw', 'adamax'], default='adam',
                        help='Choice of optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Training batch size.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--seed', type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument('--use_gpu', type=bool, default=True, help="Whether you want to use GPU for training.")

    parser.add_argument('--use_wandb', type=bool, default=False, help="Whether to use wandb to monitor training.")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="Wandb run name.")

    args = parser.parse_args()

    return args


def train(args):
    logger.info(f"Training arguments: {vars(args)}")

    # Defaults to cpu if gpu is unavailable
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Make output dir, save training args, create metrics files
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'best_f1'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'final'), exist_ok=True)

        with open(os.path.join(args.output_dir, 'training_args.json'), "w") as fp:
            json.dump(vars(args), fp, sort_keys=True, indent=4)

        # create metric output files
        train_metrics_file = os.path.join(args.output_dir, 'train_metrics.csv')
        val_metrics_file = os.path.join(args.output_dir, 'val_metrics.csv')
        fieldnames = ['epoch', 'loss', 'best_theta', 'f1', 'precision', 'recall', 'auc']

        with open(train_metrics_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        with open(val_metrics_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    if args.seed is not None:
        set_seed(args.seed)

    # Load config if provided else initialize with defaults
    if args.model_config_path:
        config = LsrConfig.from_json_file(json_file=args.model_config_path)
    else:
        config = LsrConfig()

    # Load model weights if provided
    if args.model_weights_path:
        model = LsrModel.from_pretrained(args.model_weights_path, config=config)
    else:
        model = LsrModel(config)

    if args.use_wandb:
        import wandb
        wandb.init(project="lsr", name=args.wandb_run_name)
        wandb.watch(model, log="all")

    # Note: this will override the provided model weights
    if args.pretrained_embeddings_path is not None and not config.use_bert:
        pretrained_embeddings = np.load(args.pretrained_embeddings_path)
        model.load_pretrained_word_embedding(pretrained_word_embedding=pretrained_embeddings)

    # Set to training device
    model.to(device=device)

    # Load dataset
    # Set to cpu initially (for preprocessing entire dataset first)
    logger.info("Preprocessing datasets...")
    rel2id_path = os.path.join(args.metadata_dir, "rel2id.json")
    word2id_path = os.path.join(args.metadata_dir, "word2id.json")
    ner2id_path = os.path.join(args.metadata_dir, "ner2id.json")
    train_preprocessor = LsrPreprocessor(rel2id_path=rel2id_path, word2id_path=word2id_path, ner2id_path=ner2id_path,
                                         is_train=True, device=torch.device("cpu"), config=config)
    val_preprocessor = LsrPreprocessor(rel2id_path=rel2id_path, word2id_path=word2id_path, ner2id_path=ner2id_path,
                                       device=torch.device("cpu"), config=config)
    train_dataset = DocredDataset(json_file=args.train_file, preprocessor=train_preprocessor)
    val_dataset = DocredDataset(json_file=args.validation_file, preprocessor=val_preprocessor)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=lsr_collate_fn)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=lsr_collate_fn)

    # Optimizer and parameters
    if config.use_bert:
        other_params = [p for name, p in model.named_parameters() if
                        p.requires_grad and not name.startswith("bert")]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": args.lr},
            {"params": model.bert.parameters(), "lr": 1e-5},
        ], lr=args.lr)
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = get_optimizer(args.optim, parameters, args.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    best_f1 = 0

    for epoch in range(args.num_epoch):
        logger.info(f"Epoch: {epoch + 1}/{args.num_epoch}, lr: {optimizer.param_groups[0]['lr']}")

        train_metrics = LsrMetrics(num_relations=config.num_relations)
        total_epoch_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=device)

            outputs = model(**batch)

            # Backpropagation
            loss = outputs.loss
            # TODO: Remove debug logs below
            if np.isnan(loss.item()):
                logger.info("Skipping backward prop.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # Track metrics
            total_epoch_train_loss += loss.item()
            train_metrics.add_batch(model_output=outputs, batch_data=batch)

        # Compute metrics and log at end of epoch
        best_theta, f1, precision, recall, auc = train_metrics.compute()
        avg_epoch_train_loss = total_epoch_train_loss / (step + 1)
        logger.info(f"Train loss: {avg_epoch_train_loss:.3f}, best theta: {best_theta:.3f}, "
                    f"f1: {f1:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, auc: {auc:.5f}")

        if args.use_wandb:
            wandb.log({
                "train_loss": avg_epoch_train_loss,
                "train_best_theta": best_theta,
                "train_f1": f1,
                "train_precision": precision,
                "train_recall": recall,
                "train_auc": auc
            }, step=epoch)

        # Write train metrics
        if args.output_dir is not None:
            with open(train_metrics_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({'epoch': epoch + 1, 'loss': avg_epoch_train_loss, 'best_theta': best_theta, 'f1': f1,
                                 'precision': precision, 'recall': recall, 'auc': auc})

        if epoch + 1 >= args.evaluate_epoch:
            val_metrics = LsrMetrics(num_relations=config.num_relations)
            total_epoch_val_loss = 0
            model.eval()
            for step, batch in enumerate(val_dataloader):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device=device)

                outputs = model(**batch)

                # Track metrics
                total_epoch_val_loss += loss.item()
                val_metrics.add_batch(model_output=outputs, batch_data=batch)

            # Compute metrics and log at end of epoch
            best_theta, f1, precision, recall, auc = val_metrics.compute()
            avg_epoch_val_loss = total_epoch_val_loss / (step + 1)
            logger.info(f"Val loss: {avg_epoch_val_loss:.3f}, best theta: {best_theta:.3f}, "
                        f"f1: {f1:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, auc: {auc:.5f}")

            if args.use_wandb:
                wandb.log({
                    "val_loss": avg_epoch_val_loss,
                    "val_best_theta": best_theta,
                    "val_f1": f1,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_auc": auc
                }, step=epoch)

            # Write val metrics
            if args.output_dir is not None:
                with open(val_metrics_file, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(
                        {'epoch': epoch + 1, 'loss': avg_epoch_val_loss, 'best_theta': best_theta, 'f1': f1,
                         'precision': precision, 'recall': recall, 'auc': auc})

            # Save best model so far
            if args.output_dir is not None and f1 > best_f1:
                logger.info("Best f1 so far, saving model.")
                best_f1 = f1
                model.save_pretrained(os.path.join(args.output_dir, 'best_f1'))

        if epoch + 1 >= args.decay_epoch:
            if args.optim == 'adam' and optimizer.param_groups[0]['lr'] > 1e-4:
                scheduler.step()

    # Save final model
    if args.output_dir is not None:
        logger.info("Saving final model.")
        model.save_pretrained(os.path.join(args.output_dir, 'final'))


if __name__ == "__main__":
    args = parse_args()
    train(args)
