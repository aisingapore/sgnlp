from data_class import SenticASGCNTrainArgs
from utils import parse_args_and_load_config, set_random_seed, ABSADatasetReader, BucketIterator

from tokenization import SenticASGCNTokenizer


class Trainer:
    def __init__(self, cfg: SenticASGCNTrainArgs):
        self.cfg = cfg
        tokenizer = SenticASGCNTokenizer.from_pretrained(
            "/Users/raymond/work/aimakerspace_sgnlp/sgnlp/models/sentic_asgcn/tokenizer/"
        )
        dataset = ABSADatasetReader(self.cfg, tokenizer=tokenizer)

    def _train(self):
        pass


def train_model(cfg: SenticASGCNTrainArgs):
    tokenizer = SenticASGCNTokenizer.from_pretrained(
        "/Users/raymond/work/aimakerspace_sgnlp/sgnlp/models/sentic_asgcn/tokenizer/"
    )
    absa_dataset = ABSADatasetReader(cfg, tokenizer)
    train_dataloader = BucketIterator(data=absa_dataset.train_data, batch_size=cfg.batch_size, shuffle=True)
    test_dataloader = BucketIterator(data=absa_dataset.test_data, batch_size=cfg.batch_size, shuffle=False)


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
    train_model(cfg)
