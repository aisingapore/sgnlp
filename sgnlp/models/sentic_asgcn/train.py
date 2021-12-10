from .data_class import SenticASGCNTrainArgs
from .utils import parse_args_and_load_config, set_random_seed


def train_model(cfg: SenticASGCNTrainArgs):
    pass


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
    train_model(cfg)
