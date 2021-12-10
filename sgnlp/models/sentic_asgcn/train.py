from .data_class import SenticASGCNTrainArgs
from .utils import parse_args_and_load_config


def train_model(cfg: SenticASGCNTrainArgs):
    pass


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    train_model(cfg)
