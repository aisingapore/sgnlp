
from .utils import parse_args_and_load_config
from .data_class import RSTParserArguments, RSTSegmenterArguments


def train_parser(cfg: RSTParserArguments) -> None:
    pass


def train_segmenter(cfg: RSTSegmenterArguments) -> None:
    pass


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if isinstance(cfg, RSTSegmenterArguments):
        train_segmenter(cfg)
        print(cfg)
    if isinstance(cfg, RSTParserArguments):
        train_parser(cfg)
        print(cfg)
