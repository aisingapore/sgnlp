
from .utils import parse_args_and_load_config
from .data_class import RstPointerParserTrainArgs, RstPointerSegmenterTrainArgs


def train_parser(cfg: RstPointerParserTrainArgs) -> None:
    pass


def train_segmenter(cfg: RstPointerSegmenterTrainArgs) -> None:
    pass


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if isinstance(cfg, RstPointerSegmenterTrainArgs):
        train_segmenter(cfg)
        print(cfg)
    if isinstance(cfg, RstPointerParserTrainArgs):
        train_parser(cfg)
        print(cfg)
