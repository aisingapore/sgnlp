import pathlib

from .data_class import NEAArguments
from .utils import parse_args_and_load_config


def preprocess_embedding(cfg: NEAArguments) -> None:
    """Preprocess raw w2v embedding downloaded from
    http://ai.stanford.edu/~wzou/mt/biling_mt_release.tar.gz
    and save preprocessed embedding to local directory

    Args:
        cfg (NEAArguments): NEAArguments config load from config.json
    """
    preprocessed_file = []
    full_emb_path = (
        pathlib.Path(__file__).parent
        / cfg.preprocess_embedding_args["raw_embedding_file"]
    )
    with open(full_emb_path, "r") as emb_file:
        for line in emb_file:
            line_split = line.split()
            word = line_split[0]
            word_embedding = line_split[1].replace(",", " ")
            new_line = " ".join([word, word_embedding])
            preprocessed_file.append(new_line)
    preprocessed_file.insert(0, "100229 50")

    full_save_emb_path = (
        pathlib.Path(__file__).parent
        / cfg.preprocess_embedding_args["preprocessed_embedding_file"]
    )
    with open(full_save_emb_path, "w") as emb_file:
        emb_file.write("\n".join(preprocessed_file))


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    preprocess_embedding(cfg)
