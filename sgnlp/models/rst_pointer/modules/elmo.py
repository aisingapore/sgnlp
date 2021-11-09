from typing import Tuple
from allennlp.modules.elmo import Elmo

elmo_metadata = {
    "Large": {
        "word_dim": 1024,
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
    },
    "Medium": {
        "word_dim": 512,
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json",
    },
    "Small": {
        "word_dim": 256,
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
    },
}


def initialize_elmo(size: str = "Large") -> Tuple[Elmo, int]:
    """Helper function to create Elmo object.

    Args:
        size (str): size of elmo model to use. Sizes available: ["Small", "Medium", "Large"]

    Returns:
        Tuple[Elmo, int]: return initialized elmo object and the word dimension.
    """
    metadata = elmo_metadata[size]
    return (
        Elmo(
            metadata["options_file"],
            metadata["weight_file"],
            2,
            dropout=0.5,
            requires_grad=False,
        ),
        metadata["word_dim"],
    )
