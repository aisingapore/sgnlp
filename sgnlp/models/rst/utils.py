import argparse
import json
from typing import Union


from .data_class import RSTParserArguments, RSTSegmenterArguments


def parse_args_and_load_config() -> Union[RSTParserArguments, RSTSegmenterArguments]:
    """Helper method to parse input arguments

    Returns:
        Union[RSTParserArguments, RSTSegmenterArguments]: return either a RSTParserArguments or RSTSegmenterArguments
        depending on the input arguments.
    """
    parser = argparse.ArgumentParser(description="RST Training")
    parser.add_argument(
        '--train_type', type=str, choices=['segmenter', 'parser'], required=True, help='Select which model to train.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file.')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    if args.train_type == 'parser':
        data_class = RSTParserArguments(**config)
    elif args.train_type == 'segmenter':
        data_class = RSTSegmenterArguments(**config)
    else:
        raise ValueError(f'Invalid train_type: {args.train_type}')
    return data_class
