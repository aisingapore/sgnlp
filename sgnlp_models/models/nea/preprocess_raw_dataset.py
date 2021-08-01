import pathlib
from typing import Dict, List

from .data_class import NEAArguments
from .utils import parse_args_and_load_config


def collect_dataset(input_file: str) -> Dict:
    """Read and parse lines from dataset files

    Args:
        input_file (str): path to input file

    Returns:
        Dict: return dict of dataset
    """
    dataset = {}
    lcount = 0
    with open(input_file, "r", encoding="latin-1") as f:
        for line in f:
            lcount += 1
            if lcount == 1:
                dataset["header"] = line
                continue
            parts = line.split("\t")
            assert len(parts) >= 6, f"Error: {line}"
            dataset[parts[0]] = line
    return dataset


def extract_based_on_ids(dataset: Dict, id_file: str) -> List[str]:
    """Extract ids from dataset files

    Args:
        dataset (Dict): dict of dataset
        id_file (str): id of the dataset file

    Returns:
        List[str]: return list of dataset entries
    """
    lines = []
    with open(id_file) as f:
        for line in f:
            id = line.strip()
            try:
                lines.append(dataset[id])
            except ValueError:
                print(f"Error: Invalid ID {id} in {id_file}")
    return lines


def create_dataset(dataset: Dict, lines: List[str], output_fname: str) -> None:
    """Helper method to generate dataset files

    Args:
        dataset (Dict): dict of dataset
        lines (List[str]): lines of dataset entries
        output_fname (str): output file names
    """
    with open(output_fname, "w", encoding="utf-8") as f:
        f.write(dataset["header"])
        for line in lines:
            f.write(line)


def preprocess(cfg: NEAArguments) -> None:
    """Main preprocess method

    Args:
        cfg (NEAArguments): NEAArguments config load from config file.
    """
    data_folder = pathlib.Path(__file__).parent.joinpath(
        cfg.preprocess_raw_dataset_args["data_folder"]
    )
    input_file_path = data_folder.joinpath(
        cfg.preprocess_raw_dataset_args["input_file"]
    )
    dataset = collect_dataset(input_file_path)
    for fold_idx in range(5):
        for dataset_type in ["dev", "test", "train"]:
            txt_file_path = data_folder.joinpath(f"fold_{fold_idx}").joinpath(
                f"{dataset_type}_ids.txt"
            )
            lines = extract_based_on_ids(dataset, txt_file_path)
            dataset_file_path = data_folder.joinpath(f"fold_{fold_idx}").joinpath(
                f"{dataset_type}.tsv"
            )
            create_dataset(dataset, lines, dataset_file_path)


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    preprocess(cfg)
