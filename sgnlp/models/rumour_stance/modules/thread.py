import csv
from dataclasses import dataclass
from typing import List


@dataclass
class Thread:
    """Conversation thread with stance label for each post.

    Args:
        text (List[str]): Posts in conversation thread.
        label (List[str]): Labels for stance classification ["0": "DENY", "1": "SUPPORT", "2": "QUERY", "3": "COMMENT"] or rumour verification ["0": "FALSE", "1": "TRUE", "2": "UNVERIFIED"].
    """

    text: List[str]
    label: List[str]


class ThreadPreprocessor:
    """Preprocess texts or file for model training, evaluation and inference."""

    @classmethod
    def _read_dataset_from_file(cls, input_file: str) -> List[List[str]]:
        """Read a tsv file.

        Args:
            input_file (str): Path of tsv file.

        Returns:
            List[List[str]]: Lines of tsv file.
        """
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            return list(reader)

    @classmethod
    def from_file(cls, input_file: str) -> List[Thread]:
        """Read and preprocess a tsv file containing train, development or test dataset.

        Args:
            input_file (str): Path of tsv file.

        Returns:
            List[Thread]: Processed conversation threads.
        """
        threads: List[Thread] = []
        lines: List[List[str]] = cls._read_dataset_from_file(input_file)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text = line[2].lower().split("|||||")
            label = line[1].split(",")
            threads.append(Thread(text=text, label=label))
        return threads

    @classmethod
    def from_api(cls, lines: List[List[str]]) -> List[Thread]:
        """Preprocess inputs containing conversation threads for model inference.

        Args:
            lines (List[List[str]]): Raw conversation threads.

        Returns:
            List[Thread]: Processed conversation threads.
        """
        threads: List[Thread] = []
        for line in lines:
            text = [l.lower() for l in line]
            label = ["1"] * len(line)
            threads.append(Thread(text=text, label=label))
        return threads
