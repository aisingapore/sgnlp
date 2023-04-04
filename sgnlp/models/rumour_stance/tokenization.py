from transformers import BertTokenizer


class BaseTokenizer(BertTokenizer):
    """This is used as base class for derived StanceClassificationTokenizer and RumourVerificationTokenizer.

    Args:
        vocab_file (str): File containing the vocabulary.
    """

    def __init__(self, vocab_file: str, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, **kwargs)


class StanceClassificationTokenizer(BaseTokenizer):
    """A BERT-based tokenizer for StanceClassificationModel."""


class RumourVerificationTokenizer(BaseTokenizer):
    """A BERT-based tokenizer for RumourVerificationModel."""
