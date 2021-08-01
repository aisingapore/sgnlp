from transformers import BertTokenizer


class RecconSpanExtractionTokenizer(BertTokenizer):
    """
    Constructs a Reccon Span Extraction tokenizer, derived from the Bert tokenizer.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        do_lower_case (:obj:`bool`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.


    Example::

            from sg_nlp import RecconSpanExtractionTokenizer

            tokenizer = RecconSpanExtractionTokenizer.from_pretrained("mrm8488/spanbert-finetuned-squadv2")
            text = "Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan--you must be excited !"
            inputs = tokenizer(text, return_tensors="pt")

    """

    def __init__(self, vocab_file: str, do_lower_case: bool = False, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, do_lower_case=do_lower_case, **kwargs)
