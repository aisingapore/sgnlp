from transformers import RobertaTokenizer


class RecconEmotionEntailmentTokenizer(RobertaTokenizer):
    """
    Constructs a Reccon Emotion Entailment tokenizer, derived from the RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        do_lower_case (:obj:`bool`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.


    Example::

            from sg_nlp import RecconEmotionEntailmentTokenizer

            tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
            text = "surprise <SEP> Me ? You're the one who pulled out in front of me ! <SEP> Why don't you watch where you're going ? <SEP> Why don't you watch where you're going ? Me ? You're the one who pulled out in front of me !"
            inputs = tokenizer(text, return_tensors="pt")

    """

    def __init__(
        self, vocab_file: str, merges_file: str, do_lower_case: bool = False, **kwargs
    ) -> None:
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            do_lower_case=do_lower_case,
            **kwargs
        )
