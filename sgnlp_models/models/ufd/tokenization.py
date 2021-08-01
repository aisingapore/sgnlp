from transformers import XLMRobertaTokenizer


class UFDTokenizer(XLMRobertaTokenizer):
    """
    The UFD Tokenizer class used for to generate token for the embedding model, derived from
    XLM Roberta Tokenizer class.

    Args:
        text (:obj:`str`):
            input text string to tokenize

    Example::
            tokenizer = UFDTokenizer.from_pretrained('xlm-roberta-large')
            inputs = tokenizer('Hello World!')
            inputs["input_ids"]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, text, **kwargs):
        # Return_tensor is set to "pt" else the truncating logic will be different if it is not a pt tensor
        encoding = super().__call__(
            text, return_tensors="pt", truncation=True, max_length=513, **kwargs
        )
        # Replicating how UFD paper did the truncation
        # Truncate to maximum of 512 tokens if length exceeds 512
        for key in encoding.keys():
            encoding[key] = encoding[key][:, :512]

        return encoding
