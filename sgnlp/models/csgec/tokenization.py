from transformers import PreTrainedTokenizerFast


class CSGTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

