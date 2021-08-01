from transformers import TransfoXLTokenizer


class RumourDetectionTwitterTokenizer(TransfoXLTokenizer):
    """
    This Tokenizer class performs word-level tokenization to generate tokens.

    Args:
        text (:obj:`str`):
            input text string to tokenize

    Example::
        # 1. From local vocab file
        vocab_file = 'vocab.txt'
        tokenizer = RumourDetectionTwitterTokenizer(vocab_file=vocab_file)
        tokenizer.build_vocab()
        token_ids, token_attention_masks = tokenizer.tokenize_threads(
            [
                ["The quick brown fox", "jumped over the lazy dog"],
                [
                    "Are those shy Eurasian footwear",
                    "cowboy chaps",
                    "or jolly earthmoving headgear?",
                ],
            ]
        )

        # 2. Download pretrained from Azure storage
        #TODO
    """

    def __init__(self, *args, vocab_file, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_file = vocab_file

    #         self.unk_idx = self.unk_token_id

    def tokenize_threads(self, threads, max_length=None, max_posts=None, **kwargs):
        """
        This function performs tokenization on a batch of Twitter threads and returns the token ids and attention masks for each tweet.

        Args:
            threads (List[List[str]]): A batch of threads containing the raw text from the Tweets to be tokenized.
            max_length (int): Maxmium number of tokens in a single Tweet
            max_posts (int): Maximum number of Tweets in a single thread

        Returns:
            :List[List[int]]: token ids for each token in each Tweet. Each tweet/thread would have been padded (or truncated) to `max_length`/`max_posts` respectively.
            :List[List[int]]: attention mask (0 or 1) for each token in each Tweet. Each tweet/thread would have been padded (or truncated) to `max_length`/`max_posts` respectively.
        """
        assert (
            max_length is not None
        ), "Please specify the maximum sequence length for each tweet. This is required to fit all the token indices in a single tensor."

        input_ids = []
        attention_masks = []
        for thread in threads:
            thread_input_ids = self.__call__(
                thread, max_length=max_length, **kwargs
            ).input_ids
            fully_padded_thread = [self.pad_token_id] * max_length

            token_masks = list(
                map(
                    lambda x: list(map(lambda y: 1 if y != 0 else 0, x)),
                    thread_input_ids,
                )
            )

            # pad the threads to max post length
            input_ids += [
                thread_input_ids[:max_posts]
                + [fully_padded_thread] * (max_posts - len(thread))
            ]

            attention_masks += [
                token_masks[:max_posts] + [[0] * max_length] * (max_posts - len(thread))
            ]

        return input_ids, attention_masks
