from transformers import BertConfig


class BaseConfig(BertConfig):
    """This is used as base class for derived StanceClassificationConfig and RumourVerificationConfig.

    Args:
        max_tweet_num (int, optional): Number of tweets in each bucket. Defaults to 17.
        max_tweet_length (int, optional): Number of words in each tweet. Defaults to 30.
    """

    def __init__(
        self, max_tweet_num: int = 17, max_tweet_length: int = 30, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.max_tweet_num = max_tweet_num
        self.max_tweet_length = max_tweet_length


class StanceClassificationConfig(BaseConfig):
    """Store the configuration of a StanceClassificationModel.

    Args:
        num_labels (int, optional): Number of label types (e.g. "DENY", "SUPPORT", "QUERY", "COMMENT") including padding. Defaults to 5.
        max_tweet_bucket (int, optional): Number of buckets in each thread. Defaults to 10.
    """

    def __init__(
        self, num_labels: int = 5, max_tweet_bucket: int = 10, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.max_tweet_bucket = max_tweet_bucket


class RumourVerificationConfig(BaseConfig):
    """Store the configuration of a RumourVerificationModel.

    Args:
        num_labels (int, optional): Number of label types (e.g. "0", "1", "2"). Defaults to 3.
        max_tweet_bucket (int, optional): Number of buckets in each thread. Defaults to 4.
    """

    def __init__(
        self, num_labels: int = 3, max_tweet_bucket: int = 4, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.max_tweet_bucket = max_tweet_bucket
