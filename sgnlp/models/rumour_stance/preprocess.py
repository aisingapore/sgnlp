import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .modules.features import (
    BaseFeature,
    BaseFeatures,
    RumourVerificationFeature,
    RumourVerificationFeatures,
    StanceClassificationFeature,
    StanceClassificationFeatures,
)
from .modules.thread import Thread, ThreadPreprocessor
from .tokenization import BaseTokenizer
from .utils import check_path_exists

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """This is used as base class for derived StanceClassificationModel and RumourVerificationModel.

    Args:
        tokenizer (BaseTokenizer): Tokenizer for a model.
        is_stance (bool): Whether the model type is stance classification.
        batch_size (int): Batch size for model training and evaluation.
        train_path (str): Path of training data file.
        dev_path (str): Path of development data file.
        test_path (str): Path of test data file.
        max_tweet_bucket (int): Number of buckets in each thread.
        local_rank (int, optional): The relative rank of the process within the node. Defaults to -1.
        max_tweet_num (int, optional): Number of tweets in each bucket. Defaults to 17.
        max_tweet_length (int, optional): Number of words in each tweet. Defaults to 30.
        max_seq_length (int, optional): Maximum sequence length of the inputs. Defaults to 512.

    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        is_stance: bool,
        batch_size: int,
        train_path: str,
        dev_path: str,
        test_path: str,
        max_tweet_bucket: int,
        local_rank: int = -1,
        max_tweet_num: int = 17,
        max_tweet_length: int = 30,
        max_seq_length: int = 512,
    ) -> None:
        self.tokenizer: BaseTokenizer = tokenizer
        self.is_stance: bool = is_stance
        self.batch_size: int = batch_size
        self.train_config: Dict[str, Any] = {
            "path": Path(__file__).parent / train_path,
            "sampler": RandomSampler if local_rank == -1 else DistributedSampler,
        }
        self.dev_config: Dict[str, Any] = {
            "path": Path(__file__).parent / dev_path,
            "sampler": SequentialSampler,
        }
        self.test_config: Dict[str, Any] = {
            "path": Path(__file__).parent / test_path,
            "sampler": SequentialSampler,
        }
        self.max_tweet_bucket: int = max_tweet_bucket
        self.max_tweet_num: int = max_tweet_num
        self.max_tweet_length: int = max_tweet_length
        self.max_seq_length: int = max_seq_length
        self.num_threads: int

    def __call__(self, texts: List[str]) -> List[torch.Tensor]:
        """Preprocess inputs containing conversation threads for model inference.

        Args:
            texts (List[str]): Raw conversation thread.

        Returns:
            List[torch.Tensor]: Processed conversation threads in Tensors.
        """
        dataloader = self._get_dataloader_from_api([texts])
        for batch in dataloader:
            processed_batch = [b for b in batch]
            LABEL_ID = -2 if self.is_stance else -1
            processed_batch[LABEL_ID] = None
        return processed_batch

    def get_train_dataloader(self) -> DataLoader:
        """Create dataloader for training dataset."""
        check_path_exists("Train configuration", self.train_config["path"])
        return self._get_dataloader_from_file(self.train_config)

    def get_dev_dataloader(self) -> DataLoader:
        """Create dataloader for development dataset."""
        check_path_exists("Dev configuration", self.dev_config["path"])
        return self._get_dataloader_from_file(self.dev_config)

    def get_test_dataloader(self) -> DataLoader:
        """Create dataloader for test dataset."""
        check_path_exists("Test configuration", self.test_config["path"])
        return self._get_dataloader_from_file(self.test_config)

    def _get_dataloader_from_file(
        self,
        config: Dict[str, Any],
    ) -> DataLoader:
        """Create dataloader for a dataset.

        Args:
            config (Dict[str, Any]): Configurations of training, development or test dataset.

        Returns:
            DataLoader: Dataloader for the dataset.
        """
        sampler = config["sampler"]
        threads: List[Thread] = ThreadPreprocessor.from_file(config["path"])
        return self._create_dataloader(sampler, threads)

    def _get_dataloader_from_api(
        self,
        texts: List[List[str]],
    ) -> DataLoader:
        """Create dataloader from raw conversation threads for model inference.

        Args:
            texts (List[List[str]]): Raw conversation threads.

        Returns:
            DataLoader: Dataloader for the conversation threads.
        """
        sampler = SequentialSampler
        threads: List[Thread] = ThreadPreprocessor.from_api(texts)
        return self._create_dataloader(sampler, threads)

    def _create_dataloader(
        self,
        sampler: Callable,
        threads: List[Thread],
    ) -> DataLoader:
        """Create dataloader.

        Args:
            sampler (Callable): Sampler type
            threads (List[Thread]): Conversation threads.

        Returns:
            DataLoader: Dataloader
        """
        self.num_threads = len(threads)
        features_list: List[BaseFeatures] = self._create_features_list(threads)

        dataset = (
            self._get_stance_tensors(features_list)
            if self.is_stance
            else self._get_rumour_tensors(features_list)
        )
        return DataLoader(dataset, sampler=sampler(dataset), batch_size=self.batch_size)

    def get_num_threads(self) -> int:
        """Return the number of conversation threads."""
        return self.num_threads

    def _create_features_list(
        self,
        threads: List[Thread],
    ) -> List[BaseFeatures]:
        """Create features list from conversation threads.

        Args:
            threads (List[Thread]): Conversation threads.


        Returns:
            List[BaseFeatures]: Features list from conversation threads.
        """
        features_list: List[BaseFeatures] = []

        for thread in threads:
            tweets = self._tokenise_texts(thread)
            padded_tweets = self._pad_tokens(tweets)

            if self.is_stance:
                labels = self._get_stance_labels(thread)
                padded_labels = self._pad_stance_labels(labels)

            else:
                padded_labels = [[int(thread.label[0])]]

            features: BaseFeatures = self._create_features(padded_tweets, padded_labels)
            features_list.append(features)

        return features_list

    def _tokenise_texts(
        self,
        thread: Thread,
    ) -> List[List[str]]:
        """Tokenize texts of conversation posts.

        Args:
            thread (Thread): Conversation thread.

        Returns:
            List[List[str]]: Tokenized texts.
        """
        tweets: List[List[str]] = []
        tweet_list = thread.text[: self.max_tweet_num * self.max_tweet_bucket]
        for tweet in tweet_list:
            if tweet == "":
                break
            tweet_token: List[str] = self.tokenizer.tokenize(tweet)
            if len(tweet_token) >= self.max_tweet_length - 1:
                tweet_token = tweet_token[: (self.max_tweet_length - 2)]
            tweets.append(tweet_token)
        return tweets

    def _pad_tokens(
        self,
        tweets: List[List[str]],
    ) -> List[List[List[str]]]:
        """Pad tokenized texts.

        Args:
            tweets (List[List[str]]): Tokenized texts.

        Returns:
            List[List[List[str]]: Padded tokenized texts.
        """
        padded_tweets: List[List[List[str]]] = [
            [] for _ in range(self.max_tweet_bucket)
        ]
        num_chunks = ((len(tweets) - 1) // self.max_tweet_num) + 1
        for i in range(num_chunks):
            min_idx = self.max_tweet_num * i
            max_idx = self.max_tweet_num * (i + 1)
            padded_tweets[i] = tweets[min_idx:max_idx]
        return padded_tweets

    def _get_stance_labels(
        self,
        thread: Thread,
    ) -> List[int]:
        """Generate predicated labels for stance classification.

        Args:
            thread (Thread): Conversation thread.

        Returns:
            List[int]: Predicated labels.
        """
        ID_TO_LABEL: Dict[str, int] = {
            "0": 1,  # "DENY"
            "1": 2,  # "SUPPORT"
            "2": 3,  # "QUERY"
            "3": 4,  # "COMMENT"
        }
        labels: List[str] = thread.label[: self.max_tweet_num * self.max_tweet_bucket]
        return [ID_TO_LABEL[label] for label in labels]

    def _pad_stance_labels(
        self,
        labels: List[int],
    ) -> List[List[int]]:
        """Pad labels for stance classification.

        Args:
            labels (List[int]): Predicated labels.

        Returns:
            List[List[int]]: Padded predicated labels.
        """
        padded_labels: List[List[int]] = [[] for _ in range(self.max_tweet_bucket)]
        num_chunks = ((len(labels) - 1) // self.max_tweet_num) + 1
        for i in range(num_chunks):
            min_idx = self.max_tweet_num * i
            max_idx = self.max_tweet_num * (i + 1)
            padded_labels[i] = labels[min_idx:max_idx]
        return padded_labels

    def _create_features(
        self,
        padded_tweets: List[List[List[str]]],
        padded_labels: List[List[int]],
    ) -> BaseFeatures:
        """Create features from tokenized texts and labels.

        Args:
            padded_tweets (List[List[List[str]]]): Padded tokenized texts.
            padded_labels (List[List[int]]): Padded predicated labels.

        Returns:
            BaseFeatures: Features from conversation threads.
        """
        features = (
            StanceClassificationFeatures()
            if self.is_stance
            else RumourVerificationFeatures()
        )

        for i in range(self.max_tweet_bucket):
            tweets = padded_tweets[i]
            labels = padded_labels[i] if self.is_stance else None
            feature = self._create_feature(tweets, labels)
            features.update(feature)

        if isinstance(features, RumourVerificationFeatures):
            features.single_label_id = padded_labels[0][0]

        return features

    def _create_feature(
        self,
        tweets: List[List[str]],
        labels: Optional[List[int]] = None,
    ) -> BaseFeature:
        """Create feature.

        Args:
            tweets (List[List[str]]): Padded tokenized texts.
            labels (Optional[List[int]]): Padded predicated labels.

        Returns:
            BaseFeature: Feature of a conversation thread.
        """
        ntokens: List[str] = []
        feature = (
            StanceClassificationFeature()
            if self.is_stance
            else RumourVerificationFeature()
        )

        for i, tweet in enumerate(tweets):
            if i != 0 or tweets != []:
                ntokens = ["[CLS]"]
                if (
                    isinstance(feature, StanceClassificationFeature)
                    and labels is not None
                ):
                    feature.label_ids.append(labels[i])
                    feature.label_masks.append(1)
            ntokens.extend(tweet)
            ntokens.append("[SEP]")
            tweet_input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            tweet_input_mask = [1] * len(tweet_input_ids)
            while len(tweet_input_ids) < self.max_tweet_length:
                tweet_input_ids.append(0)
                tweet_input_mask.append(0)
            feature.input_ids.extend(tweet_input_ids)
            feature.input_masks.extend(tweet_input_mask)

            feature.segment_ids = feature.segment_ids + [i % 2] * len(tweet_input_ids)

        pad_tweet_length = self.max_tweet_num - len(tweets)
        for j in range(pad_tweet_length):
            if isinstance(feature, StanceClassificationFeature):
                # Pad label_ids and label_mask
                feature.label_ids.append(0)
                feature.label_masks.append(0)

            # Pad input_ids and input_mask
            tweet_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
            tweet_input_mask = [1] * len(tweet_input_ids)
            tweet_input_ids = tweet_input_ids + [0] * (self.max_tweet_length - 2)
            tweet_input_mask = tweet_input_mask + [0] * (self.max_tweet_length - 2)
            feature.input_ids.extend(tweet_input_ids)
            feature.input_masks.extend(tweet_input_mask)

            # Pad segment_ids
            feature.segment_ids = (
                feature.segment_ids + [(len(tweets) + j) % 2] * self.max_tweet_length
            )

        # Pad input_ids, input_mask and segment_ids
        while len(feature.input_ids) < self.max_seq_length:
            feature.input_ids.append(0)
            feature.input_masks.append(0)
            feature.segment_ids.append(0)

        return feature

    def _get_stance_tensors(
        self, features_list: List[StanceClassificationFeatures]
    ) -> TensorDataset:
        """Get tensors for stance classification features list.

        Args:
            features_list (List[StanceClassificationFeatures]): Features list from conversation threads.

        Returns:
            TensorDataset: Tensors for stance classification features list.
        """
        inputs, input_mask = self._gets_tweets_tensors(features_list)

        label_ids = torch.tensor(
            [f.flatten_label_ids for f in features_list], dtype=torch.long
        )
        label_mask = torch.tensor(
            [f.flatten_label_masks for f in features_list], dtype=torch.long
        )

        return TensorDataset(*(inputs + [input_mask, label_ids, label_mask]))

    def _get_rumour_tensors(
        self, features_list: List[RumourVerificationFeatures]
    ) -> TensorDataset:
        """Get tensors for rumour verification features list.

        Args:
            features_list (List[RumourVerificationFeatures]): Features list from conversation threads.

        Returns:
            TensorDataset: Tensors for rumour verification features list.
        """
        inputs, input_mask = self._gets_tweets_tensors(features_list)
        label_ids = torch.tensor(
            [f.single_label_id for f in features_list], dtype=torch.long
        )
        return TensorDataset(*(inputs + [input_mask, label_ids]))

    def _gets_tweets_tensors(
        self,
        features_list: Union[
            List[StanceClassificationFeatures],
            List[RumourVerificationFeatures],
        ],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Create features list in tensors.

        Args:
            features_list (Union[List[StanceClassificationFeatures], List[RumourVerificationFeatures]]): Features list from conversation threads.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: input_ids, segment_ids, input_masks in tensors, and input_mask.
        """
        inputs: List[torch.Tensor] = []

        for i in range(self.max_tweet_bucket):
            inputs.append(
                torch.tensor(
                    [f.nested_input_ids[i] for f in features_list], dtype=torch.long
                )
            )
            inputs.append(
                torch.tensor(
                    [f.nested_segment_ids[i] for f in features_list], dtype=torch.long
                )
            )
            inputs.append(
                torch.tensor(
                    [f.nested_input_masks[i] for f in features_list], dtype=torch.long
                )
            )

        input_mask = torch.tensor(
            [f.flatten_input_masks for f in features_list], dtype=torch.long
        )

        return inputs, input_mask


class StanceClassificationPreprocessor(BasePreprocessor):
    """Preprocess inputs for training, evaluation and inference of StanceClassificationModel."""

    def __init__(
        self,
        is_stance: bool = True,
        batch_size: int = 2,
        train_path: str = "stance_input/stance_train.tsv",
        dev_path: str = "stance_input/stance_dev.tsv",
        test_path: str = "stance_input/stance_test.tsv",
        max_tweet_bucket: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            is_stance=is_stance,
            batch_size=batch_size,
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
            max_tweet_bucket=max_tweet_bucket,
            **kwargs,
        )


class RumourVerificationPreprocessor(BasePreprocessor):
    """Preprocess inputs for training, evaluation and inference of RumourVerificationModel."""

    def __init__(
        self,
        is_stance: bool = False,
        batch_size: int = 4,
        train_path: str = "rumour_input/rumour_train.tsv",
        dev_path: str = "rumour_input/rumour_dev.tsv",
        test_path: str = "rumour_input/rumour_test.tsv",
        max_tweet_bucket: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            is_stance=is_stance,
            batch_size=batch_size,
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
            max_tweet_bucket=max_tweet_bucket,
            **kwargs,
        )
