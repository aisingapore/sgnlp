from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput
from typing import Optional

from .config import RumourDetectionTwitterConfig
from .modules.encoder.word_encoder import WordEncoder
from .modules.encoder.position_encoder import PositionEncoder
from .modules.transformer.hierarchical_transformer import HierarchicalTransformer


@dataclass
class RumourDetectionTwitterModelOutput(ModelOutput):
    """
    Base class for outputs of Rumour Detection models

    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss, typically cross entropy. Loss function used is dependent on what is specified in RumourDetectionTwitterConfig.
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, num_classes)`):
            Raw logits for each class. num_classes = 4 by default.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class RumourDetectionTwitterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = RumourDetectionTwitterConfig
    base_model_prefix = "rdt"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RumourDetectionTwitterModel(RumourDetectionTwitterPreTrainedModel):
    """
    Class to create the Hierarhical Transformer with structure and post level attention used to evaluate Twitter threads.


    This method inherits from :obj:`RumourDetectionTwitterPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:class:`~RumourDetectionTwitterConfig`): Model configuration class with the default parameters required for the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Use the :obj:`.from_pretrained` method to load the model weights.

    Example::
            >>> # 1. From config / default parameters (untrained)
            >>> config = RumourDetectionTwitterConfig()
            >>> model = RumourDetectionTwitterModel(config)
            >>> # 2. From pretrained
            >>> config = RumourDetectionTwitterConfig.from_pretrained("https://sgnlp.blob.core.windows.net/models/rumour_detection_twitter/config.json")
            >>> model = RumourDetectionTwitterModel.from_pretrained("https://sgnlp.blob.core.windows.net/models/rumour_detection_twitter/pytorch_model.bin", config=config)
    """

    def __init__(self, config: RumourDetectionTwitterConfig):
        super().__init__(config)

        self.config = config
        self.wordEncoder = WordEncoder(self.config)
        self.positionEncoderWord = PositionEncoder(config, self.config.max_length)
        self.positionEncoderTime = PositionEncoder(config, self.config.size)
        self.hierarchicalTransformer = HierarchicalTransformer(self.config)
        if config.loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        token_ids: torch.Tensor,
        time_delay_ids: torch.Tensor,
        structure_ids: torch.Tensor,
        token_attention_mask=None,
        post_attention_mask=None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward method to compute model output given a Twitter thread.

        Args:
            token_ids (:obj:`torch.Tensor`): Token indices of all threads. Shape of (batch_size, max_posts, max_token_length)
            time_delay_ids (:obj:`torch.Tensor`): Time delay indices for each tweet in each thread. Note that this is not the actual time delay (in seconds/minutes/etc). This is the index of the binned time delay. Shape of (batch_size, max_posts)
            structure_ids (:obj:`torch.Tensor`): Contains the structure category index for each post with respect to every other post in the same thread. Shape of (batch_size, max_posts, max_posts)
            token_attention_mask (:obj:`torch.Tensor`): Tensor with elements of only 0 or 1. Indicates which tokens in each tweet should be considered (ie, which tokens are not padding). Shape of (batch_size, max_posts, max_token_length)
            post_attention_mask (:obj:`torch.Tensor`): Tensor with elements of only 0 or 1. Indicates which post in each thread should be considered (ie, which tokens are not padding). Shape of (batch_size, max_posts)
            labels (:obj:`Optional[torch.Tensor]`): Tensor with labels. Shape of (batch_size, 1). Defaults to None

        Returns:
            :obj:`torch.Tenosr`: raw prediction logits of shape (batch_size, num_classes). num_classes = 4 by default.
        """

        X = self.wordEncoder(token_ids)
        word_pos = self.prepare_word_pos(token_ids).to(X.device)
        word_pos = self.positionEncoderWord(word_pos)
        time_delay = self.positionEncoderTime(time_delay_ids)

        logits = self.hierarchicalTransformer(
            X,
            word_pos,
            time_delay,
            structure_ids,
            attention_mask_word=token_attention_mask,
            attention_mask_post=post_attention_mask,
        )

        if labels is not None:
            loss = self.loss(logits, labels)
        else:
            loss = None

        return RumourDetectionTwitterModelOutput(loss=loss, logits=logits)

    def load_pretrained_embedding(self, pretrained_embedding_path):
        """Load pretrained embedding matrix to the embedding layer

        Args:
            pretrained_embedding_path (str): path to a `.npy` file containing the pretrained embeddings. Note that the position of each word's embedding in the matrix has to correspond to the index of that word in the tokenizer's vocab.
        """

        self.wordEncoder.load_pretrained_embedding(pretrained_embedding_path)

    def prepare_word_pos(self, token_ids):
        """
        Generates the position indices for the tokens in each thread.

        Args:
            token_ids (:obj:`torch.Tensor`): Token indices of all threads. Shape of (batch_size, max_posts, max_token_length)


        Returns:
            :obj:`torch.Tenosr`: position indices of each token in each thread of each post. Shape should be equivalent to `token_ids` - (batch_size, max_posts, max_token_length).
        """

        batch_size, num_posts, num_words = token_ids.shape
        word_pos = np.repeat(
            np.expand_dims(
                np.repeat(
                    np.expand_dims(np.arange(num_words), axis=0), num_posts, axis=0
                ),
                axis=0,
            ),
            batch_size,
            axis=0,
        )
        word_pos = torch.from_numpy(word_pos)
        return word_pos
