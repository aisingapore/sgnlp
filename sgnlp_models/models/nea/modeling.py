from dataclasses import dataclass
from typing import Dict, Union, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .config import NEAConfig
from .modules.attention import Attention
from .modules.mean_over_time import MeanOverTime


@dataclass
class NEAModelOutput(ModelOutput):
    """
    Base class for outputs of NEA models

    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Regression loss. Loss function used is dependent on what is specified in NEAConfig
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, 1)`):
            Regression scores of between 0 and 1.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class NEAPreTrainedModel(PreTrainedModel):
    """
    The Neural Essay Assesssor (NEA) Pre-Trained Model used as base class for derived NEA Model.

    This model is the abstract super class for the NEA model which defines the config class types
    and weights initalization method. This class should not be used or instantiated directly,
    see NEAlModel class for usage.
    """

    config_class = NEAConfig
    base_model_prefix = "NEA"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight)
        elif (
            isinstance(module, nn.LSTM)
            or isinstance(module, nn.GRU)
            or isinstance(module, nn.RNN)
        ):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)


class NEARegPoolingModel(NEAPreTrainedModel):
    """
    Class to create the Neural Essay Assessor(NEA) Model used to evaluate essays.
    The model uses a Regression Pooling model type as described in the original research code.

    This method inherits from :obj:`NEAPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    .. note::
        This is the optimal model type as described in the NEA paper. Only this model class will work with the model
        weights stored on Azure Blob Storage.

    Args:
        config (:class:`~NEAConfig`): Model configuration class with all parameters required for
                                        the model. Initializing with a config file does not load
                                        the weights associated with the model, only the configuration.
                                        Use the :obj:`.from_pretrained` method to load the model weights.

    Example::
            # 1. From default
            config = NEAConfig()
            model = NEARegPoolingModel(config)
            # 2. From pretrained
            config = NEAConfig.from_pretrained("https://sgnlp.blob.core.windows.net/models/nea/config.json")
            model = \
            NEARegPoolingModel.from_pretrained("https://sgnlp.blob.core.windows.net/models/nea/pytorch_model.bin",
            config=config)
    """

    def __init__(self, config):
        super().__init__(config)
        self.embedding_layer = self._init_embedding_layer(config)
        self.conv_layer = self._init_conv_layer(config)
        self.dropout = self._init_dropout(config)
        self.rec_layer = self._init_rec_layer(config)
        self.agg_layer = self._init_agg_layer(config)
        self.linear_layer = self._init_linear_layer(config)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        self.loss_function = config.loss_function
        self.skip_init_bias = config.skip_init_bias

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> NEAModelOutput:
        """Forward method to compute model output given input.

        Args:
            input_ids (:obj:`torch.Tensor`): torch tensor of model input. Torch tensor
                should contain input_ids of sequences
            labels (:obj:`Optional[torch.Tensor]`): torch tensor of shape (batch_size). Defaults to None

        Returns:
            :obj:`NEAModelOutput`: model output of shape (batch_size, 1)
        """
        mask = self._generate_mask(input_ids)

        x = self.embedding_layer(input_ids)
        x = self.dropout(x)

        if self.conv_layer:
            x = self.conv(x)
        if self.rec_layer:
            x, _ = self.rec_layer(x)
        x = self.dropout(x)

        self.agg_layer.init_mask(mask)
        x = self.agg_layer(x)

        x = self.linear_layer(x)
        logits = self.sigmoid(x)

        loss = None
        if labels is not None:
            if self.loss_function == "mse":
                loss_fct = torch.nn.MSELoss()
            elif self.loss_function == "mae":
                loss_fct = torch.nn.L1Loss()
            loss = loss_fct(logits.view(-1), labels)

        return NEAModelOutput(loss=loss, logits=logits)

    def initialise_linear_bias(self, train_y: torch.Tensor) -> None:
        """Initialise bias term of linear layer according to implementation in NEA paper

        Args:
            train_y (:obj:`torch.Tensor`): tensor of train y
        """
        if not self.skip_init_bias:
            input_checks = (train_y < 0) | (train_y > 1)
            if sum(input_checks) > 0:
                raise ValueError("Train y needs to be between 0 and 1")
            initial_mean_value = train_y.mean()
            initial_bias = torch.log(initial_mean_value) - torch.log(
                1 - initial_mean_value
            )
            self.linear_layer.bias.data.fill_(initial_bias)

    def load_pretrained_embedding(self, emb_matrix: torch.Tensor) -> None:
        """Load pretrained embedding matrix in the embedding layer

        Args:
            emb_matrix (:obj:`torch.Tensor`): tensor of embedding matrix
        """
        if self.embedding_layer.weight.shape != emb_matrix.shape:
            raise ValueError(
                "Dimensions of emb_matrix do not match embedding layer's dimensions"
            )
        self.embedding_layer.weight = nn.parameter.Parameter(emb_matrix)

    def _generate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the number of non padding tokens for each instance. Input
        is the sequence of integers fed into the model

        Args:
            input_ids (torch.Tensor): shape: (batch_size * seq_len)

        Returns:
            torch.Tensor: shape: (batch_size * 1)
        """
        mask = (input_ids != 0).sum(axis=1)
        mask = torch.unsqueeze(mask, 1)

        return mask

    def _init_embedding_layer(self, config: Dict) -> nn.Embedding:
        """Initialise embedding layer with config

        Args:
            config (Dict): config from config file

        Returns:
            nn.Embedding: initialised embedding layer
        """
        return nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )

    def _init_conv_layer(self, config: Dict) -> Union[None, nn.Conv1d]:
        """Initialise convolutional layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.Conv1d]: initialised convolution layer. Returns None if
                cnn_output_dim == 0.
        """
        if config.cnn_output_dim > 0:
            layer = nn.Conv1d(
                in_channels=config.cnn_input_dim,
                out_channels=config.cnn_output_dim,
                kernel_size=config.cnn_kernel_size,
                padding=config.cnn_padding,
            )
        else:
            layer = None
        return layer

    def _init_dropout(self, config: Dict) -> nn.Dropout:
        """Initialise dropout layer

        Args:
            config (Dict): config from config file

        Returns:
            nn.Dropout: initialised dropout layer
        """
        return nn.Dropout(p=config.dropout)

    def _init_rec_layer(self, config: Dict) -> Union[None, nn.LSTM, nn.GRU, nn.RNN]:
        """Initialise recurrent layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.LSTM, nn.GRU, nn.RNN]: initialised recurrent layer.
                Returns None if rec_output_dim == 0.
        """
        if config.rec_output_dim > 0:
            if config.rec_layer_type == "lstm":
                rec_layer = nn.LSTM
            elif config.rec_layer_type == "gru":
                rec_layer = nn.GRU
            elif config.rec_layer_type == "rnn":
                rec_layer = nn.RNN

            layer = rec_layer(
                input_size=config.rec_input_dim,
                hidden_size=config.rec_output_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            layer = None
        return layer

    def _init_agg_layer(self, config: Dict) -> Union[MeanOverTime, Attention]:
        """Initialise aggregation layer with config. Aggregation layer is either
        a mean over time layer or attention layer

        Args:
            config (Dict): config from config file

        Returns:
            Union[MeanOverTime, Attention]: initialised aggregation layer
        """
        if config.aggregation == "mot":
            layer = MeanOverTime()
        if config.aggregation in ["attsum", "attmean"]:
            layer = Attention(op=config.aggregation)
        return layer

    def _init_linear_layer(self, config: Dict) -> nn.Linear:
        """Initialise linear layer

        Args:
            config (Dict): config file from config

        Returns:
            nn.Linear: initialised linear layer
        """
        return nn.Linear(
            in_features=config.linear_input_dim, out_features=config.linear_output_dim
        )


class NEARegModel(NEAPreTrainedModel):
    """
    Class to create the Neural Essay Assessor(NEA) Model used to evaluate essays.
    The model uses a Regression model type as described in the original research code.

    This method inherits from :obj:`NEAPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    .. note::
        This model class will not work with the model weights stored on Azure Blob Storage.
        Refer to :obj:`NEARegPoolingModel` to use the pretrained weights on Azure Blob Storage.

    Args:
        config (:class:`~NEAConfig`): Model configuration class with all parameters required for
                                        the model. Initializing with a config file does not load
                                        the weights associated with the model, only the configuration.
                                        Use the :obj:`.from_pretrained` method to load the model weights.

    Example::
            config = NEAConfig()
            model = NEARegModel(config)
    """

    def __init__(self, config):
        super().__init__(config)
        self.embedding_layer = self._init_embedding_layer(config)
        self.conv_layer = self._init_conv_layer(config)
        self.dropout = self._init_dropout(config)
        self.rec_layer = self._init_rec_layer(config)
        self.linear_layer = self._init_linear_layer(config)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        self.loss_function = config.loss_function
        self.skip_init_bias = config.skip_init_bias

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> NEAModelOutput:
        """Forward method to compute model output given input.

        Args:
            input_ids (:obj:`torch.Tensor`): torch tensor of model input. Torch tensor
                should contain input_ids of sequences
            labels (:obj:`Optional[torch.Tensor]`): torch tensor of shape (batch_size). Defaults to None

        Returns:
            :obj:`NEAModelOutput`: model output of shape (batch_size, 1)
        """
        mask = self._generate_mask(input_ids)

        x = self.embedding_layer(input_ids)
        x = self.dropout(x)

        if self.conv_layer:
            x = self.conv(x)
        if self.rec_layer:
            x, _ = self.rec_layer(x)
        x = self._extract_last_hidden_state(x, mask)
        x = self.dropout(x)
        x = self.linear_layer(x)
        logits = self.sigmoid(x)

        loss = None
        if labels is not None:
            if self.loss_function == "mse":
                loss_fct = torch.nn.MSELoss()
            elif self.args.loss_function == "mae":
                loss_fct = torch.nn.L1Loss()
            loss = loss_fct(logits.view(-1), labels)

        return NEAModelOutput(loss=loss, logits=logits)

    def initialise_linear_bias(self, train_y: torch.Tensor) -> None:
        """Initialise bias term of linear layer according to implementation in NEA paper

        Args:
            train_y (:obj:`torch.Tensor`): tensor of train y
        """
        if not self.skip_init_bias:
            input_checks = (train_y < 0) | (train_y > 1)
            if sum(input_checks) > 0:
                raise ValueError("Train y needs to be between 0 and 1")
            initial_mean_value = train_y.mean()
            initial_bias = torch.log(initial_mean_value) - torch.log(
                1 - initial_mean_value
            )
            self.linear_layer.bias.data.fill_(initial_bias)

    def load_pretrained_embedding(self, emb_matrix: torch.Tensor) -> None:
        """Load pretrained embedding matrix in the embedding layer

        Args:
            emb_matrix (:obj:`torch.Tensor`): tensor of embedding matrix
        """
        if self.embedding_layer.weight.shape != emb_matrix.shape:
            raise ValueError(
                "Dimensions of emb_matrix do not match embedding layer's dimensions"
            )
        self.embedding_layer.weight = nn.parameter.Parameter(emb_matrix)

    def _generate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the number of non padding tokens for each instance. Input
        is the sequence of integers fed into the model

        Args:
            input_ids (torch.Tensor): shape: (batch_size * seq_len)

        Returns:
            torch.Tensor: shape: (batch_size * 1)
        """
        mask = (input_ids != 0).sum(axis=1)
        mask = torch.unsqueeze(mask, 1)

        return mask

    def _extract_last_hidden_state(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract the last hidden state of each sequence returned from recurrent
        layer, excluding padding tokens.

        Args:
            x (torch.Tensor): output of recurrent layer. shape:(batch_size * seq_len * reccurent_dim)
            mask (torch.Tensor): length of non padding tokens. shape:(batch_size * 1)

        Returns:
            torch.Tensor: last hidden state of each instance. shape : (batch_size * recurrent_dim)
        """
        array = []
        for i in range(len(x)):
            array.append(x[i, mask[i] - 1, :])
        x = torch.cat(array)
        return x

    def _init_embedding_layer(self, config: Dict) -> nn.Embedding:
        """Initialise embedding layer with config

        Args:
            config (Dict): config from config file

        Returns:
            nn.Embedding: initialised embedding layer
        """
        return nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )

    def _init_conv_layer(self, config: Dict) -> Union[None, nn.Conv1d]:
        """Initialise convolutional layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.Conv1d]: initialised convolution layer. Returns None if
                cnn_output_dim == 0.
        """
        if config.cnn_output_dim > 0:
            layer = nn.Conv1d(
                in_channels=config.cnn_input_dim,
                out_channels=config.cnn_output_dim,
                kernel_size=config.cnn_kernel_size,
                padding=config.cnn_padding,
            )
        else:
            layer = None
        return layer

    def _init_dropout(self, config: Dict) -> nn.Dropout:
        """Initialise dropout layer

        Args:
            config (Dict): config from config file

        Returns:
            nn.Dropout: initialised dropout layer
        """
        return nn.Dropout(p=config.dropout)

    def _init_rec_layer(self, config: Dict) -> Union[None, nn.LSTM, nn.GRU, nn.RNN]:
        """Initialise recurrent layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.LSTM, nn.GRU, nn.RNN]: initialised recurrent layer.
                Returns None if rec_output_dim == 0.
        """
        if config.rec_output_dim > 0:
            if config.rec_layer_type == "lstm":
                rec_layer = nn.LSTM
            elif config.rec_layer_type == "gru":
                rec_layer = nn.GRU
            elif config.rec_layer_type == "rnn":
                rec_layer = nn.RNN

            layer = rec_layer(
                input_size=config.rec_input_dim,
                hidden_size=config.rec_output_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            layer = None
        return layer

    def _init_linear_layer(self, config: Dict) -> nn.Linear:
        """Initialise linear layer

        Args:
            config (Dict): config file from config

        Returns:
            nn.Linear: initialised linear layer
        """
        return nn.Linear(
            in_features=config.linear_input_dim, out_features=config.linear_output_dim
        )


class NEABiRegModel(NEAPreTrainedModel):
    """
    Class to create the Neural Essay Assessor(NEA) Model used to evaluate essays.
    The model uses a Bidirectional Regression model type as described in the original research code.

    This method inherits from :obj:`NEAPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    .. note::
        This model class will not work with the model weights stored on Azure Blob Storage.
        Refer to :obj:`NEARegPoolingModel` to use the pretrained weights on Azure Blob Storage.

    Args:
        config (:class:`~NEAConfig`): Model configuration class with all parameters required for
                                        the model. Initializing with a config file does not load
                                        the weights associated with the model, only the configuration.
                                        Use the :obj:`.from_pretrained` method to load the model weights.

    Example::
            config = NEAConfig()
            model = NEABiRegModel(config)
    """

    def __init__(self, config):
        super().__init__(config)
        self._validate_linear_input_dim(config)

        self.embedding_layer = self._init_embedding_layer(config)
        self.conv_layer = self._init_conv_layer(config)
        self.dropout = self._init_dropout(config)
        self.forward_rec_layer = self._init_rec_layer(config)
        self.backward_rec_layer = self._init_rec_layer(config)
        self.linear_layer = self._init_linear_layer(config)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        self.loss_function = config.loss_function
        self.skip_init_bias = config.skip_init_bias

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> NEAModelOutput:
        """Forward method to compute model output given input.

        Args:
            input_ids (:obj:`torch.Tensor`): torch tensor of model input. Torch tensor
                should contain input_ids of sequences
            labels (:obj:`Optional[torch.Tensor]`): torch tensor of shape (batch_size). Defaults to None

        Returns:
            :obj:`NEAModelOutput`: model output of shape (batch_size, 1)
        """
        mask = self._generate_mask(input_ids)

        x = self.embedding_layer(input_ids)
        x = self.dropout(x)

        if self.conv_layer:
            x = self.conv(x)

        if self.forward_rec_layer:
            x_forward, _ = self.forward_rec_layer(x)
            x_forward = self._extract_last_hidden_state(x_forward, mask, forward=True)
            x_forward = self.dropout(x_forward)
        if self.backward_rec_layer:
            x_backward, _ = self.backward_rec_layer(torch.flip(x, [1]))
            x_backward = self._extract_last_hidden_state(
                x_backward, mask, forward=False
            )
            x_backward = self.dropout(x_backward)
        x = torch.cat((x_forward, x_backward), 1)

        x = self.linear_layer(x)
        logits = self.sigmoid(x)

        loss = None
        if labels is not None:
            if self.loss_function == "mse":
                loss_fct = torch.nn.MSELoss()
            elif self.args.loss_function == "mae":
                loss_fct = torch.nn.L1Loss()
            loss = loss_fct(logits.view(-1), labels)

        return NEAModelOutput(loss=loss, logits=logits)

    def initialise_linear_bias(self, train_y: torch.Tensor) -> None:
        """Initialise bias term of linear layer according to implementation in NEA paper

        Args:
            train_y (:obj:`torch.Tensor`): tensor of train y
        """
        if not self.skip_init_bias:
            input_checks = (train_y < 0) | (train_y > 1)
            if sum(input_checks) > 0:
                raise ValueError("Train y needs to be between 0 and 1")
            initial_mean_value = train_y.mean()
            initial_bias = torch.log(initial_mean_value) - torch.log(
                1 - initial_mean_value
            )
            self.linear_layer.bias.data.fill_(initial_bias)

    def load_pretrained_embedding(self, emb_matrix: torch.Tensor) -> None:
        """Load pretrained embedding matrix in the embedding layer

        Args:
            emb_matrix (:obj:`torch.Tensor`): tensor of embedding matrix
        """
        if self.embedding_layer.weight.shape != emb_matrix.shape:
            raise ValueError(
                "Dimensions of emb_matrix do not match embedding layer's dimensions"
            )
        self.embedding_layer.weight = nn.parameter.Parameter(emb_matrix)

    def _validate_linear_input_dim(self, config: Dict) -> None:
        """Validate that linear_input_dim is 2 times the value of rec_output_dim.
        The output dimension of the recurrent layer will be 2 times of rec_output_dim
        due to the bidirectional property. Thus, linear_input_dim needs to be 2
        times rec_output_dim

        Args:
            config (Dict): config from NEAConfig

        Raises:
            ValueError: linear_input_dim should be 2 times the value of rec_output_dim
                        due to the bidirectional property.
        """
        if config.linear_input_dim != 2 * config.rec_output_dim:
            raise ValueError(
                "linear_input_dim should be 2 times the value of rec_output_dim due to the bidirectional property of \
                    the recurrent layer."
            )

    def _generate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the number of non padding tokens for each instance. Input
        is the sequence of integers fed into the model

        Args:
            input_ids (torch.Tensor): shape: (batch_size * seq_len)

        Returns:
            torch.Tensor: shape: (batch_size * 1)
        """
        mask = (input_ids != 0).sum(axis=1)
        mask = torch.unsqueeze(mask, 1)

        return mask

    def _extract_last_hidden_state(
        self, x: torch.Tensor, mask: torch.Tensor, forward: bool
    ) -> torch.Tensor:
        """Extract the last hidden state of each sequence returned from recurrent
        layer, excluding padding tokens

        Args:
            x (torch.Tensor): output of recurrent layer. shape:(batch_size * seq_len * reccurent_dim)
            mask (torch.Tensor): length of non padding tokens. shape:(batch_size * 1)

        Returns:
            torch.Tensor: last hidden state of each instance. shape : (batch_size * recurrent_dim)
        """
        array = []
        if forward:
            for i in range(len(x)):
                array.append(x[i, mask[i] - 1, :])
            x = torch.cat(array)
        else:
            for i in range(len(x)):
                array.append(x[i, [-1], :])
            x = torch.cat(array)
        return x

    def _init_embedding_layer(self, config: Dict) -> nn.Embedding:
        """Initialise embedding layer with config

        Args:
            config (Dict): config from config file

        Returns:
            nn.Embedding: initialised embedding layer
        """
        return nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )

    def _init_conv_layer(self, config: Dict) -> Union[None, nn.Conv1d]:
        """Initialise convolutional layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.Conv1d]: initialised convolution layer. Returns None if
                cnn_output_dim == 0.
        """
        if config.cnn_output_dim > 0:
            layer = nn.Conv1d(
                in_channels=config.cnn_input_dim,
                out_channels=config.cnn_output_dim,
                kernel_size=config.cnn_kernel_size,
                padding=config.cnn_padding,
            )
        else:
            layer = None
        return layer

    def _init_dropout(self, config: Dict) -> nn.Dropout:
        """Initialise dropout layer

        Args:
            config (Dict): config from config file

        Returns:
            nn.Dropout: initialised dropout layer
        """
        return nn.Dropout(p=config.dropout)

    def _init_rec_layer(self, config: Dict) -> Union[None, nn.LSTM, nn.GRU, nn.RNN]:
        """Initialise recurrent layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.LSTM, nn.GRU, nn.RNN]: initialised recurrent layer.
                Returns None if rec_output_dim == 0.
        """
        if config.rec_output_dim > 0:
            if config.rec_layer_type == "lstm":
                rec_layer = nn.LSTM
            elif config.rec_layer_type == "gru":
                rec_layer = nn.GRU
            elif config.rec_layer_type == "rnn":
                rec_layer = nn.RNN

            layer = rec_layer(
                input_size=config.rec_input_dim,
                hidden_size=config.rec_output_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            layer = None
        return layer

    def _init_linear_layer(self, config: Dict) -> nn.Linear:
        """Initialise linear layer

        Args:
            config (Dict): config file from config

        Returns:
            nn.Linear: initialised linear layer
        """
        return nn.Linear(
            in_features=config.linear_input_dim, out_features=config.linear_output_dim
        )


class NEABiRegPoolingModel(NEAPreTrainedModel):
    """
    Class to create the Neural Essay Assessor(NEA) Model used to evaluate essays.
    The model uses a Bidirectional Regression Pooling model type as described in the original research code.

    This method inherits from :obj:`NEAPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    .. note::
        This model class will not work with the model weights stored on Azure Blob Storage.
        Refer to :obj:`NEARegPoolingModel` to use the pretrained weights on Azure Blob Storage.

    Args:
        config (:class:`~NEAConfig`): Model configuration class with all parameters required for
                                        the model. Initializing with a config file does not load
                                        the weights associated with the model, only the configuration.
                                        Use the :obj:`.from_pretrained` method to load the model weights.

    Example::
            config = NEAConfig()
            model = NEABiRegPoolingModel(config)
    """

    def __init__(self, config):
        super().__init__(config)
        self._validate_linear_input_dim(config)

        self.embedding_layer = self._init_embedding_layer(config)
        self.conv_layer = self._init_conv_layer(config)
        self.dropout = self._init_dropout(config)
        self.forward_rec_layer = self._init_rec_layer(config)
        self.backward_rec_layer = self._init_rec_layer(config)
        self.agg_layer = self._init_agg_layer(config)
        self.linear_layer = self._init_linear_layer(config)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        self.loss_function = config.loss_function
        self.skip_init_bias = config.skip_init_bias

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> NEAModelOutput:
        """Forward method to compute model output given input.

        Args:
            input_ids (:obj:`torch.Tensor`): torch tensor of model input. Torch tensor
                should contain input_ids of sequences
            labels (:obj:`Optional[torch.Tensor]`): torch tensor of shape (batch_size). Defaults to None

        Returns:
            :obj:`NEAModelOutput`: model output of shape (batch_size, 1)
        """
        mask = self._generate_mask(input_ids)

        x = self.embedding_layer(input_ids)
        x = self.dropout(x)

        if self.conv_layer:
            x = self.conv(x)

        self.agg_layer.init_mask(mask)
        if self.forward_rec_layer:
            x_forward, _ = self.forward_rec_layer(x)
            x_forward = self.dropout(x_forward)
            x_forward_mean = self.agg_layer(x_forward)
        if self.backward_rec_layer:
            x_backward, _ = self.backward_rec_layer(torch.flip(x, [1]))
            x_backward = self.dropout(x_backward)
            x_backward_mean = self.agg_layer(x_backward)
        x = torch.cat((x_forward_mean, x_backward_mean), 1)

        x = self.linear_layer(x)
        logits = self.sigmoid(x)

        loss = None
        if labels is not None:
            if self.loss_function == "mse":
                loss_fct = torch.nn.MSELoss()
            elif self.args.loss_function == "mae":
                loss_fct = torch.nn.L1Loss()
            loss = loss_fct(logits.view(-1), labels)

        return NEAModelOutput(loss=loss, logits=logits)

    def initialise_linear_bias(self, train_y: torch.Tensor) -> None:
        """Initialise bias term of linear layer according to implementation in NEA paper

        Args:
            train_y (:obj:`torch.Tensor`): tensor of train y
        """
        if not self.skip_init_bias:
            input_checks = (train_y < 0) | (train_y > 1)
            if sum(input_checks) > 0:
                raise ValueError("Train y needs to be between 0 and 1")
            initial_mean_value = train_y.mean()
            initial_bias = torch.log(initial_mean_value) - torch.log(
                1 - initial_mean_value
            )
            self.linear_layer.bias.data.fill_(initial_bias)

    def load_pretrained_embedding(self, emb_matrix: torch.Tensor) -> None:
        """Load pretrained embedding matrix in the embedding layer

        Args:
            emb_matrix (:obj:`torch.Tensor`): tensor of embedding matrix
        """
        if self.embedding_layer.weight.shape != emb_matrix.shape:
            raise ValueError(
                "Dimensions of emb_matrix do not match embedding layer's dimensions"
            )
        self.embedding_layer.weight = nn.parameter.Parameter(emb_matrix)

    def _validate_linear_input_dim(self, config: Dict) -> None:
        """Validate that linear_input_dim is 2 times the value of rec_output_dim.
        The output dimension of the recurrent layer will be 2 times of rec_output_dim
        due to the bidirectional property. Thus, linear_input_dim needs to be 2
        times rec_output_dim

        Args:
            config (Dict): config from NEAConfig

        Raises:
            ValueError: linear_input_dim should be 2 times the value of rec_output_dim
                        due to the bidirectional property.
        """
        if config.linear_input_dim != 2 * config.rec_output_dim:
            raise ValueError(
                "linear_input_dim should be 2 times the value of rec_output_dim due to the bidirectional property of \
                    the recurrent layer."
            )

    def _generate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the number of non padding tokens for each instance. Input
        is the sequence of integers fed into the model

        Args:
            input_ids (torch.Tensor): shape: (batch_size * seq_len)

        Returns:
            torch.Tensor: shape: (batch_size * 1)
        """
        mask = (input_ids != 0).sum(axis=1)
        mask = torch.unsqueeze(mask, 1)

        return mask

    def _extract_last_hidden_state(
        self, x: torch.Tensor, mask: torch.Tensor, forward: bool
    ) -> torch.Tensor:
        """Extract the last hidden state of each sequence returned from recurrent
        layer

        Args:
            x (torch.Tensor): output of recurrent layer. shape:(batch_size * seq_len * reccurent_dim)
            mask (torch.Tensor): length of non padding tokens. shape:(batch_size * 1)

        Returns:
            torch.Tensor: last hidden state of each instance. shape : (batch_size * recurrent_dim)
        """
        array = []
        if forward:
            for i in range(len(x)):
                array.append(x[i, mask[i] - 1, :])
            x = torch.cat(array)
        else:
            for i in range(len(x)):
                array.append(x[i, [-1], :])
            x = torch.cat(array)
        return x

    def _init_embedding_layer(self, config: Dict) -> nn.Embedding:
        """Initialise embedding layer with config

        Args:
            config (Dict): config from config file

        Returns:
            nn.Embedding: initialised embedding layer
        """
        return nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )

    def _init_conv_layer(self, config: Dict) -> Union[None, nn.Conv1d]:
        """Initialise convolutional layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.Conv1d]: initialised convolution layer. Returns None if
                cnn_output_dim == 0.
        """
        if config.cnn_output_dim > 0:
            layer = nn.Conv1d(
                in_channels=config.cnn_input_dim,
                out_channels=config.cnn_output_dim,
                kernel_size=config.cnn_kernel_size,
                padding=config.cnn_padding,
            )
        else:
            layer = None
        return layer

    def _init_dropout(self, config: Dict) -> nn.Dropout:
        """Initialise dropout layer

        Args:
            config (Dict): config from config file

        Returns:
            nn.Dropout: initialised dropout layer
        """
        return nn.Dropout(p=config.dropout)

    def _init_rec_layer(self, config: Dict) -> Union[None, nn.LSTM, nn.GRU, nn.RNN]:
        """Initialise recurrent layer with config.

        Args:
            config (Dict): config from config file

        Returns:
            Union[None, nn.LSTM, nn.GRU, nn.RNN]: initialised recurrent layer.
                Returns None if rec_output_dim == 0.
        """
        if config.rec_output_dim > 0:
            if config.rec_layer_type == "lstm":
                rec_layer = nn.LSTM
            elif config.rec_layer_type == "gru":
                rec_layer = nn.GRU
            elif config.rec_layer_type == "rnn":
                rec_layer = nn.RNN

            layer = rec_layer(
                input_size=config.rec_input_dim,
                hidden_size=config.rec_output_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            layer = None
        return layer

    def _init_agg_layer(self, config: Dict) -> MeanOverTime:
        """Initialise aggregation layer with config. Aggregation layer is
        a mean over time layer

        Args:
            config (Dict): config from config file

        Returns:
            MeanOverTime: initialised aggregation layer
        """
        if config.aggregation == "mot":
            layer = MeanOverTime()
        return layer

    def _init_linear_layer(self, config: Dict) -> nn.Linear:
        """Initialise linear layer

        Args:
            config (Dict): config file from config

        Returns:
            nn.Linear: initialised linear layer
        """
        return nn.Linear(
            in_features=config.linear_input_dim, out_features=config.linear_output_dim
        )
