from transformers import PretrainedConfig


class NEAConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of  :class:`~NEARegModel`, :class:`~NEARegPoolingModel`,
    :class:`~NEABiRegModel`,  :class:`~NEABiRegPoolingModel`. It is used to instantiate a Neural Essay Assessor (NEA)
    model according to the specified arguments, defining the model architecture.

    Args:
        vocab_size (:obj:`int`, optional): size of dictionary of emebdding layer. Defaults to 4000.
        embedding_dim (:obj:`int`, optional): size of each embedding vector. Defaults to 50.
        dropout (:obj:`float`, optional): dropout ratio. Defaults to 0.5.
        cnn_input_dim (:obj:`int`, optional): input dim to 1d conv layer. Defaults to 0.
        cnn_output_dim (:obj:`int`, optional): output dim of 1d conv layer. Defaults to 0.
        cnn_kernel_size (:obj:`int`, optional): size of convolutional kernel. Defaults to 0.
        cnn_padding (:obj:`int`, optional): padding added to both sides of the input to 1d conv layer. Defaults to 0.
        rec_layer_type (:obj:`str`, optional): type of recurrent layer: rnn/gru/lstm. Defaults to "lstm".
        rec_input_dim (:obj:`int`, optional): input dimension of recurrent layer. Defaults to 50.
        rec_output_dim (:obj:`int`, optional): output dimension of recurrent layer. Defaults to 300.
        aggregation (:obj:`str`, optional): aggregation type between recurrent layer and linear layer:
            mot/attsum/attmean. Defaults to "mot".
        linear_input_dim (:obj:`int`, optional): input dimension of linear layer. Defaults to 300.
        linear_output_dim (:obj:`int`, optional): output dimension of linear layer. Defaults to 1.
        skip_init_bias (:obj:`bool`, optional): option to skip initialiing of bias term in linear layer.
            Defaults to False.
        loss_function (:obj:`str`, optional): Loss function used in the forward method when labels are provided:
            mse/mae. Defaults to "mse".
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        embedding_dim: int = 50,
        dropout: float = 0.5,
        cnn_input_dim: int = 0,
        cnn_output_dim: int = 0,
        cnn_kernel_size: int = 0,
        cnn_padding: int = 0,
        rec_layer_type: str = "lstm",
        rec_input_dim: int = 50,
        rec_output_dim: int = 300,
        aggregation: str = "mot",
        linear_input_dim: int = 300,
        linear_output_dim: int = 1,
        skip_init_bias: bool = False,
        loss_function: str = "mse",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.cnn_input_dim = cnn_input_dim
        self.cnn_output_dim = cnn_output_dim
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_padding = cnn_padding
        self.rec_layer_type = rec_layer_type
        self.rec_input_dim = rec_input_dim
        self.rec_output_dim = rec_output_dim
        self.aggregation = aggregation
        self.linear_input_dim = linear_input_dim
        self.linear_output_dim = linear_output_dim
        self.skip_init_bias = skip_init_bias
        self.loss_function = loss_function

        self._validate_arguments()

    def _validate_arguments(self) -> None:
        """Validate the arguments"""

        if self.rec_layer_type not in ["rnn", "gru", "lstm"]:
            raise ValueError("invalid rec_layer_type, it should be rnn/gru/lstm")

        if self.aggregation not in ["mot", "attsum", "attmean"]:
            raise ValueError("invalid aggregation, it should be mot/attsum/attmean")

        if self.loss_function not in ["mse", "mae"]:
            raise ValueError("invalid loss_function, it should be mse/mae")

        if self.skip_init_bias not in [True, False]:
            raise ValueError("invalid skip_init_bias, it should be True/False")

        # Validate dim between layers: Has cnn layer
        if self.cnn_output_dim > 0:
            # a. Check input from embedding layer
            if self.embedding_dim != self.cnn_input_dim:
                raise ValueError("embedding_dim do not match cnn_input_dim")
            # b. Check output: Has rec layer
            if self.rec_output_dim > 0 and self.cnn_output_dim != self.rec_input_dim:
                raise ValueError("cnn_output_dim do not match rec_output_dim")
            # c. Check output: no rec layer
            if (
                self.rec_output_dim == 0
                and self.cnn_output_dim != self.linear_input_dim
            ):
                raise ValueError("cnn_output_dim do not match linear_input_dim")

        # Validate dim between layers: Has rec layer
        if self.rec_output_dim > 0:
            # a. Check input: No CNN layer
            if self.cnn_output_dim == 0 and self.embedding_dim != self.rec_input_dim:
                raise ValueError("embedding_dim do not match rec_input_dim")
            # b. Check input: Has CNN layer
            if self.cnn_output_dim > 0 and self.cnn_output_dim != self.rec_input_dim:
                raise ValueError("cnn_output_dim do not match rec_input_dim")
            # c. Check output to linear layer
            if (
                self.rec_output_dim != self.linear_input_dim
                and 2 * self.rec_output_dim != self.linear_input_dim
            ):
                raise ValueError(
                    "linear_input_dim should be equals to rec_output_dim or 2 times rec_output_dim"
                )
