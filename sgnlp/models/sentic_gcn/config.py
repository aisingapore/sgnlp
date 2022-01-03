from transformers import PreTrainedConfig, BertConfig


class SenticGCNConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a
    :class:`~sgnlp.models.sentic_gcn.modeling.SenticGCNModel`.
    It is used to instantiate a SenticGCNModel network according to the specific arguments, defining the model architecture.

    Args:
        embed_dim (:obj:`int`, defaults to 300): Embedding dimension size.
        hidden_dim (:obj:`int`, defaults to 300): Size of hidden dimension.
        dropout (:obj:`float`, defaults to 0.3): Droput percentage.
        polarities_dim (:obj:`int`, defaults to 3): Size of output dimension representing available polarities (e.g. Positive, Negative, Neutral).
        device (:obj:`str`, defaults to 'cuda`): Type of torch device.
        loss_function (:obj:`str`, defaults to 'cross_entropy'): Loss function for training/eval.

    Example:

        from sgnlp.models.sentic_gcn import SenticGCNConfig

        # Initialize with default values
        config = SenticGCNConfig()
    """

    def __init__(
        self,
        embed_dim: int = 300,
        hidden_dim: int = 300,
        polarities_dim: int = 3,
        dropout: float = 0.3,
        device: str = "cuda",
        loss_function: str = "cross_entropy",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.polarities_dim = polarities_dim
        self.device = device
        self.loss_function = loss_function


class SenticGCNBertConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~sgnlp.models.sentic_gcn.modeling.SenticBertGCNModel`.
    It is used to instantiate a SenticBertGCNModel network according to the specific arguments, defining the model architecture.

    Args:
        hidden_dim (:obj:`int`, defaults to 768): The embedding dimension size for the Bert model as well as GCN dimension.
        max_seq_len (:obj:`int`, defaults to 85): The max sequence length to pad and truncate.
        dropout (:obj:`float`, defaults to 0.3): Dropout percentage.
        polarities_dim (:ob:`int`, defaults to 3): Size of output dimension representing available polarities (e.g. Positive, Negative, Neutral).
        device (:obj:`str`, defaults to 'cuda'): Type of torch device.
        loss_function (:obj:`str`, defaults to 'cross_entropy'): Loss function for training/eval.
    Example:

        from sgnlp.models.sentic_gcn import SenticGCNBertConfig

        # Initialize with default values
        config = SenticGCNBertConfig()
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        max_seq_len: int = 85,
        polarities_dim: int = 3,
        dropout: float = 0.3,
        device: str = "cuda",
        loss_function: str = "cross_entropy",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.polarities_dim = polarities_dim
        self.device = device
        self.loss_function = loss_function


class SenticGCNEmbeddingConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~SenticGCNEmbeddingModel`.
    It is used to instantiate a SenticGCN Embedding model according to the specified arguments, defining the model architecture.

    Args:
        PreTrainedConfig (:obj:`PretrainedConfig`): transformer :obj:`PreTrainedConfig` base class
    """

    def __init__(self, vocab_size: int = 17662, embed_dim: int = 300, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim


class SenticGCNBertEmbeddingConfig(BertConfig):
    """
    This is the configuration class to store the configuration of a :class:`~SenticGCNBertEmbeddingModel`.
    It is used to instantiate a SenticGCN Bert Embedding model according to the specified arguments, defining the model architecture.

    Args:
        BertConfig (:obj:`BertConfig`): transformer :obj:`BertConfig` base class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
