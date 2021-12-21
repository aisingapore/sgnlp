from transformers import PreTrainedConfig


class SenticNetGCNConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a
    :class:`~sgnlp.models.senticnet_gcn.modeling.SenticNetGCNModel`.
    It is used to instantiate a SenticNetGCNModel network according to the specific arguments, defining the model architecture.

    Args:
        embed_dim (:obj:`int`, defaults to 300): Embedding dimension size.
        hidden_dim (:obj:`int`, defaults to 300): Size of hidden dimension.
        dropout (:obj:`float`, defaults to 0.3): Droput percentage.
        polarities_dim (:obj:`int`, defaults to 3): Size of output dimension representing available polarities (e.g. Positive, Negative, Neutral).
        device (:obj:`str`, defaults to 'cuda`): Type of torch device.

    Example:

        from sgnlp.models.senticnet_gcn import SenticNetGCNConfig

        # Initialize with default values
        config = SenticNetGCNConfig()
    """

    def __init__(
        self,
        embed_dim: int = 300,
        hidden_dim: int = 300,
        polarities_dim: int = 3,
        dropout: float = 0.3,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.polarities_dim = polarities_dim
        self.device = device


class SenticNetBertGCNConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~sgnlp.models.senticnet_gcn.modeling.SenticNetBertGCNModel`.
    It is used to instantiate a SenticNetBertGCNModel network according to the specific arguments, defining the model architecture.

    Args:
        bert_model (:obj:`str`, defaults to 'bert-base-uncased'): The Bert model type to initalized from transformers package.
        hidden_dim (:obj:`int`, defaults to 768): The embedding dimension size for the Bert model as well as GCN dimension.
        max_seq_len (:obj:`int`, defaults to 85): The max sequence length to pad and truncate.
        dropout (:obj:`float`, defaults to 0.3): Dropout percentage.
        polarities_dim (:ob:`int`, defaults to 3): Size of output dimension representing available polarities (e.g. Positive, Negative, Neutral).
        device (:obj:`str`, defaults to 'cuda'): Type of torch device.
    Example:

        from sgnlp.models.senticnet_gcn import SenticNetBertGCNConfig

        # Initialize with default values
        config = SenticNetBertGCNConfig()
    """

    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        hidden_dim: int = 768,
        max_seq_len: int = 85,
        polarities_dim: int = 3,
        dropout: float = 0.3,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bert_model = bert_model
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.polarities_dim = polarities_dim
        self.device = device
