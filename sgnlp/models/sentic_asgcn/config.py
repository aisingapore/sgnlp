import torch
from transformers import PreTrainedConfig


class SenticASGCNConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a
    :class:`~sgnlp.models.sentic_asgcn.modeling.SenticASGCNModel`.
    It is used to instantiate a SenticASGCN network according to the specific arguments, defining the mdoel architecture.

    Args:
        embed_dim (:obj:`int`, defaults to 300): Embedding dimension size.
        hidden_dim (:obj:`int`, defaults to 300): Size of hidden dimension.
        polarities_dim (:obj:`int`, defaults to 3): Size of output dimension represeting available polarities (e.g. Positive, Negative, Neutral).
        device (:obj:`torch.device`, defaults to torch.device('cuda`)): Type of torch device.

    Example:

        from sgnlp.models.sentic_asgcn import SenticASGCNConfig

        # Initialize with default values
        config = SenticASGCNConfig()
    """

    def __init__(
        self,
        embed_dim=300,
        hidden_dim=300,
        polarities_dim=3,
        device=torch.device("cuda"),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.polarities_dim = polarities_dim
        self.device = device
