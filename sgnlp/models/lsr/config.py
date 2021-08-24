from typing import Tuple
from transformers import PretrainedConfig


class LsrConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~sgnlp.models.lsr.modeling.LsrModel`.
    It is used to instantiate a relation extraction model using latent structure refinement (LSR)
    according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        finetune_emb (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to finetune word embedding.
        word_embedding_shape (:obj:`Tuple[int, int]`, `optional`, defaults to (194784, 100)):
            Dimensionality of word embedding.
        ner_dim (:obj:`int`, `optional`, defaults to 20):
            Dimensionality of NER embedding.
        coref_dim (:obj:`int`, `optional`, defaults to 20):
            Dimensionality of coreference embedding.
        hidden_dim (:obj:`int`, `optional`, defaults to 120):
            Dimensionality of hidden states.
        distance_size (:obj:`int`, `optional`, defaults to 20):
            Dimensionality of distance embedding.
        num_relations (:obj:`int`, `optional`, defaults to 97):
            Number of classes for relations.
        dropout_rate (:obj:`float`, `optional`, defaults to 0.3):
            Dropout rate for encoding layer.
        dropout_emb (:obj:`float`, `optional`, defaults to 0.2):
            Dropout rate for embedding layer.
        dropout_gcn (:obj:`float`, `optional`, defaults to 0.4):
            Dropout rate for graph convolution network
        use_struct_att (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use struct attention.
        use_reasoning_block (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to use reasoning block.
        reasoner_layer_sizes (:obj:`Tuple[int, int]`, `optional`, defaults to (3, 4)):
            Number of layers in reasoning block.
        max_length (:obj:`int`, `optional`, defaults to 512):
            Max length of tokens considered in document.
        use_bert(:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use bert as encoder layer.
        initializer_range(:obj:`float`, `optional`, defaults to :obj:`0.02`):
            Initializer range for weights.

    Example::

        from sgnlp.models.lsr import LsrConfig

        # Initializing with default values
        configuration = LsrConfig()
    """

    def __init__(
            self,
            finetune_emb: bool = False,
            word_embedding_shape: Tuple[int, int] = (194784, 100),
            ner_dim: int = 20,
            coref_dim: int = 20,
            hidden_dim: int = 120,
            distance_size: int = 20,
            num_relations: int = 97,
            dropout_rate: float = 0.3,
            dropout_emb: float = 0.2,
            dropout_gcn: float = 0.4,
            use_struct_att: bool = False,
            use_reasoning_block: bool = True,
            reasoner_layer_sizes: Tuple[int, int] = (3, 4),
            max_length: int = 512,
            use_bert: bool = False,
            initializer_range: float = 0.02,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.finetune_emb = finetune_emb
        self.word_embedding_shape = word_embedding_shape
        self.ner_dim = ner_dim
        self.coref_dim = coref_dim
        self.hidden_dim = hidden_dim
        self.distance_size = distance_size
        self.num_relations = num_relations
        self.dropout_rate = dropout_rate
        self.dropout_emb = dropout_emb
        self.dropout_gcn = dropout_gcn
        self.use_struct_att = use_struct_att
        self.use_reasoning_block = use_reasoning_block
        self.reasoner_layer_sizes = reasoner_layer_sizes
        self.max_length = max_length
        self.use_bert = use_bert
        self.initializer_range = initializer_range
