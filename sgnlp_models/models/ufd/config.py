from transformers import PretrainedConfig, XLMRobertaConfig


class UFDAdaptorGlobalConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~UFDAdaptorGlobalModel`.
    It is used to instantiate a UFD Adaptor Global model according to the specified
    arguments, defining the model architecture.

    Args:
        PretrainedConfig (:obj:`PretrainedConfig`): transformer :obj:`PreTrainedConfig` base class
    """

    model_type = "adaptor_global"

    def __init__(
        self, in_dim=1024, dim_hidden=1024, out_dim=1024, initrange=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.dim_hidden = dim_hidden
        self.out_dim = out_dim
        self.initrange = initrange


class UFDAdaptorDomainConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~UFDAdaptorDomainModel`.
    It is used to instantiate a UFD Adaptor Domain model according to the specified
    arguments, defining the model architecture.

    Args:
        PretrainedConfig (:obj:`PretrainedConfig`): transformer :obj:`PreTrainedConfig` base class
    """

    model_type = "adaptor_domain"

    def __init__(
        self, in_dim=1024, dim_hidden=1024, out_dim=1024, initrange=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.dim_hidden = dim_hidden
        self.out_dim = out_dim
        self.initrange = initrange


class UFDCombineFeaturesMapConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~UFDCombineFeaturesMapModel`.
    It is used to instantiate a UFD Combine Features Map model according to the specified
    arguments, defining the model architecture.

    Args:
        PretrainedConfig (:obj:`PretrainedConfig`): transformer :obj:`PreTrainedConfig` base class
    """

    model_type = "combine_features_map"

    def __init__(self, embed_dim=1024, initrange=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.initrange = initrange


class UFDClassifierConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~UFDClassifierModel`.
    It is used to instantiate a UFD Classifier model according to the specified
    arguments, defining the model architecture.

    Args:
        PretrainedConfig (:obj:`PretrainedConfig`): transformer :obj:`PreTrainedConfig` base class
    """

    model_type = "classifier"

    def __init__(self, embed_dim=1024, num_class=2, initrange=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.initrange = initrange


class UFDEmbeddingConfig(XLMRobertaConfig):
    """
    This is the configuration class to store the configuration of a :class:`~UFDEmbeddingModel`.
    It is used to instantiate a UFD Embedding model according to the specified
    arguments, defining the model architecture.

    Args:
        PretrainedConfig (:obj:`PretrainedConfig`): transformer :obj:`PreTrainedConfig` base class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
