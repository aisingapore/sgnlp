from transformers import PretrainedConfig


class RstPointerSegmenterConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a
    :class:`~sgnlp.models.rst_pointer.modeling.RstPointerSegmenterModel`.
    It is used to instantiate a discourse segmenter pointer network according to the specified arguments, defining the
    model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        word_dim (:obj:`int`, defaults to 1024): Word embedding dimension size.
        hidden_dim (:obj:`int`, defaults to 64): Hidden dimension zie.
        dropout_prob (:obj:`float`, defaults to 0.2): Dropout probability.
        use_bilstm (:obj:`bool`, defaults to :obj:`True`): Whether to use bilstm layer.
        num_rnn_layers (:obj:`int`, defaults to 6): Number of RNN layers.
        rnn_type (:obj:`str`, defaults to "GRU"): RNN type. Supported choices: ["LSTM", "GRU"].
        is_batch_norm (:obj:`bool`, defaults to True): Whether to use batch normalization.
        elmo_size (:obj:`bool`, defaults to "Large"): Elmo size. Supported choices: ["Large", "Medium", "Small"].

    Example::

        from sgnlp.models.rst_pointer import RstPointerSegmenterConfig

        # Initialize with default values
        config = RstPointerSegmenterConfig()
    """

    def __init__(
        self,
        word_dim=1024,
        hidden_dim=64,
        dropout_prob=0.2,
        use_bilstm=True,
        num_rnn_layers=6,
        rnn_type="GRU",
        is_batch_norm=True,
        elmo_size="Large",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.use_bilstm = use_bilstm
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.is_batch_norm = is_batch_norm
        self.elmo_size = elmo_size


class RstPointerParserConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a
    :class:`~sgnlp.models.rst_pointer.modeling.RstPointerParserModel`.
    It is used to instantiate a discourse parser pointer network according to the specified arguments, defining the
    model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        word_dim (:obj:`int`, defaults to 1024): Word dimension size.
        hidden_size (:obj:`int`, defaults to 64): Hidden dimension size.
        decoder_input_size (:obj:`int`, defaults to 64): Decoder input size.
        atten_model: (:obj:`str`, defaults to "Dotproduct"):
            Attention type. Supported choices: ["Dotproduct", "Biaffine"].
        classifier_input_size (:obj:`int`, defaults to 64): Classifier input size.
        classifier_hidden_size (:obj:`int`, defaults to 64): Classifier hidden size.
        highorder (:obj:`bool`, defaults to False): Whether to incorporate higher order information.
        classes_label (:obj:`int`, defaults to 39): Number of classes to predict for.
        classifier_bias (:obj:`bool`, defaults to True): Whether to use bias for classifier.
        rnn_layers (:obj:`int`, defaults to 6): Number of RNN layers.
        dropout_e (:obj:`float`, defaults to 0.33): Dropout rate for encoder layer.
        dropout_d (:obj:`float`, defaults to 0.5): Dropout rate for decoder layer.
        dropout_c (:obj:`float`, defaults to 0.5): Dropout rate for classifier layer.
        elmo_size (:obj:`bool`, defaults to "Large"): Elmo size. Supported choices: ["Large", "Medium", "Small"].

    Example::

        from sgnlp.models.rst_pointer import RstPointerParserConfig

        # Initialize with default values
        config = RstPointerParserConfig()
    """

    def __init__(
        self,
        word_dim=1024,
        hidden_size=64,
        decoder_input_size=64,
        atten_model="Dotproduct",
        classifier_input_size=64,
        classifier_hidden_size=64,
        highorder=False,
        classes_label=39,
        classifier_bias=True,
        rnn_layers=6,
        dropout_e=0.33,
        dropout_d=0.5,
        dropout_c=0.5,
        elmo_size="Large",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.decoder_input_size = decoder_input_size
        self.atten_model = atten_model
        self.classifier_input_size = classifier_input_size
        self.classifier_hidden_size = classifier_hidden_size
        self.highorder = highorder
        self.classes_label = classes_label
        self.classifier_bias = classifier_bias
        self.rnn_layers = rnn_layers
        self.dropout_e = dropout_e
        self.dropout_d = dropout_d
        self.dropout_c = dropout_c
        self.elmo_size = elmo_size
