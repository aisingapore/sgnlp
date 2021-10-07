from transformers import PretrainedConfig


class RstPointerSegmenterConfig(PretrainedConfig):
    def __init__(
            self,
            word_dim=1024,
            hidden_dim=64,
            dropout_prob=0.2,
            use_bilstm=True,
            num_rnn_layers=6,
            rnn_type="GRU",
            with_finetuning=False,
            is_batch_norm=True,
            elmo_size="Large",
            **kwargs):
        super().__init__(**kwargs)
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.use_bilstm = use_bilstm
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.with_finetuning = with_finetuning
        self.is_batch_norm = is_batch_norm
        self.elmo_size = elmo_size


class RstPointerParserConfig(PretrainedConfig):
    def __init__(
            self,
            word_dim=1024,
            batch_size=64,
            hidden_size=64,
            decoder_input_size=64,
            atten_model='Dotproduct',
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
            **kwargs):
        super().__init__(**kwargs)
        self.word_dim = word_dim
        self.batch_size = batch_size
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
