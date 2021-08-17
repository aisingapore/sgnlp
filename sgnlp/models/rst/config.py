from transformers import PretrainedConfig


class RSTPointerNetworkConfig(PretrainedConfig):
    model_type = "rst_pointer_network"

    def __init__(
        self,
        word_dim,
        hidden_dim=64,
        dropout_prob=0.2,
        is_bi_encoder_rnn=True,
        num_rnn_layers=6,
        rnn_type="GRU",
        with_finetuning=False,
        is_batch_norm=True,
        use_cuda=True,
            **kwargs):
        super().__init__(**kwargs)
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.is_bi_encoder_rnn = is_bi_encoder_rnn
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.with_finetuning = with_finetuning
        self.is_batch_norm = is_batch_norm
        self.use_cuda = use_cuda
