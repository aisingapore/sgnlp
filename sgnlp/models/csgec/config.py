from transformers import PretrainedConfig


class CSGConfig(PretrainedConfig):
    def __init__(self,
                 source_vocab_size = 30000,
                 token_embedding_dim = 500,
                 dropout= 0.2,
                 hidden_dim= 1024,
                 kernel_size= 3,
                 num_encoders= 7,
                 num_ctx_encoders=0,
                 target_vocab_size= 30000,
                 num_decoders= 7,
                 src_max_seq_len= 150,
                 ctx_max_seq_len= 250,
                 trg_max_seq_len = 150,
                 padding_idx= 0,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.source_vocab_size = source_vocab_size
        self.token_embedding_dim = token_embedding_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_encoders = num_encoders
        self.num_ctx_encoders = num_ctx_encoders
        self.target_vocab_size = target_vocab_size
        self.num_decoders = num_decoders
        self.src_max_seq_len = src_max_seq_len
        self.ctx_max_seq_len = ctx_max_seq_len
        self.trg_max_seq_len = trg_max_seq_len
        self.padding_idx = padding_idx
        self.initializer_range = initializer_range

