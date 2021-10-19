from transformers import PretrainedConfig


class CsgConfig(PretrainedConfig):
    def __init__(
        self,
        source_vocab_size=30004,
        embedding_dim=500,
        dropout=0.2,
        hidden_dim=1024,
        kernel_size=3,
        num_encoders=7,
        num_aux_encoders=3,
        target_vocab_size=30004,
        num_decoders=7,
        src_max_seq_len=1024,
        ctx_max_seq_len=1024,
        trg_max_seq_len=1024,
        padding_idx=1,
        eos_idx=2,
        initializer_range=0.02,
        beam_size=12,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.source_vocab_size = source_vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_encoders = num_encoders
        self.num_aux_encoders = num_aux_encoders
        self.target_vocab_size = target_vocab_size
        self.num_decoders = num_decoders
        self.src_max_seq_len = src_max_seq_len
        self.ctx_max_seq_len = ctx_max_seq_len
        self.trg_max_seq_len = trg_max_seq_len
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.initializer_range = initializer_range
        self.beam_size = beam_size
