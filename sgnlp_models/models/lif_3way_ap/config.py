from transformers import PretrainedConfig

DEFAULT_CONFIG_ARGS = {
    "char_embedding_args": {
        "num_embeddings": 102,
        "embedding_dim": 20,
        "num_filters": 100,
        "kernel_size": 5,
        "padding_idx": 0,
        "stride": 1
    },
    "word_embedding_args": {
        "num_embeddings": 10002,
        "embedding_dim": 300,
        "padding_idx": 0
    },
    "p_seq_enc_args": {
        "bidirectional": True,
        "hidden_size": 150,
        "input_size": 400,
        "num_layers": 1,
        "batch_first": True
    },
    "q_seq_enc_args": {
        "bidirectional": True,
        "hidden_size": 150,
        "input_size": 400,
        "num_layers": 1,
        "batch_first": True
    },
    "c_seq_enc_args": {
        "bidirectional": True,
        "hidden_size": 150,
        "input_size": 400,
        "num_layers": 1,
        "batch_first": True
    },
    "cartesian_attn_mat_args": {
        "identity": True
    },
    "pq_attn_mat_args": {
        "identity": False,
        "projector_args": {
            "dropout": 0.3,
            "hidden_dims": 300,
            "input_dim": 300,
            "num_layers": 1
        }
    },
    "pc_attn_mat_args": {
        "identity": False,
        "projector_args": {
            "dropout": 0.3,
            "hidden_dims": 300,
            "input_dim": 300,
            "num_layers": 1
        }
    },
    "cq_attn_mat_args": {
        "identity": False,
        "projector_args": {
            "dropout": 0.3,
            "hidden_dims": 300,
            "input_dim": 300,
            "num_layers": 1
        }
    },
    "gate_qdep_penc_args": {
        "gate_args": {
            "dropout": 0.3,
            "hidden_dims": 600,
            "input_dim": 600,
            "num_layers": 1
        }
    },
    "qdep_penc_rnn_args": {
        "bidirectional": True,
        "hidden_size": 150,
        "input_size": 600,
        "num_layers": 1,
        "batch_first": True
    },
    "mfa_enc_args": {
        "projector_args": {
            "activation": "tanh",
            "dropout": 0.3,
            "hidden_dims": 1200,
            "input_dim": 300,
            "num_layers": 1
        },
        "gate_args": {
            "activation": "linear",
            "dropout": 0.3,
            "hidden_dims": 600,
            "input_dim": 600,
            "num_layers": 1
        },
        "num_factor": 4,
        "attn_pooling": "max"
    },
    "mfa_rnn_args": {
        "bidirectional": True,
        "hidden_size": 150,
        "input_size": 600,
        "num_layers": 1,
        "batch_first": True
    },
}


class LIF3WayAPConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~sg_nlp_models.L2AFModel`.
    It is used to instantiate a classification model that determines whether a question is a follow-up question of the
    current conversation for effective answer finding according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        TODO

    """

    def __init__(self,
                 char_embedding_args: dict = DEFAULT_CONFIG_ARGS["char_embedding_args"],
                 word_embedding_args: dict = DEFAULT_CONFIG_ARGS["word_embedding_args"],
                 p_seq_enc_args: dict = DEFAULT_CONFIG_ARGS["p_seq_enc_args"],
                 q_seq_enc_args: dict = DEFAULT_CONFIG_ARGS["q_seq_enc_args"],
                 c_seq_enc_args: dict = DEFAULT_CONFIG_ARGS["c_seq_enc_args"],
                 cartesian_attn_mat_args: dict = DEFAULT_CONFIG_ARGS["cartesian_attn_mat_args"],
                 pq_attn_mat_args: dict = DEFAULT_CONFIG_ARGS["pq_attn_mat_args"],
                 pc_attn_mat_args: dict = DEFAULT_CONFIG_ARGS["pc_attn_mat_args"],
                 cq_attn_mat_args: dict = DEFAULT_CONFIG_ARGS["cq_attn_mat_args"],
                 gate_qdep_penc_args: dict = DEFAULT_CONFIG_ARGS["gate_qdep_penc_args"],
                 qdep_penc_rnn_args: dict = DEFAULT_CONFIG_ARGS["qdep_penc_rnn_args"],
                 mfa_enc_args: dict = DEFAULT_CONFIG_ARGS["mfa_enc_args"],
                 mfa_rnn_args: dict = DEFAULT_CONFIG_ARGS["mfa_rnn_args"],
                 dropout: float = 0.3,
                 is_qdep_penc: bool = True,
                 is_mfa_enc: bool = True,
                 with_knowledge: bool = True,
                 is_qc_ap: bool = True,
                 shared_rnn: bool = True,
                 initializer_range: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.char_embedding_args = char_embedding_args
        self.word_embedding_args = word_embedding_args
        self.p_seq_enc_args = p_seq_enc_args
        self.q_seq_enc_args = q_seq_enc_args
        self.c_seq_enc_args = c_seq_enc_args

        self.cartesian_attn_mat_args = cartesian_attn_mat_args
        self.pq_attn_mat_args = pq_attn_mat_args
        self.pc_attn_mat_args = pc_attn_mat_args
        self.cq_attn_mat_args = cq_attn_mat_args

        self.gate_qdep_penc_args = gate_qdep_penc_args
        self.qdep_penc_rnn_args = qdep_penc_rnn_args
        self.mfa_enc_args = mfa_enc_args
        self.mfa_rnn_args = mfa_rnn_args

        self.dropout = dropout
        self.is_qdep_penc = is_qdep_penc
        self.is_mfa_enc = is_mfa_enc
        self.with_knowledge = with_knowledge
        self.is_qc_ap = is_qc_ap
        self.shared_rnn = shared_rnn

        self.initializer_range = initializer_range
