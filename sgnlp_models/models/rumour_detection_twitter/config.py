from transformers import PretrainedConfig


class RumourDetectionTwitterConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of  :class:`~RumourDetectionTwitterModel`. It is used to instantiate a Rumour Detection
    model.

    Args:
        num_classes (:obj:`int`, optional): number of classers predicted by the model. Defaults to 4.
        max_vocab (:obj:`int`, optional): vocabulary size. Defaults to 15000.
        emb_dim (:obj:`int`, optional): size of each token embedding vector. Defaults to 300.
    """

    def __init__(
        self,
        num_classes=4,
        max_vocab=15000,
        emb_dim=300,
        num_structure_index=5,
        include_key_structure=True,
        include_val_structure=True,
        word_module_version=4,
        post_module_version=3,
        train_word_emb=False,
        train_pos_emb=False,
        size=100,
        interval=10,
        include_time_interval=True,
        max_length=35,
        max_tweets=339,
        d_model=300,
        dropout_rate=0.3,
        ff_word=True,
        num_emb_layers_word=2,
        n_mha_layers_word=2,
        n_head_word=2,
        ff_post=True,
        num_emb_layers=2,
        n_mha_layers=12,
        n_head=2,
        d_feed_forward=600,
        gpu=False,
        gpu_idx=[0],
        main_gpu=[0],
        initializer_range=0.02,
        loss="cross_entropy",
        **kwargs
    ):

        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.max_vocab = max_vocab
        self.emb_dim = emb_dim
        self.num_structure_index = num_structure_index
        self.include_key_structure = include_key_structure
        self.include_val_structure = include_val_structure
        self.word_module_version = word_module_version
        self.post_module_version = post_module_version
        self.train_word_emb = train_word_emb
        self.train_pos_emb = train_pos_emb
        self.size = size  # number of bins for the time embeddings
        self.interval = interval  # time embedding interval size (unsure of units)
        self.include_time_interval = include_time_interval
        self.max_length = max_length
        self.max_tweets = max_tweets
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.ff_word = ff_word
        self.num_emb_layers_word = num_emb_layers_word
        self.n_mha_layers_word = n_mha_layers_word
        self.n_head_word = n_head_word
        self.ff_post = ff_post
        self.num_emb_layers = num_emb_layers
        self.n_mha_layers = n_mha_layers
        self.n_head = n_head
        self.d_feed_forward = d_feed_forward
        self.initializer_range = initializer_range
        self.loss = loss

        # TODO: to move these parameters outside of the model config and into the training args
        self.gpu = gpu
        self.gpu_idx = gpu_idx
        self.main_gpu = gpu_idx
