from transformers import PretrainedConfig


class CoherenceConfig(PretrainedConfig):
    model_type = "general_coherence_model"

    def __init__(
        self,
        model_size: str = "base",
        lr_start: float = 5e-06,
        lr_end: float = 1e-06,
        lr_anneal_epochs: int = 50,
        eval_interval: int = 1000,
        seed: int = 100,
        batch_size: int = 1,
        margin: float = 0.1,
        num_negs: int = 5,
        max_len: int = 600,
        num_rank_negs: int = 50,
        train_steps: int = 200,
        momentum_coefficient: float = 0.9999999,
        queue_size: int = 1000,
        contrastive_loss_weight: float = 0.85,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model_size = model_size
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_anneal_epochs = lr_anneal_epochs
        self.eval_interval = eval_interval
        self.seed = seed
        self.batch_size = batch_size
        self.margin = margin
        self.num_negs = num_negs
        self.max_len = max_len
        self.num_rank_negs = num_rank_negs
        self.train_steps = train_steps
        self.momentum_coefficient = momentum_coefficient
        self.queue_size = queue_size
        self.contrastive_loss_weight = contrastive_loss_weight
