from transformers import PretrainedConfig


class CoherenceMomentumConfig(PretrainedConfig):
    def __init__(
        self,
        model_size: str = "base",
        margin: float = 0.1,
        num_negs: int = 5,
        max_len: int = 600,
        num_rank_negs: int = 50,
        momentum_coefficient: float = 0.9999999,
        queue_size: int = 1000,
        contrastive_loss_weight: float = 0.85,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model_size = model_size
        self.margin = margin
        self.num_negs = num_negs
        self.max_len = max_len
        self.num_rank_negs = num_rank_negs
        self.momentum_coefficient = momentum_coefficient
        self.queue_size = queue_size
        self.contrastive_loss_weight = contrastive_loss_weight
