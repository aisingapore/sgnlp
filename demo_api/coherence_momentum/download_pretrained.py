from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig

config = CoherenceMomentumConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/config.json"
)
model = CoherenceMomentumModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/pytorch_model.bin",
    config=config
)
