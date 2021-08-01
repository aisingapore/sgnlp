from transformers import RobertaForSequenceClassification


class RecconEmotionEntailmentModel(RobertaForSequenceClassification):
    """The Reccon Emotion Entailment Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Args:
        config (:class:`~reccon.RecconEmotionEntailmentConfig`): Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Use the :obj:`.from_pretrained` method to load the model weights.

    Example::

            from sgnlp.models.emotion_entailment import RecconEmotionEntailmentConfig, RecconEmotionEntailmentModel, RecconEmotionEntailmentTokenizer

            # 1. load from default
            config = RecconEmotionEntailmentConfig()
            model = RecconEmotionEntailmentModel(config)
            # 2. load from pretrained
            config = RecconEmotionEntailmentConfig.from_pretrained("https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/config.json")
            model = RecconEmotionEntailmentModel.from_pretrained("https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/pytorch_model.bin", config=config)

            # Using model
            tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
            text = "surprise <SEP> Me ? You're the one who pulled out in front of me ! <SEP> Why don't you watch where you're going ? <SEP> Why don't you watch where you're going ? Me ? You're the one who pulled out in front of me !"
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)

    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        return super().forward(**kwargs)
