from transformers import RobertaConfig


class RecconEmotionEntailmentConfig(RobertaConfig):
    """
    This is the configuration class to store the configuration of a :class:`~reccon.RecconEmotionEntailmentModel`.
    It is used to instantiate a RECCON emotion entailment model according to the specified
    arguments, defining the model architecture.

    Examples::

        from sg_nlp import RecconEmotionEntailmentConfig, RecconEmotionEntailmentModel

        # Initializing the RECCON emotion entailment config
        reccon_emotion_entailment_config = RecconEmotionEntailmentConfig()

        # Initializing a model with the RECCON emotion entailment config
        model = RecconEmotionEntailmentModel(reccon_emotion_entailment_config)

        # Accessing the model configuration
        configuration = model.config
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
