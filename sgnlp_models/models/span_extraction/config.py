from transformers import BertConfig


class RecconSpanExtractionConfig(BertConfig):
    """
    This is the configuration class to store the configuration of a :class:`~reccon.RecconSpanExtractionModel`.
    It is used to instantiate a RECCON span extraction model according to the specified
    arguments, defining the model architecture.

    Examples::

        from sg_nlp import RecconSpanExtractionConfig, RecconSpanExtractionModel

        # Initializing the RECCON span extraction config
        reccon_span_extraction_config = RecconSpanExtractionConfig()

        # Initializing a model with the RECCON span extraction config
        reccon_span_extraction_model = RecconSpanExtractionModel(reccon_span_extraction_config)

        # Accessing the model configuration
        configuration = reccon_span_extraction_model.config
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
