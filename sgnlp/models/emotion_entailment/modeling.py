from typing import Optional

import torch
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
        config = RecconEmotionEntailmentConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/config.json")
        model = RecconEmotionEntailmentModel.from_pretrained("https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/pytorch_model.bin", config=config)

        # Using model
        tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
        text = "surprise <SEP> Me ? You're the one who pulled out in front of me ! <SEP> Why don't you watch where you're going ? <SEP> Why don't you watch where you're going ? Me ? You're the one who pulled out in front of me !"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)

    """

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
