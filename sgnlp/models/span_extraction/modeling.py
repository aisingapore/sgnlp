from typing import Optional

import torch
from transformers import BertForQuestionAnswering


class RecconSpanExtractionModel(BertForQuestionAnswering):
    """
    The Reccon Span Extraction Model with a span classification head on top for extractive question-answering tasks like SQuAD.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Args:
        config (:class:`~reccon.RecconSpanExtractionConfig`): Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Use the :obj:`.from_pretrained` method to load the model weights.

    Example::

        from sgnlp.models.span_extraction import RecconSpanExtractionConfig, RecconSpanExtractionTokenizer, RecconSpanExtractionModel, utils

        # 1. load from default
        config = RecconSpanExtractionConfig()
        model = RecconSpanExtractionModel(config)

        # 2. load from pretrained
        config = RecconSpanExtractionConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/reccon_span_extraction/config.json")
        model = RecconSpanExtractionModel.from_pretrained("https://storage.googleapis.com/sgnlp/models/reccon_span_extraction/pytorch_model.bin", config=config)

        # Using model
        tokenizer = RecconSpanExtractionTokenizer.from_pretrained("mrm8488/spanbert-finetuned-squadv2")
        text = {
            'context': "Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan--you must be excited !",
            'qas': [{
                'id': 'dailydialog_tr_1097_utt_1_true_cause_utt_1_span_0',
                'is_impossible': False,
                'question': "The target utterance is Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan--you must be excited ! The evidence utterance is Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan--you must be excited ! What is the causal span from context that is relevant to the target utterance's emotion happiness ?",
                'answers': [{'text': "Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan",
                'answer_start': 0}]}]}
        dataset, _, _ = utils.load_examples(text, tokenizer)
        inputs = {"input_ids": dataset[0], "attention_mask": dataset[1], "token_type_ids": dataset[2]}
        outputs = model(**inputs)
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
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
            start_positions=start_positions,
            end_positions=end_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
