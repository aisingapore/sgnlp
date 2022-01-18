from sgnlp.models.sentic_gcn import (
    SenticGCNBertModel,
    SenticGCNBertPreprocessor,
    SenticGCNBertConfig,
    SenticGCNBertPostprocessor,
)

preprocessor = SenticGCNBertPreprocessor(
    senticnet="https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle", device="cpu"
)

postprocessor = SenticGCNBertPostprocessor()

# Load model
config = SenticGCNBertConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json"
)

model = SenticGCNBertModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", config=config
)

# Inputs
inputs = [
    {
        "aspects": ["service", "decor"],
        "sentence": "Everything is always cooked to perfection , the service is excellent, the decor cool and understated.",
    },
    {
        "aspects": ["food", "portions"],
        "sentence": "The food was lousy - too sweet or too salty and the portions tiny.",
    },
    {
        "aspects": ["service"],
        "sentence": "To sum it up : service varies from good to mediorce , depending on which waiter you get ; generally it is just average ok .",
    },
]

processed_inputs, processed_indices = preprocessor(inputs)
outputs = model(processed_indices)

# Postprocessing
post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)
