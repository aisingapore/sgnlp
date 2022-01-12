from sgnlp.models.sentic_gcn import (
    SenticGCNBertModel, 
    SenticGCNBertPreprocessor, 
    SenticGCNBertConfig
    )

from sgnlp.models.sentic_gcn.postprocess import SenticGCNBertPostprocessor

preprocessor = SenticGCNBertPreprocessor(
    senticnet='https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle', 
    device='cpu'
)

postprocessor = SenticGCNBertPostprocessor()

# Load model
config = SenticGCNBertConfig.from_pretrained('https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json')

model = SenticGCNBertModel.from_pretrained(
    'https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin', 
    config=config
)

# Inputs
inputs = [
    {
        "aspect": ["Soup"],
        "sentence": "Soup is tasty but soup is a little salty. Salty soup."
    }, # 1, -1
    {
        "aspect": ["service"],
        "sentence": "Everyone that sat in the back outside agreed that it was the worst service we had ever received."
    }, # -1
    {
        "aspect": ["location", "food"],
        "sentence": "it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more ."
    } # 0, 1
]

processed_inputs, processed_indices = preprocessor(inputs)
outputs = model(processed_indices)

# Postprocessing
post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)