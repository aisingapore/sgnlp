import torch.nn.functional as F

from sgnlp.models.sentic_gcn import (
    SenticGCNBertModel, 
    SenticGCNBertPreprocessor, 
    SenticGCNBertConfig
    )

from sgnlp.models.sentic_gcn.postprocess import SenticGCNBertPostprocessor

# Load model
# path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'senticnet5.pickle')
preprocessor = SenticGCNBertPreprocessor(senticnet='https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle', device='cpu')

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
processed_inputs

outputs = model(processed_indices)
t_probs = F.softmax(outputs.logits)
t_probs = t_probs.detach().numpy()

infer_label = [t_probs.argmax(axis=-1)[idx] -1 for idx in range(len(t_probs))]

# Postprocessing
postprocessor = SenticGCNBertPostprocessor()
post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)
print(post_outputs)