from torch._C import device
from transformers import cached_path
import torch.nn.functional as F


from sgnlp.models.sentic_gcn import (
    SenticGCNBertModel, 
    SenticGCNBertPreprocessor, 
    SenticGCNBertConfig
    )

from sgnlp.models.sentic_gcn.postprocess import SenticGCNBertPostprocessor

"""
Overall steps:
1. tokenize the data
2. Get embedding matrix 
    -> self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
3. Set embedding martrix in the loaded model class
4. Run the model (train / test)
"""


# Load model
# path = '/Users/weiming/Dev/sg-nlp/sgnlp/sgnlp/models/sentic_gcn/senticnet5.pickle'
path = '../../sgnlp/models/sentic_gcn/senticnet5.pickle'
preprocessor = SenticGCNBertPreprocessor(senticnet=path, device='cpu')

config = SenticGCNBertConfig.from_pretrained('https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json')


embedded_matrix = ""

# tokenizer = SenticGCNTokenizer.from_pretrained("senticgcn")
# Other tokenizers
# BertTokenizer: 'bert-base-uncased'

model = SenticGCNBertModel.from_pretrained(
    'https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin', 
    config=config
)

# Model predict

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

# print(processed_inputs[0])
# print(infer_label[0])


# tensor_dict = preprocessor(input_batch)
# print(tensor_dict)
# output = model(**tensor_dict)
# sentiment = ""


# Postprocessing
postprocessor = SenticGCNBertPostprocessor()
post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)
print(post_outputs)