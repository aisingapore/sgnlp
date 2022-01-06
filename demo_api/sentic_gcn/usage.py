from sgnlp.models.sentic_gcn import (
    SenticGCNModel, 
    SenticGCNModelOutput, 
    SenticGCNPreTrainedModel,
    SenticGCNBertModelOutput,
    SenticGCNBertPreTrainedModel,
    SenticGCNBertModel,
    SenticGCNEmbeddingPreTrainedModel,
    SenticGCNEmbeddingModel,
    SenticGCNBertEmbeddingModel,
    SenticGCNConfig,
    SenticGCNPreprocessor,
    SenticGCNTokenizer,
    SenticGCNBertTokenizer
    )

"""
Overall steps:
1. tokenize the data
2. Get embedding matrix 
    -> self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
3. Set embedding martrix in the loaded model class
4. Run the model (train / test)
"""


# Load model
config = SenticGCNConfig.from_pretrained("") # Input JSON file

embedded_matrix = ""

tokenizer = SenticGCNTokenizer.from_pretrained("")

model = SenticGCNModel.from_pretrained(
    "", # Input model
    config=config
)

preprocessor = SenticGCNPreprocessor(tokenizer)

# Model predict

# Inputs
input_batch = {} # Dictionary


tensor_dict = preprocessor(input_batch)
raw_output = model(**tensor_dict)
acc , f1 = "" # Return the output, refer to the model class
