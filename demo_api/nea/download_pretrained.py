import nltk

from sgnlp.models.nea import NEARegPoolingModel, NEAConfig, NEATokenizer, NEAArguments
from sgnlp.models.nea.utils import download_tokenizer_files_from_azure

# Download files and model from azure blob storage
config = NEAConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/nea/config.json"
)
model = NEARegPoolingModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/nea/pytorch_model.bin", config=config
)

cfg = NEAArguments()
download_tokenizer_files_from_azure(cfg)

tokenizer = NEATokenizer.from_pretrained(cfg.tokenizer_args["save_folder"])

nltk.download("punkt")
