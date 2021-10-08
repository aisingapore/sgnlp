from sgnlp.models.csgec import (
    CSGConfig,
    CSGModel,
    download_tokenizer_files,
)

config = CSGConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/csgec/config.json")
model = CSGModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/csgec/pytorch_model.bin",
    config=config,
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/src_tokenizer/",
    "csgec_src_tokenizer",
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/ctx_tokenizer/",
    "csgec_ctx_tokenizer",
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/tgt_tokenizer/",
    "csgec_tgt_tokenizer",
)
