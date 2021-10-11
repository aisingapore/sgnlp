from sgnlp.models.csgec import (
    CsgConfig,
    CsgModel,
    CsgTokenizer,
    CsgecPreprocessor,
    CsgecPostprocessor,
    download_tokenizer_files,
)

config = CsgConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/csgec/config.json")
model = CsgModel.from_pretrained(
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
src_tokenizer = CsgTokenizer.from_pretrained("csgec_src_tokenizer")
ctx_tokenizer = CsgTokenizer.from_pretrained("csgec_ctx_tokenizer")
tgt_tokenizer = CsgTokenizer.from_pretrained("csgec_tgt_tokenizer")

preprocessor = CsgecPreprocessor(src_tokenizer=src_tokenizer, ctx_tokenizer=ctx_tokenizer)
postprocessor = CsgecPostprocessor(tgt_tokenizer=tgt_tokenizer)

texts = [
    "All of us are living in the technology realm society. Have you ever wondered why we use these tools to connect "
    "ourselves with other people? It started withthe invention of technology which has evolved tremendously over the "
    "past few decades. In the past, we travel by ship and now we can use airplane to do so. In the past, it took a few "
    "days to receive a message as we need to post our letter and now, we can use e-mail which stands for electronic "
    "message to send messages to our friends or even use our handphone to send our messages.",
    "Machines have replaced a bunch of coolies and heavy labor. Cars and trucks diminish the redundancy of long time "
    "shipment. As a result, people have more time to enjoy advantage of modern life. One can easily travel to the "
    "other half of the globe to see beautiful scenery that one dreams for his lifetime. One can also easily see his "
    "deeply loved one through internet from miles away."
]

batch_source_ids, batch_context_ids = preprocessor(texts)
predicted_ids = model.decode(batch_source_ids, batch_context_ids)
predicted_texts = postprocessor(predicted_ids)
