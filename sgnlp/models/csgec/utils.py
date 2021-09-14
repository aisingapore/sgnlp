from datasets import load_dataset
from torch.utils.data import DataLoader


def load_transform_dataset(train_data, batch_size, source_tokenizer, context_tokenizer, target_tokenizer):
    """
    train_data
        Path to the train data json file
    batch_size : int
        Batch size
    source_tokenizer : CSGTokenizerFast
        Tokenizer object for source sequences
    context_tokenizer : CSGTokenizerFast
        Tokenizer object for context sequences
    target_tokenizer : CSGTokenizerFast
        Tokenizer object for target sequences
    """

    #TODO make the cache_dir an argument
    dataset = load_dataset('json', data_files=train_data, field='data', cache_dir="/polyaxon-data/SG-NLP/crosentgec_refactor/data/")
    dataset = dataset.map(lambda x: encode_dataset(x, source_tokenizer, context_tokenizer, target_tokenizer), batched=True)
    dataset.set_format(type='torch', columns=['source_ids', 'context_ids', 'target_ids'])
    dataloader = DataLoader(dataset["train"], batch_size=batch_size)
    return dataloader


def encode_dataset(examples, source_tokenizer, context_tokenizer, target_tokenizer):
    output = {
        "source_ids": source_tokenizer(examples['source'], truncation=True, padding='max_length').input_ids,
        "context_ids": context_tokenizer(examples['context'], truncation=True, padding='max_length').input_ids,
        "target_ids": target_tokenizer(examples['target'], truncation=True, padding='max_length').input_ids,
    }
    return output

