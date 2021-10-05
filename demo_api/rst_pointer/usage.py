from sgnlp.models.rst_pointer import (
    RstPointerParserConfig,
    RstPointerParserModel,
    RstPointerSegmenterConfig,
    RstPointerSegmenterModel,
    RstPreprocessor,
    RstPostprocessor
)

# Load processors and models
preprocessor = RstPreprocessor()
postprocessor = RstPostprocessor()

segmenter_config = RstPointerSegmenterConfig.from_pretrained(
    'https://sgnlp.blob.core.windows.net/models/rst_pointer/segmenter/config.json')
segmenter = RstPointerSegmenterModel.from_pretrained(
    'https://sgnlp.blob.core.windows.net/models/rst_pointer/segmenter/pytorch_model.bin',
    config=segmenter_config)
segmenter.eval()

parser_config = RstPointerParserConfig.from_pretrained(
    'https://sgnlp.blob.core.windows.net/models/rst_pointer/parser/config.json')
parser = RstPointerParserModel.from_pretrained(
    'https://sgnlp.blob.core.windows.net/models/rst_pointer/parser/pytorch_model.bin',
    config=parser_config)
parser.eval()

sentences = [
    "James ate some cheese whilst thinking about the play.",
    "CRISPR-Cas9 is a versatile genome editing technology for studying the functions of genetic elements."
]

tokenized_sentences_ids, tokenized_sentences, lengths = preprocessor(sentences)

segmenter_output = segmenter(tokenized_sentences_ids, lengths)
end_boundaries = segmenter_output.end_boundaries

parser_output = parser(tokenized_sentences_ids, end_boundaries, lengths)

trees = postprocessor(sentences=sentences, tokenized_sentences=tokenized_sentences,
                      end_boundaries=end_boundaries,
                      discourse_tree_splits=parser_output.splits)
