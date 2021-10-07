from sgnlp.models.rst_pointer import (
    RstPointerParserConfig,
    RstPointerParserModel,
    RstPointerSegmenterConfig,
    RstPointerSegmenterModel
)

segmenter_config = RstPointerSegmenterConfig.from_pretrained(
    'https://storage.googleapis.com/sgnlp/models/rst_pointer/segmenter/config.json')
segmenter = RstPointerSegmenterModel.from_pretrained(
    'https://storage.googleapis.com/sgnlp/models/rst_pointer/segmenter/pytorch_model.bin',
    config=segmenter_config)

parser_config = RstPointerParserConfig.from_pretrained(
    'https://storage.googleapis.com/sgnlp/models/rst_pointer/parser/config.json')
parser = RstPointerParserModel.from_pretrained(
    'https://storage.googleapis.com/sgnlp/models/rst_pointer/parser/pytorch_model.bin',
    config=parser_config)
