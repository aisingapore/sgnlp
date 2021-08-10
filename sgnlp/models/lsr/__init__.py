from ...utils.requirements import check_requirements

requirements = ["networkx"]
check_requirements(requirements)

from spacy.cli import download

download('en_core_web_sm')

from .modeling import LsrModel
from .config import LsrConfig
from .preprocess import LsrPreprocessor
from .postprocess import LsrPostprocessor
