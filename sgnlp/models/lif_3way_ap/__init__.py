from ...utils.requirements import check_requirements

requirements = ["allennlp==0.8.4", "scikit-learn==0.22", "overrides==3.1.0"]
check_requirements(requirements)

from .modeling import Lif3WayApModel
