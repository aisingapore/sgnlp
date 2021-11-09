from sgnlp.utils.base_model import BaseAllenNlpModel
from .modules.allennlp.predictor import Lif3WayApPredictor


class Lif3WayApModel(BaseAllenNlpModel):
    PREDICTOR_CLASS = Lif3WayApPredictor
