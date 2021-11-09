from allennlp.predictors import Predictor


class BaseAllenNlpModel:
    PREDICTOR_CLASS: Predictor = None

    def from_pretrained(self, model_path) -> Predictor:
        return self.PREDICTOR_CLASS.from_path(model_path)
