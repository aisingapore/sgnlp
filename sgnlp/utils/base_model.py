from allennlp.predictors import Predictor


class BaseAllenNlpModel:
    PREDICTOR_CLASS: Predictor = None

    @classmethod
    def from_pretrained(cls, model_path, **kwargs) -> Predictor:
        return cls.PREDICTOR_CLASS.from_path(model_path, **kwargs)
