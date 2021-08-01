import pytest
import unittest

from sgnlp_models.models.ufd import (
    UFDAdaptorDomainModel,
    UFDAdaptorGlobalModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
    UFDModelBuilder,
    UFDModel
)


class TestUFDModelBuilderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model_map = {
            'adaptor_domain': UFDAdaptorDomainModel,
            'adaptor_global': UFDAdaptorGlobalModel,
            'maper': UFDCombineFeaturesMapModel,
            'classifier': UFDClassifierModel
        }

    @pytest.mark.slow
    def test_build_model_group(self):
        model_builder = UFDModelBuilder()
        model_group = model_builder.build_model_group()

        self.assertTrue(len(model_builder.trimmed_model_sets), len(model_group.keys()))
        for key, mod in model_group.items():
            self.assertTrue(isinstance(mod, UFDModel))
