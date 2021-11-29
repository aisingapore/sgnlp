import pathlib
import posixpath
import urllib.parse
from itertools import product
from typing import Any, Dict, List, Tuple

import torch

from .config import (
    UFDAdaptorDomainConfig,
    UFDAdaptorGlobalConfig,
    UFDCombineFeaturesMapConfig,
    UFDClassifierConfig,
)
from .modeling import (
    UFDAdaptorDomainModel,
    UFDAdaptorGlobalModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
    UFDModel,
)


class UFDModelBuilder:
    """
    Class for building the UFD model groups consisting of adaptor domain, adaptor global, combine features map and
    classifier models in their cross language cross domain configuration from pretrained model weights and config.

    The builder expects the models structure as follows:

    |models_root_path
    |----[sourcedomain_targetlanguage_targetdomain_modeltype]
    |--------config_filename
    |--------model_filename
    """

    def __init__(
        self,
        models_root_path: str = "https://storage.googleapis.com/sgnlp/models/ufd/",
        source_domains: List[str] = ["books", "music", "dvd"],
        target_languages: List[str] = ["de", "jp", "fr"],
        target_domains: List[str] = ["books", "music", "dvd"],
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        device: torch.device = torch.device("cpu"),
    ):
        self.from_remote_url = models_root_path.startswith(
            "https://"
        ) or models_root_path.startswith("http://")
        self.models_root_path = models_root_path
        self.source_domains = source_domains
        self.target_languages = target_languages
        self.target_domains = target_domains
        self.config_filename = config_filename
        self.model_filename = model_filename
        self.device = device

        self.models_group = ["adaptor_domain", "adaptor_global", "maper", "classifier"]

        model_sets = list(
            product(*[self.source_domains, self.target_languages, self.target_domains])
        )
        self.trimmed_model_sets = [lst for lst in model_sets if lst[0] != lst[2]]

    def _create_adaptor_domain(
        self, config_full_path: str, model_full_path: str
    ) -> UFDAdaptorDomainModel:
        """
        Method to create UFDAdaptorDomainConfig and UFDAdaptorDomainModel from pretrained weights.

        Args:
            config_full_path (str): full path to config file
            model_full_path (str): full path to pretrained model weights

        Returns:
            UFDAdaptorDomainModel: return the created model instance
        """
        adaptor_domain_config = UFDAdaptorDomainConfig.from_pretrained(config_full_path)
        adaptor_domain_model = (
            UFDAdaptorDomainModel.from_pretrained(
                model_full_path, config=adaptor_domain_config
            )
            .to(self.device)
            .eval()
        )
        return adaptor_domain_model

    def _create_adaptor_global(
        self, config_full_path: str, model_full_path: str
    ) -> UFDAdaptorGlobalModel:
        """
        Method to create UFDAdaptorGlobalConfig and UFDAdaptorGlobalModel from pretrained weights.

        Args:
            config_full_path (str): full path to config file
            model_full_path (str): full path to pretrained model weights

        Returns:
            UFDAdaptorGlobalModel: return the created model instance
        """
        adaptor_global_config = UFDAdaptorGlobalConfig.from_pretrained(config_full_path)
        adaptor_global_model = (
            UFDAdaptorGlobalModel.from_pretrained(
                model_full_path, config=adaptor_global_config
            )
            .to(self.device)
            .eval()
        )
        return adaptor_global_model

    def _create_maper(
        self, config_full_path: str, model_full_path: str
    ) -> UFDCombineFeaturesMapModel:
        """
        Method to create UFDCombineFeaturesMapConfig and UFDCombineFeaturesMapModel from pretrained weights.

        Args:
            config_full_path (str): full path to config file
            model_full_path (str): full path to pretrained model weights

        Returns:
            UFDCombineFeaturesMapModel: return the created model instance
        """
        maper_config = UFDCombineFeaturesMapConfig.from_pretrained(config_full_path)
        maper_model = (
            UFDCombineFeaturesMapModel.from_pretrained(
                model_full_path, config=maper_config
            )
            .to(self.device)
            .eval()
        )
        return maper_model

    def _create_classifier(
        self, config_full_path: str, model_full_path: str
    ) -> UFDClassifierModel:
        """
        Method to create UFDClassifierConfig and UFDClassifierModel from pretrained weights.

        Args:
            config_full_path (str]): full path to config file
            model_full_path (str): full path to pretrained model weights

        Returns:
            UFDClassifierModel: return the created model instance
        """
        classifier_config = UFDClassifierConfig.from_pretrained(config_full_path)
        classifier_model = (
            UFDClassifierModel.from_pretrained(
                model_full_path, config=classifier_config
            )
            .to(self.device)
            .eval()
        )
        return classifier_model

    def _build_file_path(self, model_name: str) -> Tuple[str, str]:
        """
        Helper method to form the full file path for the model weights and the config file.

        Args:
            model_name (str): name of the model to form (i.e. 'books_de_dvd_adaptor_global').

        Returns:
            Tuple[str, str]: return the full path for the config file and the model weight.
        """
        if self.from_remote_url:
            remote_config_path = posixpath.join(model_name, self.config_filename)
            remote_model_path = posixpath.join(model_name, self.model_filename)
            config_path = urllib.parse.urljoin(
                self.models_root_path, remote_config_path
            )
            model_path = urllib.parse.urljoin(self.models_root_path, remote_model_path)
        else:
            model_root_path = pathlib.Path(self.models_root_path).joinpath(model_name)
            config_path = str(model_root_path / self.config_filename)
            model_path = str(model_root_path / self.model_filename)
        return config_path, model_path

    @property
    def model_map(self) -> Dict[str, Any]:
        """
        Property to hold all model creation methods in a hashmap.

        Returns:
            Dict[str, Any]: return a dict of models name as key and model create method as values.
        """
        return {
            self.models_group[0]: self._create_adaptor_domain,
            self.models_group[1]: self._create_adaptor_global,
            self.models_group[2]: self._create_maper,
            self.models_group[3]: self._create_classifier,
        }

    def build_model_group(self) -> Dict[str, UFDModel]:
        """
        Method to generate all UFD model grouping.

        Returns:
            Dict[str, UFDModel]: return a dict of model names as keys and the created UFDModel instances as the value.
                                    Each UFDModel instances is created based on their respective adaptor_domain,
                                    adaptor_global, features_maper and classifier model instances.
        """
        models_group = {}
        for grp in self.trimmed_model_sets:
            model_name = "_".join(grp)
            ad_cfg_path, ad_model_path = self._build_file_path(
                model_name + "_adaptor_domain"
            )
            ag_cfg_path, ag_model_path = self._build_file_path(
                model_name + "_adaptor_global"
            )
            feat_maper_cfg_path, feat_maper_model_path = self._build_file_path(
                model_name + "_maper"
            )
            classifier_cfg_path, classifier_model_path = self._build_file_path(
                model_name + "_classifier"
            )
            models_group[model_name] = UFDModel(
                adaptor_domain=self.model_map[self.models_group[0]](
                    ad_cfg_path, ad_model_path
                ),
                adaptor_global=self.model_map[self.models_group[1]](
                    ag_cfg_path, ag_model_path
                ),
                feature_maper=self.model_map[self.models_group[2]](
                    feat_maper_cfg_path, feat_maper_model_path
                ),
                classifier=self.model_map[self.models_group[3]](
                    classifier_cfg_path, classifier_model_path
                ),
            )
        return models_group
