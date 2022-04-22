from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, XLMRobertaModel
from transformers.file_utils import ModelOutput


from .config import (
    UFDAdaptorGlobalConfig,
    UFDAdaptorDomainConfig,
    UFDCombineFeaturesMapConfig,
    UFDClassifierConfig,
)


@dataclass
class UFDModelOutput(ModelOutput):
    """
    Base class for outputs of UFD models

    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss. Loss function used is dependent on what is specified in UFDConfig
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, 1)`):
            Classification scores.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class UFDAdaptorGlobalPreTrainedModel(PreTrainedModel):
    """
    The UFD Adaptor Global Pre-Trained Model used as base class for derived Adaptor Global Model.

    This model is the abstract super class for the UFD Adaptor Global model which defines the config class types
    and weights initalization method. This class should not be used or instantiated directly,
    see UFDAdaptorGlobalModel class for usage.
    """

    config_class = UFDAdaptorGlobalConfig
    base_model_prefix = "UFDAdaptorGlobal"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.initrange, self.config.initrange)
            module.bias.data.zero_()


class UFDAdaptorGlobalModel(UFDAdaptorGlobalPreTrainedModel):
    """
    The UFD Adaptor Global Model used for unsupervised training for global context.

    This method inherits from :obj:`UFDAdaptorGlobalPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:class:`~UFDAdaptorGlobalConfig`): Model configuration class with all parameters required for
                                                    the model. Initializing with a config file does not load
                                                    the weights associated with the model, only the configuration.
                                                    Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.lin1 = nn.Linear(config.in_dim, config.dim_hidden)
        self.lin2 = nn.Linear(config.dim_hidden, config.out_dim)
        self.act = F.relu
        self.init_weights()

    def forward(
        self, input: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            input (:ob:'torch.FloatTensor' of shape :obj:'(batch_size, max_num_words)': Word IDs of text
        """
        x = self.lin1(input)
        x = self.act(x)
        local_features = x + input
        x = self.lin2(local_features)
        x = self.act(x) + local_features
        return x, local_features


class UFDAdaptorDomainPreTrainedModel(PreTrainedModel):
    """
    The UFD Adaptor Domain Pre-Trained Model used as base class for derived Adaptor Domain Model.

    This model is the abstract super class for the UFD Adaptor Domain model which defines the config class types
    and weights initalization method. This class should not be used or instantiated directly,
    see UFDAdaptorDomainModel class for usage.
    """

    config_class = UFDAdaptorDomainConfig
    base_model_prefix = "UFDAdaptorDomain"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.initrange, self.config.initrange)
            module.bias.data.zero_()


class UFDAdaptorDomainModel(UFDAdaptorDomainPreTrainedModel):
    """
    The UFD Adaptor Domain Model used for unsupervised training for domain context.

    This method inherits from :obj:`UFDAdaptorDomainPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:class:`~UFDAdaptorDomainConfig`): Model configuration class with all parameters required for
                                                    the model. Initializing with a config file does not load
                                                    the weights associated with the model, only the configuration.
                                                    Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.lin1 = nn.Linear(config.in_dim, config.dim_hidden)
        self.lin2 = nn.Linear(config.dim_hidden, config.out_dim)
        self.act = F.relu
        self.init_weights()

    def forward(
        self, input: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            input (:ob:'torch.FloatTensor' of shape :obj:'(batch_size, max_num_words)': Word IDs of text
        """
        x = self.act(input)
        x = self.lin1(x)
        local_features = x
        x = self.lin2(local_features)
        x = self.act(x)
        x = F.normalize(x, p=2, dim=1)
        return x, local_features


class UFDCombineFeaturesMapPreTrainedModel(PreTrainedModel):
    """
    The UFD Combine Features Map Pre-Trained Model used as base class for derived Combine Features Map Model.

    This model is the abstract super class for the UFD Combine Features Map model which defines the config
    class types and weights initalization method. This class should not be used or instantiated directly,
    see UFDCombineFeaturesMapModel class for usage.

    Args:
        PreTrainedModel ([transformers.PreTrainedModel]): transformer PreTrainedModel base class
    """

    config_class = UFDCombineFeaturesMapConfig
    base_model_prefix = "UFDCombineFeaturesMap"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.initrange, self.config.initrange)
            module.bias.data.zero_()


class UFDCombineFeaturesMapModel(UFDCombineFeaturesMapPreTrainedModel):
    """
    The UFD Combine Features Map Model used for unsupervised training for global to domain mapping.

    This method inherits from :obj:`UFDCombineFeaturesMapPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:class:`~UFDCombineFeaturesMapConfig`): Model configuration class with all parameters required for
                                                    the model. Initializing with a config file does not load
                                                    the weights associated with the model, only the configuration.
                                                    Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.fc = nn.Linear(2 * config.embed_dim, config.embed_dim)
        self.act = F.relu
        self.init_weights()

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input (:ob:'torch.FloatTensor' of shape :obj:'(batch_size, feature_size_of_both_adaptor_global_and_domain)':
                concatenated features from both adaptor global and adaptor domain models.
        """
        return self.fc(self.act(input))


class UFDClassifierPreTrainedModel(PreTrainedModel):
    """
    The UFD Classifier Pre-Trained Model used as base class for derived Classifier Model.

    This model is the abstract super class for the UFD Combine Features Map model which defines the config
    class types and weights initalization method. This class should not be used or instantiated directly,
    see UFDClassifierModel class for usage.
    """

    config_class = UFDClassifierConfig
    base_model_prefix = "UFDClassifier"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.initrange, self.config.initrange)
            module.bias.data.zero_()


class UFDClassifierModel(UFDClassifierPreTrainedModel):
    """
    The UFD Classifier Model used for supervised training for source domain data.

    This method inherits from :obj:`UFDClassifierPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:class:`~UFDClassifierConfig`): Model configuration class with all parameters required for
                                                    the model. Initializing with a config file does not load
                                                    the weights associated with the model, only the configuration.
                                                    Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.fc = nn.Linear(config.embed_dim, config.num_class)
        self.act = F.relu
        self.init_weights()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): features from UFDCombineFeaturesMapModel
            labels (Optional[torch.Tensor], optional): labels for input feature. Defaults to None.


        """
        logits = self.fc(self.act(input_tensor))
        return logits


class UFDMaxDiscriminatorModel(nn.Module):
    """
    Max Discriminator Model used for unsupervised loss functions.
    """

    def __init__(self, hidden_g=1024, initrange=0.1):
        super().__init__()
        self.l0 = nn.Linear(2 * hidden_g, 1)
        self.init_weights(initrange)
        self.act = F.relu
        self.l0.weight.data.uniform_(-initrange, initrange)
        self.l0.bias.data.zero_()

    def init_weights(self, initrange):
        self.l0.weight.data.uniform_(-initrange, initrange)
        self.l0.bias.data.zero_()

    def forward(self, f_g, f_d):
        h = torch.cat((f_g, f_d), dim=1)
        return self.l0(self.act(h))


class UFDMinDiscriminatorModel(nn.Module):
    """
    Min Discriminator Model used for unsupervised loss functions.
    """

    def __init__(self, hidden_l=1024, initrange=0.1):
        super().__init__()
        self.l0 = nn.Linear(2 * hidden_l, 1)
        self.init_weights(initrange)
        self.act = F.relu

    def init_weights(self, initrange):
        self.l0.weight.data.uniform_(-initrange, initrange)
        self.l0.bias.data.zero_()

    def forward(self, f_g, f_d):
        h = torch.cat((f_g, f_d), dim=1)
        h = self.l0(self.act(h))
        return F.normalize(h, p=2, dim=1)


class UFDDeepInfoMaxLossModel(nn.Module):
    """
    Main unsupervised deep info max loss model used for unsupervised training loss functions.
    """

    def __init__(
        self, dim_hidden=1024, initrange=0.1, alpha=0.3, beta=1, gamma=0.2, delta=1
    ):
        super().__init__()
        self.max_d = UFDMaxDiscriminatorModel(dim_hidden, initrange)
        self.min_d = UFDMinDiscriminatorModel(dim_hidden, initrange)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, x, x_n, f_g, fg_n, f_d, fd_n, y_g, y_d, yd_n):
        Ej = -F.softplus(-self.max_d(y_g, f_g)).mean()
        Em = F.softplus(self.max_d(y_g, fg_n)).mean()
        GLOBAL_A = (Em - Ej) * self.alpha

        Ej = -F.softplus(-self.max_d(x, y_g)).mean()
        Em = F.softplus(self.max_d(x_n, y_g)).mean()
        GLOBAL_B = (Em - Ej) * self.delta

        Ej = -F.softplus(-self.min_d(y_d, y_g)).mean()
        Em = F.softplus(self.min_d(yd_n, y_g)).mean()
        Local_B = (Ej - Em) * self.gamma

        return GLOBAL_A + GLOBAL_B + Local_B


class UFDEmbeddingModel(XLMRobertaModel):
    """
    The UFD Embedding Model used for to generate embeddings for model inputs.

    This method inherits from :obj:`XLMRobertaModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:class:`~UFDEmbeddingConfig`): Model configuration class with all parameters required for
                                                    the model. Initializing with a config file does not load
                                                    the weights associated with the model, only the configuration.
                                                    Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config):
        super().__init__(config)


class UFDModel(nn.Module):
    """
    The UFDModel used for running inferences. This model wraps the trained UFDAdaptorDomainModel, UFDAdaptorGlobalModel,
    UFDCombineFeaturesMapModel and the UFDClassifierModel.

    The forward pass method executes a series of forward pass of these warpped models in the sequence defined in the
    paper and research code.
    """

    def __init__(
        self,
        adaptor_domain: UFDAdaptorDomainModel,
        adaptor_global: UFDAdaptorGlobalModel,
        feature_maper: UFDCombineFeaturesMapModel,
        classifier: UFDClassifierModel,
        loss_function: str = "crossentrophyloss",
    ):
        super().__init__()
        self.adaptor_domain = adaptor_domain
        self.adaptor_global = adaptor_global
        self.feature_maper = feature_maper
        self.classifier = classifier
        self.loss_function = loss_function

    def forward(
        self, data_batch: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> UFDModelOutput:
        """
        Args:
            data_batch (torch.Tensor): input tensor batch.
            labels (Optional[torch.Tensor], optional): list of labels. Defaults to None.

        Returns:
            UFDModelOutput: output UFDModelOutput instance with loss and logits.

        Example::
            from sgnlp.models.ufd import (
                UFDModelBuilder,
                UFDPreprocessor
            )
            model_builder = UFDModelBuilder()
            model_grp = model_builder.build_model_group()
            preprocessor = UFDPreprocessor()
            texts = ['dieser film ist wirklich gut!', 'Diese Fortsetzung ist nicht so gut wie die Vorgeschichte']
            text_feature = preprocessor(texts)
            ufd_model_output = model_grp['books_de_dvd'](**text_feature)
        """
        with torch.no_grad():
            global_feat, _ = self.adaptor_global(data_batch)
            domain_feat, _ = self.adaptor_domain(data_batch)
            features = torch.cat((global_feat, domain_feat), dim=1)
            logits = self.classifier(self.feature_maper(features))

            loss = None
            if labels is not None:
                if self.loss_function == "crossentrophyloss":
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1), labels)
            return UFDModelOutput(loss=loss, logits=logits)
