# Contributing Guide

Thank you for considering contributing to SG-NLP! We believe that SG-NLP's future is closely tied with the community and community contributions will help SG-NLP to grow faster.

## How can I contribute?

- [Add a new model to `sgnlp`](#adding-a-new-model-to-sgnlp)
  - [Pre-requisites](#pre-requisites)
  - [Required Components](#required-components)
  - [Config](#config)
  - [Preprocess](#preprocess)
  - [Tokenizer](#tokenizer-optional)
  - [Modeling](#modeling)
  - [Train](#train)
  - [Eval](#eval)
  - [Logging](#logging)
  - [Utils](#utils)
  - [README](#readme)
  - [Model weights and artefacts](#model-weights-and-artefacts)
- [Submit Bug Fixes](#submitting-bug-fixes)
- [Add Documentation](#adding-documentation)

## Adding a new model to `sgnlp`

### Pre-requisites

1. Fork the `sgnlp` repository.
2. Create a Python virtual environment (verison >= 3.8)
3. Install the packages in `requirements_dev.txt` at the root of the repository using `pip install -r requirements_dev.txt`. This will install `black`, `flake8`, and `pytest` which we use for code formatting and testing.

### Required Components

Before you create a pull request to add your model to `sgnlp`, please ensure that you have the following components ready.

- Python scripts / code comprising of (we'll go into more detail below)
  - [config.py](#config)
  - [modeling.py](#modeling)
  - [preprocess.py](#preprocess)
  - [train.py](#train)
  - [eval.py](#eval)
  - [utils.py](#utils) (optional)
  - [README](#readme)
  - [requirements.txt](#requirements.txt) (discouraged)
- Model information (to be included in the README)
  - Original paper / source
  - Datasets and/or how to obtain them
  - Evaluation metrics
  - Model size
  - Training information
- Model weights and artefacts (to be submitted separately)
  - pytorch_model.bin
  - config.json
  - tokenizer_config.json (optional)

To contribute a model, add a folder for the model at `sgnlp/sgnlp/models/<model_name>`. The following components are required within this folder.

| Folder / File                             | Description                                                                                                                                                                                                                                     |
| :---------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _<model_name>_                            | Folder containing the modeling, preprocess, config, train, and eval scripts.                                                                                                                                                                    |
| _<model_name>/config_                     | Folder containing the JSON configuration files used for the train and eval scripts.                                                                                                                                                             |
| _<model_name>/config.py_                  | Script containing the model config class which inherits from HuggingFace's [`PretrainedConfig`](https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig) or its family of derived classes.            |
| _<model_name>/eval.py_                    | Script containing code to evaluate the model performance.                                                                                                                                                                                       |
| _<model_name>/modeling.py_                | Script containing the model class which inherits from HuggingFace's [`PretrainedModel`](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel) class or its family of derived classes.                       |
| _<model_name>/preprocess.py_              | Script containing code to preprocess the input text into the model's required input tensors.                                                                                                                                                    |
| _<model_name>/tokenization.py_ (optional) | Script containing the model tokenizer class which inherits from HuggingFace's [`PretrainedTokenizer`](https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer) class or its family of derived classes. |
| _<model_name>/train.py_                   | Script containing code to train the model. It is recommended to utilize the [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) class from HuggingFace.                                             |
| _<model_name>/README.md_                  | README markdown file containing model information such as model source, architecture, evaluation datasets and metrics, model size, and training information.                                                                                    |

To manage the number of dependencies installed with `sgnlp`, contributors are strongly recommended to limit their code to use the packages listed in `setup.py`. If additional dependencies are required, please introduce a check in the `__init__.py` at `sgnlp/sgnlp/models/<model_name>/__init__.py`. For example, the Latent Structure Refinement model for Relation Extraction requires the `networkx` package. The code snippet from LSR's `__init__.py` checks if `networkx` is installed when the model is imported. Users will have to install these additional dependencies separately.

```python

from ...utils.requirements import check_requirements

requirements = ["networkx"]
check_requirements(requirements)
```

### Config

Model configs contain model architecture information. Typically, this would include hyperparameters for the different layers within the model as well as the loss function. Model configs should inherit from the `PretrainedConfig` class from the `transformers` package. The following is an example from the Cross Lingual Cross Domain Sentiment Analysis model.

```python

from transformers import PretrainedConfig

class UFDClassifierConfig(PretrainedConfig):

    model_type = "classifier"

    def __init__(self, embed_dim=1024, num_class=2, initrange=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.initrange = initrange
```

For models that use or adapt pre-trained configs already available in the `transformers` package, the model config should inherit from the pre-trained config class instead. For example, this model config inherits from `BertConfig` which is a child class of `PretrainedConfig`.

```python

from transformers import BertConfig

class NewModelConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

### Preprocess

The `preprocess.py` script and its associated preprocessor class is an addition in `sgnlp`. When implementing various models, the team found that some models required more complex preprocessing. For example, some NLP models take in multiple different text inputs (ie, different utterances, multiple tweets, a single question and multiple possible answers, etc) which require different preprocessing steps. The `preprocess.py` and the preprocessor class is the team's solution to packaging all of these different steps into a single and consistent (across different models) step.

The preprocessor class inherits from the default `object` class. All preprocessing steps should be executed in the class' `__call__` method. The `__call__` method should return a dictionary containing all the necessary input tensors required by the model. The following code snippet illustrates the `__call__` method from the RECCON Span Extraction model's `RecconSpanExtractionPreprocessor`.

```python

class RecconSpanExtractionPreprocessor:

    def __call__(
        self, data_batch: Dict[str, List[str]]
    ) -> Tuple[
        BatchEncoding,
        List[Dict[str, Union[int, str]]],
        List[SquadExample],
        List[SquadFeatures],
    ]:
        self._check_values_len(data_batch)
        concatenated_batch, evidences = self._concatenate_batch(data_batch)
        dataset, examples, features = load_examples(
            concatenated_batch, self.tokenizer, evaluate=True, output_examples=True
        )

        input_ids = [torch.unsqueeze(instance[0], 0) for instance in dataset]
        attention_mask = [torch.unsqueeze(instance[1], 0) for instance in dataset]
        token_type_ids = [torch.unsqueeze(instance[2], 0) for instance in dataset]

        output = {
            "input_ids": torch.cat(input_ids, axis=0),
            "attention_mask": torch.cat(attention_mask, axis=0),
            "token_type_ids": torch.cat(token_type_ids, axis=0),
        }
        output = BatchEncoding(output)

        return output, evidences, examples, features
```

In the RECCON Span Extraction model, `output` is a dictionary with the token ids, attention masks and token type ids for the input utterance. `evidences`, `examples` and `features` are other features required in the RECCON model. The key idea here is to consolidate all the necessary preprocessing steps into a single method to reduce the effort needed to start using the models.

### Tokenizer (optional)

The `tokenizer.py` is optional if the `preprocess.py` already contains a tokenizer. All tokenizers should inherit from the `PreTrainedTokenizer` or `PreTrainedTokenizerFast` classes from the `transformers` package.

```python

from transformers import PreTrainedTokenizer

class NewModelTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

For models that use or adapt pre-trained tokenizers already available in the `transformers` package, the tokenizer should inherit from the pre-trained tokenizer class instead. For example, the RECCON Span Extraction tokenizer inherits from `BertTokenizer` which inherits from `PreTrainedTokenizer`.

```python

from transformers import BertTokenizer

class RecconSpanExtractionTokenizer(BertTokenizer):
    """
    Constructs a Reccon Span Extraction tokenizer, derived from the Bert tokenizer.
    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        do_lower_case (:obj:`bool`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.
    """

    def __init__(self, vocab_file: str, do_lower_case: bool = False, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, do_lower_case=do_lower_case, **kwargs)

```

### Modeling

There are 2 steps required to add a new model class. The first step is to introduce a `NewModelPreTrainedModel` class which handles weights instantiation, downloading and loading pretrained models. This class should inherit from the `PreTrainedModel` class from `transformers`.

The key things to define as the `config_class`, `base_model_prefix` class attributes and `_init_weights` method. The `_init_weights` method dictates how the weights for the different layers are instantiated.

```python

from transformers import PreTrainedModel
from .config import LsrConfig


class LsrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LsrConfig
    base_model_prefix = "lsr"

    def _init_weights(self, module):
        """ Initialize the weights """
        ...
```

Subsequently, the main model class should inherit from this `NewModelPreTrainedModel` class. The main model class contains the code required to execute the model's forward pass.

```python

class RumourDetectionTwitterModel(RumourDetectionTwitterPreTrainedModel):

    def __init__(self, config: RumourDetectionTwitterConfig):
        super().__init__(config)
        self.config = config
        self.wordEncoder = WordEncoder(self.config)
        self.positionEncoderWord = PositionEncoder(config, self.config.max_length)
        self.positionEncoderTime = PositionEncoder(config, self.config.size)
        self.hierarchicalTransformer = HierarchicalTransformer(self.config)
        if config.loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        token_ids: torch.Tensor,
        time_delay_ids: torch.Tensor,
        structure_ids: torch.Tensor,
        token_attention_mask=None,
        post_attention_mask=None,
        labels: Optional[torch.Tensor] = None,
    ):
        X = self.wordEncoder(token_ids)
        word_pos = self.prepare_word_pos(token_ids).to(X.device)
        word_pos = self.positionEncoderWord(word_pos)
        time_delay = self.positionEncoderTime(time_delay_ids)

        logits = self.hierarchicalTransformer(
            X,
            word_pos,
            time_delay,
            structure_ids,
            attention_mask_word=token_attention_mask,
            attention_mask_post=post_attention_mask,
        )

        if labels is not None:
            loss = self.loss(logits, labels)
        else:
            loss = None

        return RumourDetectionTwitterModelOutput(loss=loss, logits=logits)
```

There are 3 key things to note in the above implementation.

1. When initialising the model, it is important to invoke the `init_weights()` method. Note the lack of an underscore at the start of the method name. This is required so that the model weights are initialized using the methods defined in the `__init__` method specified in `NewModelPreTrainedModel`.

2. The `forward` method takes in an optional `labels` argument. If this argument is passed to the model, the `forward` method should also return the value of the loss function for that batch of inputs.

3. The `forward` method's output is an object of the `RumourDetectionTwitterModelOutput` dataclass. This dataclass is illustrated in the code snippet below.

```python

from dataclasses import dataclass
from transformers.file_utils import ModelOutput


@dataclass
class RumourDetectionTwitterModelOutput(ModelOutput):
    """
    Base class for outputs of Rumour Detection models
    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss, typically cross entropy. Loss function used is dependent on what is specified in RumourDetectionTwitterConfig.
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, num_classes)`):
            Raw logits for each class. num_classes = 4 by default.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
```

### train

`train.py` should contain a working implementation of the model training process. A user should be able to train the model from the command line using `python -m train --train_args config/config.json`.

### eval

`eval.py` should contain a working implementing of the model evaluation process. This script should load the trained model (using the information in `config/config.json`) and evaluate it against the evaluation datasets. The evaluation metrics reported should correspond to that reported in the `README.md`.

### utils

`utils.py` should contain other functions which are useful for `train.py` or `eval.py` but do not directly fit within any of the other scripts above.

### logging

When the `sgnlp` package is first imported, a logger with the base name `sgnlp` is setup with a default `StreamHandler` and `NullHandler`.  
For logging within all model scripts mentioned above, contributer should use the `logging.getLogger(__name__)` to obtain a logger specific the to the script and use that logger for logging throughout the script. Log messages will inherit the format setup when `sgnlp` package is first imported.  

```python

import logging
logger = logging.getLevel(__name__)

...

logger.info("Log message here")
# 2022-03-01 12:00:00,000 - INFO - sgnlp.models.sentic_gcn.preprocess - preprocess.py - 10 - Log message here
```

### README

The `README` for the model should provide a concise introduction to the model. The following information are required:

- Citation and link to the original paper or source that introduced the model
- Citation and link to the train, validation and test datasets that the model was trained on. If the model was trained and evaluated on licensed datasets, information should be provided on how the SG-NLP team can obtain access to the evaluation (test) dataset. The train dataset may be omitted.
- Evaluation metrics. Please cite the appropriate paper if the evaluation metric is a published benchmark.
- Model size (in terms of the size of the trained model's weights)
- Training information such as hyperparameter values and compute time and resources used to train the model.

### Model weights and artefacts

Model weights and artefects comprise of:

- Saved model weights. Specifically, the `pytorch_model.bin` file saved using the `save_pretrained` method from the model class implemented in `modeling.py`. For now, only models implemented in `PyTorch` are accepted. The team is looking into accepting models implemented in `TensorFlow` as well.
- Model config. The `config.json` generated when using the `save_pretrained` method from the model config class implemented in `config.py`.
- Any artefacts needed by the tokenizer or preprocessor.

## Submitting Bug Fixes

If you spotted a bug, please follow these steps to report them.

1. Check the issues list to see whether the bug has already been reported. If an issue has already been created, please comment on that issue with details on how to replicate the bug.
2. If there is no issue relevant to the bug, please open a GitHub with
   - Clear description of the bug
   - Information related to your environment (ie, OS, package version, Python version, etc)
   - Steps on how to replicate the bug (ie, a code snippet that we could run to encounter the bug)

## Adding Documentation

One easy way to contribute is to add or refine documentation / docstrings to the models that are currently available. `sgnlp` uses the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for our docstrings. Once the docstrings have been added or edited, please submit a pull request.
