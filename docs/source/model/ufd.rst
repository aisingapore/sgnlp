UFD: Unsupervised Domain Adaptation of a Pretrained Cross-Lingual Language Model
================================================================================

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The UFD model was proposed in `Unsupervised Domain Adaptation of a Pretrained
Cross-Lingual Language Model <https://www.ijcai.org/Proceedings/2020/508>`_
by Juntao Li, Ruidan He, Hai Ye, Hwee Tou Ng, Lidong Bing and Rui Yan.

The abstract from the paper is as follows:

*Recent research indicates that pretraining cross-lingual language models on
large-scale unlabeled texts yields significant performance improvements over
various cross-lingual and low-resource tasks. Through training on one hundred
languages and terabytes of texts, cross-lingual language models have proven to
be effective in leveraging high-resource languages to enhance low-resource
language processing and outperform monolingual models. In this paper, we
further investigate the cross-lingual and cross-domain (CLCD) setting when a
pretrained cross-lingual language model needs to adapt to new domains.
Specifically, we propose a novel unsupervised feature decomposition method that
can automatically extract domain-specific features and domain-invariant features
from the entangled pretrained cross-lingual representations, given unlabeled
raw texts in the source language. Our proposed model leverages mutual
information estimation to decompose the representations computed by a
cross-lingual model into domain-invariant and domain-specific parts.
Experimental results show that our proposed method achieves significant
performance improvements over the state-of-the-art pretrained cross-lingual
language model in the CLCD setting.*

In keeping with how the models performance are calculated in the paper, this
implementation save the best performing model weights in the supervised
training phase for the respective source domain, target language and target
domain.

Default dataset present in the paper consists of one source language (English),
three source domains (Books, Music, and DVD), three target languages (French,
German, Japanese) and three target domains (Books, Music, and DVD). This
combination results in a total of 18 cross lingual cross domain model sets of 4
models each, for a total of 72 models.

Each model set consists of the domain specific model, the domain invariant model,
the combine features maper model and the classifier model.

| Link to the `paper <https://www.ijcai.org/Proceedings/2020/508>`_
| Link to the `dataset <https://github.com/lijuntaopku/UFD/tree/main/data>`_
| Link to the original `github <https://github.com/lijuntaopku/UFD>`_


Pretrained config and weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All 72 pretrained model weights and config are available to download from Azure
Storage.

The base url for the UFD models storage folder is `https://sgnlp.blob.core.windows.net/models/ufd`.
To download a specific model set, please append the following pattern to the base
url to directly download the config file or pretrained weights.

    `<SOURCE_DOMAIN>_<TARGET_LANGUAGE>_<TARGET_DOMAIN>_<MODEL_TYPE>`

The following are the available keys for the pattern above:

| <SOURCE_DOMAIN> : `books`, `music`, `dvd`
| <TARGET_LANGUAGE> : `fr`, `de`, `jp`
| <TARGET_DOMAIN> : `books`, `music`, `dvd`
| <MODEL_TYPE> : `adaptor_domain`, `adaptor_global`, `maper`, `classifier`

*Note: `adaptor_domain` is the domain specific model, `adaptor_global` is the
domain invariant model, `maper` is the combine features mapper model and
`classifier` is the classifier model.*


Example:
To download the model set for 'books' source domain, German target language and
the 'dvd' target domain, use the following url:

| For domain specific model:
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_adaptor_domain/config.json`.
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_adaptor_domain/pytorch_model.bin`.

| For domain invariant model:
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_adaptor_global/config.json`.
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_adaptor_global/pytorch_model.bin`.

| For combine features maper model:
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_maper/config.json`.
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_maper/pytorch_model.bin`.

| For classifier model:
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_classifier/config.json`.
| `https://sgnlp.blob.core.windows.net/models/ufd/books_de_dvd_classifier/pytorch_model.bin`.

*Note: In keeping with HuggingFace pretrained models implementation. Each model
consists of a `config.json` file and the `pytorch_model.bin` model weights file.*


Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First import the neccessary modules/classes

.. code:: python

   from sgnlp_models.models.ufd import (
       UFDModelBuilder,
       UFDPreprocessor)

Next instantiate instances of the imported classes.
For the purpose of this tutorial, default arguments are used, for options available,
please refer to the API documentations.

.. code:: python

   model_builder = UFDModelBuilder()
   preprocessor = UFDPreprocessor()

Note: By default, :class:`~sgnlp_models.models.ufd.model_builder.UFDModelBuilder` will include all available pretrained models,
to target only specific model set, simply define the `source_domains`,
`target_languages` and `target_domains` input arguments.
The following shows an example for a single model set for the `books` source
domains, `German` target language and `dvd` target domain.

.. code:: python

   model_builder = UFDModelBuilder(source_domains=['books'], target_languages=['de'], target_domains=['dvd'])
   preprocessor = UFDPreprocessor()

Next step is to build the default model groups. This will download all
pretrained config and model weights from Azure storage.
Using default arguments, a total of 72 pretrained config and model weights will
be downloaded to form a total of 18 model groups consisting of 4 models
(adaptor domain model, adaptor global model, combine features maper model and classifier model)
per model group.

.. code:: python

   model_groups = model_builder.build_model_group()

The ``build_model_group()`` method call will return a dictionary of pretained
:class:`~sgnlp_models.models.ufd.modeling.UFDModel`
with the model grouping as keys. Each keys are formed via concatenating the
source domain key, the target language key and the target domain key seperated
via an underscore. (i.e. ``books_de_dvd`` for model group trained on English language ``books``
domain dataset and is the best performing model when evalulated on the German
``de`` target language and ``dvd`` target domain dataset.)

Next run the inference step with raw input text by accessing the desired model group via the dictionary key.
The output is a :class:`~sgnlp_models.models.ufd.modeling.UFDModelOutput`  type which contains the optional ``loss`` value and the ``logits``.

.. code:: python

   text = ['Wolverine is BACK Der Film ist im Grunde wie alle Teile der X-Men für Comic-Fans auf jeden Fall ein muss. \
            Hugh Jackman spielt seine Rolle wie immer so gut was ich von den ein oder anderen Darsteller leider nicht \
            sagen kann. Story und Action sind aber genug Gründe um sich die Blu-ray zu kaufen.']
   text_feature = preprocessor(text)
   output = model_group['books_de_dvd'](**text_feature)
   # UFDModelOutput(loss=None, logits=tensor([[-1.1018,  0.0944]]))

Full starter code is as follows,

.. code:: python

    from sgnlp_models.models.ufd import (
       UFDModelBuilder,
       UFDPreprocessor)
    import torch
    import torch.nn.functional as F

    model_builder = UFDModelBuilder()
    preprocessor = UFDPreprocessor()

    model_groups = model_builder.build_model_group()

    text = ['Wolverine is BACK Der Film ist im Grunde wie alle Teile der X-Men für Comic-Fans auf jeden Fall ein muss. \
            Hugh Jackman spielt seine Rolle wie immer so gut was ich von den ein oder anderen Darsteller leider nicht \
            sagen kann. Story und Action sind aber genug Gründe um sich die Blu-ray zu kaufen.']
    text_feature = preprocessor(text)
    output = model_group['books_de_dvd'](**text_feature)
    # UFDModelOutput(loss=None, logits=tensor([[-1.1018,  0.0944]]))

    logits_probabilities = F.softmax(output.logits, dim=1)
    max_output = torch.max(logits_probabilities, axis=1)
    probabilities = max_output.values.item()
    sentiments = max_output.indices.item()


Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input data to the :class:`~sgnlp_models.models.ufd.preprocess.UFDPreprocessor`
is a list of strings of the target language and target domain. The keys to the
model groups should match the input data target language and target domain,
as well as the desired source domain.


Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output from the model is a :class:`~sgnlp_models.models.ufd.modeling.UFDModelOutput`
object which containers the `logits` and optional `loss` value. For probability
and sentiment of the output, pass the `logits` thru a softmax function and get
the max value, the index of the max value represents the sentiment.


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------

Dataset consists of unlabeled text of the source language for the unsupervised training phase and text with labels of both
the source and target languages split into their respective domains.

| Link to example of `unlabeled dataset <https://github.com/lijuntaopku/UFD/blob/main/data/raw.0.6.txt>`_
| Link to example of `labeled dataset <https://github.com/lijuntaopku/UFD/tree/main/data/en>`_

Config Preparation
------------------

Aspect of the training could be configure via the `ufd_config.json` file.

+------------------------------------------+--------------------------------------------------------------------------------------+
| Configuration key                        | Description                                                                          |
+==========================================+======================================================================================+
| verbose                                  | Enable verbose logging messages.                                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+
| device                                   | Pytorch device type to set for training.                                             |
+------------------------------------------+--------------------------------------------------------------------------------------+
| data_folder                              | Folder path to dataset.                                                              |
+------------------------------------------+--------------------------------------------------------------------------------------+
| model_folder                             | Folder path to model weights.                                                        |
+------------------------------------------+--------------------------------------------------------------------------------------+
| cache_folder                             | Folder path for caching.                                                             |
+------------------------------------------+--------------------------------------------------------------------------------------+
| use_wandb                                | Use weight and biases for training logs.                                             |
+------------------------------------------+--------------------------------------------------------------------------------------+
| wandb_config/project                     | Project name for wandb.                                                              |
+------------------------------------------+--------------------------------------------------------------------------------------+
| wandb_config/tags                        | Tags label for wandb.                                                                |
+------------------------------------------+--------------------------------------------------------------------------------------+
| wandb_config/name                        | Name of a specific train run. To be updated for each different train run.            |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/unsupervised_dataset_filename | Filename to dataset file for unsupervised training.                                  |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/train_filename                | Filename for the train dataset file.                                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/val_filename                  | Filename for the validation dataset file.                                            |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/train_cache_filename          | Filename for the cache pickled after the train dataset processing.                   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/val_cache_filename            | Filename for the cache pickled after the val dataset processing.                     |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/learning_rate                 | Learning rate used for training.                                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/seed                          | Random seed number.                                                                  |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/unsupervised_model_batch_size | Batch size to use for the unsupervised training.                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/unsupervised_epochs           | Number of epochs to train for unsupervised training.                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/in_dim                        | Number of neurons for first linear layer for adaptor_domain, adaptor_global model.   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/dim_hidden                    | Number of neurons for hidden linear layer for adaptor_domain, adaptor_global model.  |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/out_dim                       | Number of neurons for last linear layer for adaptor_domain, adaptor_global model.    |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/initrange                     | Range to initialize weigths for all models.                                          |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/classifier_epochs             | Number of epochs to train for classifier training.                                   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/classifier_batch_size         | Batch size to use for the classifier training.                                       |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/num_class                     | Number of classes for sentiment analysis, set as output neurons of classifier model. |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/embedding_model_name          | Name of HuggingFace model used for embedding model.                                  |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/source_language               | Key for the dataset source language.                                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/source_domains                | List of keys for the dataset source domains.                                         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/target_languages              | List of keys for the dataset target languages.                                       |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/target_domains                | List of keys for the dataset target domains.                                         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/warmup_epochs                 | Number of warmup epochs for classifier training.                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+


Running Train Code
----------------------
To start UFD training, execute the follow code,

.. code:: python

    from sgnlp_models.models.ufd.utils import parse_args_and_load_config
    from sgnlp_models.models.ufd.train import train
    cfg = parse_args_and_load_config
    train(cfg)


Evaluating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------

Refer to training section above for dataset example.


Config Preparation
------------------

Aspect of the evaluation could be configure via the `ufd_config.json` file.

+---------------------------+---------------------------------------------------------------------------+
| Configuration key         | Description                                                               |
+===========================+===========================================================================+
| verbose                   | Enable verbose logging messages.                                          |
+---------------------------+---------------------------------------------------------------------------+
| device                    | Pytorch device type to set for evaluation.                                |
+---------------------------+---------------------------------------------------------------------------+
| data_folder               | Folder path to dataset.                                                   |
+---------------------------+---------------------------------------------------------------------------+
| model_folder              | Folder path to model weights.                                             |
+---------------------------+---------------------------------------------------------------------------+
| cache_folder              | Folder path for caching.                                                  |
+---------------------------+---------------------------------------------------------------------------+
| use_wandb                 | Use weight and biases for training logs.                                  |
+---------------------------+---------------------------------------------------------------------------+
| wandb_config/project      | Project name for wandb.                                                   |
+---------------------------+---------------------------------------------------------------------------+
| wandb_config/tags         | Tags label for wandb.                                                     |
+---------------------------+---------------------------------------------------------------------------+
| wandb_config/name         | Name of a specific train run. To be updated for each different train run. |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/result_folder   | Folder path to save evaluation results.                                   |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/result_filename | Filename of text file to save evaluation results.                         |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/test_filename   | Filename of test dataset.                                                 |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/eval_batch_size | Batch size to use for evaluation.                                         |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/config_filename | Filename of pretrained HuggingFace UFD config file.                       |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/model_filename  | Filename of pretrained HuggingFace UFD model weights.                     |
+---------------------------+---------------------------------------------------------------------------+


Running Evaluation Code
---------------------------
To start UFD evaluation, execute the following code,

.. code:: python

    from sgnlp_models.models.ufd import parse_args_and_load_config
    from sgnlp_models.models.ufd import evaluate
    cfg = parse_args_and_load_config('config/ufd_config.json')
    evaluate(cfg)
