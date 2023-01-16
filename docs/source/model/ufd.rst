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

The base url for the UFD models storage folder is `https://storage.googleapis.com/sgnlp-models/models/ufd`.
To download a specific model set, please append the following pattern to the base
url to directly download the config file or pretrained weights.

    `<SOURCE_DOMAIN>_<TARGET_LANGUAGE>_<TARGET_DOMAIN>_<MODEL_TYPE>`

The following are the available keys for the pattern above:

| <SOURCE_DOMAIN> : `books`, `music`, `dvd`
| <TARGET_LANGUAGE> : `fr`, `de`, `jp`
| <TARGET_DOMAIN> : `books`, `music`, `dvd`
| <MODEL_TYPE> : `adaptor_domain`, `adaptor_global`, `maper`, `classifier`

.. note::

   *adaptor_domain* is the domain specific model, *adaptor_global* is the
   domain invariant model, `maper` is the combine features mapper model and
   `classifier` is the classifier model.*


Example:
To download the model set for 'books' source domain, German target language and
the 'dvd' target domain, use the following url:

| For domain specific model:
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_adaptor_domain/config.json`.
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_adaptor_domain/pytorch_model.bin`.

| For domain invariant model:
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_adaptor_global/config.json`.
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_adaptor_global/pytorch_model.bin`.

| For combine features maper model:
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_maper/config.json`.
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_maper/pytorch_model.bin`.

| For classifier model:
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_classifier/config.json`.
| `https://storage.googleapis.com/sgnlp-models/models/ufd/books_de_dvd_classifier/pytorch_model.bin`.

.. note::

   In keeping with HuggingFace pretrained models implementation. Each model
   consists of a `config.json` file and the `pytorch_model.bin` model weights file.


Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First import the neccessary modules/classes

.. code:: python

   from sgnlp.models.ufd import (
       UFDModelBuilder,
       UFDPreprocessor)

Next instantiate instances of the imported classes.
For the purpose of this tutorial, default arguments are used, for options available,
please refer to the API documentations.

.. code:: python

   model_builder = UFDModelBuilder()
   preprocessor = UFDPreprocessor()

.. note::

   By default, :class:`~sgnlp.models.ufd.model_builder.UFDModelBuilder` will include all available pretrained models,
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
:class:`~sgnlp.models.ufd.modeling.UFDModel`
with the model grouping as keys. Each keys are formed via concatenating the
source domain key, the target language key and the target domain key seperated
via an underscore. (i.e. ``books_de_dvd`` for model group trained on English language ``books``
domain dataset and is the best performing model when evalulated on the German
``de`` target language and ``dvd`` target domain dataset.)

Next run the inference step with raw input text by accessing the desired model group via the dictionary key.
The output is a :class:`~sgnlp.models.ufd.modeling.UFDModelOutput`  type which contains the optional ``loss`` value and the ``logits``.

.. code:: python

   text = ['Wolverine is BACK Der Film ist im Grunde wie alle Teile der X-Men für Comic-Fans auf jeden Fall ein muss. \
            Hugh Jackman spielt seine Rolle wie immer so gut was ich von den ein oder anderen Darsteller leider nicht \
            sagen kann. Story und Action sind aber genug Gründe um sich die Blu-ray zu kaufen.']
   text_feature = preprocessor(text)
   output = model_group['books_de_dvd'](**text_feature)
   # UFDModelOutput(loss=None, logits=tensor([[-1.1018,  0.0944]]))

Full starter code is as follows,

.. code:: python

    from sgnlp.models.ufd import (
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

The input data to the :class:`~sgnlp.models.ufd.preprocess.UFDPreprocessor`
is a list of strings of the target language and target domain. The keys to the
model groups should match the input data target language and target domain,
as well as the desired source domain.


Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output from the model is a :class:`~sgnlp.models.ufd.modeling.UFDModelOutput`
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

Aspect of the training could be configure via the `ufd_config.json` file. An
example of the config file can be found
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/ufd/config/ufd_config.json>`_

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
| embedding_model_name                     | Name of HuggingFace model used for embedding model.                                  |
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
| train_args/train_cache_filename          | Optional, filename for the cache pickled after the train dataset processing.         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_args/val_cache_filename            | Optional, filename for the cache pickled after the val dataset processing.           |
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

    from sgnlp.models.ufd.utils import parse_args_and_load_config
    from sgnlp.models.ufd.train import train
    cfg = parse_args_and_load_config('config/ufd_config.json')
    train(cfg)

Evaluating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------

Refer to training section above for dataset example.


Config Preparation
------------------

Aspect of the evaluation could be configure via the `ufd_config.json` file. An
example of the config file can be found
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/ufd/config/ufd_config.json>`_


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
| embedding_model_name      | Name of HuggingFace model used for embedding model.                       |
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
| eval_args/source_language | Key for the dataset source language.                                      |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/source_domains  | List of keys for the dataset source domains.                              |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/target_languages| List of keys for the dataset target languages.                            |
+---------------------------+---------------------------------------------------------------------------+
| eval_args/target_domains  | List of keys for the dataset target domains.                              |
+---------------------------+---------------------------------------------------------------------------+


Running Evaluation Code
---------------------------
To start UFD evaluation, execute the following code,

.. code:: python

    from sgnlp.models.ufd import parse_args_and_load_config
    from sgnlp.models.ufd import evaluate
    cfg = parse_args_and_load_config('config/ufd_config.json')
    evaluate(cfg)

Using custom dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overview
---------------------------

In this example, we'll demonstrate how to train and evaluate the UFD model on a
custom dataset.

We will be using a Bengali drama reviews dataset. The dataset can be found
`here <https://github.com/sazzadcsedu/BN-Dataset>`_. Assume that we only have a
small amount of labelled data and we would like to train a sentiment analysis
model on the Bengali drama review dataset. Instead of using a pretrained model
and fine-tuning it on this small dataset, we could make use of dataset in
another language and domain to train this model.

As English dataset is easily obtainable, we could use English language
as the source language to train this model. For this example, we will use
the English data published by the authors of the UFD paper, which can be found
`here <https://github.com/lijuntaopku/UFD/tree/main/data>`_ We will need 2
datasets in English.

   1. **Labelled data in source language and source domain**: Eg. A labelled English dataset on music reviews
   2. **Unlabelled data in source language and multiple domains, including target domain**: Eg. An unlabelled English dataset of books, movies and drama reviews

We will also leave a small portion of the labelled Bengali data as validation
set during training. Eg. 100 instances of labelled data. The remaining labelled
Bengali dataset will be used as test set during evaluation

File structure
--------------
Here is the file structure for the example:

.. code::

   .
   ├── config
   │   ├── ufd_config_evaluate.json
   │   └── ufd_config_train.json
   ├── data
   │   ├── bengali
   │   │   └── drama
   │   │       ├── test.txt
   │   │       └── val.txt
   │   ├── en
   │   │   └── books
   │   │       └── train.txt
   │   └── raw_unlabelled.txt
   ├── evaluate.py
   └── train.py

For our example, we will need the following dataset:

   1. **Labelled data in source language and source domain (training)**: A labelled English dataset on music reviews. Eg. *train.txt*
   2. **Unlablled data in source language across multiple domain, including target domain (training)**: An unlabelled English dataset across DVD reviews, books review and music reviews. DVD domain is treated as an approximation of the target domain, since they are similar.  Eg. *raw_unlabelled.txt*
   3. **Labelled data in target language and target domain for validation (training)**: A labelled Bengali dataset on drama reviews. Eg. *val.txt*
   4. **Labelled data in target language and target domain as test set (evaluating)**: A labelled Bengali dataset on drama reviews. Eg. *test.txt*

Dataset Preparation
--------------------

The labelled data needs to be in *.txt* format where the labels are separated from
the text with a *tab*. Here are examples of how the dataset needs to look like:

1. A labelled English music reviews dataset, where the labels are separated from the text with a `tab`

.. code::

   0	Calvino could have written better stuff This book says nothing. He brings you on a journey through nothing that will you unfilled. Don't read it
   1	Fascinating I hightly recommend this book. An easy, quick read that could change your life
   0	its over guys This is a kids book. First few had my attention but since then its getting worse with each book.
   1	Excellent! I absolutely loved this sequel to Something Borrowed. Just as good as the first in my opinion.
   0	not good science while I appreciate what Tipler was attempting to accomplish, he fails miserabley both from a theological and a scientific perspective. skip this one!

2. An unlabelled English dataset across music, books and DVD domain. (DVD domain is used as an approximation of the target domain):

.. code::

   Spiritually and mentally inspiring! A book that allows you to question your morals and will help you discover who you really are!
   This is one my must have books. It is a masterpiece of spirituality. I'll be the first to admit, its literary quality isn't much. It is rather simplistically written, but the message behind it is so powerful that you have to read it. It will take you to enlightenment.
   This book provides a reflection that you can apply to your own life.And, a way for you to try and assess whether you are truly doing the right thing and making the most of your short time on this plane.
   I first read THE PROPHET in college back in the 60's. The book had a revival as did anything metaphysical in the turbulent 60's. It had a profound effect on me and became a book I always took with me. After graduation I joined the Peace Corps and during stressful training in country (Liberia) at times of illness and the night before I left, this book gave me great comfort. I read it before I married, just before and again after my children were born and again after two near fatal illnesses. I am always amazed that there is a chapter that reaches out to you, grabs you and offers both comfort and hope for the future.Gibran offers timeless insights and love with each word. I think that we as a nation should read AND learn the lessons here. It is definitely a time for thought and reflection this book could guide us through.
   A timeless classic.  It is a very demanding and assuming title, but Gibran backs it up with some excellent style and content.  If he had the means to publish it a century or two earlier, he could have inspired a new religion.From the mouth of an old man about to sail away to a far away destination, we hear the wisdom of life and all important aspects of it.  It is a messege.  A guide book.  A Sufi sermon. Much is put in perspective without any hint of a dogma.  There is much that hints at his birth place, Lebanon where many of the old prophets walked the Earth and where this book project first germinated most likely.Probably becuase it was written in English originally, the writing flows, it is pleasant to read, and the charcoal drawings of the author decorating the pages is a plus.  I loved the cover.

3. Labelled Bengali drama reviews dataset, where the labels are separated from the text with a `tab`. We will need a validation set and test set.

.. code::

   0	ওরে বাবা এসব কি দেখছি বাংলাদেশের নাটকে এসব চলতেছে এখন
   0	ফাল্তু মোশারফ
   1	ফাটা ফাটি সুপার
   1	দারুণ একটা
   1	নিশো ভাই সেরা সেরা

The data folder needs to be named according to some rules for ease of tuning
configuration in the config file when there are multiple languages and domains.

.. code::

   .
   └── data
       ├── bengali
       │   └── drama
       │       ├── test.txt
       │       └── val.txt
       ├── en
       │   └── books
       │       └── train.txt
       └── raw_unlabelled.txt

Here are the rules:

   1. Level 1 folder should be named with the source language and target languages. Eg. bengali
   2. Level 2 folder should be named with the source domain under source language folders or target domain under target language folder. Use the same name if there are same domains across different languages
   3. Level 3 files should be named consistently across source languages or target languages. For example, if there are multiple source domains (eg. books and music), all the training data in the source language should be named as *train.txt*. On the other hand, if there are multiple target domains, all the validation data across the target domains should be named as *val.txt* while all the test data across the target domains should be named as *test.txt*


Training
-----------------

First, we will need to create the config file for training on the data. We will
use the default config modified with the dataset that we are using.

.. note::

   Note that the source language, source domain, target language and target domain
   in the config needs to be same as the name of the folders.

Here is the config file that we will be using for training, *ufd_config_train.json*:

.. code::

   {
      "verbose": false,
      "device": "cuda",
      "data_folder": "data/",
      "model_folder": "model/",
      "cache_folder": "cache/",
      "embedding_model_name": "xlm-roberta-large",
      "use_wandb": false,
      "train_args": {
         "unsupervised_dataset_filename": "raw_unlabelled.txt",
         "train_filename": "train.txt",
         "val_filename": "val.txt",
         "train_cache_filename": "train_dataset.pickle",
         "val_cache_filename": "val_dataset.pickle",
         "learning_rate": 0.00001,
         "seed": 0,
         "unsupervised_model_batch_size": 16,
         "unsupervised_epochs": 30,
         "in_dim": 1024,
         "dim_hidden": 1024,
         "out_dim": 1024,
         "initrange": 0.1,
         "classifier_epochs": 60,
         "classifier_batch_size": 16,
         "num_class": 2,
         "source_language": "en",
         "source_domains": ["books"],
         "target_domains": ["drama"],
         "target_languages": ["bengali"],
         "warmup_epochs": 5
      },
   }

We will then call the train function on this config in *train.py*:

.. code:: python

    from sgnlp.models.ufd import parse_args_and_load_config
    from sgnlp.models.ufd import train
    cfg = parse_args_and_load_config('config/ufd_config_train.json')
    train(cfg)


Evaluating
------------------

To evaluate, we will also first create the config file for evaluation then we
will call the evaluate function. Here is the *ufd_config_evaluate.json*:

.. code::

   {
      "verbose": false,
      "device": "cuda",
      "data_folder": "data/",
      "model_folder": "model/",
      "cache_folder": "cache/",
      "embedding_model_name": "xlm-roberta-large",
      "use_wandb": false,
      "eval_args":{
         "result_folder": "result/",
         "result_filename": "results.log",
         "test_filename": "test.txt",
         "eval_batch_size": 8,
         "config_filename": "config.json",
         "model_filename": "pytorch_model.bin",
         "source_language": "en",
         "source_domains": ["drama"],
         "target_domains": ["books"],
         "target_languages": ["bengali"]
      }
   }

Here is *evaluate.py*:

.. code:: python

    from sgnlp.models.ufd import parse_args_and_load_config
    from sgnlp.models.ufd import evaluate
    cfg = parse_args_and_load_config('config/ufd_config_evaluate.json')
    evaluate(cfg)

Using multiple languages and domain
-------------------------------------

As it takes abit of experimenting to identify the optimal language and domain
that gives the the best result for the model, we have designed the code to
allow experimenting with multiple languages and domains in a single config file.

For example, if we have found another English labelled dataset on music which we
also want to try out, we can add the music dataset to the data folder. The new
project folder structure will be as such:

.. code::

   .
   .
   ├── data
   .   .
   .   .
   │   ├── en
   │   │   ├── books
   │   │   │   └── train.txt
   │   │   └── music
   │   │       └── train.txt
   │   └── raw_unlabelled.txt
   .
   .

In the config file for both train and evaluate, we will add an additional
music element to the source_domains as such:

.. code::

   {
      ...
      "train_args": {
         ...
         "source_domains": ["books", "music"],
         ...
      },
   }

The same changes can be made if we want to experiment on multiple target
language and target domains



