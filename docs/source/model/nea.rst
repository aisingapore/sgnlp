NEA: A Neural Approach to Automated Essay Scoring
=================================================

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The NEA model was proposed in `A Neural Approach to Automated Essay Scoring
<https://aclanthology.org/D16-1193/>`_ by Kaveh Taghipour and Hwee Tou Ng.

The abstract from the paper is as follows:

*Traditional automated essay scoring systems rely on carefully designed features
to evaluate and score essays. The performance of such systems is tightly bound
to the quality of the underlying features. However, it is laborious to manually
design the most informative features for such a system. In this paper, we
develop an approach based on recurrent neural networks to learn the relation
between an essay and its assigned score, without any feature engineering.
We explore several neural network models for the task of automated essay
scoring and perform some analysis to get some insights of the models.
The results show that our best system, which is based on long short-term memory
networks, outperforms a strong baseline by 5.6% in terms of quadratic weighted
Kappa, without requiring any feature engineering.*

| Link to the `paper <https://aclanthology.org/D16-1193/>`_
| Link to the `dataset <https://github.com/nusnlp/nea/tree/master/data>`_
| Link to the original `github <https://github.com/nusnlp/nea>`_

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The pretrained model can be downloaded and accessed as follows:

.. code:: python

    from sgnlp_models.models.nea import (
        NEAArguments,
        NEAPreprocessor,
        NEAConfig,
        NEARegPoolingModel,
        NEATokenizer,
        download_tokenizer_files_from_azure)
    from sgnlp_models.models.nea.utils import convert_to_dataset_friendly_scores

    # Download tokenizer files from azure
    cfg = NEAArguments()
    download_tokenizer_files_from_azure(cfg)

    # Load model and preprocessor
    config = NEAConfig.from_pretrained('https://sgnlp.blob.core.windows.net/models/nea/config.json')
    model = NEARegPoolingModel.from_pretrained('https://sgnlp.blob.core.windows.net/models/nea/pytorch_model.bin',
                                                    config=config)
    tokenizer = NEATokenizer.from_pretrained(cfg.tokenizer_args["save_folder"])
    preprocessor = NEAPreprocessor(tokenizer=tokenizer)

    # Model predict
    instance = """Dear @CAPS1 @CAPS2, I believe that using computers will benefit us in many ways like talking and becoming
                friends will others through websites like facebook and mysace. Using computers can help us find
                coordibates, locations, and able ourselfs to millions of information. Also computers will benefit us by
                helping with jobs as in planning a house plan and typing a @NUM1 page report for one of our jobs in less
                than writing it. Now lets go into the wonder world of technology. Using a computer will help us in life by
                talking or making friends on line. Many people have myspace, facebooks, aim, these all benefit us by
                having conversations with one another. Many people believe computers are bad but how can you make friends
                if you can never talk to them? I am very fortunate for having a computer that can help with not only
                school work but my social life and how I make friends. Computers help us with finding our locations,
                coordibates and millions of information online. If we didn't go on the internet a lot we wouldn't know how
                to go onto websites that @MONTH1 help us with locations and coordinates like @LOCATION1. Would you rather
                use a computer or be in @LOCATION3. When your supposed to be vacationing in @LOCATION2. Million of
                information is found on the internet. You can as almost every question and a computer will have it. Would
                you rather easily draw up a house plan on the computers or take @NUM1 hours doing one by hand with ugly
                erazer marks all over it, you are garrenteed that to find a job with a drawing like that. Also when
                appling for a job many workers must write very long papers like a @NUM3 word essay on why this job fits
                you the most, and many people I know don't like writing @NUM3 words non-stopp for hours when it could take
                them I hav an a computer. That is why computers we needed a lot now adays. I hope this essay has impacted
                your descion on computers because they are great machines to work with. The other day I showed my mom how
                to use a computer and she said it was the greatest invention sense sliced bread! Now go out and buy a
                computer to help you chat online with friends, find locations and millions of information on one click of
                the button and help your self with getting a job with neat, prepared, printed work that your boss will
                love."""

    tokens = preprocessor([instance])
    output = model(**tokens)
    # NEAModelOutput(loss=None, logits=tensor([[0.8115]], grad_fn=<SigmoidBackward>))

    score = int(convert_to_dataset_friendly_scores(output.logits.detach().numpy(), cfg.preprocess_data_args.prompt_id))
    # 10

Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input data to the :class:`~sgnlp_models.models.nea.preprocess.NEAPreprocessor` is a list of strings.


Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output from the model is a :class:`~sgnlp_models.models.nea.modeling.NEAModelOutput`
object which contains the `logits` and optional `loss` value. To obtain a score
as define in the paper, pass the `logits` and `prompt_id` to the
`convert_to_dataset_friendly_scores` function.


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------

Dataset preparation per the original code for NEA includes a raw dataset processing step as follows,
this step can be skipped if the dataset is already prepared.


| 1) First download the data from the `github <https://github.com/nusnlp/nea/tree/master/data>`_ to your local project directory.
| 2) Next download the raw dataset (training_set_rel3.tsv) from `Kaggle <https://www.kaggle.com/c/asap-aes/data>`_. to your local project directory.
| 3) Next update the `preprocess_data_args` section of the `nea_config.json` file.
| 4) Lastly execute the `preprocess_raw_dataset.py` script.

| Link to original instruction for `dataset preparation <https://github.com/nusnlp/nea>`_
| Link to dataset `starter file <https://github.com/nusnlp/nea/tree/master/data>`_

Config Preparation
------------------

Aspect of the training could be configure via the `nea_config.json` file.

+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration key                                     | Description                                                                                                                                                                 |
+=======================================================+=============================================================================================================================================================================+
| use_wandb                                             | Use weight and biases for training logs.                                                                                                                                    |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| wandb_config/project                                  | Project name for wandb.                                                                                                                                                     |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| wandb_config/tags                                     | Tags label for wandb.                                                                                                                                                       |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| wandb_config/name                                     | Name of a specific train run. To be updated for each different train run.                                                                                                   |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_type                                            | NEA model type to use for training.                                                                                                                                         |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| emb_path                                              | File path to the embedding file. Refer to `original github <https://github.com/nusnlp/nea/blob/master/FAQ.md>`_ for reference.                                              |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/train_path                       | File path to the train dataset file.                                                                                                                                        |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/dev_path                         | File path to the dev dataset file.                                                                                                                                          |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/test_path                        | File path to the test dataset file.                                                                                                                                         |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/prompt_id                        | Prompt ID to filter from dataset for training.                                                                                                                              |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/maxlen                           | Maximum allowed number of words during training.                                                                                                                            |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/to_lower                         | Flag to indicate if dataset should be set to lower case.                                                                                                                    |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/score_index                      | Score index to use for scoring predictions.                                                                                                                                 |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/azure_path                             | Root directory path to Azure storage for NEA files.                                                                                                                         |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/files                                  | Files name for tokenizers files required to construct NEATokenizer.                                                                                                         |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/vocab_train_file                       | File path to vocab file for training NEATokenizer.                                                                                                                          |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/save_folder                            | Folder path to save downloaded tokenizer files from Azure storage.                                                                                                          |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_raw_dataset_args/data_folder               | Folder path to raw dataset.                                                                                                                                                 |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_raw_dataset_args/input_file                | File name of raw dataset file in `data_folder`.                                                                                                                             |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_embedding_args/raw_embedding_file          | File name of raw embeddings file.                                                                                                                                           |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_embedding_args/preprocessed_embedding_file | File name of preprocessed embeddings.                                                                                                                                       |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| train_args                                            | For all train_args option, please refer to HuggingFace `TrainingArguments <https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments>`_. |
+-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Running Train Code
----------------------
To start NEA training, execute the following code,

.. code:: python

    # Download NLTK package
    import nltk
    nltk.download('punkt')

    from sgnlp_models.models.nea.utils import parse_args_and_load_config
    from sgnlp_models.models.nea import train
    cfg = parse_args_and_load_config('config/nea_config.json')
    train(cfg)


Evaluating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------

Refer to training section above for dataset example.


Config Preparation
------------------

Aspect of the evaluation could be configure via the `nea_config.json` file.

+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration key                | Description                                                                                                                                                                             |
+==================================+=========================================================================================================================================================================================+
| use_wandb                        | Use weight and biases for training logs.                                                                                                                                                |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| wandb_config/project             | Project name for wandb.                                                                                                                                                                 |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| wandb_config/tags                | Tags label for wandb.                                                                                                                                                                   |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| wandb_config/name                | Name of a specific train run. To be updated for each different train run.                                                                                                               |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_type                       | NEA model type to use for training.                                                                                                                                                     |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| emb_path                         | File path to the embedding file. Refer to `original github <https://github.com/nusnlp/nea/blob/master/FAQ.md>`_ for reference.                                                          |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/train_path  | File path to the train dataset file.                                                                                                                                                    |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/dev_path    | File path to the dev dataset file.                                                                                                                                                      |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/test_path   | File path to the test dataset file.                                                                                                                                                     |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/prompt_id   | Prompt ID to filter from dataset for training.                                                                                                                                          |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/maxlen      | Maximum allowed number of words during training.                                                                                                                                        |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/to_lower    | Flag to indicate if dataset should be set to lower case.                                                                                                                                |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess_data_args/score_index | Score index to use for scoring predictions.                                                                                                                                             |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/azure_path        | Root directory path to Azure storage for NEA files.                                                                                                                                     |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/files             | Files name for tokenizers files required to construct NEATokenizer.                                                                                                                     |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/vocab_train_file  | File path to vocab file for training NEATokenizer.                                                                                                                                      |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tokenizer_args/save_folder       | Folder path to save downloaded tokenizer files from Azure storage.                                                                                                                      |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| eval_args/results_path           | File path of evaluation results.                                                                                                                                                        |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| eval_args/trainer_args           | For all eval_args/trainer_args option, please refer to HuggingFace `TrainingArguments <https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments>`_. |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Running Evaluation Code
---------------------------
To start NEA evaluation, execute the following code,

.. code:: python

    # Download NLTK package
    import nltk
    nltk.download('punkt')

    from sgnlp_models.models.nea.utils import parse_args_and_load_config
    from sgnlp_models.models.nea import train
    cfg = parse_args_and_load_config('config/nea_config.json')
    eval(cfg)
