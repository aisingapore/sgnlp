Sentic-GCN: Aspect-Based Sentiment Analysis via Affective Knowledge Enhanced Graph Convolutional Networks
=========================================================================================================

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Sentic-GCN model was proposed in `Aspect-Based Sentiment Analysis via Affective Knowledge Enhanced
Graph Convolutional Networks <https://www.sentic.net/sentic-gcn.pdf>`_ by Liang, Bin and Su, Hang and
Gui, Lin and Cambria, Erik and Xu, Ruifeng.

The abstract from the paper is as follows:

*Aspect-based sentiment analysis is a fine-grained sentiment analysis task, which needs to detection the
sentiment polarity towards a given aspect. Recently, graph neural models over the dependency tree are
widely applied for aspect- based sentiment analysis. Most existing works, however, they generally focus
on learning the dependency information from contextual words to aspect words based on the dependency tree
of the sentence, which lacks the exploitation of contextual affective knowledge with regard to the
specific aspect. In this pa- per, we propose a graph convolutional network based on SenticNet to leverage
the affective dependencies of the sentence according to the specific aspect, called Sentic GCN. To be
specific, we explore a novel solution to construct the graph neural networks via integrating the affective
knowledge from SenticNet to en- hance the dependency graphs of sentences. Based on it, both the
dependencies of contextual words and aspect words and the affective information between opinion words and
the aspect are considered by the novel affective enhanced graph model. Experimental results on multiple
public benchmark datasets il- lustrate that our proposed model can beat state-of-the-art methods.*

In keeping with how the models performance are calculated in the paper, this implementation save the best
performing model weights for both Sentic-GCN model and the Sentic-GCN Bert model.

Default dataset presented in the paper are the SemEval 2014 (Laptop, Restaurant), SemEval 2015
(Restaurant), SemEval 2016 (Restaurant). However, please note that the dataset format have been further
processed from original source, please see dataset link below for the processed datasets.

| Link to the `paper <https://www.sentic.net/sentic-gcn.pdf>`_
| Link to the `dataset <https://github.com/BinLiang-NLP/Sentic-GCN/tree/main/datasets>`_
| Link to the original `github <https://github.com/BinLiang-NLP/Sentic-GCN>`_


Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Sentic-GCN model pretrained on the SemEval 2014/2015/2016 data can be loaded and accessed with the
following code:

.. code:: python

    from sgnlp.models.sentic_gcn import(
        SenticGCNConfig,
        SenticGCNModel,
        SenticGCNEmbeddingConfig,
        SenticGCNEmbeddingModel,
        SenticGCNTokenizer,
        SenticGCNPreprocessor,
        SenticGCNPostprocessor,
        download_tokenizer_files,
    )

    download_tokenizer_files(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_tokenizer/",
        "senticgcn_tokenizer")
    tokenizer = SenticGCNTokenizer.from_pretrained("senticgcn_tokenizer")

    config = SenticGCNConfig.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn/config.json"
    )
    model = SenticGCNModel.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn/pytorch_model.bin",
        config=config
    )

    embed_config = SenticGCNEmbeddingConfig.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_embedding_model/config.json"
    )
    embed_model = SenticGCNEmbeddingModel.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_embedding_model/pytorch_model.bin",
        config=embed_config
    )

    preprocessor = SenticGCNPreprocessor(tokenizer=tokenizer, embedding_model=embed_model)
    postprocessor = SenticGCNPostprocessor()

    inputs = [
        {
            "aspects": ["Soup"],
            "sentence": "The soup is a little salty."
        },
        {
            "aspects": ["service"],
            "sentence": """Everyone that sat in the back outside agreed that it was the worst service we
                           had ever received."""
        },
        {
            "aspects": ["location", "food"],
            "sentence": """it 's located in a strip mall near the beverly center , not the greatest
                           location , but the food keeps me coming back for more ."""
        }
    ]

    processed_inputs, processed_indices = preprocessor(inputs)
    raw_outputs = model(processed_indices)

    post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=raw_outputs)

    print(post_outputs[0])
    # {'sentence': ['The', 'soup', 'is', 'a', 'little', 'salty.'],
    #  'aspects': [1],
    #  'labels': [-1]}

    print(post_outputs[1])
    # {'sentence': ['Everyone', 'that', 'sat', 'in', 'the', 'back', 'outside', 'agreed', 'that', 'it',
    #               'was', 'the', 'worst', 'service', 'we', 'had', 'ever', 'received.'],
    #  'aspects': [13],
    #  'labels': [-1]}

    print(post_outputs[2])
    # {'sentence': ['it', "'s", 'located', 'in', 'a', 'strip', 'mall', 'near', 'the', 'beverly',
    #               'center', ',', 'not', 'the', 'greatest', 'location', ',', 'but', 'the', 'food',
    #               'keeps', 'me', 'coming', 'back', 'for', 'more', '.'],
    #  'aspects': [15, 19],
    #  'labels': [0, 1]}


The Sentic-GCN Bert model pretrained on the SemEval 2014/2015/2016 data can be loaded and accessed
with the following code:

.. code:: python

    from sgnlp.models.sentic_gcn import(
        SenticGCNBertConfig,
        SenticGCNBertModel,
        SenticGCNBertEmbeddingConfig,
        SenticGCNBertEmbeddingModel,
        SenticGCNBertTokenizer,
        SenticGCNBertPreprocessor,
        SenticGCNBertPostprocessor
    )

    tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")

    config = SenticGCNBertConfig.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json"
    )
    model = SenticGCNBertModel.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin",
        config=config
    )

    embed_config = SenticGCNBertEmbeddingConfig.from_pretrained("bert-base-uncased")
    embed_model = SenticGCNBertEmbeddingModel.from_pretrained("bert-base-uncased",
        config=embed_config
    )

    preprocessor = SenticGCNBertPreprocessor(tokenizer=tokenizer, embedding_model=embed_model)
    postprocessor = SenticGCNBertPostprocessor()

    inputs = [
        {
            "aspects": ["Soup"],
            "sentence": "The soup is a little salty."
        },
        {
            "aspects": ["service"],
            "sentence": """Everyone that sat in the back outside agreed that it was the worst service we
                           had ever received."""
        },
        {
            "aspects": ["location", "food"],
            "sentence": """it 's located in a strip mall near the beverly center , not the greatest
                           location , but the food keeps me coming back for more ."""
        }
    ]

    processed_inputs, processed_indices = preprocessor(inputs)
    raw_outputs = model(processed_indices)

    post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=raw_outputs)

    print(post_outputs[0])
    # {'sentence': ['The', 'soup', 'is', 'a', 'little', 'salty.'],
    #  'aspects': [1],
    #  'labels': [-1]}

    print(post_outputs[1])
    # {'sentence': ['Everyone', 'that', 'sat', 'in', 'the', 'back', 'outside', 'agreed', 'that', 'it',
    #               'was', 'the', 'worst', 'service', 'we', 'had', 'ever', 'received.'],
    #  'aspects': [13],
    #  'labels': [-1]}

    print(post_outputs[2])
    # {'sentence': ['it', "'s", 'located', 'in', 'a', 'strip', 'mall', 'near', 'the', 'beverly',
    #               'center', ',', 'not', 'the', 'greatest', 'location', ',', 'but', 'the', 'food',
    #               'keeps', 'me', 'coming', 'back', 'for', 'more', '.'],
    #  'aspects': [15, 19],
    #  'labels': [0, 1]}


Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The input data needs to be a dictionary with the following keys:

+----------------------+-----------------------------------------------------------------------------------------------+
| Key                  | Meaning                                                                                       |
+----------------------+-----------------------------------------------------------------------------------------------+
| aspects              | A list of aspect(s) which must also be found in the sentence.                                 |
+----------------------+-----------------------------------------------------------------------------------------------+
| sentence             | A sentence which also contains all the aspects.                                               |
+----------------------+-----------------------------------------------------------------------------------------------+

The value(s) for aspects must be a list and each aspect must also exists in the sentence. If aspect have more than one
occurances in the sentence, each aspect will be treated as an input instance.


Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The output returned from :class:`~sgnlp.models.sentic_gcn.postprocess.SenticGCNPostprocessor` and
:class:`~sgnlp.models.sentic_gcn.postprocess.SenticGCNBertPostprocessor` consists of a list of dictionary
containing each processed input entries. Each entry consists of the following:

1. sentence: The input sentence in tokenized form.
2. aspects: A list of indices which denotes each index position in the tokenized input sentence.
3. labels: A list of prediction for each aspects in order. -1 denote negative sentiment, 0 denote neutral sentiment and 1 denote positive sentiment.

The logits can be accessed from the model output returned from the model.


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------
Prepare the training and evaluation dataset in the format that is the same as the datasets from the
author's repo. Please refer to the sample dataset
`here <https://github.com/BinLiang-NLP/Sentic-GCN/tree/main/datasets/semeval14>`__ for reference.


Config Preparation
------------------

Aspect of the training could be configured via the `sentic_gcn_config.json` and `sentic_gcn_bert_config.json`
file. An example of the Sentic-GCN config file can be found
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/sentic_gcn/config/sentic_gcn_config.json>`_
and example of the Sentic-GCN Bert config file can be found
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/sentic_gcn/config/sentic_gcn_bert_config.json>`_

+------------------------------------------+--------------------------------------------------------------------------------------+
| Configuration key                        | Description                                                                          |
+==========================================+======================================================================================+
| senticnet_word_file_path                 | File path to the SenticNet 5.0 file.                                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_preprocessed_senticnet              | Flag to indicate if the processed SenticNet dictionary should be pickled.            |
+------------------------------------------+--------------------------------------------------------------------------------------+
| saved_preprocessed_senticnet_file_path   | Pickle file path for saving processed SenticNet dictionary.                          |
+------------------------------------------+--------------------------------------------------------------------------------------+
| spacy_pipeline                           | Spacy pre-trained pipeline to load for preprocessing.                                |
+------------------------------------------+--------------------------------------------------------------------------------------+
| word_vec_file_path                       | File path to word vectors file for generating embeddings. (e.g. GloVe vectors.)      |
+------------------------------------------+--------------------------------------------------------------------------------------+
| dataset_train                            | List of training dataset files path.                                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| dataset_test                             | List of testing dataset files path.                                                  |
+------------------------------------------+--------------------------------------------------------------------------------------+
| valset_ratio                             | Ratio for train validation split.                                                    |
+------------------------------------------+--------------------------------------------------------------------------------------+
| model                                    | The model type to train. Either 'senticgcn' or 'senticgcnbert'.                      |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_best_model                          | Flag to indicate if best model should saved.                                         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_model_path                          | Folder path to save best performing model during training.                           |
+------------------------------------------+--------------------------------------------------------------------------------------+
| tokenizer                                | The tokenizer type to use for dataset preprocessing.                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| train_tokenizer                          | Flag to indicate if tokenizer should be trained using train and test datasets.       |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_tokenizer                           | Flag to indicate if trained tokenizer should be saved.                               |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_tokenizer_path                      | Folder path to save trained tokenizer.                                               |
+------------------------------------------+--------------------------------------------------------------------------------------+
| embedding_model                          | Embedding model type to use for training.                                            |
+------------------------------------------+--------------------------------------------------------------------------------------+
| build_embedding_model                    | Flag to indicate if embedding model should be trained on input word vectors.         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_embedding_model                     | Flag to indicate if trained embedding model should be saved.                         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_embedding_model_path                | Folder path to save trained embedding model.                                         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_results                             | Flag to indicate if training results should be saved.                                |
+------------------------------------------+--------------------------------------------------------------------------------------+
| save_result_folder                       | Folder path for saving training results.                                             |
+------------------------------------------+--------------------------------------------------------------------------------------+
| initializer                              | torch.nn.initializer type for initializing model weights.                            |
+------------------------------------------+--------------------------------------------------------------------------------------+
| optimizer                                | torch.nn.optimizer type for training.                                                |
+------------------------------------------+--------------------------------------------------------------------------------------+
| loss_function                            | Loss function to use for training.                                                   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| learning_rate                            | Learning rate for training.                                                          |
+------------------------------------------+--------------------------------------------------------------------------------------+
| l2reg                                    | l2reg value to set for training.                                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+
| epochs                                   | Number of epoch to train.                                                            |
+------------------------------------------+--------------------------------------------------------------------------------------+
| batch_size                               | Batch size to set for dataloader.                                                    |
+------------------------------------------+--------------------------------------------------------------------------------------+
| log_step                                 | Print training results for every log_step.                                           |
+------------------------------------------+--------------------------------------------------------------------------------------+
| embed_dim                                | Size of embedding dimension.                                                         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| hidden_dim                               | Size of hidden layer for GCN.                                                        |
+------------------------------------------+--------------------------------------------------------------------------------------+
| polarities_dim                           | Size of output layer.                                                                |
+------------------------------------------+--------------------------------------------------------------------------------------+
| dropout                                  | Dropout ratio for dropout layer.                                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+
| seed                                     | Random seed to set prior to training.                                                |
+------------------------------------------+--------------------------------------------------------------------------------------+
| device                                   | torch.device to set for training.                                                    |
+------------------------------------------+--------------------------------------------------------------------------------------+
| repeats                                  | Number of times to repeat whole training cycle.                                      |
+------------------------------------------+--------------------------------------------------------------------------------------+
| patience                                 | Patience value for early stopping.                                                   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| max_len                                  | Maximum length for input tensor.                                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+


Running Train Code
------------------
To start training Sentic-GCN or Sentic-GCN Bert model, execute the following code:

.. code:: python

    from sgnlp.models.sentic_gcn.train import SenticGCNTrainer, SenticGCNBertTrainer
    from sgnlp.models.sentic_gcn.utils import parse_args_and_load_config, set_random_seed

    cfg = parse_args_and_load_config()
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
    trainer = SenticGCNTrainer(cfg) if cfg.model == "senticgcn" else SenticGCNBertTrainer(cfg)
    trainer.train()


Evaluating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------

Refer to training section above for dataset example.


Config Preparation
------------------

Aspect of the training could be configured via the `sentic_gcn_config.json` and `sentic_gcn_bert_config.json`
file. An example of the Sentic-GCN config file can be found
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/sentic_gcn/config/sentic_gcn_config.json>`_
and example of the Sentic-GCN Bert config file can be found
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/sentic_gcn/config/sentic_gcn_bert_config.json>`_

+------------------------------------------+--------------------------------------------------------------------------------------+
| Configuration key                        | Description                                                                          |
+==========================================+======================================================================================+
| eval_args/model                          | The model type to evaluate. Either 'senticgcn' or 'senticgcnbert'.                   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| eval_args/model                          | Path to model folder, could be cloud storage, local folder or HuggingFace model hub. |
+------------------------------------------+--------------------------------------------------------------------------------------+
| tokenizer                                | The tokenizer type to use for dataset preprocessing.                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| embedding_model                          | The embedding model type to use for dataset preprocessing.                           |
+------------------------------------------+--------------------------------------------------------------------------------------+
| config_filename                          | Config file name to load from model folder and embedding model folder.               |
+------------------------------------------+--------------------------------------------------------------------------------------+
| model_filename                           | Model file name to load from model folder and embedding model folder.                |
+------------------------------------------+--------------------------------------------------------------------------------------+
| test_filename                            | File path to test dataset.                                                           |
+------------------------------------------+--------------------------------------------------------------------------------------+
| senticnet                                | File path to pickled processed senticnet.                                            |
+------------------------------------------+--------------------------------------------------------------------------------------+
| spacy_pipeline                           | Spacy pre-trained pipeline to load for dataset preprocesing.                         |
+------------------------------------------+--------------------------------------------------------------------------------------+
| result_folder                            | Folder to save evaluation results.                                                   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| eval_batch_size                          | Batch size for evaluator dataloader.                                                 |
+------------------------------------------+--------------------------------------------------------------------------------------+
| seed                                     | Random seed to set for evaluation.                                                   |
+------------------------------------------+--------------------------------------------------------------------------------------+
| device                                   | torch.device to set for tensors.                                                     |
+------------------------------------------+--------------------------------------------------------------------------------------+


Running the Evaluation Code
---------------------------
To start evaluating Sentic-GCN or Sentic-GCN Bert model, execute the following code:

.. code:: python

    from sgnlp.models.sentic_gcn.eval import SenticGCNEvaluator, SenticGCNBertEvaluator
    from sgnlp.models.sentic_gcn.utils import parse_args_and_load_config, set_random_seed

    cfg = parse_args_and_load_config()
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
    evaluator = SenticGCNEvaluator(cfg) if cfg.model == "senticgcn" else SenticGCNBertEvaluator(cfg)
    evaluator.evaluate()
