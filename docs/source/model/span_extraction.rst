RECCON: Span Extraction Model
================================================================================

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The RECCON Span Extraction model was proposed in `Recognizing Emotion Cause
in Conversations <https://arxiv.org/abs/2012.11820>`_ by Soujanya Poria, Navonil
Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai
Jian, Pengfei Hong, Romila Ghosh, Abhinaba Roy, Niyati Chhaya, Alexande Gelbukh
and Rada Mihalcea.

The abstract from the paper is the following:

*Recognizing the cause behind emotions in text is a fundamental yet
under-explored area of research in NLP. Advances in this area hold the potential
to improve interpretability and performance in affect-based models. Identifying
emotion causes at the utterance level in conversations is particularly
challenging due to the intermingling dynamic among the interlocutors. To this
end, we introduce the task of recognizing emotion cause in conversations with an
accompanying dataset named RECCON. Furthermore, we define different cause types
based on the source of the causes and establish strong transformer-based
baselines to address two different sub-tasks of RECCON: 1) Causal Span
Extraction and 2) Causal Emotion Entailment.*

| Link to the `paper <https://arxiv.org/abs/2012.11820>`_
| Link to the `dataset <https://github.com/declare-lab/RECCON/tree/main/data>`_
| Link to the original `github <https://github.com/declare-lab/RECCON>`_


Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The model pretrained on the RECCON data can be loaded and accessed with the
following code:

.. code:: python

    from sgnlp.models.span_extraction import (
        RecconSpanExtractionConfig,
        RecconSpanExtractionModel,
        RecconSpanExtractionTokenizer,
        RecconSpanExtractionPreprocessor,
        RecconSpanExtractionPostprocessor,
    )

    config = RecconSpanExtractionConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/reccon_span_extraction/config.json"
    )
    tokenizer = RecconSpanExtractionTokenizer.from_pretrained(
        "mrm8488/spanbert-finetuned-squadv2"
    )
    model = RecconSpanExtractionModel.from_pretrained(
        "https://storage.googleapis.com/sgnlp-models/models/reccon_span_extraction/pytorch_model.bin",
        config=config,
    )
    preprocessor = RecconSpanExtractionPreprocessor(tokenizer)
    postprocessor = RecconSpanExtractionPostprocessor()

    input_batch = {
        "emotion": ["surprise", "surprise"],
        "target_utterance": [
            "Hi George ! It's good to see you !",
            "Hi George ! It's good to see you !",
        ],
        "evidence_utterance": [
            "Linda ? Is that you ? I haven't seen you in ages !",
            "Hi George ! It's good to see you !",
        ],
        "conversation_history": [
            "Linda ? Is that you ? I haven't seen you in ages ! Hi George ! It's good to see you !",
            "Linda ? Is that you ? I haven't seen you in ages ! Hi George ! It's good to see you !",
        ],
    }

    tensor_dict, evidences, examples, features = preprocessor(input_batch)
    raw_output = model(**tensor_dict)
    context, evidence_span, probability = postprocessor(raw_output, evidences, examples, features)

    print(context)
    # [['Linda ? Is that you ? ', "I haven't seen you in ages !"], ['Hi George ! ', "It's good to see you !"]]
    print(evidence_span)
    # [[0, 1], [0, 1]]
    print(probability)
    # [[-1, 0.943615029866203], [-1, 0.8712913786944898]]

Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The input data needs to be a dictionary with the following keys:

+----------------------+-----------------------------------------------------------------------------------------------+
| Key                  | Meaning                                                                                       |
+----------------------+-----------------------------------------------------------------------------------------------+
| emotion              | Emotion of the target utterance                                                               |
+----------------------+-----------------------------------------------------------------------------------------------+
| target_utterance     | Utterance whose emotion cause we are interested in                                            |
+----------------------+-----------------------------------------------------------------------------------------------+
| evidence_utterance   | Potential evidence utterance for causing emotion in target utterance                          |
+----------------------+-----------------------------------------------------------------------------------------------+
| conversation_history | All utterances from the beginning of the conversation till and including the target utterance |
+----------------------+-----------------------------------------------------------------------------------------------+

The values need to be a list of str and the list need to be of the same length
across all keys. Refer to the original paper for more details of the inputs
required.

Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are 3 outputs returned from :class:`~sgnlp.models.span_extraction.postprocess.RecconSpanExtractionPostprocessor`.

1. Context: This shows the span extracted from the evidence utterance. This is a list of list of str.

2. Evidence span: This indicates whether the corresponding span is a causal span. This is a list of list of int.

3. Probability: This indicates the probability of the corresponding span being a causal span. -1 indicates that the span is non causal.

The start and end logits can be accessed from the raw output returned from the model.


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------
Prepare the training and evaluation dataset in the format that is the same
as the RECCON dataset in the authors' repo. You can refer to the sample dataset
`here <https://github.com/declare-lab/RECCON/tree/main/data/subtask2/fold1>`__.
Use the dataset with context.

Config Preparation
------------------
Create a copy of the config file. Update the following parameters:
`train_data_path`, `val_data_path` and `train_args/output_dir`. For the other parameters,
you can either use the default values or modify it. You can refer to an example
of the config file
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/span_extraction/config/span_extraction_config.json>`__.

+----------------------------------------+---------------------------------------------------------------------------------------------------+
| Configuration key                      | Description                                                                                       |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| model_name                             | Pretrained model to use for training                                                              |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| train_data_path                        | Folder path of training data                                                                      |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| val_data_path                          | Folder path of validation data                                                                    |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| max_seq_length                         | Maximum sequence length                                                                           |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| doc_stride                             | Document stride                                                                                   |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| max_query_length                       | Maximum query length                                                                              |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| train_args/output_dir                  | Folder path to save trained model weights                                                         |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| train_args/gradient_accumulation_steps | Number of updates steps to accumulate the gradients for, before performing a backward/update pass |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| train_args/num_train_epochs            | Total number of training epochs to perform                                                        |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| train_args/per_device_train_batch_size | Training batch size                                                                               |
+----------------------------------------+---------------------------------------------------------------------------------------------------+
| train_args/warmup_ratio                | Ratio of total training steps used for a linear warmup from 0 to learning_rate                    |
+----------------------------------------+---------------------------------------------------------------------------------------------------+

You may refer to the other *train_args* parameters `here <https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments>`__.

Running Train Code
----------------------
Import :func:`~sgnlp.models.span_extraction.train.train` and
:func:`~sgnlp.models.span_extraction.utils.parse_args_and_load_config`
function. Set the path to the config file as the argument for the
:func:`~sgnlp.models.span_extraction.utils.parse_args_and_load_config`
function. Run :func:`~sgnlp.models.span_extraction.train.train` on the
config.

.. code:: python

    import json
    from sgnlp.models.span_extraction import train
    from sgnlp.models.span_extraction.utils import parse_args_and_load_config

    cfg = parse_args_and_load_config('config/span_extraction_config.json')
    train(cfg)

Evaluating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------
Prepare the test dataset in the format that is the same
as the RECCON dataset in the authors' repo. You can refer to the sample dataset
`here <https://github.com/declare-lab/RECCON/tree/main/data/subtask2/fold1>`__.
Use the dataset with context.

Config Preparation
------------------
Create a copy of the config file. Update the following parameters:
`eval_args/trained_model_dir` , `eval_args/x_test_path` and `results_path`.
For the other parameters, you can either use the default values or modify it.
You can refer to an example of the config file
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/span_extraction/config/span_extraction_config.json>`__.


+-------------------------------------+---------------------------------------------------------------------------------------------------+
| Configuration key                   | Description                                                                                       |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| model_name                          | Pretrained model to use for training                                                              |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| test_data_path                      | Folder path of test data                                                                          |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| max_seq_length                      | Maximum sequence length                                                                           |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| doc_stride                          | Document stride                                                                                   |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| max_query_length                    | Maximum query length                                                                              |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/trained_model_dir         | Folder path to load trained model weights                                                         |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/results_path              | Number of updates steps to accumulate the gradients for, before performing a backward/update pass |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/batch_size                | Batch size for prediction                                                                         |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/n_best_size               | n best size                                                                                       |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/null_score_diff_threshold | Null score difference threshold                                                                   |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/sliding_window            | Whether to use sliding window                                                                     |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/no_cuda                   | Whether to use cuda                                                                               |
+-------------------------------------+---------------------------------------------------------------------------------------------------+
| eval_args/max_answer_length         | Maximum answer length                                                                             |
+-------------------------------------+---------------------------------------------------------------------------------------------------+

Running Evaluation Code
---------------------------
Import :func:`~sgnlp.models.span_extraction.eval.evaluate` and
:func:`~sgnlp.models.span_extraction.utils.parse_args_and_load_config`
function. Set the path to the config file as the argument for the
:func:`~sgnlp.models.span_extraction.utils.parse_args_and_load_config`
function. Run :func:`~sgnlp.models.span_extraction.eval.evaluate` on the
config.

.. code:: python

    import json
    from sgnlp.models.span_extraction import evaluate
    from sgnlp.models.span_extraction.utils import parse_args_and_load_config

    cfg = parse_args_and_load_config('config/span_extraction_config.json')
    evaluate(cfg)