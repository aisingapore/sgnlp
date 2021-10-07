RECCON: Emotion Entailment Model
================================================================================

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The RECCON Emotion Entailment model was proposed in `Recognizing Emotion Cause
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

    from sgnlp.models.emotion_entailment import (
        RecconEmotionEntailmentConfig,
        RecconEmotionEntailmentModel,
        RecconEmotionEntailmentTokenizer,
        RecconEmotionEntailmentPreprocessor,
        RecconEmotionEntailmentPostprocessor,
    )

    config = RecconEmotionEntailmentConfig.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/config.json"
    )
    model = RecconEmotionEntailmentModel.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/pytorch_model.bin",
        config=config,
    )
    tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
    preprocessor = RecconEmotionEntailmentPreprocessor(tokenizer)
    postprocess = RecconEmotionEntailmentPostprocessor()

    input_batch = {
        "emotion": ["happiness", "happiness"],
        "target_utterance": ["Thank you very much .", "Thank you very much ."],
        "evidence_utterance": [
            "How can I forget my old friend ?",
            "My best wishes to you and the bride !",
        ],
        "conversation_history": [
            "It's very thoughtful of you to invite me to your wedding . How can I forget my old friend ? My best wishes to you and the bride ! Thank you very much .",
            "It's very thoughtful of you to invite me to your wedding . How can I forget my old friend ? My best wishes to you and the bride ! Thank you very much .",
        ],
    }
    input_dict = preprocessor(input_batch)
    raw_output = model(**input_dict)
    output = postprocess(raw_output)
    print(output)
    # [0, 1]

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
The output returned from :class:`~sgnlp.models.emotion_entailment.postprocess.RecconEmotionEntailmentPostprocessor`
instance is a list of int.

1 indicates that the evidence_utterance at the corresponding index caused the
corresponding emotion in the target_utterance, while 0 indicates that the
evidence_utterance at the corresponding index did not cause the corresponding
emotion in the target_utterance.

The logits can be accessed from the raw output returned from the model.


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------
Prepare the training and evaluation dataset in the format that is the same
as the RECCON dataset in the authors' repo. You can refer to the sample dataset
`here <https://github.com/declare-lab/RECCON/tree/main/data/subtask2/fold1>`_.
Use the dataset with context.

Config Preparation
------------------
Create a copy of the config file. Update the following parameters:
`x_train_path`, `x_valid_path` and `train_args/output_dir`. For the other parameters,
you can either use the default values or modify it. You can refer to an example
of the config file
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/emotion_entailment/config/emotion_entailment_config.json>`_.

+-----------------------+-----------------------------------------------+
| Configuration key     | Description                                   |
+-----------------------+-----------------------------------------------+
| model_name            | Pretrained model to use for training          |
+-----------------------+-----------------------------------------------+
| x_train_path          | Folder path to training data                  |
+-----------------------+-----------------------------------------------+
| x_valid_path          | Folder path to validation data                |
+-----------------------+-----------------------------------------------+
| max_seq_length        | Maximum length of input sequence              |
+-----------------------+-----------------------------------------------+
| train_args/output_dir | Folder path for model weights and checkpoints |
+-----------------------+-----------------------------------------------+

You may refer to the other *train_args* parameters `here <https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments>`_.

Running Train Code
----------------------
Import :func:`~sgnlp.models.emotion_entailment.train.train` and
:func:`~sgnlp.models.emotion_entailment.utils.parse_args_and_load_config`
function. Set the path to the config file as the argument for the
:func:`~sgnlp.models.emotion_entailment.utils.parse_args_and_load_config`
function. Run :func:`~sgnlp.models.emotion_entailment.train.train` on the
config.

.. code:: python

    import json
    from sgnlp.models.emotion_entailment import train
    from sgnlp.models.emotion_entailment.utils import parse_args_and_load_config

    cfg = parse_args_and_load_config('config/emotion_entailment_config.json')
    train(cfg)

Evaluating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset Preparation
-------------------
Prepare the test dataset in the format that is the same
as the RECCON dataset in the authors' repo. You can refer to the sample dataset
`here <https://github.com/declare-lab/RECCON/tree/main/data/subtask2/fold1>`_.
Use the dataset with context.

Config Preparation
------------------
Create a copy of the config file. Update the following parameters:
`eval_args/trained_model_dir` , `eval_args/x_test_path` and `results_path`.
For the other parameters, you can either use the default values or modify it.
You can refer to an example of the config file
`here <https://github.com/aimakerspace/sgnlp/blob/main/sgnlp/models/emotion_entailment/config/emotion_entailment_config.json>`_.

+--------------------------------------+---------------------------------------+
| Configuration key                    | Description                           |
+--------------------------------------+---------------------------------------+
| model_name                           | Pretrained model to use for training  |
+--------------------------------------+---------------------------------------+
| max_seq_length                       | Maximum length of input sequence      |
+--------------------------------------+---------------------------------------+
| eval_args/trained_model_dir          | Folder path for trained model weights |
+--------------------------------------+---------------------------------------+
| eval_args/x_test_path                | Folder path of test data              |
+--------------------------------------+---------------------------------------+
| eval_args/results_path               | Folder path to save the test result   |
+--------------------------------------+---------------------------------------+
| eval_args/per_device_eval_batch_size | Batch size for prediction             |
+--------------------------------------+---------------------------------------+
| eval_args/no_cuda                    | Whether to use cuda for prediction    |
+--------------------------------------+---------------------------------------+


Running Evaluation Code
---------------------------
Import :func:`~sgnlp.models.emotion_entailment.eval.evaluate` and
:func:`~sgnlp.models.emotion_entailment.utils.parse_args_and_load_config`
function. Set the path to the config file as the argument for the
:func:`~sgnlp.models.emotion_entailment.utils.parse_args_and_load_config`
function. Run :func:`~sgnlp.models.emotion_entailment.eval.evaluate` on the
config.

.. code:: python

    import json
    from sgnlp.models.emotion_entailment import evaluate
    from sgnlp.models.emotion_entailment.utils import parse_args_and_load_config

    cfg = parse_args_and_load_config('config/emotion_entailment_config.json')
    evaluate(cfg)