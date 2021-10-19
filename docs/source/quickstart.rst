Quickstart
==========

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sgnlp is tested on Python 3.8+ and Pytorch 1.8+.

Install sgnlp with Python's pip package manager.

.. code:: python

    pip install sgnlp


Basics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package is built around the following classes for each model, similar to how
the `transformers <https://huggingface.co/transformers/index.html>`_ package
is built.

1. **Preprocessor**: Preprocesses input data into a format that can be fed into the model
2. **Config**: Stores the configuration of the model.
3. **Model**: Pytorch models which can work with pretrained weights.
4. **Tokenizer**: Stores the vocabulary for each model and encodes text into a format that can be fed into Model.
5. **Postprocessor** (available for some models): Postprocesses the raw output of the model into a format that can be easily interpreted.

Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
