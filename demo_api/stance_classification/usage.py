from sgnlp.models.rumour_stance import (
    StanceClassificationConfig,
    StanceClassificationModel,
    StanceClassificationPostprocessor,
    StanceClassificationPreprocessor,
    StanceClassificationTokenizer,
)

config = StanceClassificationConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/config.json"
)

model = StanceClassificationModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/pytorch_model.bin",
    config=config,
)

tokenizer = StanceClassificationTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=False,
)

preprocessor = StanceClassificationPreprocessor(tokenizer=tokenizer)

postprocessor = StanceClassificationPostprocessor()

inputs = [
    "Update - French interior ministry says incident which led to evacuation of Trocadero near the Eiffel Tower in Paris was a false alarm",
    "@SkyNewsBreak At least theyre alert.",
    "@Golden_Gaytime @SkyNewsBreak thank God. This is already horrific enough.",
    "@SkyNewsBreak @LTCPeterLerner Have you thought of confirming information before posting it on Twitter? Might be helpful.",
    "@SkyNewsBreak: French interior ministry says incident which led to evacuation of Trocadero in Paris was false alarm @kristaalyce",
    "@theloon @SkyNewsBreak @LTCPeterLerner  I am not posting any info from now on which is not confirmed. Creates confusion.",
    "@collywobbles54 @SkyNewsBreak @LTCPeterLerner - Also creates unnecessary anxiety/fear for anyone with relatives/friends in France.",
    "@theloon @SkyNewsBreak @LTCPeterLerner  of course.",
]

processed_inputs = preprocessor(inputs)

outputs = model(*processed_inputs)

print(postprocessor(inputs, outputs))
