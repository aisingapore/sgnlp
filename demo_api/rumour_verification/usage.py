from sgnlp.models.rumour_stance import (
    RumourVerificationConfig,
    RumourVerificationModel,
    RumourVerificationPostprocessor,
    RumourVerificationPreprocessor,
    RumourVerificationTokenizer,
)

config = RumourVerificationConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/config.json"
)

model = RumourVerificationModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/pytorch_model.bin",
    config=config,
)

tokenizer = RumourVerificationTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=False,
)

preprocessor = RumourVerificationPreprocessor(tokenizer=tokenizer)

postprocessor = RumourVerificationPostprocessor()

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

print(postprocessor(outputs))
