from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor

config = CoherenceMomentumConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/config.json"
)
model = CoherenceMomentumModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/pytorch_model.bin",
    config=config
)

preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)

text1 = "Companies listed below reported quarterly profit substantially different from the average of analysts ' " \
        "estimates . The companies are followed by at least three analysts , and had a minimum five-cent change in " \
        "actual earnings per share . Estimated and actual results involving losses are omitted . The percent " \
        "difference compares actual profit with the 30-day estimate where at least three analysts have issues " \
        "forecasts in the past 30 days . Otherwise , actual profit is compared with the 300-day estimate . " \
        "Source : Zacks Investment Research"
text2 = "The companies are followed by at least three analysts , and had a minimum five-cent change in actual " \
        "earnings per share . The percent difference compares actual profit with the 30-day estimate where at least " \
        "three analysts have issues forecasts in the past 30 days . Otherwise , actual profit is compared with the " \
        "300-day estimate . Source : Zacks Investment Research. Companies listed below reported quarterly profit " \
        "substantially different from the average of analysts ' estimates . Estimated and actual results involving " \
        "losses are omitted ."

text1_tensor = preprocessor([text1])
text2_tensor = preprocessor([text2])

text1_score = model.get_main_score(text1_tensor["tokenized_texts"]).item()
text2_score = model.get_main_score(text2_tensor["tokenized_texts"]).item()

print(text1_score, text2_score)
