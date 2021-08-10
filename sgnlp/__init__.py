import spacy

# Download spacy model if not installed
spacy_model = 'en_core_web_sm'

if not spacy.util.is_package(spacy_model):
    spacy.cli.download(spacy_model)
