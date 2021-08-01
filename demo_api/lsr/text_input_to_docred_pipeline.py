import textdistance
import numpy as np
import allennlp_models.tagging  # Needed by Predictor
from allennlp.predictors.predictor import Predictor


class TextInputToDocredPipeline:
    """
    This pipeline transforms a text input into the DocRED format.
    It uses pretrained NER and coreference models from allennlp.
    """

    def __init__(self):
        self.ner = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2020-06-24.tar.gz")
        self.coref = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
        self.ner._tokenizer.spacy = self.coref._spacy  # Set spacy model of ner to coref (ensure same model is used)

    def preprocess(self, text):
        text = text.replace('\n', '')  # Remove newline tokens. Because of inconsistencies between coref and ner model.
        text = ' '.join(text.split())  # Remove multi-spaces. Because coref model does not behave consistently with it.

        ner_pred = self.ner.predict(sentence=text)
        named_entities = parse_allennlp_ner_pred(ner_pred)

        coref_pred = self.coref.predict(document=text)
        coref_clusters = parse_allennlp_coref_pred(coref_pred)

        assert coref_pred['document'] == ner_pred['words']  # Check that the tokenization is the same

        named_entities_coref_clusters = remove_non_named_entities(named_entities, coref_clusters)

        # Get sents tokens
        spacy_coref = self.coref._spacy
        spacy_doc = spacy_coref(text)
        sents_tokens = []
        for sent in spacy_doc.sents:
            sents_tokens.append([token.text for token in sent])

        return {
            'vertexSet': transform_to_docred_format(named_entities_coref_clusters, sents_tokens),
            'sents': sents_tokens
        }


def parse_allennlp_ner_pred(ner_pred):
    """Parses allennlp fine-grained-ner prediction to get tokens, start and end indices, and ner type"""
    named_entities = []
    for i, tag in enumerate(ner_pred['tags']):
        splits = tag.split('-')
        if splits[0] == 'B':  # Indicates start
            start_idx = i
        elif splits[0] == 'L':  # Indicates end
            end_idx = i + 1
            assert end_idx > start_idx
            named_entities.append({
                'tokens': ner_pred['words'][start_idx:end_idx],
                'pos': [start_idx, end_idx],
                'type': splits[1]
            })
        elif splits[0] == 'U':  # Indicates that the entity is a single token (start + end)
            named_entities.append({
                'tokens': ner_pred['words'][i:i + 1],
                'pos': [i, i + 1],
                'type': splits[1]
            })

    return named_entities


def parse_allennlp_coref_pred(coref_pred):
    """Parses allennlp coref prediction to get tokens, and start and indices"""
    coref_clusters = []
    document = coref_pred['document']
    for cluster in coref_pred['clusters']:
        coref_cluster = []
        for indices_span in cluster:
            start_idx, end_idx = indices_span[0], indices_span[1] + 1
            coref_cluster.append({'tokens': document[start_idx:end_idx], 'pos': [start_idx, end_idx]})
        coref_clusters.append(coref_cluster)
    return coref_clusters


def span_overlap(span_indices1, span_indices2):
    """Boolean check if two spans overlap.

    This does an exclusive overlap check (i.e. (0-2) and (2-3) does not overlap).
    """
    return span_indices1[0] < span_indices2[1] and span_indices2[0] < span_indices1[1]


def remove_non_named_entities(named_entities, coref_clusters):
    """Remove non named entities from coref clusters i.e. Keep recognized named entities in the coref clusters."""
    named_entities_coref_clusters = []
    ner_used_count = np.zeros(len(named_entities))
    for cluster in coref_clusters:
        named_entities_coref_cluster = dict()
        for coref_span in cluster:
            for ner_idx, ner_span in enumerate(named_entities):
                if span_overlap(coref_span['pos'], ner_span['pos']):
                    # Choose ner span if there's an overlap
                    named_entities_coref_cluster[(ner_span['pos'][0], ner_span['pos'][1])] = ner_span
                    ner_used_count[ner_idx] += 1
        named_entities_coref_cluster = list(named_entities_coref_cluster.values())
        if len(named_entities_coref_cluster) > 1:
            # Resolve named entities that match poorly with others in cluster
            named_entities_coref_cluster = remove_dissimilar_entities(named_entities_coref_cluster)
        if len(named_entities_coref_cluster) > 0:
            # Only append when there is at least one named entity in coref cluster
            named_entities_coref_clusters.append(named_entities_coref_cluster)

    # Add in the remaining ner that do not occur in a cluster i.e. single mention
    remaining_entities = np.array(named_entities, dtype='object')[ner_used_count == 0]
    for entity in remaining_entities:
        named_entities_coref_clusters.append([entity])
    return named_entities_coref_clusters


def get_pairwise_jaccard_scores(named_entities_coref_cluster):
    size = len(named_entities_coref_cluster)
    scores = np.zeros(shape=(size, size))
    for i in range(size):
        for j in range(i + 1, size):
            scores[i, j] = textdistance.jaccard(named_entities_coref_cluster[i]['tokens'],
                                                named_entities_coref_cluster[j]['tokens'])
    scores = scores + scores.T  # Copy upper triangle to lower triangle, assumes scores is an upper triangle matrix with 0 diagonal
    return scores


def remove_dissimilar_entities(named_entities_coref_cluster):
    jaccard_scores = get_pairwise_jaccard_scores(named_entities_coref_cluster)
    mean_jaccard_scores = jaccard_scores.mean(axis=1)
    threshold_ratio = 0.3  # arbitrary choice
    threshold = max(mean_jaccard_scores) * threshold_ratio
    filtered_cluster = np.array(named_entities_coref_cluster)  # Convert to numpy array to use numpy indexing
    filtered_cluster = filtered_cluster[mean_jaccard_scores > threshold]
    return filtered_cluster.tolist()


# Maps ontonotes ner labels to docred ner labels
ontonotes2docred_ner_labels = {
    "CARDINAL": "NUM",
    "DATE": "TIME",
    "EVENT": "MISC",
    "FAC": "LOC",
    "GPE": "LOC",
    "LANGUAGE": "MISC",
    "LAW": "MISC",
    "LOC": "LOC",
    "MONEY": "NUM",
    "NORP": "ORG",
    "ORDINAL": "NUM",
    "ORG": "ORG",
    "PERCENT": "NUM",
    "PERSON": "PER",
    "PRODUCT": "MISC",
    "QUANTITY": "NUM",
    "TIME": "TIME",
    "WORK_OF_ART": "MISC"
}


def transform_to_docred_format(named_entities_coref_clusters, sents_tokens):
    """Transform input to a format similar to the DocRED dataset"""
    sent_starts = [0]
    sent_length = 0
    for sent_tokens in sents_tokens:
        sent_length += len(sent_tokens)
        sent_starts.append(sent_length)

    processed_clusters = []
    for cluster in named_entities_coref_clusters:
        processed_cluster = []
        for span in cluster:
            start_idx, end_idx = span['pos']
            ner_type = span['type']
            # Find next sentence idx
            for sent_idx, sent_start_idx in enumerate(sent_starts[1:], 1):
                if start_idx < sent_start_idx:
                    break
            # Adjust to current sentence idx
            sent_idx = sent_idx - 1
            offset = sent_starts[sent_idx]  # Offset to shift idx
            start_idx = start_idx - offset
            # Edge case of entity crossing sentence (keep first part only)
            end_idx = min(end_idx, sent_starts[sent_idx + 1])
            end_idx = end_idx - offset
            str_tokens = sents_tokens[sent_idx][start_idx:end_idx]
            processed_cluster.append({
                'name': ' '.join(str_tokens),
                'pos': [start_idx, end_idx],
                'sent_id': sent_idx,
                'type': ontonotes2docred_ner_labels[ner_type],  # Map from ontonotes to docred ner label
            })
        processed_clusters.append(processed_cluster)
    return processed_clusters
