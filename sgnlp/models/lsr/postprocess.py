"""Functionality for postprocessing :class:`~sgnlp.models.lsr.modeling.LsrModelOutput`"""

import json
import numpy as np
from torch import sigmoid
from .utils import idx2ht


class LsrPostprocessor:
    """This class processes :class:`~sgnlp.models.lsr.modeling.LsrModelOutput` to a readable format.

    Args:
        rel2id (:obj:`dict`):
            Relation to id mapping.
        rel_info (:obj:`dict`):
            Relation to description mapping.
        pred_threshold (:obj:`float`, `optional`, defaults to 0.3):
            Threshold for relation prediction to be returned.
    """

    def __init__(self, rel2id: dict, rel_info: dict, pred_threshold: float = 0.3):
        self.rel2id = rel2id
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.rel_info = rel_info
        self.pred_threshold = pred_threshold

    @staticmethod
    def from_file_paths(rel2id_path: str, rel_info_path: str, pred_threshold: float = 0.3):
        """Constructs LsrPostprocessor from relevant DocRED files.

        Args:
            rel2id_path (:obj:`str`):
                Path to relation to id mapping file.
            rel_info_path (:obj:`str`):
                Path to relation info file. This is a mapping from relation to relation description.
            pred_threshold (:obj:`float`, `optional`, defaults to 0.3):
                Threshold for relation prediction to be returned.

        Returns:
            postprocessor (:class:`~sgnlp.models.lsr.postprocess.LsrPostprocessor`)
        """
        rel2id = json.load(open(rel2id_path))
        rel_info = json.load(open(rel_info_path))
        rel_info['Na'] = 'No relation'
        return LsrPostprocessor(rel2id, rel_info, pred_threshold)

    def __call__(self, prediction, data):
        """
        Args:
            prediction (:class:`torch.FloatTensor`):
                Prediction of :class:`~sgnlp.models.lsr.modeling.LsrModelOutput`.
            data:
                DocRED-like data that was used as input in preprocessing step.

        Returns:
            List of dictionary that includes the document, the entity clusters found in the document, and
            the predicted relations between the entity clusters.
        """

        output = []
        for prediction_instance, data_instance in zip(prediction, data):
            document = [item for sublist in data_instance['sents'] for item in sublist]  # Flatten nested list tokens
            num_entities = len(data_instance['vertexSet'])
            total_relation_combinations = num_entities * (num_entities - 1)

            pred = sigmoid(prediction_instance).data.cpu().numpy()
            pred = pred[:total_relation_combinations]

            above_threshold_indices = zip(*np.where(pred > self.pred_threshold))

            relations = []
            for h_t_idx, rel_idx in above_threshold_indices:
                h_idx, t_idx = idx2ht(h_t_idx, num_entities)
                rel_description = self.rel_info[self.id2rel[rel_idx]]

                # Typecasts are to allow JSON serializable (Numpy types generally not json serializable by default)
                relations.append({
                    "score": float(pred[h_t_idx, rel_idx]),
                    "relation": rel_description,
                    "object_idx": int(h_idx),
                    "subject_idx": int(t_idx)
                })

            # Compute sentence start indices
            sentence_start_idx = [0]
            sentence_start_idx_counter = 0
            for sent in data_instance['sents']:
                sentence_start_idx_counter += len(sent)
                sentence_start_idx.append(sentence_start_idx_counter)

            clusters = []
            for vertex_set in data_instance['vertexSet']:
                cluster = []
                for entity in vertex_set:
                    sent_id = entity['sent_id']  # sent_id that entity appears in
                    pos_adjustment = sentence_start_idx[sent_id]  # start idx of sent
                    pos = list(entity['pos'])
                    pos = [pos[0] + pos_adjustment,
                           pos[1] + pos_adjustment]  # adjust pos by adding start of sentence idx
                    cluster.append(pos)
                clusters.append(cluster)

            output.append({
                "clusters": clusters,
                "document": document,
                "relations": relations
            })

        return output
