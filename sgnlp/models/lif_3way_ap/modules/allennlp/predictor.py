from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register("lif_3way_ap_predictor")
class Lif3WayApPredictor(Predictor):
    """
    Predictor Class
    """

    def predict_instance(self, instance: Instance) -> JsonDict:
        output = self._model.forward_on_instance(instance)
        label_probs = output["label_probs"]

        output_json = {
            "label_probs": label_probs,
        }
        return sanitize(output_json)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        output_json_list = []
        for i in range(len(outputs)):
            lp = outputs[i]["label_probs"]
            output_json_list.append(
                {
                    "label_probs": lp,
                }
            )

        return sanitize(output_json_list)

    @overrides
    def _json_to_instance(self, qa: JsonDict) -> Instance:
        paragraph = qa["context"]
        tokenized_paragraph = self._dataset_reader._tokenizer.tokenize(paragraph)

        prev_q_text_list = [q.strip().replace("\n", "") for q in qa["prev_qs"]]
        start_idx = max(
            0, len(prev_q_text_list) - self._dataset_reader._num_context_answers
        )
        prev_q_text_list = prev_q_text_list[start_idx:]
        prev_ans_text_list = qa["prev_ans"][start_idx:]
        prev_ans_text_list = prev_ans_text_list[start_idx:]

        candidate = qa["candidate"].strip()

        return self._dataset_reader.text_to_instance(
            prev_q_text_list,
            prev_ans_text_list,
            paragraph,
            candidate,
            tokenized_paragraph,
        )
