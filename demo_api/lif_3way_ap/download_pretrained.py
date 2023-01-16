"""Run this script during build time to download the pretrained models and relevant files first"""

from sgnlp.models.lif_3way_ap import Lif3WayApModel
from sgnlp.models.lif_3way_ap.modules.allennlp.model import Lif3WayApAllenNlpModel
from sgnlp.models.lif_3way_ap.modules.allennlp.predictor import Lif3WayApPredictor
from sgnlp.models.lif_3way_ap.modules.allennlp.dataset_reader import (
    Lif3WayApDatasetReader,
)

# Downloads pretrained model
model = Lif3WayApModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/lif_3way_ap/model.tar.gz",
    predictor_name="lif_3way_ap_predictor",
)
