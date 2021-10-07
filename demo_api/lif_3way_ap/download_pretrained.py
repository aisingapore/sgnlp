"""Run this script during build time to download the pretrained models and relevant files first"""

from sgnlp.models.lif_3way_ap import (
    LIF3WayAPModel,
    LIF3WayAPConfig
)

# Downloads pretrained config and model
config = LIF3WayAPConfig.from_pretrained('https://storage.googleapis.com/sgnlp/models/lif_3way_ap/config.json')
model = LIF3WayAPModel.from_pretrained('https://storage.googleapis.com/sgnlp/models/lif_3way_ap/pytorch_model.bin',
                                       config=config)
