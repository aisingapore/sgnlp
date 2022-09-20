import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetModel, XLNetConfig
from transformers import PreTrainedModel
from .config import CoherenceMomentumConfig


class CoherenceMomentumPreTrainedModel(PreTrainedModel):
    config_class = CoherenceMomentumConfig
    base_model_prefix = "coherence_momentum"


class CoherenceMomentumModel(CoherenceMomentumPreTrainedModel):
    def __init__(self, config):

        super().__init__(config)
        self.momentum_coefficient = config.momentum_coefficient

        self.encoder_name = f"xlnet-{config.model_size}-cased"
        self.encoder_config = XLNetConfig.from_pretrained(self.encoder_name)
        self.main_encoder = XLNetModel(self.encoder_config)
        self.momentum_encoder = XLNetModel(self.encoder_config)

        if config.model_size == "base":
            hidden_size = 768
        elif config.model_size == "large":
            hidden_size = 1024

        self.queue = []
        self.queue_size = config.queue_size
        self.con_loss_weight = config.contrastive_loss_weight
        self.num_negs = config.num_negs
        self.margin = config.margin
        self.cosim = nn.CosineSimilarity()
        self.sub_margin = lambda z: z - config.margin

        self.conlinear = nn.Linear(hidden_size, 1)

    def init_encoders(self):
        self.main_encoder = XLNetModel.from_pretrained(self.encoder_name)
        self.momentum_encoder = XLNetModel.from_pretrained(self.encoder_name)

    def get_main_score(self, doc):
        rep = self.main_encoder(input_ids=doc).last_hidden_state[:, -1, :]
        score = self.conlinear(rep).view(-1)
        return score

    def get_momentum_rep(self, doc):
        rep = self.momentum_encoder(input_ids=doc).last_hidden_state[:, -1, :]
        return rep.detach()

    def get_cos_sim(self, pos_rep, pos_slice):
        pos_sim = self.cosim(pos_rep, pos_slice)
        neg_sims = [self.cosim(pos_rep, neg_x.view(1, -1)) for neg_x in self.queue]
        return pos_sim, neg_sims

    def update_momentum_encoder(self):
        with torch.no_grad():
            for main, moco in zip(
                self.main_encoder.parameters(), self.momentum_encoder.parameters()
            ):
                moco.data = (moco.data * self.momentum_coefficient) + (
                    main.data * (1 - self.momentum_coefficient)
                )

    def forward(self, pos_doc, pos_slice, neg_docs):
        pos_rep = self.main_encoder(input_ids=pos_doc).last_hidden_state[:, -1, :]
        pos_score = self.conlinear(pos_rep).view(-1)

        pos_slice_rep = self.get_momentum_rep(pos_slice)

        neg_scores = list(map(self.get_main_score, list(neg_docs)))
        neg_moco_rep = list(map(self.get_momentum_rep, list(neg_docs)))

        if len(self.queue) >= self.queue_size:  # global negative queue size
            del self.queue[: self.num_negs]
        self.queue.extend(neg_moco_rep[0])

        pos_sim, neg_sims = self.get_cos_sim(pos_rep, pos_slice_rep)

        sim_contra_loss = self.sim_contrastive_loss(pos_sim, neg_sims)
        contra_loss = self.contrastive_loss(pos_score, neg_scores[0])

        full_loss = (self.con_loss_weight * contra_loss) + (
            (1 - self.con_loss_weight) * sim_contra_loss
        )

        return full_loss

    def eval_forward(self, pos_doc, neg_docs):
        pos_score = self.get_main_score(pos_doc)
        neg_scores = torch.stack(list(map(self.get_main_score, list(neg_docs))))
        return pos_score.detach(), neg_scores[0].detach()

    def sim_contrastive_loss(self, pos_sim, neg_sims):
        neg_sims_sub = torch.stack(list(map(self.sub_margin, neg_sims))).view(-1)
        all_sims = torch.cat((neg_sims_sub, pos_sim), dim=-1)
        lsmax = -1 * F.log_softmax(all_sims, dim=-1)
        loss = lsmax[-1]
        return loss

    def contrastive_loss(self, pos_score, neg_scores):
        neg_scores_sub = torch.stack(list(map(self.sub_margin, neg_scores)))
        all_scores = torch.cat((neg_scores_sub, pos_score), dim=-1)
        lsmax = -1 * F.log_softmax(all_scores, dim=-1)
        pos_loss = lsmax[-1]
        return pos_loss
