import torch
import os
import sys
import torch.nn as nn
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
from position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from transformer import TransformerEncoder, TransformerEncoderLayer


class MatchERT(nn.Module):
    def __init__(self, d_global, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation,
                 normalize_before):
        super(MatchERT, self).__init__()
        assert (d_model % 2 == 0)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pos_encoder = PositionEmbeddingSine(d_model // 2, normalize=True, scale=2.0)
        self.seg_encoder = nn.Embedding(4, d_model)
        self.classifier = nn.Linear(d_model, 1)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_global, src_local, tgt_global, tgt_local, normalize=False):
        ##########################################################
        # Global features are not used in the final model
        # Keep the API here for future study
        ##########################################################
        # src_global = self.remap(src_global)
        # tgt_global = self.remap(tgt_global)    
        # if normalize:
        #     src_global = F.normalize(src_global, p=2, dim=-1)
        #     tgt_global = F.normalize(tgt_global, p=2, dim=-1)

        bsize, fsize, h, w = src_local.size()
        pos_embed = self.pos_encoder(src_local.new_ones((1, h, w))).expand(bsize, fsize, h, w)
        cls_embed = self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long)).permute(0, 2, 1)
        sep_embed = self.seg_encoder(src_local.new_ones((bsize, 1), dtype=torch.long)).permute(0, 2, 1)
        src_local = src_local.flatten(2) + self.seg_encoder(
            2 * src_local.new_ones((bsize, 1), dtype=torch.long)).permute(0, 2, 1) + pos_embed.flatten(2)
        tgt_local = tgt_local.flatten(2) + self.seg_encoder(
            3 * src_local.new_ones((bsize, 1), dtype=torch.long)).permute(0, 2, 1) + pos_embed.flatten(2)
        # src_global = src_global.unsqueeze(1) + self.seg_encoder(4 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)
        # tgt_global = tgt_global.unsqueeze(1) + self.seg_encoder(5 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)

        # global features were not used in the final model
        input_feats = torch.cat([cls_embed, src_local, sep_embed, tgt_local], -1).permute(2, 0, 1)
        logits = self.encoder(input_feats)[0]
        return self.classifier(logits).view(-1)


class Matcher(nn.Module):
    def __init__(self, num_global_features: int = 128,
                 num_local_features: int = 128,
                 dropout: float = 0.,
                 detach: bool = False,
                 norm_layer: Optional[str] = None,
                 normalize: bool = False,
                 set_bn_eval: bool = False,
                 remap: bool = False,
                 normalize_weight: bool = False,
                 ert_seq_len: int = 102,
                 ert_dim_feedforward=1024,
                 ert_nhead: int = 4,
                 ert_num_encoder_layers: int = 6,
                 ert_dropout: int = 0.1,
                 ert_activation: str = 'relu',
                 ert_normalize_before: bool = False):
        super(Matcher, self).__init__()
        self.backbone_features = 2048
        if num_local_features != self.backbone_features:
            self.remap_local = nn.Conv2d(self.backbone_features, num_local_features, kernel_size=1, stride=1, padding=0)
            nn.init.zeros_(self.remap_local.bias)

        self.matcher = MatchERT(
            d_global=num_global_features, d_model=num_local_features,
            nhead=ert_nhead, num_encoder_layers=ert_num_encoder_layers,
            dim_feedforward=ert_dim_feedforward, dropout=ert_dropout,
            activation=ert_activation, normalize_before=ert_normalize_before
        )

    def forward(self, src_global=None, src_local=None, tgt_global=None, tgt_local=None):
        src_local, tgt_local = self.remap_local(src_local), self.remap_local(tgt_local)
        logits = self.matcher(src_global=src_global, src_local=src_local, tgt_global=tgt_global,
                              tgt_local=tgt_local)
        return logits, (src_global, src_local), (tgt_global, tgt_local)


import torch.nn.functional as F


class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyWithLogits, self).__init__()

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)


class Matcher_Loss(nn.Module):
    def __init__(self, num_global_features: int = 128,
                 num_local_features: int = 128,
                 dropout: float = 0.1,
                 detach: bool = False,
                 norm_layer: Optional[str] = None,
                 normalize: bool = False,
                 set_bn_eval: bool = False,
                 remap: bool = False,
                 normalize_weight: bool = False,
                 ert_seq_len: int = 102,
                 ert_dim_feedforward=1024,
                 ert_nhead: int = 4,
                 ert_num_encoder_layers: int = 6,
                 ert_dropout: int = 0.0,
                 ert_activation: str = 'relu',
                 ert_normalize_before: bool = False):
        super(Matcher_Loss, self).__init__()
        self.matcher = Matcher(num_global_features=num_global_features,
                               num_local_features=num_local_features,
                               dropout=dropout,
                               detach=detach,
                               norm_layer=norm_layer,
                               normalize=normalize,
                               set_bn_eval=set_bn_eval,
                               remap=remap,
                               normalize_weight=normalize_weight,
                               ert_seq_len=ert_seq_len,
                               ert_dim_feedforward=ert_dim_feedforward,
                               ert_nhead=ert_nhead,
                               ert_num_encoder_layers=ert_num_encoder_layers,
                               ert_dropout=ert_dropout,
                               ert_activation=ert_activation,
                               ert_normalize_before=ert_normalize_before)
        self.class_loss = BinaryCrossEntropyWithLogits()

    def forward(self, features, feature_maps, target):
        target_mask = torch.tensor(target.reshape(-1, 1) == target.reshape(1, -1), dtype=torch.long)
        # print("1 target_mask.shape:", target_mask.shape, torch.sum(target_mask, dim=1))
        target_mask -= torch.eye(target.size(0), dtype=torch.long).cuda()
        # self.neighbor_k = min([torch.min(torch.sum(target_mask, dim=1)), self.neighbor_k])
        # print("target_mask.shape:", target_mask.shape, torch.sum(target_mask, dim=1))
        feature_similarity = features @ features.T
        # print("feature_similarity.shape:", feature_similarity.shape, feature_similarity)
        min_pos_score, pos_idx = torch.min((feature_similarity - 1) * target_mask, -1)
        # print("min_pos_score:", min_pos_score, "pos_idx:", pos_idx)
        feature_similarity -= torch.eye(target.size(0), dtype=torch.long).cuda()
        max_neg_score, neg_idx = torch.max(feature_similarity * (1 - target_mask), -1)
        # print("max_neg_score:", max_neg_score, "neg_idx:", neg_idx)

        pos_features_map, neg_features_map = feature_maps[pos_idx], feature_maps[neg_idx]
        # print("target:", target)
        # anchor_feature_map, pos_features_map, neg_features_map = feature_maps[::3], feature_maps[1::3], feature_maps[
        #                                                                                                 2::3]
        # print("pos_features_map.shape, neg_features_map.shape:", pos_features_map.shape, neg_features_map.shape)
        pos_logits, _, _ = self.matcher(src_local=feature_maps, tgt_local=pos_features_map)
        neg_logits, _, _ = self.matcher(src_local=feature_maps, tgt_local=neg_features_map)
        logits = torch.cat([pos_logits, neg_logits], 0)

        bsize = logits.size(0)
        labels = logits.new_ones(logits.size())
        # labels = torch.rand(logits.size()).cuda()*0.3
        # print("logits.shape:", logits.shape, "labels.shape:", labels.shape)
        labels[(bsize // 2):] = 0
        # labels[:(bsize // 2)] += 0.7
        loss = self.class_loss(logits, labels)
        # print("loss:", loss)
        return loss


if __name__ == '__main__':
    src_local = torch.randn((100, 2048, 7, 7))
    tgt_local = torch.randn((100, 2048, 7, 7))
    matcher = Matcher()
    logits, _, _ = matcher(src_local=src_local, tgt_local=tgt_local)
    print(logits.shape)
