"""Training / eval helpers around FaCLIP dual-cache + ODA.

Cache writes are disabled under eval() for frozen inference.
"""
import torch
import torch.nn as nn

from .faclip import FaCLIP
from .loss import BinaryDiceLoss, FocalLoss


class FaCLIP_Trainer(nn.Module):
    def __init__(
            self,
            learning_rate=0.01,
            device='cuda',
            feature_dim=768,
            cache_memory_size=25,
            cache_momentum=0.8,
            cache_alpha=0.2,
            cache_n_thres=100,
            cache_obs_history_thres=200,
            cache_bandwidth_scale=1.0,
            oda_temp_init=0.1,
    ):
        super(FaCLIP_Trainer, self).__init__()

        self.device = device
        self.lambda_oda = 0.05  # λ_ODA in the paper (Eq. 22)

        # Same losses as the original trainer (Eq. 23)
        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        self.clip_model = FaCLIP(
            feature_dim=feature_dim,
            device=device,
            cache_memory_size=cache_memory_size,
            cache_momentum=cache_momentum,
            cache_alpha=cache_alpha,
            cache_n_thres=cache_n_thres,
            cache_obs_history_thres=cache_obs_history_thres,
            cache_bandwidth_scale=cache_bandwidth_scale,
            oda_temp_init=oda_temp_init,
        ).to(device)

        self.params_to_update = [
            p for p in self.clip_model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            self.params_to_update, lr=learning_rate, betas=(0.5, 0.999))

    def save(self, path):
        torch.save(self.clip_model.state_dict(), path)

    def load(self, path):
        self.clip_model.load_state_dict(
            torch.load(path, map_location=self.device), strict=False)

    def train_one_batch(
            self,
            image_features,
            text_features,
            anomaly_score,
            anomaly_maps,
            is_anomaly,
            gt_mask,
            patch_tokens=None,
    ):
        """L_total = L_cls + L_seg + λ_ODA * L_ODA (Eq. 22)."""
        _, _, loss_oda = self.clip_model(
            image_features, text_features, patch_tokens, return_oda_loss=True)

        if not isinstance(anomaly_maps, list):
            anomaly_maps = [anomaly_maps]

        gt = gt_mask.to(self.device).clone().squeeze()
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        is_anomaly = is_anomaly.to(self.device).clone()
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0

        # L_cls = Focal(p_cls, y)  (Eq. 23)
        loss_cls = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))

        # L_seg = Focal(p_seg, S) + Dice(p_seg^(1), S) + Dice(p_seg^(0), 1-S)
        loss_seg = 0
        for am in anomaly_maps:
            loss_seg = (
                loss_seg
                + self.loss_focal(am, gt)
                + self.loss_dice(am[:, 1, :, :], gt)
                + self.loss_dice(am[:, 0, :, :], 1 - gt)
            )

        loss = loss_cls + loss_seg + self.lambda_oda * loss_oda

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self, feature_loader):
        self.clip_model.train()
        loss_list = []
        for batch in feature_loader:
            # Expected batch:
            #   image_features, text_features, anomaly_score, anomaly_maps,
            #   is_anomaly, gt_mask [, patch_tokens]
            image_features = batch[0].to(self.device)
            text_features = batch[1].to(self.device)
            anomaly_score = batch[2].to(self.device)
            anomaly_maps = batch[3]
            if isinstance(anomaly_maps, (list, tuple)):
                anomaly_maps = [am.to(self.device) for am in anomaly_maps]
            else:
                anomaly_maps = anomaly_maps.to(self.device)
            is_anomaly = batch[4].to(self.device)
            gt_mask = batch[5].to(self.device)
            patch_tokens = batch[6] if len(batch) > 6 else None

            loss = self.train_one_batch(
                image_features,
                text_features,
                anomaly_score,
                anomaly_maps,
                is_anomaly,
                gt_mask,
                patch_tokens,
            )
            loss_list.append(loss.item())
        return sum(loss_list) / max(len(loss_list), 1)

    @torch.no_grad()
    def refine_features(self, image_features, text_features, patch_tokens=None):
        """Frozen-cache inference: eval() disables cache writes (non-transductive)."""
        self.clip_model.eval()
        return self.clip_model(
            image_features, text_features, patch_tokens, return_oda_loss=False)