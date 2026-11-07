"""FA-CLIP core wrapper: dual-cache refinement + ODA.

Paper pipeline (Sec. III): F_I, F_T -> dual cache -> F_I^E, F_T^E -> L_ODA.
"""
import torch
from torch import nn

from .cache import ImageMemoryCache, MemoryCache
from .oda import ODALoss
from .prompts import PROMPT_ABNORMAL, PROMPT_NORMAL, PROMPT_TEMPLATES, build_prompt_sentences


class FaCLIP(nn.Module):
    """Fine-Grained Aligned CLIP — cache + ODA modules (manuscript defaults)."""

    def __init__(
            self,
            feature_dim: int = 768,
            device: str = 'cuda',
            use_cache: bool = True,
            cache_memory_size: int = 25,
            cache_momentum: float = 0.8,
            cache_alpha: float = 0.2,
            cache_n_thres: int = 100,
            cache_obs_history_thres: int = 200,
            cache_bandwidth_scale: float = 1.0,
            oda_temp_init: float = 0.1,
    ):
        super(FaCLIP, self).__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.use_cache = use_cache

        # Prompt construction
        self.prompt_normal = list(PROMPT_NORMAL)
        self.prompt_abnormal = list(PROMPT_ABNORMAL)
        self.prompt_templates = list(PROMPT_TEMPLATES)

        if self.use_cache:
            self.text_memory_cache = MemoryCache(
                feature_dim=feature_dim,
                memory_size=cache_memory_size,
                momentum=cache_momentum,
                alpha=cache_alpha,
                device=device,
                n_thres=cache_n_thres,
                obs_history_thres=cache_obs_history_thres,
                bandwidth_scale=cache_bandwidth_scale,
            )
            self.image_memory_cache = ImageMemoryCache(
                feature_dim=feature_dim,
                memory_size=cache_memory_size,
                momentum=cache_momentum,
                alpha=cache_alpha,
                device=device,
            )

        self.criterion = ODALoss(
            temp=float(oda_temp_init),
            feature_dim=feature_dim,
        )

    @staticmethod
    def build_prompt_sentences(class_name: str):
        return build_prompt_sentences(class_name)

    def refine(self, image_features, text_features, patch_tokens=None):
        """Apply dual-cache refinement: (F_I, F_T) -> (F_I^E, F_T^E).

        At eval time, cache writes are disabled (frozen, non-transductive).
        """
        if not self.use_cache:
            return image_features, text_features

        text_features, _ = self.text_memory_cache(text_features, image_features)
        image_features, _ = self.image_memory_cache(image_features, patch_tokens)
        return image_features, text_features

    def _prepare_oda_features(self, image_features, text_features):
        """Build paired F_I^E / F_T^E for L_ODA (Eqs. 19-21)."""
        fi = image_features.float()
        if text_features.dim() == 3:
            ft = text_features.float().mean(dim=-1)
        else:
            ft = text_features.float()

        if fi.dim() == 1:
            fi = fi.unsqueeze(0)
        if ft.dim() == 1:
            ft = ft.unsqueeze(0)

        if fi.shape[0] == 1:
            fi = fi.expand(2, -1).contiguous()
            ft = ft.expand(2, -1).contiguous()
        return fi, ft

    def forward(self, image_features, text_features, patch_tokens=None, return_oda_loss=True):
        """Dual-cache refine, then compute L_ODA."""
        image_e, text_e = self.refine(image_features, text_features, patch_tokens)

        if not return_oda_loss:
            return image_e, text_e

        fi_e, ft_e = self._prepare_oda_features(image_e, text_e)
        loss_oda = self.criterion(fi_e, ft_e)
        if isinstance(loss_oda, torch.Tensor) and loss_oda.numel() > 1:
            loss_oda = loss_oda.mean()
        return image_e, text_e, loss_oda
