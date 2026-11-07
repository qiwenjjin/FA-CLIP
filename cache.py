"""Dual-cache modules: ITE-gated text cache and momentum image cache.

Aligns with paper Sec. III-B and the frozen, non-transductive inference protocol.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MemoryCache(nn.Module):
    def __init__(self, feature_dim=768, memory_size=25, momentum=0.8, alpha=0.2, device='cuda',
                 n_thres=100, obs_history_thres=200, bandwidth_scale=1.0):
        """ITE-gated text cache.

        n_thres:           |H| needed before ITE gating (N_thres)
        obs_history_thres: Phase-II observational collection length
        bandwidth_scale:   multiplier on Silverman bandwidth in NW regression
        """
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.alpha = alpha
        self.n_thres = int(n_thres)
        self.obs_history_thres = int(obs_history_thres)
        self.bandwidth_scale = float(bandwidth_scale)

        # Buffers so caches / ITE state are saved in checkpoints and restored at test time.
        self.register_buffer(
            'text_fine_cache',
            F.normalize(torch.rand(memory_size, feature_dim), dim=-1)
        )
        self.register_buffer(
            'frozen_cache_for_outcome',
            torch.zeros(memory_size, feature_dim)
        )
        self.register_buffer('frozen_cache_initialized', torch.tensor(False))

        self.extractor = nn.Linear(2 * feature_dim, feature_dim, bias=False)

        self.register_buffer('treatment_history', torch.empty(0))
        self.register_buffer('covariate_history', torch.empty(0, feature_dim))
        self.register_buffer('outcome_history', torch.empty(0))
        self.history_max_size = 1000

        self.k_pos = 5
        self.temperature = 0.07  # temperature for peakedness proxy Y

        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = 50  # N_warmup in the paper

    def _reshape_to_2d(self, x):
        """Flatten features of varying ranks to 2D."""
        if x.dim() == 3:
            B, D1, D2 = x.shape
            if D1 == self.feature_dim:
                return x.transpose(1, 2).reshape(B * D2, D1), (B, D1, D2)
            elif D2 == self.feature_dim:
                return x.reshape(B * D1, D2), (B, D1, D2)
            else:
                if D2 == 2:
                    return x.transpose(1, 2).reshape(B * D2, D1), (B, D1, D2)
                else:
                    return x.reshape(B * D1, D2), (B, D1, D2)
        elif x.dim() == 2:
            if x.shape[0] == self.feature_dim:
                return x.t(), x.shape
            elif x.shape[1] == self.feature_dim:
                return x, x.shape
            else:
                return x, x.shape
        return x, None

    def _reshape_back(self, x, original_shape):
        """Restore the original feature layout."""
        if original_shape is None:
            return x
        if len(original_shape) == 3:
            B, D1, D2 = original_shape
            if D1 == self.feature_dim:
                return x.reshape(B, D2, D1).transpose(1, 2)
            else:
                return x.reshape(B, D1, D2)
        elif len(original_shape) == 2:
            if original_shape[0] == self.feature_dim:
                return x.t()
            else:
                return x
        return x

    def get_score(self, query, mem):
        """Softmax similarity scores between queries and memory slots."""
        score = query @ mem.t()
        score_query = F.softmax(score, dim=0)
        score_mem = F.softmax(score, dim=1)
        return score_query, score_mem

    def initialize_frozen_cache(self):
        """Freeze C_ref once after warm-up."""
        if (not bool(self.frozen_cache_initialized.item())
                and self.training_step >= self.warmup_steps):
            self.register_buffer(
                'frozen_cache_for_outcome',
                self.text_fine_cache.detach().clone()
            )
            self.frozen_cache_initialized.fill_(True)

    def compute_outcome(self, features):
        """
        Peakedness proxy Y via softmax over similarities to frozen C_ref.
        """
        if not bool(self.frozen_cache_initialized.item()):
            return torch.zeros(features.shape[0], device=features.device)

        features_norm = F.normalize(features, dim=-1)
        similarities = features_norm @ self.frozen_cache_for_outcome.detach().t()

        # Softmax normalization
        probs = F.softmax(similarities / self.temperature, dim=1)

        max_prob = probs.max(dim=1)[0]
        mean_prob = probs.mean(dim=1)

        # Difference form
        outcome = max_prob - mean_prob

        return outcome

    def apply_cache_enhancement(self, features):
        """Apply residual cache enhancement."""
        features_norm = F.normalize(features, dim=-1)
        cache = self.text_fine_cache.detach()
        _, softmax_score_cache = self.get_score(features_norm, cache)
        cache_enhanced = softmax_score_cache @ cache
        enhanced = self.alpha * cache_enhanced + (1 - self.alpha) * features_norm
        return F.normalize(enhanced, dim=-1)

    def collect_observational_data(self, features):
        """Collect Phase-II observational (F_T, T, Y) records."""
        N = features.shape[0]
        T = torch.bernoulli(torch.full((N,), 0.5, device=self.device))
        features_for_outcome = features.clone()

        for i in range(N):
            if T[i] == 1:
                features_for_outcome[i] = self.apply_cache_enhancement(features[i:i + 1]).squeeze()

        Y = self.compute_outcome(features_for_outcome)
        return features, T, Y, features_for_outcome

    def update_history(self, X, T, Y):
        """Append to ITE history H."""
        cov = torch.cat([self.covariate_history, X.detach()], dim=0)
        treat = torch.cat([self.treatment_history, T.detach()], dim=0)
        out = torch.cat([self.outcome_history, Y.detach()], dim=0)

        if cov.shape[0] > self.history_max_size:
            cov = cov[-self.history_max_size:]
            treat = treat[-self.history_max_size:]
            out = out[-self.history_max_size:]

        self.register_buffer('covariate_history', cov)
        self.register_buffer('treatment_history', treat)
        self.register_buffer('outcome_history', out)

    def silverman_bandwidth(self, X):
        """Silverman's rule for bandwidth"""
        n, d = X.shape
        std = X.std(dim=0)
        q75 = torch.quantile(X, 0.75, dim=0)
        q25 = torch.quantile(X, 0.25, dim=0)
        iqr = (q75 - q25) / 1.34
        scale = torch.minimum(std, iqr)
        bandwidth = 0.9 * scale * (n ** (-1 / 5))
        return bandwidth.mean() * float(getattr(self, 'bandwidth_scale', 1.0))

    def estimate_conditional_expectation(self, X_train, Y_train, X_test, k=5):
        """Nadaraya–Watson kernel regression."""
        k = min(k, len(X_train))
        if k == 0:
            return torch.zeros(X_test.shape[0], device=self.device)

        distances = torch.cdist(X_test, X_train, p=2)
        _, nearest_indices = torch.topk(distances, k, largest=False, dim=1)
        nearest_distances = distances.gather(1, nearest_indices)

        bandwidth = self.silverman_bandwidth(X_train)
        kernel_weights = torch.exp(-0.5 * ((nearest_distances / (bandwidth + 1e-8)) ** 2))
        kernel_weights = kernel_weights / (kernel_weights.sum(dim=1, keepdim=True) + 1e-8)

        if Y_train.dim() == 1:
            Y_train = Y_train.unsqueeze(1)

        nearest_Y = Y_train[nearest_indices]

        if nearest_Y.dim() == 2:
            mu = (kernel_weights * nearest_Y).sum(dim=1)
        else:
            mu = (kernel_weights.unsqueeze(-1) * nearest_Y).sum(dim=1)

        return mu.squeeze()

    def individual_treatment_effect(self, X_new):
        """Estimate ITE for new covariates."""
        n_thres = int(getattr(self, 'n_thres', 100))
        if len(self.covariate_history) < n_thres:
            return torch.zeros(X_new.shape[0], device=self.device)

        X_hist = self.covariate_history
        T_hist = self.treatment_history
        Y_hist = self.outcome_history

        treat_mask = (T_hist == 1)
        control_mask = (T_hist == 0)
        treat_indices = treat_mask.nonzero(as_tuple=True)[0]
        control_indices = control_mask.nonzero(as_tuple=True)[0]

        if len(treat_indices) == 0 or len(control_indices) == 0:
            return torch.zeros(X_new.shape[0], device=self.device)

        X_treat = X_hist[treat_indices]
        Y_treat = Y_hist[treat_indices]
        X_control = X_hist[control_indices]
        Y_control = Y_hist[control_indices]

        mu_1_new = self.estimate_conditional_expectation(X_treat, Y_treat, X_new, k=5)
        mu_0_new = self.estimate_conditional_expectation(X_control, Y_control, X_new, k=5)

        ite = mu_1_new - mu_0_new
        return ite

    @torch.no_grad()
    def read(self, text_token):
        """Read / optionally enhance text features."""
        text_flat, orig_shape = self._reshape_to_2d(text_token)
        base_features = F.normalize(text_flat, dim=-1)

        if self.training:
            if self.training_step < self.warmup_steps:
                enhanced_features = self.apply_cache_enhancement(base_features)
                return self._reshape_back(enhanced_features, orig_shape)

            obs_thres = int(getattr(self, 'obs_history_thres', 200))
            if len(self.covariate_history) < obs_thres:
                if not bool(self.frozen_cache_initialized.item()):
                    self.initialize_frozen_cache()

                X, T, Y, enhanced_or_not = self.collect_observational_data(base_features)
                self.update_history(X, T, Y)
                return self._reshape_back(enhanced_or_not, orig_shape)

        n_thres = int(getattr(self, 'n_thres', 100))
        if len(self.covariate_history) >= n_thres:
            enhance_mask = self._gate_enhance_mask(base_features)
            enhanced_features = base_features.clone()
            if enhance_mask.any():
                enhanced_features[enhance_mask] = self.apply_cache_enhancement(
                    base_features[enhance_mask]
                )
            return self._reshape_back(enhanced_features, orig_shape)
        else:
            # History not ready yet: enhance all text features
            enhanced_features = self.apply_cache_enhancement(base_features)
            return self._reshape_back(enhanced_features, orig_shape)

    def _gate_enhance_mask(self, base_features):
        """Enhance when estimated ITE > 0 (paper default)."""
        ite = self.individual_treatment_effect(base_features)
        return ite > 0
        
    @torch.no_grad()
    def write(self, image_token):
        """Write to the text cache during training only."""
        if not self.training:
            return

        m, d = self.text_fine_cache.shape

        if isinstance(image_token, list):
            base_features = image_token[-1].clone()
        else:
            base_features = image_token.clone()

        if base_features.dim() == 1:
            base_features = base_features.unsqueeze(0)
        elif base_features.dim() == 3:
            B, N, d = base_features.shape
            base_features = base_features.reshape(-1, d)

        base_features = F.normalize(base_features, dim=-1)
        cache = self.text_fine_cache

        softmax_score_query, softmax_score_cache = self.get_score(base_features, cache)

        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)
        updated_cache = cache.clone()

        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i, as_tuple=False)

            if len(idx) > 0:
                idx_1d = idx[:, 0]
                weights = softmax_score_query[idx_1d, i]
                norm_weights = weights / (weights.sum() + 1e-8)

                weighted_features = (norm_weights.unsqueeze(1) * base_features[idx_1d]).sum(dim=0)
                updated_cache[i] = self.momentum * cache[i] + \
                                   (1 - self.momentum) * weighted_features

        updated_cache = F.normalize(updated_cache, dim=-1)
        # Replace buffers instead of inplace copy_ into graph-referenced storage
        self.register_buffer('text_fine_cache', updated_cache.detach().clone())

    def forward(self, text_token, image_token=None):
        """Forward: read (+ enhance), write during training."""
        if self.training:
            # Avoid inplace += on buffers
            self.register_buffer(
                'training_step',
                (self.training_step + 1).detach().to(dtype=torch.long)
            )

        enhanced_text = self.read(text_token)

        text_flat, orig_shape = self._reshape_to_2d(text_token)
        enhanced_flat, _ = self._reshape_to_2d(enhanced_text)

        if text_flat.shape[0] == enhanced_flat.shape[0]:
            concat_features = torch.cat((text_flat, enhanced_flat), dim=-1)
            final_features = self.alpha * self.extractor(concat_features) + text_flat
        else:
            final_features = enhanced_flat

        final_output = self._reshape_back(final_features, orig_shape)

        if self.training and image_token is not None:
            self.write(image_token)

        return final_output, 0.0


class ImageMemoryCache(nn.Module):
    def __init__(self, feature_dim=768, memory_size=25, momentum=0.8, alpha=0.2, device='cuda'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.alpha = alpha

        self.register_buffer(
            'image_fine_cache',
            F.normalize(torch.rand(self.memory_size, feature_dim), dim=-1)
        )
        self.extractor = nn.Linear(2 * feature_dim, feature_dim, bias=False)

    def get_score(self, query, mem):
        score = query @ mem.t()
        score_query = F.softmax(score, dim=0)
        score_mem = F.softmax(score, dim=1)
        return score_query, score_mem

    def read(self, x):
        base_features = F.normalize(x, dim=-1)
        cache = self.image_fine_cache.detach()
        _, softmax_score_cache = self.get_score(base_features, cache)
        fine_feature = softmax_score_cache @ cache
        return fine_feature

    @torch.no_grad()
    def write(self, x):
        """Update image cache from image tokens; training-only writes; frozen at inference."""
        if not self.training:
            return

        m, d = self.image_fine_cache.shape

        base_features = F.normalize(x, dim=-1)
        cache = self.image_fine_cache

        softmax_score_query, softmax_score_cache = self.get_score(base_features, cache)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)

        updated_cache = cache.clone()
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / (torch.max(softmax_score_query[:, i]) + 1e-8))
                updated_cache[i] = self.momentum * cache[i] + (1 - self.momentum) * torch.sum(
                    score * base_features[idx.squeeze(1)], dim=0)

        updated_cache = F.normalize(updated_cache, dim=-1)
        self.register_buffer('image_fine_cache', updated_cache.detach().clone())

    def forward(self, image_token):
        # Handle input rank
        if image_token.dim() == 1:
            image_token = image_token.unsqueeze(0)

        # Query the image cache with image features
        fine_feature = self.read(image_token)

        # Residual enhancement
        concat_features = torch.cat((image_token, fine_feature), dim=-1)
        enhanced_image_feature = self.alpha * self.extractor(concat_features) + image_token

        # Update cache during training; no writes at inference
        if self.training:
            self.write(image_token)
            loss = F.l1_loss(
                F.normalize(enhanced_image_feature, dim=-1),
                F.normalize(image_token, dim=-1),
                reduction='mean'
            )
        else:
            loss = 0.0

        return enhanced_image_feature, loss