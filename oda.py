"""Orthogonally Decoupled Alignment (ODA) loss — paper Sec. III-C."""
import torch
from torch import nn
from torch.nn import functional as F


class ODALoss(nn.Module):
    """Orthogonally Decoupled Alignment (ODA) contrastive loss."""
    def __init__(self, temp=0.1, gamma_init=-1.0, momentum=0.9, feature_dim=768):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(float(temp)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.momentum = momentum
        # Momentum-updated semantic basis
        self.register_buffer('semantic_basis', F.normalize(torch.randn(feature_dim), dim=0))

    @torch.no_grad()
    def _update_basis(self, features):
        """Momentum-update the semantic basis via a new buffer (avoid inplace writes)."""
        new_basis = F.normalize(features.mean(dim=0), dim=0)
        updated_basis = self.momentum * self.semantic_basis + (1 - self.momentum) * new_basis
        self.register_buffer('semantic_basis', F.normalize(updated_basis, dim=0).detach())

    def _feature_disentanglement(self, features, basis=None):
        """Decompose features into parallel / orthogonal parts w.r.t. the semantic basis."""
        if basis is None:
            basis = self.semantic_basis.detach()
        # Projection coefficients 
        dot_product = torch.mv(features, basis)  # [N]
        # Parallel (shared-semantic) component
        parallel_component = dot_product[:, None] * basis[None, :]  # [N, D]
        # Orthogonal (instance-specific) component
        orthogonal_component = features - parallel_component
        return parallel_component, orthogonal_component

    def forward(self, image_features, text_features):
        # Feature normalization
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Update the semantic basis during training 
        if self.training and image_features.size(0) > 1:  # update when batch size > 1
            self._update_basis(image_features.detach())

        # Disentangle with the global basis S (detached; no grad through S)
        basis = self.semantic_basis.detach()
        img_parallel, img_ortho = self._feature_disentanglement(image_features, basis)
        txt_parallel, txt_ortho = self._feature_disentanglement(text_features, basis)

        # Dynamic ODA weights: alpha = sigmoid(gamma * r), beta = 1 - alpha
        img_ratio = torch.norm(img_parallel, dim=1) / (torch.norm(image_features, dim=1) + 1e-8)
        txt_ratio = torch.norm(txt_parallel, dim=1) / (torch.norm(text_features, dim=1) + 1e-8)
        # Detach ratio so gamma still learns without fragile dual paths through norms
        alpha_img = torch.sigmoid(self.gamma * img_ratio.detach())
        alpha_txt = torch.sigmoid(self.gamma * txt_ratio.detach())
        beta_img = 1 - alpha_img
        beta_txt = 1 - alpha_txt

        # Disentangled similarities
        sim_parallel_i2t = img_parallel @ txt_parallel.T  # [N, N]
        sim_ortho_i2t = img_ortho @ txt_ortho.T  # [N, N]
        sim_parallel_t2i = txt_parallel @ img_parallel.T  # [N, N]
        sim_ortho_t2i = txt_ortho @ img_ortho.T  # [N, N]

        # Apply dynamic weights
        sim_i2t = (alpha_img.unsqueeze(1) * sim_parallel_i2t +
                   beta_img.unsqueeze(1) * sim_ortho_i2t)
        sim_t2i = (alpha_txt.unsqueeze(1) * sim_parallel_t2i +
                   beta_txt.unsqueeze(1) * sim_ortho_t2i)

        # Temperature scaling (tau_emp = exp(temp))
        temp = self.temp.exp()
        logits_i2t = sim_i2t / temp
        logits_t2i = sim_t2i / temp

        # Contrastive labels and loss
        labels = torch.arange(image_features.size(0), device=image_features.device)
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)

        return (loss_i2t + loss_t2i) / 2
